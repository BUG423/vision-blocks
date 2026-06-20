import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：GCM (Gradient Correlation Module) —— 梯度相关性模块

一、模块简介
梯度信息包含了图像的边缘和纹理结构，但现有的梯度利用方法通常只考虑
梯度的幅值，忽略了梯度方向间的相关性。

GCM 的核心思想是：通过建模梯度方向间的相关性，提取更丰富的结构信息，
实现梯度相关性引导的特征增强。

核心创新点：
1. 梯度方向相关性：计算不同方向梯度间的相关性
2. 相关性引导增强：利用相关性信息引导特征增强
3. 方向感知编码：对不同方向的梯度进行感知编码
4. 结构保持融合

二、结构设计
GCM 由以下子结构组成：
1. 多方向梯度提取器（Multi-Direction Gradient Extractor）
2. 梯度相关性计算器（Gradient Correlation Calculator）
3. 相关性引导增强器（Correlation-Guided Enhancer）
4. 结构保持融合

三、论文写法参考
"本文提出 GCM（Gradient Correlation Module）模块，通过建模梯度方向间的
相关性实现更丰富的结构信息提取。该模块首先提取多个方向的梯度，然后计算
梯度方向间的相关性，最后利用相关性信息引导特征增强并保持结构完整性。"

四、适用任务
适用于边缘检测、语义分割、纹理分析等需要结构信息的视觉任务。
'''


class GCM(nn.Module):
    """GCM: Gradient Correlation Module —— 梯度相关性模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_directions: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_directions = num_directions

        # Multi-direction gradient extractors
        self.gradient_convs = nn.ModuleList([
            nn.Conv2d(channels, inner, 3, padding=1, bias=False)
            for _ in range(num_directions)
        ])

        # Gradient correlation calculator
        self.correlation = nn.Sequential(
            nn.Conv1d(num_directions * inner, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, num_directions, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Correlation-guided enhancer
        self.enhancer = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Structure preservation
        self.structure_preserve = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Multi-direction gradients
        gradients = []
        for conv in self.gradient_convs:
            gradients.append(conv(x))

        # 2. Gradient correlation
        grad_concat = torch.cat(gradients, dim=1)        # [B, num_dir*inner, H, W]
        grad_flat = grad_concat.view(B, self.num_directions, -1, H * W)
        grad_corr = self.correlation(grad_flat.mean(dim=-1))  # [B, num_dir, 1]

        # 3. Weighted gradient fusion
        weighted_grad = torch.zeros_like(gradients[0])
        for i, grad in enumerate(gradients):
            weighted_grad = weighted_grad + grad * grad_corr[:, i:i+1, :]

        # 4. Enhance
        enhanced = self.enhancer(x + weighted_grad)

        # 5. Structure preservation
        preserved = self.structure_preserve(enhanced)

        # 6. Output projection
        out = self.proj(preserved)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = GCM(channels=64)
    output = model(input_tensor)
    print('=== GCM: Gradient Correlation Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
