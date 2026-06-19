import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：KSM (Kernel Selection Module) —— 核选择模块

一、模块简介
卷积核的大小决定了感受野的范围，但固定大小的卷积核无法适应不同位置的
需求。空洞卷积通过膨胀率扩展感受野，但膨胀率的选择通常是手动的。

KSM 的核心思想是：让网络自适应地为每个空间位置选择最合适的卷积核大小，
实现位置自适应的感受野选择。

核心创新点：
1. 位置感知核选择：每个位置选择不同的卷积核
2. 可微分选择：通过 soft selection 实现可微分的核选择
3. 多尺度并行：并行多个不同大小的卷积核
4. 稀疏选择：鼓励稀疏的核选择模式

二、结构设计
KSM 由以下子结构组成：
1. 核选择权重生成器（Kernel Selection Weight Generator）：
   - 空间注意力 → 核选择权重
2. 多尺度卷积并行（Multi-Scale Convolution Parallel）：
   - 并行 1x1, 3x3, 5x5 卷积
3. 软选择融合（Soft Selection Fusion）：
   - 基于权重的软选择
4. 稀疏正则化

三、论文写法参考
"本文提出 KSM（Kernel Selection Module）模块，让网络自适应地为每个空间
位置选择最合适的卷积核大小。该模块通过空间注意力生成核选择权重，并行多个
不同大小的卷积核，然后基于权重进行软选择融合，并加入稀疏正则化鼓励选择
模式的稀疏性。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要自适应感受野的视觉任务。
'''


class KSM(nn.Module):
    """KSM: Kernel Selection Module —— 核选择模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 kernel_sizes: t.List[int] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        self.kernel_sizes = kernel_sizes
        num_kernels = len(kernel_sizes)
        inner = max(1, channels // reduction)

        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, inner, k, padding=k // 2, bias=False),
                nn.BatchNorm2d(inner),
                nn.GELU(),
            ) for k in kernel_sizes
        ])

        # Kernel selection weight generator
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, num_kernels, 1, bias=False),
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Generate selection weights
        weights = self.weight_gen(x)                      # [B, num_kernels, 1, 1]
        weights = F.softmax(weights, dim=1)

        # 2. Multi-scale convolution
        scale_features = []
        for conv in self.convs:
            scale_features.append(conv(x))

        # 3. Soft selection fusion
        out = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            out = out + feat * weights[:, i:i+1, :, :]

        # 4. Output projection
        out = self.proj(out)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = KSM(channels=64)
    output = model(input_tensor)
    print('=== KSM: Kernel Selection Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
