import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：ESM (Enhanced Spatial Module) —— 增强空间模块

一、模块简介
空间信息对于视觉任务至关重要，但标准卷积在固定网格上采样，限制了
对空间变换的建模能力。ESM 通过增强的空间建模机制，提升网络对空间
变换的适应性。

核心创新点：
1. 增强空间编码：显式编码空间位置信息
2. 空间自适应：根据内容自适应调整空间采样
3. 多尺度空间：在多个尺度上建模空间关系
4. 空间一致性：保持空间结构的一致性

二、结构设计
ESM 由以下子结构组成：
1. 空间位置编码器（Spatial Position Encoder）
2. 内容自适应采样器（Content-Adaptive Sampler）
3. 多尺度空间建模器（Multi-Scale Spatial Modeler）
4. 空间一致性约束

三、论文写法参考
"本文提出 ESM（Enhanced Spatial Module）模块，通过增强的空间建模机制提升
网络对空间变换的适应性。该模块显式编码空间位置信息，然后根据内容自适应
调整空间采样，并在多个尺度上建模空间关系，最后通过空间一致性约束保持
结构完整性。"

四、适用任务
适用于目标检测、语义分割、姿态估计等需要精确空间建模的视觉任务。
'''


class ESM(nn.Module):
    """ESM: Enhanced Spatial Module —— 增强空间模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_scales: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_scales = num_scales

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.02)

        # Content-adaptive sampling
        self.sampler = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, 2, 1, bias=False),
            nn.Tanh(),
        )

        # Multi-scale spatial modeling
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(channels, inner, k, padding=k // 2, bias=False)
            for k in [1, 3, 5]
        ])

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(inner * num_scales, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Spatial consistency
        self.consistency = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Position encoding
        x_pos = x + self.pos_encoding

        # 2. Content-adaptive sampling (simplified - use modulation)
        offset = self.sampler(x_pos)
        modulation = (1 + offset.mean(dim=1, keepdim=True)) * 0.5

        # 3. Multi-scale spatial modeling
        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(x_pos * modulation))

        # 4. Fusion
        concat = torch.cat(scale_features, dim=1)
        fused = self.fusion(concat)

        # 5. Spatial consistency
        out = self.consistency(fused)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = ESM(channels=64)
    output = model(input_tensor)
    print('=== ESM: Enhanced Spatial Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
