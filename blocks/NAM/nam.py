import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：NAM (Neural Attention Module) —— 神经注意力模块

一、模块简介
注意力机制是深度学习的核心技术，但标准的注意力计算通常在单一尺度上
进行，忽略了多尺度信息的互补性。

NAM 的核心思想是：通过多尺度神经注意力机制，在不同尺度上计算注意力，
然后融合多尺度注意力信息，实现更全面的特征关注。

核心创新点：
1. 多尺度注意力：在多个尺度上并行计算注意力
2. 尺度间交互：不同尺度的注意力相互影响
3. 注意力融合：自适应融合多尺度注意力
4. 神经调制：注意力结果调制特征

二、结构设计
NAM 由以下子结构组成：
1. 多尺度注意力计算器（Multi-Scale Attention Calculator）
2. 尺度间交互器（Scale Interactor）
3. 注意力融合器（Attention Fusion）
4. 神经调制器（Neural Modulator）

三、论文写法参考
"本文提出 NAM（Neural Attention Module）模块，通过多尺度神经注意力机制
实现更全面的特征关注。该模块在多个尺度上并行计算注意力，然后通过尺度间
交互和注意力融合获得多尺度注意力信息，最后用注意力结果调制特征。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要多尺度注意力的视觉任务。
'''


class NAM(nn.Module):
    """NAM: Neural Attention Module —— 神经注意力模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_scales: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_scales = num_scales

        # Multi-scale attention
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, inner, k, padding=k // 2, bias=False),
                nn.BatchNorm2d(inner),
                nn.GELU(),
            ) for k in [1, 3, 5]
        ])

        self.scale_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inner, inner, 1, bias=False),
                nn.Sigmoid(),
            ) for _ in range(num_scales)
        ])

        # Scale interaction
        self.interaction = nn.Sequential(
            nn.Conv1d(inner * num_scales, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, num_scales, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Attention fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Neural modulator
        self.modulator = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Multi-scale attention
        scale_features = []
        scale_attentions = []
        for conv, attn in zip(self.scale_convs, self.scale_attn):
            feat = conv(x)
            att = attn(feat)
            scale_features.append(feat)
            scale_attentions.append(att)

        # 2. Scale interaction
        att_concat = torch.cat([a.view(B, -1, 1) for a in scale_attentions],
                               dim=1)
        weights = self.interaction(att_concat)             # [B, num_scales, 1]

        # 3. Attention fusion
        fused = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            fused = fused + feat * weights[:, i:i+1, :, :]

        fused = self.fusion(fused)

        # 4. Neural modulation
        mod = self.modulator(fused)
        out = x * mod

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = NAM(channels=64)
    output = model(input_tensor)
    print('=== NAM: Neural Attention Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
