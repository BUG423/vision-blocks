import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：GFM (Global Fusion Module) —— 全局融合模块

一、模块简介
全局信息对于理解图像的整体语义至关重要，但全局特征通常缺乏局部细节。
GFM 通过将全局语义与局部细节进行自适应融合，实现全局-局部的互补增强。

核心创新点：
1. 全局语义提取：提取图像的全局语义信息
2. 局部细节保持：保持图像的局部细节信息
3. 自适应融合：根据内容自适应融合全局和局部信息
4. 语义引导：全局语义引导局部特征增强

二、结构设计
GFM 由以下子结构组成：
1. 全局语义提取器（Global Semantic Extractor）
2. 局部细节增强器（Local Detail Enhancer）
3. 自适应融合门（Adaptive Fusion Gate）
4. 语义引导增强器（Semantic-Guided Enhancer）

三、论文写法参考
"本文提出 GFM（Global Fusion Module）模块，通过将全局语义与局部细节
进行自适应融合实现互补增强。该模块提取全局语义信息和局部细节信息，
然后通过自适应融合门融合两者，最后用全局语义引导局部特征增强。"

四、适用任务
适用于图像分类、语义分割、场景理解等需要全局-局部融合的视觉任务。
'''


class GFM(nn.Module):
    """GFM: Global Fusion Module —— 全局融合模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Global semantic extractor
        self.global_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Local detail enhancer
        self.local_enhancer = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Adaptive fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Semantic-guided enhancer
        self.semantic_guided = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Global semantic
        global_feat = self.global_extractor(x).expand(-1, -1, H, W)

        # 2. Local detail
        local_feat = self.local_enhancer(x)

        # 3. Adaptive fusion
        concat = torch.cat([global_feat, local_feat], dim=1)
        gate = self.fusion_gate(concat)
        fused = global_feat * gate + local_feat * (1 - gate)

        # 4. Semantic-guided enhancement
        out = self.semantic_guided(fused)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = GFM(channels=64)
    output = model(input_tensor)
    print('=== GFM: Global Fusion Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
