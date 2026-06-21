import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：HPM (Hierarchical Prediction Module) —— 层次预测模块

一、模块简介
视觉预测任务通常需要在多个层次上进行预测。HPM 通过层次化的预测结构，
在不同层次上生成预测结果，然后融合多层次的预测信息。

核心创新点：
1. 层次化预测：在多个层次上独立预测
2. 层次间传递：预测信息在层次间传递
3. 渐进式精炼：预测结果逐层次精炼
4. 多层次融合：融合多层次的预测信息

二、结构设计
HPM 由以下子结构组成：
1. 层次预测器（Hierarchical Predictor）
2. 层次间传递器（Inter-Level Passer）
3. 渐进式精炼器（Progressive Refiner）
4. 多层次融合器（Multi-Level Fusion）

三、论文写法参考
"本文提出 HPM（Hierarchical Prediction Module）模块，通过层次化的预测结构
在多个层次上生成预测结果。该模块在不同层次上独立预测，然后通过层次间传递
和渐进式精炼逐步提升预测质量，最后融合多层次的预测信息。"

四、适用任务
适用于语义分割、目标检测、深度估计等需要多层次预测的视觉任务。
'''


class HPM(nn.Module):
    """HPM: Hierarchical Prediction Module —— 层次预测模块"""

    def __init__(self, channels: int, num_classes: int,
                 reduction: int = 4, num_levels: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_levels = num_levels

        # Hierarchical predictors
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, inner, 3, padding=1, bias=False),
                nn.BatchNorm2d(inner),
                nn.GELU(),
                nn.Conv2d(inner, num_classes, 1, bias=False),
            ) for _ in range(num_levels)
        ])

        # Inter-level passer
        self.passers = nn.ModuleList([
            nn.Conv2d(num_classes, channels, 1, bias=False)
            for _ in range(num_levels - 1)
        ])

        # Progressive refiner
        self.refiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_classes, inner, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(inner, num_classes, 1, bias=False),
            ) for _ in range(num_levels)
        ])

        # Multi-level fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes * num_levels, num_classes, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        predictions = []
        current = x

        for i in range(self.num_levels):
            # Predict
            pred = self.predictors[i](current)

            # Refine
            refined = self.refiners[i](pred)
            pred = pred + refined

            predictions.append(pred)

            # Pass to next level
            if i < self.num_levels - 1:
                current = self.passers[i](pred)

        # Multi-level fusion
        concat = torch.cat(predictions, dim=1)
        weights = self.fusion(concat)

        # Weighted sum
        out = sum(p * weights[:, i:i+1] for i, p in enumerate(predictions))

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = HPM(channels=64, num_classes=10)
    output = model(input_tensor)
    print('=== HPM: Hierarchical Prediction Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
