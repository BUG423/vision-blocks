import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：HPM-BCL (Hierarchical Prediction Module - BCL) —— 层次预测模块（BCL版）

一、模块简介
本模块是 HPM 针对 BCL 时序数据格式的适配版本。通过层次化的预测结构，
在不同层次上生成时序预测结果。

二、结构设计
HPM-BCL 由层次预测器、层次间传递器、渐进式精炼器和多层次融合器组成。
'''


class HPM_BCL(nn.Module):
    """HPM-BCL: Hierarchical Prediction Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, num_classes: int,
                 reduction: int = 4, num_levels: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_levels = num_levels

        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, inner, 3, padding=1, bias=False),
                nn.BatchNorm1d(inner),
                nn.GELU(),
                nn.Conv1d(inner, num_classes, 1, bias=False),
            ) for _ in range(num_levels)
        ])
        self.passers = nn.ModuleList([
            nn.Conv1d(num_classes, channels, 1, bias=False)
            for _ in range(num_levels - 1)
        ])
        self.refiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_classes, inner, 1, bias=False),
                nn.GELU(),
                nn.Conv1d(inner, num_classes, 1, bias=False),
            ) for _ in range(num_levels)
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(num_classes * num_levels, num_classes, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        predictions = []
        current = x
        for i in range(self.num_levels):
            pred = self.predictors[i](current)
            refined = self.refiners[i](pred)
            pred = pred + refined
            predictions.append(pred)
            if i < self.num_levels - 1:
                current = self.passers[i](pred)
        concat = torch.cat(predictions, dim=1)
        weights = self.fusion(concat)
        out = sum(p * weights[:, i:i+1] for i, p in enumerate(predictions))
        return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = HPM_BCL(channels=64, seq_len=128, num_classes=10)
    print('HPM-BCL:', model(x).shape)
