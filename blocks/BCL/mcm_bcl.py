import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：MCM-BCL (Multi-Scale Context Module - BCL) —— 多尺度上下文模块（BCL版）

一、模块简介
本模块是 MCM 针对 BCL 时序数据格式的适配版本。通过多尺度时间上下文
提取和自适应尺度权重学习，实现时序特征的多尺度增强。

二、结构设计
MCM-BCL 由多尺度提取器、尺度权重学习器、尺度交互器和全局-局部融合器组成。
'''


class MCM_BCL(nn.Module):
    """MCM-BCL: Multi-Scale Context Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, inner, k, padding=k // 2, groups=inner,
                          bias=False),
                nn.BatchNorm1d(inner),
                nn.GELU(),
            ) for k in [1, 3, 5, 7]
        ])

        self.scale_weight = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inner * 4, 4, 1, bias=False),
            nn.Softmax(dim=1),
        )

        self.proj = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_features = [conv(x) for conv in self.scale_convs]
        concat = torch.cat(scale_features, dim=1)
        weights = self.scale_weight(concat)
        combined = sum(f * weights[:, i:i+1, :] for i, f in enumerate(scale_features))
        out = self.proj(combined)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = MCM_BCL(channels=64, seq_len=128)
    print('MCM-BCL:', model(x).shape)
