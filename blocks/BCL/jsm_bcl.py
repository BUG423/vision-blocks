import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：JSM-BCL (Joint Selection Module - BCL) —— 联合选择模块（BCL版）

一、模块简介
本模块是 JSM 针对 BCL 时序数据格式的适配版本。通过联合在时间和通道两个
维度上进行选择实现更精确的特征筛选。

二、结构设计
JSM-BCL 由时间选择器、通道选择器、选择一致性约束和联合激活组成。
'''


class JSM_BCL(nn.Module):
    """JSM-BCL: Joint Selection Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 sparse_ratio: float = 0.5):
        super().__init__()
        inner = max(1, channels // reduction)
        self.sparse_ratio = sparse_ratio

        self.temporal_selector = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.channel_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.consistency = nn.Sequential(
            nn.Conv1d(channels + 1, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        temporal_weight = self.temporal_selector(x)
        channel_weight = self.channel_selector(x)
        concat = torch.cat([x * channel_weight,
                           temporal_weight.expand(-1, C, -1)], dim=1)
        consistent = self.consistency(concat)
        joint_weight = consistent * temporal_weight.expand_as(consistent)
        out = self.proj(x * joint_weight)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = JSM_BCL(channels=64, seq_len=128)
    print('JSM-BCL:', model(x).shape)
