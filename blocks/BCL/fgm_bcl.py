import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：FGM-BCL (Feature Gating Module - BCL) —— 特征门控模块（BCL版）

一、模块简介
本模块是 FGM 针对 BCL 时序数据格式的适配版本。通过协作门控机制实现
时序通道间的相互影响特征选择。

二、结构设计
FGM-BCL 由协作门控生成器、局部上下文门控、双向通道交互和稀疏激活组成。
'''


class FGM_BCL(nn.Module):
    """FGM-BCL: Feature Gating Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.coop_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.local_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coop = self.coop_gate(x)
        local = self.local_gate(x)
        gate = coop * local
        out = self.proj(x * gate)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = FGM_BCL(channels=64, seq_len=128)
    print('FGM-BCL:', model(x).shape)
