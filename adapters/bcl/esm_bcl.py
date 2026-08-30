import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：ESM-BCL (Enhanced Spatial Module - BCL) —— 增强空间模块（BCL版）

一、模块简介
本模块是 ESM 针对 BCL 时序数据格式的适配版本。通过增强的时间建模机制，
提升网络对时序变换的适应性。

二、结构设计
ESM-BCL 由时间位置编码器、内容自适应采样器、多尺度时间建模器和时间一致性约束组成。
'''


class ESM_BCL(nn.Module):
    """ESM-BCL: Enhanced Spatial Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.pos_encoding = nn.Parameter(torch.randn(1, channels, 1) * 0.02)
        self.sampler = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, 1, 1, bias=False),
            nn.Tanh(),
        )
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(channels, inner, k, padding=k // 2, bias=False)
            for k in [1, 3, 5]
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(inner * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.consistency = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_pos = x + self.pos_encoding
        offset = self.sampler(x_pos)
        modulation = (1 + offset) * 0.5
        scale_features = [conv(x_pos * modulation) for conv in self.scale_convs]
        concat = torch.cat(scale_features, dim=1)
        fused = self.fusion(concat)
        out = self.consistency(fused)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = ESM_BCL(channels=64, seq_len=128)
    print('ESM-BCL:', model(x).shape)
