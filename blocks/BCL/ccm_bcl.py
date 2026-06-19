import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：CCM-BCL (Channel Correlation Module - BCL) —— 通道相关性模块（BCL版）

一、模块简介
本模块是 CCM 针对 BCL 时序数据格式的适配版本。通过建模时序通道间的
统计相关性，实现更精确的通道特征增强。

二、结构设计
CCM-BCL 由通道相关性计算器、低秩近似器、相关性引导增强器和正则化组成。
'''


class CCM_BCL(nn.Module):
    """CCM-BCL: Channel Correlation Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 rank: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.rank = min(rank, channels)

        self.down_proj = nn.Conv1d(channels, self.rank, 1, bias=False)
        self.up_proj = nn.Conv1d(self.rank, channels, 1, bias=False)

        self.enhancer = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = self.down_proj(x)
        x_corr = self.up_proj(x_low)
        enhance_weight = self.enhancer(x_corr)
        enhanced = x * enhance_weight
        out = self.proj(enhanced)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = CCM_BCL(channels=64, seq_len=128)
    print('CCM-BCL:', model(x).shape)
