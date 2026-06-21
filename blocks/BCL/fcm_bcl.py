import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：FCM-BCL (Feature Correlation Module - BCL) —— 特征相关性模块（BCL版）

一、模块简介
本模块是 FCM 针对 BCL 时序数据格式的适配版本。通过建模时序特征间的
全局相关性模式实现更有效的特征交互。

二、结构设计
FCM-BCL 由低秩相关性计算、相关性模式学习、相关性引导增强和输出精炼组成。
'''


class FCM_BCL(nn.Module):
    """FCM-BCL: Feature Correlation Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 rank: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.rank = min(rank, channels)

        self.down_proj = nn.Conv1d(channels, self.rank, 1, bias=False)
        self.up_proj = nn.Conv1d(self.rank, channels, 1, bias=False)
        self.pattern_learner = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.enhancer = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = self.down_proj(x)
        x_corr = self.up_proj(x_low)
        pattern = self.pattern_learner(x_corr)
        enhanced = self.enhancer(x * pattern)
        out = self.refine(enhanced)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = FCM_BCL(channels=64, seq_len=128)
    print('FCM-BCL:', model(x).shape)
