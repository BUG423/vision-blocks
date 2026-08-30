import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：CLM-BCL (Context Learning Module - BCL) —— 上下文学习模块（BCL版）

一、模块简介
本模块是 CLM 针对 BCL 时序数据格式的适配版本。通过学习多种时序上下文
类型并自适应融合，实现更全面的时序上下文理解。

二、结构设计
CLM-BCL 由多类型上下文提取器、上下文选择器、上下文交互器和学习型融合器组成。
'''


class CLM_BCL(nn.Module):
    """CLM-BCL: Context Learning Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.temporal_ctx = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
        )
        self.channel_ctx = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )
        self.selector = nn.Sequential(
            nn.Conv1d(inner * 3, 3, 1, bias=False),
            nn.Softmax(dim=1),
        )
        self.interaction = nn.Conv1d(inner, inner, 3, padding=1,
                                     groups=inner, bias=False)
        self.proj = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        temporal = self.temporal_ctx(x)
        channel = self.channel_ctx(x).expand(-1, -1, T)
        global_ = self.global_ctx(x).expand(-1, -1, T)
        concat = torch.cat([temporal, channel, global_], dim=1)
        weights = self.selector(concat)
        fused = temporal * weights[:, 0:1] + channel * weights[:, 1:2] + global_ * weights[:, 2:3]
        interacted = self.interaction(fused)
        out = self.proj(interacted)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = CLM_BCL(channels=64, seq_len=128)
    print('CLM-BCL:', model(x).shape)
