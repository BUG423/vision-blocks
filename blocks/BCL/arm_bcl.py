import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：ARM-BCL (Attention Refinement Module - BCL) —— 注意力精炼模块（BCL版）

一、模块简介
本模块是 ARM 针对 BCL 时序数据格式的适配版本。通过迭代精炼机制逐步
提升时序注意力图的质量。

二、结构设计
ARM-BCL 由初始注意力生成器、精炼迭代器、精炼门控组成。
'''


class ARM_BCL(nn.Module):
    """ARM-BCL: Attention Refinement Module for BCL format"""

    def __init__(self, channels: int, seq_len: int,
                 reduction: int = 4, num_iterations: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_iterations = num_iterations

        self.initial_attn = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine_conv = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.refine_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.initial_attn(x)
        for _ in range(self.num_iterations):
            refined = self.refine_conv(x * attn)
            gate = self.refine_gate(refined)
            attn = attn + gate * (refined.sigmoid() - attn)
        return x * attn + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = ARM_BCL(channels=64, seq_len=128)
    print('ARM-BCL:', model(x).shape)
