import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：IPM-BCL (Iterative Processing Module - BCL) —— 迭代处理模块（BCL版）

一、模块简介
本模块是 IPM 针对 BCL 时序数据格式的适配版本。通过迭代处理机制让
时序特征在多次迭代中逐步优化。

二、结构设计
IPM-BCL 由迭代处理器、残差累积器、迭代门控和自适应迭代控制器组成。
'''


class IPM_BCL(nn.Module):
    """IPM-BCL: Iterative Processing Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 max_iterations: int = 5):
        super().__init__()
        inner = max(1, channels // reduction)
        self.max_iterations = max_iterations

        self.processor = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.accumulator = nn.Conv1d(channels, channels, 1, bias=False)
        self.iter_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        complexity = self.controller(x)
        num_iter = int((complexity.mean() * self.max_iterations).item()) + 1
        num_iter = min(num_iter, self.max_iterations)
        accumulated = torch.zeros_like(x)
        current = x
        for _ in range(num_iter):
            processed = self.processor(current)
            gate = self.iter_gate(processed)
            current = current + processed * gate
            accumulated = accumulated + processed
        out = self.accumulator(accumulated)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = IPM_BCL(channels=64, seq_len=128)
    print('IPM-BCL:', model(x).shape)
