import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：GCM-BCL (Gradient Correlation Module - BCL) —— 梯度相关性模块（BCL版）

一、模块简介
本模块是 GCM 针对 BCL 时序数据格式的适配版本。通过建模时序梯度方向间的
相关性，实现梯度相关性引导的时序特征增强。

二、结构设计
GCM-BCL 由多方向梯度提取器、梯度相关性计算器、相关性引导增强器和结构保持融合组成。
'''


class GCM_BCL(nn.Module):
    """GCM-BCL: Gradient Correlation Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 num_directions: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_directions = num_directions

        self.gradient_convs = nn.ModuleList([
            nn.Conv1d(channels, inner, 3, padding=1, bias=False)
            for _ in range(num_directions)
        ])
        self.correlation = nn.Sequential(
            nn.Conv1d(num_directions * inner, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, num_directions, 1, bias=False),
            nn.Softmax(dim=1),
        )
        self.enhancer = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        gradients = [conv(x) for conv in self.gradient_convs]
        grad_concat = torch.cat(gradients, dim=1)
        grad_corr = self.correlation(grad_concat.mean(dim=-1))
        weighted_grad = torch.zeros_like(gradients[0])
        for i, grad in enumerate(gradients):
            weighted_grad = weighted_grad + grad * grad_corr[:, i:i+1, :]
        enhanced = self.enhancer(x + weighted_grad)
        out = self.proj(enhanced)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = GCM_BCL(channels=64, seq_len=128)
    print('GCM-BCL:', model(x).shape)
