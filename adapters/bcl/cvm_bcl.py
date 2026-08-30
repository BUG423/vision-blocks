import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：CVM-BCL (Channel Variance Module - BCL) —— 通道方差模块（BCL版）

一、模块简介
本模块是 CVM 针对 BCL 时序数据格式的适配版本。通过通道方差估计实现
自适应的时序通道特征增强。

二、结构设计
CVM-BCL 由方差估计器、方差引导增强器、方差归一化器和残差融合组成。
'''


class CVM_BCL(nn.Module):
    """CVM-BCL: Channel Variance Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.var_estimator = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Softplus(),
        )
        self.enhancer = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.normalizer = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = self.var_estimator(x).view(x.size(0), x.size(1), 1)
        enhance_weight = self.enhancer(x)
        enhanced = x * enhance_weight * (1 + var)
        normalized = self.normalizer(enhanced)
        out = self.proj(normalized)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = CVM_BCL(channels=64, seq_len=128)
    print('CVM-BCL:', model(x).shape)
