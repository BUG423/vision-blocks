import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：DWM-BCL (Dynamic Weight Module - BCL) —— 动态权重模块（BCL版）

一、模块简介
本模块是 DWM 针对 BCL 时序数据格式的适配版本。通过轻量级的权重预测
网络动态生成时序卷积核的调制因子。

二、结构设计
DWM-BCL 由内容编码器、调制因子预测器、渐进式调制和平滑性约束组成。
'''


class DWM_BCL(nn.Module):
    """DWM-BCL: Dynamic Weight Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(inner, inner, 1, bias=False),
                nn.GELU(),
                nn.Conv1d(inner, channels, 1, bias=False),
                nn.Tanh(),
            ) for _ in range(3)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content = self.encoder(x)
        modulated = x
        for predictor in self.predictors:
            factor = predictor(content)
            modulated = modulated * (1 + factor * 0.1)
        out = self.proj(modulated)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = DWM_BCL(channels=64, seq_len=128)
    print('DWM-BCL:', model(x).shape)
