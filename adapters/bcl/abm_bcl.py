import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：ABM-BCL (Adaptive Batch Module - BCL) —— 自适应批归一化模块（BCL版）

一、模块简介
本模块是 ABM 针对 BCL 时序数据格式的适配版本。通过内容感知的自适应机制，
实现时序数据的动态归一化。

二、结构设计
ABM-BCL 由内容编码器、统计量预测器、双重调制器和渐进式归一化层组成。
'''


class ABM_BCL(nn.Module):
    """ABM-BCL: Adaptive Batch Normalization Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )
        self.mean_predictor = nn.Conv1d(inner, channels, 1, bias=False)
        self.var_predictor = nn.Conv1d(inner, channels, 1, bias=False)
        self.running_mean = nn.Parameter(torch.zeros(channels))
        self.running_var = nn.Parameter(torch.ones(channels))
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        content = self.encoder(x)
        pred_mean = self.mean_predictor(content).view(B, C, 1)
        pred_var = self.var_predictor(content).view(B, C, 1)
        mean = self.running_mean.view(1, C, 1) + pred_mean
        var = self.running_var.view(1, C, 1) * (1 + pred_var)
        x_norm = (x - mean) / (var.sqrt() + 1e-5)
        out = self.proj(x_norm)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = ABM_BCL(channels=64, seq_len=128)
    print('ABM-BCL:', model(x).shape)
