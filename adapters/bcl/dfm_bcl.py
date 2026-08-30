import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：DFM-BCL (Dynamic Feature Module - BCL) —— 动态特征模块（BCL版）

一、模块简介
本模块是 DFM 针对 BCL 时序数据格式的适配版本。通过动态生成特征处理
参数，实现输入自适应的时序特征增强。

二、结构设计
DFM-BCL 由内容编码器、参数预测器、动态特征处理器和输出精炼组成。
'''


class DFM_BCL(nn.Module):
    """DFM-BCL: Dynamic Feature Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )
        self.scale_predictor = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.shift_predictor = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Tanh(),
        )
        self.processor = nn.Sequential(
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
        B, C, T = x.shape
        content = self.encoder(x)
        scale = self.scale_predictor(content).expand(-1, -1, T)
        shift = self.shift_predictor(content).expand(-1, -1, T)
        dynamic = x * scale + shift
        processed = self.processor(dynamic)
        out = self.refine(processed)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = DFM_BCL(channels=64, seq_len=128)
    print('DFM-BCL:', model(x).shape)
