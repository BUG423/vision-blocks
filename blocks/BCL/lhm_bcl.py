import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：LHM-BCL (Local Histogram Module - BCL) —— 局部直方图模块（BCL版）

一、模块简介
本模块是 LHM 针对 BCL 时序数据格式的适配版本。通过可微分的局部直方图
近似计算，将时序分布统计信息引入特征增强。

二、结构设计
LHM-BCL 由可微分直方图计算器、分布特征提取器、分布感知增强器和残差融合组成。
'''


class LHM_BCL(nn.Module):
    """LHM-BCL: Local Histogram Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 num_bins: int = 16):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_bins = num_bins

        self.bin_boundaries = nn.Parameter(
            torch.linspace(-1, 1, num_bins + 1).view(1, 1, -1)
        )

        self.dist_extractor = nn.Sequential(
            nn.Conv1d(num_bins, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )

        self.enhancer = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def _soft_binning(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_flat = x.view(B * C, 1, -1)
        boundaries = self.bin_boundaries
        bin_width = boundaries[0, 0, 1] - boundaries[0, 0, 0]
        diff = (x_flat.unsqueeze(-1) - boundaries.view(1, 1, 1, -1)) / (bin_width + 1e-6)
        soft_assign = torch.exp(-diff ** 2 / 2)[:, :, :, :self.num_bins]
        hist = soft_assign.sum(dim=2)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)
        return hist.view(B, C, self.num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        hist = self._soft_binning(x)
        dist_feat = self.dist_extractor(hist).unsqueeze(-1).expand(-1, -1, T)
        enhanced = self.enhancer(x + dist_feat)
        gate = self.gate(enhanced)
        return enhanced * gate + x * (1 - gate)


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = LHM_BCL(channels=64, seq_len=128)
    print('LHM-BCL:', model(x).shape)
