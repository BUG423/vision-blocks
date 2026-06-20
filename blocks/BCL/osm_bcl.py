import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：OSM-BCL (Offset Spatial Mixing - BCL) —— 偏移空间混合模块（BCL版）

一、模块简介
本模块是 OSM 针对 BCL 时序数据格式的适配版本。通过时序偏移学习实现
更稳定的特征增强。

二、结构设计
OSM-BCL 由偏移预测器、偏移连续性约束、多尺度偏移融合和偏移感知聚合组成。
'''


class OSM_BCL(nn.Module):
    """OSM-BCL: Offset Spatial Mixing Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.offset_predictor = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.feature_convs = nn.ModuleList([
            nn.Conv1d(channels, inner, k, padding=k // 2, groups=inner,
                      bias=False)
            for k in [1, 3, 5]
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(inner * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        offsets = self.offset_predictor(x)
        offset_norm = offsets.norm(dim=1, keepdim=True)
        offset_weight = torch.sigmoid(offset_norm)
        scale_features = [conv(x) for conv in self.feature_convs]
        weighted = [f * offset_weight for f in scale_features]
        fused = self.fusion(torch.cat(weighted, dim=1))
        gate = self.gate(fused)
        return fused * gate + x * (1 - gate)


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = OSM_BCL(channels=64, seq_len=128)
    print('OSM-BCL:', model(x).shape)
