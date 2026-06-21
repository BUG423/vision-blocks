import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：GFM-BCL (Global Fusion Module - BCL) —— 全局融合模块（BCL版）

一、模块简介
本模块是 GFM 针对 BCL 时序数据格式的适配版本。通过将全局语义与局部细节
进行自适应融合实现互补增强。

二、结构设计
GFM-BCL 由全局语义提取器、局部细节增强器、自适应融合门和语义引导增强器组成。
'''


class GFM_BCL(nn.Module):
    """GFM-BCL: Global Fusion Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.global_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.local_enhancer = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.semantic_guided = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        global_feat = self.global_extractor(x).expand(-1, -1, T)
        local_feat = self.local_enhancer(x)
        concat = torch.cat([global_feat, local_feat], dim=1)
        gate = self.fusion_gate(concat)
        fused = global_feat * gate + local_feat * (1 - gate)
        out = self.semantic_guided(fused)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = GFM_BCL(channels=64, seq_len=128)
    print('GFM-BCL:', model(x).shape)
