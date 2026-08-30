import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：NAM-BCL (Neural Attention Module - BCL) —— 神经注意力模块（BCL版）

一、模块简介
本模块是 NAM 针对 BCL 时序数据格式的适配版本。通过多尺度神经注意力机制，
实现更全面的时序特征关注。

二、结构设计
NAM-BCL 由多尺度注意力计算器、尺度间交互器、注意力融合器和神经调制器组成。
'''


class NAM_BCL(nn.Module):
    """NAM-BCL: Neural Attention Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, inner, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(inner),
                nn.GELU(),
            ) for k in [1, 3, 5]
        ])
        self.scale_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(inner, inner, 1, bias=False),
                nn.Sigmoid(),
            ) for _ in range(3)
        ])
        self.interaction = nn.Sequential(
            nn.Conv1d(inner * 3, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, 3, 1, bias=False),
            nn.Softmax(dim=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.modulator = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        scale_features = []
        scale_attentions = []
        for conv, attn in zip(self.scale_convs, self.scale_attn):
            feat = conv(x)
            att = attn(feat)
            scale_features.append(feat)
            scale_attentions.append(att)
        att_concat = torch.cat([a.view(B, -1, 1) for a in scale_attentions], dim=1)
        weights = self.interaction(att_concat)
        fused = sum(f * weights[:, i:i+1, :] for i, f in enumerate(scale_features))
        fused = self.fusion(fused)
        mod = self.modulator(fused)
        out = x * mod
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = NAM_BCL(channels=64, seq_len=128)
    print('NAM-BCL:', model(x).shape)
