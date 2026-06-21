import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：KBM-BCL (Knowledge Bridge Module - BCL) —— 知识桥接模块（BCL版）

一、模块简介
本模块是 KBM 针对 BCL 时序数据格式的适配版本。通过知识桥接机制实现
跨层/跨分支的有效时序特征融合。

二、结构设计
KBM-BCL 由知识编码器、语义对齐器、桥接传递器和自适应强度控制器组成。
'''


class KBM_BCL(nn.Module):
    """KBM-BCL: Knowledge Bridge Module for BCL format"""

    def __init__(self, channels_in: int, channels_out: int,
                 seq_len: int, reduction: int = 4):
        super().__init__()
        inner_in = max(1, channels_in // reduction)
        inner_out = max(1, channels_out // reduction)

        self.encoder = nn.Sequential(
            nn.Conv1d(channels_in, inner_in, 1, bias=False),
            nn.GELU(),
        )
        self.aligner = nn.Sequential(
            nn.Conv1d(inner_in, inner_out, 1, bias=False),
            nn.BatchNorm1d(inner_out),
            nn.GELU(),
        )
        self.bridge = nn.Sequential(
            nn.Conv1d(inner_out, inner_out, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner_out),
            nn.GELU(),
            nn.Conv1d(inner_out, channels_out, 1, bias=False),
        )
        self.strength = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels_out, channels_out, 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(channels_out, channels_out, 1, bias=False),
            nn.BatchNorm1d(channels_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, T = x.shape
        encoded = self.encoder(x)
        aligned = self.aligner(encoded)
        bridged = self.bridge(aligned)
        strength = self.strength(bridged)
        bridged = bridged * strength
        out = self.refine(bridged)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = KBM_BCL(channels_in=64, channels_out=128, seq_len=128)
    print('KBM-BCL:', model(x).shape)
