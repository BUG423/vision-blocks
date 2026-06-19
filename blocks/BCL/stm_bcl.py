import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：STM-BCL (Spatial-Channel Transformer Module - BCL) —— 空间-通道变换模块（BCL版）

一、模块简介
本模块是 STM 针对 BCL 时序数据格式的适配版本。通过时间-通道联合变换机制，
实现时序特征的精确建模。

二、结构设计
STM-BCL 由时间编码器、通道编码器、交叉注意力和联合融合组成。
'''


class STM_BCL(nn.Module):
    """STM-BCL: Spatial-Channel Transformer Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 num_heads: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_heads = num_heads
        self.head_dim = inner // num_heads

        self.temporal_enc = nn.Conv1d(channels, inner, 1, bias=False)
        self.channel_enc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
        )

        self.t2c_q = nn.Conv1d(inner, inner, 1, bias=False)
        self.t2c_k = nn.Conv1d(inner, inner, 1, bias=False)
        self.t2c_v = nn.Conv1d(inner, inner, 1, bias=False)

        self.fusion_gate = nn.Sequential(
            nn.Conv1d(inner * 2, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

        self.scale = self.head_dim ** -0.5

    def _cross_attn(self, q, k, v):
        B, C, T = q.shape
        h = self.num_heads
        d = self.head_dim

        q = q.view(B, h, d, T).permute(0, 1, 3, 2)
        k = k.view(B, h, d, -1).permute(0, 1, 3, 2)
        v = v.view(B, h, d, -1).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).contiguous()
        return out.view(B, C, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_feat = self.temporal_enc(x)
        c_feat = self.channel_enc(x)

        t_q = self.t2c_q(t_feat)
        t2c = self._cross_attn(t_q, c_feat, c_feat)

        c2t = c_feat.expand_as(t_feat) * t_feat

        concat = torch.cat([t2c, c2t], dim=1)
        gate = self.fusion_gate(concat)
        fused = gate * t2c + (1 - gate) * c2t

        out = self.proj(fused)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = STM_BCL(channels=64, seq_len=128)
    print('STM-BCL:', model(x).shape)
