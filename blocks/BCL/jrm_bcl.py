import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：JRM-BCL (Joint Reasoning Module - BCL) —— 联合推理模块（BCL版）

一、模块简介
本模块是 JRM 针对 BCL 时序数据格式的适配版本。通过联合推理机制同时建模
多种类型的时序关系。

二、结构设计
JRM-BCL 由时间关系编码器、通道关系编码器、上下文关系编码器和联合推理器组成。
'''


class JRM_BCL(nn.Module):
    """JRM-BCL: Joint Reasoning Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 num_heads: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_heads = num_heads
        self.head_dim = inner // num_heads

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
        )
        self.channel_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )
        self.context_q = nn.Conv1d(inner, inner, 1, bias=False)
        self.context_k = nn.Conv1d(inner, inner, 1, bias=False)
        self.context_v = nn.Conv1d(inner, inner, 1, bias=False)
        self.reasoner = nn.Sequential(
            nn.Conv1d(inner * 3, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        temporal = self.temporal_encoder(x)
        channel = self.channel_encoder(x).expand(-1, -1, T)
        q = self.context_q(temporal).view(B, self.num_heads, self.head_dim, T).permute(0, 1, 3, 2)
        k = self.context_k(temporal).view(B, self.num_heads, self.head_dim, T).permute(0, 1, 3, 2)
        v = self.context_v(temporal).view(B, self.num_heads, self.head_dim, T).permute(0, 1, 3, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        context = torch.matmul(attn, v).permute(0, 1, 3, 2).contiguous().view(B, -1, T)
        concat = torch.cat([temporal, channel, context], dim=1)
        reasoned = self.reasoner(concat)
        gate = self.gate(reasoned)
        fused = reasoned * gate + x * (1 - gate)
        out = self.proj(fused)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = JRM_BCL(channels=64, seq_len=128)
    print('JRM-BCL:', model(x).shape)
