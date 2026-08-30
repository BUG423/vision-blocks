import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：BFM-BCL (Batch Fusion Module - BCL) —— 批融合模块（BCL版）

一、模块简介
本模块是 BFM 针对 BCL 时序数据格式的适配版本。利用批次维度的统计信息
进行时序特征融合。

二、结构设计
BFM-BCL 由批统计提取器、样本间注意力、自适应融合门和输出投影组成。
'''


class BFM_BCL(nn.Module):
    """BFM-BCL: Batch Fusion Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )
        self.q_proj = nn.Conv1d(inner, inner, 1, bias=False)
        self.k_proj = nn.Conv1d(inner, inner, 1, bias=False)
        self.v_proj = nn.Conv1d(inner, inner, 1, bias=False)
        self.out_proj = nn.Conv1d(inner, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        feat = self.extractor(x).view(B, -1, 1)
        q = self.q_proj(feat)
        k = self.k_proj(feat)
        v = self.v_proj(feat)
        attn = torch.matmul(q.transpose(-1, -2), k) / (feat.shape[1] ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-1, -2))
        out = self.out_proj(out).view(B, C, 1)
        gate = self.gate(out)
        fused = out.expand_as(x) * gate + x * (1 - gate)
        return fused


if __name__ == '__main__':
    x = torch.randn(4, 64, 128)
    model = BFM_BCL(channels=64, seq_len=128)
    print('BFM-BCL:', model(x).shape)
