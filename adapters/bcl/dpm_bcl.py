import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：DPM-BCL (Dense Prediction Module - BCL) —— 密集预测模块（BCL版）

一、模块简介
本模块是 DPM 针对 BCL 时序数据格式的适配版本。通过密集的预测头网络，
对每个时间位置进行自适应的特征预测。

二、结构设计
DPM-BCL 由全局上下文提取器、局部特征增强器、自适应融合器和密集预测头组成。
'''


class DPM_BCL(nn.Module):
    """DPM-BCL: Dense Prediction Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, num_classes: int,
                 reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.local_enhance = nn.Sequential(
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
        self.head = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, num_classes, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        global_feat = self.global_context(x).expand(-1, -1, T)
        local_feat = self.local_enhance(x)
        concat = torch.cat([global_feat, local_feat], dim=1)
        gate = self.fusion_gate(concat)
        fused = global_feat * gate + local_feat * (1 - gate)
        out = self.head(fused)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = DPM_BCL(channels=64, seq_len=128, num_classes=10)
    print('DPM-BCL:', model(x).shape)
