import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：PAM-BCL (Phase Alignment Module - BCL) —— 相位对齐模块（BCL版）

一、模块简介
本模块是 PAM 针对 BCL 时序数据格式的适配版本。通过时间频率域的相位估计
和自适应对齐机制，实现时序特征的相位一致性融合。

二、结构设计
PAM-BCL 由相位估计器、相位对齐器、相位一致性约束和残差融合组成。
'''


class PAM_BCL(nn.Module):
    """PAM-BCL: Phase Alignment Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.phase_conv = nn.Conv1d(channels, 8, 3, padding=1, bias=False)
        self.phase_shift = nn.Sequential(
            nn.Conv1d(8, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Tanh(),
        )
        self.consistency = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase_map = self.phase_conv(x)
        phase_offset = self.phase_shift(phase_map)
        aligned = x + phase_offset * 0.1
        consistent = self.consistency(aligned)
        gate = self.gate(consistent)
        return consistent * gate + x * (1 - gate)


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = PAM_BCL(channels=64, seq_len=128)
    print('PAM-BCL:', model(x).shape)
