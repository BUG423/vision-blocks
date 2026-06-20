import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：FTM-BCL (Frequency Transform Module - BCL) —— 频率变换模块（BCL版）

一、模块简介
本模块是 FTM 针对 BCL 时序数据格式的适配版本。通过轻量级的频率变换
近似，实现高效的时序频域特征处理。

二、结构设计
FTM-BCL 由频带选择器、频域增强器、门控和逆变换融合组成。
'''


class FTM_BCL(nn.Module):
    """FTM-BCL: Frequency Transform Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.band_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, 8, 1, bias=False),
            nn.Softmax(dim=1),
        )
        self.freq_enhance = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _haar_transform_1d(x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if T % 2 != 0:
            x = F.pad(x, (0, 1))
            T += 1
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        low = (x_even + x_odd) / 2
        high = (x_even - x_odd) / 2
        return torch.cat([low, high], dim=-1)

    @staticmethod
    def _haar_inverse_1d(x: torch.Tensor, orig_t: int) -> torch.Tensor:
        B, C, T = x.shape
        low = x[:, :, :T // 2]
        high = x[:, :, T // 2:]
        x_even = low + high
        x_odd = low - high
        x_out = torch.zeros(B, C, T, device=x.device, dtype=x.dtype)
        x_out[:, :, 0::2] = x_even
        x_out[:, :, 1::2] = x_odd
        return x_out[:, :, :orig_t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_freq = self._haar_transform_1d(x)
        weights = self.band_selector(x)
        enhanced = self.freq_enhance(x_freq)
        gate = self.gate(enhanced)
        x_freq = x_freq * gate + enhanced * (1 - gate)
        out = self._haar_inverse_1d(x_freq, T)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = FTM_BCL(channels=64, seq_len=128)
    print('FTM-BCL:', model(x).shape)
