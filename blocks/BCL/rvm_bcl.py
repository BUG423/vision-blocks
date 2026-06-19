import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：RVM-BCL (Random Variation Module - BCL) —— 随机变异模块（BCL版）

一、模块简介
本模块是 RVM 针对 BCL 时序数据格式的适配版本。在网络内部引入可控的
随机变异，提升时序模型的鲁棒性。

二、结构设计
RVM-BCL 由温度控制器、噪声缩放器、特征调制器和方差保持归一化组成。
'''


class RVM_BCL(nn.Module):
    """RVM-BCL: Random Variation Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 temperature: float = 1.0):
        super().__init__()
        inner = max(1, channels // reduction)

        self.temperature = nn.Parameter(torch.tensor(temperature).log())
        self.noise_scale = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Softplus(),
        )
        self.modulator = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.temperature)
        noise = torch.randn_like(x) * temp
        scale = self.noise_scale(x)
        noise = noise * scale
        modulated = self.modulator(x + noise)
        out = self.norm(modulated)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = RVM_BCL(channels=64, seq_len=128)
    print('RVM-BCL:', model(x).shape)
