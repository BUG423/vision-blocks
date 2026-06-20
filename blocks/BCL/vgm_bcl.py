import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：VGM-BCL (Variational Gaussian Mixing - BCL) —— 变分高斯混合模块（BCL版）

一、模块简介
本模块是 VGM 针对 BCL 时序数据格式的适配版本。将时序特征建模为高斯分布，
通过变分推断学习分布参数，实现不确定性感知的时序特征混合。

二、结构设计
VGM-BCL 由高斯参数编码器、重参数化采样、不确定性感知混合和输出投影组成。
'''


class VGM_BCL(nn.Module):
    """VGM-BCL: Variational Gaussian Mixing Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.mu_encoder = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.logvar_encoder = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.mixer = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        z = self.reparameterize(mu, logvar)
        uncertainty = torch.exp(-logvar)
        mixed = self.mixer(torch.cat([z * uncertainty, x], dim=1))
        out = self.proj(mixed)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = VGM_BCL(channels=64, seq_len=128)
    model.train()
    print('VGM-BCL (train):', model(x).shape)
    model.eval()
    print('VGM-BCL (eval):', model(x).shape)
