import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：KSM-BCL (Kernel Selection Module - BCL) —— 核选择模块（BCL版）

一、模块简介
本模块是 KSM 针对 BCL 时序数据格式的适配版本。让网络自适应地为每个
时间位置选择最合适的卷积核大小。

二、结构设计
KSM-BCL 由多尺度卷积并行、核选择权重生成器和软选择融合组成。
'''


class KSM_BCL(nn.Module):
    """KSM-BCL: Kernel Selection Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4,
                 kernel_sizes: t.List[int] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        self.kernel_sizes = kernel_sizes
        num_kernels = len(kernel_sizes)
        inner = max(1, channels // reduction)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, inner, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(inner),
                nn.GELU(),
            ) for k in kernel_sizes
        ])

        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, num_kernels, 1, bias=False),
        )

        self.proj = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.weight_gen(x)
        weights = F.softmax(weights, dim=1)
        scale_features = [conv(x) for conv in self.convs]
        out = sum(f * weights[:, i:i+1, :] for i, f in enumerate(scale_features))
        out = self.proj(out)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = KSM_BCL(channels=64, seq_len=128)
    print('KSM-BCL:', model(x).shape)
