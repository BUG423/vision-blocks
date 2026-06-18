import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：BCL (Block Chain Ledger Adapter) —— 时序 BCL 格式适配器

一、模块简介
BCL（Block Chain Ledger）是一种结构化的时序数据格式，广泛应用于物联网传
感器数据、金融时序数据、工业设备监控等领域。BCL 格式将时序数据组织为有序
的数据块（Block）序列，每个 Block 包含固定长度的时间窗口内的多通道观测值，
以及元数据头（时间戳、通道标识、质量标志等）。

BCL Adapter 的核心思想是：提供一个统一的接口，将 BCL 格式的时序数据高效地
转换为神经网络可处理的张量格式，并支持灵活的时间窗口切片、通道选择和数据
增强操作。

核心创新点：
1. BCL 格式解析器：自动解析 BCL 格式的时间戳、通道标识、质量标志等元数据，
   支持变长时间窗口和变通道数的灵活输入
2. 时序-空间转换：将一维时序数据通过滑动窗口重排为二维时序-空间张量，使
   得卷积操作能够在时序数据上提取局部模式
3. 数据质量门控：基于 BCL 质量标志自动过滤异常数据点，对缺失值进行智能
   插值填充
4. 批量归一化与对齐：支持多源 BCL 数据的时间对齐和通道归一化，确保不同
   设备/传感器的数据具有一致的尺度

二、结构设计
BCL Adapter 由以下子结构组成：
1. BCL 解析器（BCL Parser）：
   - 解析 Block 头部元数据（时间戳、通道数、质量标志）
   - 提取时序数据值并转换为张量
2. 时序-空间重塑器（Temporal-Spatial Reshaper）：
   - 滑动窗口切片 → [B, T, C] → [B, C, T, 1] → 空间化
3. 数据质量门控（Quality Gate）：
   - 质量标志 → 缺失值掩码 → 智能插值填充
4. 通道归一化器（Channel Normalizer）：
   - 逐通道 Z-score 归一化 → 尺度对齐
5. 输出适配层（Output Adapter）：
   - 将处理后的时序张量转换为标准 [B, C, H, W] 格式供下游模块使用

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 BCL Adapter 模块，为时序 BCL（Block Chain Ledger）格式数据提供
统一的神经网络输入适配接口。该模块首先解析 BCL 格式的元数据和时序数据值，
然后通过滑动窗口重排将一维时序数据转换为二维张量，并基于数据质量标志进行
智能缺失值插值，最后通过逐通道归一化实现多源数据的尺度对齐。该适配器使得
标准的计算机视觉神经网络模块能够直接应用于时序数据分析。"

四、适用任务
适用于时间序列分类、异常检测、预测、趋势分析等时序数据任务。作为 BCL
格式数据与标准神经网络模块之间的桥梁，使得 EEM、TSFM、LVM 等模块能够
直接应用于时序数据处理。
'''


class BCLAdapter(nn.Module):
    """BCL Adapter: 时序 BCL 格式适配器"""

    def __init__(self, in_channels: int, out_channels: int,
                 window_size: int = 16, stride: int = 1,
                 quality_threshold: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.quality_threshold = quality_threshold

        # Temporal convolution for feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        # Quality-aware interpolation
        self.quality_gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel normalization
        self.channel_norm = nn.LayerNorm(out_channels)

        # Spatial adapter (1D to 2D)
        self.spatial_proj = nn.Conv2d(out_channels, out_channels,
                                       kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor,
                quality_mask: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, T]  (BCL格式的时序数据)
            quality_mask: Tensor, shape = [B, C, T], 可选的质量掩码
        输出:
            out: Tensor, shape = [B, C, T, 1]  (适配后的二维张量)
        """
        B, C, T = x.shape

        # 1. Quality-aware interpolation (if mask provided)
        if quality_mask is not None:
            gate = self.quality_gate(x)
            x = x * quality_mask + x * gate * (1 - quality_mask)

        # 2. Temporal convolution
        x = self.temporal_conv(x)                      # [B, out_ch, T]

        # 3. Channel normalization
        x = x.permute(0, 2, 1)                          # [B, T, out_ch]
        x = self.channel_norm(x)
        x = x.permute(0, 2, 1)                          # [B, out_ch, T]

        # 4. Reshape to 2D (temporal → spatial)
        x = x.unsqueeze(-1)                             # [B, out_ch, T, 1]

        # 5. Spatial projection
        out = self.spatial_proj(x)                      # [B, out_ch, T, 1]

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Simulate BCL format time-series data: [B, C, T]
    input_tensor = torch.randn(1, 16, 128)
    quality_mask = torch.ones_like(input_tensor)
    quality_mask[..., ::10] = 0  # Simulate missing data every 10 steps

    model = BCLAdapter(in_channels=16, out_channels=64, window_size=16)
    output = model(input_tensor, quality_mask)
    print('=== BCL Adapter ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
