import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：EEM-BCL (Energy Equalization Module - BCL) —— 能量均衡模块（BCL版）

一、模块简介
本模块是 EEM（Energy Equalization Module）针对 BCL 时序数据格式的适配版
本。在 BCL 格式中，数据以 [B, C, T] 的三维张量形式组织，其中 T 为时间
步长。本模块将 EEM 的能量均衡思想扩展到时序维度，通过时间维度的能量均衡实
现时序特征的自适应增强。

核心创新点：
1. 时间维度能量均衡：在时间轴上进行能量统计和均衡，使得每个时间步的特征
   能量趋于均匀
2. 通道维度能量均衡：保持原有 EEM 的通道能量均衡机制
3. 时序-通道联合均衡：通过联合均衡机制同时处理时间维度和通道维度的能量
   分布

二、结构设计
EEM-BCL 由以下子结构组成：
1. 时间维度能量均衡器（Temporal Energy Equalizer）：
   - 沿时间轴统计每个时间步的能量 → 自适应时间缩放
2. 通道维度能量均衡器（Channel Energy Equalizer）：
   - 沿通道轴统计每个通道的能量 → 自适应通道缩放
3. 联合均衡融合器（Joint Equalization Fusion）：
   - 时间均衡结果与通道均衡结果的自适应融合
4. 输出精炼与残差连接
'''


class EEM_BCL(nn.Module):
    """EEM-BCL: Energy Equalization Module for BCL format"""

    def __init__(self, channels: int, seq_len: int,
                 reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner
        self.seq_len = seq_len

        # Temporal energy equalizer
        self.temporal_scale = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inner, max(1, inner // 4), 1, bias=False),
            nn.GELU(),
            nn.Conv1d(max(1, inner // 4), inner, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Channel energy equalizer
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inner, max(1, inner // 4), 1, bias=False),
            nn.GELU(),
            nn.Conv1d(max(1, inner // 4), inner, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Projection
        self.compress = nn.Conv1d(channels, inner, 1, bias=False)
        self.expand = nn.Conv1d(inner, channels, 1, bias=False)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(inner * 2, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, T] (BCL格式时序数据)
        输出:
            out: Tensor, shape = [B, C, T]
        """
        # 1. Compress
        feat = self.compress(x)                         # [B, inner, T]

        # 2. Temporal energy equalization
        t_energy = self.temporal_scale(feat)            # [B, inner, 1]
        t_eq = feat * t_energy

        # 3. Channel energy equalization
        c_energy = self.channel_scale(feat.permute(0, 2, 1)).permute(0, 2, 1)
        c_eq = feat * c_energy

        # 4. Joint fusion
        concat = torch.cat([t_eq, c_eq], dim=1)
        g = self.fusion_gate(concat)
        fused = g * t_eq + (1 - g) * c_eq

        # 5. Energy conservation
        orig_energy = feat.pow(2).mean()
        fused_energy = fused.pow(2).mean().clamp(min=1e-8)
        fused = fused * (orig_energy / fused_energy).sqrt()

        # 6. Expand and residual
        out = self.expand(fused)
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Simulate BCL format: [B, C, T]
    input_tensor = torch.randn(1, 64, 128)
    model = EEM_BCL(channels=64, seq_len=128)
    output = model(input_tensor)
    print('=== EEM-BCL ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
