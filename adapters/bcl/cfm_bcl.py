import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：CFM-BCL (Channel Frequency Mixer - BCL) —— 通道频率混合模块（BCL版）

一、模块简介
本模块是 CFM（Channel Frequency Mixer）针对 BCL 时序数据格式的适配版本。
在 BCL 格式中，时序数据以 [B, C, T] 的形式组织，时间维度替代了空间维度。
本模块将 CFM 的频域通道混合思想扩展到时序数据，通过 DCT 变换在时间频率域
进行通道间的信息交互。

核心创新点：
1. 时间频率域通道混合：将时序特征通过 DCT 变换到时间频率域，在频域进行
   跨通道信息交换
2. 时间频率感知缩放：根据每个通道的时间频率特性自适应调整混合权重
3. 残差频率门控：控制频率混合强度，避免过度混合

二、结构设计
CFM-BCL 由以下子结构组成：
1. 时间频率变换器（Temporal Frequency Transform）：
   - 1D DCT 将时序特征变换到频率域
2. 通道频率混合器（Channel Frequency Mixer）：
   - 可学习频率混合矩阵 → 跨通道频率信息交换
3. 频率感知缩放器（Frequency-Aware Scaler）：
   - 通道频率特性编码 → 自适应缩放因子
4. 残差门控与逆变换
'''


class CFM_BCL(nn.Module):
    """CFM-BCL: Channel Frequency Mixer for BCL format"""

    def __init__(self, channels: int, seq_len: int,
                 reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len

        # Frequency-aware channel scaler
        self.channel_scaler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(1, channels // reduction), 1, bias=False),
            nn.GELU(),
            nn.Conv1d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel frequency mixing matrix
        self.mix_matrix = nn.Parameter(
            torch.eye(channels).unsqueeze(0)
        )

        # Residual gate
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _dct_1d(x: torch.Tensor) -> torch.Tensor:
        """Simple 1D DCT approximation."""
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
    def _idct_1d(x: torch.Tensor, orig_t: int) -> torch.Tensor:
        """Inverse 1D DCT."""
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
        """
        输入:
            x: Tensor, shape = [B, C, T] (BCL格式时序数据)
        输出:
            out: Tensor, shape = [B, C, T]
        """
        B, C, T = x.shape

        # 1. Frequency-aware scaling
        scale = self.channel_scaler(x)
        x_scaled = x * scale

        # 2. Transform to frequency domain
        x_freq = self._dct_1d(x_scaled)

        # 3. Channel frequency mixing
        x_flat = x_freq.view(B, C, -1)
        mix_weight = torch.sigmoid(self.mix_matrix)
        x_mixed = torch.bmm(mix_weight.expand(B, -1, -1), x_flat)
        x_mixed = x_mixed.view(B, C, -1)

        # 4. Residual gate
        gate = self.gate(x_mixed)
        x_out = x_mixed * gate + x_freq * (1 - gate)

        # 5. Inverse transform
        out = self._idct_1d(x_out, T)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 128)
    model = CFM_BCL(channels=64, seq_len=128)
    output = model(input_tensor)
    print('=== CFM-BCL ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
