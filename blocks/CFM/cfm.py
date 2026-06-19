import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：CFM (Channel Frequency Mixer) —— 通道频率混合模块

一、模块简介
在卷积神经网络中，通道间的交互通常通过 1x1 卷积或通道注意力实现，这些
方法在空间域上操作，难以直接捕捉通道间的频域关联。实际上，不同通道的特征
在频域上呈现出互补的频率模式——某些通道主要携带低频全局信息，而另一些通道
则编码高频细节。

CFM 的核心思想是：将通道特征变换到频域空间，通过可学习的频率混合矩阵实现
跨通道的频率信息交互，让高频通道和低频通道能够相互补充，从而增强通道间的
信息流动。

核心创新点：
1. 频域通道混合：将通道特征通过 DCT 变换到频域，在频域空间进行通道间
   的信息交换，比空间域混合更具全局性
2. 频率感知缩放：根据每个通道的频率特性（高频/低频）自适应调整混合权重
3. 残差频率门控：通过可学习门控控制频率混合的强度，避免过度混合导致的
   信息损失
4. 逆变换保真：混合后的频域特征通过 IDCT 变换回空间域，保持特征完整性

二、结构设计
CFM 由以下子结构组成：
1. 频域变换器（Frequency Transform）：
   - 2D DCT 将空间特征变换到频域
2. 通道频率混合器（Channel Frequency Mixer）：
   - 可学习频率混合矩阵 → 跨通道频率信息交换
3. 频率感知缩放器（Frequency-Aware Scaler）：
   - 通道频率特性编码 → 自适应缩放因子
4. 残差门控与逆变换（Residual Gate & Inverse Transform）：
   - 门控混合强度控制 → 2D IDCT 逆变换

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 CFM（Channel Frequency Mixer）模块，通过频域通道混合实现跨通道
的频率信息交互。该模块首先将通道特征通过 DCT 变换到频域空间，然后用可学习
的频率混合矩阵进行跨通道信息交换，同时根据每个通道的频率特性自适应调整
混合权重；最后通过残差门控控制混合强度，并经 IDCT 逆变换回空间域。该设计
使高频通道和低频通道能够相互补充，增强了通道间的信息流动。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要增强通道间频率信息交互的任务。
'''


class CFM(nn.Module):
    """CFM: Channel Frequency Mixer —— 通道频率混合模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 mix_groups: int = 8):
        super().__init__()
        self.channels = channels
        self.mix_groups = mix_groups
        self.group_size = channels // mix_groups

        # Frequency-aware channel scaler
        self.channel_scaler = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel frequency mixing matrix (learnable)
        self.mix_matrix = nn.Parameter(
            torch.eye(channels).unsqueeze(0).unsqueeze(0)
        )

        # Residual gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _dct_2d(x: torch.Tensor) -> torch.Tensor:
        """Simple 2D DCT approximation using Haar-like transform."""
        B, C, H, W = x.shape
        # Pad to even size
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        # Horizontal transform
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        low_w = (x_even + x_odd) / 2
        high_w = (x_even - x_odd) / 2
        x_freq = torch.cat([low_w, high_w], dim=-1)

        # Vertical transform
        x_even = x_freq[:, :, 0::2, :]
        x_odd = x_freq[:, :, 1::2, :]
        low_h = (x_even + x_odd) / 2
        high_h = (x_even - x_odd) / 2
        x_freq = torch.cat([low_h, high_h], dim=2)

        return x_freq

    @staticmethod
    def _idct_2d(x: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        """Inverse 2D DCT (inverse of _dct_2d)."""
        B, C, H, W = x.shape

        # Inverse vertical
        low_h = x[:, :, :H // 2, :]
        high_h = x[:, :, H // 2:, :]
        x_even = low_h + high_h
        x_odd = low_h - high_h
        x_spatial = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        x_spatial[:, :, 0::2, :] = x_even
        x_spatial[:, :, 1::2, :] = x_odd

        # Inverse horizontal
        H2 = x_spatial.shape[2]
        low_w = x_spatial[:, :, :, :W // 2]
        high_w = x_spatial[:, :, :, W // 2:]
        x_even = low_w + high_w
        x_odd = low_w - high_w
        x_out = torch.zeros(B, C, H2, W, device=x.device, dtype=x.dtype)
        x_out[:, :, :, 0::2] = x_even
        x_out[:, :, :, 1::2] = x_odd

        return x_out[:, :, :orig_h, :orig_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Frequency-aware channel scaling
        scale = self.channel_scaler(x)                 # [B, C, 1, 1]
        x_scaled = x * scale

        # 2. Transform to frequency domain
        x_freq = self._dct_2d(x_scaled)               # [B, C, H, W]

        # 3. Channel frequency mixing
        # Reshape for matrix multiplication: [B, C, H*W]
        x_flat = x_freq.view(B, C, -1)
        # Apply mixing: [B, C, C] @ [B, C, H*W] -> [B, C, H*W]
        mix_weight = torch.sigmoid(self.mix_matrix)
        x_mixed = torch.bmm(mix_weight.expand(B, -1, -1), x_flat)
        x_mixed = x_mixed.view(B, C, H, W)

        # 4. Residual gate
        gate = self.gate(x_mixed)
        x_out = x_mixed * gate + x_freq * (1 - gate)

        # 5. Inverse transform back to spatial domain
        out = self._idct_2d(x_out, H, W)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = CFM(channels=64)
    output = model(input_tensor)
    print('=== CFM: Channel Frequency Mixer ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
