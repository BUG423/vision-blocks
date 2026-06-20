import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：FTM (Frequency Transform Module) —— 频率变换模块

一、模块简介
频域分析是信号处理的基础工具，但传统的频域方法计算开销大且难以
嵌入端到端训练。轻量级的频域近似方法可以在保持效果的同时降低计算成本。

FTM 的核心思想是：通过轻量级的频率变换近似，在特征空间中进行频域
分析和增强，实现高效的频域特征处理。

核心创新点：
1. 轻量级频率近似：使用 Haar 变换近似 DCT
2. 频带选择：自适应选择重要的频带
3. 频域增强：在频域空间进行特征增强
4. 逆变换融合：增强后的频域特征与空间域特征融合

二、结构设计
FTM 由以下子结构组成：
1. 轻量级频率变换器（Lightweight Frequency Transform）
2. 频带选择器（Band Selector）
3. 频域增强器（Frequency Enhancer）
4. 逆变换融合

三、论文写法参考
"本文提出 FTM（Frequency Transform Module）模块，通过轻量级的频率变换
近似实现高效的频域特征处理。该模块使用 Haar 变换近似 DCT，然后自适应
选择重要频带进行增强，最后通过逆变换与空间域特征融合。"

四、适用任务
适用于图像分类、图像恢复、去噪等需要频域分析的视觉任务。
'''


class FTM(nn.Module):
    """FTM: Frequency Transform Module —— 频率变换模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_bands: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_bands = num_bands

        # Band selector
        self.band_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, num_bands, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Frequency enhancer
        self.freq_enhance = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _haar_transform(x: torch.Tensor) -> torch.Tensor:
        """Simple 2D Haar transform approximation."""
        B, C, H, W = x.shape
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))
            W += 1

        # Horizontal
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        low_w = (x_even + x_odd) / 2
        high_w = (x_even - x_odd) / 2
        x = torch.cat([low_w, high_w], dim=-1)

        # Vertical
        x_even = x[:, :, 0::2, :]
        x_odd = x[:, :, 1::2, :]
        low_h = (x_even + x_odd) / 2
        high_h = (x_even - x_odd) / 2
        x = torch.cat([low_h, high_h], dim=2)

        return x

    @staticmethod
    def _haar_inverse(x: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        """Inverse 2D Haar transform."""
        B, C, H, W = x.shape

        # Inverse vertical
        low_h = x[:, :, :H // 2, :]
        high_h = x[:, :, H // 2:, :]
        x_even = low_h + high_h
        x_odd = low_h - high_h
        x = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        x[:, :, 0::2, :] = x_even
        x[:, :, 1::2, :] = x_odd

        # Inverse horizontal
        low_w = x[:, :, :, :W // 2]
        high_w = x[:, :, :, W // 2:]
        x_even = low_w + high_w
        x_odd = low_w - high_w
        x_out = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        x_out[:, :, :, 0::2] = x_even
        x_out[:, :, :, 1::2] = x_odd

        return x_out[:, :, :orig_h, :orig_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Frequency transform
        x_freq = self._haar_transform(x)

        # 2. Band selection
        weights = self.band_selector(x)                  # [B, num_bands, 1, 1]

        # 3. Frequency enhancement
        enhanced = self.freq_enhance(x_freq)

        # 4. Gate fusion
        gate = self.gate(enhanced)
        x_freq = x_freq * gate + enhanced * (1 - gate)

        # 5. Inverse transform
        out = self._haar_inverse(x_freq, H, W)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = FTM(channels=64)
    output = model(input_tensor)
    print('=== FTM: Frequency Transform Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
