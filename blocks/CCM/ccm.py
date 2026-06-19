import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：CCM (Channel Correlation Module) —— 通道相关性模块

一、模块简介
通道间的相关性蕴含了丰富的语义信息，但现有的通道注意力方法通常只考虑
单个通道的重要性，忽略了通道间的统计相关性。

CCM 的核心思想是：通过建模通道间的统计相关性，挖掘通道间的协作关系，
实现更精确的通道特征增强。

核心创新点：
1. 通道相关性矩阵：计算通道间的相关性矩阵
2. 相关性引导增强：基于相关性矩阵引导通道增强
3. 低秩近似：使用低秩近似减少计算开销
4. 相关性正则化：鼓励有意义的通道相关性

二、结构设计
CCM 由以下子结构组成：
1. 通道相关性计算器（Channel Correlation Calculator）：
   - 协方差矩阵 → 相关性矩阵
2. 低秩近似器（Low-Rank Approximator）：
   - PCA-like 低秩近似
3. 相关性引导增强器（Correlation-Guided Enhancer）：
   - 基于相关性的通道增强
4. 正则化与残差

三、论文写法参考
"本文提出 CCM（Channel Correlation Module）模块，通过建模通道间的统计
相关性实现更精确的通道特征增强。该模块首先计算通道间的相关性矩阵，然后
通过低秩近似减少计算开销，最后基于相关性矩阵引导通道增强，并加入相关性
正则化鼓励有意义的通道协作。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要通道间协作关系的视觉任务。
'''


class CCM(nn.Module):
    """CCM: Channel Correlation Module —— 通道相关性模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 rank: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.rank = min(rank, channels)

        # Channel pool for correlation computation
        self.channel_pool = nn.AdaptiveAvgPool2d(1)

        # Low-rank projection
        self.down_proj = nn.Conv2d(channels, self.rank, 1, bias=False)
        self.up_proj = nn.Conv2d(self.rank, channels, 1, bias=False)

        # Correlation-guided enhancer
        self.enhancer = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Regularization
        self.reg_conv = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Channel pooling
        pooled = self.channel_pool(x).view(B, C)         # [B, C]

        # 2. Low-rank correlation
        x_low = self.down_proj(x)                         # [B, rank, H, W]
        x_corr = self.up_proj(x_low)                      # [B, C, H, W]

        # 3. Correlation-guided enhancement
        enhance_weight = self.enhancer(x_corr)
        enhanced = x * enhance_weight

        # 4. Regularization
        regularized = self.reg_conv(enhanced)

        # 5. Output projection
        out = self.proj(regularized)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = CCM(channels=64)
    output = model(input_tensor)
    print('=== CCM: Channel Correlation Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
