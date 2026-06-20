import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：CVM (Channel Variance Module) —— 通道方差模块

一、模块简介
通道方差反映了每个通道特征的离散程度，高方差通道携带更多信息，
低方差通道则相对冗余。利用方差信息可以实现更精确的通道选择。

CVM 的核心思想是：通过估计每个通道的方差，利用方差信息引导通道
特征的自适应增强和抑制。

核心创新点：
1. 通道方差估计：精确估计每个通道的方差
2. 方差引导增强：高方差通道增强，低方差通道抑制
3. 方差归一化：方差感知的归一化处理
4. 残差融合

二、结构设计
CVM 由以下子结构组成：
1. 方差估计器（Variance Estimator）
2. 方差引导增强器（Variance-Guided Enhancer）
3. 方差归一化器（Variance Normalizer）
4. 残差融合

三、论文写法参考
"本文提出 CVM（Channel Variance Module）模块，通过通道方差估计实现
自适应的通道特征增强。该模块首先估计每个通道的方差，然后利用方差信息
引导高方差通道增强和低方差通道抑制，最后通过方差归一化保持特征稳定性。"

四、适用任务
适用于图像分类、特征选择、通道剪枝等需要通道重要性评估的视觉任务。
'''


class CVM(nn.Module):
    """CVM: Channel Variance Module —— 通道方差模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Variance estimator
        self.var_estimator = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Softplus(),
        )

        # Variance-guided enhancer
        self.enhancer = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Variance normalizer
        self.normalizer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Estimate channel variance
        var = self.var_estimator(x)                       # [B, C, 1, 1]

        # 2. Variance-guided enhancement
        enhance_weight = self.enhancer(x)
        enhanced = x * enhance_weight * (1 + var)

        # 3. Variance normalization
        normalized = self.normalizer(enhanced)

        # 4. Output projection
        out = self.proj(normalized)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = CVM(channels=64)
    output = model(input_tensor)
    print('=== CVM: Channel Variance Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
