import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：DWM (Dynamic Weight Module) —— 动态权重模块

一、模块简介
卷积核的权重在推理时是固定的，无法适应不同的输入内容。动态卷积通过
预测卷积核权重来实现输入自适应处理，但通常需要生成大量权重参数。

DWM 的核心思想是：通过轻量级的权重预测网络，动态生成卷积核的调制因子，
而非直接生成完整权重，从而在保持动态性的同时控制计算开销。

核心创新点：
1. 轻量级权重预测：预测调制因子而非完整权重
2. 内容感知：权重预测基于输入内容
3. 渐进式调制：多层级渐进式调整权重
4. 正则化约束：权重变化的平滑性约束

二、结构设计
DWM 由以下子结构组成：
1. 内容编码器（Content Encoder）：
   - 全局池化 + MLP 编码输入内容
2. 调制因子预测器（Modulation Factor Predictor）：
   - 预测通道级调制因子
3. 渐进式调制（Progressive Modulation）：
   - 多层级渐进式权重调整
4. 平滑性约束

三、论文写法参考
"本文提出 DWM（Dynamic Weight Module）模块，通过轻量级的权重预测网络
动态生成卷积核的调制因子。该模块首先编码输入内容特征，然后预测通道级
调制因子对卷积权重进行动态调整，并通过渐进式调制和平滑性约束保持权重
变化的稳定性。"

四、适用任务
适用于图像分类、目标检测、风格迁移等需要输入自适应处理的视觉任务。
'''


class DWM(nn.Module):
    """DWM: Dynamic Weight Module —— 动态权重模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_levels: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_levels = num_levels

        # Content encoder
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Modulation factor predictors (multi-level)
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inner, inner, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(inner, channels, 1, bias=False),
                nn.Tanh(),
            ) for _ in range(num_levels)
        ])

        # Smoothness constraint
        self.smooth_conv = nn.Conv2d(channels, channels, 3, padding=1,
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
        # 1. Content encoding
        content = self.encoder(x)                        # [B, inner, 1, 1]

        # 2. Multi-level modulation
        modulated = x
        for predictor in self.predictors:
            factor = predictor(content)                  # [B, C, 1, 1]
            factor = factor.expand_as(modulated)
            modulated = modulated * (1 + factor * 0.1)

        # 3. Smoothness constraint
        smoothed = self.smooth_conv(modulated)
        modulated = modulated * 0.5 + smoothed * 0.5

        # 4. Output projection
        out = self.proj(modulated)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = DWM(channels=64)
    output = model(input_tensor)
    print('=== DWM: Dynamic Weight Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
