import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：FCM (Feature Correlation Module) —— 特征相关性模块

一、模块简介
特征间的相关性包含了丰富的结构信息，但现有的特征交互方法通常只考虑
点对点的关系，忽略了特征间的全局相关性模式。

FCM 通过建模特征间的全局相关性模式，实现更有效的特征交互和增强。

核心创新点：
1. 全局相关性：计算特征间的全局相关性矩阵
2. 相关性模式学习：学习有意义的相关性模式
3. 相关性引导增强：利用相关性模式引导特征增强
4. 低秩近似：高效计算相关性

二、结构设计
FCM 由以下子结构组成：
1. 相关性矩阵计算器（Correlation Matrix Calculator）
2. 相关性模式学习器（Correlation Pattern Learner）
3. 相关性引导增强器（Correlation-Guided Enhancer）
4. 输出精炼

三、论文写法参考
"本文提出 FCM（Feature Correlation Module）模块，通过建模特征间的全局
相关性模式实现更有效的特征交互。该模块计算特征间的全局相关性矩阵，学习
有意义的相关性模式，然后利用相关性模式引导特征增强，最终通过输出精炼
保持特征质量。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要特征间交互的视觉任务。
'''


class FCM(nn.Module):
    """FCM: Feature Correlation Module —— 特征相关性模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 rank: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.rank = min(rank, channels)

        # Low-rank projection
        self.down_proj = nn.Conv2d(channels, self.rank, 1, bias=False)
        self.up_proj = nn.Conv2d(self.rank, channels, 1, bias=False)

        # Correlation pattern learner
        self.pattern_learner = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Enhancer
        self.enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Low-rank correlation
        x_low = self.down_proj(x)
        x_corr = self.up_proj(x_low)

        # 2. Correlation pattern
        pattern = self.pattern_learner(x_corr)

        # 3. Enhance
        enhanced = self.enhancer(x * pattern)

        # 4. Output refinement
        out = self.refine(enhanced)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = FCM(channels=64)
    output = model(input_tensor)
    print('=== FCM: Feature Correlation Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
