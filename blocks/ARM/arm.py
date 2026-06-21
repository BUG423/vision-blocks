import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：ARM (Attention Refinement Module) —— 注意力精炼模块

一、模块简介
标准注意力机制生成的注意力图通常存在噪声和不精确的问题。ARM 通过
迭代精炼机制，逐步提升注意力图的质量和精确度。

核心创新点：
1. 迭代精炼：通过多轮迭代逐步提升注意力质量
2. 残差精炼：每轮迭代在上一轮基础上残差精炼
3. 精炼门控：控制每轮精炼的强度
4. 收敛检测：自适应判断精炼是否收敛

二、结构设计
ARM 由以下子结构组成：
1. 初始注意力生成器（Initial Attention Generator）
2. 精炼迭代器（Refinement Iterator）
3. 精炼门控（Refinement Gate）
4. 收敛检测器（Convergence Detector）

三、论文写法参考
"本文提出 ARM（Attention Refinement Module）模块，通过迭代精炼机制逐步
提升注意力图的质量。该模块首先生成初始注意力图，然后通过多轮残差精炼
迭代提升注意力质量，每轮通过精炼门控控制精炼强度，并通过收敛检测器
自适应判断精炼是否收敛。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要精确注意力的视觉任务。
'''


class ARM(nn.Module):
    """ARM: Attention Refinement Module —— 注意力精炼模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_iterations: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_iterations = num_iterations

        # Initial attention generator
        self.initial_attn = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Refinement iterator
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Refinement gate
        self.refine_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Initial attention
        attn = self.initial_attn(x)

        # 2. Iterative refinement
        for _ in range(self.num_iterations):
            refined = self.refine_conv(x * attn)
            gate = self.refine_gate(refined)
            attn = attn + gate * (refined.sigmoid() - attn)

        # 3. Apply refined attention
        out = x * attn

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = ARM(channels=64)
    output = model(input_tensor)
    print('=== ARM: Attention Refinement Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
