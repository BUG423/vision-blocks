import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：IPM (Iterative Processing Module) —— 迭代处理模块

一、模块简介
深度网络的特征处理通常是单次前向传播，但某些任务需要多次迭代处理
才能获得最佳结果。IPM 通过迭代处理机制，让特征在多次迭代中逐步优化。

核心创新点：
1. 迭代处理：通过多次迭代逐步优化特征
2. 残差累积：每次迭代的结果累积到最终结果
3. 迭代门控：控制每次迭代的处理强度
4. 自适应迭代次数：根据输入复杂度调整迭代次数

二、结构设计
IPM 由以下子结构组成：
1. 迭代处理器（Iterative Processor）
2. 残差累积器（Residual Accumulator）
3. 迭代门控（Iteration Gate）
4. 自适应迭代控制器（Adaptive Iteration Controller）

三、论文写法参考
"本文提出 IPM（Iterative Processing Module）模块，通过迭代处理机制让特征
在多次迭代中逐步优化。该模块每次迭代对特征进行处理，通过残差累积累积
迭代结果，并通过迭代门控控制处理强度，最后通过自适应迭代控制器根据输入
复杂度调整迭代次数。"

四、适用任务
适用于图像恢复、去噪、超分辨率等需要迭代优化的视觉任务。
'''


class IPM(nn.Module):
    """IPM: Iterative Processing Module —— 迭代处理模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 max_iterations: int = 5):
        super().__init__()
        inner = max(1, channels // reduction)
        self.max_iterations = max_iterations

        # Iterative processor
        self.processor = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Residual accumulator
        self.accumulator = nn.Conv2d(channels, channels, 1, bias=False)

        # Iteration gate
        self.iter_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Adaptive iteration controller
        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Determine iteration count
        complexity = self.controller(x)                   # [B, 1, 1, 1]
        num_iter = int((complexity.mean() * self.max_iterations).item()) + 1
        num_iter = min(num_iter, self.max_iterations)

        # 2. Iterative processing
        accumulated = torch.zeros_like(x)
        current = x

        for _ in range(num_iter):
            processed = self.processor(current)
            gate = self.iter_gate(processed)
            current = current + processed * gate
            accumulated = accumulated + processed

        # 3. Final accumulation
        out = self.accumulator(accumulated)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = IPM(channels=64)
    output = model(input_tensor)
    print('=== IPM: Iterative Processing Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
