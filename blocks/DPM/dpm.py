import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：DPM (Dense Prediction Module) —— 密集预测模块

一、模块简介
密集预测任务（如语义分割、深度估计）需要同时利用全局语义信息和
局部细节信息。现有的方法通常通过多尺度特征融合来实现，但缺乏
自适应的信息选择机制。

DPM 的核心思想是：通过密集的预测头网络，对每个空间位置进行自适应的
特征预测，同时利用全局和局部信息。

核心创新点：
1. 密集预测头：对每个空间位置独立预测
2. 全局-局部融合：同时利用全局语义和局部细节
3. 自适应权重：每个位置自适应选择信息来源
4. 多任务支持：支持多种密集预测任务

二、结构设计
DPM 由以下子结构组成：
1. 全局上下文提取器（Global Context Extractor）
2. 局部特征增强器（Local Feature Enhancer）
3. 自适应融合器（Adaptive Fusion）
4. 密集预测头（Dense Prediction Head）

三、论文写法参考
"本文提出 DPM（Dense Prediction Module）模块，通过密集的预测头网络实现
自适应的特征预测。该模块同时提取全局上下文和局部特征，然后通过自适应
融合器为每个位置选择最合适的信息来源，最后通过密集预测头输出预测结果。"

四、适用任务
适用于语义分割、深度估计、目标检测等密集预测视觉任务。
'''


class DPM(nn.Module):
    """DPM: Dense Prediction Module —— 密集预测模块"""

    def __init__(self, channels: int, num_classes: int,
                 reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Global context extractor
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Local feature enhancer
        self.local_enhance = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Dense prediction head
        self.head = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, num_classes, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Global context
        global_feat = self.global_context(x).expand(-1, -1, H, W)

        # 2. Local enhancement
        local_feat = self.local_enhance(x)

        # 3. Adaptive fusion
        concat = torch.cat([global_feat, local_feat], dim=1)
        gate = self.fusion_gate(concat)
        fused = global_feat * gate + local_feat * (1 - gate)

        # 4. Dense prediction
        out = self.head(fused)

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = DPM(channels=64, num_classes=10)
    output = model(input_tensor)
    print('=== DPM: Dense Prediction Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
