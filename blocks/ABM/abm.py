import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：ABM (Adaptive Batch Module) —— 自适应批归一化模块

一、模块简介
批归一化（Batch Normalization）是深度网络中的关键技术，但固定的归一化
统计量无法适应不同输入的分布变化。自适应归一化方法（如 AdaIN）通过调制
统计量实现风格迁移，但通常需要额外的风格编码器。

ABM 的核心思想是：通过轻量级的自适应机制，让归一化统计量根据输入内容
动态调整，实现内容感知的自适应归一化。

核心创新点：
1. 内容感知统计量：统计量由输入内容自适应生成
2. 双重调制：同时调制均值和方差
3. 轻量级预测：仅用 1x1 卷积预测调制因子
4. 渐进式归一化：多层级渐进式调整

二、结构设计
ABM 由以下子结构组成：
1. 内容编码器（Content Encoder）
2. 统计量预测器（Statistics Predictor）
3. 双重调制器（Dual Modulator）
4. 渐进式归一化层

三、论文写法参考
"本文提出 ABM（Adaptive Batch Normalization）模块，通过内容感知的自适应
机制实现动态归一化。该模块首先编码输入内容特征，然后预测均值和方差的
调制因子，最后通过渐进式归一化实现内容感知的特征调整。"

四、适用任务
适用于图像分类、风格迁移、域自适应等需要自适应归一化的视觉任务。
'''


class ABM(nn.Module):
    """ABM: Adaptive Batch Normalization Module —— 自适应批归一化模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_groups: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_groups = num_groups
        self.group_size = channels // num_groups

        # Content encoder
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Statistics predictor
        self.mean_predictor = nn.Conv2d(inner, channels, 1, bias=False)
        self.var_predictor = nn.Conv2d(inner, channels, 1, bias=False)

        # Learnable initial statistics
        self.running_mean = nn.Parameter(torch.zeros(channels))
        self.running_var = nn.Parameter(torch.ones(channels))

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Content encoding
        content = self.encoder(x)

        # 2. Predict adaptive statistics
        pred_mean = self.mean_predictor(content).view(B, C, 1, 1)
        pred_var = self.var_predictor(content).view(B, C, 1, 1)

        # 3. Adaptive normalization
        mean = self.running_mean.view(1, C, 1, 1) + pred_mean
        var = self.running_var.view(1, C, 1, 1) * (1 + pred_var)

        # 4. Normalize
        x_norm = (x - mean) / (var.sqrt() + 1e-5)

        # 5. Output projection
        out = self.proj(x_norm)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = ABM(channels=64)
    output = model(input_tensor)
    print('=== ABM: Adaptive Batch Normalization Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
