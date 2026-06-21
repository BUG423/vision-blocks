import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：DFM (Dynamic Feature Module) —— 动态特征模块

一、模块简介
特征在不同输入下应该有不同的处理方式。DFM 通过动态生成特征处理
参数，实现输入自适应的特征增强。

核心创新点：
1. 动态参数生成：根据输入动态生成处理参数
2. 内容感知：处理方式随输入内容变化
3. 轻量级预测：仅生成调制参数而非完整权重
4. 实时适应：推理时实时调整处理策略

二、结构设计
DFM 由以下子结构组成：
1. 内容编码器（Content Encoder）
2. 参数预测器（Parameter Predictor）
3. 动态特征处理器（Dynamic Feature Processor）
4. 输出精炼

三、论文写法参考
"本文提出 DFM（Dynamic Feature Module）模块，通过动态生成特征处理参数实现
输入自适应的特征增强。该模块首先编码输入内容特征，然后预测动态处理参数，
最后用动态参数对特征进行自适应处理。"

四、适用任务
适用于图像分类、风格迁移、域自适应等需要输入自适应处理的视觉任务。
'''


class DFM(nn.Module):
    """DFM: Dynamic Feature Module —— 动态特征模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Content encoder
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Parameter predictor
        self.scale_predictor = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.shift_predictor = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Tanh(),
        )

        # Dynamic processor
        self.processor = nn.Sequential(
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
        B, C, H, W = x.shape

        # 1. Content encoding
        content = self.encoder(x)

        # 2. Dynamic parameter prediction
        scale = self.scale_predictor(content).expand(-1, -1, H, W)
        shift = self.shift_predictor(content).expand(-1, -1, H, W)

        # 3. Dynamic processing
        dynamic = x * scale + shift

        # 4. Process
        processed = self.processor(dynamic)

        # 5. Output refinement
        out = self.refine(processed)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = DFM(channels=64)
    output = model(input_tensor)
    print('=== DFM: Dynamic Feature Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
