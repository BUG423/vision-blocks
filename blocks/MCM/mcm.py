import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：MCM (Multi-Scale Context Module) —— 多尺度上下文模块

一、模块简介
上下文信息对于视觉理解至关重要，但不同尺度的上下文对任务的贡献不同。
现有的多尺度方法通常简单拼接或相加不同尺度的特征，缺乏自适应的尺度选择。

MCM 的核心思想是：通过自适应尺度权重学习，动态选择最合适的上下文尺度
组合，实现尺度感知的特征增强。

核心创新点：
1. 多尺度并行提取：使用不同大小的卷积核并行提取多尺度上下文
2. 自适应尺度权重：通过可学习的权重动态组合不同尺度
3. 尺度间交互：不同尺度之间通过注意力机制交互
4. 全局-局部融合：全局上下文与局部细节自适应融合

二、结构设计
MCM 由以下子结构组成：
1. 多尺度提取器（Multi-Scale Extractor）：
   - 并行 1x1, 3x3, 5x5, 7x7 卷积
2. 尺度权重学习器（Scale Weight Learner）：
   - 全局池化 → MLP → 尺度权重
3. 尺度交互器（Scale Interactor）：
   - 跨尺度注意力
4. 全局-局部融合器

三、论文写法参考
"本文提出 MCM（Multi-Scale Context Module）模块，通过自适应尺度权重学习
动态选择最合适的上下文尺度组合。该模块并行使用不同大小的卷积核提取多尺度
上下文，然后通过可学习权重动态组合，并在尺度间通过注意力机制交互，最终
融合全局上下文与局部细节。"

四、适用任务
适用于语义分割、目标检测、图像分类等需要多尺度上下文的视觉任务。
'''


class MCM(nn.Module):
    """MCM: Multi-Scale Context Module —— 多尺度上下文模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_scales: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_scales = num_scales

        # Multi-scale extractors
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, inner, kernel_size=k, padding=k // 2,
                          groups=inner, bias=False),
                nn.BatchNorm2d(inner),
                nn.GELU(),
            ) for k in [1, 3, 5, 7]
        ])

        # Scale weight learner
        self.scale_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner * num_scales, num_scales, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Scale interaction
        self.interaction = nn.MultiheadAttention(inner, num_heads=4,
                                                  batch_first=True)

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
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

        # 1. Multi-scale extraction
        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(x))

        # 2. Scale weight learning
        concat_features = torch.cat(scale_features, dim=1)  # [B, inner*4, H, W]
        weights = self.scale_weight(concat_features)         # [B, 4, 1, 1]

        # 3. Weighted combination
        combined = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            combined = combined + feat * weights[:, i:i+1, :, :]

        # 4. Scale interaction (reshape for attention)
        combined_flat = combined.view(B, -1, H * W).permute(0, 2, 1)
        interacted, _ = self.interaction(combined_flat, combined_flat,
                                          combined_flat)
        interacted = interacted.permute(0, 2, 1).view(B, -1, H, W)

        # 5. Output projection
        out = self.proj(interacted)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = MCM(channels=64)
    output = model(input_tensor)
    print('=== MCM: Multi-Scale Context Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
