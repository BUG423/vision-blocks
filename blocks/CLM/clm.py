import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：CLM (Context Learning Module) —— 上下文学习模块

一、模块简介
上下文信息对于视觉理解至关重要，但现有的上下文建模方法通常只考虑
单一类型的上下文。CLM 通过学习多种上下文类型并自适应融合，实现
更全面的上下文理解。

核心创新点：
1. 多类型上下文：学习空间、通道、全局等多种上下文
2. 上下文选择：自适应选择最相关的上下文类型
3. 上下文交互：不同类型上下文相互补充
4. 学习型融合：通过学习确定最佳融合方式

二、结构设计
CLM 由以下子结构组成：
1. 多类型上下文提取器（Multi-Type Context Extractor）
2. 上下文选择器（Context Selector）
3. 上下文交互器（Context Interactor）
4. 学习型融合器（Learnable Fusion）

三、论文写法参考
"本文提出 CLM（Context Learning Module）模块，通过学习多种上下文类型并
自适应融合实现更全面的上下文理解。该模块提取空间、通道、全局等多种上下文，
然后自适应选择最相关的上下文类型，并通过上下文交互和学习型融合获得最终
的上下文表示。"

四、适用任务
适用于语义分割、场景理解、图像分类等需要丰富上下文的视觉任务。
'''


class CLM(nn.Module):
    """CLM: Context Learning Module —— 上下文学习模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_contexts: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_contexts = num_contexts

        # Spatial context
        self.spatial_ctx = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Channel context
        self.channel_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Global context
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Context selector
        self.selector = nn.Sequential(
            nn.Conv2d(inner * num_contexts, num_contexts, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Context interaction
        self.interaction = nn.Conv2d(inner, inner, 3, padding=1,
                                     groups=inner, bias=False)

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Multi-type context
        spatial = self.spatial_ctx(x)
        channel = self.channel_ctx(x).expand(-1, -1, H, W)
        global_ = self.global_ctx(x).expand(-1, -1, H, W)

        # 2. Context selection
        concat = torch.cat([spatial, channel, global_], dim=1)
        weights = self.selector(concat)

        # 3. Weighted fusion
        fused = (spatial * weights[:, 0:1] +
                 channel * weights[:, 1:2] +
                 global_ * weights[:, 2:3])

        # 4. Context interaction
        interacted = self.interaction(fused)

        # 5. Output projection
        out = self.proj(interacted)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = CLM(channels=64)
    output = model(input_tensor)
    print('=== CLM: Context Learning Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
