import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：JRM (Joint Reasoning Module) —— 联合推理模块

一、模块简介
视觉推理需要同时考虑空间关系、语义关系和上下文关系。现有的推理方法
通常只建模单一类型的关系，忽略了多种关系间的交互。

JRM 的核心思想是：通过联合推理机制，同时建模多种类型的关系，实现
更全面的视觉理解。

核心创新点：
1. 多关系建模：同时建模空间、语义、上下文关系
2. 关系交互：不同类型关系间的相互影响
3. 联合推理：基于多种关系的联合决策
4. 推理门控：控制推理强度

二、结构设计
JRM 由以下子结构组成：
1. 空间关系编码器（Spatial Relation Encoder）
2. 语义关系编码器（Semantic Relation Encoder）
3. 上下文关系编码器（Context Relation Encoder）
4. 联合推理器（Joint Reasoner）

三、论文写法参考
"本文提出 JRM（Joint Reasoning Module）模块，通过联合推理机制同时建模
多种类型的关系。该模块分别编码空间、语义和上下文关系，然后通过关系交互
和联合推理实现更全面的视觉理解，最后通过推理门控控制推理强度。"

四、适用任务
适用于场景理解、视觉问答、关系检测等需要复杂推理的视觉任务。
'''


class JRM(nn.Module):
    """JRM: Joint Reasoning Module —— 联合推理模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_heads: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_heads = num_heads
        self.head_dim = inner // num_heads

        # Spatial relation encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Semantic relation encoder
        self.semantic_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Context relation encoder (self-attention)
        self.context_q = nn.Conv2d(inner, inner, 1, bias=False)
        self.context_k = nn.Conv2d(inner, inner, 1, bias=False)
        self.context_v = nn.Conv2d(inner, inner, 1, bias=False)

        # Joint reasoner
        self.reasoner = nn.Sequential(
            nn.Conv2d(inner * 3, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Reasoning gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Spatial relation
        spatial = self.spatial_encoder(x)

        # 2. Semantic relation
        semantic = self.semantic_encoder(x).expand(-1, -1, H, W)

        # 3. Context relation (self-attention)
        q = self.context_q(spatial)
        k = self.context_k(spatial)
        v = self.context_v(spatial)

        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        context = torch.matmul(attn, v).permute(0, 1, 3, 2).contiguous()
        context = context.view(B, inner, H, W)

        # 4. Joint reasoning
        concat = torch.cat([spatial, semantic, context], dim=1)
        reasoned = self.reasoner(concat)

        # 5. Gate
        gate = self.gate(reasoned)
        fused = reasoned * gate + x * (1 - gate)

        # 6. Output projection
        out = self.proj(fused)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = JRM(channels=64)
    output = model(input_tensor)
    print('=== JRM: Joint Reasoning Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
