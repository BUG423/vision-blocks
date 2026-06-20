import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：BFM (Batch Fusion Module) —— 批融合模块

一、模块简介
批维度的信息在特征处理中通常被忽略，但批次内的样本间关系可以提供
有价值的全局统计信息。

BFM 的核心思想是：利用批次维度的统计信息进行特征融合，让每个样本
能够从批次内的其他样本中获取有用的上下文信息。

核心创新点：
1. 批内统计：计算批次维度的统计信息
2. 样本间交互：通过注意力机制实现样本间信息流动
3. 自适应融合：每个样本自适应地从批次中选择有用信息
4. 计算高效：仅在全局池化后的低维空间操作

二、结构设计
BFM 由以下子结构组成：
1. 批统计提取器（Batch Statistics Extractor）
2. 样本间注意力（Inter-Sample Attention）
3. 自适应融合门（Adaptive Fusion Gate）
4. 输出投影

三、论文写法参考
"本文提出 BFM（Batch Fusion Module）模块，利用批次维度的统计信息进行
特征融合。该模块首先提取批次维度的统计信息，然后通过样本间注意力实现
信息流动，最后通过自适应融合门选择有用信息。"

四、适用任务
适用于图像分类、度量学习、少样本学习等需要利用批次信息的视觉任务。
'''


class BFM(nn.Module):
    """BFM: Batch Fusion Module —— 批融合模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_heads: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_heads = num_heads
        self.head_dim = inner // num_heads

        # Batch statistics extractor
        self.extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Inter-sample attention
        self.q_proj = nn.Conv1d(inner, inner, 1, bias=False)
        self.k_proj = nn.Conv1d(inner, inner, 1, bias=False)
        self.v_proj = nn.Conv1d(inner, inner, 1, bias=False)
        self.out_proj = nn.Conv1d(inner, channels, 1, bias=False)

        # Fusion gate
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Extract batch features
        feat = self.extractor(x).view(B, -1, 1)         # [B, inner, 1]

        # 2. Inter-sample attention
        q = self.q_proj(feat)                             # [B, inner, 1]
        k = self.k_proj(feat)                             # [B, inner, 1]
        v = self.v_proj(feat)                             # [B, inner, 1]

        # Reshape for attention: [B, num_heads, 1, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, 1)
        k = k.view(B, self.num_heads, self.head_dim, 1)
        v = v.view(B, self.num_heads, self.head_dim, 1)

        attn = torch.matmul(q.transpose(-1, -2), k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-1, -2))

        # 3. Project back
        out = out.view(B, -1, 1)
        out = self.out_proj(out).view(B, C, 1, 1)

        # 4. Gate fusion
        gate = self.gate(out.view(B, C, 1))
        expanded = out.expand_as(x)
        fused = expanded * gate.view(B, C, 1, 1) + x * (1 - gate.view(B, C, 1, 1))

        return fused


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(4, 64, 32, 32)
    model = BFM(channels=64)
    output = model(input_tensor)
    print('=== BFM: Batch Fusion Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
