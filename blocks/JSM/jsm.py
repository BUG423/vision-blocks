import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-20

'''
模块名称：JSM (Joint Selection Module) —— 联合选择模块

一、模块简介
特征选择是深度网络中的关键操作，但现有的选择方法通常只考虑单一维度的
选择。JSM 通过联合在空间和通道两个维度上进行选择，实现更精确的特征筛选。

核心创新点：
1. 联合选择：同时在空间和通道维度上选择
2. 双维度门控：空间门控和通道门控联合工作
3. 选择一致性：保持两个维度选择的一致性
4. 稀疏激活：鼓励稀疏的特征选择

二、结构设计
JSM 由以下子结构组成：
1. 空间选择器（Spatial Selector）
2. 通道选择器（Channel Selector）
3. 选择一致性约束（Selection Consistency）
4. 联合激活（Joint Activation）

三、论文写法参考
"本文提出 JSM（Joint Selection Module）模块，通过联合在空间和通道两个
维度上进行选择实现更精确的特征筛选。该模块分别生成空间选择权重和通道
选择权重，然后通过选择一致性约束保持两者的一致性，最后通过联合激活
实现稀疏的特征选择。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要精确特征选择的视觉任务。
'''


class JSM(nn.Module):
    """JSM: Joint Selection Module —— 联合选择模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 sparse_ratio: float = 0.5):
        super().__init__()
        inner = max(1, channels // reduction)
        self.sparse_ratio = sparse_ratio

        # Spatial selector
        self.spatial_selector = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # Channel selector
        self.channel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Selection consistency
        self.consistency = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Spatial selection
        spatial_weight = self.spatial_selector(x)         # [B, 1, H, W]

        # 2. Channel selection
        channel_weight = self.channel_selector(x)         # [B, C, 1, 1]

        # 3. Selection consistency
        concat = torch.cat([x * channel_weight,
                           spatial_weight.expand(-1, C, -1, -1)], dim=1)
        consistent = self.consistency(concat)

        # 4. Joint activation with sparsity
        joint_weight = consistent * spatial_weight.expand_as(consistent)

        # Apply sparsity
        joint_flat = joint_weight.view(B, C, -1)
        k = int(C * self.sparse_ratio * H * W)
        if k > 0:
            _, topk_idx = joint_flat.topk(min(k, C * H * W), dim=2)
            sparse_mask = torch.zeros_like(joint_flat)
            sparse_mask.scatter_(2, topk_idx, 1.0)
            joint_weight = joint_weight * sparse_mask.view(B, C, H, W)

        # 5. Apply selection
        out = self.proj(x * joint_weight)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = JSM(channels=64)
    output = model(input_tensor)
    print('=== JSM: Joint Selection Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
