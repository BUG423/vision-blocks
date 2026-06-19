import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：FGM (Feature Gating Module) —— 特征门控模块

一、模块简介
特征选择是深度网络中的关键操作，但现有的门控机制通常只考虑单个通道的
重要性，忽略了通道间的协作关系。

FGM 的核心思想是：通过协作门控机制，让通道之间相互影响门控决策，实现
更精确的特征选择。

核心创新点：
1. 协作门控：通道门控值由所有通道共同决定
2. 局部上下文感知：门控考虑局部空间上下文
3. 双向门控：前向和后向通道交互
4. 稀疏激活：鼓励稀疏的特征选择

二、结构设计
FGM 由以下子结构组成：
1. 协作门控生成器（Cooperative Gate Generator）：
   - 通道交互 → 协作门控值
2. 局部上下文门控（Local Context Gate）：
   - 空间上下文 → 局部门控
3. 双向融合（Bidirectional Fusion）：
   - 前向/后向通道交互
4. 稀疏激活与残差

三、论文写法参考
"本文提出 FGM（Feature Gating Module）模块，通过协作门控机制实现通道间
相互影响的特征选择。该模块让通道门控值由所有通道共同决定，并结合局部空间
上下文和双向通道交互，最终通过稀疏激活鼓励精确的特征选择。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要精确特征选择的视觉任务。
'''


class FGM(nn.Module):
    """FGM: Feature Gating Module —— 特征门控模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 sparse_ratio: float = 0.5):
        super().__init__()
        inner = max(1, channels // reduction)
        self.sparse_ratio = sparse_ratio

        # Cooperative gate generator
        self.coop_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Local context gate
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        # Bidirectional channel interaction
        self.forward_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.backward_conv = nn.Conv1d(channels, channels, 1, bias=False)

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
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

        # 1. Cooperative gate
        coop = self.coop_gate(x)                         # [B, C, 1, 1]

        # 2. Local context gate
        local = self.local_gate(x)                       # [B, C, H, W]

        # 3. Bidirectional channel interaction
        x_flat = x.view(B, C, -1)                        # [B, C, H*W]
        x_mean = x_flat.mean(dim=-1, keepdim=True)       # [B, C, 1]
        forward = self.forward_conv(x_mean).sigmoid()     # [B, C, 1]
        backward = self.backward_conv(x_mean).sigmoid()   # [B, C, 1]
        bidir = (forward + backward) / 2                  # [B, C, 1]

        # 4. Combine gates
        gate = coop * local * bidir

        # 5. Sparse activation
        gate_flat = gate.view(B, C, -1)
        k = int(C * self.sparse_ratio)
        _, topk_idx = gate_flat.topk(k, dim=1)
        sparse_mask = torch.zeros_like(gate_flat)
        sparse_mask.scatter_(1, topk_idx, 1.0)
        gate = gate * sparse_mask.view(B, C, H, W)

        # 6. Apply gate
        out = self.proj(x * gate)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = FGM(channels=64)
    output = model(input_tensor)
    print('=== FGM: Feature Gating Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
