import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：DEM-BCL (Dense Evolution Module - BCL) —— 密集进化模块（BCL版）

一、模块简介
本模块是 DEM（Dense Evolution Module）针对 BCL 时序数据格式的适配版本。
在 BCL 格式中，时序数据需要在时间维度上进行特征进化。本模块将 DEM 的
"变异-选择-保留"机制扩展到时序数据，通过时间维度的特征进化增强时序
表征能力。

核心创新点：
1. 时间维度变异生成：通过不同大小的 1D 卷积生成时序特征变体
2. 信息熵适应度评估：基于时间局部方差估计每个变体的信息量
3. Top-k 自然选择：选择信息量最高的变体进行保留
4. 进化融合：选择后的变体与原始特征融合

二、结构设计
DEM-BCL 由以下子结构组成：
1. 变异生成器（Variation Generator）：
   - 多尺度 1D 卷积（1x1, 3x1, 5x1）生成特征变体
2. 适应度评估器（Fitness Evaluator）：
   - 时间局部方差估计信息量
3. 选择器（Selector）：
   - Top-k 选择保留高质量变体
4. 进化融合器（Evolution Fusion）
'''


class DEM_BCL(nn.Module):
    """DEM-BCL: Dense Evolution Module for BCL format"""

    def __init__(self, channels: int, seq_len: int,
                 reduction: int = 4, num_variants: int = 4,
                 top_k: int = 2):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_variants = num_variants
        self.top_k = min(top_k, num_variants)

        # Variation generator
        self.variants = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, inner, 1, bias=False),
                nn.BatchNorm1d(inner),
                nn.GELU(),
                nn.Conv1d(inner, channels, 1, bias=False),
            ) for _ in range(num_variants)
        ])

        # Fitness evaluator
        self.fitness_conv = nn.Conv1d(channels, 1, 1, bias=False)

        # Evolution fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def _estimate_fitness(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate fitness via temporal local variance."""
        mu = F.avg_pool1d(x, 3, stride=1, padding=1)
        mu_sq = F.avg_pool1d(x * x, 3, stride=1, padding=1)
        var = (mu_sq - mu * mu).clamp(min=1e-6)
        fitness = self.fitness_conv(var)
        return fitness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, T] (BCL格式时序数据)
        输出:
            out: Tensor, shape = [B, C, T]
        """
        B, C, T = x.shape

        # 1. Generate variants
        variant_list = []
        for variant_fn in self.variants:
            v = variant_fn(x) + x
            variant_list.append(v)

        # 2. Evaluate fitness
        fitness_scores = []
        for v in variant_list:
            fitness = self._estimate_fitness(v)
            fitness_map = fitness.mean(dim=-1)  # [B, C]
            fitness_scores.append(fitness_map.mean(dim=-1))  # [B]

        # Stack: [B, num_variants]
        fitness_stack = torch.stack(fitness_scores, dim=1)

        # 3. Select top-k
        _, top_indices = fitness_stack.topk(self.top_k, dim=1)

        # Gather and average selected variants
        selected = []
        for i in range(self.top_k):
            idx = top_indices[:, i]
            for b in range(B):
                selected.append(variant_list[idx[b]][b:b+1])
        selected = torch.cat(selected, dim=0)
        selected = selected.view(B, self.top_k, C, T).mean(dim=1)

        # 4. Evolution fusion
        gate = self.fusion_gate(selected)
        out = selected * gate + x * (1 - gate)

        # 5. Refine
        out = self.refine(out)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 128)
    model = DEM_BCL(channels=64, seq_len=128)
    output = model(input_tensor)
    print('=== DEM-BCL ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
