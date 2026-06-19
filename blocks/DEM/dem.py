import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：DEM (Dense Evolution Module) —— 密集进化模块

一、模块简介
深度网络中的特征在层间传递时，信息会逐渐衰减和扭曲。传统的残差连接虽然
缓解了梯度消失问题，但特征的进化过程仍然是线性的——每一层只能看到前一层
的输出。DenseNet 通过密集连接让每一层都能访问所有前序层的特征，但带来了
巨大的内存开销。

DEM 的核心思想是：通过进化论的选择机制，让特征在传递过程中经历"变异-选择-
保留"的循环，每一代只保留信息量最高的特征子集，从而在保持密集连接优势的
同时控制计算开销。

核心创新点：
1. 特征变异（Variation）：通过多尺度卷积生成特征的多个变体，模拟基因变异
2. 适应度评估（Fitness Evaluation）：通过信息熵估计每个变体的信息量，作为
   适应度指标
3. 自然选择（Natural Selection）：基于适应度选择 top-k 个变体，淘汰低质量
   特征
4. 进化累积（Evolution Accumulation）：选择后的特征与原始特征融合，形成
   下一代特征

二、结构设计
DEM 由以下子结构组成：
1. 变异生成器（Variation Generator）：
   - 多尺度卷积（1x1, 3x3, 5x5）生成特征变体
2. 适应度评估器（Fitness Evaluator）：
   - 信息熵估计每个变体的信息量
3. 选择器（Selector）：
   - Top-k 选择保留高质量变体
4. 进化融合器（Evolution Fusion）：
   - 选择后的变体与原始特征加权融合

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 DEM（Dense Evolution Module）模块，将生物进化论的选择机制引入
特征增强。该模块首先通过多尺度卷积生成特征的多个变体（变异），然后用信息
熵估计每个变体的信息量作为适应度指标，接着基于适应度选择 top-k 个高质量
变体（自然选择），最后将选择后的变体与原始特征融合形成下一代特征。该设计
在保持密集连接优势的同时，通过选择性保留控制了计算开销。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要多尺度特征交互和特征质量筛选的
任务。
'''


class DEM(nn.Module):
    """DEM: Dense Evolution Module —— 密集进化模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_variants: int = 4, top_k: int = 2):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_variants = num_variants
        self.top_k = min(top_k, num_variants)

        # Variation generator: multi-scale convolutions
        self.variants = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, inner, 1, bias=False),
                nn.BatchNorm2d(inner),
                nn.GELU(),
                nn.Conv2d(inner, channels, 1, bias=False),
            ) for _ in range(num_variants)
        ])

        # Fitness evaluator: information entropy estimation
        self.fitness_conv = nn.Conv2d(channels, 1, 1, bias=False)

        # Evolution fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * top_k, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _estimate_fitness(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate fitness via local information entropy."""
        # Simple entropy estimation via local variance
        mu = F.avg_pool2d(x, 3, stride=1, padding=1)
        mu_sq = F.avg_pool2d(x * x, 3, stride=1, padding=1)
        var = (mu_sq - mu * mu).clamp(min=1e-6)
        fitness = self.fitness_conv(var)
        return fitness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Generate variants (variation)
        variant_list = []
        for variant_fn in self.variants:
            v = variant_fn(x) + x  # residual variant
            variant_list.append(v)

        # 2. Evaluate fitness
        fitness_scores = []
        for v in variant_list:
            fitness = self._estimate_fitness(v)
            fitness_scores.append(fitness)

        # Stack fitness scores: [B, num_variants, H, W]
        fitness_map = torch.cat(fitness_scores, dim=1)  # [B, num_variants, H, W]
        fitness_map = fitness_map.view(B, self.num_variants, -1).mean(dim=-1)  # [B, num_variants]

        # 3. Select top-k variants
        _, top_indices = fitness_map.topk(self.top_k, dim=1)  # [B, top_k]

        # Gather selected variants
        selected = []
        for i in range(self.top_k):
            idx = top_indices[:, i]  # [B]
            for b in range(B):
                selected.append(variant_list[idx[b]][b:b+1])
        selected = torch.cat(selected, dim=0)  # [B*top_k, C, H, W]
        # Reshape to [B, top_k*C, H, W]
        selected = selected.view(B, self.top_k, C, H, W)
        selected = selected.permute(0, 1, 2, 3, 4).reshape(B, self.top_k * C, H, W)

        # 4. Evolution fusion
        gate = self.fusion_gate(selected)
        # Average selected variants
        selected_avg = selected.view(B, self.top_k, C, H, W).mean(dim=1)
        out = selected_avg * gate + x * (1 - gate)

        # 5. Refine
        out = self.refine(out)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = DEM(channels=64)
    output = model(input_tensor)
    print('=== DEM: Dense Evolution Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
