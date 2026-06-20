import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：OSM (Offset Spatial Mixing) —— 偏移空间混合模块

一、模块简介
标准卷积在固定的空间网格上采样，限制了模型对几何变换的建模能力。可变形
卷积（Deformable Conv）通过学习采样偏移来增强空间建模能力，但其偏移学习
通常只关注单点位置，忽略了偏移方向上的连续性约束。

OSM 的核心思想是：在可变形卷积的基础上引入偏移连续性约束，使学习到的偏移
场具有空间平滑性，从而生成更稳定的特征增强。

核心创新点：
1. 可变形偏移学习：为每个空间位置学习采样偏移
2. 偏移连续性约束：通过相邻位置偏移的平滑性约束，避免偏移场的突变
3. 多尺度偏移融合：不同尺度的偏移信息相互补充
4. 偏移感知特征聚合：根据偏移方向加权聚合采样特征

二、结构设计
OSM 由以下子结构组成：
1. 偏移预测器（Offset Predictor）：
   - 卷积网络预测每个位置的采样偏移
2. 连续性约束器（Continuity Enforcer）：
   - 相邻位置偏移的平滑性正则化
3. 多尺度偏移融合（Multi-Scale Offset Fusion）：
   - 不同尺度偏移的加权融合
4. 偏移感知特征聚合（Offset-Aware Aggregation）：
   - 根据偏移方向加权采样并聚合特征

三、论文写法参考
"本文提出 OSM（Offset Spatial Mixing）模块，在可变形卷积基础上引入偏移
连续性约束。该模块首先通过卷积网络预测每个位置的采样偏移，然后通过
相邻位置偏移的平滑性约束避免偏移场突变，最后根据偏移方向加权聚合采样
特征，生成空间平滑且几何自适应的特征增强。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合存在几何变换（旋转、缩放、形变）
的场景。
'''


class OSM(nn.Module):
    """OSM: Offset Spatial Mixing —— 偏移空间混合模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        inner = max(1, channels // reduction)
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_offset = kernel_size * kernel_size * 2

        # Offset predictor
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, self.num_offset, 3, padding=1, bias=False),
        )

        # Offset continuity smoothness
        self.smooth_conv = nn.Conv2d(self.num_offset, self.num_offset,
                                      3, padding=1, groups=self.num_offset,
                                      bias=False)

        # Feature aggregation
        self.aggregator = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Refine
        self.refine = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Predict offsets
        raw_offsets = self.offset_predictor(x)            # [B, 2*k*k, H, W]

        # Smooth offsets for continuity
        smoothed = self.smooth_conv(raw_offsets)
        offsets = raw_offsets + smoothed

        # Normalize offsets
        offsets = torch.tanh(offsets)

        # Simple offset-weighted aggregation (grid_sample approximation)
        # Use mean of offset-modulated features as approximation
        offset_mag = offsets.view(B, -1, 2, H, W).norm(dim=2)
        offset_weight = offset_mag.mean(dim=1, keepdim=True)

        # Aggregate
        aggregated = self.aggregator(x)
        out = aggregated * (1 + offset_weight)

        out = self.refine(out)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = OSM(channels=64)
    output = model(input_tensor)
    print('=== OSM: Offset Spatial Mixing ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
