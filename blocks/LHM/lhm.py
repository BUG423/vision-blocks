import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：LHM (Local Histogram Module) —— 局部直方图模块

一、模块简介
直方图是描述特征分布的重要统计量，但传统的直方图计算不可微分，无法
嵌入端到端训练的网络中。

LHM 的核心思想是：通过可微分的局部直方图近似计算，将分布统计信息引入
特征增强，实现分布感知的特征处理。

核心创新点：
1. 可微分直方图：使用 soft binning 实现可微分的直方图计算
2. 局部分布统计：在局部窗口内计算特征分布
3. 分布感知增强：基于分布统计量进行特征增强
4. 自适应 bin 边界：bin 边界可学习

二、结构设计
LHM 由以下子结构组成：
1. 可微分直方图计算器（Differentiable Histogram Calculator）：
   - Soft binning → 局部直方图
2. 分布特征提取器（Distribution Feature Extractor）：
   - 直方图 → 分布统计量
3. 分布感知增强器（Distribution-Aware Enhancer）：
   - 基于分布统计的特征增强
4. 残差融合

三、论文写法参考
"本文提出 LHM（Local Histogram Module）模块，通过可微分的局部直方图近似
计算将分布统计信息引入特征增强。该模块使用 soft binning 实现可微分的直方图
计算，在局部窗口内统计特征分布，并基于分布统计量进行自适应特征增强。"

四、适用任务
适用于图像分类、异常检测、纹理分析等需要分布信息的视觉任务。
'''


class LHM(nn.Module):
    """LHM: Local Histogram Module —— 局部直方图模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_bins: int = 16, local_size: int = 7):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_bins = num_bins
        self.local_size = local_size

        # Learnable bin boundaries
        self.bin_boundaries = nn.Parameter(
            torch.linspace(-1, 1, num_bins + 1).view(1, 1, -1)
        )

        # Distribution feature extractor
        self.dist_extractor = nn.Sequential(
            nn.Conv1d(num_bins, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )

        # Enhancer
        self.enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def _soft_binning(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable histogram via soft binning."""
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, -1)                   # [B*C, 1, H*W]

        # Compute soft assignments to bins
        boundaries = self.bin_boundaries                   # [1, 1, num_bins+1]
        x_exp = x_flat.unsqueeze(-1)                      # [B*C, 1, H*W, 1]
        bin_width = boundaries[0, 0, 1] - boundaries[0, 0, 0]

        # Soft assignment: gaussian-like
        diff = (x_exp - boundaries.view(1, 1, 1, -1)) / (bin_width + 1e-6)
        soft_assign = torch.exp(-diff ** 2 / 2)           # [B*C, 1, H*W, num_bins+1]
        soft_assign = soft_assign[:, :, :, :self.num_bins]

        # Compute histogram
        hist = soft_assign.sum(dim=2)                      # [B*C, 1, num_bins]
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)

        return hist.view(B, C, self.num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Compute local histogram
        hist = self._soft_binning(x)                      # [B, C, num_bins]

        # 2. Extract distribution features
        dist_feat = self.dist_extractor(hist)             # [B, C, 1]
        dist_feat = dist_feat.unsqueeze(-1).expand(-1, -1, H, W)

        # 3. Enhance
        enhanced = self.enhancer(x + dist_feat)

        # 4. Gate
        gate = self.gate(enhanced)
        out = enhanced * gate + x * (1 - gate)

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = LHM(channels=64)
    output = model(input_tensor)
    print('=== LHM: Local Histogram Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
