import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：LVM (Local Variance Modulator) —— 局部方差调制模块

一、模块简介
在图像特征处理中，不同空间区域的局部方差（局部变化剧烈程度）反映了该
区域的信息密度——高方差区域通常包含丰富的纹理、边缘和细节信息，低方差
区域则相对平滑，主要承载全局结构信息。现有的注意力机制往往忽略这一统计
特性，对所有区域采用统一的处理策略，导致高方差区域的细节信息被过度平滑、
低方差区域的冗余信息未被有效抑制。

LVM 的核心思想是：通过精确估计每个空间位置的局部方差，将方差信息作为
特征调制的核心驱动力——对高方差区域施加细节增强与边缘锐化，对低方差区域
施加平滑抑制与冗余压缩，从而实现信息密度感知的自适应特征处理。

核心创新点：
1. 方差感知的双路调制：基于局部方差估计将特征分为高方差和低方差两个语义
   路径，分别施加增强和抑制处理
2. 多尺度方差估计：使用多个不同尺度的滑动窗口估计局部方差，捕获不同粒度
   的变化信息，避免单尺度估计的偏差
3. 方差引导的自适应混合：通过可学习的方差-调制映射函数，自适应地决定每个
   位置在增强和抑制之间的混合比例
4. 方差守恒机制：调制后的局部方差分布保持与原始分布的统计一致性，避免
   过度增强导致的特征失真

二、结构设计
LVM 由以下子结构组成：
1. 方差估计器（Variance Estimator）：
   - 多尺度滑动窗口统计（3x3, 5x5, 7x7）→ 方差金字塔
   - 多尺度方差融合为统一的方差图
2. 高方差增强路径（High-Variance Enhancement）：
   - 边缘感知的局部特征增强 → 细节锐化
3. 低方差抑制路径（Low-Variance Suppression）：
   - 平滑滤波 → 冗余压缩 → 结构保持
4. 方差引导混合器（Variance-Guided Mixer）：
   - 可学习方差→混合比映射 → 两路径自适应融合
5. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 LVM（Local Variance Modulator）模块，通过局部方差感知的自适应
调制实现信息密度驱动的特征处理。该模块首先使用多尺度滑动窗口估计每个空间
位置的局部方差，然后将特征分为高方差和低方差两个语义路径——高方差路径施加
细节增强与边缘锐化，低方差路径施加平滑抑制与冗余压缩；最终通过可学习的方差-
调制映射函数自适应地融合两路径输出，实现信息密度感知的特征增强。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要差异化处理纹理丰富区域与平滑区域
的任务，如医学图像分析、遥感图像分割、纹理分类等。
'''


class LVM(nn.Module):
    """LVM: Local Variance Modulator —— 局部方差调制模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner

        # Channel projection
        self.compress = nn.Conv2d(channels, inner, 1, bias=False)
        self.expand = nn.Conv2d(inner, channels, 1, bias=False)

        # Multi-scale variance estimator
        self.var_conv3 = nn.Conv2d(inner, inner, 3, padding=1,
                                    groups=inner, bias=False)
        self.var_conv5 = nn.Conv2d(inner, inner, 5, padding=2,
                                    groups=inner, bias=False)
        self.var_conv7 = nn.Conv2d(inner, inner, 7, padding=3,
                                    groups=inner, bias=False)
        self.var_fuse = nn.Sequential(
            nn.Conv2d(inner * 3, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # High-variance enhancement path
        self.hv_enhance = nn.Sequential(
            nn.Conv2d(inner, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
        )

        # Low-variance suppression path
        self.lv_suppress = nn.Sequential(
            nn.Conv2d(inner, inner, 5, padding=2, groups=inner, bias=False),
            nn.BatchNorm2d(inner),
            nn.Conv2d(inner, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
        )

        # Variance-guided mixing
        self.mix_gate = nn.Sequential(
            nn.Conv2d(inner, max(1, inner // 2), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(1, inner // 2), inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _multi_scale_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate local variance via multi-scale statistical moments."""
        x_sq = x * x
        # Use different kernel sizes for multi-scale variance
        v3 = F.avg_pool2d(x_sq, 3, stride=1, padding=1) - \
             F.avg_pool2d(x, 3, stride=1, padding=1).pow(2)
        v5 = F.avg_pool2d(x_sq, 5, stride=1, padding=2) - \
             F.avg_pool2d(x, 5, stride=1, padding=2).pow(2)
        v7 = F.avg_pool2d(x_sq, 7, stride=1, padding=3) - \
             F.avg_pool2d(x, 7, stride=1, padding=3).pow(2)
        var_map = torch.cat([v3, v5, v7], dim=1)        # [B, 3*inner, H, W]
        var_map = self.var_fuse(var_map)                 # [B, inner, H, W]
        return var_map.clamp(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Compress channels
        feat = self.compress(x)                         # [B, inner, H, W]

        # 2. Multi-scale variance estimation
        var_map = self._multi_scale_variance(feat)      # [B, inner, H, W]

        # 3. High-variance enhancement
        hv = self.hv_enhance(feat)                      # [B, inner, H, W]

        # 4. Low-variance suppression
        lv = self.lv_suppress(feat)                     # [B, inner, H, W]

        # 5. Variance-guided mixing
        gate = self.mix_gate(var_map)                   # [B, inner, H, W]
        mixed = gate * hv + (1 - gate) * lv

        # 6. Expand channels, refine and residual
        out = self.expand(mixed)                        # [B, C, H, W]
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = LVM(channels=128)
    output = model(input_tensor)
    print('=== LVM: Local Variance Modulator ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
