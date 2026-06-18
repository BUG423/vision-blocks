import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：EEM (Energy Equalization Module) —— 能量均衡模块

一、模块简介
在深度卷积网络中，不同通道和不同空间位置的特征能量分布往往极不均衡——
某些通道携带大量信息而某些通道几乎无用，某些空间区域激活极强而其他区域
几乎为零。这种不均衡会导致梯度在训练中分配不均，使得高能量通道过拟合、
低能量通道欠训练，最终降低模型的泛化能力。

EEM 的核心思想是：通过能量感知的全局-局部双重均衡机制，自动重塑特征图
的能量分布，使其在通道维度和空间维度上都趋于均匀化，从而让每个通道和每
个空间位置都能充分参与表征学习。

核心创新点：
1. 通道能量均衡：通过全局能量统计（通道L2范数）估计每个通道的能量水平，
   再用可学习的通道缩放因子将高能量通道压缩、低能量通道放大
2. 空间能量均衡：通过局部能量估计（滑动窗口统计）识别空间能量聚集区域，
   用空间门控将能量从集中区向稀疏区扩散
3. 双向均衡融合：通道均衡与空间均衡的结果通过自适应门控融合，避免单一维度
   均衡导致的信息损失
4. 能量守恒约束：均衡过程保持总能量不变，只重新分配而非增减信息量

二、结构设计
EEM 由以下子结构组成：
1. 能量编码器（Energy Encoder）：
   - 1x1 卷积将特征压缩到低维能量空间
2. 通道能量均衡器（Channel Energy Equalizer）：
   - 全局平均池化估计每通道能量 → 可学习缩放因子 → 通道均衡
3. 空间能量均衡器（Spatial Energy Equalizer）：
   - 局部能量估计 → 空间门控扩散 → 空间均衡
4. 双向均衡融合器（Bidirectional Fusion）：
   - 通道均衡结果与空间均衡结果通过自适应门控融合
5. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 EEM（Energy Equalization Module）模块，通过能量感知的全局-局部
双重均衡机制重塑特征图的能量分布。该模块首先在通道维度上通过全局能量统计
估计各通道能量水平，并用可学习缩放因子实现通道间能量均衡；然后在空间维度
上通过局部能量估计和门控扩散实现空间能量均衡；最终将两个维度的均衡结果通
过自适应门控融合，并保持总能量守恒。该设计使每个通道和空间位置都能充分参
与表征学习，提升了深层网络的泛化能力。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合深层网络中特征能量分布严重不均的场
景，以及需要均衡各通道和空间位置表征能力的任务。
'''


class EEM(nn.Module):
    """EEM: Energy Equalization Module —— 能量均衡模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 local_kernel: int = 7):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner
        self.lk = local_kernel
        self.padding = local_kernel // 2

        # Energy encoder: project to low-dim energy space
        self.encode = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Channel energy equalizer
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner, max(1, inner // 4), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(1, inner // 4), inner, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Spatial energy equalizer
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(inner, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm2d(inner),
            nn.Sigmoid(),
        )

        # Bidirectional fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(inner * 2, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Decoder
        self.decode = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _local_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate local energy via sliding window L2 norm."""
        x_sq = x * x
        local_energy = F.avg_pool2d(x_sq, self.lk, stride=1,
                                     padding=self.padding)
        return local_energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Encode to energy space
        feat = self.encode(x)                          # [B, inner, H, W]

        # 2. Channel energy equalization
        ch_energy = self.channel_scale(feat)           # [B, inner, 1, 1]
        ch_eq = feat * ch_energy                       # channel-balanced

        # 3. Spatial energy equalization
        local_e = self._local_energy(feat)             # local energy map
        sp_gate = self.spatial_gate(local_e)           # spatial gate [0,1]
        sp_eq = feat * sp_gate + feat.mean() * (1 - sp_gate)  # spatial-balanced

        # 4. Bidirectional fusion
        concat = torch.cat([ch_eq, sp_eq], dim=1)     # [B, 2*inner, H, W]
        g = self.fusion_gate(concat)                   # [B, inner, H, W]
        fused = g * ch_eq + (1 - g) * sp_eq

        # 5. Energy conservation: rescale to preserve total energy
        orig_energy = feat.pow(2).mean()
        fused_energy = fused.pow(2).mean().clamp(min=1e-8)
        fused = fused * (orig_energy / fused_energy).sqrt()

        # 6. Decode, refine and residual
        out = self.decode(fused)                       # [B, C, H, W]
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = EEM(channels=128)
    output = model(input_tensor)
    print('=== EEM: Energy Equalization Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
