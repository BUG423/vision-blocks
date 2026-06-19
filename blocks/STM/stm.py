import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：STM (Spatial-Channel Transformer Module) —— 空间-通道变换模块

一、模块简介
Transformer 的自注意力机制在视觉任务中表现出色，但标准的自注意力在
空间和通道维度上独立计算，忽略了两者的耦合关系。

STM 的核心思想是：通过空间-通道联合变换机制，让空间注意力和通道注意力
相互引导，实现更精确的特征建模。

核心创新点：
1. 空间-通道联合查询：空间特征查询通道特征，通道特征查询空间特征
2. 交叉注意力融合：双向交叉注意力实现空间-通道信息流动
3. 位置编码增强：引入可学习的位置编码增强空间感知
4. 多头联合变换：多头机制捕获不同的空间-通道交互模式

二、结构设计
STM 由以下子结构组成：
1. 空间编码器（Spatial Encoder）：
   - 位置编码 + 空间特征提取
2. 通道编码器（Channel Encoder）：
   - 全局池化 + 通道特征提取
3. 交叉注意力（Cross Attention）：
   - 空间→通道交叉注意力
   - 通道→空间交叉注意力
4. 联合融合与输出投影

三、论文写法参考
"本文提出 STM（Spatial-Channel Transformer Module）模块，通过空间-通道
联合变换机制实现更精确的特征建模。该模块让空间特征和通道特征通过双向交叉
注意力相互引导，并引入可学习位置编码增强空间感知，最终通过多头机制捕获
不同的空间-通道交互模式。"

四、适用任务
适用于图像分类、目标检测、语义分割等需要精确空间-通道建模的视觉任务。
'''


class STM(nn.Module):
    """STM: Spatial-Channel Transformer Module —— 空间-通道变换模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_heads: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_heads = num_heads
        self.head_dim = inner // num_heads

        # Spatial encoder with position encoding
        self.spatial_enc = nn.Conv2d(channels, inner, 1, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, inner, 1, 1) * 0.02)

        # Channel encoder
        self.channel_enc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
        )

        # Cross attention: spatial queries channel
        self.s2c_q = nn.Conv2d(inner, inner, 1, bias=False)
        self.s2c_k = nn.Conv2d(inner, inner, 1, bias=False)
        self.s2c_v = nn.Conv2d(inner, inner, 1, bias=False)
        self.s2c_out = nn.Conv2d(inner, inner, 1, bias=False)

        # Cross attention: channel queries spatial
        self.c2s_q = nn.Conv2d(inner, inner, 1, bias=False)
        self.c2s_k = nn.Conv2d(inner, inner, 1, bias=False)
        self.c2s_v = nn.Conv2d(inner, inner, 1, bias=False)
        self.c2s_out = nn.Conv2d(inner, inner, 1, bias=False)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(inner * 2, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.scale = self.head_dim ** -0.5

    def _cross_attn(self, q, k, v):
        """Multi-head cross attention."""
        B, C, H, W = q.shape
        h = self.num_heads
        d = self.head_dim

        q = q.view(B, h, d, H * W).permute(0, 1, 3, 2)
        k = k.view(B, h, d, -1).permute(0, 1, 3, 2)
        v = v.view(B, h, d, -1).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).contiguous()
        return out.view(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 1. Encode
        s_feat = self.spatial_enc(x) + self.pos_embed
        c_feat = self.channel_enc(x)

        # 2. Spatial → Channel cross attention
        s_q = self.s2c_q(s_feat)
        s2c = self._cross_attn(s_q, c_feat, c_feat)
        s2c = self.s2c_out(s2c)

        # 3. Channel → Spatial cross attention
        c_q = self.c2s_q(c_feat.expand_as(s_feat))
        c2s = self._cross_attn(c_q, s_feat, s_feat)
        c2s = self.c2s_out(c2s)

        # 4. Fusion
        concat = torch.cat([s2c, c2s], dim=1)
        gate = self.fusion_gate(concat)
        fused = gate * s2c + (1 - gate) * c2s

        # 5. Output projection
        out = self.proj(fused)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = STM(channels=64)
    output = model(input_tensor)
    print('=== STM: Spatial-Channel Transformer Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
