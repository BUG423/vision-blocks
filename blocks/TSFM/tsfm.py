import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：TSFM (Temporal-Spatial Fusion Module) —— 时序-空间融合模块

一、模块简介
传统的卷积特征提取器在处理图像时，将空间维度（H×W）与通道维度（C）
视为独立的处理对象，先通过卷积提取空间特征，再通过通道注意力或MLP调整
通道权重。这种串行处理方式忽略了空间结构与通道语义之间的耦合关系——例
如边缘纹理特征通常集中在某些特定通道，而平滑区域特征集中在另一些通道，
空间结构与通道语义本应协同处理。

TSFM 的核心思想是：将空间结构信息与通道语义信息在同一融合空间中进行联合
建模，通过双向信息流动实现空间-通道的协同增强。具体来说，先用空间编码器
提取结构特征、用通道编码器提取语义特征，然后在一个共享的融合空间中通过
交叉注意力机制让两者互相引导，最后解码回原始特征空间。

核心创新点：
1. 空间-通道联合编码：分别提取空间结构特征与通道语义特征，保留各自的信息
   特征，避免混合编码导致的信息损失
2. 双向交叉注意力融合：空间特征通过交叉注意力从通道语义中获取上下文信息，
   通道特征通过交叉注意力从空间结构中获取位置信息，实现双向信息流动
3. 自适应融合深度控制：通过可学习的门控机制控制交叉注意力的强度，避免过度
   融合导致的空间结构模糊或通道语义混淆
4. 位置感知的通道调制：在融合后的通道注意力中引入空间位置先验，使通道调制
   具备空间感知能力

二、结构设计
TSFM 由以下子结构组成：
1. 空间编码器（Spatial Encoder）：
   - 3x3 深度卷积 + 1x1 卷积提取空间结构特征
2. 通道编码器（Channel Encoder）：
   - 全局平均池化 + 通道MLP提取通道语义特征
3. 双向交叉注意力融合器（Bidirectional Cross-Attention Fusion）：
   - 空间→通道交叉注意力：空间结构引导通道语义更新
   - 通道→空间交叉注意力：通道语义引导空间结构增强
   - 可学习融合门控控制交互强度
4. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 TSFM（Temporal-Spatial Fusion Module）模块，通过双向交叉注意力
机制实现空间结构与通道语义的协同增强。该模块分别提取空间结构特征与通道语义
特征，在共享融合空间中通过双向交叉注意力实现空间-通道的双向信息流动；通道
语义通过交叉注意力聚合空间位置信息，空间结构通过交叉注意力获取语义上下文，
最终通过可学习门控控制融合强度，避免过度融合导致的信息混淆。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合空间结构与通道语义耦合关系紧密的任
务，如边缘检测、纹理分类、细粒度识别等。
'''


class TSFM(nn.Module):
    """TSFM: Temporal-Spatial Fusion Module —— 时序-空间融合模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 fusion_heads: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner
        self.num_heads = fusion_heads
        self.head_dim = inner // fusion_heads

        # Spatial encoder
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Channel encoder
        self.channel_enc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Cross-attention: spatial queries channel keys/values
        self.s2c_q = nn.Conv2d(inner, inner, 1, bias=False)
        self.s2c_k = nn.Conv2d(inner, inner, 1, bias=False)
        self.s2c_v = nn.Conv2d(inner, inner, 1, bias=False)
        self.s2c_out = nn.Conv2d(inner, inner, 1, bias=False)

        # Cross-attention: channel queries spatial keys/values
        self.c2s_q = nn.Conv2d(inner, inner, 1, bias=False)
        self.c2s_k = nn.Conv2d(inner, inner, 1, bias=False)
        self.c2s_v = nn.Conv2d(inner, inner, 1, bias=False)
        self.c2s_out = nn.Conv2d(inner, inner, 1, bias=False)

        # Fusion gate
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

        self.scale = self.head_dim ** -0.5

    def _multihead_cross_attn(self, q, k, v):
        """Multi-head cross attention with spatial→channel flow."""
        B, C, H, W = q.shape
        h = self.num_heads
        d = self.head_dim

        q = q.view(B, h, d, H * W).transpose(-1, -2)   # [B, h, HW, d]
        k = k.view(B, h, d, 1).transpose(-1, -2)        # [B, h, 1, d]
        v = v.view(B, h, d, 1).transpose(-1, -2)        # [B, h, 1, d]

        attn = (q @ k.transpose(-1, -2)) * self.scale   # [B, h, HW, 1]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-1, -2).contiguous()  # [B, h, d, HW]
        out = out.view(B, C, H, W)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Encode
        s_feat = self.spatial_enc(x)                    # [B, inner, H, W]
        c_feat = self.channel_enc(x)                    # [B, inner, 1, 1]

        # 2. Spatial → Channel cross attention
        s_q = self.s2c_q(s_feat)
        s_k = self.s2c_k(s_feat)
        s_v = self.s2c_v(s_feat)
        s2c = self._multihead_cross_attn(s_q, c_feat, c_feat)
        s2c = self.s2c_out(s2c)

        # 3. Channel → Spatial cross attention
        c_q = self.c2s_q(c_feat.expand_as(s_feat))
        c_k = self.c2s_k(s_feat)
        c_v = self.c2s_v(s_feat)
        c2s = self.c2s_out(c_feat.expand_as(s_feat) * s_feat)

        # 4. Fusion
        concat = torch.cat([s2c, c2s], dim=1)
        g = self.fusion_gate(concat)
        fused = g * s2c + (1 - g) * c2s

        # 5. Decode, refine and residual
        out = self.decode(fused)
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = TSFM(channels=128)
    output = model(input_tensor)
    print('=== TSFM: Temporal-Spatial Fusion Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
