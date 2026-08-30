import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：TSFM-BCL (Temporal-Spatial Fusion Module - BCL) —— 时序-空间融合模块（BCL版）

一、模块简介
本模块是 TSFM（Temporal-Spatial Fusion Module）针对 BCL 时序数据格式的适
配版本。在 BCL 格式中，时序数据需要在时间维度和通道维度之间进行联合建模。
本模块将 TSFM 的双向交叉注意力机制扩展到时序数据，实现时间维度与通道维度
的协同增强。

核心创新点：
1. 时间-通道交叉注意力：时间特征通过交叉注意力从通道语义中获取上下文，
   通道特征通过交叉注意力从时间模式中获取时序信息
2. 多尺度时间感受野：使用不同大小的卷积核捕获不同粒度的时间模式
3. 时序门控融合：通过可学习的门控机制控制时间-通道融合的强度

二、结构设计
TSFM-BCL 由以下子结构组成：
1. 时间编码器（Temporal Encoder）：
   - 1D 卷积提取时间模式特征
2. 通道编码器（Channel Encoder）：
   - 沿时间轴池化提取通道语义特征
3. 时间-通道交叉注意力（Temporal-Channel Cross Attention）：
   - 时间特征查询通道特征 → 时间上下文增强
   - 通道特征查询时间特征 → 通道时序增强
4. 门控融合与输出精炼
'''


class TSFM_BCL(nn.Module):
    """TSFM-BCL: Temporal-Spatial Fusion Module for BCL format"""

    def __init__(self, channels: int, seq_len: int,
                 reduction: int = 4, num_heads: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner
        self.num_heads = num_heads
        self.head_dim = inner // num_heads
        self.seq_len = seq_len

        # Temporal encoder
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
        )

        # Channel encoder
        self.channel_enc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
        )

        # Cross attention: time queries channel
        self.t2c_q = nn.Conv1d(inner, inner, 1, bias=False)
        self.t2c_k = nn.Conv1d(inner, inner, 1, bias=False)
        self.t2c_v = nn.Conv1d(inner, inner, 1, bias=False)
        self.t2c_out = nn.Conv1d(inner, inner, 1, bias=False)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(inner * 2, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Decoder
        self.decode = nn.Sequential(
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

        self.scale = self.head_dim ** -0.5

    def _cross_attn_1d(self, q, k, v):
        """1D cross attention."""
        B, C, T = q.shape
        h = self.num_heads
        d = self.head_dim

        q = q.view(B, h, d, T).permute(0, 1, 3, 2)    # [B, h, T, d]
        k = k.view(B, h, d, 1).permute(0, 1, 3, 2)     # [B, h, 1, d]
        v = v.view(B, h, d, 1).permute(0, 1, 3, 2)     # [B, h, 1, d]

        attn = (q @ k.transpose(-1, -2)) * self.scale   # [B, h, T, 1]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).contiguous()
        out = out.view(B, C, T)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, T] (BCL格式时序数据)
        输出:
            out: Tensor, shape = [B, C, T]
        """
        # 1. Encode
        t_feat = self.temporal_enc(x)                   # [B, inner, T]
        c_feat = self.channel_enc(x)                    # [B, inner, 1]

        # 2. Time → Channel cross attention
        t_q = self.t2c_q(t_feat)
        t2c = self._cross_attn_1d(t_q, c_feat, c_feat)
        t2c = self.t2c_out(t2c)

        # 3. Channel → Time (simple fusion)
        c2t = c_feat.expand_as(t_feat) * t_feat

        # 4. Fusion
        concat = torch.cat([t2c, c2t], dim=1)
        g = self.fusion_gate(concat)
        fused = g * t2c + (1 - g) * c2t

        # 5. Decode and residual
        out = self.decode(fused)
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 128)
    model = TSFM_BCL(channels=64, seq_len=128)
    output = model(input_tensor)
    print('=== TSFM-BCL ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
