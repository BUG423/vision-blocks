import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：LVM-BCL (Local Variance Modulator - BCL) —— 局部方差调制模块（BCL版）

一、模块简介
本模块是 LVM（Local Variance Modulator）针对 BCL 时序数据格式的适配版本。
在 BCL 格式中，局部方差需要在时间维度上进行估计——高方差时间步表示数据变
化剧烈（可能是事件发生点），低方差时间步表示数据平稳。本模块将 LVM 的方差
感知双路径调制扩展到时序数据，实现信息密度感知的时序特征增强。

核心创新点：
1. 时间维度方差估计：使用多尺度滑动窗口在时间轴上估计局部方差
2. 高方差时间步增强：对变化剧烈的时间步施加特征增强
3. 低方差时间步抑制：对平稳时间步施加冗余压缩
4. 方差引导的时间混合：通过可学习映射自适应融合增强和抑制路径

二、结构设计
LVM-BCL 由以下子结构组成：
1. 时间方差估计器（Temporal Variance Estimator）：
   - 多尺度滑动窗口统计（3, 5, 7）→ 方差图
2. 高方差增强路径（High-Variance Enhancement）：
   - 1D 卷积特征增强
3. 低方差抑制路径（Low-Variance Suppression）：
   - 深度 1D 卷积平滑
4. 方差引导混合器与输出精炼
'''


class LVM_BCL(nn.Module):
    """LVM-BCL: Local Variance Modulator for BCL format"""

    def __init__(self, channels: int, seq_len: int,
                 reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner

        # Projection
        self.compress = nn.Conv1d(channels, inner, 1, bias=False)
        self.expand = nn.Conv1d(inner, channels, 1, bias=False)

        # Multi-scale temporal variance estimator
        self.var_conv3 = nn.Conv1d(inner, inner, 3, padding=1,
                                    groups=inner, bias=False)
        self.var_conv5 = nn.Conv1d(inner, inner, 5, padding=2,
                                    groups=inner, bias=False)
        self.var_conv7 = nn.Conv1d(inner, inner, 7, padding=3,
                                    groups=inner, bias=False)
        self.var_fuse = nn.Sequential(
            nn.Conv1d(inner * 3, inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # High-variance enhancement
        self.hv_enhance = nn.Sequential(
            nn.Conv1d(inner, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
        )

        # Low-variance suppression
        self.lv_suppress = nn.Sequential(
            nn.Conv1d(inner, inner, 5, padding=2, groups=inner, bias=False),
            nn.BatchNorm1d(inner),
            nn.Conv1d(inner, inner, 1, bias=False),
            nn.BatchNorm1d(inner),
        )

        # Variance-guided mixing
        self.mix_gate = nn.Sequential(
            nn.Conv1d(inner, max(1, inner // 2), 1, bias=False),
            nn.GELU(),
            nn.Conv1d(max(1, inner // 2), inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def _multi_scale_temporal_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate temporal variance via multi-scale statistics."""
        x_sq = x * x
        v3 = F.avg_pool1d(x_sq, 3, stride=1, padding=1) - \
             F.avg_pool1d(x, 3, stride=1, padding=1).pow(2)
        v5 = F.avg_pool1d(x_sq, 5, stride=1, padding=2) - \
             F.avg_pool1d(x, 5, stride=1, padding=2).pow(2)
        v7 = F.avg_pool1d(x_sq, 7, stride=1, padding=3) - \
             F.avg_pool1d(x, 7, stride=1, padding=3).pow(2)
        var_map = torch.cat([v3, v5, v7], dim=1)
        var_map = self.var_fuse(var_map)
        return var_map.clamp(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, T] (BCL格式时序数据)
        输出:
            out: Tensor, shape = [B, C, T]
        """
        # 1. Compress
        feat = self.compress(x)                         # [B, inner, T]

        # 2. Temporal variance estimation
        var_map = self._multi_scale_temporal_variance(feat)

        # 3. High-variance enhancement
        hv = self.hv_enhance(feat)

        # 4. Low-variance suppression
        lv = self.lv_suppress(feat)

        # 5. Variance-guided mixing
        gate = self.mix_gate(var_map)
        mixed = gate * hv + (1 - gate) * lv

        # 6. Expand and residual
        out = self.expand(mixed)
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 128)
    model = LVM_BCL(channels=64, seq_len=128)
    output = model(input_tensor)
    print('=== LVM-BCL ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
