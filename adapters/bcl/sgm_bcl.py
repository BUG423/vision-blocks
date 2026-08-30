import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：SGM-BCL (Spatial Gradient Modulator - BCL) —— 空间梯度调制模块（BCL版）

一、模块简介
本模块是 SGM（Spatial Gradient Modulator）针对 BCL 时序数据格式的适配版本。
在 BCL 格式中，梯度信息体现在时间维度上——高梯度时间步表示数据变化剧烈（可
能是事件发生点），低梯度时间步表示数据平稳。本模块将 SGM 的梯度感知调制
思想扩展到时序数据，通过时间梯度实现时序特征的自适应增强。

核心创新点：
1. 时间梯度提取：使用 1D 差分算子计算时序特征的时间梯度
2. 梯度幅值调制：根据梯度幅值对特征进行自适应增强/抑制
3. 方向感知增强：正梯度（上升）和负梯度（下降）分别施加不同的调制策略

二、结构设计
SGM-BCL 由以下子结构组成：
1. 时间梯度提取器（Temporal Gradient Extractor）：
   - 1D 差分计算时间梯度
2. 梯度编码器（Gradient Encoder）：
   - 幅值编码 → 信息密度权重
   - 方向编码 → 方向选择性权重
3. 方向感知调制器（Direction-Aware Modulator）：
   - 基于方向权重的特征选择性增强
4. 梯度一致性残差
'''


class SGM_BCL(nn.Module):
    """SGM-BCL: Spatial Gradient Modulator for BCL format"""

    def __init__(self, channels: int, seq_len: int,
                 reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Gradient difference kernels
        diff_kernel = torch.tensor([-1, 0, 1], dtype=torch.float32) / 2
        self.register_buffer('diff_kernel', diff_kernel.view(1, 1, 3))

        # Magnitude encoder
        self.magnitude_encoder = nn.Sequential(
            nn.Conv1d(1, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Direction encoder
        self.direction_encoder = nn.Sequential(
            nn.Conv1d(1, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Feature modulation
        self.modulate = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def _compute_gradient(self, x: torch.Tensor) -> t.Tuple[torch.Tensor,
                                                              torch.Tensor]:
        """Compute temporal gradient magnitude and direction."""
        x_gray = x.mean(dim=1, keepdim=True)
        grad = F.conv1d(x_gray, self.diff_kernel, padding=1)
        magnitude = grad.abs()
        direction = grad
        return magnitude, direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, T] (BCL格式时序数据)
        输出:
            out: Tensor, shape = [B, C, T]
        """
        # 1. Compute gradients
        magnitude, direction = self._compute_gradient(x)

        # 2. Encode gradient information
        mag_weight = self.magnitude_encoder(magnitude)
        dir_weight = self.direction_encoder(direction)

        # 3. Direction-aware modulation
        combined_weight = mag_weight * dir_weight
        modulated = self.modulate(x * combined_weight)

        # 4. Residual
        return modulated + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 128)
    model = SGM_BCL(channels=64, seq_len=128)
    output = model(input_tensor)
    print('=== SGM-BCL ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
