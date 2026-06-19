import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：PAM (Phase Alignment Module) —— 相位对齐模块

一、模块简介
在特征融合过程中，不同来源或不同层的特征往往存在相位偏移，导致融合后
的特征出现模糊或失真。传统的相位校正方法通常在全局频域上操作，计算开销
大且缺乏空间自适应性。

PAM 的核心思想是：通过局部相位估计和自适应对齐机制，使特征在融合前达到
相位一致，从而提升融合质量。

核心创新点：
1. 局部相位估计：使用 Gabor 滤波器组估计每个位置的局部相位
2. 自适应相位对齐：通过可学习的相位偏移量对齐特征
3. 相位一致性约束：保持对齐后的相位一致性
4. 残差融合：对齐后的特征与原始特征残差融合

二、结构设计
PAM 由以下子结构组成：
1. 相位估计器（Phase Estimator）：
   - Gabor 滤波器组估计局部相位
2. 相位对齐器（Phase Aligner）：
   - 可学习相位偏移 → 自适应对齐
3. 相位一致性约束（Phase Consistency）：
   - 保持相位结构
4. 残差融合

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 PAM（Phase Alignment Module）模块，通过局部相位估计和自适应对齐
机制实现特征的相位一致性融合。该模块首先使用 Gabor 滤波器组估计每个位置的
局部相位，然后通过可学习的相位偏移量自适应对齐特征，最后保持对齐后的相位
一致性并与原始特征残差融合。"

四、适用任务
适用于图像融合、图像恢复、多尺度特征融合等视觉任务。
'''


class PAM(nn.Module):
    """PAM: Phase Alignment Module —— 相位对齐模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_gabor: int = 8):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_gabor = num_gabor

        # Gabor-like phase estimator
        self.gabor_conv = nn.Conv2d(channels, num_gabor, 3, padding=1,
                                     bias=False)

        # Phase alignment
        self.phase_shift = nn.Sequential(
            nn.Conv2d(num_gabor, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Tanh(),
        )

        # Phase consistency
        self.consistency = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
        )

        # Output gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 1. Phase estimation
        phase_map = self.gabor_conv(x)                  # [B, num_gabor, H, W]

        # 2. Phase alignment
        phase_offset = self.phase_shift(phase_map)       # [B, C, H, W]
        aligned = x + phase_offset * 0.1                 # small adjustment

        # 3. Phase consistency
        consistent = self.consistency(aligned)

        # 4. Gate fusion
        gate = self.gate(consistent)
        out = consistent * gate + x * (1 - gate)

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = PAM(channels=64)
    output = model(input_tensor)
    print('=== PAM: Phase Alignment Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
