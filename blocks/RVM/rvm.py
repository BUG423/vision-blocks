import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：RVM (Random Variation Module) —— 随机变异模块

一、模块简介
数据增强是提升模型泛化能力的重要手段，但通常只在训练阶段的输入端进行，
网络内部缺乏随机性。

RVM 的核心思想是：在网络内部引入可控的随机变异，让网络在推理时也能获得
一定的随机性，提升鲁棒性和集成效果。

核心创新点：
1. 可控随机注入：在特征图中注入可控的随机噪声
2. 温度控制：通过温度参数控制随机强度
3. 特征级增强：在特征空间而非输入空间进行随机增强
4. 训练-推理一致性：训练时启用随机性，推理时可控

二、结构设计
RVM 由以下子结构组成：
1. 随机噪声生成器（Random Noise Generator）：
   - 高斯噪声 + 噪声缩放
2. 温度控制器（Temperature Controller）：
   - 可学习温度参数
3. 特征调制器（Feature Modulator）：
   - 噪声与特征融合
4. 方差保持归一化

三、论文写法参考
"本文提出 RVM（Random Variation Module）模块，在网络内部引入可控的随机
变异。该模块通过高斯噪声生成器注入随机性，并用可学习的温度参数控制随机
强度，最后通过方差保持归一化确保特征稳定性。该设计让网络在推理时也能获得
一定的随机性，提升鲁棒性。"

四、适用任务
适用于图像分类、目标检测等需要提升鲁棒性和集成效果的视觉任务。
'''


class RVM(nn.Module):
    """RVM: Random Variation Module —— 随机变异模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 temperature: float = 1.0):
        super().__init__()
        inner = max(1, channels // reduction)

        # Temperature controller
        self.temperature = nn.Parameter(
            torch.tensor(temperature).log()
        )

        # Noise scaling
        self.noise_scale = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Softplus(),
        )

        # Feature modulator
        self.modulator = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Variance-preserving normalization
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Temperature-controlled noise
        temp = torch.exp(self.temperature)
        noise = torch.randn_like(x) * temp

        # 2. Noise scaling
        scale = self.noise_scale(x)
        noise = noise * scale

        # 3. Feature modulation
        modulated = self.modulator(x + noise)

        # 4. Variance-preserving normalization
        out = self.norm(modulated)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = RVM(channels=64)
    output = model(input_tensor)
    print('=== RVM: Random Variation Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
