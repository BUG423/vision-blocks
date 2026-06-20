import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：VGM (Variational Gaussian Mixing) —— 变分高斯混合模块

一、模块简介
深度网络的特征分布通常假设为确定性的，但实际上特征具有内在的不确定性。
利用这种不确定性可以实现更鲁棒的特征增强——高不确定性区域需要更谨慎的
处理，低不确定性区域可以更积极的增强。

VGM 的核心思想是：将特征建模为高斯分布，通过变分推断学习分布参数，然后
利用分布的不确定性信息实现自适应的特征混合和增强。

核心创新点：
1. 高斯分布建模：将每个特征通道建模为高斯分布，学习均值和方差
2. 变分推断：通过重参数化技巧实现可微的分布学习
3. 不确定性感知混合：利用分布方差作为不确定性指标指导混合
4. 采样增强：通过分布采样生成增强特征

二、结构设计
VGM 由以下子结构组成：
1. 分布参数估计器（Distribution Parameter Estimator）：
   - 估计高斯分布的均值和方差
2. 重参数化采样（Reparameterization Sampling）：
   - 通过重参数化技巧生成采样特征
3. 不确定性感知混合（Uncertainty-Aware Mixing）：
   - 利用方差指导特征混合权重
4. 输出精炼与残差连接

三、论文写法参考
"本文提出 VGM（Variational Gaussian Mixing）模块，将特征建模为高斯分布，
通过变分推断学习分布参数。该模块首先估计每个特征通道的均值和方差，然后
通过重参数化技巧生成采样特征，最后利用方差作为不确定性指标实现自适应的
特征混合，增强了特征的鲁棒性和表达能力。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要建模不确定性的场景。
'''


class VGM(nn.Module):
    """VGM: Variational Gaussian Mixing —— 变分高斯混合模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_samples: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.num_samples = num_samples

        # Distribution parameter estimator
        self.mean_estimator = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        self.log_var_estimator = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Uncertainty-aware mixer
        self.mixer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Refine
        self.refine = nn.Conv2d(channels, channels, 1, bias=False)

    def reparameterize(self, mean: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # Estimate distribution parameters
        mean = self.mean_estimator(x)
        log_var = self.log_var_estimator(x)

        # Reparameterization sampling
        sampled = self.reparameterize(mean, log_var)

        # Uncertainty-aware mixing
        var = torch.exp(log_var)
        uncertainty = var / (var.max() + 1e-6)

        mixed_input = torch.cat([sampled, uncertainty], dim=1)
        gate = self.mixer(mixed_input)

        out = gate * sampled + (1 - gate) * mean
        out = self.refine(out)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = VGM(channels=64)
    model.train()
    output = model(input_tensor)
    print('=== VGM: Variational Gaussian Mixing ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        model.eval()
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
