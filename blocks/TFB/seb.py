import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：An Efficient Hybrid Vision Transformer for TinyML Applications (TinyNeXt) (ICCV 2025)
# 论文链接：https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf
# 代码来源：https://github.com/yuffeenn/TinyNeXt
# 原始许可证：MIT
# 模块出处：classification/models/modules.py 的 SeBlock / SeModule / Mlp 类
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：去除 einops 与 Add/Mul 包装类（数值等价）；保留 SeModule 的
# max(C//reduction, 8) 隐层下限与 Hardsigmoid 门控；三残差结构不变。
# 前向数值逻辑与原 SeBlock 逐算子一致。

'''
模块名称：SEB (TinyNeXt SeBlock) —— TinyNeXt 通道激励卷积块

一、模块简介
SEB 是 TinyNeXt 中与 FormerBlock 并列的轻量 CNN 块，用于在纯卷积阶段
注入通道注意力。它将传统 SE（Squeeze-and-Excitation）门控与深度可分离
局部卷积、瓶颈 MLP 串联成三残差结构：SE 分支用全局平均池化压缩通道
统计后经两层 1x1 卷积产生通道权重（Hardsigmoid 激活，对量化友好），
局部 DW 3x3 分支补充空间上下文，MLP 分支完成通道混合。相比标准 SE 块，
SeBlock 将三者均以残差方式叠加，训练更稳定，且全部算子（BN、DW Conv、
1x1 Conv、Hardsigmoid）均为 TinyML 友好算子，可在微控制器上低比特部署。

二、结构设计
输入 [B, C, H, W]，输出 [B, C, H, W]：
1. SE 激励分支（残差）：
   - BatchNorm2d(C)
   - AdaptiveAvgPool2d(1) → [B, C, 1, 1]
   - 1x1 Conv(C→max(C//r, 8)) → BN → ReLU → 1x1 Conv(→C) → Hardsigmoid
   - 残差加回 x + BN(x) * gate(BN(x))（与原 SeModule 的 x*se(x) 一致）
2. 局部卷积分支（残差）：3x3 深度可分离卷积（groups=C），x = x + dw(x)
3. MLP 分支（残差）：
   - BatchNorm2d(C) → 1x1 Conv(ratio*C) → GELU → 1x1 Conv(C)，残差加回
   （ratio 默认 2.0，与原 Mlp 一致）

三、论文写法参考
若在论文中引用该模块，可描述为："We adopt the SeBlock from TinyNeXt
(ICCV 2025), which stacks squeeze-and-excitation gating, depthwise local
convolution and a bottleneck MLP as three residual branches for TinyML."
并引用：An Efficient Hybrid Vision Transformer for TinyML Applications,
ICCV 2025.

四、适用任务
TinyML 图像分类、微型检测/分割等边缘任务；也可作为即插即用的通道注意力
块嵌入 CNN 主干，适合需要低比特量化部署的场景。
'''


class SEB(nn.Module):
    """SEB: TinyNeXt SeBlock —— SE 门控 + 局部 DW 卷积 + MLP 三残差块"""

    def __init__(self, channels: int, mlp_ratio: float = 2.0, reduction: int = 4):
        super().__init__()
        self.channels = channels
        hidden_channel = max(channels // reduction, 8)

        self.norm_se = nn.BatchNorm2d(channels)
        # SeModule 原式：BN(x) * gate(BN(x))，故门控与被乘对象共用 norm_se 输出
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, channels, kernel_size=1, bias=False),
            nn.Hardsigmoid(),
        )

        self.local = nn.Conv2d(channels, channels, 3, 1, 1, 1, groups=channels,
                               bias=False)

        hidden = int(mlp_ratio * channels)
        self.norm_mlp = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"channel mismatch: expect {self.channels}, got {C}"
        se_feat = self.norm_se(x)
        x = x + se_feat * self.se_gate(se_feat)
        x = x + self.local(x)
        x = x + self.mlp(self.norm_mlp(x))
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SEB(channels=128)
    # SE 门控内 BN 作用于 [B, C//r, 1, 1]，训练模式下需 batch>1；形状自检用 eval
    model.eval()
    output = model(input_tensor)
    print('=== SEB: TinyNeXt SeBlock ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
