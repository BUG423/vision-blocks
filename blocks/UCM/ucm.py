import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale (ICCV 2025)
# 论文链接：https://arxiv.org/abs/2508.09000
# 代码来源：https://github.com/ai-paperwithcode/UniConvNet
# 原始许可证：MIT
# 模块出处：models/UniConvNet.py 的 ConvMod 类（含 LayerNorm 辅助类）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：仅提取 ConvMod 及其 channels-first LayerNorm 辅助类，去除
# timm / DCNv3 依赖与整网训练代码；将硬编码的三级 DW 卷积核 7/9/11
# 参数化为 kernel_size, kernel_size+2, kernel_size+4（默认 7 时与原文
# 完全一致，且保持"逐级扩张感受野"设计）；补 channels % 4 == 0 断言。
# 前向数值逻辑与原 ConvMod 逐算子一致。

'''
模块名称：UCM (UniConvNet ConvMod) —— UniConvNet 卷积调制模块

一、模块简介
ConvNet 的有效感受野（Effective Receptive Field, EFR）通常远小于理论
感受野，且其分布趋于各向同性高斯。UniConvNet 指出：盲目扩大卷积核虽能
扩张 EFR，却会破坏 EFR 的渐近高斯性，导致边界响应失真。UCM（原论文的
ConvMod）在扩张 EFR 的同时维持其渐近高斯分布：通过三级级联的"调制-局部"
双分支结构，逐级使用递增的深度可分离卷积核（默认 7→9→11）平滑地扩大
感受野，并用乘性注意力式调制（a * v）对特征进行门控，使能量集中在中心、
边界平滑衰减，从而保持 EFR 的高斯形态。整个模块纯卷积实现，无动态卷积
或自注意力的显存开销，可即插即用于任意尺度的 ConvNet。

二、结构设计
输入 [B, C, H, W]，要求 C % 4 == 0，输出 [B, C, H, W]：
1. 一级调制（C/4 ×4 → C/2）：
   - channels-first LayerNorm
   - 四等分 x0,x1,x2,x3（各 C/4）
   - 调制支路 a1 = 1x1 → GELU → 7x7 DW（默认），mul = v11(a1 ⊙ v1(x0))
   - 局部支路 x1' = conv3_1(v12(x1)) + a1
   - 拼接 (x1', mul) → [B, C/2, H, W]
2. 二级调制（C/2 → 3C/4）：
   - LayerNorm(C/2)；a2 = 1x1 → GELU → 9x9 DW
   - mul = v21(a2 ⊙ v2(·))；局部支路 conv3_2(v22(x2)) + proj2(a2)
   - 拼接 → [B, 3C/4, H, W]
3. 三级调制（3C/4 → C）：
   - LayerNorm(3C/4)；a3 = 1x1 → GELU → 11x11 DW
   - mul = v31(a3 ⊙ v3(·))；局部支路 conv3_3(v32(x3)) + proj3(a3)
   - 拼接 → [B, C, H, W]
每级均以递增的 DW 核扩张 EFR，乘性调制维持响应的渐近高斯分布。

三、论文写法参考
若在论文中引用该模块，可描述为："We adopt the ConvMod block from
UniConvNet (ICCV 2025), which expands the effective receptive field via
cascaded growing depthwise kernels while maintaining an asymptotically
Gaussian EFR distribution through multiplicative modulation."并引用：
UniConvNet, ICCV 2025, arXiv:2508.09000.

四、适用任务
图像分类、目标检测、语义分割等视觉主干任务；特别适合需要大有效感受野
又要求边界响应平滑的场景（如高分辨率分割、小目标检测），可替换 ConvNeXt/
RepLKNet 等主干中的基本块。
'''


class _LayerNorm(nn.Module):
    """channels-first LayerNorm（原 UniConvNet 的 LayerNorm 辅助类）"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class UCM(nn.Module):
    """UCM: UniConvNet ConvMod —— 扩张有效感受野并保持渐近高斯分布的卷积调制模块"""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4"
        c4 = channels // 4
        c2 = channels // 2
        c34 = channels * 3 // 4
        # 逐级扩张的 DW 核：kernel_size, +2, +4（默认 7/9/11，与原文一致）
        k1, k2, k3 = kernel_size, kernel_size + 2, kernel_size + 4

        self.dim = channels

        self.norm1 = _LayerNorm(channels, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv2d(c4, c4, 1),
            nn.GELU(),
            nn.Conv2d(c4, c4, k1, padding=k1 // 2, groups=c4),
        )
        self.v1 = nn.Conv2d(c4, c4, 1)
        self.v11 = nn.Conv2d(c4, c4, 1)
        self.v12 = nn.Conv2d(c4, c4, 1)
        self.conv3_1 = nn.Conv2d(c4, c4, 3, padding=1, groups=c4)

        self.norm2 = _LayerNorm(c2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv2d(c2, c2, 1),
            nn.GELU(),
            nn.Conv2d(c2, c2, k2, padding=k2 // 2, groups=c2),
        )
        self.v2 = nn.Conv2d(c2, c2, 1)
        self.v21 = nn.Conv2d(c2, c2, 1)
        self.v22 = nn.Conv2d(c4, c4, 1)
        self.proj2 = nn.Conv2d(c2, c4, 1)
        self.conv3_2 = nn.Conv2d(c4, c4, 3, padding=1, groups=c4)

        self.norm3 = _LayerNorm(c34, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv2d(c34, c34, 1),
            nn.GELU(),
            nn.Conv2d(c34, c34, k3, padding=k3 // 2, groups=c34),
        )
        self.v3 = nn.Conv2d(c34, c34, 1)
        self.v31 = nn.Conv2d(c34, c34, 1)
        self.v32 = nn.Conv2d(c4, c4, 1)
        self.proj3 = nn.Conv2d(c34, c4, 1)
        self.conv3_3 = nn.Conv2d(c4, c4, 3, padding=1, groups=c4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.dim, f"channel mismatch: expect {self.dim}, got {C}"

        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)

        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = UCM(channels=128)
    output = model(input_tensor)
    print('=== UCM: UniConvNet ConvMod ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
