import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight SR (CVPR 2026)
# 论文链接：https://arxiv.org/abs/2603.11680
# 代码来源：https://github.com/hokiyoshi/UCAN
# 原始许可证：Apache-2.0
# 模块出处：basicsr/archs/ucan_arch.py 的 LKSA 类
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删除 timm / einops / basicsr 依赖与 ARCH_REGISTRY；参数 dim→channels、ratio→reduction
#          （语义不变：reduction 为动态 3×3 核分支的瓶颈倒数）；删除未使用的 mid=int(dim*ratio)
#          与 self.k_size（forward 未引用）；原类自带 4D 接口，静态大核分解（水平/垂直、
#          空洞、可选大核扩展）与动态 3×3 核分支、1×1 门控融合的数值公式原样保留：
#          out = conv1x1(x) * (large_kernel_path(x) + conv3_branch(x))。

'''
模块名称：LKS (Large-Kernel Spatial Attention) —— 大核空间注意力

一、模块简介
轻量超分辨率网络受限于小卷积核，感受野不足，难以利用长程空间上下文。
UCAN 提出的 LKS（原 LKSA）用"静态大核分解 + 动态 3×3 核"的空间注意力
实现超大感受野：静态路径把 k×k 大核分解为水平/垂直两串深度可分离卷积，
并叠加空洞卷积与更大核的扩展段，以极低参数量获得 k∈{7,11,23,35,41,53}
的空间覆盖；动态路径用 1×1 降维 + 3×3 + 1×1 升维的瓶颈卷积补充局部细节。
两路相加后经 1×1 卷积生成逐通道门控，对输入做乘性空间调制。

核心创新点：
1. 大核空间注意力：把 k×k 大卷积核分解为 (1,k)+(k,1) 两串深度可分离卷积，
   并用空洞卷积与额外大核段把感受野扩展到 35×35 及以上；
2. 动态 3×3 核分支：以 ratio（reduction）瓶颈卷积补充局部纹理，与大核路径互补；
3. 乘性融合：1×1 卷积的门控 × (大核响应 + 动态核响应)，形成空间注意力图；
4. 轻量即插即用：纯 Conv2d 实现，可直接替换 SR / 通用视觉主干中的空间注意力。

二、结构设计
LKS 由以下子结构组成（形状以输入 [B, C, H, W] 计，k 默认 35）：
1. 静态大核分解路径（全部 depth-wise，groups=C）：
   - conv_h：(1, core_k) 水平卷积，core_k = 5（k=35 时）；
   - conv_v：(core_k, 1) 垂直卷积；
   - conv_dh / conv_dv：同形空洞卷积，dilation = dil（k=35 时为 3）；
   - conv_big_h / conv_big_v：(1, extra_k)/(extra_k, 1) 大核扩展段，
     extra_k = 11（k=35 时），dilation = dil；k<35 时为 Identity；
   张量流转：[B,C,H,W] → … → large_kn [B,C,H,W]（padding 保证尺寸不变）。
2. 动态 3×3 核分支 conv3：
   Conv2d(C→C/reduction, 1×1) → GELU → Conv2d(C/reduction→C/reduction, 3×3)
   → GELU → Conv2d(C/reduction→C, 1×1)，得 [B,C,H,W]。
3. 1×1 融合门控：channel = Conv2d(C→C, 1×1)；
   输出 out = channel(x) * (large_kn + conv3(x))，[B,C,H,W] → [B,C,H,W]。
k 与 (core, dilate, extra) 的对应见 STATIC_SPECS：
7→(3,1,无)、11→(5,1,无)、23→(5,2,无)、35→(5,3,11)、41→(5,3,13)、53→(5,3,17)。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 UCAN（CVPR 2026）提出的大核空间注意力 LKS：将 k×k 大卷积核
分解为水平/垂直深度可分离卷积并叠加空洞与大核扩展段，配合 ratio 瓶颈的
动态 3×3 核分支与 1×1 门控融合，在几乎不增加参数的前提下获得超大空间
感受野。"（原论文引用格式：LKS 为 UCAN 中 Large-Kernel Spatial Attention 模块）

四、适用任务
适用于图像超分辨率、去噪、修复等轻量复原任务，也可作为通用空间注意力
即插即用到检测/分割/生成主干。适合需要大感受野但参数/算力受限的轻量模型。
'''


class LKS(nn.Module):
    """LKS: Large-Kernel Spatial Attention —— 大核空间注意力（原 UCAN 的 LKSA）"""

    # 静态大核分解规格：k → (core 核, 空洞率, 可选大核扩展)
    STATIC_SPECS: t.Dict[int, t.Dict[str, int]] = {
        7: dict(core=3, dilate=1),
        11: dict(core=5, dilate=1),
        23: dict(core=5, dilate=2),
        35: dict(core=5, dilate=3, extra=11),
        41: dict(core=5, dilate=3, extra=13),
        53: dict(core=5, dilate=3, extra=17),
    }

    def __init__(self, channels: int, k: int = 35, reduction: int = 4):
        super().__init__()
        assert k in self.STATIC_SPECS, f'Unsupported k={k}, must be one of {sorted(self.STATIC_SPECS)}'
        assert channels % reduction == 0, \
            f'channels {channels} must be divisible by reduction {reduction}'
        spec = self.STATIC_SPECS[k]
        core_k = spec['core']
        dil = spec['dilate']
        extra_k = spec.get('extra', 0)

        self.channels = channels
        self.k = k
        self.reduction = reduction

        # ---------- 静态大核分解路径（depth-wise） ----------
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, core_k),
                                padding=(0, (core_k - 1) // 2), groups=channels)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=(core_k, 1),
                                padding=((core_k - 1) // 2, 0), groups=channels)
        self.conv_dh = nn.Conv2d(channels, channels, kernel_size=(1, core_k),
                                 padding=(0, dil * (core_k // 2)), dilation=dil, groups=channels)
        self.conv_dv = nn.Conv2d(channels, channels, kernel_size=(core_k, 1),
                                 padding=(dil * (core_k // 2), 0), dilation=dil, groups=channels)

        if extra_k:  # k >= 35 时启用大核扩展段
            self.conv_big_h = nn.Conv2d(channels, channels, kernel_size=(1, extra_k),
                                        padding=(0, dil * (extra_k // 2)), dilation=dil, groups=channels)
            self.conv_big_v = nn.Conv2d(channels, channels, kernel_size=(extra_k, 1),
                                        padding=(dil * (extra_k // 2), 0), dilation=dil, groups=channels)
        else:
            self.conv_big_h = self.conv_big_v = nn.Identity()

        # ---------- 动态 3×3 核分支（ratio 瓶颈） ----------
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels // reduction, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, 1, 0),
        )

        # ---------- 1×1 门控融合 ----------
        self.channel = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        # ---------------- 静态大核分解路径 ----------------
        y = self.conv_h(x)
        y = self.conv_v(y)
        y = self.conv_dh(y)
        y = self.conv_dv(y)
        y = self.conv_big_h(y)
        large_kn = self.conv_big_v(y)                       # [B, C, H, W]

        # 门控 × (大核响应 + 动态 3×3 核响应)
        out = self.channel(x) * (large_kn + self.conv3(x))  # [B, C, H, W]

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = LKS(channels=128)
    output = model(input_tensor)
    print('=== LKS: Large-Kernel Spatial Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
