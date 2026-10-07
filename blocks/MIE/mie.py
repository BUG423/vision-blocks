import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices (ICCV 2025)
# 论文链接：https://arxiv.org/abs/2507.01838
# 代码来源：https://github.com/AVC2-UESTC/MobileIE
# 原始许可证：Apache-2.0
# 模块出处：model/lle.py 的 MobileIELLENet（body/att/att1 核心增强单元）+ model/utils.py 的
#          MBRConv3 / MBRConv1 / FST
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：抽取 MobileIELLENet 中可复用的轻量增强单元（FST 包裹的 MBRConv3 主体 +
#          SE 式双路注意力 att/att1），数学与原 forward 的 x1/x2/x3/x4 计算逐行一致；
#          网级 3 通道 I/O 适配器（head 的 MBRConv5→PReLU→MBRConv3、tail/tail_warm）
#          与训练专用 forward_warm/DropBlock、部署用 slim() 重参数化不属块级逻辑，删除；
#          MBRConv5（仅 head 使用）随 head 一并删除；MBRConv 的多分支 concat+1x1 压缩
#          与 FST 二次门控 (w1·x)·(w2·x)+bias 公式原样保留，仅统一 nn 写法并补 shape 断言。

'''
模块名称：MIE (MobileIE Enhancement Block) —— MobileIE 轻量增强块

一、模块简介
MobileIE 面向移动端实时图像增强（低照度增强等），核心是用极低参数量的
多分支重参数卷积（MBRConv）替换常规卷积：训练期并联 3×3 / 1×1 / 十字形
(3×1 + 1×3) 多分支及其 BatchNorm 副本，推理期可融合为单个方卷积。
本模块提取其中可复用的增强级联单元：FST 包裹的 MBRConv3 主体 + SE 式双路
注意力（全局通道注意力 att 与由通道最大值驱动的像素注意力 att1）。
核心创新点：
1. MBRConv 多分支感受野（方形 + 十字交叉）以极少参数覆盖多尺度空间模式；
2. FST 特征变换对主体输出做二次门控 (w1·x)·(w2·x)+bias，引入乘性非线性；
3. 双路注意力：att 先对特征做全局池化后的通道重标定，再用 att1 以通道最大
   响应图为线索生成像素级权重，二者相乘对主体特征逐点加权；
4. 全部算子为纯 conv/BN/PReLU，对 TFLite/移动端部署友好。

二、结构设计
输入 x: [B, C, H, W]，中间形状均保持 [B, C, H, W]（att 输出 [B, C, 1, 1] 广播）：
1. 主体 body = FST(MBRConv3(C→C))：
   - MBRConv3：conv3×3 / conv1×1 / conv3×1 / conv1×3 四分支各输出 C·rep_scale
     通道，连同各自 BN 副本 concat 为 8·rep_scale·C 通道，再 1×1 压缩回 C；
   - FST：out = (w1·x1)·(w2·x1) + bias，w1,w2 为标量参数，bias 为 (1,C,1,1)；
2. 通道注意力 att = AdaptiveAvgPool2d(1) → MBRConv1(C→C) → Sigmoid，
   得 x2 ∈ [B, C, 1, 1]；
3. 像素注意力线索：max_out = max_channel(x2 ⊙ x1) ∈ [B, 1, H, W]，
   att1 = MBRConv1(1→C) → Sigmoid 得 x3 ∈ [B, C, H, W]；
4. 输出 out = x2 ⊙ x3 ⊙ x1，形状 [B, C, H, W]。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 MobileIE（ICCV 2025）提出的轻量增强块 MIE：以多分支重参数卷积
MBRConv3（3×3/1×1/十字交叉分支训练并联、推理可融合）作为主体，并用 FST
对主体输出施加二次乘性门控；随后通过全局通道注意力与像素注意力的双路加权
对特征逐点重标定，在几乎不增加计算量的前提下显著提升增强质量。"
（原论文引用格式：@inproceedings{mobileie2025, title={MobileIE: An Extremely
Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile
Devices}, booktitle={ICCV}, year={2025}}）

四、适用任务
适用于低照度图像增强、图像修复、实时 ISP 后处理等底层视觉增强任务，也可作为
任意骨干中的轻量特征增强单元（输入输出同通道数）。特别适合移动端/边缘端
对算力与访存敏感的部署场景。
'''


class MBRConv3(nn.Module):
    """MBRConv3：四分支（3×3/1×1/十字 H/十字 V）+ 各自 BN 副本 concat 后 1×1 压缩。"""

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn = nn.BatchNorm2d(out_channels * rep_scale)
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.BatchNorm2d(out_channels * rep_scale)
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.BatchNorm2d(out_channels * rep_scale)
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.BatchNorm2d(out_channels * rep_scale)
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        x = torch.cat(
            [x0, x1, x2, x3,
             self.conv_bn(x0),
             self.conv1_bn(x1),
             self.conv_crossh_bn(x2),
             self.conv_crossv_bn(x3)],
            1,
        )
        return self.conv_out(x)


class MBRConv1(nn.Module):
    """MBRConv1：1×1 卷积与其 BN 副本 concat 后 1×1 压缩。"""

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.BatchNorm2d(out_channels * rep_scale)
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x0 = self.conv(inp)
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        return self.conv_out(x)


class FST(nn.Module):
    """FST：二次乘性门控 (w1·x1)·(w2·x1) + bias（与原 model/utils.FST 逐式一致）。"""

    def __init__(self, block1: nn.Module, channels: int):
        super().__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias


class MIE(nn.Module):
    """MIE: MobileIE Enhancement Block —— MobileIE 轻量增强块"""

    def __init__(self, channels: int, rep_scale: int = 4):
        super().__init__()
        self.channels = channels
        # 原 MobileIELLENet.body：FST(MBRConv3(channels, channels))
        self.body = FST(MBRConv3(channels, channels, rep_scale=rep_scale), channels)
        # 原 MobileIELLENet.att：GAP → MBRConv1 → Sigmoid
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid(),
        )
        # 原 MobileIELLENet.att1：通道最大值线索 → MBRConv1(1→C) → Sigmoid
        self.att1 = nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        x1 = self.body(x)                                              # [B, C, H, W]
        x2 = self.att(x1)                                              # [B, C, 1, 1]
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)           # [B, 1, H, W]
        x3 = self.att1(max_out)                                        # [B, C, H, W]
        out = torch.mul(x2, x3) * x1
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # att 中 BN 作用在 1×1 空间上，train 模式要求 batch>1（与原仓库训练行为一致）
    input_tensor = torch.randn(2, 128, 32, 32)
    model = MIE(channels=128)
    output = model(input_tensor)
    print('=== MIE: MobileIE Enhancement Block ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
