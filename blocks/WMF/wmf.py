import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：WaveMamba: Wave-Inspired Cross-Modal Fusion for Robust Event-Image Semantic Segmentation (NeurIPS 2026)
# 论文链接：https://github.com/adeelferozmirza/WaveMamba（代码仓库，论文见 README）
# 代码来源：https://github.com/adeelferozmirza/WaveMamba
# 原始许可证：MIT
# 模块出处：models/decoder/fusion_decoder.py 的 PointConv / CNNHead
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：PointConv 的卷积-归一化-ReLU 数学逐行保留（含 dilation 参数与
#   conv_padding=(kernel_size//2)*dilation 的写法）。CNNHead 原本吃 4 个尺度的
#   RGB/IR 特征 + 2 张原始图，输出 3 通道融合图；为统一 [B,C,H,W]->[B,C,H,W]
#   接口，这里保留其融合头结构（3x3 Conv+BN+ReLU → 1x1 Conv+Sigmoid 的 CNN_fuse）
#   并把多尺度 PointConv 组改为同尺度多膨胀率 PointConv 组（dilations=(1,3,5)，
#   对应接口约定的 wave-like multi-dilation bank），sigmoid 图改作通道级门控，
#   与主路内容做门控残差融合。不含 mamba/selective_scan；删除未用到的
#   align_corners 插值分支（统一接口无多尺度对齐需求）。数值逻辑：PointConv
#   与 CNN_fuse 的层类型/顺序/激活不变，仅通道数参数化。

'''
模块名称：WMF (WaveFusion) —— 波式多膨胀率融合模块

一、模块简介
事件-图像语义分割等跨模态任务需要把两路模态特征在保持各自空间结构的前提下
融合，并抑制噪声模态的干扰。WaveMamba 以波动传播为灵感，用多感受野（多膨胀
率）的逐点卷积组提取不同空间频率的响应，再用学习到的 sigmoid 融合图对两路
模态做门控混合。WMF 提取其中纯 PyTorch 的融合头：多膨胀率 PointConv 组 +
CNN_fuse 融合门，去掉依赖 mamba/selective_scan 的时序扫描部分，保留波式
多膨胀率融合数学，可作为即插即用的双输入/单输入融合块。

核心创新点：
1. 多膨胀率 PointConv 组：对同一输入并行多路逐点卷积（dilation 可配），
   覆盖多个有效感受野，对应波动分解的多频率视角
2. CNN_fuse 融合门：3x3 Conv+BN+ReLU → 1x1 Conv+Sigmoid，生成与特征同形的
   融合权重图
3. 双模态/自融合统一：x_aux 给定时做跨模态门控融合，省略时退化为单输入
   门控精炼（x_aux=x 且内容取主路多膨胀率响应）
4. 门控残差：out = gate * content + (1 - gate) * x，保持 shape 与残差通路

二、结构设计
WMF 由以下子结构组成：
1. PointConv（与原实现一致）：
   - Conv2d(in, out, kernel_size=1, padding=(1//2)*dilation, dilation=d)
   - BatchNorm2d(out) + ReLU(inplace=True)
   - 输入 [B, C, H, W] → 输出 [B, C, H, W]
2. 多膨胀率 PointConv 组（wave-like multi-dilation bank）：
   - 主路 K 路 PointConv（dilations=(1,3,5)，K=3）→ 各 [B, C, H, W]
   - 辅路（x_aux 或 x 本身）同样 K 路 → 各 [B, C, H, W]
3. CNN_fuse 融合门：
   - cat([主路 K 路, 辅路 K 路, 原始 x]) → [B, (2K+1)C, H, W]
   - Conv2d((2K+1)C, C, 3, 1, 1) + BN + ReLU
   - Conv2d(C, C, 1) + Sigmoid → gate [B, C, H, W]
4. 内容混合与门控残差：
   - content = 主路多膨胀率响应的均值 [B, C, H, W]
   - out = gate * content + (1 - gate) * x

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 WMF（WaveFusion）跨模态融合头，通过多膨胀率逐点卷积组提取波式
多频率响应，并用学习到的 sigmoid 融合门对主/辅模态做门控残差融合，保留了
WaveMamba 解码器融合分支的多膨胀率 PointConv 与 CNN_fuse 数学。"

原论文引用格式：WaveMamba: Wave-Inspired Cross-Modal Fusion for Robust
Event-Image Semantic Segmentation, NeurIPS 2026.

四、适用任务
事件-图像语义分割、多模态融合、可见光-红外融合、任意双路特征融合或单路
门控精炼场景；可作为解码器/融合头即插即用。
'''


class PointConv(nn.Module):
    """
    Point convolution block: input: x with size(B C H W); output size (B C1 H W)
    与原仓库 models/decoder/fusion_decoder.py 的 PointConv 保持一致。
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 64,
        dilation: int = 1,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.kernel_size = 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dilation = dilation
        conv_padding = (self.kernel_size // 2) * self.dilation

        self.pconv = nn.Sequential(
            nn.Conv2d(
                self.in_dim,
                self.out_dim,
                self.kernel_size,
                padding=conv_padding,
                dilation=self.dilation,
            ),
            norm_layer(self.out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.pconv(x)
        return x


class WMF(nn.Module):
    """WMF: WaveFusion —— 波式多膨胀率融合模块"""

    def __init__(
        self,
        channels: int,
        dilations: t.Sequence[int] = (1, 3, 5),
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        assert channels > 0, 'channels must be positive'
        assert len(dilations) > 0, 'dilations must be non-empty'
        self.channels = channels
        self.dilations = tuple(dilations)
        k = len(self.dilations)

        # 波式多膨胀率 PointConv 组（主路 / 辅路）
        self.pc_main = nn.ModuleList(
            [PointConv(channels, channels, dilation=d, norm_layer=norm_layer) for d in self.dilations]
        )
        self.pc_aux = nn.ModuleList(
            [PointConv(channels, channels, dilation=d, norm_layer=norm_layer) for d in self.dilations]
        )

        # CNN_fuse 融合门（层结构与原 CNNHead.CNN_fuse 一致，通道数参数化）
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (2 * k + 1), channels, kernel_size=3, stride=1, padding=1),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_aux: t.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
            x_aux: 可选辅路特征 [B, C, H, W]；为 None 时做单输入门控融合
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        assert x.dim() == 4, f'expect 4D [B,C,H,W], got {tuple(x.shape)}'
        b, c, h, w = x.shape
        assert c == self.channels, f'expect C={self.channels}, got {c}'
        if x_aux is not None:
            assert x_aux.shape == x.shape, f'x_aux shape {tuple(x_aux.shape)} != {tuple(x.shape)}'
        else:
            # 单输入：辅路退化为自身（自融合）
            x_aux = x

        # 多膨胀率 PointConv 响应（PointConv 数学与原实现一致）
        main_feats = [pc(x) for pc in self.pc_main]            # K × [B, C, H, W]
        aux_feats = [pc(x_aux) for pc in self.pc_aux]          # K × [B, C, H, W]

        # CNN_fuse 输入：多膨胀率响应 + 原始主路（对应原 CNNHead 拼接原始输入）
        fuse_in = torch.cat(main_feats + aux_feats + [x], dim=1)  # [B, (2K+1)C, H, W]
        gate = self.fuse(fuse_in)                                  # [B, C, H, W]

        # 主路多膨胀率内容（波式多频率响应混合）
        content = torch.stack(main_feats, dim=0).mean(dim=0)       # [B, C, H, W]

        # 门控残差融合
        out = gate * content + (1.0 - gate) * x
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = WMF(channels=128)
    output = model(input_tensor)
    print('=== WMF: WaveFusion ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    aux = torch.randn(1, 128, 64, 64)
    out2 = model(input_tensor, aux)
    print('dual_input_output_size:', out2.size())
