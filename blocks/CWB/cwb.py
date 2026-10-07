import math
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：CWNet: Causal Wavelet Network for Low-Light Image Enhancement (ICCV 2025)
# 论文链接：https://openaccess.thecvf.com/（ICCV 2025；代码仓库见下）
# 代码来源：https://github.com/bywlzts/CWNet-Causal-Wavelet-Network
# 原始许可证：MIT
# 模块出处：models/archs/CWNet.py 的 ProcessBlock（因果小波分解/重建路径）
#   + models/archs/arch_util.py 的 dwt_init / iwt_init（DWT/IDWT）
#   + models/archs/wtconv/wtconv2d.py 的 WTConv2d（LL 增强）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：因果 Haar 分解/重建 dwt_init / iwt_init 数学逐行保留（子采样 +
#   加减组合的可逆因果小波）。LL 增强 WTConv2d 的多级小波卷积-逐级重建结构
#   与 wavelet_transform / inverse_wavelet_transform 的 conv2d /
#   conv_transpose2d 数学不变；原实现用 pywt.Wavelet('db1') 生成滤波器，
#   这里改为直接以 register_buffer 注册 db1(Haar) 系数（1/√2），消除 pywt
#   依赖，滤波器数值与 pywt db1 一致。高频增强 Depth_conv、水平/垂直/对角
#   wave-conv 核（create_wave_conv）与 conv_fusechannel 均逐行保留。按提取
#   规范删除 SS2D6(mamba) 高频扫描与 LightBlock/FFC(posenhance) 路径（依赖
#   mamba 与 FFC 扩展，超出纯 torch 范围）；因果小波分解-增强-重建主干不变。

'''
模块名称：CWB (Causal Wavelet Block) —— 因果小波变换块

一、模块简介
低光图像增强需要在放大暗区信号的同时不放大噪声，且要保留清晰的边缘结构。
CWNet 提出因果小波网络：用可逆的因果 Haar 小波把特征分解为低频 LL 与
高频 HL/LH/HH 子带，对低频做多级小波卷积增强、对高频做深度卷积增强，再用
固定方向核（水平/垂直/对角）把低频的结构信息引导回高频子带，最后经因果
重建并残差回原特征。由于分解/重建是精确可逆的线性变换，信息在子带间流动
是“因果”的（由粗到细、由低频引导高频），有利于低光增强中的结构保持与噪声
抑制。本模块提取其中纯 PyTorch 的因果小波变换块，可即插即用。

核心创新点：
1. 因果 Haar 分解/重建：子采样 + 加减组合，严格可逆（dwt_init / iwt_init）
2. 多级小波卷积增强（WTConv2d）：对 LL 逐级做 db1 小波变换-子带卷积-逆变换
3. 方向核引导：水平/垂直/对角固定核把 LL 结构注入 HL/LH/HH
4. 高频深度卷积增强 + 通道融合 + 残差重建

二、结构设计
CWB 由以下子结构组成：
1. 因果 DWT（dwt_init）：
   - x → [LL, HL, LH, HH]，各 [B, C, H/2, W/2]
   - LL = x1+x2+x3+x4, HL = -x1-x2+x3+x4, LH = -x1+x2-x3+x4, HH = x1-x2-x3+x4
     （x1..x4 为 2×2 子采样块，各除以 2）
2. LL 增强（WTConv2d × n_ll）：
   - 每级：db1 小波变换 → 子带分组卷积 + scale → 逆变换，与 base 卷积残差
3. 高频增强（Depth_conv）：
   - cat(HL, LH, HH) 沿 batch 维拼接 → 3×3 深度卷积 + 1×1 点卷积
4. 方向核引导（create_wave_conv）：
   - 水平核 [[1,0,-1]×3]、垂直核 [[1,1,1],[0,0,0],[-1,-1,-1]]、
     对角核 [[0,1,0],[1,-4,1],[0,1,0]]，权重=核 repeat 到 [C, C, 3, 3]
   - ll_hl = horizontal_conv(ll) 等，与对应高频子带沿通道拼接
5. 通道融合与因果 IDWT：
   - conv_fusechannel: Conv2d(2C, C, 1) + LayerNorm
   - iwt_init 把 [LL, e_high] 四组重建为 [B, C, H, W]，再加残差 xori

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 CWB（Causal Wavelet Block）进行因果小波变换：以可逆 Haar 小波
把特征分解为低/高频子带，对低频做多级小波卷积增强、对高频做深度卷积增强，
并用固定方向核把低频结构引导至高频子带，最后经因果重建并残差输出，从而在
低光增强中保持结构并抑制噪声（CWNet, ICCV 2025）。"

原论文引用格式：CWNet: Causal Wavelet Network for Low-Light Image
Enhancement, ICCV 2025.

四、适用任务
低光图像增强、图像去噪、去雨去雾等低层视觉恢复任务；也可作为任意骨干中的
多频带特征增强块。
'''


# ---------------------------------------------------------------------- #
# 因果 Haar 分解 / 重建（与原 dwt_init / iwt_init 一致）
# ---------------------------------------------------------------------- #
def _dwt_init(x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def _iwt_init(x: torch.Tensor) -> torch.Tensor:
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch = int(in_batch / (r ** 2))
    out_channel = in_channel
    out_height = r * in_height
    out_width = r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    """因果 Haar 分解（与原 arch_util.DWT 一致，fuseh=False 路径）。"""

    def __init__(self, fuseh: bool = False):
        super().__init__()
        self.requires_grad = False
        self.fuseh = fuseh

    def forward(self, x: torch.Tensor):
        if self.fuseh:
            x_LL, x_HL, x_LH, x_HH = _dwt_init(x)
            return x_LL, torch.cat((x_HL, x_LH, x_HH), dim=0)
        return _dwt_init(x)


class IDWT(nn.Module):
    """因果 Haar 重建（与原 arch_util.IDWT 一致）。"""

    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _iwt_init(x)


# ---------------------------------------------------------------------- #
# 纯 torch 小波滤波器与小波卷积（替代 pywt，滤波器注册为 buffer）
# ---------------------------------------------------------------------- #
def _create_wavelet_filter(in_size: int, out_size: int, dtype: torch.dtype = torch.float):
    """db1(Haar) 滤波器，数值与 pywt.Wavelet('db1') 一致，去掉 pywt 依赖。

    原 wavelet.create_wavelet_filter 对 dec/dec_lo、rec 系数做 [::-1]（rec 再
    flip）后做外积堆叠并按通道数 repeat；此处逐行等价实现。
    """
    s = 1.0 / math.sqrt(2.0)
    # w.dec_lo=[s,s], w.dec_hi=[-s,s]；原实现 dec_* = w.dec_*[::-1]
    dec_lo = torch.tensor([s, s], dtype=dtype)
    dec_hi = torch.tensor([s, -s], dtype=dtype)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )  # [4, 2, 2]
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)  # [4*in, 1, 2, 2]

    # w.rec_lo=[s,s], w.rec_hi=[s,-s]；原实现 rec_* = w.rec_*[::-1].flip(0)
    rec_lo = torch.tensor([s, s], dtype=dtype)
    rec_hi = torch.tensor([s, -s], dtype=dtype)
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )  # [4, 2, 2]
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)  # [4*out, 1, 2, 2]

    return dec_filters, rec_filters


def _wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def _inverse_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    """可学习通道缩放（与原 wtconv2d._ScaleModule 一致）。"""

    def __init__(self, dims: t.Sequence[int], init_scale: float = 1.0):
        super().__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    """多级小波卷积（与原 wtconv2d.WTConv2d 一致；滤波器改为 buffer）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 1,
    ):
        super().__init__()
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_filter, iwt_filter = _create_wavelet_filter(in_channels, in_channels, torch.float)
        self.register_buffer('wt_filter', wt_filter)
        self.register_buffer('iwt_filter', iwt_filter)

        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding='same',
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding='same',
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(
                x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels
            )
        else:
            self.do_stride = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = _wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = _inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


# ---------------------------------------------------------------------- #
# 高频深度卷积增强（与原 Depth_conv 一致）
# ---------------------------------------------------------------------- #
class Depth_conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# ---------------------------------------------------------------------- #
# 通道 LayerNorm（与原 arch_util.LayerNorm 的 WithBias 路径一致）
# ---------------------------------------------------------------------- #
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class LayerNorm(nn.Module):
    """WithBias 通道 LayerNorm，输入 [B, C, H, W]。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.body = WithBias_LayerNorm(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x3 = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        y = self.body(x3)
        return y.reshape(b, h, w, c).permute(0, 3, 1, 2)


class CWB(nn.Module):
    """CWB: Causal Wavelet Block —— 因果小波变换块"""

    def __init__(
        self,
        channels: int,
        n_ll_enhance: int = 2,
        wt_levels: int = 3,
        wt_kernel_size: int = 5,
    ):
        super().__init__()
        assert channels > 0, 'channels must be positive'
        self.channels = channels
        self.dim = channels

        # 因果 Haar 分解 / 重建
        self.dwt = DWT(fuseh=False)
        self.idwt = IDWT()

        # LL 多级小波卷积增强
        self.llenhance = nn.ModuleList(
            [
                WTConv2d(channels, channels, kernel_size=wt_kernel_size, wt_levels=wt_levels)
                for _ in range(n_ll_enhance)
            ]
        )

        # 高频深度卷积增强
        self.hhenhance = Depth_conv(self.dim, self.dim)

        # 方向核引导（水平 / 垂直 / 对角），权重=固定核 repeat，与原 create_wave_conv 一致
        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self._create_wave_conv()

        # 通道融合
        self.conv_fusechannel = nn.Conv2d(self.dim * 2, self.dim, 1, stride=1, bias=False)
        self.norm2 = LayerNorm(self.dim)

    def _create_conv_layer(self, kernel: torch.Tensor) -> nn.Conv2d:
        conv = nn.Conv2d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1, bias=False
        )
        conv.weight.data = kernel.repeat(self.dim, self.dim, 1, 1)
        return conv

    def _create_wave_conv(self):
        horizontal_kernel = torch.tensor(
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        vertical_kernel = torch.tensor(
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        diagonal_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        horizontal_conv = self._create_conv_layer(horizontal_kernel)
        vertical_conv = self._create_conv_layer(vertical_kernel)
        diagonal_conv = self._create_conv_layer(diagonal_kernel)
        return horizontal_conv, vertical_conv, diagonal_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        assert x.dim() == 4, f'expect 4D [B,C,H,W], got {tuple(x.shape)}'
        b, c, h, w = x.shape
        assert c == self.channels, f'expect C={self.channels}, got {c}'
        assert h % 2 == 0 and w % 2 == 0, f'H,W must be even for causal DWT, got {h}x{w}'

        xori = x
        ll, hl, lh, hh = self.dwt(x)                    # 各 [B, C, H/2, W/2]

        for layer in self.llenhance:
            ll = layer(ll)                              # LL 多级小波卷积增强

        hh = torch.cat((hl, lh, hh), dim=0)             # [3B, C, H/2, W/2]
        hh = self.hhenhance(hh)                         # 高频深度卷积增强
        e_hl, e_lh, e_hh = hh[:b, ...], hh[b:2 * b, ...], hh[2 * b:, ...]

        ll_hl = self.horizontal_conv(ll)                # 方向核引导
        ll_lh = self.vertical_conv(ll)
        ll_hh = self.diagonal_conv(ll)

        e_hl = torch.cat((e_hl, ll_hl), dim=1)          # [B, 2C, H/2, W/2]
        e_lh = torch.cat((e_lh, ll_lh), dim=1)
        e_hh = torch.cat((e_hh, ll_hh), dim=1)

        e_high = torch.cat((e_hl, e_lh, e_hh), dim=0)   # [3B, 2C, H/2, W/2]
        e_high = self.conv_fusechannel(e_high)          # [3B, C, H/2, W/2]
        e_high = self.norm2(e_high)

        x_out = self.idwt(torch.cat((ll, e_high), dim=0)) + xori   # 因果重建 + 残差
        return x_out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CWB(channels=128)
    output = model(input_tensor)
    print('=== CWB: Causal Wavelet Block ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
