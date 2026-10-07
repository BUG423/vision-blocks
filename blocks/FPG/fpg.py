import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network (CVPR 2026)
# 论文链接：https://arxiv.org/abs/2602.20537
# 代码来源：https://github.com/fhjdqaq/PFGNet
# 原始许可证：Apache-2.0
# 模块出处：openstl/models/pfg_model.py 的 MidPFG 类（L48）及依赖
#          openstl/modules/pfg_modules.py 的 MSInit / PFG，
#          openstl/modules/layers/pfg.py 的 GRN / _RepDWLite / PFGA
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删除 openstl 方法框架 / registry / timm（DropPath 以纯 PyTorch 重写，
#          trunc_normal_ 未被本模块使用故不引入）；原 MidPFG 前向接收时空张量
#          [B, T, C, H, W] 并按 x.view(b, t*c, h, w) 把时间维折进通道再折回，
#          本提取取 T=1 语义（t*c = c），直接在单张 [B, C, H, W] 特征图上运行，
#          频率分解（Sobel/Laplacian/局部方差）与中心/外周门控（PFGA）的逐像素
#          数学完全一致；MSInit 多尺度初始化与 PFG 块（PFGA token 混合 +
#          GLU 通道混合 + GRN + LayerScale + DropPath）公式原样保留。

'''
模块名称：FPG (Frequency-guided Peripheral Gating) —— 频率引导外周门控

一、模块简介
全卷积网络在时空预测（视频预报等）中缺少全局感受野时，常退化为局部
平滑，难以区分"平坦背景"与"需要精细外周上下文的运动区域"。PFGNet
提出频率引导的外周门控（PFGA）：用固定的频率算子（Sobel 梯度、
Laplacian、局部方差）刻画每个像素的频率活跃度，据此对多个大核
外周聚合分支做逐像素 softmax 门控，并用可学习中心抑制项显式压制
中心响应，使网络自适应地放大外周上下文、抑制中心冗余。

核心创新点：
1. 频率分解：固定 Sobel_x / Sobel_y / Laplacian 核 + 局部方差，得到
   三张频率图（梯度幅值、Laplacian 幅值、局部方差），逐通道平均；
2. 频率引导门控：1x1 卷积把 3 张频率图映射为 len(K_list) 个尺度的
   门控 logits，softmax(dim=1) 得到逐像素尺度权重；
3. 外周聚合 + 中心抑制：每个尺度用 DW(1xK)+DW(Kx1) 近似 KxK 外周卷积，
   减去 tanh(beta) * DW3x3 中心路径，显式抑制中心、突出外周；
4. 多尺度融合：各尺度外周响应按频率门控权重逐像素加权求和；
5. 配套 PFG 块：GLU 式通道混合（1x1 → DW → 1x1）+ GRN 全局响应归一化
   + LayerScale 残差，稳定训练。

二、结构设计
FPG 由以下子结构组成（输入/输出均为 [B, C, H, W]）：
1. MSInit（原 cel，多尺度初始化）：len(k_list) 个分支，每分支
   _RepDWLite(DW(1xK)+DW(Kx1)+DW(3x3)+DW(1x1)) + 1x1 降到 C/len(k_list)，
   拼接后 GroupNorm(1, C) + GELU；
2. depth 个 PFG 块（默认 depth=4，与 PFG_Model 的 N_T 一致）：
   a) token 混合：GroupNorm → PFGA → GELU → GRN → Dropout → LayerScale 残差；
   b) 通道混合：GroupNorm → 1x1 升到 2E → 切成 u,v → v 过 DW3x3 →
      silu(u)*v → 1x1 降回 C → GRN → Dropout → LayerScale 残差；
   E = max(C, int(C * mlp_ratio))，mlp_ratio 默认 4.0；
3. PFGA 内部形状流转：x [B,C,H,W] → 各尺度外周响应 peris[i] [B,C,H,W]，
   频率图 Freq [B,3,H,W] → gate_head → logits [B,len(K_list),H,W] →
   softmax(dim=1) → alpha，Y = Σ_i peris[i] * alpha[:, i:i+1]。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 PFGNet（CVPR 2026）提出的频率引导外周门控模块 FPG：以固定
Sobel/Laplacian/局部方差算子分解出频率活跃度图，逐像素地对多尺度外周
聚合分支做 softmax 门控，并通过可学习中心抑制项突出外周上下文，在纯
卷积结构下获得自适应的频率-外周感受野调节能力。"

四、适用任务
原论文面向视频时空预测（Moving MNIST / KTH / WeatherBench 等，配合
SimVP 式编解码器）。作为通用特征算子，也可用于图像分类/分割/检测主干
中的特征细化块，尤其适合需要大感受野外周上下文且对局部频率敏感的任务
（运动估计、去模糊、显著性检测）。
'''

__all__ = ['FPG']


class _DropPath(nn.Module):
    """Stochastic depth（纯 PyTorch 重写 timm.layers.DropPath，行为一致）。"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GRN(nn.Module):
    """Global Response Normalization（ConvNeXt V2 风格，原 layers.pfg.GRN）。

    y = x + gamma * (x / ||x||_2) + beta
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))   # learnable scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))   # learnable bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)     # L2 over spatial dims
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta


class _RepDWLite(nn.Module):
    """轻量重参数化深度可分离组合：DW(1xK) -> DW(Kx1) + DW(3x3) + DW(1x1)。

    原 layers.pfg._RepDWLite，数值逻辑不变。
    """

    def __init__(self, dim: int, K: int, stride: int = 1):
        super().__init__()

        # separable large-kernel approximation
        self.dw_h = nn.Conv2d(
            dim, dim, kernel_size=(1, K),
            stride=(1, stride), padding=(0, K // 2),
            groups=dim, bias=False
        )
        self.dw_v = nn.Conv2d(
            dim, dim, kernel_size=(K, 1),
            stride=(stride, 1), padding=(K // 2, 0),
            groups=dim, bias=False
        )

        # complementary 3x3 + identity-like DW(1x1)
        self.dw_s = nn.Conv2d(
            dim, dim, kernel_size=3,
            stride=stride, padding=1, dilation=1,
            groups=dim, bias=False
        )
        self.dw_i = nn.Conv2d(
            dim, dim, kernel_size=1,
            stride=stride, groups=dim, bias=False
        )
        nn.init.dirac_(self.dw_i.weight)  # start as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_v(self.dw_h(x)) + self.dw_s(x) + self.dw_i(x)


class PFGA(nn.Module):
    """Peripheral-Frequency Guided Aggregation（原 layers.pfg.PFGA，公式不变）。

    - 多尺度大核深度可分离外周分支（DW(1xK) + DW(Kx1)）
    - 逐像素频率门控（Sobel / Laplacian / 局部方差线索）
    - 可选中心抑制（tanh(beta) * DW3x3 中心路径）
    """

    class Branch(nn.Module):
        def __init__(self, dim: int, K: int, center_suppress: bool = True):
            super().__init__()
            self.center_suppress = center_suppress

            # approximate KxK with DW(1xK) + DW(Kx1)
            self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, K),
                                  padding=(0, K // 2), groups=dim, bias=False)
            self.dw_v = nn.Conv2d(dim, dim, kernel_size=(K, 1),
                                  padding=(K // 2, 0), groups=dim, bias=False)

            # optional 3x3 center path for suppression
            if self.center_suppress:
                self.dw_c = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                                      groups=dim, bias=False)
                self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            else:
                self.register_parameter('beta', None)
                self.dw_c = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.dw_v(self.dw_h(x))
            if self.center_suppress:
                center = self.dw_c(x)
                y = y - torch.tanh(self.beta) * center   # explicit center suppression
            return y

    def __init__(self, dim: int, K_list: t.Sequence[int] = (9, 15, 31),
                 use_grn: bool = False, center_suppress: bool = True):
        super().__init__()
        self.dim = dim
        self.K_list = tuple(K_list)

        # multi-scale peripheral branches
        self.branches = nn.ModuleList([
            PFGA.Branch(dim, K, center_suppress=center_suppress) for K in self.K_list
        ])

        # fixed frequency filters (buffers): Sobel x/y + Laplacian
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplace", laplace, persistent=False)

        # 1x1 conv to produce per-scale gating logits
        self.gate_head = nn.Conv2d(3, len(self.K_list), kernel_size=1, bias=True)

        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(dim)

    # depthwise apply fixed 3x3 kernels to all channels
    def _depthwise_filter(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = k.repeat(C, 1, 1, 1)
        return F.conv2d(x, w, padding=1, groups=C)

    # build frequency maps: gradient magnitude, Laplacian magnitude, local variance
    def _freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)

        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        mean = F.avg_pool2d(x, 3, 1, 1)
        mean2 = F.avg_pool2d(x * x, 3, 1, 1)
        var = torch.clamp(mean2 - mean * mean, min=0.)

        f1 = grad_mag.mean(dim=1, keepdim=True)
        f2 = lap.abs().mean(dim=1, keepdim=True)
        f3 = var.mean(dim=1, keepdim=True)
        return torch.cat([f1, f2, f3], dim=1)  # (B,3,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # peripheral responses at multiple scales
        peris = [b(x) for b in self.branches]

        # per-pixel softmax over scales from frequency cues
        Freq = self._freq_maps(x)
        logits = self.gate_head(Freq)
        alpha = torch.softmax(logits, dim=1)  # (B,K,H,W)

        # pixel-wise fusion
        Y = 0.
        for i, y in enumerate(peris):
            Y = Y + y * alpha[:, i:i + 1, :, :]

        if self.use_grn:
            Y = self.grn(Y)
        return Y


class MSInit(nn.Module):
    """多尺度初始化（原 pfg_modules.MSInit，即 MidPFG 的 cel）。"""

    def __init__(self, in_ch: int, out_ch: int, k_list: t.Sequence[int] = (3, 5, 7),
                 stride: int = 1, use_gn: bool = True):
        super().__init__()
        # equal split for branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                _RepDWLite(in_ch, K=k, stride=stride),
                nn.Conv2d(in_ch, out_ch // len(k_list), 1, bias=False)
            ) for k in k_list
        ])

        # handle remainder channels
        gap = out_ch - (out_ch // len(k_list)) * len(k_list)
        self.tail = nn.Identity() if gap == 0 else nn.Sequential(
            _RepDWLite(in_ch, K=k_list[0], stride=stride),
            nn.Conv2d(in_ch, gap, 1, bias=False)
        )

        self.fuse = nn.Identity()
        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.branches]
        if not isinstance(self.tail, nn.Identity):
            parts.append(self.tail(x))
        y = torch.cat(parts, dim=1)
        return self.act(self.norm(y))


class PFG(nn.Module):
    """主 PFG 块（原 pfg_modules.PFG）：PFGA token 混合 + GLU 通道混合。"""

    def __init__(self,
                 dim: int,
                 groups_pw: int = 1,
                 layerscale_init: float = 1e-6,
                 act_layer: t.Callable[..., nn.Module] = nn.GELU,
                 drop: float = 0.0,
                 drop_path: float = 0.0,
                 pfga_K: t.Sequence[int] = (9, 15, 31),
                 mlp_ratio: float = 4.0,
                 dw_kernel: int = 3):
        super().__init__()
        self.dim = dim

        # lightweight norms before token/channel mixers
        self.norm_dw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.norm_pw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)

        # token mixing by PFGA
        self.tm = PFGA(dim, K_list=pfga_K, use_grn=False)

        # GRN after each stage
        self.grn_dw = GRN(dim)
        self.grn_pw = GRN(dim)

        self.mlp_ratio = mlp_ratio
        self.dw_kernel = dw_kernel

        # GLU-like channel mixing
        E = max(dim, int(dim * self.mlp_ratio))

        self.pw_in = nn.Conv2d(dim, 2 * E, kernel_size=1, bias=True, groups=groups_pw)
        self.dw_v = nn.Conv2d(E, E, kernel_size=self.dw_kernel, padding=1, groups=E, bias=False)
        self.pw_out = nn.Conv2d(E, dim, kernel_size=1, bias=True, groups=groups_pw)

        self.act = act_layer()

        # LayerScale (kept exactly as original)
        self.gamma_dw = nn.Parameter(torch.ones(dim) * layerscale_init)
        self.gamma_pw = nn.Parameter(torch.ones(dim) * layerscale_init)

        self.dropout_dw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.dropout_pw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self._init_params()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma_dw', 'gamma_pw'}

    def _init_params(self):
        # standard conv/norm init, identical to original
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) token mixing
        y = self.norm_dw(x)
        y = self.tm(y)
        y = self.act(y)
        y = self.grn_dw(y)
        y = self.dropout_dw(y)
        x = x + self.drop_path(y * self.gamma_dw.view(1, self.dim, 1, 1))

        # (2) channel mixing (GLU style)
        z = self.norm_pw(x)
        uv = self.pw_in(z)
        u, v = torch.chunk(uv, 2, dim=1)
        v = self.dw_v(v)
        z = F.silu(u) * v
        z = self.pw_out(z)

        z = self.grn_pw(z)
        z = self.dropout_pw(z)
        x = x + self.drop_path(z * self.gamma_pw.view(1, self.dim, 1, 1))
        return x


class FPG(nn.Module):
    """FPG: Frequency-guided Peripheral Gating —— 频率引导外周门控

    原 MidPFG 在 [B,T,C,H,W] 上把 T 折进通道后运行；本类取 T=1 语义，
    直接对单张 [B,C,H,W] 特征图做 MSInit + PFG 堆叠，门控数学不变。
    """

    def __init__(self, channels: int, depth: int = 4,
                 groups_pw: int = 1, layerscale_init: float = 1e-6,
                 cel_k: t.Sequence[int] = (3, 5, 7),
                 drop: float = 0.0, drop_path: float = 0.0,
                 pfga_K: t.Sequence[int] = (9, 15, 31),
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = channels
        self.cel = MSInit(channels, channels, k_list=cel_k, use_gn=True)

        if drop_path > 0 and depth > 0:
            dpr = [x.item() for x in torch.linspace(1e-2, drop_path, depth)]
        else:
            dpr = [0.0] * depth

        self.blocks = nn.Sequential(*[
            PFG(
                channels,
                groups_pw=groups_pw,
                layerscale_init=layerscale_init,
                act_layer=nn.GELU,
                drop=drop,
                drop_path=(drop_path if i == depth - 1 else dpr[i]),
                pfga_K=pfga_K,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        assert x.size(1) == self.channels, f'expected C={self.channels}, got {x.size(1)}'
        x = self.cel(x)
        x = self.blocks(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 32x32 控制 CPU 上大核 DW 的耗时；depth 取默认 4（与 PFG_Model 的 N_T 一致）
    input_tensor = torch.randn(1, 64, 32, 32)
    model = FPG(channels=64)
    output = model(input_tensor)
    print('=== FPG: Frequency-guided Peripheral Gating ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
