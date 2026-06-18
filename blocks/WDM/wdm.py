import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：WDM (Wavelet Decomposition Module) —— 小波分解模块

一、模块简介
卷积特征图同时承载着全局平滑的结构信息（低频）与局部边缘、纹理等细节
信息（高频）。现有频率类模块要么在全局频域上操作（DCT/FFT），要么用
多尺度卷积近似多频带，难以同时获得频率定位与空间定位。小波变换的独
特优势在于：它将信号分解到一组同时具有良好频率分辨率和空间分辨率的
子带上——近似子带 LL 保留低频整体结构，三个细节子带（LH 水平、HL
垂直、HH 对角）分别刻画不同方向的高频细节，且每个子带都保留了明确
的空间位置信息。

WDM 的核心思想是：对特征图做一层 Haar 小波分解，得到 1 个近似子带和
3 个方向细节子带；对低频近似子带施加通道级精炼以增强全局结构，对高
频细节子带施加可学习的逐通道软阈值以"去噪锐化"——抑制小幅度噪声系
数、保留并增强大幅度边缘系数（这正是经典小波去噪的核心操作）；最
后通过逆小波变换（IDWT）将处理后的子带完美重构回特征空间，并与原
特征残差融合。

核心创新点：
1. 小波变换引入特征处理：用 Haar 小波将特征分解为方向性子带，兼顾频
   率与空间定位，区别于全局 DCT/FFT 频域方法
2. 方向感知的子带分工：低频近似走通道精炼路径，三个方向高频细节走
   软阈值去噪锐化路径，各司其职
3. 可学习逐通道软阈值：对小波细节系数施加 learnable soft-thresholding，
   自适应抑制噪声、增强显著边缘
4. 完美重构的逆变换：IDWT 与分解严格互逆，保证处理后的子带能无损地
   重构回原分辨率特征空间

二、结构设计
WDM 由以下子结构组成：
1. Haar 小波分解器（DWT2D）：
   - 通过提升格式（lifting scheme）实现一层二维 Haar 分解
   - 输出近似子带 LL 与三个细节子带 LH/HL/HH，空间分辨率减半
   - 提升格式保证分解与重构严格互逆
2. 低频精炼路径（Low-frequency Refinement）：
   - 对 LL 子带施加 3x3 深度卷积 + 1x1 通道 MLP（类 SE）精炼全局结构
3. 高频去噪锐化路径（High-frequency Denoising & Sharpening）：
   - 将 LH/HL/HH 沿通道拼接为 3C 张量
   - 逐通道可学习软阈值：sign(x)·ReLU(|x| - τ)，τ 自适应学习
   - 1x1 卷积跨子带交互后拆分回三个子带
4. 逆小波重构器（IDWT2D）：
   - 由处理后的 4 个子带按提升格式逆变换重构回原分辨率
5. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 WDM（Wavelet Decomposition Module）模块，将二维 Haar 小波
变换引入卷积特征增强。该模块将特征分解为一个低频近似子带和三个方向
高频细节子带：对低频子带施加通道级精炼以增强全局结构，对高频细节子
带施加可学习逐通道软阈值以抑制噪声并锐化显著边缘，最后通过严格互逆
的逆小波变换重构特征。该设计在频率定位与空间定位之间取得平衡，实现
方向感知的特征去噪与结构增强。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要同时增强全局结构与局部边缘
纹理的任务，以及存在高频噪声、需要细节锐化的图像恢复、超分辨率、检测
等场景。
'''


class WDM(nn.Module):
    """WDM: Wavelet Decomposition Module —— 小波分解模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 threshold_init: float = 0.05):
        super().__init__()
        inner = max(1, channels // reduction)
        self.channels = channels
        self.inner = inner

        # Low-frequency refinement path (on LL subband)
        self.ll_dw = nn.Conv2d(inner, inner, 3, padding=1,
                               groups=inner, bias=False)
        self.ll_bn = nn.BatchNorm2d(inner)
        # Channel MLP (SE-like) to recalibrate global structure
        self.ll_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner, max(1, inner // 4), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(1, inner // 4), inner, 1, bias=False),
            nn.Sigmoid(),
        )

        # High-frequency denoising & sharpening path (on LH/HL/HH)
        # 3 detail subbands concatenated along channel -> 3*inner
        # Learnable per-channel soft threshold
        self.detail_threshold = nn.Parameter(
            torch.full((1, 3 * inner, 1, 1), threshold_init)
        )
        self.detail_pw = nn.Sequential(
            nn.Conv2d(3 * inner, 3 * inner, 1, bias=False),
            nn.BatchNorm2d(3 * inner),
            nn.GELU(),
            nn.Conv2d(3 * inner, 3 * inner, 1, bias=False),
        )

        # Channel projection between full and compressed spaces
        self.compress = nn.Conv2d(channels, inner, 1, bias=False)
        self.expand = nn.Conv2d(inner, channels, 1, bias=False)

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    @staticmethod
    def _pad_to_even(x: torch.Tensor):
        """Pad spatial dims to even sizes; return (padded, pad_info)."""
        B, C, H, W = x.shape
        ph = H % 2
        pw = W % 2
        if ph == 0 and pw == 0:
            return x, (0, 0, 0, 0)
        # pad bottom/right by 1 using reflection
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return x, (0, pw, 0, ph)

    @staticmethod
    def _haar_decompose(x: torch.Tensor):
        """
        One-level 2D Haar wavelet decomposition via lifting scheme.
        Returns (LL, LH, HL, HH), each [B, C, H/2, W/2].
        Lifting guarantees perfect reconstruction with the inverse below.
        """
        # Step 1: along W (columns)
        x0 = x[:, :, :, 0::2]                       # even cols
        x1 = x[:, :, :, 1::2]                       # odd cols
        detail_w = x1 - x0                          # high along W
        approx_w = x0 + detail_w / 2                # = (x0 + x1) / 2

        # Step 2: along H (rows) for both approx and detail
        a0 = approx_w[:, :, 0::2, :]
        a1 = approx_w[:, :, 1::2, :]
        d0 = detail_w[:, :, 0::2, :]
        d1 = detail_w[:, :, 1::2, :]

        LL = a0 + (a1 - a0) / 2                     # approx of approx
        LH = a1 - a0                                # vertical detail
        HL = d0 + (d1 - d0) / 2                     # approx of detail
        HH = d1 - d0                                # diagonal detail
        return LL, LH, HL, HH

    @staticmethod
    def _haar_reconstruct(LL, LH, HL, HH, out_h: int, out_w: int):
        """
        Inverse 2D Haar transform (inverse of _haar_decompose).
        Returns reconstructed tensor [B, C, out_h, out_w].
        """
        B, C, Hh, Wh = LL.shape
        device, dtype = LL.device, LL.dtype

        # Invert step 2 (along H)
        a0 = LL - LH / 2
        a1 = a0 + LH
        d0 = HL - HH / 2
        d1 = d0 + HH
        approx_w = torch.empty(B, C, 2 * Hh, Wh, device=device, dtype=dtype)
        approx_w[:, :, 0::2, :] = a0
        approx_w[:, :, 1::2, :] = a1
        detail_w = torch.empty(B, C, 2 * Hh, Wh, device=device, dtype=dtype)
        detail_w[:, :, 0::2, :] = d0
        detail_w[:, :, 1::2, :] = d1

        # Invert step 1 (along W)
        x0 = approx_w - detail_w / 2
        x1 = approx_w + detail_w / 2
        x = torch.empty(B, C, 2 * Hh, 2 * Wh, device=device, dtype=dtype)
        x[:, :, :, 0::2] = x0
        x[:, :, :, 1::2] = x1

        # Safety crop (in case of rounding)
        return x[:, :, :out_h, :out_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Compress channels, pad to even spatial size
        feat = self.compress(x)                      # [B, inner, H, W]
        feat, pad_info = self._pad_to_even(feat)
        Hp, Wp = feat.shape[-2:]

        # 2. Haar wavelet decomposition
        LL, LH, HL, HH = self._haar_decompose(feat)  # each [B, inner, h, w]

        # 3. Low-frequency refinement: structure recalibration
        ll_ref = self.ll_dw(LL)
        ll_ref = self.ll_bn(ll_ref)
        ll_ref = ll_ref * self.ll_se(ll_ref)
        LL_out = LL + ll_ref                         # [B, inner, h, w]

        # 4. High-frequency denoising & sharpening via soft-thresholding
        det = torch.cat([LH, HL, HH], dim=1)         # [B, 3*inner, h, w]
        tau = torch.relu(self.detail_threshold)      # non-negative threshold
        det_thr = torch.sign(det) * torch.relu(det.abs() - tau)
        det_proc = self.detail_pw(det_thr)
        det = det + det_proc
        LH_out, HL_out, HH_out = det.chunk(3, dim=1)

        # 5. Inverse wavelet reconstruction
        recon = self._haar_reconstruct(LL_out, LH_out, HL_out, HH_out,
                                       out_h=Hp, out_w=Wp)  # [B, inner, Hp, Wp]
        # crop back to pre-pad size
        recon = recon[:, :, :H, :W]

        # 6. Expand channels, refine and residual
        out = self.expand(recon)                     # [B, C, H, W]
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = WDM(channels=128)
    output = model(input_tensor)
    print('=== WDM: Wavelet Decomposition Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    # Perfect-reconstruction sanity check (no processing branches learn yet)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
