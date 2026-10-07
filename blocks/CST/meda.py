import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：CUST: Clustered Unit-level Similarity Transformer for Lightweight Image Super-Resolution (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2607.11088
# 代码来源：https://github.com/jwgdmkj/CUST
# 原始许可证：MIT (Copyright (c) 2024 Jeongsoo Kim)
# 模块出处：CUST_arch.py 的 MEDA 类（L505-562），含 Attention（L416-458）、
#   Low_to_high_MS_v2（L462-502）、patch_divide/patch_reverse（L304-414）与
#   LayerNorm/dwconv/ConvFFN 辅助类
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：仅做等价重写——(1) 删除 einops 依赖，rearrange 手写为 view/permute；
#   (2) 原 forward(x, ps) 的逐块 patch_size 改为构造参数 patch_size（MainBlock 中
#   各块使用的 [12,16,20,24,...] 之一，默认 16），step=patch_size-2 逻辑不变；
#   (3) 重叠 patch 划分/逆划分、低频-高频误差调制、门控位置编码注意力、ConvFFN
#   的数值逻辑逐行保持；(4) 原 Attention 的 pe 路径要求 qk_dim==dim（view 硬约束），
#   这里补 assert 显式化该约束（不改变数值）。

'''
模块名称：MEDA (Multi-scale Error Decomposition Attention) —— 多尺度误差分解注意力

一、模块简介
轻量超分网络既要捕捉高频细节又要维持低频结构，而单一尺度的注意力难以兼顾。
CUST 在聚簇单元级相似度注意力（CST）之外提出 MEDA：先用"低到高"的多尺度误差
分解显式提取特征在不同下采样层级间的残差（误差）成分，并用空间门控调制其注入
强度；再把特征裁成互相重叠的 patch（stride < patch_size），在每个 patch 内做
带门控与位置编码的注意力，最后用重叠累加-平均的逆变换还原。重叠划分让边界区域
的 token 也能获得充分上下文，误差分解让注意力聚焦在真正需要精炼的频率成分上。

核心创新点：
1. 多尺度误差分解：对特征做 2 倍/4 倍下采样-上采样，提取层间残差 err2/err4，
   经精炼器与空间门控后加权注入（可学习 scale 初始为 0）
2. 重叠 patch 划分：step = patch_size - 2，边界块贴边裁剪，逆变换对重叠区取平均
3. 门控位置编码注意力：深度卷积位置编码加到注意力输出上，再经 sigmoid 门控调制
4. ConvFFN 前馈：Linear-GELU-(+dwconv)-Linear，作用于 patch 逆变换后的 token 序列

二、结构设计
MEDA 张量形状流转如下（C=channels，ps=patch_size，step=ps-2）：
1. 多尺度误差分解（Low_to_high_MS_v2）：
   x_d2 = AvgPool(x, H/2, W/2) -> 上采样得 x_u2，err2 = x - x_u2；
   x_d4 = AvgPool(x_d2, H/4, W/4) -> 上采样得 x_u4，err4 = x_u2 - x_u4；
   refined = 1x1(GELU(dwconv3x3_dilated(err2+err4)))；
   gate = Sigmoid(1x1(GELU(1x1(cat(|err2|,|err4|)))))；
   out = x + scale * refined * gate，[B,C,H,W]
2. 重叠 patch 划分：[B,C,H,W] -> [B, n, C, ps, ps] -> token 化 [(B*n), ps^2, C]
3. 门控注意力（Attention）：qkv = Linear(C->3C) 拆为 q,k,v（qk_dim=C）；
   pe = dwconv3x3(q)；attn = softmax((q@k^T)*qk_dim^-0.5) @ v + pe；out *= Sigmoid(gate(x))；
   再经 Linear 投影，残差相加，还原 [B, n, C, ps, ps]
4. 重叠逆变换：累加回 [B,C,H,W]，重叠区域除以重叠次数
5. ConvFFN：token 化 [B,HW,C] -> Linear -> GELU -> (+5x5 dwconv) -> Linear，残差相加
   还原 [B,C,H,W]

三、论文写法参考
若在论文中引用该模块，可描述为：
"我们采用 CUST 提出的多尺度误差分解注意力（MEDA）[Kim et al., ECCV 2026]，
通过层级残差提取与空间门控显式调制高低频误差成分，并在重叠 patch 内执行带
门控位置编码的注意力，以增强轻量超分网络的细节重建能力。"
（原论文引用格式：Kim et al., "CUST: Clustered Unit-level Similarity Transformer
for Lightweight Image Super-Resolution", ECCV 2026.）

四、适用任务
轻量图像超分辨率、图像复原（去噪/去雨）、需要频率分解与边界上下文的密集预测任务。
'''
__all__ = ['MEDA']


##############################################################
## LN and ConvFFN (helpers, from CUST_arch.py)
##############################################################
class LayerNorm(nn.Module):
    """可选通道优先布局的 LayerNorm"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, channel_first: bool = True):
        super().__init__()
        self.channel_first = channel_first
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.channel_first:
            return self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class dwconv(nn.Module):
    """token 域上的深度卷积：[B,L,C] <-> [B,C,H,W] 来回变换后做 5x5 dwconv"""

    def __init__(self, hidden_features: int, kernel_size: int = 5):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2,
            groups=hidden_features,
        )

    def forward(self, x: torch.Tensor, x_size: t.Tuple[int, int]) -> torch.Tensor:
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.depthwise_conv(x)
        x = x.view(B, C, -1).transpose(1, 2)
        return x


class ConvFFN(nn.Module):
    """Linear-GELU-(+dwconv)-Linear 前馈，作用于 token 序列 [B,L,C]"""

    def __init__(self, in_features: int, hidden_features: t.Optional[int] = None,
                 out_features: t.Optional[int] = None, kernel_size: int = 5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, x_size: t.Tuple[int, int]) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


##############################################################
## Intra-Window Attn: overlapping patch divide / reverse
##############################################################
def patch_divide(x: torch.Tensor, step: int, ps: int):
    """把特征裁成互相重叠的 ps×ps patch。
    Args:
        x: [B, C, H, W]
        step: 划分步长（ps-2）
        ps: patch 边长
    Returns:
        crop_x: [B, n, C, ps, ps]，nh/nw 为纵向/横向裁剪数
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh

    crop_x = torch.stack(crop_x, dim=0)                # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x: torch.Tensor, x: torch.Tensor, step: int, ps: int) -> torch.Tensor:
    """把重叠 patch 累加还原为特征图，重叠区域除以重叠次数。
    Args:
        crop_x: [B, n, C, ps, ps]
        x: 原特征 [B, C, H, W]（提供形状与累加缓存）
        step: 划分步长
        ps: patch 边长
    Returns:
        output: [B, C, H, W]
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1

    # 重叠两次的条带区域除以 2（纵横重叠区各除一次，角部 4 次累加被除成 1）
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2

    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2

    return output


##############################################################
## Intra-Window Attn: gated attention with positional encoding
##############################################################
class Attention(nn.Module):
    """patch 内门控注意力（含深度卷积位置编码）"""

    def __init__(self, dim: int, heads: int, qk_dim: int):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.act = nn.GELU()
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        ws = int(N ** 0.5)

        qkv = self.qkv(x)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        z = self.act(self.gate(x))

        # 位置编码取自 q（要求 qk_dim == dim，原实现 view 硬约束）
        pe = self.pe(q.transpose(1, 2).view(B, C, ws, ws)).view(B, C, N).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v) + pe

        out = out * z
        return self.proj(out)


##############################################################
## Frequency Modulation
##############################################################
class Low_to_high_MS_v2(nn.Module):
    """低到高的多尺度误差分解与门控注入"""

    def __init__(self, dim: int):
        super().__init__()
        self.error_refiner = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.gate_gen = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. 层次化误差提取
        x_d2 = F.adaptive_avg_pool2d(x, (H // 2, W // 2))
        x_u2 = F.interpolate(x_d2, size=(H, W), mode='bilinear', align_corners=False)
        err2 = x - x_u2

        x_d4 = F.adaptive_avg_pool2d(x_d2, (H // 4, W // 4))
        x_u4 = F.interpolate(x_d4, size=(H, W), mode='bilinear', align_corners=False)
        err4 = x_u2 - x_u4

        # 2. Refiner & Gate
        refined_error = self.error_refiner(err2 + err4)
        error_energies = torch.cat([err2.abs(), err4.abs()], dim=1)
        spatial_gate = self.gate_gen(error_energies)

        return x + (self.scale * refined_error * spatial_gate)


##############################################################
## MEDA block
##############################################################
class MEDA(nn.Module):
    """MEDA: Multi-scale Error Decomposition Attention —— 多尺度误差分解注意力"""

    def __init__(self, channels: int, patch_size: int = 16,
                 qk_dim: t.Optional[int] = None, ffn_scale: float = 2.0,
                 num_heads: int = 1):
        super().__init__()
        if qk_dim is None:
            qk_dim = channels
        assert qk_dim == channels, '原实现 pe 路径要求 qk_dim == channels'
        self.patch_size = patch_size

        self.norm1 = LayerNorm(channels, channel_first=False)
        self.norm2 = LayerNorm(channels, channel_first=False)

        self.lth = Low_to_high_MS_v2(channels)
        self.attn = Attention(channels, num_heads, qk_dim)
        self.ffn = ConvFFN(channels, int(channels * ffn_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        step = ps - 2

        x = self.lth(x)

        # Patch Divide - LN - ATTN
        crop_x, nh, nw = patch_divide(x, step, ps)          # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = crop_x.reshape(b * n, ph * pw, c)          # (b n) (h w) c

        crop_x = self.attn(self.norm1(crop_x)) + crop_x
        crop_x = crop_x.view(b, n, ph, pw, c).permute(0, 1, 4, 2, 3).contiguous()  # b n c h w

        # Patch Reverse - LN - ConvFFN
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)                    # b (h w) c
        x = self.ffn(self.norm2(x), x_size=(h, w)) + x
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)          # b c h w
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = MEDA(channels=128)
    output = model(input_tensor)
    print('=== MEDA: Multi-scale Error Decomposition Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
