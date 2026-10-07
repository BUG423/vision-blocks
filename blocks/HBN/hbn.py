import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization (NeurIPS 2025)
# 论文链接：https://arxiv.org/abs/2503.04598
# 代码来源：https://github.com/BryceZhuo/HybridNorm
# 原始许可证：Apache-2.0（LICENSE 位于 MoE_model/ 与 dense_model/ 下）
# 模块出处：MoE_model/olmo/model.py 的 OLMoEBlock.forward（L806-887，norm 位置调度）、
#          OLMoBlock 的 q_norm/k_norm/v_norm 构造（L428-465）与 attention 内 QKV 归一化
#          （L656-662）、RMSLayerNorm（L224-251）；配置见 MoE_model/configs/exps/
#          MoE-1B-7B-HybridNorm.yaml（attn_norm=false, ffn_norm=true,
#          ffn_residual_after_norm=true, use_query/key/value_norm=true, layer_norm_type=rms）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：去除 OLMo 配置系统 / registry / 激活检查点 / MoE 专家结构，仅保留
#          HybridNorm 的两段归一化数学：(1) QKV-norm——注意力内部对 q/k/v 逐头
#          RMS 归一化（归一化轴= head_dim，fp32 方差 + rsqrt(var+eps)，可选 affine，
#          与 RMSLayerNorm.forward 逐行同构）；(2) FFN Post-Norm——
#          h = ff_norm(x); out = h + sublayer(h)（对应 ffn_norm=True 且
#          ffn_residual_after_norm=True 的残差取归一化张量语义，见 OLMoEBlock.forward
#          L875-887 的 og_x 赋值位置）。层归一化类型按论文配置取 rms（RMSNorm）。
#          统一 4D 接口下 forward 把该 FFN Post-Norm 残差形式包装到任意子层外围
#          （默认子层为空 → 纯 norm-and-passthrough，out = RMSNorm(x)）；
#          QKV-norm 以 apply_qkv_norm 暴露供注意力内部调用（q/k/v: [..., num_heads, head_dim]），
#          亦提供 qkv_norm_2d 直接对 [B,C,H,W] 按头组归一化。数学公式不变。

'''
模块名称：HBN (Hybrid Normalization) —— 混合归一化

一、模块简介
Transformer 训练中，Post-Norm（残差后归一化）梯度性质差、深层易发散但表达强；
Pre-Norm（残差前归一化）训练稳定、可堆很深，但残差流不受约束、上限略低。
HybridNorm 提出把两者优势拼进同一 block：注意力子层不做 pre/post 包裹，而是在
注意力内部对 Q/K/V 做逐头归一化（QKV-norm，稳定打分尺度）；FFN 子层则采用
Post-Norm 式的"归一化残差"结构——先归一化，再让子层与残差都从归一化张量出发
（out = Norm(x) + FFN(Norm(x))）。该混合方案兼顾训练稳定性与最终性能，且几乎
不增加参数。

核心创新点：
1. QKV-norm：注意力内对 q/k/v 逐头 RMS 归一化（归一化轴= head_dim），稳定注意力
   打分尺度，替代对整层的 pre/post 包裹；
2. FFN Post-Norm：h = Norm(x)；out = h + FFN(h)——残差取归一化后的张量
   （ffn_residual_after_norm=True），使残差流本身受归一化约束；
3. 零结构侵入：不改注意力/FFN 内部结构，只改归一化放置，可即插即用；
4. 训练稳定 + 性能：1B 稠密 / 1B-7B MoE 上同时优于 Pre-Norm 与 Post-Norm。

二、结构设计
以输入 [B, C, H, W]、N = H*W、head_dim = C / num_heads 计：
1. QKV-norm（apply_qkv_norm）：q/k/v: [..., num_heads, head_dim] → 每头
   x_fp32 * rsqrt(mean(x_fp32^2, -1) + eps) * weight（RMS 归一化，与原
   RMSLayerNorm 一致：fp32 计算后转回原 dtype，affine 可选、默认有 weight 无 bias）；
2. FFN Post-Norm 包装（forward）：
   h = ff_norm(x)（C 维 RMSNorm）→ out = h + sublayer(h)；
   默认 sublayer=None（空子层）→ out = h = RMSNorm(x)，即纯 norm-and-passthrough；
   传入任意 nn.Module（FFN/卷积/注意力等）即得论文的 FFN Post-Norm 包装；
3. qkv_norm_2d(x)：把 [B,C,H,W] 通道按 num_heads 分组，每组以 head_dim 为归一化
   轴做同一 RMS 归一化（QKV-norm 的 2D 直接适配）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 HybridNorm（NeurIPS 2025）的混合归一化策略 HBN：注意力内部对 Q/K/V
做逐头 RMS 归一化（QKV-norm）以稳定打分尺度，FFN 子层采用 Post-Norm 式归一化残差
（h=Norm(x)，out=h+FFN(h)），在不改动子层结构的前提下同时获得 Pre-Norm 的训练
稳定性与 Post-Norm 的表达能力。"
（原论文 BibTeX：Zhuo et al., arXiv:2503.04598）

四、适用任务
适用于深层 Transformer 训练（语言模型、ViT、检测/分割主干）——尤其大深度、
大规模预训练场景；可作为任意注意力/FFN 子层外围的即插即用归一化替换
（替代单一 LayerNorm），或独立用作 2D 特征的 RMS 归一化模块。
'''


class _RMSNorm(nn.Module):
    """RMS 归一化（原 RMSLayerNorm，layer_norm_type=rms 路径）：fp32 方差 + rsqrt。"""

    def __init__(self, size: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与原 forward 同构：autocast 关闭下强制 fp32 计算再转回原 dtype
        og_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(og_dtype)
        if self.weight is not None:
            return self.weight * x
        return x


class HBN(nn.Module):
    """HBN: Hybrid Normalization —— 混合归一化（QKV-norm + FFN Post-Norm）"""

    def __init__(self, channels: int, eps: float = 1e-5, num_heads: int = 1,
                 elementwise_affine: bool = True,
                 sublayer: t.Optional[nn.Module] = None):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels {channels} must be divisible by num_heads {num_heads}'
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.eps = eps

        # QKV-norm：三个独立仿射参数、同一 RMS 公式（原 q_norm/k_norm/v_norm）
        self.q_norm = _RMSNorm(self.head_dim, eps, elementwise_affine)
        self.k_norm = _RMSNorm(self.head_dim, eps, elementwise_affine)
        self.v_norm = _RMSNorm(self.head_dim, eps, elementwise_affine)
        # FFN Post-Norm 用归一化（原 ff_norm，C 维）
        self.ff_norm = _RMSNorm(channels, eps, elementwise_affine)

        # 被包装的子层（FFN/注意力等）；None = 空子层（norm-and-passthrough）
        self.sublayer = sublayer

    def apply_qkv_norm(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) \
            -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QKV-norm：对 q/k/v 逐头 RMS 归一化（归一化轴= head_dim）。

        q/k/v: [..., num_heads, head_dim]，与原 attention 内
        q = self.q_norm(q).to(dtype) 等调用完全同构。
        """
        return self.q_norm(q), self.k_norm(k), self.v_norm(v)

    def qkv_norm_2d(self, x: torch.Tensor) -> torch.Tensor:
        """QKV-norm 的 2D 直接适配：[B, C, H, W] → 同形，按头组做 RMS 归一化。"""
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        t = x.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)  # [B, nh, N, hd]
        t = self.v_norm(t)  # 同源特征取 V 路径归一化（q/k 同公式，见 apply_qkv_norm）
        return t.transpose(-1, -2).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        HybridNorm FFN Post-Norm 包装（ffn_norm=True, ffn_residual_after_norm=True）：
            h = ff_norm(x); out = h + sublayer(h)（sublayer 为空时 out = h）。
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        # RMSNorm 在特征维（C）上计算：转成 channel-last 与原 RMSLayerNorm 同构
        h = self.ff_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.sublayer is None:
            return h
        return h + self.sublayer(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = HBN(channels=128)
    output = model(input_tensor)
    print('=== HBN: Hybrid Normalization ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
