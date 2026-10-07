import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers (CVPR 2026)
# 论文链接：https://arxiv.org/abs/2603.09582
# 代码来源：https://github.com/EdwardChasel/BinaryAttention
# 原始许可证：Apache-2.0
# 模块出处：models.py 的 Attention 类（量化算子来自 utils.py 的 binarize / symquantize / round_ste）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删除 timm 依赖、模型注册 registry、ImageNet cfg、相对位置偏置分支（attn_bias，
#          非注意力核心且依赖固定 input_size）；token 接口 [B,N,C] 包装为统一 4D 接口
#          [B,C,H,W]（内部 flatten(2).transpose / transpose.reshape 还原，不改数值）；
#          1-bit QK 量化（mean-abs scale × STE sign）与 PV 量化（attn 无符号均匀 + V 对称均匀）
#          公式与 STE 反传掩码原样保留；原硬编码的 8-bit 改为参数 pv_bits=8、qk_bits=1（论文默认）。

'''
模块名称：BQA (Binary QK-Attention) —— 一比特 QK 注意力

一、模块简介
标准自注意力的 QK^T 矩阵乘是 Vision/Diffusion Transformer 的主要算力与
访存瓶颈：其计算量随 token 数平方增长，且需读写完整的浮点 Q、K 张量。
BinaryAttention 观察到注意力图 softmax 后是高度冗余的概率分布，对数值
精度不敏感，因此把 Q、K 压缩到 1-bit（符号 + 每头缩放因子），用 XNOR 风格
的点积近似注意力打分，同时对注意力概率与 V 做 8-bit 均匀量化。

核心创新点：
1. 1-bit QK 量化：每头用平均绝对值作为缩放，符号函数二值化，配合
   Straight-Through Estimator（STE）保持可微训练；
2. PV 联动量化：softmax 后的注意力概率做无符号均匀量化，V 做对称均匀量化，
   默认 8-bit；
3. 精度几乎无损：1-bit QK + 8-bit PV 在分类与扩散 Transformer 上接近全精度；
4. 即插即用：可直接替换 ViT/DiT 中的标准 Multi-Head Attention。

二、结构设计
BQA 由以下子结构组成（形状以输入 [B, C, H, W]、N = H*W 计）：
1. QKV 线性投影：nn.Linear(C, 3C)，随后 reshape/permute 为
   q, k, v: [B, num_heads, N, head_dim]，head_dim = C / num_heads；
2. 1-bit QK 量化（attn_quant=True 时）：
   - scale s = mean_{N, head_dim}(|x|) ∈ [B, H, 1, 1]；
   - sign(x) ∈ {-1, +1}，STE 反传（|x|>1 处梯度置零）；
   - q̂ = s ⊙ sign(q)，k̂ 同理；
3. 注意力打分：attn = (q̂ @ k̂^T) * head_dim^{-0.5}，softmax(dim=-1)；
4. PV 量化（pv_quant=True 时）：
   - attn：无符号均匀量化，步长 1/(2^{pv_bits}-1)，clamp 到 [0, qmax]；
   - v：对称均匀量化，步长由 |v| 在 token 维的最大值决定，STE 反传
     （|v|≥2 处梯度置零，clip_val = [-2, 2]）；
5. 输出投影：out = (attn @ v) → [B, N, C] → Linear(C, C) → reshape 回 [B, C, H, W]。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 BinaryAttention（CVPR 2026）提出的一比特 QK 注意力 BQA 替换标准
自注意力：对 Query/Key 做 1-bit 符号量化（STE 直通估计）并保持每头平均绝对值
缩放，对注意力概率与 Value 做 8-bit 均匀量化，在几乎不损失精度的前提下显著
降低注意力的计算与访存开销。"

四、适用任务
适用于图像分类、目标检测、语义分割、扩散生成等以 Vision Transformer /
Diffusion Transformer 为主干的任务。尤其适合部署侧对低比特推理敏感的场景
（边缘端 ViT、DiT 加速）。作为即插即用注意力模块，可直接替换主干中的 MHA。
'''


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradients（原 utils.round_ste）。"""
    zhat = z.round()
    return z + (zhat - z).detach()


def _ste_sign(x: torch.Tensor) -> torch.Tensor:
    """符号二值化 + STE：前向 ∈ {-1, +1}；反传仅在 |x| ≤ 1 处透传梯度。

    与原 utils.STESign 完全一致（原 backward 将 x>1 / x<-1 处梯度置零）。
    """
    sign_x = x.sign() + (x == 0).type_as(x)          # x == 0 映射为 +1
    mask = ((x >= -1) & (x <= 1)).type_as(x)
    return x * mask + (sign_x - x * mask).detach()


def _sym_quantize_ste(x: torch.Tensor, bits: int,
                      clip_val: t.Tuple[float, float] = (-2.0, 2.0)) -> torch.Tensor:
    """对称均匀量化 + STE：反传仅在 clip_val 开区间内透传梯度。

    与原 utils.SymQuantizer（layerwise=False）完全一致：scale 由 |x| 在 dim=-2
    （token 维）上的最大值给出并 detach，量化位宽为 bits，clip_val 仅影响梯度。
    """
    max_input = torch.max(torch.abs(x), dim=-2, keepdim=True)[0].expand_as(x).detach()
    s = (2 ** (bits - 1) - 1) / (max_input + 1e-6)
    q = torch.round(x * s).div(s + 1e-6)
    mask = ((x > clip_val[0]) & (x < clip_val[1])).type_as(x)
    return x * mask + (q - x * mask).detach()


class BQA(nn.Module):
    """BQA: Binary QK-Attention —— 一比特 QK 注意力"""

    def __init__(self, channels: int, num_heads: int = 8,
                 attn_quant: bool = True, pv_quant: bool = True,
                 qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.,
                 qk_bits: int = 1, pv_bits: int = 8):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels {channels} must be divisible by num_heads {num_heads}'
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.attn_quant = attn_quant
        self.pv_quant = pv_quant
        self.qk_bits = qk_bits
        self.pv_bits = pv_bits

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    # ---- quantizers（与原 Attention._quantize / _quantize_p / _quantize_v 对应）----

    def _quantize_qk(self, x: torch.Tensor) -> torch.Tensor:
        """Q/K 量化。qk_bits=1 为论文的 1-bit 路径：mean-abs scale × STE sign。"""
        if self.qk_bits == 1:
            s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
            return s * _ste_sign(x)
        # 非论文默认：多比特对称均匀量化（扩展，便于消融）
        return _sym_quantize_ste(x, self.qk_bits)

    def _quantize_p(self, x: torch.Tensor) -> torch.Tensor:
        """注意力概率无符号均匀量化（原 _quantize_p，qmax = 2^{pv_bits}-1）。"""
        qmax = float(2 ** self.pv_bits - 1)
        s = 1.0 / qmax
        q = round_ste(x / s).clamp(0, qmax)
        return s * q

    def _quantize_v(self, x: torch.Tensor) -> torch.Tensor:
        """V 对称均匀量化（原 _quantize_v + SymQuantizer，clip_val=[-2, 2]）。"""
        return _sym_quantize_ste(x, self.pv_bits, clip_val=(-2.0, 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        N = H * W

        # [B, C, H, W] -> [B, N, C]（token 化，统一接口包装）
        tokens = x.flatten(2).transpose(1, 2)                                    # [B, N, C]

        qkv = self.qkv(tokens)                                                   # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                         # [B, H, N, D]

        if self.attn_quant:
            q = self._quantize_qk(q)
            k = self._quantize_qk(k)

            attn = (q @ k.transpose(-2, -1)) * self.scale                        # [B, H, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if self.pv_quant:
                attn = self._quantize_p(attn)
                v = self._quantize_v(v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)                        # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)

        # [B, N, C] -> [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 全精度注意力的空间代价为 O((H*W)^2)，smoke 用 32x32 控制内存
    input_tensor = torch.randn(1, 64, 32, 32)
    model = BQA(channels=64)
    output = model(input_tensor)
    print('=== BQA: Binary QK-Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
