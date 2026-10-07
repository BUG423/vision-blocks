import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Beyond Sequential Distance: Inter-Modal Distance Invariant Position Encoding (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2603.10863
# 代码来源：https://github.com/lchen1019/DIPE
# 原始许可证：MIT
# 模块出处：qwenvl/models/modeling_llm_eagar.py 的 apply_multimodal_rotary_pos_emb（L164-174）、
#          decoupled_eager_attention_forward（L36-107）、Qwen25_SigLIPRotaryEmbedding（L110-145）；
#          锚定位置构造见 qwenvl/data/rope2d.py 的 static_position_ids（L37-120）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：去除 transformers 依赖（ROPE_INIT_FUNCTIONS / Qwen2Config / Cache 等），
#          以纯 PyTorch 复刻默认 RoPE 的 inv_freq 与 cos/sin 构造（attention_scaling=1.0），
#          数值公式与 Qwen25_SigLIPRotaryEmbedding.forward 完全一致；
#          apply_multimodal_rotary_pos_emb 的相位重排原样保留（注意 mrope_section * 2
#          是 Python 列表重复 [16,24,24]→[16,24,24,16,24,24]，配合 i%3 循环取 (t,h,w)
#          三轴相位，使 rotate_half 的前后两半频率一致——此处不得改写为元素乘 2）；
#          锚定静态位置（inter-modal 距离不变的核心）原样保留：每段 token 的位置全部
#          填为段起始索引（full_like），使跨模态相对距离只由段锚点决定；
#          decoupled_eager_attention_forward 的双相位选择（intra 用动态相位、inter 用
#          静态相位：torch.where(mask_intra, attn_dyn, attn_static)）抽为
#          decoupled_scores 方法，分块循环与 HF Cache/packing 装配删除；
#          统一 4D 接口：forward 把 [B,C,H,W] 视作单幅 2D 特征图（t 轴恒为 0 的
#          h/w 网格位置），经内部 2D RoPE 施加同一相位重排后还原形状。

'''
模块名称：DIP (Inter-modal Distance Invariant Position Encoding) —— 跨模态距离不变位置编码

一、模块简介
多模态大模型（如 Qwen2.5-VL 类）用 M-RoPE 给文本 token 一维位置、给视觉 token
(t,h,w) 三维网格位置。此时文本与视觉之间的"相对距离"依赖双方在序列中的绝对位置，
图像插入位置/长度一变，跨模态注意力的距离先验就漂移，长上下文外推与多图插值不稳。
DIPE 提出把位置相位解耦为两套：动态相位（真实 2D/3D 网格位置，刻画模态内结构）
与锚定静态相位（每段 token 共享段起始锚点，刻画跨模态关系）。注意力打分时，
模态内（intra）配对用动态相位、跨模态（inter）配对用静态相位，使跨模态距离只由
段锚点决定，对段内排布与长度保持不变（inter-modal distance invariant）。

核心创新点：
1. 锚定静态位置（anchored static position ids）：每段（文本/图像）内所有 token 的
   位置统一填为该段起始索引，跨段相对距离 = 锚点之差，与段内细节无关；
2. 解耦双相位注意力（decoupled eager attention）：q 同时施加动态/静态两套 RoPE，
   按 intra/inter 掩码逐元素选择打分，key 侧统一用动态相位；
3. mrope 相位重排：head_dim 按 mrope_section*2 列表重复切成 6 段，第 i 段取
   (t,h,w)[i%3] 轴相位，前后两半频率一致以配合 rotate_half；
4. 即插即用：可作为 2D 特征图的位置编码插件，或暴露 apply_rope / decoupled_scores
   供自定义注意力调用。

二、结构设计
以输入 [B, C, H, W]、N = H*W、head_dim = C / num_heads 计：
1. 位置构造 build_position_ids：动态 2D 网格 → [3, B, N]（t 轴全 0、h/w 网格展平，
   对应原 rope2d 中 llm_grid_t=1 的图像 token）；build_static_position_ids：锚定
   填充 [3, B, N] 全为 anchor（原 llm_pos_ids_list_static 的 full_like 逻辑）；
2. 2D RoPE 嵌入（复刻 Qwen25_SigLIPRotaryEmbedding）：
   inv_freq = 1 / base^(2i/head_dim)，freqs = inv_freq @ pos → [3, B, N, head_dim/2]，
   emb = cat(freqs, freqs) → cos, sin ∈ [3, B, N, head_dim]（attention_scaling=1）；
3. 相位重排 apply_rope（原 apply_multimodal_rotary_pos_emb）：
   mrope_section 列表 *2 → 6 段 split，第 i 段取 cos/sin 的第 (i%3) 个网格分量，
   cat 后 unsqueeze 得 [B, 1, N, head_dim]，x_embed = x*cos + rotate_half(x)*sin；
4. forward：x → [B, num_heads, N, head_dim] → apply_rope（内部 2D 动态相位；
   mode='static' 时改用锚定相位）→ 还原 [B, C, H, W]，形状保持；
5. decoupled_scores：q/k: [B, H, L, D] 与两套 (cos, sin)，返回
   where(intra, q_dyn@k_dyn, q_static@k_dyn)*scale，并按 inter|intra 掩码置 -1e7
   （原 decoupled_eager_attention_forward 的选择与掩码逻辑）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 DIPE（ECCV 2026）提出的跨模态距离不变位置编码 DIP：对模态内注意力
使用真实网格位置的动态 RoPE 相位，对跨模态注意力使用段锚定的静态相位（每段 token
共享段起始位置），两类配对按 intra/inter 掩码在打分阶段解耦选择，使跨模态相对
距离只由段锚点决定，对段内长度与排布保持不变，显著改善长上下文多模态外推稳定性。"
（原论文 BibTeX 见 arXiv:2603.10863）

四、适用任务
适用于多模态理解（图文/多图/长上下文 VQA）、多模态检索、以及任何需要把 M-RoPE
风格位置编码注入 2D 视觉特征的即插即用场景。对图像插入位置敏感、需要跨模态
距离稳健性的任务（长文档、交错图文、多图插值）收益最大。
'''


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input（原 modeling_utils.rotate_half）。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class _DIPRotaryEmbedding(nn.Module):
    """2D 适配的 Qwen25_SigLIPRotaryEmbedding：由 [3, B, L] 位置生成 cos/sin。"""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f'head_dim must be even, got {head_dim}'
        # 与 transformers ROPE_INIT_FUNCTIONS['default'] 一致：arange(0, dim, 2)/dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.attention_scaling = 1.0  # 默认 rope_type 的 attention_scaling

    def forward(self, position_ids: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [3, B, L]（t/h/w 三轴位置）
        返回 cos, sin: [3, B, L, head_dim]
        """
        # 与原 forward 逐行同构（强制 float32 计算后转回输入 dtype）
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()          # [3, B, 1, L]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # [3, B, L, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)                              # [3, B, L, head_dim]
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin


class DIP(nn.Module):
    """DIP: Inter-modal Distance Invariant Position Encoding —— 跨模态距离不变位置编码"""

    def __init__(self, channels: int, mrope_section: t.Optional[t.List[int]] = None,
                 num_heads: int = 1, base: float = 10000.0):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels {channels} must be divisible by num_heads {num_heads}'
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # 论文/Qwen2.5-VL 默认 mrope_section=[16, 24, 24]（和为 head_dim/2=64，head_dim=128）
        if mrope_section is None:
            assert self.head_dim == 128, \
                'default mrope_section=[16, 24, 24] requires head_dim=128; ' \
                'pass mrope_section explicitly for other head_dim'
            mrope_section = [16, 24, 24]
        assert sum(mrope_section) == self.head_dim // 2, \
            f'sum(mrope_section)={sum(mrope_section)} must equal head_dim//2={self.head_dim // 2}'
        self.mrope_section = list(mrope_section)

        self.rope_emb = _DIPRotaryEmbedding(self.head_dim, base=base)

    # ---- 位置构造（对应 qwenvl/data/rope2d.py 的 position_ids / static_position_ids）----

    def build_position_ids(self, batch: int, h: int, w: int,
                           device: t.Optional[torch.device] = None,
                           dtype: torch.dtype = torch.long) -> torch.Tensor:
        """动态 2D 网格位置（t 轴恒 0）→ [3, B, H*W]。

        对应原 get_rope_index_3 中单幅图像（llm_grid_t=1）的 (t_index, h_index, w_index)。
        """
        t_index = torch.zeros(h * w, device=device, dtype=dtype)
        h_index = torch.arange(h, device=device, dtype=dtype).view(-1, 1).expand(h, w).reshape(-1)
        w_index = torch.arange(w, device=device, dtype=dtype).view(1, -1).expand(h, w).reshape(-1)
        pos = torch.stack([t_index, h_index, w_index], dim=0)                 # [3, H*W]
        return pos.unsqueeze(1).expand(3, batch, h * w).contiguous()

    def build_static_position_ids(self, batch: int, h: int, w: int, anchor: int = 0,
                                  device: t.Optional[torch.device] = None,
                                  dtype: torch.dtype = torch.long) -> torch.Tensor:
        """锚定静态位置 → [3, B, H*W]，全填 anchor（原 full_like 锚定逻辑）。

        inter-modal 距离不变的核心：段内所有 token 共享段起始锚点，跨段相对距离
        只由锚点差决定。单幅特征图默认 anchor=0。
        """
        pos = self.build_position_ids(batch, h, w, device=device, dtype=dtype)
        return torch.full_like(pos, anchor)

    def get_cos_sin(self, position_ids: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """由 [3, B, L] 位置生成 cos/sin（[3, B, L, head_dim]）。"""
        return self.rope_emb(position_ids)

    # ---- 相位重排（原 apply_multimodal_rotary_pos_emb，数值逻辑逐行保留）----

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """解耦 RoPE 相位重排 + 旋转。

        x:   [B, num_heads, L, head_dim]
        cos: [3, B, L, head_dim], sin 同形（来自 get_cos_sin）
        返回: 与 x 同形。
        注意：mrope_section * 2 是列表重复（[16,24,24]→[16,24,24,16,24,24]），
        split 后第 i 段取第 (i%3) 个网格轴相位，保证 rotate_half 前后两半频率一致。
        """
        mrope_section = self.mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed

    # ---- 解耦双相位打分（原 decoupled_eager_attention_forward 的相位选择）----

    def decoupled_scores(self, static_query: torch.Tensor, dynamic_query: torch.Tensor,
                         key: torch.Tensor,
                         inter_mask: torch.Tensor, intra_mask: torch.Tensor,
                         scaling: t.Optional[float] = None) -> torch.Tensor:
        """inter-modal 距离不变的解耦打分选择，返回未 softmax 的注意力打分。

        static_query / dynamic_query / key: [B, num_heads, L, head_dim]
        （调用方先用 apply_rope 施加锚定静态相位 / 动态相位；key 用动态相位。）
        inter_mask / intra_mask: [B, 1, L, L] 或可广播布尔张量（True=允许）。
        返回: [B, num_heads, L, L]，inter 配对取静态打分、intra 配对取动态打分，
        两者皆不允许的位置置 -1e7（原 min_val 逻辑）。
        """
        if scaling is None:
            scaling = self.head_dim ** -0.5
        attn_static = torch.matmul(static_query, key.transpose(-1, -2)) * scaling
        attn_dynamic = torch.matmul(dynamic_query, key.transpose(-1, -2)) * scaling

        attn_weights = torch.where(intra_mask, attn_dynamic, attn_static)
        mask_keep = intra_mask | inter_mask
        min_val = torch.tensor(-1e7, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(mask_keep, attn_weights, min_val)
        return attn_weights

    def forward(self, x: torch.Tensor, mode: str = 'dynamic') -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        mode: 'dynamic' 用 2D 网格动态相位（模态内距离结构）；
              'static'  用锚定静态相位（anchor=0，inter-modal 距离不变）。
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        assert mode in ('dynamic', 'static'), f'unknown mode {mode!r}'
        N = H * W

        # [B, C, H, W] -> [B, num_heads, N, head_dim]
        tokens = x.reshape(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

        if mode == 'dynamic':
            pos = self.build_position_ids(B, H, W, device=x.device)
        else:
            pos = self.build_static_position_ids(B, H, W, anchor=0, device=x.device)
        cos, sin = self.get_cos_sin(pos)

        out = self.apply_rope(tokens, cos, sin)                             # [B, num_heads, N, head_dim]

        # 还原 [B, C, H, W]
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = DIP(channels=128)
    output = model(input_tensor)
    print('=== DIP: Inter-modal Distance Invariant Position Encoding ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
