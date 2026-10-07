import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：CUST: Clustered Unit-level Similarity Transformer for Lightweight Image Super-Resolution (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2607.11088
# 代码来源：https://github.com/jwgdmkj/CUST
# 原始许可证：MIT (Copyright (c) 2024 Jeongsoo Kim)
# 模块出处：CUST_arch.py 的 CUSTAttention 类（L86-263）与 CUSTBlock 类（L266-299），含 LayerNorm/dwconv/ConvFFN 辅助类
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：仅做等价重写——(1) 删除 einops/torchvision/basicsr 依赖，rearrange 手写为 view/permute；
#   (2) 原 window_size 重命名为 patch_size 以统一接口（数值默认 8 不变），group_size 默认 9 与原类一致；
#   (3) 保留 num_heads 形参以满足统一接口，但不参与计算：原注意力为全通道单头（scale=dim**-0.5），
#   改多头会改变数值逻辑，故忽略之；(4) 簇分配(argmax)-按簇排序-分块滑窗 KV-同簇掩码-门控-逆排序
#   的 CUSTAttention 数值逻辑逐行保持；(5) CUSTBlock 的深度卷积位置编码 + Pre-Norm 残差注意力
#   + ConvFFN 结构原样保留；(6) 零填充张量补上 dtype 以兼容半精度（数值不变）。

'''
模块名称：CST (Clustered Unit-level Similarity Transformer attention) —— 聚簇单元级相似度 Transformer 注意力

一、模块简介
轻量图像超分辨率网络受限于参数预算，难以负担全局自注意力的二次复杂度，
而窗口注意力又把感受野锁死在窗口内部，跨窗信息只能靠堆叠层来间接传递。
CUST 提出"聚簇单元级相似度"注意力：把特征图切成窗口（window），再把相邻的
group_size×group_size 个窗口组成一个组（group）；组内每个最小单元（窗口内的
patch token）根据其与各窗口代表向量的相似度，被硬分配到最相似的窗口簇中。
随后 token 按簇排序并分块，每个查询块只与"前半块+本块+后半块"的滑窗键值做
注意力，且掩码保证只与同簇 token 交互——从而以接近线性的代价获得跨窗的、
按内容相似度组织的长程信息流。输出再经可学习门控调制，兼顾精度与效率。

核心创新点：
1. 单元级相似度聚簇：以窗口均值代表为参照，把每个 patch token 硬分配（argmax）到最相似窗口簇
2. 排序-分块滑窗 KV：按簇排序后以窗口大小为块，KV 取前后半块+本块的重叠滑窗，扩大簇内感受野
3. 同簇掩码注意力：仅同簇（同 assign_id）token 可交互，跨簇被置 -1e4 后 softmax
4. 门控输出：sigmoid 门控作用于排序后的 token，再投影并逆排序还原空间布局

二、结构设计
CST 由 CUSTBlock 外壳与 CUSTAttention 核心组成，张量形状流转如下：
1. 深度卷积位置编码（pe, 3x3 dwconv）：[B,C,H,W] -> [B,C,H,W]，x = x + pe(x)
2. Pre-Norm + CUSTAttention（残差）：
   a. 反射填充到 (window*group) 的整数倍：[B,C,H,W] -> [B,C,Hp,Wp]
   b. 窗口-组划分：[B,C,Hp,Wp] -> [B, gh*gw, gs^2, ws^2, C]
      （gh*gw 个组，每组 gs^2 个窗口，每窗 ws^2 个 token）
   c. 相似度：窗口代表 = 组内 token 均值（detach），
      sim = einsum('b g w p c, b g k c -> b g w p k')，得 [B, ng, gs^2, ws^2, gs^2]
   d. cana：按 sim.argmax 簇分配 -> argsort 排序 -> 每 ws^2 个 token 一块
      -> 滑窗 KV（前后各半块填充）-> q@k^T * dim^-0.5 -> 同簇掩码(-1e4) -> softmax
      -> @v -> sigmoid 门控 -> 输出投影 -> 逆排序，回到 [B, gh*gw, gs^2, ws^2, C]
   e. 逆窗口-组划分并裁掉填充：[B,C,Hp,Wp] -> [B,C,H,W]
3. Pre-Norm + ConvFFN（残差）：token 化 [B,HW,C] -> Linear -> GELU ->
   (+5x5 深度卷积) -> Linear -> 还原 [B,C,H,W]，残差相加

三、论文写法参考
若在论文中引用该模块，可描述为：
"本文采用 CUST 提出的聚簇单元级相似度注意力（CST）[Kim et al., ECCV 2026]，
在窗口组内依据单元级相似度对 token 进行硬聚簇，并在同簇约束下执行分块滑窗
注意力，以较低计算开销实现跨窗口的内容自适应信息交互。"
（原论文引用格式：Kim et al., "CUST: Clustered Unit-level Similarity Transformer
for Lightweight Image Super-Resolution", ECCV 2026.）

四、适用任务
轻量图像超分辨率、图像复原（去噪/去雨/去雾）等低层视觉任务；也可作为通用
视觉主干中的即插即用注意力块，尤其适合需要跨窗长程交互但算力受限的场景。
'''
__all__ = ['CST']


##############################################################
## LN and ConvFFN (helpers, from CUST_arch.py)
##############################################################
class LayerNorm(nn.Module):
    """通道优先布局的 LayerNorm（内部转置后调用 nn.LayerNorm）"""

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
        # x: [B, L, C]
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
## Inter-Window Attn: clustered unit-level similarity attention
##############################################################
class CUSTAttention(nn.Module):
    """聚簇单元级相似度注意力核心（原 CUST_arch.CUSTAttention，数学逻辑不变）"""

    def __init__(self, dim: int, window_size: int = 8, group_size: int = 9):
        super().__init__()
        self.window_size = window_size
        self.group_size = group_size
        self.scale = dim ** -0.5

        hidden_dim = dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # gate before proj
        self.gate_proj = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()

    ############ window group partition and reverse ############
    def window_group_partition(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ws, gs = self.window_size, self.group_size

        # Pad 到 (ws*gs) 的整数倍
        target_unit = ws * gs
        pad_h = (target_unit - H % target_unit) % target_unit
        pad_w = (target_unit - W % target_unit) % target_unit
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        H_pad, W_pad = x.shape[2], x.shape[3]
        gh, gw = H_pad // target_unit, W_pad // target_unit

        # [B C (gh gs ws) (gw gs ws)] -> [B gh gw gs gs ws ws C] -> [B, gh*gw, gs^2, ws^2, C]
        x = x.view(B, C, gh, gs, ws, gw, gs, ws)
        x = x.permute(0, 2, 5, 3, 6, 4, 7, 1)
        x = x.contiguous().view(B, gh * gw, gs * gs, ws * ws, C)
        return x, pad_h, pad_w

    def window_group_reverse(self, x: torch.Tensor, original_shape: t.Tuple[int, ...],
                             padded_size: t.Tuple[int, int]) -> torch.Tensor:
        b, ng, gs_sq, ws_sq, chan = x.shape
        ws, gs = self.window_size, self.group_size
        _, _, H, W = original_shape

        H_pad, W_pad = H + padded_size[0], W + padded_size[1]
        gh, gw = H_pad // (ws * gs), W_pad // (ws * gs)

        x = x.view(b, gh, gw, gs, gs, ws, ws, chan)
        x = x.permute(0, 7, 1, 3, 5, 2, 4, 6)
        x = x.contiguous().view(b, chan, H_pad, W_pad)
        if padded_size[0] > 0 or padded_size[1] > 0:
            x = x[:, :, :H, :W]
        return x

    ############ clustered unit-level attention ############
    def cana(self, x_grouped: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """把 token 按相似度簇分配排序后做同簇掩码滑窗注意力。
        x_grouped: [B, ng, gs^2, ws^2, C]；sim: [B, ng, gs^2, ws^2, gs^2]
        """
        # 注意：此处 gs/ws 为窗口数/每窗 token 数（即 group_size^2 / window_size^2），沿用原实现命名
        B, ng, gs, ws, chan = x_grouped.shape
        device = x_grouped.device

        x_grouped = x_grouped.view(B * ng, gs * ws, chan)

        assign_id = sim.argmax(dim=-1).view(B * ng, gs * ws)
        sorting_indices = torch.argsort(assign_id, dim=1)

        # x 与 id 按排序索引重排
        gather_idx = sorting_indices.unsqueeze(-1).expand(-1, -1, chan)
        x_sorted = torch.gather(x_grouped, 1, gather_idx)          # [B*ng, gs*ws, C]
        id_sorted = torch.gather(assign_id, 1, sorting_indices)    # [B*ng, gs*ws]

        cs = self.window_size ** 2      # chunk_size
        nc = (gs * ws) // cs            # num_chunk

        # A) Query：排序后按 cs 个 token 一块
        q_chunks = x_sorted.view(B * ng, nc, cs, chan)
        q_ids = id_sorted.view(B * ng, nc, cs)

        # B) KV：前后各补半块，unfold 得 (前半块+本块+后半块) 的滑窗，stride=cs
        pad_x = torch.zeros(B * ng, cs // 2, chan, device=device, dtype=x_sorted.dtype)
        pad_x = torch.cat([pad_x, x_sorted, pad_x], dim=1)
        pad_id = torch.full((B * ng, cs // 2), -1, device=device, dtype=id_sorted.dtype)
        pad_id = torch.cat([pad_id, id_sorted, pad_id], dim=1)

        kv_chunks = pad_x.unfold(1, cs * 2, cs).permute(0, 1, 3, 2)
        kv_ids = pad_id.unfold(1, cs * 2, cs)

        # C) Attn with same-cluster masking
        q = self.to_q(q_chunks)      # [BG, Chunks, cs, C]
        k = self.to_k(kv_chunks)     # [BG, Chunks, 2*cs, C]
        v = self.to_v(kv_chunks)     # [BG, Chunks, 2*cs, C]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 仅同簇（id 相等）可交互，其余置 -1e4
        mask = (q_ids.unsqueeze(-1) == kv_ids.unsqueeze(-2))
        min_val = -1e4
        attn = attn.masked_fill(~mask, min_val)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        gate = self.act(self.gate_proj(x_sorted)).view(B * ng, gs, ws, -1)
        out = out * gate

        # D) 逆排序还原
        out = out.view(B * ng, gs * ws, chan)
        out = self.proj(out)
        inverse_indices = torch.argsort(sorting_indices, dim=1)
        inverse_indices = inverse_indices.unsqueeze(-1).expand(-1, -1, chan)
        out = torch.gather(out, 1, inverse_indices)
        out = out.view(B, ng, gs, ws, chan)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, C, H, W]"""
        batch, chan, H, W = x.shape

        x_grouped, pad_h, pad_w = self.window_group_partition(x)

        # 相似度：窗口代表 = 窗内 token 均值（detach，不参与梯度）
        sim = x_grouped.detach().mean(dim=3)                                    # [B, ng, gs^2, C]
        sim = torch.einsum('b g w p c, b g k c -> b g w p k', x_grouped, sim)   # [B, ng, gs^2, ws^2, gs^2]
        cana_out = self.cana(x_grouped, sim)

        x = self.window_group_reverse(cana_out, x.shape, (pad_h, pad_w))
        return x


##############################################################
## Block: CST (CUSTBlock structure)
##############################################################
class CST(nn.Module):
    """CST: Clustered unit-level Similarity Transformer attention —— 聚簇单元级相似度注意力块"""

    def __init__(self, channels: int, num_heads: int = 4, patch_size: int = 8,
                 group_size: int = 9, ffn_scale: float = 2.0):
        super().__init__()
        # num_heads 仅为统一接口保留：原实现为全通道单头注意力，不参与计算
        self.num_heads = num_heads
        self.pe = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

        # Attention Path
        self.norm1 = LayerNorm(channels)
        self.attn = CUSTAttention(channels, window_size=patch_size, group_size=group_size)

        # FFN Path
        self.norm2 = LayerNorm(channels)
        self.ffn = ConvFFN(channels, int(channels * ffn_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        x = x + self.pe(x)

        # Attention (Pre-Norm & Residual)
        x = x + self.attn(self.norm1(x))

        # FFN (Pre-Norm & Residual)
        shortcut = x
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)      # b c h w -> b (h w) c
        x = self.ffn(x, (H, W))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)          # b (h w) c -> b c h w

        x = shortcut + x
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CST(channels=128)
    output = model(input_tensor)
    print('=== CST: Clustered unit-level Similarity Transformer attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
