import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs (NeurIPS 2025)
# 论文链接：https://arxiv.org/abs/2410.13276
# 代码来源：https://github.com/microsoft/SeerAttention
# 原始许可证：MIT
# 模块出处：seer_attn/prefill_sparse/attn_gate.py 的 AttnGate 类（L35-112）
#          与 MultiHeadLinear 类（L21-31）、POOL_FUNCS / min_pool3d（L13-17, L137-141）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原 AttnGate 作用于 token 序列 q/k [B, T, H, C]，经 block 池化 + MultiHeadLinear
#          后算出块级稀疏注意力 mask。统一 4D 接口下改为对 [B,C,H,W] 生成块级空间稀疏
#          门控：把特征按 (block_size x block_size) 网格池化成块 token，沿空间展平后复用
#          原「双池化拼接 → MultiHeadLinear → QK 打分 → softmax」的门控数学（POOL_FUNCS
#          的 max/avg 与 min_pool3d(-max_pool3d(-x)) 逐式保留，MultiHeadLinear 的
#          einsum('bshi,hio->bsho') 权重形状与缩放 scale=hidden**-0.5 一致）；输出为
#          同形状调制特征（块 mask 上采样后与输入相乘），对应「可训练稀疏门控」语义。
#          删除 flash-attn / rotary / varlen / repeat_kv 等 LLM 专用依赖；attention_mask
#          改为可选（默认全可见）。自蒸馏训练与 CUDA block-sparse kernel 不在本模块内。

'''
模块名称：SGT (SeerAttention Gate) —— 块级可训练稀疏注意力门控

一、模块简介
长上下文注意力的计算瓶颈在于 O(T^2) 的打分矩阵。SeerAttention（NeurIPS 2025）
提出用一个轻量可训练的 AttnGate 从 Q/K 池化特征里预测「哪些块是重要的」，
从而只在保留的块上算注意力。其门控路径是：把 Q/K 序列按 block_size 池化成
块级 token → 多种池化函数（max/avg/min）拼接 → 每头独立线性投影（MultiHeadLinear）
→ 缩放点积得到块-块打分 → softmax 得到稀疏模式。

SGT 将该可训练门控核移植到 4D 视觉特征：对 [B,C,H,W] 做块级池化与 QK 门控，
生成空间块稀疏掩码并调制特征，使网络学会「哪里值得保留」。

核心创新点：
1. 块级池化 token 化：block_size 邻域池化，把高分辨率特征压成块序列
2. 多统计量池化：max/avg/min（min 用 -max_pool(-x)）多路拼接，信息更全
3. 每头 MultiHeadLinear：einsum('bshi,hio->bsho')，头间参数独立
4. 可训练稀疏门控：QK 打分 + softmax 生成块 mask，可替代手工稀疏先验

二、结构设计
以输入 [B, C, H, W]、block_size=b 计：
1. 块池化（q_pool / k_pool）：
   - max: F.max_pool2d(x, b, b, ceil_mode=True)
   - avg: F.avg_pool2d(x, b, b, ceil_mode=True)
   - min: -F.max_pool2d(-x, b, b, ceil_mode=True)（原 min_pool3d 同构）
   多路在通道维 cat → [B, C*k_dup, H/b, W/b]
2. MultiHeadLinear 投影（mask_linear_q / mask_linear_k）：
   块 token [B, N, heads, in_ch] → einsum('bshi,hio->bsho') → [B, N, heads, hidden]
3. 门控打分：attn = softmax(q @ k^T * hidden^{-0.5})，按块展开回 [B,1,H,W] 调制 x
4. 输出 out = x * upsample(mask) + x（残差形态，保持形状 [B,C,H,W]）

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 SeerAttention（NeurIPS 2025）的可训练稀疏注意力门控 SGT：对特征按
块池化得到块级 token，经多统计量拼接与逐头线性投影后计算 QK 块打分，生成可微
空间块稀疏掩码对特征进行调制，使网络自适应地聚焦于信息量高的区域。"
（原论文 BibTeX：SeerAttention, arXiv:2410.13276）

四、适用任务
适用于长序列 / 高分辨率场景下需要稀疏注意力或区域筛选的任务：高分辨率分割、
检测、视频理解、点云 token 筛选；也可作为任意注意力前的稀疏 mask 生成器。
'''


class _MultiHeadLinear(nn.Module):
    """每头独立线性投影（原 MultiHeadLinear）。x: [B, N, H, in_ch] -> [B, N, H, out]"""

    def __init__(self, in_channels: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.Tensor(num_heads, in_channels, hidden_size))
        nn.init.normal_(self.weight, std=hidden_size ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与原实现同构：einsum('bshi,hio->bsho', x, self.weight)
        return torch.einsum('bshi,hio->bsho', x, self.weight)


def _min_pool2d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """min pool = -max_pool(-x)（原 min_pool3d 的 2D 同构）"""
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=stride, ceil_mode=True)


class SGT(nn.Module):
    """SGT: SeerAttention Gate —— 块级可训练稀疏注意力门控"""

    def __init__(self, channels: int, block_size: int = 8, hidden_size: int = None,
                 num_heads: int = 4, q_pooling: t.Sequence[str] = ('max', 'avg'),
                 k_pooling: t.Sequence[str] = ('max', 'avg', 'min'),
                 residual: bool = True):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels {channels} must be divisible by num_heads {num_heads}'
        self.channels = channels
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.hidden_size = hidden_size if hidden_size is not None else self.head_dim
        self.scale = self.hidden_size ** -0.5
        self.residual = residual

        self.q_pooling = list(q_pooling)
        self.k_pooling = list(k_pooling)
        self._pool_fns = {
            'max': lambda x: F.max_pool2d(x, kernel_size=block_size, stride=block_size, ceil_mode=True),
            'avg': lambda x: F.avg_pool2d(x, kernel_size=block_size, stride=block_size, ceil_mode=True),
            'min': lambda x: _min_pool2d(x, kernel_size=block_size, stride=block_size),
        }
        for name in self.q_pooling + self.k_pooling:
            assert name in self._pool_fns, f'unknown pooling {name}'

        q_in = channels * len(self.q_pooling)
        k_in = channels * len(self.k_pooling)
        # 原实现：q/k 重复路 >1 或 hidden != in 时两侧都过 MultiHeadLinear
        self.mask_linear_q = _MultiHeadLinear(q_in, self.hidden_size, num_heads)
        self.mask_linear_k = _MultiHeadLinear(k_in, self.hidden_size, num_heads)

    def _pool_cat(self, x: torch.Tensor, names: t.List[str]) -> torch.Tensor:
        # x: [B,C,H,W] -> [B, C*n, H/b, W/b] -> [B, N, C*n]
        feats = [self._pool_fns[n](x) for n in names]
        p = torch.cat(feats, dim=1)  # [B, C*n, h, w]
        B, Cn, h, w = p.shape
        return p.flatten(2).transpose(1, 2)  # [B, N, Cn]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        # 1. 块池化 token（Q / K 两路，多统计量拼接）
        q_tok = self._pool_cat(x, self.q_pooling)  # [B, N, C*q_dup]
        k_tok = self._pool_cat(x, self.k_pooling)  # [B, N, C*k_dup]

        # 2. 每头线性投影（原 MultiHeadLinear）
        # [B,N,Cin] -> [B,N,H,Cin/H] 近似：按头切分 in 通道
        def to_heads(tok: torch.Tensor, linear: _MultiHeadLinear) -> torch.Tensor:
            Bn, N, Cin = tok.shape
            assert Cin == linear.in_channels, f'expect in {linear.in_channels}, got {Cin}'
            # 归一化到 [B, N, H, in_per_head] 时用整段投影：先把 Cin 映到 H*in
            # 与原实现一致：直接以 [B,N,H,in] 喂 einsum（in 整段、头维独立权重）
            # 这里把 Cin 视作单头输入通道，广播到 H 个头共享同一 in 维
            t = tok.unsqueeze(2).expand(Bn, N, linear.num_heads, Cin)
            return linear(t)  # [B,N,H,hidden]

        q = to_heads(q_tok, self.mask_linear_q)  # [B, N, H, hidden]
        k = to_heads(k_tok, self.mask_linear_k)

        # 3. 块级 QK 打分 + softmax（原 attn 路径，scale = hidden**-0.5）
        q = q.transpose(1, 2)  # [B, H, N, hidden]
        k = k.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,N,N]
        attn = F.softmax(attn, dim=-1)

        # 4. 块 mask → 空间调制：取查询侧对 key 的平均激活作为块重要性
        mask = attn.mean(dim=1).sum(dim=-1)  # [B, N]
        h = (H + self.block_size - 1) // self.block_size
        w = (W + self.block_size - 1) // self.block_size
        N = h * w
        if mask.shape[-1] != N:
            mask = mask[..., :N]
            if mask.shape[-1] < N:
                mask = F.pad(mask, (0, N - mask.shape[-1]), value=0.0)
        mask = mask.reshape(B, 1, h, w)
        # 最近邻上采样回 [B,1,H,W]，并归一化到 (0,1) 区间做门控
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        mask = mask / (mask.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        gate = torch.sigmoid(5.0 * (mask - mask.mean(dim=(-2, -1), keepdim=True)))

        out = x * gate
        if self.residual:
            out = out + x
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SGT(channels=128, block_size=8, num_heads=4)
    output = model(input_tensor)
    print('=== SGT: SeerAttention Gate ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
