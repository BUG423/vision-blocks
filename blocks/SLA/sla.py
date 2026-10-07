import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：SAT: Selective Aggregation Transformer for Image Super-Resolution (CVPR 2026 Findings)
# 论文链接：https://arxiv.org/abs/2604.07994
# 代码来源：https://github.com/PhuTran1005/SAT
# 原始许可证：MIT
# 模块出处：basicsr/archs/sat_arch.py 的 SAA 类（L492-545）及其依赖 cluster_and_merge（L386-489）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删除 basicsr registry / einops / timm / numpy 依赖与 L_SA、WindowAttention、
#          DynamicPosBias、Block 等无关模块（SAA 自身不引用它们，仅被 Block 交替堆叠）；
#          token 接口 [B,N,C] 包装为统一 4D 接口 [B,C,H,W]（内部 flatten/reshape，不改数值）；
#          torch.einsum("bnc,bnk->bkc", ...) 用 matmul 等价改写（非 einops，数值一致）；
#          原 forward 的 prev_attn / image 形参保留为可选 prev_attn（原实现即未消费，
#          纯 [B,C,H,W]->[B,C,H,W] 路径 prev_attn=None 即可独立运行）；
#          聚类合并、范数保持、交叉注意力的公式与随机子采样逻辑原样保留。

'''
模块名称：SLA (Selective Aggregation Attention) —— 选择性聚合注意力

一、模块简介
图像超分辨等密集预测任务中，特征图 token 数量巨大，标准自注意力的
O(N^2) 复杂度难以承受；而现有加速方法（窗口注意力、稀疏注意力）往往
对前景/背景 token 一视同仁，导致高频细节与大面积平坦区域被同等计算。
SAT 提出的选择性聚合注意力（SAA）另辟蹊径：先用轻量聚类把相似的
"不重要"（低信息量）token 合并成少量代表 token，再让全部原始 token
对这些代表 token 做交叉注意力——即"选择性聚合"。

核心创新点：
1. 聚类合并压缩（Cluster-Merge）：在归一化特征上以密度-距离打分
   （γ = ρ × δ）选出 K 个聚类中心，其余 token 按余弦相似度归并；
2. 范数保持（Norm Preservation）：合并后的代表 token 被重新缩放到
   原始 token 的最大范数，避免聚合造成的幅值衰减；
3. 选择性交叉注意力：Q 来自全部 N 个 token，K/V 来自压缩后的 K 个
   代表 token，复杂度从 O(N^2) 降到 O(N·K)，K = M·N；
4. 前景比例可调：M（默认 0.03）控制保留的代表 token 数量。

二、结构设计
SLA 由以下子结构组成（输入 [B, C, H, W]，N = H*W）：
1. Token 化：[B, C, H, W] → [B, N, C]；
2. Cluster-Merge（cluster_and_merge）：
   - 特征 L2 归一化后区域分层随机子采样 S = min(N, max(2K, 4K)) 个点；
   - 余弦相似度矩阵上取 top-k 均值作为密度 ρ，δ = 1 − max sim(更高密度点)；
   - γ = ρ·δ 的 top-K 作为聚类中心，全体 token 按余弦相似度指派；
   - 同簇 token 取均值 → [B, K, C]，K = int(M·N)；
3. 范数保持：T_avg 按 max_norm / avg_norm 缩放（avg_norm≈0 时保持原样）；
4. 选择性交叉注意力：
   - q = Linear(C → cr)(x)：[B, H, N, cr/H]；
   - k = Linear(C → cr)(T_avg)，v = Linear(C → C)(T_avg)；
   - attn = softmax(q k^T · (cr/H)^{-0.5})，out = attn·v → Linear(C → C)；
   - cr = int(C · c_ratio)，默认 c_ratio=0.5；
5. 输出 reshape 回 [B, C, H, W]（SAA 自身不含残差，原 Block 在外部加残差/MLP）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 SAT（CVPR 2026 Findings）提出的选择性聚合注意力 SLA：通过对
低信息量 token 进行密度-距离聚类合并并将代表 token 范数对齐到原始特征，
使全部 query 仅对少量聚合后的代表 token 做交叉注意力，在保持高频重建质量
的同时将注意力复杂度由 O(N^2) 降至 O(N·K)。"

四、适用任务
适用于图像超分辨（SR）、去噪、去模糊等密集预测任务，也可用于高分辨率
特征图上的通用视觉主干。适合 token 数量大、前景稀疏、背景可压缩的场景。
'''

__all__ = ['SLA']


def cluster_and_merge(x: torch.Tensor, cluster_num: int,
                      subsample_factor: int = 4) -> torch.Tensor:
    """密度-距离聚类 + 簇内均值合并（原 sat_arch.cluster_and_merge，数值逻辑不变）。

    输入 x: [B, N, C]，输出: [B, cluster_num, C]。
    """
    B, N, C = x.shape
    device = x.device
    K = cluster_num

    x_proj = x

    x_norm = F.normalize(x_proj, dim=-1)  # (B, N, D) where D = proj_dim or C

    S = min(N, max(2 * K, subsample_factor * K))  # Ensure S >= 2K, cap at N

    samples_per_region = S // K
    sub_idx = []
    for i in range(K):
        start_idx = i * (N // K)
        end_idx = (i + 1) * (N // K) if i < K - 1 else N
        region_size = end_idx - start_idx
        n_samples = min(samples_per_region, region_size)

        if region_size > 0:
            region_perm = torch.randperm(region_size, device=device)[:n_samples]
            sub_idx.append(start_idx + region_perm)

    # Add random samples to reach S if needed
    sub_idx = torch.cat(sub_idx)
    if len(sub_idx) < S:
        remaining = S - len(sub_idx)
        all_idx = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sub_idx] = False
        additional = all_idx[mask][torch.randperm((~mask).sum(), device=device)[:remaining]]
        sub_idx = torch.cat([sub_idx, additional])

    x_norm_sub = x_norm[:, sub_idx]  # (B, S, D)

    # Cosine similarity (normalized dot product)
    sim_sub = x_norm_sub @ x_norm_sub.transpose(1, 2)  # (B, S, S)
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1)

    # Mean of top-k similarities
    k = min(K, S - 1)
    sim_topk_sub, _ = torch.topk(sim_sub, k=k, dim=-1)  # (B, S, k)
    density_sub = sim_topk_sub.mean(dim=-1)  # (B, S)
    density_sub = density_sub + torch.rand_like(density_sub) * 1e-6

    # Mask for points with higher density
    mask_higher_density = (density_sub[:, None, :] > density_sub[:, :, None]).float()  # (B, S, S)

    # For points with higher density, keep similarity; otherwise set to very negative
    masked_sim_sub = sim_sub * mask_higher_density - 1e9 * (1.0 - mask_higher_density)

    # Maximum similarity to higher-density points
    max_sim_to_higher, _ = masked_sim_sub.max(dim=-1)  # (B, S)

    # Convert to distance: δ = 1 - similarity
    delta_sub = 1.0 - max_sim_to_higher  # (B, S)

    # Handle points with maximum density (no higher-density neighbors)
    max_density_mask_sub = (mask_higher_density.sum(dim=-1) == 0)  # (B, S)

    # For max density points, use maximum distance in subsample
    min_sim_global = sim_sub.min(dim=-1)[0]  # (B, S)
    max_dist_global = 1.0 - min_sim_global
    delta_sub[max_density_mask_sub] = max_dist_global[max_density_mask_sub]

    # Ensure delta is non-negative
    delta_sub = torch.clamp(delta_sub, min=0.0)

    # Score: γ = ρ × δ
    score_sub = density_sub * delta_sub  # (B, S)

    # Select top-K scoring points as cluster centers
    _, center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1)  # (B, K)

    # Map back to original indices
    center_idx = sub_idx[center_idx_in_sub]  # (B, K)

    # Get center representations (normalized)
    centers_norm = torch.gather(
        x_norm,
        1,
        center_idx[..., None].expand(B, K, x_norm.shape[-1])
    )  # (B, K, D)

    # Use cosine similarity (consistent with center selection)
    sim_token_center = x_norm @ centers_norm.transpose(1, 2)  # (B, N, K)

    # Assign to cluster with highest similarity
    assign_idx = sim_token_center.argmax(dim=-1)  # (B, N)

    # One-hot encoding of assignments
    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x)  # (B, N, K)

    # Count tokens per cluster
    cluster_counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, K)

    # Weighted average: sum tokens per cluster, then normalize
    # 原式 torch.einsum("bnc,bnk->bkc", x, one_hot)，matmul 等价改写
    out = torch.matmul(x.transpose(1, 2), one_hot).transpose(1, 2)  # (B, K, C)
    out = out / cluster_counts.transpose(1, 2)

    return out


class SLA(nn.Module):
    """SLA: Selective Aggregation Attention —— 选择性聚合注意力"""

    def __init__(self, channels: int, num_heads: int = 8, window_size: int = 8,
                 c_ratio: float = 0.5, M: float = 0.03,
                 qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels {channels} must be divided by num_heads {num_heads}'
        self.dim = channels
        self.num_heads = num_heads
        self.cr = int(channels * c_ratio)
        assert self.cr % num_heads == 0, \
            f'compressed dim {self.cr} must be divided by num_heads {num_heads}'
        self.scale = (self.cr // num_heads) ** -0.5
        self.M = M  # Ratio for NF (foreground size)
        # window_size 仅为统一接口保留：SAA 本身是全局 token 聚合注意力，不使用窗口
        self.window_size = window_size

        # QKV projections
        self.q = nn.Linear(channels, self.cr, bias=qkv_bias)
        self.k = nn.Linear(channels, self.cr, bias=qkv_bias)
        self.v = nn.Linear(channels, channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, prev_attn: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
            prev_attn: 可选的跨层注意力缓存（原 SAT 接口保留；SAA 核心不消费它，
                       默认 None 时为纯 [B, C, H, W] -> [B, C, H, W] 路径）
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.dim, f'expected C={self.dim}, got {C}'
        N = H * W

        # [B, C, H, W] -> [B, N, C]（token 化，统一接口包装）
        tokens = x.flatten(2).transpose(1, 2)                                    # [B, N, C]
        T_unimp = tokens
        NF = int(self.M * N)

        # Average and cluster-merge background tokens
        T_avg = cluster_and_merge(T_unimp, NF)
        # Norm preservation
        norms = torch.norm(T_unimp, dim=-1)  # B x num_unimp
        max_norm = norms.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # B x 1 x 1
        avg_norm = torch.norm(T_avg, dim=-1, keepdim=True)  # B x 1 x 1
        epsilon = 1e-6
        mask = avg_norm > epsilon  # B x 1 x 1
        scaled = (T_avg / (avg_norm + epsilon)) * max_norm
        T_avg = torch.where(mask, scaled, T_avg)

        # Concat for KV_comp
        KV_comp = T_avg
        K_size = KV_comp.shape[1]

        # Cross-Attention
        q = self.q(tokens).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(KV_comp).reshape(B, K_size, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(KV_comp).reshape(B, K_size, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # [B, N, C] -> [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    # 聚类需要 NF = int(M*N) >= 1，N = H*W 需 ≥ 34（M=0.03）；用 32x32 兼顾速度
    input_tensor = torch.randn(1, 64, 32, 32)
    model = SLA(channels=64)
    output = model(input_tensor)
    print('=== SLA: Selective Aggregation Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    # prev_attn 可选路径（默认 None 已验证）
    output2 = model(input_tensor, prev_attn=None)
    print('output_size(prev_attn=None):', output2.size())
