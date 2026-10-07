import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Selective Synergistic Learning for Video Object-Centric Learning (SSync) (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2606.15527
# 代码来源：https://github.com/wjun0830/SSync
# 原始许可证：MIT (Copyright (c) 2023 Maximilian Seitzer and Andrii Zadaianchuk)
# 模块出处：SSync/modules/groupers.py 的 SlotAttention 类（L15-96），slot 初始化取自
#   SSync/modules/initializers.py 的 RandomInit（L14-28），MLP 结构取自 SSync/modules/networks.py
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：仅做等价重写与接口统一——(1) 删除 SSync 的 registry/构建系统与 timm 依赖，
#   MLP 手写为 LayerNorm-Linear-ReLU-Linear+残差（与 networks.MLP(initial_layer_norm=True,
#   residual=True) 等价）；(2) 槽注意力迭代竞争数学（槽内 LayerNorm -> q、特征 LayerNorm -> k/v、
#   softmax(dim=1) 槽竞争、eps 后重归一化、GRU 更新、MLP 精炼）逐行保持不变；
#   (3) 为满足 4D 接口增加包装层：空间展平为 token、RandomInit 槽初始化、输出投影把槽按
#   注意力软分配还原为 [B,C,H,W]——包装层为接口统一所加，核心槽竞争数学不变。

'''
模块名称：SLT (Slot Attention) —— 槽注意力

一、模块简介
无监督物体表征学习希望把场景分解为若干可组合的"物体槽"（object slot），
而槽注意力（Slot Attention, Locatello et al., ICML 2020）是其中的核心竞争
机制：一组可学习的槽向量通过迭代的注意力竞争输入特征，每个槽"争夺"与其
最相关的 token（softmax 在槽维上做归一化，形成槽间竞争而非 token 间竞争），
再经 GRU 与 MLP 精炼，迭代数轮后得到一组彼此分离的物体级表征。SSync 在
视频物体中心学习中沿用该分组器，并以选择性协同学习组织时序上的槽对应。
本模块保留槽竞争的完整数学，并外加统一 4D 接口所需的展平/还原包装。

核心创新点：
1. 槽间竞争归一化：softmax(dim=1) 沿槽维归一化，token 在槽之间被竞争性分配
2. 迭代精炼：GRUCell 以注意力加权更新量逐步写入槽状态，共 n_iters 轮
3. 双 LayerNorm：特征与槽分别归一化后再投影到共享 kvq 维做点积
4. 重归一化稳定性：softmax 结果加 eps 后按 token 求和重归一化，避免数值退化

二、结构设计
SLT 由槽注意力核心与 4D 包装组成，张量形状流转如下（S=num_slots，M=H*W）：
1. 包装展平：x [B,C,H,W] -> tokens [B,M,C]
2. 槽初始化（RandomInit，包装层新增）：可学习 mean/log_std 采样 slots [B,S,D]
   （D=slot_dim，初始 std=D^-0.5）
3. 特征投影：tokens -> LayerNorm(C) -> to_k/to_v（无偏置 Linear）-> k,v [B,M,K]
   （K=kvq_dim，默认 D）
4. 迭代 n_iters 轮：
   a. slots -> LayerNorm(D) -> to_q -> q [B,S,K]
   b. dots = (q @ k^T) * K^-0.5 -> [B,S,M]；pre_norm_attn = softmax(dim=1)（槽竞争）
   c. attn = (pre_norm_attn + eps) / sum(-1, keepdim)；updates = attn @ v [B,S,K]
   d. GRUCell(updates, slots) 或 slots + updates -> [B,S,D]
   e. MLP：LayerNorm -> Linear(D,4D) -> ReLU -> Linear(4D,D) + 残差
5. 包装投影（接口统一新增）：to_out(slots) [B,S,C]；按最终注意力软分配还原
   tokens_out[b,m,c] = sum_s attn[b,s,m] * slot_feat[b,s,c] -> [B,M,C]
   -> 重排为 [B,C,H,W]

三、论文写法参考
若在论文中引用该模块，可描述为：
"我们采用槽注意力（SLT）[Locatello et al., ICML 2020; SSync, ECCV 2026] 对特征
进行物体中心分组：一组可学习槽通过迭代的槽间竞争注意力聚合 token，并经 GRU 与
MLP 精炼，得到分离的物体级表征。"
（原论文引用格式：Locatello et al., "Object-Centric Learning with Slot Attention",
ICML 2020；SSync authors, "Selective Synergistic Learning for Video Object-Centric
Learning", ECCV 2026.）

四、适用任务
无监督/自监督物体中心表征学习、视频物体分解与跟踪、场景分解、组合生成；
也可作为特征重组模块用于检测/分割中需要物体级聚合的场景。
'''
__all__ = ['SLT']


class _SlotMLP(nn.Module):
    """槽精炼 MLP：LayerNorm -> Linear -> ReLU -> Linear + 残差（等价原 networks.MLP）"""

    def __init__(self, slot_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(slot_dim)
        self.fc1 = nn.Linear(slot_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, slot_dim)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return slots + self.fc2(self.act(self.fc1(self.norm(slots))))


class SLT(nn.Module):
    """SLT: Slot Attention —— 槽注意力（槽竞争 + 4D 接口包装）"""

    def __init__(self, channels: int, slot_dim: int = 64, num_slots: int = 7,
                 n_iters: int = 3, kvq_dim: t.Optional[int] = None,
                 hidden_dim: t.Optional[int] = None, eps: float = 1e-8,
                 use_gru: bool = True, use_mlp: bool = True):
        super().__init__()
        assert n_iters >= 1
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(channels, kvq_dim, bias=False)
        self.to_v = nn.Linear(channels, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim
        self.mlp = _SlotMLP(slot_dim, hidden_dim) if use_mlp else None

        self.norm_features = nn.LayerNorm(channels)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim ** -0.5

        # ---- 包装层（接口统一所加，非原 SlotAttention 组件）----
        # 槽初始化：RandomInit（SSync/modules/initializers.py）
        initial_std = slot_dim ** -0.5
        self.slot_mean = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slot_log_std = nn.Parameter(
            torch.log(torch.ones(1, 1, slot_dim) * initial_std))
        # 槽 -> 通道输出投影（把槽表征还原为空间特征图所需）
        self.to_out = nn.Linear(slot_dim, channels)

    def _init_slots(self, batch_size: int) -> torch.Tensor:
        noise = torch.randn(batch_size, self.num_slots, self.slot_dim,
                            device=self.slot_mean.device)
        return self.slot_mean + noise * self.slot_log_std.exp()

    def step(self, slots: torch.Tensor, keys: torch.Tensor,
             values: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行一轮槽注意力迭代。"""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots_ = torch.einsum("bsd, bfd -> bsf", queries, keys)
        dots = dots_ * self.scale
        pre_norm_attn = torch.softmax(dots, dim=1)     # 沿槽维竞争
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru is not None:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn, dots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        # 包装：空间展平为 token [B, HW, C]
        features = x.flatten(2).transpose(1, 2)

        slots = self._init_slots(B)                    # [B, S, D]

        features = self.norm_features(features)
        keys = self.to_k(features)                     # [B, HW, K]
        values = self.to_v(features)                   # [B, HW, K]

        for _ in range(self.n_iters):
            slots, pre_norm_attn, dots = self.step(slots, keys, values)

        # 包装：槽经软分配还原为 [B, HW, C] -> [B, C, H, W]
        attn = pre_norm_attn                           # [B, S, HW]
        slot_feat = self.to_out(slots)                 # [B, S, C]
        tokens_out = torch.einsum("bsf, bsc -> bfc", attn, slot_feat)  # [B, HW, C]
        out = tokens_out.transpose(1, 2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SLT(channels=128)
    output = model(input_tensor)
    print('=== SLT: Slot Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
