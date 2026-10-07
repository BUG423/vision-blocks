import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 论文：Beyond the Birkhoff Polytope: Spectral-Sphere-Constrained Hyper-Connections
#       (NeurIPS 2026)
# 论文链接：https://arxiv.org/abs/2603.20896
# 代码来源：https://github.com/6zHAOyi/s2HC
# 原始许可证：Apache-2.0
# 模块出处：litgpt/hyper_conn/shc.py 的 SHyperConnections 类（含 get_expand_reduce_stream_functions
#          的 expand/reduce 流包装）；variant='hc' 对应 litgpt/hyper_conn/hyper_connections.py
#          的 HyperConnections 类（Appendix J, Algorithm 2, arXiv:2409.19606）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删去 einops 依赖（rearrange/einsum/Reduce 全部改为 torch.reshape/permute/
#          einsum 等价写法，收缩下标与原式一致）、functools.partial / random.randrange /
#          torch.utils._pytree（原 decorate_branch 的树展平接口不适用于本块）；核心混合
#          数学（谱球约束参数化：skew-symmetric 生成元 + Cayley 变换 + 对角 σ + Uz 提升
#          到 J 的正交补，以及 alpha/beta 的动态-静态合成、sigmoid/tanh 缩放）逐行保留；
#          原 token 接口 (b s, ..., d) 包装为统一 4D 接口 [B,C,H,W]：内部
#          expand 流 repeat→(B*s) → channels-last 展平 (B*s,T,C) → 核心 width/depth
#          混合 → branch 子网络 → reduce 流 sum over s。expand/reduce 与 branch 属包装
#          层（已注明），核心混合公式未改；channel_first 选项固定为 4D channel-first，
#          residual_transform / num_fracs(frac-connections) / num_input_views 等原仓库
#          扩展未纳入（均为 Identity / 1 的默认路径）。

'''
模块名称：HYC (Hyper-Connections, s2HC) —— 谱球约束超连接

一、模块简介
残差连接是深度网络的基本构件，但标准残差只有一条恒等旁路：每一层的输出
以固定权重 1 加回主干。Hyper-Connections（HC）将单一残差流扩展为多条并行
残差流，在每一层用输入相关的动态权重做"宽度连接"（把多条流混合出分支网络
的输入）与"深度连接"（把分支输出按权重写回多条流），从而让网络学习层间信息
的路由与保留。然而原始 HC 的动态混合矩阵在参数化上是无约束的，其谱性质
（奇异值/正交性）不受控，训练稳定性与可解释性受限。

s2HC（NeurIPS 2026）的核心思想是：把超连接的残差混合矩阵约束到"谱球"上——
不再直接回归一个任意矩阵，而是用斜对称生成元经 Cayley 变换得到正交因子，
配一组 tanh 压到 [-1,1] 的对角尺度 σ，再经 Helmert 型正交基 Uz 提升到
均匀混合矩阵 J 的正交补空间。这样得到的混合矩阵 H = J + Uz·(U·diag(σ)·Vᵀ)·Uzᵀ
天然具有受控谱（偏离 Birkhoff 单纯形中心 J 的部分是正交缩放），动态系数
仅预测低维的生成元参数而非整个矩阵。

核心创新点：
1. 谱球约束参数化：混合矩阵 = 均匀矩阵 J + Uz·E·Uzᵀ，E = U·diag(σ)·Vᵀ，
   U/V 由斜对称矩阵经 Cayley 变换（(I+A)⁻¹(I-A)）得到正交因子，σ ∈ (-1,1)^{s-1}；
2. 动态-静态分解：所有动态系数（预混合 α_pre、残差混合生成元、深度混合 β）
   由多流拼接特征经无偏置线性层预测，乘小尺度（1e-2）后加到静态可学习偏置，
   保证初始接近恒等/均匀混合；
3. 多残差流：s 条并行残差流（默认 4），宽度连接混合出分支输入，深度连接
   以 β 加权写回，等价于层间的可学习信息路由；
4. 深度 β 门控：β = 2·sigmoid(·) ∈ (0,2)，分支输出可增强或抑制各残差流。

二、结构设计
HYC 由以下子结构组成（形状以输入 [B,C,H,W]、s = num_streams、T = H*W 计；
以下 1-3 为论文核心混合，4-6 为本块包装层）：
1. 流展开（包装）：x [B,C,H,W] → repeat 成 (B*s, C, H, W)，转 channels-last
   并展平空间得 (B*s, T, C)，即原仓库 expand_fn = repeat('b ... -> (b s) ...')；
2. 宽度连接（核心，SHyperConnections.width_connection 等价）：
   - 多流拼接 (B,T,s,C) → reshape (B,T,s*C) → RMSNorm(s*C)；
   - α_pre = sigmoid(Linear(s*C→s)·1e-2 + static_alpha_pre) ∈ (0,1)^{s×1}；
   - 残差混合：动态 (Linear(s*C→(s-1)²)) 切成 z_U/z_V/z_S 三段，分别乘
     residual_rot_scale_u/v、residual_val_scale 后加静态偏置；
     z_U = γ_u·tanh(·), z_V = γ_v·tanh(·), σ = tanh(·)；
     A_U/A_V 以上三角填充后减转置成斜对称；
     U = (I+A_U)⁻¹(I-A_U), V = (I+A_V)⁻¹(I-A_V)（Cayley，正交）；
     E = U·(σ.unsqueeze(-1)*Vᵀ), X = Uz·E·Uzᵀ, H = J + X ∈ R^{s×s}；
   - α = cat([α_pre, H], dim=-1) ∈ R^{s×(s+1)}；β = 2·sigmoid(Linear(s*C→s)·1e-2
     + static_beta) ∈ (0,2)^s；
   - 混合：mix_h = einsum('...st,...sd->...td', α, streams)，t=0 为分支输入
     (B,T,C)，t=1..s 为新残差流 (B,T,s,C) → (B*s,T,C)；
3. 深度连接（核心）：output = einsum('btd,bts->btsd', branch_out, β) →
   (B*s,T,C)，residuals = output + residuals，Dropout；
4. 分支子网络（包装）：默认 Conv3x3(C,C) - GELU - Conv3x3(C,C)，输入/输出
   [B,C,H,W]；可通过 branch 参数替换为任意保持 [B,C,H,W] 形状的子网络；
5. 流归约（包装）：(B*s,C,H,W) → reshape (B,s,C,H,W) → sum over s，
   即原仓库 reduce_fn = reduce('(b s) ... -> b ...', 'sum')；
6. variant='hc'（对照）：原始 HyperConnections 参数化——α = tanh(normed·W_α)·1e-2
   + static_alpha（static_alpha = [流选择列 | I_s]，norm 按流独立 RMSNorm(C)），
   β = tanh(normed·w_β)·1e-2 + static_beta（无 sigmoid），混合与深度连接同构。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 s2HC（NeurIPS 2026）的谱球约束超连接 HYC 替换标准残差连接：将
单一残差流扩展为 s 条并行流，每层以动态-静态分解的 α 做宽度混合产生分支输入，
分支输出经 β 门控做深度混合写回各流；残差混合矩阵约束为 H = J + Uz·(U diag(σ)
Vᵀ)·Uzᵀ（U/V 为 Cayley 变换正交因子），使混合谱受控、训练更稳定。"

原论文引用格式：
Beyond the Birkhoff Polytope: Spectral-Sphere-Constrained Hyper-Connections.
NeurIPS 2026. arXiv:2603.20896.

四、适用任务
适用于分类、检测、分割等需要堆叠多层残差块的视觉任务，可替换 ResNet/ViT
中的标准残差连接。特别适合深网络（层间信息路由收益大）与训练不稳定的场景。
注意：本块为 4D [B,C,H,W] 版本，内部含多流展开/归约与默认分支子网络包装；
若需在 Transformer token 布局中使用原论文的层包装形式，应直接引用上游
s2HC 仓库的 SHyperConnections。
'''


class _RMSNorm(nn.Module):
    """原仓库 RMSNorm：F.normalize(x, dim=-1) * sqrt(dim) * (gamma + 1)"""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


class _SHCCore(nn.Module):
    """s2HC 谱球约束超连接核心混合（SHyperConnections），channels-last (B*s, T, C) 进出。

    仅重排张量布局与命名，宽度/深度连接的公式与 shc.py 逐行一致。
    """

    def __init__(self, channels: int, num_streams: int,
                 layer_index: t.Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert num_streams >= 2, 'SHyperConnections core needs num_streams >= 2'
        s = num_streams
        sm1 = s - 1
        d = channels

        init_residual_index = (torch.randint(0, s, (1,)).item()
                               if layer_index is None else layer_index) % s
        in_dim = int(d * s)
        self.norm = _RMSNorm(in_dim)
        self.num_streams = s

        J = torch.ones(s, s) / s
        self.register_buffer('J', J)

        Uz = torch.zeros(s, sm1)
        for i in range(1, s):
            val = 1.0 / math.sqrt(i * (i + 1))
            Uz[:i, i - 1] = val
            Uz[i, i - 1] = -i * val
        self.register_buffer('Uz', Uz)

        I_sm1 = torch.eye(sm1)
        self.register_buffer('eye_sm1', I_sm1)

        triu_i, triu_j = torch.triu_indices(sm1, sm1, offset=1)
        self.register_buffer('triu_i', triu_i)
        self.register_buffer('triu_j', triu_j)

        # Pre
        self.to_alpha_pre = nn.Linear(in_dim, s, bias=False)
        nn.init.zeros_(self.to_alpha_pre.weight)
        init_alpha0 = torch.ones((s, 1)) * -1
        init_alpha0[init_residual_index, 0] = 1.
        self.static_alpha_pre = nn.Parameter(init_alpha0)

        # Residual（谱球约束生成元参数）
        self.to_alpha_residual = nn.Linear(in_dim, sm1 ** 2, bias=False)
        nn.init.zeros_(self.to_alpha_residual.weight)
        init_bias = torch.zeros(sm1 ** 2)
        init_bias[-sm1:] = 4.0
        self.static_alpha_residual = nn.Parameter(init_bias)

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_rot_scale_u = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_rot_scale_v = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_val_scale = nn.Parameter(torch.ones(1) * 1e-2)

        self.gamma_u = nn.Parameter(torch.ones(1))
        self.gamma_v = nn.Parameter(torch.ones(1))

        # Beta
        self.to_beta = nn.Linear(in_dim, s, bias=False)
        nn.init.zeros_(self.to_beta.weight)
        beta_init = torch.ones(s) * -1.
        beta_init[init_residual_index] = 1.
        self.static_beta = nn.Parameter(beta_init)
        self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)

    def width_connection(self, residuals: torch.Tensor
                         ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B*s, T, C) -> 分支输入 (B,T,C), 新残差流 (B*s,T,C), beta (B,T,s)"""
        s = self.num_streams
        sm1 = s - 1
        B = residuals.shape[0] // s
        T, d = residuals.shape[1], residuals.shape[2]

        # (B*s, T, C) -> (B, T, s, C)
        x = residuals.reshape(B, s, T, d).permute(0, 2, 1, 3)
        normed = self.norm(x.reshape(B, T, s * d))

        # Pre
        dynamic_pre = self.to_alpha_pre(normed).unsqueeze(-1)     # (B,T,s,1)
        alpha_pre = (dynamic_pre * self.pre_branch_scale) + self.static_alpha_pre
        alpha_pre = alpha_pre.sigmoid()

        # Residual：动态生成元参数 → 斜对称 → Cayley 正交 → Uz 提升到 J 的正交补
        dynamic_res = self.to_alpha_residual(normed)              # (B,T,sm1^2)

        k = sm1 * (sm1 - 1) // 2
        dyn_U = dynamic_res[..., :k] * self.residual_rot_scale_u
        dyn_V = dynamic_res[..., k:2 * k] * self.residual_rot_scale_v
        dyn_S = dynamic_res[..., 2 * k:] * self.residual_val_scale

        stat_U = self.static_alpha_residual[:k]
        stat_V = self.static_alpha_residual[k:2 * k]
        stat_S = self.static_alpha_residual[2 * k:]

        z_U = self.gamma_u * torch.tanh(dyn_U + stat_U)           # (B,T,k)
        z_V = self.gamma_v * torch.tanh(dyn_V + stat_V)           # (B,T,k)
        sigma = torch.tanh(dyn_S + stat_S)                        # (B,T,sm1)

        A_U = z_U.new_zeros(*z_U.shape[:-1], sm1, sm1)
        A_V = z_V.new_zeros(*z_V.shape[:-1], sm1, sm1)

        A_U[..., self.triu_i, self.triu_j] = z_U
        A_V[..., self.triu_i, self.triu_j] = z_V

        A_U = A_U - A_U.transpose(-1, -2)
        A_V = A_V - A_V.transpose(-1, -2)

        U_core = torch.linalg.solve(self.eye_sm1 + A_U, self.eye_sm1 - A_U)
        V_core = torch.linalg.solve(self.eye_sm1 + A_V, self.eye_sm1 - A_V)

        E_core = U_core @ (sigma.unsqueeze(-1) * V_core.transpose(-1, -2))
        X = self.Uz @ E_core @ self.Uz.T                          # (B,T,s,s)
        alpha_residual = self.J + X

        alpha = torch.cat((alpha_pre, alpha_residual), dim=-1)    # (B,T,s,s+1)

        # Beta
        dc_weight = self.to_beta(normed)                          # (B,T,s)
        beta = (dc_weight * self.h_post_scale) + self.static_beta
        beta = beta.sigmoid() * 2

        # 混合：'... s t, ... s d -> ... t d'；t=0 为分支输入，t=1..s 为残差流
        mix_h = torch.einsum('btsu,btsd->btud', alpha, x)         # (B,T,s+1,d)
        branch_input = mix_h[:, :, 0, :]                          # (B,T,d)
        new_residuals = mix_h[:, :, 1:, :]                        # (B,T,s,d)

        # (B,T,s,d) -> (B*s,T,d)，s 为批次内侧维
        new_residuals = new_residuals.permute(0, 2, 1, 3).reshape(B * s, T, d)
        return branch_input, new_residuals, beta

    def depth_connection(self, branch_output: torch.Tensor,
                         residuals: torch.Tensor,
                         beta: torch.Tensor) -> torch.Tensor:
        """分支输出 (B,T,C) 写回残差流 (B*s,T,C)"""
        s = self.num_streams
        B = residuals.shape[0] // s
        T, d = residuals.shape[1], residuals.shape[2]

        # einsum(branch_output, beta, 'b t d, b t s -> b t s d')
        output = torch.einsum('btd,bts->btsd', branch_output, beta)
        output = output.permute(0, 2, 1, 3).reshape(B * s, T, d)

        residuals = output + residuals
        return self.dropout(residuals)


class _HCCore(nn.Module):
    """原始 HyperConnections 核心混合（num_fracs=1 / num_input_views=1），channels-last。

    对照实现（variant='hc'），公式与 hyper_connections.py 一致；frac-connections
    （num_fracs>1）扩展未纳入。
    """

    def __init__(self, channels: int, num_streams: int,
                 layer_index: t.Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert num_streams >= 2, 'HyperConnections core needs num_streams >= 2'
        s = num_streams
        d = channels

        self.norm = _RMSNorm(d)
        self.act = nn.Tanh()
        self.num_streams = s

        init_residual_index = (torch.randint(0, s, (1,)).item()
                               if layer_index is None else layer_index) % s

        # 宽度连接：static_alpha = [流选择列 | I_s]
        init_alpha0 = torch.zeros((s, 1))
        init_alpha0[init_residual_index, :] = 1.
        self.static_alpha = nn.Parameter(
            torch.cat((init_alpha0, torch.eye(s)), dim=1))         # (s, s+1)

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(d, s + 1))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # 深度连接：beta（无 sigmoid）
        self.static_beta = nn.Parameter(torch.ones(s))
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(d))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)

    def width_connection(self, residuals: torch.Tensor
                         ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.num_streams
        B = residuals.shape[0] // s
        T, d = residuals.shape[1], residuals.shape[2]

        x = residuals.reshape(B, s, T, d).permute(0, 2, 1, 3)     # (B,T,s,d)
        normed = self.norm(x)

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)      # (B,T,s,s+1)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        dc_weight = self.act(normed @ self.dynamic_beta_fn)       # (B,T,s)
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        mix_h = torch.einsum('btsu,btsd->btud', alpha, x)         # (B,T,s+1,d)
        branch_input = mix_h[:, :, 0, :]
        new_residuals = mix_h[:, :, 1:, :].permute(0, 2, 1, 3).reshape(B * s, T, d)
        return branch_input, new_residuals, beta

    def depth_connection(self, branch_output: torch.Tensor,
                         residuals: torch.Tensor,
                         beta: torch.Tensor) -> torch.Tensor:
        s = self.num_streams
        B = residuals.shape[0] // s
        T, d = residuals.shape[1], residuals.shape[2]

        output = torch.einsum('btd,bts->btsd', branch_output, beta)
        output = output.permute(0, 2, 1, 3).reshape(B * s, T, d)

        residuals = output + residuals
        return self.dropout(residuals)


class HYC(nn.Module):
    """HYC: spectral-sphere-constrained Hyper-Connections —— 谱球约束超连接（多残差流）"""

    def __init__(self, channels: int, num_streams: int = 4, variant: str = 'shc',
                 layer_index: t.Optional[int] = None, dropout: float = 0.0,
                 branch: t.Optional[nn.Module] = None):
        super().__init__()
        assert channels > 0, 'channels must be positive'
        assert num_streams >= 1, 'num_streams must be >= 1'
        variant = variant.strip().lower()
        assert variant in ('shc', 'hc'), "variant must be 'shc' or 'hc'"

        self.channels = channels
        self.num_streams = num_streams
        self.variant = variant

        # ── 包装层：分支子网络（论文核心混合之外的部分，可替换） ──
        if branch is None:
            self.branch: nn.Module = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
        else:
            self.branch = branch

        if num_streams == 1:
            # 原仓库 disable 路径：退化为标准残差（Residual 类）
            self.core = None
            self.dropout = nn.Dropout(dropout)
        else:
            core_cls = _SHCCore if variant == 'shc' else _HCCore
            self.core = core_cls(channels, num_streams,
                                 layer_index=layer_index, dropout=dropout)
            self.dropout = None

    def _expand_streams(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,H,W] -> (B*s,C,H,W)，批次索引 = b*s + 流号（等价 repeat 'b ... -> (b s) ...'）"""
        B, C, H, W = x.shape
        s = self.num_streams
        return x.unsqueeze(1).expand(B, s, C, H, W).reshape(B * s, C, H, W)

    def _reduce_streams(self, x: torch.Tensor) -> torch.Tensor:
        """(B*s,C,H,W) -> [B,C,H,W]，对 s 条流求和（等价 reduce '(b s) ... -> b ...'）"""
        B = x.shape[0] // self.num_streams
        s = self.num_streams
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        return x.reshape(B, s, C, H, W).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        assert x.dim() == 4, f'expected [B,C,H,W], got {tuple(x.shape)}'
        assert x.shape[1] == self.channels, \
            f'channel mismatch: x has {x.shape[1]}, module configured with {self.channels}'

        if self.core is None:
            # num_streams == 1：标准残差
            return self.dropout(self.branch(x) + x)

        B, C, H, W = x.shape
        s = self.num_streams

        # 1) 流展开 + channels-last 展平空间
        x_exp = self._expand_streams(x)                    # (B*s, C, H, W)
        x_cl = x_exp.permute(0, 2, 3, 1)                   # (B*s, H, W, C)
        residuals = x_cl.reshape(B * s, H * W, C)          # (B*s, T, C)

        # 2) 宽度连接（核心混合）
        branch_input, residuals, beta = self.core.width_connection(residuals)

        # 3) 分支子网络（包装）：(B,T,C) <-> [B,C,H,W]
        b_in = branch_input.reshape(B, H, W, C).permute(0, 3, 1, 2)   # [B,C,H,W]
        b_out = self.branch(b_in)                                     # [B,C,H,W]
        b_out = b_out.permute(0, 2, 3, 1).reshape(B, H * W, C)        # (B,T,C)

        # 4) 深度连接（核心混合）
        residuals = self.core.depth_connection(b_out, residuals, beta)  # (B*s,T,C)

        # 5) 流归约（包装）
        out = residuals.reshape(B * s, H, W, C).permute(0, 3, 1, 2)   # (B*s,C,H,W)
        return self._reduce_streams(out)                               # [B,C,H,W]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = HYC(channels=128)
    output = model(input_tensor)
    print('=== HYC: Hyper-Connections (s2HC) ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    model_hc = HYC(channels=128, variant='hc', num_streams=4)
    out_hc = model_hc(input_tensor)
    print("variant='hc' output_size:", out_hc.size())
    print("variant='hc' params:", count_parameters(model_hc))
    model_s1 = HYC(channels=128, num_streams=1)
    print('num_streams=1 output_size:', model_s1(input_tensor).size())
