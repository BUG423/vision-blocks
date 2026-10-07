import math
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Linear Recurrent Unit with Semantic Modulation for Image Super-Resolution (CVPR 2026 Findings)
# 论文链接：https://arxiv.org/abs/2606.19901
# 代码来源：https://github.com/MingyuChoi-run/LSM
# 原始许可证：Apache-2.0
# 模块出处：basicsr/archs/lsm_arch.py 的 SemanticLRU 类（L517-612）与 LRUcore 类（L391-494）；
#          并行前缀扫描 associative_scan / binary_operator_diag 取自 basicsr/utils/img_util.py（L223-320）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：1) Learned_dict 原为 Block.embedding（nn.Embedding(num_tokens, dim)，
#             uniform_(-1/num_tokens, 1/num_tokens)）的外部输入，现内化为本模块的 nn.Embedding，
#             初始化分布保持一致，forward 内 expand(B,-1,-1) 等价原 repeat(B,1,1)；
#          2) np.log 改为 math.log（标量，数值相同）；jax.tree_util 的树展平替换为元组处理，
#             associative_scan / binary_operator_diag 算法与默认 axis=0 扫描约定原样保留；
#          3) 删除未使用的 hidden_dim、dropout、kwargs；d_state 与 dict_size 的隐含耦合
#             （Mk 三段 chunk 的宽度 = d_state = dict_size/4）改为显式 assert；
#          4) token 接口 [B,HW,C] 包装为统一 4D 接口 [B,C,H,W]（内部 flatten(2).transpose /
#             transpose.reshape 还原，不改数值）；num_heads 仅保留接口占位（原 SemanticLRU 无头分解）；
#          5) 去除 einops/timm/basicsr 依赖，SMU 语义字典调制与 LRU 递推公式逐行保留。

'''
模块名称：SLU (Semantic-modulated Linear Recurrent Unit) —— 语义调制线性循环单元

一、模块简介
图像超分主干中的长程依赖建模要么靠二次复杂度的自注意力，要么靠固定核
卷积。LSM 提出用线性循环单元（LRU）在展平的空间 token 序列上做递推状态
传递，并用可学习语义字典（semantic dictionary）对递推进行调制：每个空间
token 与字典原型做余弦相似度得到语义匹配分布 Mk，Mk 一方面作为字典值
的检索权重生成增强特征 Y_enhance，另一方面被切分为 A/B/C 三段，分别调制
LRU 的递推衰减 λ、输入注入 B·u 与状态读出 C·h。为使递推沿着语义相近的
token 进行，token 先按语义分组排序（Gumbel 硬分配 + 稳定排序）送入扫描，
扫描后再逆排序还原空间顺序。

核心创新点：
1. 语义调制 LRU：用语义匹配分布 Mk 同时调制递推的衰减、注入与读出，
   使线性递推获得内容相关性；
2. 语义邻域排序：按字典原型的 Gumbel 硬分配对 token 重排，让循环扫描
   沿语义相近的 token 链进行；
3. 并行前缀扫描：递推用 associative_scan 并行展开，训练高效；
4. 频域/空间混合输出：LRU 分支（out_proj→C/2）与字典检索分支（C/2）
   拼接后还原为完整通道。

二、结构设计
SLU 由以下子结构组成（形状以输入 [B, C, H, W]、N = H*W 计）：
1. token 化：x → [B, N, C]；语义字典 Learned_dict: [B, dict_size, C]
   （内部 nn.Embedding，uniform(-1/dict_size, 1/dict_size) 初始化）；
2. QKV 投影：wq: C→C/3，wk: C→C/3，wv: C→C/2；
3. SMU 语义匹配：SMU = norm(Q_U) @ norm(K_D)^T ∈ [B, N, dict_size]，
   乘 (1 + clamp(scale,0,1) * log(dict_size))，softmax 得 Mk；
4. 字典增强：Y_enhance = Mk @ V_D ∈ [B, N, C/2]；
5. Mk 三段调制：Mk_AB, Mk_C = chunk(Mk, 2)；Mk_A, Mk_B = chunk(Mk_AB, 2)；
   Mk_A ← 1 + (1 - exp(-exp(nu_log))) * (Mk_A - mean) / (amax + 1e-12)；
6. 语义邻域排序：logsoftmax(SMU) → Gumbel-softmax 硬分配 → argmax 分组号
   → 稳定排序 sort_idx 与其逆置换 rev_sort_idx；
7. 2D 投影 + CPE：in_proj 1×1 Conv2d → 乘 sigmoid(CPE(x))（3×3 深度卷积）；
8. LRUcore 扫描：categorize(U, sort_idx) → 并行扫描（λ 由 nu_log/theta_log
   参数化，B/C 为复投影，D 为直通）→ categorize(Y_scan, rev_sort_idx)
   → out_proj: C→C/2；
9. 输出拼接：cat([Y_LRU, Y_enhance], -1) → [B, N, C] → [B, C, H, W]。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 LSM（CVPR 2026 Findings）提出的语义调制线性循环单元 SLU 进行
空间长程依赖建模：以可学习语义字典产生语义匹配分布，对线性循环单元的
递推衰减、输入注入与状态读出三段进行内容调制，并按语义分组对 token 排序
后做并行前缀扫描，在保持线性复杂度的同时获得内容自适应的递推建模。"

四、适用任务
适用于图像超分、去噪等密集恢复任务中的主干特征建模，也可用于分割/检测
主干的中后段长程依赖模块。适合 token 数较大、需要线性复杂度序列建模的场景。
'''


def _slice_along_axis(start, end, stride=None, axis=0):
    return (slice(None),) * axis + (slice(start, end, stride),)


def _interleave(a: torch.Tensor, b: torch.Tensor, axis: int) -> torch.Tensor:
    """交错拼接相邻扫描结果（原 img_util._interleave）。"""
    if b_trunc := (a.shape[axis] == b.shape[axis] + 1):
        pad = [0, 0] * b.ndim
        pad[(b.ndim - axis - 1) * 2 + 1] = 1
        b = F.pad(b, pad)
    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        interleaved = interleaved[_slice_along_axis(0, b.shape[axis] + a.shape[axis] - 1, axis=axis)]
    return interleaved


def _binary_operator_diag(q_i: t.Tuple[torch.Tensor, torch.Tensor],
                          q_j: t.Tuple[torch.Tensor, torch.Tensor]):
    """对角递推的二元算子： (A_i, b_i) ⊕ (A_j, b_j) = (A_j*A_i, A_j*b_i + b_j)。"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


def _associative_scan(operator, elems, axis: int = 0, reverse: bool = False):
    """JAX 语义的并行前缀扫描（原 img_util.associative_scan）。

    elems 为两个同形张量构成的元组；算法、切片与交错逻辑与原实现逐行对应，
    仅将 jax.tree_util 的 pytree 展平替换为元组处理（本模块唯一调用形态）。
    """
    elems_flat = list(elems)
    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat):
        return list(operator(tuple(a_flat), tuple(b_flat)))

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    def _scan(elems):
        num_elems = elems[0].shape[axis]
        if num_elems < 2:
            return elems

        # 相邻配对归约
        reduced_elems = combine(
            [elem[_slice_along_axis(0, -1, stride=2, axis=axis)] for elem in elems],
            [elem[_slice_along_axis(1, None, stride=2, axis=axis)] for elem in elems])

        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = combine(
                [e[_slice_along_axis(0, -1, axis=axis)] for e in odd_elems],
                [e[_slice_along_axis(2, None, stride=2, axis=axis)] for e in elems])
        else:
            even_elems = combine(
                odd_elems,
                [e[_slice_along_axis(2, None, stride=2, axis=axis)] for e in elems])

        even_elems = [
            torch.cat([elem[_slice_along_axis(0, 1, axis=axis)], result], dim=axis)
            if result.shape.numel() > 0 and elem.shape[axis] > 0 else
            result if result.shape.numel() > 0 else
            elem[_slice_along_axis(0, 1, axis=axis)]
            for (elem, result) in zip(elems, even_elems)]

        return [_interleave(a, b, axis=axis) for a, b in zip(even_elems, odd_elems)]

    scans = _scan(elems_flat)
    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]
    return tuple(scans)


def _index_reverse(index: torch.Tensor) -> torch.Tensor:
    """排序索引的逆置换（原 lsm_arch.index_reverse）。"""
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def _categorize(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """按 index 在 dim=-1 上重排 token（原 lsm_arch.categorize）。"""
    dim = index.dim()
    assert x.shape[:dim] == index.shape, \
        'x ({}) and index ({}) shape incompatible'.format(x.shape, index.shape)
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    return torch.gather(x, dim=dim - 1, index=index)


class _LRUCore(nn.Module):
    """对角线性循环单元核心（原 lsm_arch.LRUcore）：λ 参数化 + 复 B/C 投影 + D 直通。"""

    def __init__(self, in_features: int, out_features: int, state_features: int,
                 rmin: float = 0.0, rmax: float = 1.0, max_phase: float = 6.283):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

        # D parameter
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features))

        # Lambda initialization
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        # Gamma initialization
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))))

        # Complex input projection
        self.B_re = nn.Parameter(torch.randn([state_features, in_features]) / math.sqrt(2 * in_features))
        self.B_im = nn.Parameter(torch.randn([state_features, in_features]) / math.sqrt(2 * in_features))

        # Complex output projection
        self.C_re = nn.Parameter(torch.randn([out_features, state_features]) / math.sqrt(state_features))
        self.C_im = nn.Parameter(torch.randn([out_features, state_features]) / math.sqrt(state_features))

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))              # (N,)
        lambda_phase = torch.exp(self.theta_log)                     # (N,)
        lambda_re = lambda_abs * torch.cos(lambda_phase)             # (N,)
        lambda_im = lambda_abs * torch.sin(lambda_phase)             # (N,)
        lambdas = torch.complex(lambda_re, lambda_im)                # complex (N,)

        gammas = torch.exp(self.gamma_log).unsqueeze(-1)             # (N,1)
        B_re_scaled = self.B_re * gammas                             # (N, in_features)
        B_im_scaled = self.B_im * gammas

        return lambdas, B_re_scaled, B_im_scaled, self.C_re, self.C_im, self.D

    def _scan_1d(self, seq_1d, lambdas, B_re, B_im, C_re, C_im, D,
                 promptA, promptB, promptC, state=None):
        B, L, _ = seq_1d.shape
        N = lambdas.shape[0]

        # promptA 调制递推衰减
        alpha = promptA.mean(dim=0)

        lam_base = lambdas.view(1, 1, N).expand(B, L, N)
        lam_expand = lam_base * alpha.unsqueeze(0).to(lam_base)

        # Bu
        Bu_re = seq_1d @ B_re.T
        Bu_im = seq_1d @ B_im.T
        Bu_cplx = torch.complex(Bu_re, Bu_im)

        # promptB 调制输入注入
        Bu_cplx = Bu_cplx * promptB.to(Bu_cplx)

        if state is not None:
            if state.ndim == 1:
                Bu_cplx[:, 0, :] = Bu_cplx[:, 0, :] + lambdas[None, :] * state[None, :]
            else:
                Bu_cplx[:, 0, :] = Bu_cplx[:, 0, :] + lambdas[None, :] * state

        # 并行前缀扫描
        inner_states = _associative_scan(_binary_operator_diag, (lam_expand, Bu_cplx))[1]

        # promptC 调制状态读出
        promptC_re, promptC_im = promptC.chunk(2, dim=-1)
        inner_real = inner_states.real * promptC_re
        inner_imag = inner_states.imag * promptC_im

        real_part = torch.einsum('bln,on->blo', inner_real, C_re)
        imag_part = torch.einsum('bln,on->blo', inner_imag, C_im)

        return real_part - imag_part + seq_1d @ D.T

    def forward(self, x_1d, promptA, promptB, promptC):
        lambdas, B_re, B_im, C_re, C_im, D = self.ss_params()
        return self._scan_1d(x_1d, lambdas, B_re, B_im, C_re, C_im, D,
                             promptA, promptB, promptC, state=None)


class SLU(nn.Module):
    """SLU: Semantic-modulated Linear Recurrent Unit —— 语义调制线性循环单元"""

    def __init__(self, channels: int, num_heads: int = 8, dict_size: int = 64,
                 d_state: t.Optional[int] = None, rmin: float = 0.9, rmax: float = 0.99,
                 max_phase: float = 2 * math.pi):
        super().__init__()
        assert channels > 0 and dict_size > 0 and dict_size % 4 == 0, \
            f'channels must be > 0 and dict_size must be divisible by 4, got {channels}, {dict_size}'
        self.channels = channels
        self.num_heads = num_heads          # 原 SemanticLRU 无头分解，保留仅为统一接口
        self.num_tokens = dict_size         # 原参数名 num_tokens
        # Mk 三段 chunk 宽度约束：Mk_A/Mk_B/Mk_C 经 promptA/B/C 进入 LRUcore，
        # 必须满足 d_state = dict_size / 4（原实现 num_tokens=128, d_state=32）
        self.d_state = dict_size // 4 if d_state is None else d_state
        assert self.d_state == dict_size // 4, \
            f'd_state must equal dict_size//4 ({dict_size // 4}), got {self.d_state}'

        # LRU
        self.lru_core = _LRUCore(
            in_features=channels, out_features=channels, state_features=self.d_state,
            rmin=rmin, rmax=rmax, max_phase=max_phase)
        self.out_proj = nn.Linear(channels, channels // 2, bias=True)

        # Preprocess
        self.in_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.CPE = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)

        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.wq = nn.Linear(channels, channels // 3, bias=True)
        self.wk = nn.Linear(channels, channels // 3, bias=True)
        self.wv = nn.Linear(channels, channels // 2, bias=True)
        self.scale = nn.Parameter(torch.ones([self.num_tokens]) * 0.5, requires_grad=True)

        # 语义字典（原 Block.embedding，外部传入 Learned_dict；此处内化，初始化一致）
        self.embedding = nn.Embedding(self.num_tokens, channels)
        self.embedding.weight.data.uniform_(-1.0 / self.num_tokens, 1.0 / self.num_tokens)

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
        U = x.flatten(2).transpose(1, 2)                                       # [B, N, C]
        Learned_dict = self.embedding.weight.unsqueeze(0).expand(B, -1, -1)    # [B, T, C]

        # QKV generation
        Q_U = self.wq(U)                                                       # [B, N, C/3]
        K_D = self.wk(Learned_dict)                                            # [B, T, C/3]
        V_D = self.wv(Learned_dict)                                            # [B, T, C/2]

        # SMU
        SMU = F.normalize(Q_U, dim=-1) @ F.normalize(K_D, dim=-1).transpose(-2, -1)  # [B, N, T]
        scale = torch.clamp(self.scale, 0, 1)
        SMU = SMU * (1 + scale * math.log(self.num_tokens))
        Mk = self.softmax(SMU)                                                 # [B, N, T]

        # Mk * V
        Y_enhance = (Mk @ V_D).reshape(B, N, C // 2)                           # [B, N, C/2]

        # Mk Chunk
        Mk_AB, Mk_C = Mk.chunk(2, dim=-1)
        Mk_A, Mk_B = Mk_AB.chunk(2, dim=-1)
        Mk_A = 1 + (1 - torch.exp(-torch.exp(self.lru_core.nu_log))) * \
            (Mk_A - Mk_A.mean(-1, True)) / (Mk_A.amax(-1, True) + 1e-12)

        # Semantic neighbor sorting
        pred_cls = self.logsoftmax(SMU)                                        # [B, N, T]
        cls_policy = F.gumbel_softmax(pred_cls, hard=True, dim=-1)             # [B, N, T]
        group_idx = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, N)
        _, sort_idx = torch.sort(group_idx, dim=-1, stable=False)
        rev_sort_idx = _index_reverse(sort_idx)

        # 2D projection
        U = U.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        U = self.in_proj(U)
        U = U * torch.sigmoid(self.CPE(U))
        U = U.view(B, C, N).permute(0, 2, 1).contiguous()                      # [B, N, C]

        semantic_U = _categorize(U, sort_idx)
        Y_scan = self.lru_core(semantic_U, Mk_A, Mk_B, Mk_C)                   # [B, N, C]
        Y_LRU = _categorize(Y_scan, rev_sort_idx)
        Y_LRU = self.out_proj(Y_LRU)                                           # [B, N, C/2]

        Y_out = torch.cat([Y_LRU, Y_enhance], dim=-1)                          # [B, N, C]

        # [B, N, C] -> [B, C, H, W]
        out = Y_out.transpose(1, 2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 语义排序含 O(N log N) 与逐 batch 逆置换循环，smoke 用 32x32 控制耗时
    input_tensor = torch.randn(1, 64, 32, 32)
    model = SLU(channels=64)
    output = model(input_tensor)
    print('=== SLU: Semantic-modulated Linear Recurrent Unit ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
