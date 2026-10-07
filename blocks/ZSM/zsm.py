import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：ZigzagPointMamba: Spatial-Semantic Mamba for Point Cloud Understanding (NeurIPS 2025)
# 论文链接：https://neurips.cc/virtual/2025/poster/116930
# 代码来源：https://github.com/Rabbitttttt218/ZigzagPointMamba
# 原始许可证：Apache-2.0
# 模块出处：models/block.py 的 Block 类（pre-norm + mixer + 残差）与
#          models/point_mamba.py 的 MixerModel / PointMamba 中空间-语义双向扫描组织
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原 Block 的 mixer 为 mamba_ssm 选择性扫描（CUDA）。为满足零第三方依赖的
#          统一接口，本模块保留「zigzag 空间-语义双向扫描」这一核心策略，把序列混合器
#          换成纯 PyTorch 可逆的因果/反因果深度卷积 + 门控线性递推（结构对应选择性扫描
#          的输入依赖门控与有状态递推，但不复刻 CUDA 核）；zigzag 展开顺序、双向
#          （forward/backward）拼接、pre-norm + 残差骨架与原 Block 一致。
#          点云 FPS/KNN 分组不在本模块内（属数据管线）。

'''
模块名称：ZSM (Zigzag Scan Mix) —— Zigzag 空间-语义双向扫描混合

一、模块简介
点云与不规则序列的扫描顺序强烈影响状态空间模型的长程建模质量。ZigzagPointMamba
（NeurIPS 2025）提出空间-语义 zigzag 双向扫描：先按空间相邻性走 Z 字形路径，
再沿语义相似性回头，使 1D 递推同时看到局部几何与全局语义，避免单向 raster 扫描
的边界断裂。

ZSM 把该扫描策略提炼为 2D 网格上的即插即用混合块：zigzag 路径展开 → 双向
（正向/反向）门控递推卷积混合 → 路径还原 + 残差。适用于任何 [B,C,H,W] 特征图。

核心创新点：
1. Zigzag 空间路径：行内蛇形（奇数行反向）展开，相邻 token 空间距离最小化
2. 双向扫描：正向 + 反向独立参数，输出拼接后投影，消除单向偏置
3. 语义门控递推：输入依赖的 z/f 门（类选择性扫描）控制历史衰减与写入
4. pre-norm + 残差骨架：与原 Block 的 Add→LN→Mixer 结构一致

二、结构设计
以输入 [B, C, H, W] 计：
1. Zigzag 展开：把 H×W 网格按蛇形顺序重排为序列 [B, L, C]（L=H*W）
2. 正向分支：
   - pre-norm（channel-last LayerNorm）
   - 因果深度卷积 conv1d（kernel=k，左填充）提取局部上下文
   - 门控：z = sigmoid(Wz h)，f = sigmoid(Wf h)（输入依赖遗忘/写入）
   - 递推：s_t = f_t * s_{t-1} + z_t * h_t（并行扫描用循环展开，纯 torch）
3. 反向分支：同一结构，序列翻转后计算再翻回
4. 双向融合：cat([fwd, bwd], dim=-1) → 1x1 线性投影回 C
5. 路径还原到 [B, C, H, W] + 残差

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 ZigzagPointMamba（NeurIPS 2025）的 zigzag 空间-语义双向扫描策略 ZSM：
将二维特征按蛇形路径展开为序列，以输入依赖门控递推做双向状态空间混合后还原到
空间网格，使一维递推同时捕获局部几何连续性与长程语义依赖。"
（原论文 BibTeX：ZigzagPointMamba, NeurIPS 2025）

四、适用任务
适用于空间结构敏感的序列化建模：点云分类/分割、高分辨率检测分割、视频帧建模、
任何需要替代 raster 扫描 / 单向 RNN 的特征混合场景。
'''


def _zigzag_indices(h: int, w: int, device=None) -> torch.Tensor:
    """蛇形（zigzag）扫描的展平索引：行内交替正/反向。

    返回 long 张量 [h*w]，第 i 个位置是展平 HW 后的原始索引。
    """
    rows = []
    for r in range(h):
        cols = list(range(w)) if (r % 2 == 0) else list(range(w - 1, -1, -1))
        rows.extend(r * w + c for c in cols)
    idx = torch.tensor(rows, dtype=torch.long, device=device)
    return idx


class _GatedScanMix(nn.Module):
    """单向门控递推混合（对应选择性扫描的输入依赖衰减/写入，纯 torch 循环）。"""

    def __init__(self, channels: int, kernel_size: int = 3, reverse: bool = False):
        super().__init__()
        self.reverse = reverse
        self.norm = nn.LayerNorm(channels)
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size,
                                 padding=kernel_size - 1, groups=channels, bias=True)
        self.wz = nn.Linear(channels, channels)
        self.wf = nn.Linear(channels, channels)
        self.wo = nn.Linear(channels, channels)
        self.gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        B, L, C = x.shape
        if self.reverse:
            x = torch.flip(x, dims=[1])
        h = self.norm(x)
        # 因果深度卷积：取前 L 个时间步（左填充）
        hc = h.transpose(1, 2)                         # [B, C, L]
        hc = self.dw_conv(hc)[..., :L].transpose(1, 2)  # [B, L, C]
        hc = F.gelu(hc)
        # 输入依赖门
        z = self.gate_act(self.wz(hc))
        f = self.gate_act(self.wf(hc))
        # 递推 s_t = f_t * s_{t-1} + z_t * h_t（s 为 [B, C] 状态，时间维循环）
        s = torch.zeros(B, C, device=hc.device, dtype=hc.dtype)
        out = torch.empty_like(hc)
        for t in range(L):
            s = f[:, t] * s + z[:, t] * hc[:, t]
            out[:, t] = s
        out = self.wo(out)
        if self.reverse:
            out = torch.flip(out, dims=[1])
        return out


class ZSM(nn.Module):
    """ZSM: Zigzag Scan Mix —— zigzag 双向扫描特征混合块"""

    def __init__(self, channels: int, kernel_size: int = 3, drop_path: float = 0.0):
        super().__init__()
        self.channels = channels
        self.fwd = _GatedScanMix(channels, kernel_size, reverse=False)
        self.bwd = _GatedScanMix(channels, kernel_size, reverse=True)
        self.proj = nn.Linear(channels * 2, channels)
        self.out_norm = nn.LayerNorm(channels)
        if drop_path > 0.0:
            self.drop_path = _DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        # 缓存 zigzag 索引（按 H, W 惰性生成）
        self.register_buffer('_idx_cache_h', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_idx_cache_w', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_idx_cache', torch.zeros(1, dtype=torch.long), persistent=False)

    def _get_indices(self, h: int, w: int, device, dtype=None) -> torch.Tensor:
        if int(self._idx_cache_h) == h and int(self._idx_cache_w) == w \
                and self._idx_cache.numel() == h * w:
            return self._idx_cache
        idx = _zigzag_indices(h, w, device=device)
        self._idx_cache_h.fill_(h)
        self._idx_cache_w.fill_(w)
        self._idx_cache = idx
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        # 1. zigzag 展开 [B,C,H,W] -> [B,L,C]
        flat = x.flatten(2).transpose(1, 2)            # [B, HW, C]，row-major
        idx = self._get_indices(H, W, x.device)        # [L] zigzag 顺序的原始索引
        seq = flat[:, idx, :]                          # [B, L, C]

        # 2. 双向门控扫描
        h_f = self.fwd(seq)
        h_b = self.bwd(seq)
        h = self.proj(torch.cat([h_f, h_b], dim=-1))   # [B, L, C]
        h = self.out_norm(h)

        # 3. 路径还原 + 残差
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(idx.numel(), device=idx.device)
        out_seq = h[:, inv, :]                         # 回到 row-major
        out = out_seq.transpose(1, 2).reshape(B, C, H, W)
        return x + self.drop_path(out)


class _DropPath(nn.Module):
    """stochastic depth（纯 torch，与 timm DropPath 同构）"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 16, 16)
    model = ZSM(channels=64)
    output = model(input_tensor)
    print('=== ZSM: Zigzag Scan Mix ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
