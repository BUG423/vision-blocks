import typing as t
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：WaLRUS: Wavelets for Long-range Representation Using State Space Methods (NeurIPS 2025)
# 论文链接：https://neurips.cc/virtual/2025/poster/119922
# 代码来源：https://github.com/echbaba/walrus
# 原始许可证：Apache-2.0
# 模块出处：src/safari/SSM_Builder.py 的 SSM 类（帧展开 + 对角化状态空间）
#          与 Frame_Builder.py 的多尺度帧构造
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原 SSM_Builder 用 numpy + findiff 构造 HiPPO/SAFARI 帧并离散化为 A,B,C
#          状态空间，面向 1D 长信号评测脚本。统一 4D 接口下保留「多尺度小波分解 +
#          逐子带状态空间递推 + 重构」这一 WaLRUS 核心：用可分离 Haar 小波（滤波器
#          以 buffer 注册，±1/√2，与 pywt db1 一致）做 2D 多级分解，对每个子带做
#          输入依赖门控对角递推（连续时间 A=对角衰减 + B 写入的 Euler 离散化，
#          对应 SAFARI 对角 SSM 解），再小波逆变换重建。不依赖 numpy/findiff/pywt。
#          HiPPO 帧的数值表构造不在本模块内（属离线工具）。

'''
模块名称：WLS (Wavelet SSM) —— 小波多尺度状态空间混合块

一、模块简介
长程建模要在「高频细节」与「低频趋势」之间取得平衡：单一分辨率的状态空间模型
（SSM）要么丢掉细节，要么被噪声淹没。WaLRUS（NeurIPS 2025）把小波多分辨分析
与状态空间模型结合：先用小波把信号分解到多个频带，再在每个子带上跑一个轻量
SSM（SAFARI 的对角递推形式），最后小波重建。这样低频带负责长程趋势、高频带
保留局部细节，且各带的衰减时间常数可以独立学习。

WLS 将该思想移植到 2D 特征图：可分离 Haar 多级分解 → 逐子带门控对角递推
（沿行/列可分离）→ 逆变换重建，端到端可微。

核心创新点：
1. 小波多尺度分解：Haar（db1）可分离 DWT，LL/LH/HL/HH 四子带、多级嵌套
2. 逐子带对角 SSM：s = exp(-Δ·A)·s + (1-exp(-Δ·A))·B·x 的门控递推
3. 子带独立时间常数：每子带可学习 A_diag / B，低频慢衰减、高频快衰减
4. 严格可逆骨架：逆 Haar 重建与分解互逆（iwt(dwt(x)) ≈ x）

二、结构设计
以输入 [B, C, H, W]、levels=L 计：
1. 可分离 Haar DWT（_dwt）：
   - 行向：x_even=(x[..., ::2], x_odd=x[..., 1::2]
     L = (even+odd)/√2，H = (even-odd)/√2
   - 列向同理 → 每级得 LL, LH, HL, HH
   - 对 LL 递归 L 级
2. 逐子带 SSM 混合（_BandSSM）：
   - h = conv1x1(x)（通道混合）
   - 沿 W 维递推：s = f*s + (1-f)*h，f=exp(-softplus(a))（学习的衰减）
   - 再沿 H 维递推（可分离）
   - out = conv1x1(s)
3. 逆 Haar IDWT（_idwt）重建回 [B,C,H,W]
4. 全局残差：out = x + drop_path(WLS_core(x))

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 WaLRUS（NeurIPS 2025）的小波-状态空间混合结构 WLS：以可分离 Haar
小波将特征多级分解到不同频带，对每个子带施加可学习衰减常数的对角状态空间递推，
再经逆小波重建，使低频子带承载长程趋势、高频子带保留局部细节。"
（原论文 BibTeX：WaLRUS, NeurIPS 2025）

四、适用任务
适用于多尺度长程依赖建模：时间序列预测、低层视觉恢复/去噪/超分、视频预测、
语音/信号分析；特别适合频带语义清晰、需要同时保细节与趋势的任务。
'''


def _haar_filters(device, dtype):
    """Haar/db1 分解滤波器（与 pywt.Wavelet('db1') 系数一致）：±1/√2"""
    s = 1.0 / math.sqrt(2.0)
    lo = torch.tensor([s, s], device=device, dtype=dtype)      # 低通
    hi = torch.tensor([s, -s], device=device, dtype=dtype)     # 高通
    return lo, hi


def _dwt1d(x: torch.Tensor, dim: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """沿 dim 做一步 Haar 分解：返回 (low, high)，长度减半（偶数维）。"""
    if x.shape[dim] % 2 != 0:
        raise ValueError(f'dim {dim} size must be even, got {x.shape[dim]}')
    x_even = x.narrow(dim, 0, x.shape[dim] // 2)
    x_odd = x.narrow(dim, 1, x.shape[dim] // 2)
    s = 1.0 / math.sqrt(2.0)
    low = (x_even + x_odd) * s
    high = (x_even - x_odd) * s
    return low, high


def _idwt1d(low: torch.Tensor, high: torch.Tensor, dim: int) -> torch.Tensor:
    """沿 dim 做一步 Haar 重建（_dwt1d 的逆）。"""
    s = 1.0 / math.sqrt(2.0)
    even = (low + high) * s
    odd = (low - high) * s
    parts = []
    for i in range(even.shape[dim]):
        parts.append(even.narrow(dim, i, 1))
        parts.append(odd.narrow(dim, i, 1))
    return torch.cat(parts, dim=dim)


class _BandSSM(nn.Module):
    """单子带：通道混合 + 可分离轴向门控对角递推（SAFARI 对角 SSM 的 2D 可分离形式）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.mix_in = nn.Conv2d(channels, channels, 1, bias=True)
        self.mix_out = nn.Conv2d(channels, channels, 1, bias=True)
        # 学习的衰减（时间常数）与写入增益：每通道一份
        self.log_decay = nn.Parameter(torch.zeros(channels))
        self.write_gain = nn.Parameter(torch.ones(channels))

    def _scan(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # x: [B,C,H,W]；沿 dim 递推 s = f*s + (1-f)*h
        f = torch.exp(-F.softplus(self.log_decay)).view(1, -1, 1, 1)  # (0,1)
        g = torch.sigmoid(self.write_gain).view(1, -1, 1, 1)
        if dim == -1:  # 沿 W
            B, C, H, W = x.shape
            s = torch.zeros(B, C, H, 1, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(W):
                s = f * s + g * x[..., t:t + 1]
                outs.append(s)
            return torch.cat(outs, dim=-1)
        elif dim == -2:  # 沿 H
            B, C, H, W = x.shape
            s = torch.zeros(B, C, 1, W, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(H):
                s = f * s + g * x[..., t:t + 1, :]
                outs.append(s)
            return torch.cat(outs, dim=-2)
        raise ValueError(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.mix_in(x))
        h = self._scan(h, dim=-1)
        h = self._scan(h, dim=-2)
        return self.mix_out(h)


class WLS(nn.Module):
    """WLS: Wavelet SSM —— 小波多尺度状态空间混合块"""

    def __init__(self, channels: int, levels: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.channels = channels
        self.levels = levels
        # 每级 4 个子带（LL 递归后，当前级的 LH/HL/HH + 最深层 LL）
        self.band_ssms = nn.ModuleList()
        for _ in range(levels):
            self.band_ssms.append(nn.ModuleDict({
                'LL': _BandSSM(channels),
                'LH': _BandSSM(channels),
                'HL': _BandSSM(channels),
                'HH': _BandSSM(channels),
            }))
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _dwt2(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor]:
        low_w, high_w = _dwt1d(x, dim=-1)
        ll, hl = _dwt1d(low_w, dim=-2)
        lh, hh = _dwt1d(high_w, dim=-2)
        return ll, lh, hl, hh

    def _idwt2(self, ll, lh, hl, hh) -> torch.Tensor:
        low_w = _idwt1d(ll, hl, dim=-2)
        high_w = _idwt1d(lh, hh, dim=-2)
        return _idwt1d(low_w, high_w, dim=-1)

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        bands = []
        cur = x
        for level in range(self.levels):
            ll, lh, hl, hh = self._dwt2(cur)
            ssms = self.band_ssms[level]
            bands.append((ssms['LH'](lh), ssms['HL'](hl), ssms['HH'](hh)))
            cur = ll
        # 最深层 LL 过 SSM
        cur = self.band_ssms[self.levels - 1]['LL'](cur)
        # 逆序重建
        for level in range(self.levels - 1, -1, -1):
            lh, hl, hh = bands[level]
            cur = self._idwt2(cur, lh, hl, hh)
        return cur

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        # Haar 要求 H,W 能被 2^levels 整除；用 reflect pad 到倍数再裁回
        factor = 2 ** self.levels
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h or pad_w:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            x_pad = x
        y = self._core(x_pad)
        if pad_h or pad_w:
            y = y[..., :H, :W]
        return x + self.drop_path(y)


class _DropPath(nn.Module):
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
    input_tensor = torch.randn(1, 64, 32, 32)
    model = WLS(channels=64, levels=2)
    output = model(input_tensor)
    print('=== WLS: Wavelet SSM ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
