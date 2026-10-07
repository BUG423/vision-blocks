import math
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：HRDiT: Training-Free High-Resolution Image Generation with Off-the-Shelf DiT (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2608.07003
# 代码来源：https://github.com/zylwithxy/HRDiT
# 原始许可证：MIT
# 模块出处：hrdit/spa.py 的 _phi / build_bundle_id_variants / set_spa
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原始 SPA 在 DiT 注意力内部把 RoPE 位置 id 经 bundle 映射生成多组滑动变体，
#   对每个变体的注意力输出取平均（Eq.6）。为满足统一 4D 特征接口，本文件保留 _phi 与
#   build_bundle_id_variants 的数学完全不变，类 SPA 额外提供 align_pos_ids（输出对齐后
#   的位置 id 变体，可直接喂给 RoPE）以及 shape-preserving 的 forward(x)：用同一套
#   _phi bundle 逻辑生成对齐的 2D 正弦位置编码，对全部变体取平均（对应论文对 bundle
#   映射取平均），再由可学习 1x1 卷积投影为位置偏置加性调制到特征上（learned positional
#   bias 包装方案）。位置 id 的构造/映射数学不变；新增的仅是把对齐后的位置信息注入
#   特征的轻量包装，并补齐 shape assert。

'''
模块名称：SPA (Spatial Position Alignment) —— 空间位置对齐模块

一、模块简介
高分辨率生成中，直接把原始像素坐标送入 RoPE 会让位置编码超出训练分辨率的
分布，导致生成质量下降。SPA 的核心思想是：把连续的行列位置 id 经“束映射”
(bundle mapping) _phi 折叠为少量粗粒度束索引，并让束边界滑动产生多组等价的
位置划分，再对这些划分下的模型输出取平均。由于束索引保持在训练时的取值范围
内，RoPE 始终处于 in-distribution 状态，从而实现免训练的高分辨率生成。

核心创新点：
1. 束映射 _phi：把 x < n1 的位置映射为束 0，其余映射为 ceil((x+1-n1)/size)
2. 边界滑动变体：固定束宽 s，滑动首束边界 n1，得到一组互补的束划分
3. 多变体平均：对每个束划分下的输出取均值，等价于平均注意力图（Eq.6）
4. 与 RoPE 解耦：只改位置 id，不改特征/权重，可即插即用

二、结构设计
SPA 由以下子结构组成：
1. 位置 id 生成（align_pos_ids）：
   - 生成 H×W 网格的位置 id [H*W, 3]（列 1 为行号，列 2 为列号）
   - 经 build_bundle_id_variants 得到多组束索引 id 变体
2. 束映射 _phi(x, n1, size)：
   - x < n1 → 0，否则 → ceil((x + 1 - n1) / size)（整数上取整）
3. 变体构造 build_bundle_id_variants：
   - s_row = max(1, ceil(max_row / (group_num - 1)))，s_col 同理
   - 基准变体 variant(s_row, s_col)
   - 行边界滑动 variant(n, s_col), n = 1..s_row-1
   - 列边界滑动 variant(s_row, m), m = 1..s_col-1
4. forward 的位置偏置包装（统一接口用）：
   - 对每个变体的束索引做 2D 正弦编码 [H*W, C]
   - 对变体取平均 → [1, C, H, W]
   - 1x1 卷积投影 + 残差调制：out = x + bias

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 SPA（Spatial Position Alignment）对位置编码做束对齐：将连续位置 id
经束映射折叠为粗粒度束索引，并对束边界滑动产生的多组划分下的输出取平均，使
RoPE 始终处于训练分布内，从而支持免训练的高分辨率生成（HRDiT, ECCV 2026）。"

原论文引用格式：HRDiT: Training-Free High-Resolution Image Generation with
Off-the-Shelf DiT, ECCV 2026.

四、适用任务
高分辨率图像生成（免训练放大）、DiT/Transformer 位置编码对齐、任意需要把
细粒度坐标折叠为粗粒度束索引的多尺度位置建模场景。
'''


def _phi(x: torch.Tensor, n1: int, size: int) -> torch.Tensor:
    """Bundle mapping: 0 for x < n1, else ceil((x + 1 - n1) / size)."""
    return torch.where(x < n1, torch.zeros_like(x), (x + 1 - n1 + size - 1) // size)


def build_bundle_id_variants(img_ids: torch.Tensor, group_num: int) -> t.List[torch.Tensor]:
    """Build the bundle-index variants of ``img_ids`` used by SPA.

    ``img_ids`` is [T_img, 3] with column 1 the row index and column 2 the
    column index. ``group_num`` sets how many bundles each axis is divided into.
    Returns one variant per sliding position of the bundle boundaries.
    """
    rows = img_ids[:, 1].long()
    cols = img_ids[:, 2].long()
    s_row = max(1, math.ceil(rows.max().item() / (group_num - 1)))
    s_col = max(1, math.ceil(cols.max().item() / (group_num - 1)))

    def variant(n1_row: int, n1_col: int) -> torch.Tensor:
        ids = img_ids.clone()
        ids[:, 1] = _phi(rows, n1_row, s_row).to(img_ids.dtype)
        ids[:, 2] = _phi(cols, n1_col, s_col).to(img_ids.dtype)
        return ids

    variants = [variant(s_row, s_col)]
    variants += [variant(n, s_col) for n in range(1, s_row)]
    variants += [variant(s_row, m) for m in range(1, s_col)]
    return variants


class SPA(nn.Module):
    """SPA: Spatial Position Alignment —— 空间位置对齐模块"""

    def __init__(
        self,
        channels: int,
        group_num: int = 80,
        num_variants: int = 8,
        spa_on: bool = True,
    ):
        super().__init__()
        assert channels > 0, 'channels must be positive'
        assert group_num > 1, 'group_num must be > 1'
        self.channels = channels
        self.group_num = group_num
        # 前向最多使用的变体数上限；<=0 表示使用全部滑动变体（与原实现一致）
        self.num_variants = num_variants
        self.spa_on = spa_on
        # 把对齐后的位置编码投影为通道级位置偏置
        self.pos_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    # ------------------------------------------------------------------ #
    # 位置 id：与原实现一致的 bundle 数学
    # ------------------------------------------------------------------ #
    def grid_pos_ids(self, h: int, w: int, device: t.Optional[torch.device] = None) -> torch.Tensor:
        """Generate [H*W, 3] position ids for an H×W grid (col1=row, col2=col)."""
        rows = torch.arange(h, device=device, dtype=torch.float32)
        cols = torch.arange(w, device=device, dtype=torch.float32)
        row_grid = rows.view(h, 1).expand(h, w).reshape(-1)
        col_grid = cols.view(1, w).expand(h, w).reshape(-1)
        ids = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        ids[:, 1] = row_grid
        ids[:, 2] = col_grid
        return ids

    def align_pos_ids(
        self,
        h: int,
        w: int,
        device: t.Optional[torch.device] = None,
    ) -> t.List[torch.Tensor]:
        """Return SPA bundle-id variants of the H×W grid, one tensor [H*W, 3] each."""
        img_ids = self.grid_pos_ids(h, w, device=device)
        return build_bundle_id_variants(img_ids, self.group_num)

    def get_aligned_pos_ids(
        self,
        h: int,
        w: int,
        device: t.Optional[torch.device] = None,
    ) -> t.List[torch.Tensor]:
        """Alias of align_pos_ids."""
        return self.align_pos_ids(h, w, device=device)

    def set_spa(self, on: bool) -> None:
        """Enable or disable SPA modulation in forward."""
        self.spa_on = bool(on)

    # ------------------------------------------------------------------ #
    # 2D 正弦位置编码：作用在 bundle 后的行/列索引上
    # ------------------------------------------------------------------ #
    def _sin_cos_embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [T, 3] -> [T, channels] sinusoidal encoding of bundled row/col."""
        c = self.channels
        device = ids.device
        row = ids[:, 1].float().unsqueeze(1)  # [T, 1]
        col = ids[:, 2].float().unsqueeze(1)  # [T, 1]

        half = (c + 1) // 2
        # 标准正弦位置编码频率
        div = torch.exp(
            torch.arange(0, half, 2, device=device).float()
            * (-math.log(10000.0) / max(half, 1))
        )  # [half/2]

        def _axis_pe(pos: torch.Tensor) -> torch.Tensor:
            pe = torch.zeros(pos.shape[0], half, device=device)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        pe = torch.cat([_axis_pe(row), _axis_pe(col)], dim=1)  # [T, 2*half]
        return pe[:, :c]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        assert x.dim() == 4, f'expect 4D [B,C,H,W], got {tuple(x.shape)}'
        b, c, h, w = x.shape
        assert c == self.channels, f'expect C={self.channels}, got {c}'

        if not self.spa_on:
            return x

        variants = self.align_pos_ids(h, w, device=x.device)
        if self.num_variants and self.num_variants > 0:
            variants = variants[: self.num_variants]

        # 对 bundle 变体取平均（对应论文 Eq.6 对多划分输出取平均）
        pe = torch.stack([self._sin_cos_embed(v) for v in variants], dim=0).mean(dim=0)
        pe = pe.transpose(0, 1).reshape(1, c, h, w)  # [1, C, H, W]
        bias = self.pos_proj(pe)  # [1, C, H, W]
        return x + bias


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SPA(channels=128)
    output = model(input_tensor)
    print('=== SPA: Spatial Position Alignment ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    # 多变体路径：更大分辨率 + 更小 group_num
    big = torch.randn(1, 128, 128, 128)
    model_big = SPA(channels=128, group_num=8, num_variants=0)
    variants = model_big.align_pos_ids(128, 128)
    out_big = model_big(big)
    print('variants:', len(variants), 'variant_shape:', tuple(variants[0].shape))
    print('big_input_size:', big.size(), 'big_output_size:', out_big.size())
