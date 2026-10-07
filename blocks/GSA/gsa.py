import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Global Regulation and Excitation via Attention Tuning for Stereo Matching (GREAT-Stereo) (ICCV 2025)
# 论文链接：https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Global_Regulation_and_Excitation_via_Attention_Tuning_for_Stereo_Matching_ICCV_2025_paper.pdf
# 代码来源：https://github.com/JarvisLee0423/GREAT-Stereo
# 原始许可证：Apache-2.0
# 模块出处：models/great_stereo/transformers.py 的 AttentionLayer 类（含 to_3d/to_4d 辅助函数）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原 AttentionLayer 为左右特征图之间的交叉注意力；本块按统一接口
# 暴露为单输入自注意力（feats1=feats2=x，注意力公式逐算子一致）。去除 einops，
# 以 view/permute 等价实现 to_3d/to_4d 与多头切分/合并；保留 sink_competition
# 全局竞争重归一化（原文 TransformerBlock 中启用）、scale=(C/h)^{-1/2}、
# softmax 维度（dim=-1）与 eps 常数不变。VolumeAttention（代价体相关）不提取。

'''
模块名称：GSA (GREAT Spatial Attention) —— GREAT 空间注意力模块

一、模块简介
GREAT-Stereo 面向立体匹配任务提出"全局调节与激励"的注意力调制机制：
标准注意力在长序列（高分辨率视差搜索）下容易出现注意力分布塌缩，缺乏
全局竞争约束。GSA 提取其空间注意力核（AttentionLayer）：通过可学习的
Q/KV 1x1 投影构造多头注意力，并引入 sink_competition（全局竞争）机制——
softmax 后加 eps 并按行重新归一化，强制不同空间位置之间形成竞争，抑制
某一位置独占注意力质量，从而得到更平滑、全局一致的注意力分布。该模块
以纯张量操作实现空间注意力（内部 to_3d/to_4d 在 4D 与 3D 序列间转换），
不含代价体相关算子，可即插即用于 2D 特征图。

二、结构设计
输入 [B, C, H, W]，输出 [B, C, H, W]（内部 to_3d：[B,H,W,C] → [B*H, W, C]）：
1. 预归一化（pre_norm=True，原文默认）：
   - LayerNorm(in_channels) 作用于展平后的空间序列（norm 与 norm_context
     各自独立，对应原文交叉注意力的两个输入；自注意力形态下两者同源）
2. 线性投影：
   - to_q: 1x1 Conv(in → hidden)；to_kv: 1x1 Conv(in → 2*hidden)
   - query 来自 feats1，key/value 来自 feats2（自注意力时均为 x）
   - chunk(2, dim=-1) 切出 key、value
3. 多头切分：[bh, w, n*c] → [bh*n, w, c]，scale = (hidden // num_heads)^{-1/2}
4. 相似度与注意力：
   - similarity = Q K^T * scale（einsum "bid,bjd->bij"，沿 w 维）
   - sink_competition=True（原文 TransformerBlock 配置）：
     raw_attn = softmax(sim, -1) + eps；raw_attn /= sum(raw_attn, -1, keepdim)
   - 否则 raw_attn = softmax(sim, -1)
5. 聚合输出：out = attn · V；多头合并 → 1x1 Conv(hidden → out)
6. 后归一化（pre_norm=False 时对 3D 序列 LayerNorm）；to_4d 还原 [B, C, H, W]

三、论文写法参考
若在论文中引用该模块，可描述为："We adopt the spatial attention layer from
GREAT-Stereo (ICCV 2025) with global regulation (sink competition): after
softmax an eps-shifted renormalization enforces global competition among
spatial positions."并引用：Global Regulation and Excitation via Attention
Tuning for Stereo Matching, ICCV 2025.

四、适用任务
立体匹配（原文场景）、光流/深度估计等需要全局一致注意力分布的任务；也
可作为通用空间注意力块用于分类、检测、分割主干中需要全局建模的位置。
'''


def to_3d(inputs: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """[B, C, H, W] -> [B*H, W, C]（dim=1）或 [B, H*W, C]（dim=2）"""
    b, c, h, w = inputs.shape
    x = inputs.permute(0, 2, 3, 1).contiguous()
    if dim == 1:
        return x.view(b * h, w, c)
    return x.view(b, h * w, c)


def to_4d(inputs: torch.Tensor, b: int, h: int, w: int = None,
          dim: int = 1) -> torch.Tensor:
    """[B*H, W, C]（dim=1）或 [B, H*W, C]（dim=2）-> [B, C, H, W]"""
    if dim == 1:
        bh, ww, c = inputs.shape
        return inputs.view(b, h, ww, c).permute(0, 3, 1, 2).contiguous()
    bb, hw, c = inputs.shape
    return inputs.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def _split_heads(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[bh, w, n*c] -> [bh*n, w, c]（等价于 einops 'bh w (n c) -> (bh n) w c'）"""
    bh, w, nc = t.shape
    c = nc // num_heads
    return t.view(bh, w, num_heads, c).permute(0, 2, 1, 3).contiguous().view(
        bh * num_heads, w, c)


def _merge_heads(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[bh*n, w, c] -> [bh, w, n*c]（等价于 einops '(bh n) w c -> bh w (n c)'）"""
    bn, w, c = t.shape
    bh = bn // num_heads
    return t.view(bh, num_heads, w, c).permute(0, 2, 1, 3).contiguous().view(
        bh, w, num_heads * c)


class GSA(nn.Module):
    """GSA: GREAT Spatial Attention —— GREAT 空间注意力（全局竞争）模块"""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0,
                 pre_norm: bool = True, sink_competition: bool = True,
                 qkv_bias: bool = True, eps: float = 1e-6):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.eps = eps
        self.pre_norm = pre_norm
        self.sink_competition = sink_competition
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        self.norm = nn.LayerNorm(channels, eps=eps)
        self.norm_context = nn.LayerNorm(channels, eps=eps) if pre_norm else None

        self.to_q = nn.Conv2d(channels, channels, kernel_size=1, bias=qkv_bias)
        self.to_kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=qkv_bias)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        b, c, h, w = x.shape
        if self.pre_norm:
            feats1 = to_4d(self.norm(to_3d(x)), b, h)
            feats2 = to_4d(self.norm_context(to_3d(x)), b, h)
        else:
            feats1 = feats2 = x

        query = to_3d(self.to_q(feats1))
        key, value = to_3d(self.to_kv(feats2)).chunk(2, dim=-1)

        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)

        similarity_matrix = torch.einsum("bid, bjd -> bij", query, key) * self.scale

        if self.sink_competition:
            raw_attn = F.softmax(similarity_matrix, dim=-1) + self.eps
            raw_attn = raw_attn / torch.sum(raw_attn, dim=(-1,), keepdim=True)
        else:
            raw_attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(raw_attn)

        out = torch.einsum("bij, bjd -> bid", attn, value)
        out = _merge_heads(out, self.num_heads)
        out = to_3d(self.to_out(to_4d(out, b, h)))
        if not self.pre_norm:
            out = self.norm(out)
        out = to_4d(out, b, h)

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = GSA(channels=128)
    output = model(input_tensor)
    print('=== GSA: GREAT Spatial Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
