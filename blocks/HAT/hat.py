import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Multi-modality Image Fusion under Adverse Weather (AMG-Fuse) (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2606.26812
# 代码来源：https://github.com/ixilai/AMG-Fuse
# 原始许可证：MIT
# 模块出处：model/amgfuse.py 的 Attention_histogram（L146，DHSA）+ TransformerBlock（L22）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：Attention_histogram 的动态直方图自注意力数学逐行保留：输入半通道
#   沿 H/W 双向排序、v 沿空间维排序并对 q/k 做 gather、双分支 box/交错分组
#   注意力（softmax_1 = exp/(sum+1) 归一化，非标准 softmax，保持不变）、
#   scatter 逆排序、out1*out2 逐元相乘、半通道按 idx_w/idx_h 逆排序。原实现
#   用 einops.rearrange 做 (head c)/(factor hw)/(c factor) 重排，这里改为等价
#   的 view/permute/reshape（einops 语义：括号内左侧为外层维）。TransformerBlock
#   的 norm1-attn-norm2-ffn 双残差结构与 FeedForward（3x3/5x5 双深度卷积分支）
#   逐行保留。为避免就地修改调用方输入，forward 开头 clone 一份（数值不变）。
#   删除未调用的 normalize 死代码。不改动 softmax 维度与归一化方式。

'''
模块名称：HAT (Histogram Attention) —— 直方图注意力模块

一、模块简介
恶劣天气（雨、雾、雪）下的多模态图像融合需要在动态范围剧烈变化的区域
（反光、暗区、噪声）保持稳定的选择性增强。AMG-Fuse 提出动态范围直方图
自注意力（DHSA）：先把半数通道按空间坐标排序，构造出近似“直方图序”的
特征排布，再沿排序后的空间维做分组注意力；同时用两种互补的分组方式
（box 分组与交错分组）各算一路注意力并逐元相乘，使响应同时受两种分组
一致性约束，从而抑制动态范围异常值、增强结构一致性。再配合双核前馈
（3x3/5x5 并行深度卷积）与双残差 TransformerBlock，形成稳健的融合块。

核心创新点：
1. 空间排序直方图化：半通道沿 H、W 双向排序，得到稳定的动态范围序
2. 空间维排序 + gather：对 v 沿空间维排序，q/k 同步 gather，保持对齐
3. 双分组注意力相乘：box 分组与交错分组两路 softmax_1 注意力逐元相乘
4. softmax_1 归一化：exp(x)/(sum(exp)+1)，带保守衰减，区别于标准 softmax
5. 逆排序还原：scatter 按排序索引还原空间/直方图顺序

二、结构设计
HAT 由以下子结构组成（TransformerBlock 结构）：
1. LayerNorm（WithBias，默认）：[B, C, H, W] → 按通道 LayerNorm
2. Attention_histogram（DHSA）：
   - qkv: 1x1 Conv → 3x3 深度卷积 → 5 路 [q1, k1, q2, k2, v]，各 [B, C, H, W]
   - 半通道 H/W 双向排序；v 沿空间维排序，q/k gather 对齐
   - reshape_attn(True/False)：pad 到 factor(=num_heads) 整除 →
     (head c)×(factor hw) 或 (head c)×(hw factor) 重排为 b head (c factor) hw
   - q/k L2 归一化 → attn = (q @ k^T) * temperature → softmax_1 → @v
   - 两路 scatter 逆排序后逐元相乘 → 1x1 投影 → 半通道逆排序
   - 输出 [B, C, H, W]
3. FeedForward：
   - 1x1 投影到 2*hidden → 3x3/5x5 并行深度卷积 + ReLU → 各自再 3x3/5x5
     深度卷积 → cat → 1x1 投影回 C
4. 双残差：x = x + attn(norm1(x))；x = x + ffn(norm2(x))

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 HAT（Histogram Attention）块进行动态范围稳健的特征增强，其核心
为动态范围直方图自注意力（DHSA）：对半数通道做空间排序以构造直方图序，
沿排序后的空间维计算 box 分组与交错分组两路 softmax_1 注意力并逐元相乘，
配合双核前馈与双残差结构，提升恶劣天气多模态融合的稳定性。"

原论文引用格式：Multi-modality Image Fusion under Adverse Weather, ECCV 2026.

四、适用任务
恶劣天气下的多模态图像融合、图像恢复/增强、低光增强、去雨去雾；也可作为
任意视觉骨干中的动态范围稳健注意力块。
'''


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    # 'b c h w -> b (h w) c'
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # 'b (h w) c -> b c h w'
    b, hw, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, LayerNorm_type: str = 'WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class Attention_histogram(nn.Module):
    """Dynamic-range Histogram Self-Attention (DHSA) —— 动态范围直方图自注意力"""

    def __init__(self, dim: int, num_heads: int, bias: bool = True, ifBox: bool = True):
        super().__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x: torch.Tensor, factor: int):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x: torch.Tensor, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def reshape_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ifBox: bool,
    ) -> torch.Tensor:
        """双分组注意力：box 分组（ifBox=True）或交错分组（ifBox=False）。

        重排语义与原 einops 一致：
          ifBox=True : 'b (head c) (factor hw) -> b head (c factor) hw'
          ifBox=False: 'b (head c) (hw factor) -> b head (c factor) hw'
        """
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        n_heads = self.num_heads
        c_head = c // n_heads
        factor = self.factor

        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            if ifBox:
                # (head c) 为 head 外层；(factor hw) 为 factor 外层
                t = t.view(b, n_heads, c_head, factor, hw)
                return t.reshape(b, n_heads, c_head * factor, hw)
            # (hw factor) 为 hw 外层
            t = t.view(b, n_heads, c_head, hw, factor)
            t = t.permute(0, 1, 2, 4, 3).contiguous()
            return t.reshape(b, n_heads, c_head * factor, hw)

        def _from_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(b, n_heads, c_head, factor, hw)
            if ifBox:
                return t.reshape(b, n_heads * c_head, factor * hw)
            t = t.permute(0, 1, 2, 4, 3).contiguous()
            return t.reshape(b, n_heads * c_head, hw * factor)

        q = _to_heads(q)
        k = _to_heads(k)
        v = _to_heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = attn @ v
        out = _from_heads(out)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # clone：避免就地修改调用方输入（原实现直接写入 x[:, :c//2]）
        x = x.clone()
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,h,w

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool = True):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias,
        )
        self.dwconv5x5 = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
            groups=hidden_features * 2, bias=bias,
        )
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(
            hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
            groups=hidden_features, bias=bias,
        )
        self.dwconv5x5_1 = nn.Conv2d(
            hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
            groups=hidden_features, bias=bias,
        )

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)
        return x


class HAT(nn.Module):
    """HAT: Histogram Attention —— 直方图注意力模块"""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.66,
        bias: bool = True,
        LayerNorm_type: str = 'WithBias',
    ):
        super().__init__()
        assert channels > 0, 'channels must be positive'
        assert channels % num_heads == 0, 'channels must be divisible by num_heads'
        assert channels % 2 == 0, 'channels must be even (half-channel sort)'
        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = LayerNorm(channels, LayerNorm_type)
        self.attn = Attention_histogram(channels, num_heads, bias)
        self.norm2 = LayerNorm(channels, LayerNorm_type)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        assert x.dim() == 4, f'expect 4D [B,C,H,W], got {tuple(x.shape)}'
        assert x.shape[1] == self.channels, f'expect C={self.channels}, got {x.shape[1]}'
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = HAT(channels=128)
    output = model(input_tensor)
    print('=== HAT: Histogram Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
