import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：An Efficient Hybrid Vision Transformer for TinyML Applications (TinyNeXt) (ICCV 2025)
# 论文链接：https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf
# 代码来源：https://github.com/yuffeenn/TinyNeXt
# 原始许可证：MIT
# 模块出处：classification/models/modules.py 的 FormerBlock / Attention / Mlp 类
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：去除 einops 与 Add/MatMul/Mul 包装类（仅运算符封装，数值等价）；
# 将单头精简注意力推广为 num_heads 头（默认 num_heads=1 时与原文逐算子一致：
# scale=C^{-1/2}，K 与输入共享无投影、仅 Q/V 有线性投影）；nn.Linear 作用于
# [B,N,C] 的用法与原实现一致。前向数值逻辑不变。

'''
模块名称：TFB (TinyNeXt FormerBlock) —— TinyNeXt 精简注意力前馈块

一、模块简介
TinyML 场景下，Transformer 块中的标准自注意力存在大量访存密集操作
（Q/K/V 三组投影、大中间张量），在微控制器级别的算力与内存上难以部署。
TinyNeXt 提出 FormerBlock：用"精简注意力"替代标准注意力——去掉 Key 投影
（直接以输入特征为 Key），仅保留 Q、V 两个线性投影，将矩阵乘法从 4 次
降到 2 次，显著减少参数量、计算量与内存占用，同时保留全局建模能力。
在此基础上，块内再串联一个深度可分离 3x3 卷积分支补足局部信息，以及
一个瓶颈 MLP 分支提供通道混合，三者均采用残差连接。整个块无 LayerNorm、
仅用 BatchNorm，对量化与低比特推理友好，适合 TinyML 分类/检测部署。

二、结构设计
输入 [B, C, H, W]，输出 [B, C, H, W]：
1. 精简注意力分支（残差）：
   - BatchNorm2d(C)
   - 展平为 [B, N, C]（N = H*W）
   - Q = Linear(C→C)(x)，K = x（无投影），V = Linear(C→C)(x)
   - attn = softmax(Q K^T * C^{-1/2}) ∈ [B, N, N]
   - out = attn · V → 还原 [B, C, H, W]，x = x + out
   - num_heads>1 时按头切分，每头维度 d=C/h，scale=d^{-1/2}（默认 1 头与原文一致）
2. 局部卷积分支（残差）：3x3 深度可分离卷积（groups=C），x = x + dw(x)
3. MLP 分支（残差）：
   - BatchNorm2d(C)
   - 1x1 Conv → GELU → 1x1 Conv（隐层为 ratio*C，默认 ratio=2.0）
   - x = x + mlp(x)

三、论文写法参考
若在论文中引用该模块，可描述为："We adopt the FormerBlock from TinyNeXt
(ICCV 2025), a lean attention block that removes the key projection and keeps
only query/value linear maps, reducing memory-bound operations for TinyML."
并引用：An Efficient Hybrid Vision Transformer for TinyML Applications,
ICCV 2025.

四、适用任务
面向 TinyML 的图像分类、关键词检测、微型目标检测等资源受限任务；也可
作为轻量注意力块嵌入任意 CNN/混合主干，适合对参数量、内存带宽敏感的
边缘部署场景。
'''


class TFB(nn.Module):
    """TFB: TinyNeXt FormerBlock —— 精简注意力前馈块（去 Key 投影）"""

    def __init__(self, channels: int, num_heads: int = 1, mlp_ratio: float = 2.0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_attn = nn.BatchNorm2d(channels)
        # 精简注意力：仅 Q、V 两个投影，K 直接取输入（原文数学不变）
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)

        self.local = nn.Conv2d(channels, channels, 3, 1, 1, 1, groups=channels,
                               bias=False)

        hidden = int(mlp_ratio * channels)
        self.norm_mlp = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, N, C]（与原 view+transpose 顺序一致）
        x = x.view(B, C, -1).transpose(-1, -2).contiguous()
        q = self.q_proj(x)
        k = x
        v = self.v_proj(x)
        if self.num_heads > 1:
            N = q.shape[1]
            q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale,
                                 dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, N, C)
        else:
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale,
                                 dim=-1)
            out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).view(B, C, H, W).contiguous()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"channel mismatch: expect {self.channels}, got {C}"
        x = x + self._attention(self.norm_attn(x))
        x = x + self.local(x)
        x = x + self.mlp(self.norm_mlp(x))
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = TFB(channels=128)
    output = model(input_tensor)
    print('=== TFB: TinyNeXt FormerBlock ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
