import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Restormer: Efficient Transformer for High-Resolution Image Restoration (IEEE TPAMI 2022)
# 论文链接：https://arxiv.org/abs/2111.09881
# 模块提出：Restormer (Zamir et al., TPAMI 2022) 的 BCL 时序适配版
# 日期：2026-10-09

'''
模块名称：MDTA-BCL (Multi-DConv Head Transposed Attention - BCL) —— 多深度卷积头转置自注意力（BCL版）

一、模块简介
本模块是将 Restormer 的 MDTA 机制适配至 BCL 格式一维时序数据（[B, C, T]）的版本。
在长序列时序数据建模中，传统自注意力随序列长度 T 呈二次方 O(T^2) 复杂度，在长程信号（心电、工业传感、金融高频行情）中
容易发生显存爆炸。MDTA-BCL 将转置注意力引入时序领域：通过沿时间维度执行 1D 深度卷积提取局部时序动态，并在通道维度
计算交叉转置注意力矩阵 [C, C]，使整体计算复杂度关于时序长度 T 呈线性 O(C^2 T)。

二、结构设计
MDTA-BCL 由以下子结构组成（输入 [B, C, T]）：
1. 投影与时序深度卷积：
   - Conv1d(C, 3C, 1) 生成级联 Q, K, V 特征；
   - Conv1d(3C, 3C, 3, padding=1, groups=3C) 提取局部邻近时序上下文；
   - chunk 切分为 Q, K, V，各 [B, C, T]。
2. 多头重塑与时间归一化：
   - 将 Q, K, V 重排为 [B, num_heads, C_head, T]（C_head = C // num_heads）；
   - 在时间维度 T 上执行 F.normalize(..., dim=-1)。
3. 通道转置注意力与加权：
   - 注意力图 A = Softmax(temperature * (Q @ K^T), dim=-1)，形状 [B, num_heads, C_head, C_head]；
   - 输出特征 out = A @ V，形状 [B, num_heads, C_head, T]。
4. 重塑、投影与残差相加：
   - 重塑为 [B, C, T] 并经 Conv1d(C, C, 1) 输出投影；
   - 与输入 x 残差相加输出。
'''


class MDTA_BCL(nn.Module):
    """MDTA-BCL: Multi-DConv Head Transposed Attention for BCL time-series format"""

    def __init__(self, channels: int, seq_len: int = 128,
                 num_heads: int = 8, bias: bool = False):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(
            channels * 3, channels * 3, kernel_size=3, stride=1, padding=1,
            groups=channels * 3, bias=bias
        )
        self.project_out = nn.Conv1d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, C, T]
        out: [B, C, T]
        """
        b, c, t = x.shape
        head_dim = c // self.num_heads

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重排为多头格式 [B, num_heads, head_dim, T]
        q = q.view(b, self.num_heads, head_dim, t)
        k = k.view(b, self.num_heads, head_dim, t)
        v = v.view(b, self.num_heads, head_dim, t)

        # 沿时间维度归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 通道维转置注意力 [B, num_heads, head_dim, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 加权 Value 并重塑
        out = attn @ v
        out = out.view(b, c, t)
        out = self.project_out(out)
        return out + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = MDTA_BCL(channels=64, seq_len=128)
    print('MDTA-BCL:', model(x).shape)
