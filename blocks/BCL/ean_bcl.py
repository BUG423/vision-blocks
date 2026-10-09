import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Beyond Self-Attention: External Attention Using Two Linear Layers for Visual Tasks (IEEE TPAMI 2023)
# 论文链接：https://arxiv.org/abs/2105.02358
# 模块提出：EANet (Guo et al., TPAMI 2023) 的 BCL 时序适配版
# 日期：2026-10-09

'''
模块名称：EAN-BCL (External Attention Network - BCL) —— 外部注意力模块（BCL版）

一、模块简介
本模块是将 EANet 的外部注意力机制适配至 BCL 格式一维时序数据（[B, C, T]）的版本。
在时序数据中，样本间往往具有相似的模式规律（如传感器周期性心跳、金融行情相似态势）。
EAN-BCL 利用两组全局共享的可学习记忆单元 M_k 与 M_v 作为跨样本的时序先验词典。
通过双重归一化机制（Double Normalization：沿时间维度 Softmax + 沿记忆维度 L1 归一化），
以线性计算复杂度 O(S * T) 实现高质量时序特征增强。

二、结构设计
EAN-BCL 由以下子结构组成（输入 [B, C, T]）：
1. 投影层：Conv1d(C, C, 1) 对输入特征进行预处理；
2. 记忆单元交互（Memory Unit 0）：
   - Linear_0（Conv1d(C, S, 1, bias=False)）计算输入时序特征与外置字典 M_k 的相关度，得到 [B, S, T]；
3. 双重归一化（Double Normalization）：
   - 第一级：Softmax(attn, dim=-1)，在时间维度 T 归一化；
   - 第二级：attn / (1e-9 + attn.sum(dim=1, keepdim=True))，在记忆维度做 L1 归一化；
4. 记忆单元重构（Memory Unit 1）：
   - Linear_1（Conv1d(S, C, 1, bias=False)）利用外置字典 M_v 聚合重构特征，恢复至 [B, C, T]；
5. 投影与残差激活：
   - Conv1d(C, C, 1) + BatchNorm1d(C) 投影精炼；
   - 与输入 x 残差相加后经 ReLU 激活输出。
'''


class EAN_BCL(nn.Module):
    """EAN-BCL: External Attention Network for BCL time-series format"""

    def __init__(self, channels: int, seq_len: int = 128, memory_size: int = 64):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.memory_size = memory_size

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.linear_0 = nn.Conv1d(channels, memory_size, kernel_size=1, bias=False)
        self.linear_1 = nn.Conv1d(memory_size, channels, kernel_size=1, bias=False)

        # 遵循原论文初始化策略：M_v 初始为 M_k 的转置
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2).clone()

        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, C, T]
        out: [B, C, T]
        """
        idn = x
        x_proj = self.conv1(x)

        # 记忆单元 0 与双重归一化
        attn = self.linear_0(x_proj)  # [B, S, T]
        attn = F.softmax(attn, dim=-1)  # 时间维度归一化
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # 记忆维度 L1 归一化

        # 记忆单元 1 聚合
        out = self.linear_1(attn)  # [B, C, T]
        out = self.conv2(out)
        out = self.relu(out + idn)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = EAN_BCL(channels=64, seq_len=128)
    print('EAN-BCL:', model(x).shape)
