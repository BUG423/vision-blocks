import typing as t
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Beyond Self-Attention: External Attention Using Two Linear Layers for Visual Tasks (IEEE TPAMI 2023)
# 论文链接：https://arxiv.org/abs/2105.02358
# 代码来源：https://github.com/MenghaoGuo/EANet
# 原始许可证：The Clear BSD License
# 模块出处：model_torch.py 的 External_attention 类
# 提取者：vision-blocks Agent
# 日期：2026-10-09
# 重构说明：1. 剥离针对具体骨干网络（ResNet/Jittor/SynchronizedBatchNorm）的外部依赖，使用标准 nn.BatchNorm2d；
#          2. 接口对齐规范：参数 c 改为 channels，作为构造函数首参；
#          3. 4D 输入形状保持：[B, C, H, W] -> [B, C, H, W]；
#          4. 核心数值逻辑严格保持一致：两组外置记忆单元（M_k 与 M_v）进行特征交互，
#             双重归一化机制（Double Normalization：空间维度 Softmax + 记忆维度 L1 归一化），
#             记忆字典经 1x1 卷积重构后配合残差连接与 ReLU 激活输出。

'''
模块名称：EAN (External Attention Network) —— 外部注意力模块

一、模块简介
传统自注意力机制计算输入特征内部所有 token 之间的两两交互（Self-Attention），计算复杂度与 token 数量呈二次方
O(N^2) 关系，且忽略了不同样本之间的潜在关联。EANet（IEEE TPAMI 2023）提出了外部注意力（External Attention），
利用两组小型、可学习、跨数据集共享的外置记忆单元（Memory Units）替代原本的自注意力机制。
输入特征首先与第一组记忆单元 M_k 计算相似度，再利用该注意力权重聚合第二组记忆单元 M_v。为了解决 Softmax 对注意力图
尺度敏感的问题，EAN 提出了双重归一化（Double Normalization：先在空间维度做 Softmax，再在记忆维度做 L1 归一化）。
外部注意力计算复杂度关于输入尺寸呈线性关系 O(S * N)（S 为记忆单元尺寸，通常 S << N），同时天然具备跨样本的全局先验建模能力。

核心创新点：
1. 外置记忆单元：使用两个可学习的共享记忆字典 M_k 和 M_v 建模整个数据集的概念先验，而非单一图像内部的自相关；
2. 线性计算复杂度：计算复杂度为 O(S * N)，显著低于标准自注意力的 O(N^2)；
3. 双重归一化机制（Double Normalization）：沿空间维度计算 Softmax，沿记忆单元维度计算 L1 归一化，平衡特征分布；
4. 即插即用：形状严格保持 [B, C, H, W] -> [B, C, H, W]，纯 PyTorch 零外部依赖。

二、结构设计
EAN 由以下子结构组成（输入 [B, C, H, W]）：
1. 输入投影：Conv2d(C, C, 1) 对输入特征进行预处理，展平为 [B, C, N]（N = H*W）；
2. 记忆单元交互（Memory Unit 0）：
   - Linear_0（nn.Conv1d(C, S, 1, bias=False)）计算输入与外置字典 M_k 的相关度，得到 [B, S, N]；
3. 双重归一化（Double Normalization）：
   - 第一级：Softmax(attn, dim=-1)，在空间 token 维度归一化；
   - 第二级：attn / (1e-9 + attn.sum(dim=1, keepdim=True))，在记忆维度做 L1 归一化；
4. 记忆单元聚合（Memory Unit 1）：
   - Linear_1（nn.Conv1d(S, C, 1, bias=False)）利用外置字典 M_v 聚合重构特征，恢复至 [B, C, N]；
5. 空间重塑与残差融合：
   - 重塑为 [B, C, H, W]；
   - Conv2d(C, C, 1) + BatchNorm2d 投影精炼；
   - 与残差分支相加后经 ReLU 激活输出。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 EANet（IEEE TPAMI 2023）提出的外部注意力模块（EAN）：
通过两组可学习的跨样本共享记忆单元与双重归一化机制（Double Normalization），
将自注意力的二次方复杂度降为与图像大小成线性的 O(S·N)，在捕捉全局先验表征的同时兼顾计算与显存效率。"

四、适用任务
适用于图像分类、目标检测、语义分割、点云分析及边缘端高效视觉主干网络。
'''


class EAN(nn.Module):
    """EAN: External Attention Network (TPAMI 2023)"""

    def __init__(self, channels: int, memory_size: int = 64):
        super().__init__()
        self.channels = channels
        self.memory_size = memory_size

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.linear_0 = nn.Conv1d(channels, memory_size, kernel_size=1, bias=False)
        self.linear_1 = nn.Conv1d(memory_size, channels, kernel_size=1, bias=False)

        # 遵循原论文初始化策略：M_v 初始为 M_k 的转置
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2).clone()

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, C, H, W]
        out: [B, C, H, W]
        """
        idn = x
        x_proj = self.conv1(x)

        b, c, h, w = x_proj.size()
        x_flat = x_proj.view(b, c, h * w)  # [B, C, N]

        # 记忆单元 0 与双重归一化
        attn = self.linear_0(x_flat)  # [B, S, N]
        attn = F.softmax(attn, dim=-1)  # 空间维度归一化
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # 记忆维度 L1 归一化

        # 记忆单元 1 聚合与重塑
        out = self.linear_1(attn)  # [B, C, N]
        out = out.view(b, c, h, w)

        out = self.conv2(out)
        out = out + idn
        out = self.relu(out)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = EAN(channels=128)
    output = model(input_tensor)
    print('=== EAN: External Attention Network ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
