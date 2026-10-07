import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：SAM+D: Parameter-Efficient Dimensional Lifting of SAM via Depth-Routed LoRA and Depth Shifting (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2607.29033
# 代码来源：https://github.com/JerrySongCST/SAM-Plus-D
# 原始许可证：MIT (Copyright (c) 2026 Yu Song)
# 模块出处：modeling/dsm.py 的 depth_shift 函数（L4-31，DSM 中的深度移位算子）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原函数输入 (D,H,W,C) 的 3D 医学体数据，沿切片维 D 做"前向邻域移位 +
#   后向邻域移位 + 其余通道保留"的零参数通道移位（边界切片保留自身特征，非零填充
#   非循环回绕）。统一 2D 接口下将特征图的 H 维视作"深度/切片"维，把同一套
#   shift-and-merge 数值结构原样移植到 [B,C,H,W]：前 n_shift 个通道由上一行移入，
#   接着 n_shift 个通道由下一行移入，其余通道不变，首末行保留自身特征。移位与
#   合并的数值逻辑完全一致，零可学习参数，与论文一致。通道移位比例 n_shift =
#   int(C * shift_ratio) 与原实现相同。注意：简写中的"沿通道循环移位"会改变
#   边界行为与合并语义，故未采用，保持原邻域移位结构。

'''
模块名称：DPS (Depth Shift) —— 深度移位

一、模块简介
把二维基础模型（如 SAM）提升到三维医学体数据时，逐切片独立推理会丢失切片间的
连续解剖结构。SAM+D 提出 DSM（Depth Shifting Module）：借鉴时序移位模块（TSM）
的思路，以零参数的方式把一部分通道沿深度（切片）维向前移位、另一部分向后移位，
使每个切片在保持自身特征的同时获得相邻切片的信息，从而在几乎不增加参数与计算
的前提下实现跨切片信息流。移位后的"移位-合并"结构（前向移位 + 后向移位 + 保留
通道）是该模块的核心数学；边界切片保留自身特征，避免零填充带来的不连续。
本模块把该思想忠实移植到统一的 2D 特征图接口上。

核心创新点：
1. 零参数深度移位：仅用张量索引赋值实现跨切片信息流动，不引入任何可学习权重
2. 双向移位-合并：前 n_shift 通道前向移位、次 n_shift 通道后向移位、其余通道保留
3. 边界保真：边界切片保持自身特征，避免零填充/循环回绕造成的伪影
4. 与注意力/卷积无缝衔接：可插在任意 block 前作为廉价的邻域上下文注入

二、结构设计
DPS 为零参数算子，张量形状流转如下（shift_ratio 默认 0.25）：
1. 计算移位通道数：n_shift = int(C * shift_ratio)
2. 前向移位（沿深度维，2D 接口下为 H 维）：
   out[:, :n_shift, 1:, :] = x[:, :n_shift, :-1, :]
   （第 i 个切片/行获得第 i-1 个的前 n_shift 个通道）
3. 后向移位：
   out[:, n_shift:2*n_shift, :-1, :] = x[:, n_shift:2*n_shift, 1:, :]
   （第 i 个切片/行获得第 i+1 个的次 n_shift 个通道）
4. 其余通道（约 50%）不变；边界切片/行保留自身特征
   输出形状与输入一致：[B,C,H,W] -> [B,C,H,W]

三、论文写法参考
若在论文中引用该模块，可描述为：
"我们采用 SAM+D 提出的深度移位（DPS）[ECCV 2026]，以零参数方式将部分通道沿
深度维做双向邻域移位，使每个位置在保留自身特征的同时聚合相邻切片/行的上下文，
实现低成本的跨深度信息流动。"
（原论文引用格式：Song et al., "SAM+D: Parameter-Efficient Dimensional Lifting of
SAM via Depth-Routed LoRA and Depth Shifting", ECCV 2026.）

四、适用任务
三维医学图像分割/分类（CT/MRI 体数据）、视频逐帧推理、任意具有"深度/序列"轴的
密集预测任务；也可作为 2D 特征图上沿空间维的廉价上下文注入即插即用模块。
'''
__all__ = ['DPS']


class DPS(nn.Module):
    """DPS: Depth Shift —— 深度移位（零参数通道移位）"""

    def __init__(self, channels: int, shift_ratio: float = 0.25):
        super().__init__()
        assert 0.0 <= shift_ratio <= 0.5, 'shift_ratio 应在 [0, 0.5] 内'
        self.channels = channels
        self.shift_ratio = shift_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'输入通道 {C} 与初始化 channels {self.channels} 不一致'
        n_shift = int(C * self.shift_ratio)

        out = x.clone()
        # 前向移位：位置 i 从 i-1 获得前 n_shift 个通道（边界位置保留自身特征）
        out[:, :n_shift, 1:, :] = x[:, :n_shift, :-1, :]
        # 后向移位：位置 i 从 i+1 获得次 n_shift 个通道（边界位置保留自身特征）
        out[:, n_shift:2 * n_shift, :-1, :] = x[:, n_shift:2 * n_shift, 1:, :]
        # 其余通道不变（由 clone 保留）
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = DPS(channels=128)
    output = model(input_tensor)
    print('=== DPS: Depth Shift ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
