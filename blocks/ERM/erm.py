import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：ERM (Edge Response Module) —— 边缘响应模块

一、模块简介
边缘信息是视觉理解的基础，但现有的边缘检测方法通常作为独立任务或后处理
步骤，未与特征提取紧密耦合。

ERM 的核心思想是：在特征提取过程中显式建模边缘响应，让网络自适应地增强
边缘相关特征，实现边缘感知的特征学习。

核心创新点：
1. 显式边缘响应：通过可学习的边缘检测器提取边缘响应图
2. 边缘增强特征：根据边缘响应增强边缘区域特征
3. 边缘-非边缘分离：将特征分为边缘和非边缘两部分分别处理
4. 自适应融合：边缘和非边缘特征自适应融合

二、结构设计
ERM 由以下子结构组成：
1. 边缘检测器（Edge Detector）：
   - 可学习的边缘检测卷积核
2. 边缘增强器（Edge Enhancer）：
   - 基于边缘响应的特征增强
3. 边缘-非边缘分离器（Edge-NonEdge Separator）：
   - 特征分离与独立处理
4. 自适应融合器

三、论文写法参考
"本文提出 ERM（Edge Response Module）模块，在特征提取过程中显式建模边缘
响应。该模块通过可学习的边缘检测器提取边缘响应图，然后基于边缘响应增强
边缘区域特征，并将特征分为边缘和非边缘两部分分别处理，最终自适应融合。"

四、适用任务
适用于边缘检测、语义分割、目标检测等需要边缘信息的视觉任务。
'''


class ERM(nn.Module):
    """ERM: Edge Response Module —— 边缘响应模块"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        # Edge detector (learnable)
        self.edge_detector = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # Edge enhancer
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Non-edge processor
        self.nonedge_process = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 1. Edge detection
        edge_map = self.edge_detector(x)                 # [B, 1, H, W]

        # 2. Edge enhancement
        edge_feat = self.edge_enhance(x) * edge_map

        # 3. Non-edge processing
        nonedge_feat = self.nonedge_process(x) * (1 - edge_map)

        # 4. Adaptive fusion
        concat = torch.cat([edge_feat, nonedge_feat], dim=1)
        gate = self.fusion_gate(concat)
        out = edge_feat * gate + nonedge_feat * (1 - gate)

        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = ERM(channels=64)
    output = model(input_tensor)
    print('=== ERM: Edge Response Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
