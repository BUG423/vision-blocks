import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：ERM-BCL (Edge Response Module - BCL) —— 边缘响应模块（BCL版）

一、模块简介
本模块是 ERM 针对 BCL 时序数据格式的适配版本。在时序数据中，"边缘"
对应于数据的突变点或事件发生点。本模块通过检测时序突变点并增强相关特征。

二、结构设计
ERM-BCL 由边缘检测器、边缘增强器、边缘-非边缘分离器和自适应融合器组成。
'''


class ERM_BCL(nn.Module):
    """ERM-BCL: Edge Response Module for BCL format"""

    def __init__(self, channels: int, seq_len: int, reduction: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)

        self.edge_detector = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.edge_enhance = nn.Sequential(
            nn.Conv1d(channels, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.nonedge_process = nn.Sequential(
            nn.Conv1d(channels, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm1d(inner),
            nn.GELU(),
            nn.Conv1d(inner, channels, 1, bias=False),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_map = self.edge_detector(x)
        edge_feat = self.edge_enhance(x) * edge_map
        nonedge_feat = self.nonedge_process(x) * (1 - edge_map)
        concat = torch.cat([edge_feat, nonedge_feat], dim=1)
        gate = self.fusion_gate(concat)
        return edge_feat * gate + nonedge_feat * (1 - gate) + x


if __name__ == '__main__':
    x = torch.randn(1, 64, 128)
    model = ERM_BCL(channels=64, seq_len=128)
    print('ERM-BCL:', model(x).shape)
