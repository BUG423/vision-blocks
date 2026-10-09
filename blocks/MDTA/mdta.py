import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Restormer: Efficient Transformer for High-Resolution Image Restoration (IEEE TPAMI 2022)
# 论文链接：https://arxiv.org/abs/2111.09881
# 代码来源：https://github.com/swz30/Restormer
# 原始许可证：MIT
# 模块出处：basicsr/models/archs/restormer_arch.py 的 Attention 类
# 提取者：vision-blocks Agent
# 日期：2026-10-09
# 重构说明：1. 删除 basicsr、einops 等外部依赖，使用纯 PyTorch 的 view/permute 重构多头重排与还原，保持数值完全等价；
#          2. 接口对齐规范：参数 dim 改为 channels，作为构造函数首参；
#          3. 4D 输入形状保持：[B, C, H, W] -> [B, C, H, W]；
#          4. 核心数值逻辑严格保持一致：1x1 卷积升维生成 QKV 特征，3x3 深度可分离卷积注入局部空间上下文，
#             沿空间维度 L2 归一化后计算通道维转置自注意力（Transposed Self-Attention），
#             由可学习温度参数缩放并计算 Softmax，加权 Value 后经 1x1 卷积输出。

'''
模块名称：MDTA (Multi-DConv Head Transposed Self-Attention) —— 多深度卷积头转置自注意力

一、模块简介
标准空间自注意力的计算复杂度为输入空间分辨率的二次方 O((HW)^2)，在处理高分辨率计算机视觉任务（如图像恢复、
超分辨率、密集预测等）时面临极大的显存与算力开销。Restormer（IEEE TPAMI 2022）提出了多深度卷积头转置自注意力
（MDTA），将注意力的计算维度由空间维转换到通道维，计算跨通道协方差而非逐像素空间注意力图。
此外，MDTA 在 QKV 投影阶段引入 3×3 深度可分离卷积（Depthwise Conv），在计算全局通道交互前隐式编码局部空间上下文。
MDTA 的计算复杂度随空间分辨率线性增长 O(C^2 HW)，既保留了自注意力的长程动态特性，又极大降低了高分辨率下的计算负载。

核心创新点：
1. 通道维转置注意力：计算通道维的交叉注意力矩阵 [C, C]，复杂度关于空间分辨率线性，专为高分辨率设计；
2. 深度卷积空间上下文嵌入：在 QKV 线性投影后接 3×3 深度卷积，在进入多头通道注意力前注入局部邻域信息；
3. 通道间 L2 归一化与可学习温度：对 Query 与 Key 在空间维度做 L2 归一化，配合可学习温度参数稳定 Softmax 分布；
4. 即插即用：形状严格保持 [B, C, H, W] -> [B, C, H, W]，纯 PyTorch 零外部依赖。

二、结构设计
MDTA 由以下子结构组成（输入形状 [B, C, H, W]）：
1. 投影与深度卷积：
   - Conv2d(C, 3C, 1) 生成级联 QKV 特征；
   - Conv2d(3C, 3C, 3, padding=1, groups=3C) 提取局部空间特征；
   - chunk 切分为 Q, K, V，各 [B, C, H, W]。
2. 多头重塑与空间归一化：
   - 将 Q, K, V 重排为 [B, num_heads, C_head, HW]（C_head = C // num_heads）；
   - 在 HW 维度执行 F.normalize(..., dim=-1)。
3. 通道转置注意力与加权：
   - 注意力图 A = Softmax(temperature * (Q @ K^T), dim=-1)，形状 [B, num_heads, C_head, C_head]；
   - 输出特征 out = A @ V，形状 [B, num_heads, C_head, HW]。
4. 重塑与输出投影：
   - 还原为 [B, C, H, W]；
   - Conv2d(C, C, 1) 输出投影。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 Restormer（IEEE TPAMI 2022）提出的多深度卷积头转置注意力模块（MDTA）：
通过在 QKV 投影中引入 3×3 深度可分离卷积捕捉局部上下文，并在通道维度计算转置自注意力矩阵，
将标准自注意力的二次复杂度降至与特征图尺寸成线性关系，在保证长程通道依赖的同时显著提升计算效率。"

四、适用任务
适用于图像恢复（去噪、去雨、去雾、去模糊）、超分辨率、图像增强、语义分割及骨干网络的高分辨率特征提取阶段。
'''


class MDTA(nn.Module):
    """MDTA: Multi-DConv Head Transposed Self-Attention (Restormer, TPAMI 2022)"""

    def __init__(self, channels: int, num_heads: int = 8, bias: bool = False):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, stride=1, padding=1,
            groups=channels * 3, bias=bias
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, C, H, W]
        out: [B, C, H, W]
        """
        b, c, h, w = x.shape
        head_dim = c // self.num_heads

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重排为多头格式 [B, num_heads, head_dim, H*W]
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        # 沿空间维度归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 通道维转置注意力 [B, num_heads, head_dim, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 加权 Value 并还原空间形状
        out = attn @ v
        out = out.view(b, c, h, w)
        out = self.project_out(out)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = MDTA(channels=128)
    output = model(input_tensor)
    print('=== MDTA: Multi-DConv Head Transposed Self-Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
