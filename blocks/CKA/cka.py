import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：SL²A-INR: Single-Layer Learnable Activation for Implicit Neural Representation (ICCV 2025)
# 论文链接：https://arxiv.org/abs/2409.10836
# 代码来源：https://github.com/Iceage7/SL2A-INR
# 原始许可证：MIT
# 模块出处：ChebyKANLayer.py 的 ChebyKANLayer 类（参考 SL2A_INR.py 的 ChebyLayer 用法）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原 ChebyKANLayer 是作用于特征向量 [N, input_dim]→[N, output_dim] 的
# 全连接映射（系数张量 [in, out, degree+1]）。本块按统一接口暴露为 [B,C,H,W] 上的
# 逐通道可学习激活（系数 [C, degree+1]，每通道一条 Chebyshev 级数）；tanh 归一化、
# cos(k·acos(x)) 基函数递推与系数求和公式与原实现完全一致，未改任何数值逻辑。
# 纯 torch，无第三方依赖。

'''
模块名称：CKA (ChebyKAN Learnable Activation) —— Chebyshev 可学习激活模块

一、模块简介
SL²A-INR 指出：隐式神经表示（INR）中固定激活函数（ReLU、Sine 等）限制了
单层网络的表达能力，而 Chebyshev 多项式基可以紧凑地逼近复杂函数。受
Kolmogorov-Arnold Networks 启发，ChebyKAN 用 Chebyshev 多项式替代样条系数：
将输入 tanh 归一化到 [-1, 1]，以 cos(k·acos(x)) 构造第 k 阶 Chebyshev 基
T_k(x)，再用可学习系数做线性组合。这样"激活"本身成为可学习函数，且基函数
在 [-1,1] 上正交、数值稳定。CKA 将该机制适配为卷积特征图上的逐通道可学习
激活（原实现为特征向量上的全连接层），每个通道拥有独立的 Chebyshev 级数
系数，作为即插即用的激活/调制模块使用。

二、结构设计
输入 [B, C, H, W]，输出 [B, C, H, W]（形状保持）：
1. tanh 归一化：x = tanh(x) ∈ (-1, 1)（Chebyshev 多项式定义域约束）
2. 基函数展开：对每个元素计算 T_k = cos(k · acos(tanh(x)))，k = 0..degree
   （等价于原式：expand → acos → 乘 arange[0..degree] → cos）
3. 系数组合：y[..., c] = Σ_k coeff[c, k] · T_k，coeff ∈ [C, degree+1] 可学习
   （等价于原 einsum "bid,iod->bo" 的通道对角形式）
4. 初始化：默认 xavier_uniform，与原 ChebyKANLayer 一致（另支持 kaiming/
   orthogonal/uniform/normal）
shape 流转：[B,C,H,W] → tanh → [B,C,H,W,degree+1]（基函数）→ 加权求和 → [B,C,H,W]

三、论文写法参考
若在论文中引用该模块，可描述为："We use a Chebyshev-basis learnable activation
(SL²A-INR, ICCV 2025): each channel is modulated by a learnable Chebyshev series
T_k(x)=cos(k·acos(tanh x)), replacing fixed nonlinearities with orthogonal
basis expansions."并引用：SL²A-INR: Single-Layer Learnable Activation for
Implicit Neural Representation, ICCV 2025, arXiv:2409.10836.

四、适用任务
隐式神经表示（坐标网络/INR）、图像拟合、神经场重建；作为可学习激活也可
用于分类/分割主干中替代固定激活，或用于调制分支增强表达能力。
'''


class CKA(nn.Module):
    """CKA: ChebyKAN Learnable Activation —— 逐通道 Chebyshev 可学习激活"""

    def __init__(self, channels: int, degree: int = 4,
                 init_method: str = "xavier_uniform"):
        super().__init__()
        self.channels = channels
        self.degree = degree

        # 原 ChebyKANLayer 系数 [input_dim, output_dim, degree+1]；
        # 本块为逐通道激活形式，系数 [channels, degree+1]（通道对角）
        self.cheby_coeffs = nn.Parameter(torch.empty(channels, degree + 1))

        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.cheby_coeffs)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.cheby_coeffs, a=0, mode='fan_in',
                                     nonlinearity='relu')
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.cheby_coeffs, a=0, mode='fan_in',
                                    nonlinearity='relu')
        elif init_method == "orthogonal":
            nn.init.orthogonal_(self.cheby_coeffs)
        elif init_method == "uniform":
            nn.init.uniform_(self.cheby_coeffs, a=-0.5, b=0.5)
        elif init_method == "normal":
            nn.init.normal_(self.cheby_coeffs, mean=0.0,
                            std=1 / (channels * (degree + 1)))
        else:
            raise ValueError(f"unknown init_method: {init_method}")

        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"channel mismatch: expect {self.channels}, got {C}"

        # Chebyshev 多项式定义于 [-1, 1]，先用 tanh 归一化（与原文一致）
        x = torch.tanh(x)
        # 展开 degree+1 份：cos(k * acos(x))，k = 0..degree
        x = x.unsqueeze(-1).expand(-1, -1, -1, -1, self.degree + 1)
        x = x.acos()
        x = x * self.arange
        x = x.cos()                                   # [B, C, H, W, degree+1]
        # 系数线性组合（原 einsum "bid,iod->bo" 的逐通道形式）
        y = torch.einsum('bchwk,ck->bchw', x, self.cheby_coeffs)
        return y


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CKA(channels=128)
    output = model(input_tensor)
    print('=== CKA: ChebyKAN Learnable Activation ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
