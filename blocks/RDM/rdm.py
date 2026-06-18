import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：RDM (Reaction-Diffusion Module) —— 反应扩散模块

一、模块简介
生物形态发生中的 Turing 反应扩散模型揭示了一个深刻现象：两种化学物质
（激活子与抑制子）在空间扩散与局部非线性反应的耦合下，能自组织地涌现
出斑点、条纹等有序的纹理结构。这一"扩散平滑 + 反应激发"的动力学机制，
天然适合刻画卷积特征中"信息全局铺展"与"局部对比增强"这两种需求的
耦合——扩散让信息在空间上铺展互补，反应让特征在局部产生对比与结构。

现有模块大多静态地做一次空间聚合或注意力调制，缺少显式的时间动力学演
化过程。RDM 的核心思想是：将特征投影为激活子场 u 与抑制子场 v 两个
相互作用的"形态发生场"，在网络内部迭代若干步反应扩散方程：
    ∂u/∂t = Du·∇²u + f(u, v)
    ∂v/∂t = Dv·∇²v + g(u, v)
其中 ∇² 为离散拉普拉斯算子（扩散项），Du/Dv 为可学习扩散系数，f/g 为
可学习的非线性反应动力学。每一步中，扩散项将信息在邻域间平滑铺展，
反应项根据 (u,v) 的耦合产生局部增强或抑制；多步迭代后，特征自发涌现
出更规整的结构化表达。最后将 u、v 融合回特征空间并做残差连接。

核心创新点：
1. 反应扩散动力学引入特征处理：将 Turing 模型作为即插即用的可微动力学
   模块嵌入网络，区别于单次静态聚合的注意力/卷积模块
2. 激活子-抑制子双场耦合：用两个相互作用场显式建模"促进-抑制"的对立
   统一，反应项由跨场 MLP 学习非线性动力学
3. 可学习扩散系数与拉普拉斯算子：Du、Dv 可学习，控制信息铺展强度；
   离散拉普拉斯实现空间扩散
4. 稳定的有界迭代：每步用 tanh 对更新量做有界饱和，保证多步演化数值
   稳定，同时保留反应扩散的方向性结构生成能力

二、结构设计
RDM 由以下子结构组成：
1. 双场投影（Dual-field Projection）：
   - 两个 1x1 卷积将特征投影为激活子场 u 与抑制子场 v [B, inner, H, W]
2. 扩散算子（Diffusion Operator）：
   - 固定的 5 点离散拉普拉斯核（深度卷积）实现 ∇²
   - 可学习扩散系数 Du、Dv 控制铺展强度
3. 反应动力学（Reaction Kinetics）：
   - 将 (u, v) 拼接，经 1x1 MLP 产生反应项 (f, g)，建模非线性交互
4. 迭代演化（Iterative Evolution）：
   - 欧拉步迭代 T 步：u ← u + dt·tanh(Du·∇²u + f(u,v))，v 同理
   - tanh 有界饱和保证多步稳定
5. 双场融合与解码（Fusion & Decode）：
   - 拼接 (u, v) 经 1x1 卷积融合并解码回原通道空间
6. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 RDM（Reaction-Diffusion Module）模块，将 Turing 反应扩散动
力学引入卷积特征增强。该模块将特征投影为激活子与抑制子两个相互作用场，
以离散拉普拉斯算子实现空间扩散、以可学习 MLP 实现非线性反应，迭代若
干步反应扩散方程使特征自发涌现出结构化表达。扩散项负责信息的空间铺展
互补，反应项负责局部对比的激发与抑制，二者耦合实现了动态的结构生成式
特征增强。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合需要全局信息铺展与局部结构增强
协同的任务，以及纹理、边界、周期性结构显著的场景（如分割、检测、医学
图像、遥感图像）。
'''


class RDM(nn.Module):
    """RDM: Reaction-Diffusion Module —— 反应扩散模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 num_steps: int = 4):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner
        self.num_steps = num_steps

        # Dual-field projection: feature -> activator u, inhibitor v
        self.proj_u = nn.Conv2d(channels, inner, 1, bias=False)
        self.proj_v = nn.Conv2d(channels, inner, 1, bias=False)

        # Learnable diffusion coefficients (activator diffuses less than
        # inhibitor in classic Turing patterns; here both are learnable)
        self.du = nn.Parameter(torch.tensor(0.1))
        self.dv = nn.Parameter(torch.tensor(0.2))
        # Learnable step size (kept moderate via sigmoid for stability)
        self.dt_logit = nn.Parameter(torch.tensor(0.0))

        # Fixed 5-point discrete Laplacian kernel for diffusion
        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

        # Reaction kinetics: nonlinear interaction of (u, v) -> (f, g)
        self.reaction = nn.Sequential(
            nn.Conv2d(2 * inner, 2 * inner, 1, bias=False),
            nn.BatchNorm2d(2 * inner),
            nn.GELU(),
            nn.Conv2d(2 * inner, 2 * inner, 1, bias=False),
        )

        # Fusion & decode: (u, v) -> channels
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * inner, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise 5-point Laplacian ∇²x."""
        B, C, H, W = x.shape
        weight = self.lap_kernel.expand(C, 1, 3, 3).contiguous()
        return F.conv2d(x, weight, bias=None, stride=1, padding=1, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 1. Project to dual morphogen fields
        u = self.proj_u(x)                            # [B, inner, H, W]
        v = self.proj_v(x)                            # [B, inner, H, W]

        # Learnable, non-negative diffusion coefficients and bounded step size
        du = F.softplus(self.du)
        dv = F.softplus(self.dv)
        dt = torch.sigmoid(self.dt_logit)             # in (0, 1)

        # 2. Iterative reaction-diffusion evolution
        for _ in range(self.num_steps):
            lap_u = self._laplacian(u)               # diffusion of u
            lap_v = self._laplacian(v)               # diffusion of v
            react = self.reaction(torch.cat([u, v], dim=1))  # [B, 2*inner, H, W]
            fu, gv = react.chunk(2, dim=1)
            # Bounded Euler step: tanh saturates the update for stability
            u = u + dt * torch.tanh(du * lap_u + fu)
            v = v + dt * torch.tanh(dv * lap_v + gv)

        # 3. Fuse dual fields and decode back to feature space
        out = self.fuse(torch.cat([u, v], dim=1))     # [B, C, H, W]
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = RDM(channels=128)
    output = model(input_tensor)
    print('=== RDM: Reaction-Diffusion Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    print('steps:', model.num_steps)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
