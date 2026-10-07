import math
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：AD-GBC: Anisotropic Granular-Ball Skip-Connection Refiner for UNet-Based Medical Image Segmentation (CVPR 2026)
# 论文链接：https://github.com/SiaShen-dot/AD-GBC （官方实现仓库，README 未提供 arXiv 链接）
# 代码来源：https://github.com/SiaShen-dot/AD-GBC
# 原始许可证：MIT
# 模块出处：archs_GBC.py 的 GranularBall 类（L10-65）、Lo2 类（L71-168）、
#          Lo2Block 类（L171-199）与 DWConv 类（L202-211）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：1) 将原网络中分离使用的 GranularBall（跳跃连接细化器）与 Lo2Block（Lo2 局部
#             算子块）融合为单个即插即用模块：先各向异性粒球聚类重加权细化，再接 Lo2
#             局部算子（顺序组合，二者数值公式均逐行保留）；不移植 Rolling-UNet 整网；
#          2) GranularBall 原 forward 的 tau=1.0 硬编码参数改为使用 self.tau（初始化默认
#             1.0，数值不变，属硬编码改参数）；返回值仅保留细化后的特征张量，
#             att/sigma/dif 等辅助损失中间量不再返回（原为 UNet 训练 loss 打包服务）；
#          3) 删除 Lo2 未使用的 shift_size/pad 死代码；Lo2 的 fc5/x1+x 维度要求
#             hidden==in（原实现固定 mlp_ratio=1）改为显式 assert；
#          4) timm 的 trunc_normal_ / DropPath 用 nn.init.trunc_normal_ 与纯 torch
#             DropPath 重写（截断 ±2σ 与原一致）；utils 的 nn/torch 统一为显式 import；
#          5) token 接口 [B,N,C] 包装为统一 4D 接口 [B,C,H,W]（内部 flatten(2).transpose /
#             transpose.reshape 还原，不改数值）。

'''
模块名称：L2B (Lo2Block) —— 各向异性粒球重加权局部算子块

一、模块简介
UNet 的跳跃连接直接把编码器特征加到解码器，容易引入噪声与冗余区域，
在医学小目标分割上尤为明显。AD-GBC 提出各向异性粒球（Granular-Ball）
细化器：用可学习的 K 个粒球中心与对角协方差（softplus 保正）在特征空间
做软聚类，以马氏距离软分配权重重构特征并残差细化，实现区域自适应的
跳跃连接重加权。配套的 Lo2 局部算子块用方向错位滚动（OR-MLP）在水平/
垂直两个方向逐通道位移聚合长条感受野，再融合深度-逐点卷积（DSC）分支，
以很低的开销补充局部空间混合。

核心创新点：
1. 各向异性粒球聚类：每个粒球带对角协方差（或标量半径），软分配
   att = softmax(-dist²/τ)，重构 = att @ centers，区域重加权；
2. OR-MLP 方向错位滚动：沿 H/W 对每个通道按通道序号循环位移后做
   两次线性映射，形成方向解耦的长条感受野；
3. DSC 分支融合：3×3 深度卷积 + 1×1 逐点卷积与 OR-MLP 路径拼接输出；
4. 轻量即插即用：不依赖注意力，可作为任意 4D 特征的细化块。

二、结构设计
L2B 由以下子结构组成（形状以输入 [B, C, H, W]、N = H*W、K = num_balls 计）：
1. 粒球重加权细化（GranularBall）：
   - 可选投影 proj_in: C→proj_dim（1×1 Conv+BN）；
   - z_flat: [B, N, d]，与 centers: [K, d] 求差 dif: [B, N, K, d]；
   - sigma = softplus(log_sigma)（对角 [K, d]）或 softplus(log_radius)（[K, 1]）；
   - dist2 = Σ(dif/sigma)² ∈ [B, N, K]；att = softmax(-dist2/τ)；
   - recon = att @ centers → [B, d, H, W]（可选 proj_out+BN）；
   - out = recon + x（use_residual=True）→ refine（3×3 Conv+BN+ReLU）；
2. Lo2 局部算子块（token 化 [B, N, C]）：
   - OR-MLP 支路 1：按通道滚动 H → fc1+GELU → 滚动 W → fc2（+x 残差）；
   - OR-MLP 支路 2：按通道滚动 -W → fc3+GELU → 滚动 H → fc4（+x 残差）；
   - 拼接 → LayerNorm(2C) → fc5(2C→C)（+x 残差）；
   - DSC 支路：DWConv（3×3 dw + 1×1 pw）→ ReLU → BN → [B, N, C]；
   - 拼接两支路 → fc6(2C→C) → Dropout；
3. 输出：[B, N, C] → [B, C, H, W]（形状保持）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 AD-GBC（CVPR 2026）提出的 L2B 块：以各向异性粒球软聚类对
特征做区域自适应重加权细化（对角协方差马氏距离 + 软分配重构 + 残差），
随后经 Lo2 局部算子（方向错位滚动 OR-MLP 与深度-逐点卷积双支路融合）
补充方向化解耦的局部空间混合，有效抑制跳跃连接引入的噪声区域。"

四、适用任务
适用于医学图像分割等 UNet 类编解码结构（跳跃连接细化、瓶颈/解码器
特征增强），也可作为通用 4D 特征细化块用于检测/分割主干。尤其适合
小目标、边界模糊、需区域自适应抑制噪声的密集预测任务。
'''


class _DropPath(nn.Module):
    """随机深度（timm DropPath 的纯 torch 等价实现）。"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _DWConv(nn.Module):
    """深度-逐点卷积（原 archs_GBC.DWConv）。"""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.dwconv(x))


class _GranularBall(nn.Module):
    """各向异性粒球软聚类细化器（原 archs_GBC.GranularBall）。"""

    def __init__(self, in_ch: int, num_balls: int = 32, proj_dim: t.Optional[int] = None,
                 use_residual: bool = True, use_diag_cov: bool = True, tau: float = 1.0):
        super().__init__()
        self.in_ch = in_ch
        self.proj_dim = proj_dim or in_ch
        self.num_balls = num_balls
        self.use_residual = use_residual
        self.use_diag_cov = use_diag_cov
        self.tau = tau

        self.centers = nn.Parameter(torch.randn(num_balls, self.proj_dim) * 0.01)
        if use_diag_cov:
            self.log_sigma = nn.Parameter(torch.zeros(num_balls, self.proj_dim))   # diag std
        else:
            self.log_radius = nn.Parameter(torch.zeros(num_balls, 1))              # scalar std

        if self.proj_dim != in_ch:
            self.proj_in = nn.Conv2d(in_ch, self.proj_dim, 1, bias=True)
            self.bn_in = nn.BatchNorm2d(self.proj_dim)
            self.proj_out = nn.Conv2d(self.proj_dim, in_ch, 1, bias=False)
            self.bn_out = nn.BatchNorm2d(in_ch)
        else:
            self.proj_in = self.proj_out = None

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        z = self.bn_in(self.proj_in(x)) if self.proj_in is not None else x
        d = z.shape[1]
        z_flat = z.view(B, d, H * W).permute(0, 2, 1)                              # (B,N,d)

        # 球距离（缩放欧氏/对角马氏）
        dif = z_flat.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)         # (B,N,K,d)
        if self.use_diag_cov:
            sigma = F.softplus(self.log_sigma).unsqueeze(0).unsqueeze(0)           # (1,1,K,d)
        else:
            sigma = F.softplus(self.log_radius).unsqueeze(0).unsqueeze(0)          # (1,1,K,1)
        dif_scaled = dif / sigma
        dist2 = (dif_scaled ** 2).sum(-1)                                          # (B,N,K)

        att = F.softmax(-dist2 / max(1e-6, self.tau), dim=-1)                      # 软集合归属

        recon_flat = torch.matmul(att, self.centers)                               # (B,N,d)
        recon = recon_flat.permute(0, 2, 1).view(B, d, H, W)

        if self.proj_out is not None:
            recon = self.bn_out(self.proj_out(recon))

        out = recon + x if self.use_residual else recon
        return self.refine(out)


class _Lo2(nn.Module):
    """方向错位滚动 OR-MLP + DSC 局部算子（原 archs_GBC.Lo2）。"""

    def __init__(self, in_features: int, hidden_features: t.Optional[int] = None,
                 out_features: t.Optional[int] = None, act_layer: t.Callable = nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features == in_features and out_features == in_features, (
            f'Lo2 原实现要求 hidden==in==out（fc5 输入 2*hidden、x1+x 残差），'
            f'got in={in_features}, hidden={hidden_features}, out={out_features}')
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc5 = nn.Linear(in_features * 2, hidden_features)
        self.fc6 = nn.Linear(hidden_features * 2, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = _DWConv(hidden_features)
        self.act1 = act_layer()
        self.act2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_features * 2)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        ### OR-MLP 支路 1（H 向滚动 → fc1 → W 向滚动 → fc2）
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc1(x_shift_r)
        x_shift_r = self.act1(x_shift_r)
        x_shift_r = self.drop(x_shift_r)
        xn = x_shift_r.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc2(x_shift_c)
        x_1 = self.drop(x_shift_c)

        ### OR-MLP 支路 2（-W 向滚动 → fc3 → H 向滚动 → fc4）
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, -shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc3(x_shift_c)
        x_shift_c = self.act1(x_shift_c)
        x_shift_c = self.drop(x_shift_c)
        xn = x_shift_c.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc4(x_shift_r)
        x_2 = self.drop(x_shift_r)

        x_1 = torch.add(x_1, x)
        x_2 = torch.add(x_2, x)
        x1 = torch.cat([x_1, x_2], dim=2)
        x1 = self.norm1(x1)
        x1 = self.fc5(x1)
        x1 = self.drop(x1)
        x1 = torch.add(x1, x)

        ### DSC
        x2 = x.transpose(1, 2).view(B, C, H, W)
        x2 = self.dwconv(x2)
        x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = x2.flatten(2).transpose(1, 2)

        x3 = torch.cat([x1, x2], dim=2)
        x3 = self.fc6(x3)
        x3 = self.drop(x3)
        return x3


class L2B(nn.Module):
    """L2B: Lo2Block with Anisotropic Granular-Ball Reweighting —— 各向异性粒球重加权局部算子块"""

    def __init__(self, channels: int, num_balls: int = 8, proj_dim: t.Optional[int] = None,
                 use_residual: bool = True, use_diag_cov: bool = True, tau: float = 1.0,
                 mlp_ratio: float = 1.0, drop: float = 0., drop_path: float = 0.):
        super().__init__()
        self.channels = channels
        self.gbc = _GranularBall(
            in_ch=channels, num_balls=num_balls, proj_dim=proj_dim,
            use_residual=use_residual, use_diag_cov=use_diag_cov, tau=tau)
        self.lo2 = _Lo2(
            in_features=channels, hidden_features=int(channels * mlp_ratio),
            out_features=channels, drop=drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        # 粒球区域重加权细化 [B, C, H, W]
        out = self.gbc(x)

        # [B, C, H, W] -> [B, N, C]（token 化，统一接口包装）
        tokens = out.flatten(2).transpose(1, 2)

        # Lo2 局部算子（原 Lo2Block.forward = drop_path(mlp(x, H, W))）
        tokens = self.drop_path(self.lo2(tokens, H, W))

        # [B, N, C] -> [B, C, H, W]
        return tokens.transpose(1, 2).reshape(B, C, H, W)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = L2B(channels=128)
    output = model(input_tensor)
    print('=== L2B: Lo2Block with Anisotropic Granular-Ball Reweighting ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
