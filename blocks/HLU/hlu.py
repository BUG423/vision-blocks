import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Hybrid-LUT: Channel-Aware Hybrid Lookup Table and Filtering for Efficient Image Denoising (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2608.11646
# 代码来源：https://github.com/Ai-ZL/Hybrid-LUT
# 原始许可证：MIT
# 模块出处：models/units.py 的 HDUnit 类（L7-90）/ HLUnit 类（L317-391），
#          查表应用方式参考 models/luts.py 的 HLLUT/HDLUT 类，通道感知混合取自
#          models/net_model.py（L160-178）与 models/lut_model.py（L68-100）的
#          weight_lut + grid_sample + softmax 融合逻辑
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：1) 仅保留推理单元：原 units.py 的训练态 CNN（conv1-6 + PixelShuffle，
#             经 transfer.py 蒸馏为 LUT）不移植，直接以可学习 LUT 参数实现推理查表；
#          2) 三线性 LUT 插值：HDUnit 原为 2 像素邻域（水平/对角）整数索引查表，
#             为满足三线性插值改用 HLUnit 的 3 像素邻域几何（'h' 水平三点 / 'l' 形三点），
#             LUT 表 (L,L,L) 以 F.grid_sample(mode='bilinear', align_corners=True) 在 5D
#             张量上做三线性插值（坐标 = 2*val-1 ∈ [-1,1]，越界 padding_mode='border'）；
#          3) 通道感知混合逐式保留：mean/var = avg_pool2d(k=5, pad=2) →
#             var_norm = clamp(var/0.01, 0, 1) → 构造 grid(x=var_norm*2-1, y=0) →
#             F.grid_sample(weight_lut) → softmax(dim=1) → num_luts 支路加权求和；
#             为支持多通道输入，把原单通道 (B,H,W) 的 grid 扩为按通道独立的
#             (B*C, H, W)（各通道用自身局部方差取混合权重，即通道感知）；
#          4) 旋转集成沿用原推理模式：4 个 rot90 方向累加后除以 avg_factor=2
#             （HDUnit/HLUnit 类属性）；tanh 输出与原 units 一致；
#          5) LUT 量化/取整（*127、round、clamp）与训练蒸馏管线不移植（推理前向仅
#             插值查表）；静态权重 LUT 初始化为可学习参数（默认 0 初值）。

'''
模块名称：HLU (Hybrid LUT Unit) —— 通道感知混合查找表单元

一、模块简介
纯 CNN 去噪算子推理开销高，传统 LUT 查表则以整数索引换取极快推理，但
表容量有限、对不同内容一刀切。Hybrid-LUT 提出通道感知的混合查找表与
滤波：把像素邻域映射到输出残差的函数蒸馏进多张查找表，再按输入内容的
局部活跃度（方差）动态混合多张不同“强度”的 LUT——平坦区用强平滑表、
纹理区用弱平滑表——从而以查表级开销逼近 CNN 的自适应去噪能力。本单元
仅实现推理侧：可学习 LUT 的连续插值查表（三线性）+ 通道感知混合权重
（1D 权重 LUT 的 grid_sample 插值 + softmax 融合），不含训练到 LUT 的
蒸馏管线。

核心创新点：
1. 三线性 LUT 插值：3 像素邻域值作坐标对 (L,L,L) 表连续插值，
   兼顾表容量与平滑性（相对整数索引可微、无阶梯伪影）；
2. 通道感知混合：局部方差归一化后作为坐标查 1D 权重 LUT
   （grid_sample 线性插值），softmax 产生 num_luts 张表的逐像素
   逐通道融合权重；
3. 多强度表混合：num_luts 张 LUT（强/中/弱平滑）内容自适应融合，
   等效一个超大 LUT 的条件化切片；
4. 推理开销与 LUT 同级：仅查表 + 插值 + 逐点加权。

二、结构设计
HLU 由以下子结构组成（形状以输入 [B, C, H, W]、N = B*C、L = lut_size 计）：
1. 邻域展开（ktype='h' 水平三点 a=(0,0), b=(0,1), c=(0,2)；
   ktype='l' 形三点 a=(0,0), b=(0,1), c=(1,1)）：
   按通道 reshape 为 [N, 1, H, W]，replicate 填充 (0,2,0,2) 后取 3 个偏移窗；
2. LUT 分支 k = 1..num_luts：对 4 个 rot90 方向分别做
   grid = stack([2a-1, 2b-1, 2c-1], -1) ∈ [N, 1, H, W, 3]，
   对 lut_k ∈ [L, L, L] 三线性插值（grid_sample, align_corners=True），
   逆旋转累加后除以 avg_factor=2；
3. 通道感知混合：
   - mean = avg_pool2d(x, 5, 1, 2)；var = avg_pool2d((x-mean)², 5, 1, 2)；
   - var_norm = clamp(var / 0.01, 0, 1)；
   - grid_mix = (var_norm*2-1, 0) ∈ [N, H, W, 2]；
   - weights = softmax(grid_sample(weight_lutᵀ.view(1, num_luts, 1, mix_size),
     grid_mix), dim=1) ∈ [N, num_luts, H, W]；
   - out = Σ_k weights_k ⊙ lut_out_k；
4. 输出：tanh(out) → [B, C, H, W]（形状保持，值域 [-1, 1]，与原 units 一致；
   原网络由调用方再与上采样输入相加，此处不外加残差）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 Hybrid-LUT（ECCV 2026）提出的通道感知混合查找表单元 HLU：
对像素三邻域以三线性插值查可学习 LUT 得到多张不同平滑强度的映射，
并由各通道局部方差经 1D 权重 LUT 插值产生 softmax 融合权重，实现
内容自适应的查表级高效特征/图像精修。"

四、适用任务
适用于图像去噪、去伪影等像素级恢复的高效推理，也可作为即插即用的
局部映射精修块用于超分/低级视觉任务；适合部署侧对算力敏感、需要
近似内容自适应滤波的场景。
'''


class HLU(nn.Module):
    """HLU: Hybrid LUT Unit —— 通道感知混合查找表单元"""

    # 旋转集成（原 HDUnit/HLUnit.rot_dict / avg_factor 的推理用法）
    rot_dict = [0, 1, 2, 3]
    avg_factor = 2.
    pad_dict = (0, 2, 0, 2)   # F.pad(left, right, top, bottom)，3 像素邻域

    def __init__(self, channels: int, num_luts: int = 3, lut_size: int = 16,
                 mix_size: int = 64, ktype: str = 'h'):
        super().__init__()
        assert ktype in ('h', 'l'), f"ktype must be 'h' or 'l', got {ktype!r}"
        assert channels > 0 and num_luts > 0 and lut_size > 1 and mix_size > 1, \
            f'illegal dims: channels={channels}, num_luts={num_luts}, lut_size={lut_size}, mix_size={mix_size}'
        self.channels = channels
        self.num_luts = num_luts
        self.lut_size = lut_size
        self.mix_size = mix_size
        self.ktype = ktype

        # num_luts 张 3D LUT（三线性插值，原整数索引 LUT 的连续化参数形式）
        self.luts = nn.Parameter(torch.zeros(num_luts, lut_size, lut_size, lut_size))
        # 通道感知混合权重 LUT（原 weight_lut_msb/lsb: (64, 3)）
        self.mix_lut = nn.Parameter(torch.zeros(mix_size, num_luts))

    def _trilinear_lookup(self, lut: torch.Tensor, img_a: torch.Tensor,
                          img_b: torch.Tensor, img_c: torch.Tensor) -> torch.Tensor:
        """对 (L,L,L) LUT 在 (a,b,c) 邻域坐标处做三线性插值。

        lut: [L, L, L]；img_*: [N, 1, H, W]（取值约在 [0,1]）→ 输出 [N, 1, H, W]。
        坐标映射 val → 2*val-1 ∈ [-1, 1]（align_corners=True）。
        """
        L = self.lut_size
        # [N, 1, H, W, 3]：grid_sample 5D 输入要求 grid 最后一维为 3 个插值坐标
        grid = torch.stack([img_a * 2.0 - 1.0, img_b * 2.0 - 1.0, img_c * 2.0 - 1.0], dim=-1)
        lut_t = lut.view(1, 1, L, L, L).expand(img_a.shape[0], -1, -1, -1, -1)
        out = F.grid_sample(lut_t, grid, mode='bilinear', padding_mode='border',
                            align_corners=True)                     # [N, 1, 1, H, W]
        return out.squeeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        # 按通道独立处理（原 units 均 reshape(B*C, 1, H, W)）
        x_in = x.reshape(B * C, 1, H, W)

        # ---- num_luts 张 LUT 分支（各自含 4 方向旋转集成）----
        lut_outs = []
        for k in range(self.num_luts):
            out_k = 0.0
            for r in self.rot_dict:
                x_rot = torch.rot90(x_in, r, [2, 3])
                _, _, Hr, Wr = x_rot.shape
                x_pad = F.pad(x_rot, self.pad_dict, mode='replicate')
                if self.ktype == 'h':
                    img_a = x_pad[:, :, 0:0 + Hr, 0:0 + Wr]
                    img_b = x_pad[:, :, 0:0 + Hr, 1:1 + Wr]
                    img_c = x_pad[:, :, 0:0 + Hr, 2:2 + Wr]
                else:  # 'l'
                    img_a = x_pad[:, :, 0:0 + Hr, 0:0 + Wr]
                    img_b = x_pad[:, :, 0:0 + Hr, 1:1 + Wr]
                    img_c = x_pad[:, :, 1:1 + Hr, 1:1 + Wr]
                val = self._trilinear_lookup(self.luts[k], img_a, img_b, img_c)
                out_k = out_k + torch.rot90(val, 4 - r, [2, 3])
            lut_outs.append(out_k / self.avg_factor)

        # ---- 通道感知混合（原 net_model/lut_model 的 weight_lut 融合）----
        mean = F.avg_pool2d(x_in, kernel_size=5, stride=1, padding=2)
        variance = F.avg_pool2d((x_in - mean) ** 2, kernel_size=5, stride=1, padding=2)
        var_norm = torch.clamp(variance / 0.01, 0, 1)

        N, _, _, _ = x_in.shape
        grid_mix = torch.zeros(N, H, W, 2, device=x.device, dtype=x.dtype)
        grid_mix[:, :, :, 0] = var_norm.squeeze(1) * 2.0 - 1.0   # x 坐标 = 归一化局部方差
        grid_mix[:, :, :, 1] = 0                                  # y 固定（1D 权重 LUT）

        lut_tensor = self.mix_lut.T.view(1, self.num_luts, 1, self.mix_size)
        weights = F.grid_sample(lut_tensor.expand(N, -1, -1, -1), grid_mix,
                                align_corners=True)               # [N, num_luts, H, W]
        weights = F.softmax(weights, dim=1)

        out = 0.0
        for k in range(self.num_luts):
            out = out + weights[:, k:k + 1, :, :] * lut_outs[k]

        out = torch.tanh(out)                                     # 原 units 的输出激活
        return out.reshape(B, C, H, W)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = HLU(channels=128)
    output = model(input_tensor)
    print('=== HLU: Hybrid LUT Unit ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
