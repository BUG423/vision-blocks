import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：LUT-Fuse: Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables (ICCV 2025)
# 论文链接：https://arxiv.org/abs/2509.00346
# 代码来源：https://github.com/zyb5/LUT-Fuse
# 原始许可证：MIT
# 模块出处：scripts/calculate.py 的 OptimizableLUT / Generator_for_info /
#          apply_fusion_4d_with_interpolation（推理路径）；fine_tune_lut.py 的 TV_4D 属
#          蒸馏训练正则，不在提取范围
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：保留 4D 可学习 LUT 的四线性插值（嵌套 lerp）与低阶/上下文编码公式原样
#          （尺度 𝒯 取官方代码的 16.0，论文正文写 17 但 ckpt 为 16^4，以代码为准）；
#          上下文编码器 Generator_for_info 结构（5 个 3×3 卷积块 + 残差）原样保留；
#          RGB↔YCbCr 往返与官方一致（LUT 只输出融合亮度 Y，再与可见光 CbCr 合成 RGB）。
#          双输入 IR+VIS 融合接口为 forward(x, x_aux)；x_aux=None 时退化为单张量 LUT
#          映射（逐通道取 Nv/Ni、Sobel 梯度 Gv、上下文 Sj，同一 LUT 共享，属接口适配，
#          插值数学不变）。num_luts=3 对应论文低阶近似编码的三个查找元素 Ni/Nv/Gv，
#          与上下文轴 Sj 共同构成 4D LUT（论文固定为 3）。蒸馏训练管线（TV/单调性正则、
#          MM-Net 教师）整体不在范围。

'''
模块名称：LTF (LUT Fusion Unit) —— 可学习查找表融合单元

一、模块简介
LUT-Fuse 将红外-可见光融合网络（MM-Net）的能力蒸馏进一张可学习的四维查找表
（MM-LUT），使推理退化为查表 + 多线性插值，速度比轻量融合网络快一个数量级，
可在移动端实时运行。MM-LUT 的查找元素由两部分编码：低阶近似编码（零阶强度
Ni/Nv 与一阶梯度 Gv，受 Taylor 展开启发）与高层联合场景上下文编码 Sj（一个
5 层 3×3 卷积的小网络，输入为 IR+VIS 拼接）。每个像素以四元组 (Nv, Gv, Sj, Ni)
在 4D 网格上做四线性插值得到融合亮度，再与可见光色度合成 RGB。
核心创新点：
1. 低阶近似编码：以强度+梯度这类低阶量作为查表坐标，计算开销极低且贴合
   融合任务对显著热目标与纹理细节的需求；
2. 高层联合场景上下文编码：可学习的场景编码器补充低阶量缺失的语义信息；
3. 可学习 LUT + 蒸馏：LUT 节点值与编码器参数可微训练，避免量化 LUT 的
   精度损失（本模块只保留推理路径，蒸馏训练不在范围）。

二、结构设计
输入约定值域 [0, 255]（与官方 I_A*255 一致）：
1. 上下文编码器 Φs = Generator_for_info(in_channels = C+1)：
   Conv3×3→LeakyReLU(0.2)→IN + 3×(Conv3×3→LeakyReLU→IN) + Dropout(0.5)→Conv3×3→Sigmoid，
   输出 Sj ∈ [0,1]，缩放 Sj·255/𝒯（𝒯=16.0）；
2. 低阶编码（双输入模式，x 为可见光 RGB，x_aux 为红外 1 通道）：
   - Nv：x 的 YCbCr 亮度 Y，缩放 ·255/𝒯；
   - Gv：Y 的 Sobel 梯度幅值，逐图 min-max 归一化后 ·255/𝒯；
   - Ni：x_aux 强度 /𝒯；
   （单张量模式 x_aux=None：逐通道取 Nv=Ni=x_c/𝒯，Gv 由 x_c 梯度得到，
   Sj 由 Φs(cat(x, x 的通道均值)) 得到——属接口适配，插值不变）；
3. 索引与插值：k=⌊a⌋ 等 floor 索引，ceil 钳制到 [0, 15]，alpha 为小数部分，
   按 (Nv, Gv, Sj, Ni) 轴序在 16×16×16×16×1 的 LUT 上做四线性插值
   （嵌套 lerp 16 个格点，与原实现逐项一致）；
4. 输出重建（双输入模式）：fusion_y 与可见光 CbCr 拼接后 YCbCr→RGB，
   输出 [B, 3, H, W]；单张量模式直接堆叠逐通道查表结果 [B, C, H, W]。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 LUT-Fuse（ICCV 2025）提出的多模态融合查找表 MM-LUT 作为推理单元：
以可见光亮度、Sobel 梯度、红外强度等低阶近似编码连同可学习的联合场景上下文
编码构成 4D 查表坐标，在 16^4 的可学习查找表上做四线性插值，并以可见光色度
重建 RGB，实现亚毫秒级的红外-可见光融合。"
（原论文引用格式：@inproceedings{yi2025lut, title={LUT-Fuse: Towards Extremely
Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up
Tables}, booktitle={ICCV}, year={2025}}）

四、适用任务
适用于红外-可见光图像融合等多模态融合任务，以及任意可由像素级查找映射近似的
图像增强/映射任务（单张量模式）。适合对延迟极端敏感的实时/边缘部署场景。
'''


def _rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] → YCbCr [0,1]（与原 scripts/calculate.rgb_to_ycbcr 逐式一致）。"""
    return torch.stack(
        (0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
         128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
         128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
        dim=1)


def _ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """YCbCr [0,1] → RGB [0,1]（与原 scripts/calculate.ycbcr_to_rgb 逐式一致）。"""
    return torch.stack(
        (img[:, 0, :, :] + (img[:, 2, :, :] - 0.5) * 1.402,
         img[:, 0, :, :] - (img[:, 1, :, :] - 0.5) * 0.344136 - (img[:, 2, :, :] - 0.5) * 0.714136,
         img[:, 0, :, :] + (img[:, 1, :, :] - 0.5) * 1.772),
        dim=1)


def _lerp(v1: torch.Tensor, v2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return v1 * (1 - alpha) + v2 * alpha


def _quad_lerp(lut: torch.Tensor,
               y_f, y_c, g_f, g_c, s_f, s_c, i_f, i_c,
               y_a, g_a, s_a, i_a) -> torch.Tensor:
    """4D 四线性插值：与原 apply_fusion_4d_with_interpolation 的嵌套 lerp 逐项一致。

    lut 轴序 (Nv, Gv, Sj, Ni)，各索引为 [B, H, W] 长整型，alpha 为 [B, H, W, 1]。
    """
    return _lerp(
        _lerp(
            _lerp(
                _lerp(lut[y_f, g_f, s_f, i_f], lut[y_f, g_f, s_f, i_c], i_a),
                _lerp(lut[y_f, g_f, s_c, i_f], lut[y_f, g_f, s_c, i_c], i_a),
                s_a,
            ),
            _lerp(
                _lerp(lut[y_f, g_c, s_f, i_f], lut[y_f, g_c, s_f, i_c], i_a),
                _lerp(lut[y_f, g_c, s_c, i_f], lut[y_f, g_c, s_c, i_c], i_a),
                s_a,
            ),
            g_a,
        ),
        _lerp(
            _lerp(
                _lerp(lut[y_c, g_f, s_f, i_f], lut[y_c, g_f, s_f, i_c], i_a),
                _lerp(lut[y_c, g_f, s_c, i_f], lut[y_c, g_f, s_c, i_c], i_a),
                s_a,
            ),
            _lerp(
                _lerp(lut[y_c, g_c, s_f, i_f], lut[y_c, g_c, s_f, i_c], i_a),
                _lerp(lut[y_c, g_c, s_c, i_f], lut[y_c, g_c, s_c, i_c], i_a),
                s_a,
            ),
            g_a,
        ),
        y_a,
    )


def _generator_block(in_filters: int, out_filters: int, normalization: bool = False) -> t.List[nn.Module]:
    """原 scripts/calculate.generator_block。"""
    layers: t.List[nn.Module] = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class _ContextGenerator(nn.Module):
    """高层联合场景上下文编码器（原 Generator_for_info，5 个 3×3 卷积块 + 残差）。"""

    def __init__(self, in_channels: int = 4, hidden: int = 16, drop_p: float = 0.5):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(hidden, affine=True),
        )
        self.mid_layer = nn.Sequential(
            *_generator_block(hidden, hidden, normalization=True),
            *_generator_block(hidden, hidden, normalization=True),
            *_generator_block(hidden, hidden, normalization=True),
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Conv2d(hidden, 1, 3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, img_input: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(img_input)
        identity = x
        out = self.mid_layer(x)
        out += identity
        out = self.output_layer(out)
        return out


class LTF(nn.Module):
    """LTF: LUT Fusion Unit —— 可学习查找表融合单元"""

    def __init__(self, channels: int, num_luts: int = 3, lut_dim: int = 16,
                 value_scale: float = 16.0, context_hidden: int = 16, drop_p: float = 0.5):
        super().__init__()
        # num_luts：低阶近似编码的查找元素数（Ni/Nv/Gv 三组），与上下文轴 Sj 构成 4D
        # LUT；论文与官方实现固定为 3。
        assert num_luts == 3, 'MM-LUT 低阶查找元素固定为 Ni/Nv/Gv 三个（num_luts=3）'
        self.channels = channels
        self.num_luts = num_luts
        self.lut_dim = lut_dim
        self.value_scale = value_scale  # 论文符号 𝒯；官方代码取 16.0（ckpt 16^4）

        # 4D 可学习 LUT，轴序 (Nv, Gv, Sj, Ni)，每格输出 1 个融合亮度值
        self.lut = nn.Parameter(torch.zeros(lut_dim, lut_dim, lut_dim, lut_dim, 1))
        # 上下文编码器：输入 cat(主图, 辅图/强度图) 共 channels+1 通道
        # （官方 channels=3 时恰为 Generator_for_info(in_channels=4)）
        self.context = _ContextGenerator(in_channels=channels + 1, hidden=context_hidden, drop_p=drop_p)

        self.register_buffer('sobel_x', torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    # ---- 编码辅助（公式与原 apply_fusion_4d_with_interpolation 一致）----

    def _gradient_scaled(self, y01: torch.Tensor) -> torch.Tensor:
        """Sobel 梯度幅值 → 逐图 min-max 归一化 → ·255/𝒯，得 Gv 坐标 [B, H, W]。"""
        grad_x = F.conv2d(y01, self.sobel_x, padding=1)
        grad_y = F.conv2d(y01, self.sobel_y, padding=1)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)                    # [B, 1, H, W]
        min_val = gradient.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
        max_val = gradient.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        gradient_normalized = (gradient - min_val) / (max_val - min_val + 1e-8)
        gradient_scaled = (gradient_normalized * 255.) / self.value_scale
        return gradient_scaled.squeeze(1)                                   # [B, H, W]

    def _floor_ceil_alpha(self, scaled: torch.Tensor, dim: int):
        f = torch.floor(scaled).long()
        c = torch.clamp(f + 1, 0, dim - 1)
        a = (scaled - f).unsqueeze(-1)                                      # [B, H, W, 1]
        return f, c, a

    def _fuse_from_coords(self, y_scaled, g_scaled, s_scaled, i_scaled) -> torch.Tensor:
        """由四个缩放坐标查表插值，返回融合亮度 [B, 1, H, W]。"""
        y_f, y_c, y_a = self._floor_ceil_alpha(y_scaled, self.lut.shape[0])
        g_f, g_c, g_a = self._floor_ceil_alpha(g_scaled, self.lut.shape[1])
        s_f, s_c, s_a = self._floor_ceil_alpha(s_scaled, self.lut.shape[2])
        i_f, i_c, i_a = self._floor_ceil_alpha(i_scaled, self.lut.shape[3])
        fusion_result = _quad_lerp(self.lut, y_f, y_c, g_f, g_c, s_f, s_c, i_f, i_c,
                                   y_a, g_a, s_a, i_a)                      # [B, H, W, 1]
        return fusion_result.permute(0, 3, 1, 2)                            # [B, 1, H, W]

    def forward(self, x: torch.Tensor, x_aux: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]，值域 [0, 255]。
               x_aux 给出时为可见光 RGB（C=3）；否则为单张量 LUT 映射。
            x_aux: Tensor, shape = [B, 1, H, W]，红外强度（可选，值域 [0, 255]）。
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        if x_aux is not None:
            # ---- 双输入 IR+VIS 融合（官方推理路径，公式逐式一致）----
            assert self.channels == 3, '融合模式要求 channels=3（可见光 RGB）'
            assert x_aux.shape == (B, 1, H, W), f'x_aux 应为 [B,1,H,W]，得到 {tuple(x_aux.shape)}'

            image_cat = torch.cat((x, x_aux), dim=1)                        # [0, 255]
            context = self.context(image_cat)                               # [B, 1, H, W] ∈ [0,1]
            context_scaled = (context * 255. / self.value_scale).squeeze(1)  # [B, H, W]
            infrared_scaled = x_aux / self.value_scale                      # [0, 𝒯)

            ycbcr_vis = _rgb_to_ycbcr(x / 255.)                             # [B, 3, H, W]
            ycbcr_vis_scaled = ycbcr_vis * 255.0 / self.value_scale

            y_vi_scaled = ycbcr_vis_scaled[:, 0, :, :]                      # [B, H, W]
            cb_cr = ycbcr_vis[:, 1:, :, :]                                  # [B, 2, H, W]
            ir_scaled = infrared_scaled[:, 0, :, :]                         # [B, H, W]
            g_scaled = self._gradient_scaled(ycbcr_vis[:, :1, :, :])        # [B, H, W]

            fusion_y = self._fuse_from_coords(y_vi_scaled, g_scaled, context_scaled, ir_scaled)
            fusion_ycbcr = torch.cat([fusion_y, cb_cr], dim=1)
            return _ycbcr_to_rgb(fusion_ycbcr)                              # [B, 3, H, W]

        # ---- 单张量 LUT 映射（接口适配：逐通道取 Nv=Ni=x_c，插值数学不变）----
        assert self.channels >= 1
        context_in = torch.cat((x, x.mean(dim=1, keepdim=True)), dim=1)     # [B, C+1, H, W]
        context_scaled = (self.context(context_in) * 255. / self.value_scale).squeeze(1)

        outs = []
        for c in range(self.channels):
            p = x[:, c:c + 1, :, :]                                         # [B, 1, H, W]
            y_scaled = p.squeeze(1) / self.value_scale                      # Nv
            i_scaled = p.squeeze(1) / self.value_scale                      # Ni（单模态以自身为辅图）
            g_scaled = self._gradient_scaled(p / 255.)                      # Gv
            outs.append(self._fuse_from_coords(y_scaled, g_scaled, context_scaled, i_scaled))
        return torch.cat(outs, dim=1)                                       # [B, C, H, W]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 单张量 LUT 映射（x_aux=None），值域 [0, 255] 与官方一致
    input_tensor = torch.rand(1, 3, 32, 32) * 255.
    model = LTF(channels=3)
    output = model(input_tensor)
    print('=== LTF: LUT Fusion Unit ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))

    # 双输入 IR+VIS 融合路径
    vis = torch.rand(1, 3, 32, 32) * 255.
    ir = torch.rand(1, 1, 32, 32) * 255.
    fused = model(vis, ir)
    print('fusion_mode output_size:', fused.size())
