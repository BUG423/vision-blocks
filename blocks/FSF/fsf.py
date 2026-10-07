import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Spectral Scalpel: Amplifying Adjacent Action Discrepancy via Frequency-Selective Filtering for Skeleton-Based Action Segmentation (CVPR 2026)
# 论文链接：https://github.com/HaoyuJi/SpecScalpel （官方实现仓库，README 未提供 arXiv 链接）
# 代码来源：https://github.com/HaoyuJi/SpecScalpel
# 原始许可证：MIT
# 模块出处：libs/models/SpecScalpel.py 的 FFT_conv_filter 类（L65-96）与
#          FFT_filter 类（L117-171，含 CustomProcessing 类 L99-115）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：1) 1D→2D 适配：原对时间维的 torch.fft.rfft(..., dim=2) / irfft 改为对空间维的
#             torch.fft.rfft2 / irfft2(s=(H,W))，频谱形状由 (N,C,T/2+1,V) 变为 (B,C,H,W/2+1)；
#             可学习实/虚 1×1 调制卷积由 Conv1d 改为 Conv2d（仍先把实部/虚部沿 batch 维拼接
#             后做两次 1×1 卷积再切回，公式不变）；Gfilter 播种尺寸 (T1∈{40,48,56,64}, joint_num)
#             改为 (H 频基, W 频基) 并保留运行时 F.interpolate(..., mode='nearest') 到当前
#             频谱尺寸；动态路由 CustomProcessing 的 AdaptiveAvgPool2d((64, joint_num)) 因图像
#             宽度可变改为固定 (route_pool, route_pool) 池化栅格（64,64），三层线性结构保留；
#          2) 残差结构原样保留：实/虚调制支路 out_ifft + x；选择滤波支路 (out_ifft + x)/2；
#             静态 scale 原硬编码 (4,1,64,1,1)（64=n_features=C）改为 (num_filters,1,channels,1,1)；
#          3) 删除 FFT_conv_filter 中未使用的 batchnorm/relu 死代码与骨架任务专用 mask
#             （统一接口无 mask）；两阶段顺序组合作为单模块（原分别用于时序编码器与主干）。

'''
模块名称：FSF (FFT Selective Filter) —— 频域选择性滤波单元

一、模块简介
相邻动作片段的差异往往集中在特定频带，时域卷积难以针对性放大。
Spectral Scalpel 提出频域选择性滤波：把特征变换到频域，用可学习滤波器
对频谱做选择性调制，从而放大相邻动作的判别性频率成分。本模块把原论文的
两级频域算子整合为即插即用单元：（1）实/虚可学习调制——对 rfft 频谱的
实部与虚部各做两次 1×1 卷积再合成复频谱（内容自适应的频域门控）；
（2）选择性滤波组——多枚可学习频域滤波器（不同频带基尺寸，运行时插值到
当前频谱）分别滤波后，按输入幅值动态路由（softmax）与静态可学习尺度
加权融合。原算子作用于骨架序列的时间维，本单元适配为对图像空间维做
rfft2/irfft2，调制与残差结构保持不变。

核心创新点：
1. 实/虚分离调制：频谱实部虚部拼接后经两层 1×1 卷积再拆回，
   等价于复数乘性滤波器的低秩可学习参数化；
2. 选择性滤波组：多枚不同频带偏好的滤波器（nearest 插值到全频谱），
   由输入幅值经三层线性动态路由 + 静态可学习尺度共同 softmax 加权；
3. 双残差设计：调制支路 out + x，滤波支路 (out + x)/2，训练稳定；
4. 复杂度 O(CHW log(HW))，即插即用于任意 4D 特征。

二、结构设计
FSF 由两级子结构顺序组成（形状以输入 [B, C, H, W]、Wf = W//2+1 计）：
1. 实/虚调制级（对应原 FFT_conv_filter）：
   - x_fft = rfft2(x) ∈ [B, C, H, Wf]（复数）；
   - cat([real, imag], dim=0) → [2B, C, H, Wf]；
   - conv1x1_mag → conv1x1_mag2（两层 1×1 Conv2d）；
   - chunk 回 real/imag → complex → irfft2(s=(H,W)) → out + x（残差）；
2. 选择性滤波级（对应原 FFT_filter）：
   - x_fft = rfft2(x)；
   - num_filters 枚 Gfilter（1,1,h_k,w_k，ones 初始化）nearest 插值到
     (H, Wf) 后与 x_fft 相乘，各自 irfft2 得 out_k ∈ [B, C, H, W]；
   - 动态路由：|x| → AdaptiveAvgPool2d(route_pool²) → Linear(r→4) →
     Linear(r→4) → flatten → Linear(16→4) → [num_filters, B, C]
     → unsqueeze×2 → softmax(dim=0) 得 scale_dy；
   - 静态尺度：scale ∈ [num_filters, 1, C, 1, 1] → softmax(dim=0) 得 scale_pa；
   - out = Σ_k out_k · scale_dy[k] 与 Σ_k out_k · scale_pa[k] 取均值；
   - 残差：(out + x)/2。
3. 输出：[B, C, H, W]（形状保持）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 Spectral Scalpel（CVPR 2026）提出的频域选择性滤波单元 FSF：
对空间特征做二维实值 FFT，先以实/虚两路 1×1 卷积对复频谱进行可学习
调制，再以多枚不同频带的可学习滤波器进行选择性滤波，由输入幅值动态
路由与静态可学习尺度共同加权融合，配合双残差结构放大判别性频率成分。"

四、适用任务
原用于骨架动作分割的频域建模；适配后可用于图像分类/检测/分割主干中的
频域增强模块、图像恢复（超分/去噪/去模糊）中的频带选择性精修，以及任何
需要按频带放大细微差异的判别式任务。
'''


class _RealImagModulation(nn.Module):
    """频谱实/虚可学习调制（原 SpecScalpel.FFT_conv_filter，1D→2D）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1x1_mag = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_mag2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_fft = torch.fft.rfft2(x)                                   # [B, C, H, W//2+1]

        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        x_fft_real_imag = torch.cat([x_fft_real, x_fft_imag], dim=0)  # [2B, C, H, Wf]
        x_fft_real_imag = self.conv1x1_mag(x_fft_real_imag)
        x_fft_real_imag = self.conv1x1_mag2(x_fft_real_imag)

        x_fft_real, x_fft_imag = torch.chunk(x_fft_real_imag, 2, dim=0)

        out_fft = torch.complex(x_fft_real, x_fft_imag)
        out_ifft = torch.fft.irfft2(out_fft, s=(H, W))
        out_ifft = out_ifft + x                                      # 残差（与原一致）
        return out_ifft


class _CustomRouting(nn.Module):
    """幅值动态路由（原 SpecScalpel.CustomProcessing，1D→2D）：池化 → 三层线性 → num_filters 权重。"""

    def __init__(self, num_filters: int = 4, route_pool: int = 64):
        super().__init__()
        self.num_filters = num_filters
        self.route_pool = route_pool
        self.pool = nn.AdaptiveAvgPool2d((route_pool, route_pool))
        self.linear1 = nn.Linear(route_pool, num_filters)
        self.linear2 = nn.Linear(route_pool, num_filters)
        self.linear3 = nn.Linear(num_filters * num_filters, num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [num_filters, B, C]
        x = self.pool(x)                                              # [B, C, r, r]
        x = self.linear1(x)                                           # [B, C, r, num_filters]
        x = self.linear2(x.transpose(2, 3))                           # [B, C, num_filters, num_filters]
        x = x.transpose(2, 3)                                         # [B, C, num_filters, num_filters]
        x = x.flatten(2)                                              # [B, C, num_filters²]
        x = self.linear3(x)                                           # [B, C, num_filters]
        return x.permute(2, 0, 1)                                     # [num_filters, B, C]


class _SelectiveFilterBank(nn.Module):
    """多滤波器频域选择性滤波（原 SpecScalpel.FFT_filter，1D→2D）。"""

    def __init__(self, channels: int, num_filters: int = 4,
                 filter_sizes: t.Sequence[t.Tuple[int, int]] = ((40, 32), (48, 32), (56, 32), (64, 32)),
                 route_pool: int = 64):
        super().__init__()
        assert len(filter_sizes) == num_filters, \
            f'filter_sizes length {len(filter_sizes)} must equal num_filters {num_filters}'
        self.num_filters = num_filters
        # 各滤波器以不同频带基尺寸播种（原 T1∈{40,48,56,64}），运行时插值到当前频谱尺寸
        self.gfilters = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, h, w)) for (h, w) in filter_sizes])
        # 静态可学习尺度（原硬编码 4,1,64,1,1 的 64=n_features，此处改为 channels）
        self.scale = nn.Parameter(torch.zeros(num_filters, 1, channels, 1, 1))
        self.choose_filter = _CustomRouting(num_filters, route_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_fft = torch.fft.rfft2(x)                                   # [B, C, H, W//2+1]
        Hf, Wf = x_fft.shape[-2:]

        # 动态路由权重（由输入幅值决定）
        routing = self.choose_filter(torch.abs(x))                    # [num_filters, B, C]
        scale_dy = routing.unsqueeze(-1).unsqueeze(-1)                # [num_filters, B, C, 1, 1]
        scale_dy = scale_dy.softmax(dim=0)

        # 选择性滤波：每枚滤波器插值到当前频谱后滤波并逆变换
        out_iffts = []
        for k in range(self.num_filters):
            expanded_filter = F.interpolate(self.gfilters[k], size=(Hf, Wf), mode='nearest')
            out_fft_k = x_fft * expanded_filter
            out_iffts.append(torch.fft.irfft2(out_fft_k, s=(H, W)))

        scale_pa = self.scale.softmax(dim=0)                          # [num_filters, 1, C, 1, 1]

        out_ifft_dy = sum(out_iffts[k] * scale_dy[k] for k in range(self.num_filters))
        out_ifft_pa = sum(out_iffts[k] * scale_pa[k] for k in range(self.num_filters))

        out_ifft = (out_ifft_dy + out_ifft_pa) / 2
        out_ifft = (out_ifft + x) / 2                                 # 残差（与原一致）
        return out_ifft


class FSF(nn.Module):
    """FSF: FFT Selective Filter —— 频域选择性滤波单元"""

    def __init__(self, channels: int, num_filters: int = 4,
                 filter_sizes: t.Sequence[t.Tuple[int, int]] = ((40, 32), (48, 32), (56, 32), (64, 32)),
                 route_pool: int = 64):
        super().__init__()
        self.channels = channels
        self.num_filters = num_filters
        # 阶段 1：实/虚可学习调制（原 FFT_conv_filter）
        self.real_imag_mod = _RealImagModulation(channels)
        # 阶段 2：频域选择性滤波组（原 FFT_filter）
        self.selective = _SelectiveFilterBank(
            channels, num_filters=num_filters,
            filter_sizes=filter_sizes, route_pool=route_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        out = self.real_imag_mod(x)      # 实/虚调制 + 残差
        out = self.selective(out)        # 选择性滤波 + 残差
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = FSF(channels=128)
    output = model(input_tensor)
    print('=== FSF: FFT Selective Filter ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
