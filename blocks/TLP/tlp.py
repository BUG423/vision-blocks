import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Alias-Free ViT: Fractional Shift Invariance via Linear Attention (NeurIPS 2025)
# 论文链接：https://github.com/hmichaeli/alias_free_vit（见仓库 README / Citation）
# 代码来源：https://github.com/hmichaeli/alias_free_vit
# 原始许可证：Apache-2.0
# 模块出处：af_ops.py 的 TruncLPF 类
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删去 torch.cuda.amp.autocast 装饰器与全局 ENABLE_FFT_AMP 开关，改为
#          显式 dtype 策略（fp16/bfloat16 提升到 float32 再做 FFT，与原注释
#          "make sure the data is FP32" 一致；fp32/fp64 保持原精度，数值不变；
#          计算完成后 cast 回输入 dtype——原实现在 half 输入下输出 fp32，此处为
#          统一形状-类型保持接口做的策略调整，不改滤波公式）；
#          删去未参与 forward 的固定死参数 fixed_size（仅出现在 extra_repr）；
#          cutoff 索引语义、N%4==0 的 Nyquist 修正、cutoff_low==1 的 DC 边界情形、
#          垂直维保留两端低频带的切片写法逐行保留；channels 为统一接口参数
#          （逐通道共享同一滤波器，不改变滤波数学）。

'''
模块名称：TLP (TruncLPF, Truncated Low-Pass Filter) —— 截断式 FFT 低通滤波

一、模块简介
理想低通滤波在频域上是矩形窗：保留 |f| ≤ f_c 的频率分量，完全置零高频。
这类滤波是"无混叠（alias-free）"算子的基础——任何非线性或下采样之前先做
理想低通，可避免高频能量折叠到低频带产生混叠伪影。Alias-Free ViT
（NeurIPS 2025）指出，标准 ViT 的 patchify / 下采样 / 非线性激活都会引入
混叠，导致模型对输入的小幅平移极其敏感（平移等变性差）；引入可微的
截断式低通滤波算子（TruncLPF）是其消除混叠、提升平移鲁棒性的核心组件。

TruncLPF 的核心思想是：在二维实值 FFT（rfft2）谱上直接把高频区域置零
（而非乘一个软窗），再逆变换回空间域。这种"截断"实现即理想低通，计算
简单、无额外可学习参数，且对任意通道数共享同一频域掩码语义。

核心创新点：
1. 截断即理想低通：频域矩形窗等价于 sinc 卷积核，真正消除带外能量；
2. 实值 FFT 半谱处理：用 rfft2/irfft2 节省一半频谱存储，仅需处理
   半谱的高频区域；
3. Nyquist 修正：当空间尺寸 N % 4 == 0 时对截断索引减 1，保持对称性；
4. DC 边界情形：只保留直流分量时的特殊切片写法，避免退化为无效切片。

二、结构设计
TLP 由以下子结构组成（无学习参数；形状以输入 [B,C,H,W] 计）：
1. 截断索引计算：cutoff_low = int((N * cutoff) // 2) + 1，其中 N = W
   （末维尺寸）；若 N % 4 == 0 则 cutoff_low -= 1；
2. 二维实值 FFT：x_fft = torch.fft.rfft2(x)，谱形状 [B,C,H,W//2+1]；
3. 垂直维（dim=-2，全尺寸 H）置零：
   - cutoff_low == 1 时：x_fft[..., 1:, :] = 0（仅保留 DC 行）；
   - 否则：x_fft[..., cutoff_low: -cutoff_low+1, :] = 0（保留两端低频带）；
4. 水平维（dim=-1，半谱 W//2+1）置零：x_fft[..., :, cutoff_low:] = 0
   （仅保留最低 cutoff_low 个频率）；
5. 逆变换：out = torch.fft.irfft2(x_fft, s=(H, W))，回到 [B,C,H,W]。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 Alias-Free ViT（NeurIPS 2025）的截断式 FFT 低通滤波 TruncLPF
对特征施加理想低通：在二维实值 FFT 谱上截断高频分量后逆变换回空间域，
以消除下采样/非线性引起的混叠伪影，提升模型的平移鲁棒性与等变性。"

原论文引用格式：
Alias-Free ViT: Fractional Shift Invariance via Linear Attention. NeurIPS 2025.

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用的抗混叠
滤波插件置于下采样、patch-merging、非线性激活前后。特别适合对平移敏感
（需要平移等变/鲁棒）的场景，如医学影像、细粒度识别、安全敏感评测。
注意：本模块对输入末维 W 计算截断索引（与原实现一致），非正方形输入时
请确认 cutoff 语义符合预期；FFT 计算在 float32/float64 下进行。
'''


class TLP(nn.Module):
    """TLP: TruncLPF —— 截断式 FFT 理想低通滤波（频域置零高频，无学习参数）"""

    def __init__(self, channels: int, cutoff: float = 0.5):
        super().__init__()
        assert channels > 0, 'channels must be positive'
        assert 0.0 < cutoff <= 1.0, 'cutoff must be in (0, 1]'
        self.channels = channels
        self.cutoff = cutoff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        assert x.dim() == 4, f'expected [B,C,H,W], got {tuple(x.shape)}'
        assert x.shape[1] == self.channels, \
            f'channel mismatch: x has {x.shape[1]}, module configured with {self.channels}'

        # FFT 数值稳定性：半精度提升到 float32（原 autocast/float() 路径的意图）
        out_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        N = x.shape[-1]

        # Compute cutoff indices（原样保留）
        cutoff_low = int((N * self.cutoff) // 2) + 1
        if N % 4 == 0:
            cutoff_low = cutoff_low - 1

        # Perform 2D RFFT
        x_fft = torch.fft.rfft2(x)

        # Zero out high frequencies in both dimensions
        # Vertical direction (full dimension)
        if cutoff_low == 1:
            # edge case where keeping only DC frequency - end index is "0"
            x_fft[..., cutoff_low:, :] = 0
        else:
            x_fft[..., cutoff_low: -cutoff_low + 1, :] = 0

        x_fft[..., :, cutoff_low:] = 0

        # Inverse transform with original dimensions
        out = torch.fft.irfft2(x_fft, s=(x.shape[-2], x.shape[-1]))

        return out.to(out_dtype)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, cutoff={self.cutoff}"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = TLP(channels=128)
    output = model(input_tensor)
    print('=== TLP: TruncLPF ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
