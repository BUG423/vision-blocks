import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Multi-modality Image Fusion under Adverse Weather: Mask-Guided Feature Restoration and Interaction (AMG-Fuse) (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2606.26812
# 代码来源：https://github.com/ixilai/AMG-Fuse
# 原始许可证：MIT (Copyright (c) 2026 AMG-Fuse Authors)
# 模块出处：model/amgfuse.py 的 ChannelFusionAttention 类（L685-763）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：仅做等价重写——(1) 删除 einops 依赖，rearrange 手写为 reshape；
#   (2) 原 forward(x, y, mask, fuse) 三输出 (out_vi, out_ir, fuse) 适配为统一接口
#   forward(x1, x2=None, mask=None, fuse=None) -> [B,C,H,W]，返回融合支路 fuse 输出；
#   x2=None 时退化为 x1 自融合（x2=x1，fuse=x1，mask=0.5 均衡权重），文档中说明；
#   (3) 通道维注意力（b head c (h w)）、L2 归一化、temperature 缩放、SE 门控融合
#   的 ChannelFusionAttention 数值逻辑逐行保持；(4) SE 隐层 dim//reduction 加 max(1,·)
#   防止小通道数崩溃（通道数≥16 时与原实现完全一致）。

'''
模块名称：CFA (Channel Fusion Attention) —— 通道融合注意力

一、模块简介
恶劣天气下的多模态图像融合（可见光-红外）需要在恢复被天气退化遮蔽的特征的同时，
跨模态交互互补信息。AMG-Fuse 提出通道融合注意力（ChannelFusionAttention）：
把每个通道的空间特征图视作一个 token，在通道维上做注意力，从而让"哪个通道的
空间内容该被融合"由内容本身决定。具体地，两个模态分支各自产生查询（q_vi 来自
被掩码调制的可见光特征，q_ir 来自被反掩码调制的红外特征），而键值对 (k, v) 来自
当前融合特征；两个查询分别与共享键值做通道注意力后，再用 SE 风格的双通道门控
（a_vi, a_ir）加权求和得到新的融合特征。这使融合权重既依赖各模态自身的空间内容，
又依赖当前融合状态，实现"以融合特征为桥"的双向模态交互。

核心创新点：
1. 通道维注意力：以 HW 维空间图作为通道 token 的特征，沿通道轴做缩放点积注意力
2. 双查询-共享键值：可见光/红外各自生成 q，k/v 来自融合特征，实现跨模态信息汇聚
3. 掩码调制输入：q 分支输入分别乘 mask 与 (1-mask)，把天气退化掩码注入注意力
4. SE 双门控融合：对两支输出拼接后经 SE 生成 (a_vi, a_ir)，加权求和产生融合特征

二、结构设计
CFA 由以下子结构组成，张量形状流转如下（C=channels，N=H*W，head=num_heads）：
1. 掩码调制：x1 = x1 * mask，x2 = x2 * (1 - mask)，[B,C,H,W]
2. 查询分支：q_vi = dwconv3x3(1x1(x1))，q_ir = dwconv3x3(1x1(x2))，[B,C,H,W]
3. 键值分支：kv = dwconv3x3(1x1(fuse)) 后 chunk 为 k, v，各 [B,C,H,W]
4. 头切分与重排：[B,C,H,W] -> [B, head, c, N]（c=C/head）；
   q/k 沿最后一维 L2 归一化
5. 通道注意力：attn_vi = softmax((q_vi @ k^T) * temperature)，同理 attn_ir；
   out_vi = attn_vi @ v，out_ir = attn_ir @ v，[B, head, c, N] -> [B,C,H,W]
6. SE 门控融合：cat([out_vi, out_ir]) [B,2C,H,W] -> GAP -> 1x1(C/16) -> ReLU
   -> 1x1(2) -> Sigmoid -> (a_vi, a_ir) [B,2,1,1]；
   fuse = a_vi * out_vi + a_ir * out_ir
7. 输出投影：out_vi/out_ir/fuse 各过 1x1 卷积；本模块返回 fuse [B,C,H,W]

三、论文写法参考
若在论文中引用该模块，可描述为：
"我们采用 AMG-Fuse 提出的通道融合注意力（CFA）[ECCV 2026]，在通道维上以融合
特征为共享键值、以掩码调制后的双模态特征为查询执行注意力，并通过 SE 双门控
加权融合两支输出，实现恶劣天气退化下的多模态特征交互与恢复。"
（原论文引用格式：AMG-Fuse authors, "Multi-modality Image Fusion under Adverse
Weather: Mask-Guided Feature Restoration and Interaction", ECCV 2026.）

四、适用任务
多模态图像融合（可见光-红外）、恶劣天气图像恢复、多源特征交互；也可作为通用
的双分支/单张量通道注意力即插即用模块用于融合类视觉任务。
'''
__all__ = ['CFA']


class CFA(nn.Module):
    """CFA: Channel Fusion Attention —— 通道融合注意力"""

    def __init__(self, channels: int, num_heads: int = 4, bias: bool = False,
                 reduction: int = 16):
        super().__init__()
        assert channels % num_heads == 0, 'channels 必须能被 num_heads 整除'
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)

        # 查询分支：可见光 / 红外各一套 1x1 + 3x3 dwconv
        self.q_vi = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.q_dwconv_vi = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                                     padding=1, groups=channels, bias=bias)

        self.q_ir = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.q_dwconv_ir = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                                     padding=1, groups=channels, bias=bias)

        # 键值分支：来自融合特征
        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1,
                                   padding=1, groups=channels * 2, bias=bias)

        self.project_out_vi = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.project_out_ir = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.project_out_f = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

        # SE 双门控：由两支输出生成 (a_vi, a_ir)
        hidden = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, head, c, H*W]
        b, c, h, w = x.shape
        return x.reshape(b, self.num_heads, c // self.num_heads, h * w)

    def _from_heads(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # [B, head, c, H*W] -> [B, C, H, W]
        b, head, c, _ = x.shape
        return x.reshape(b, head * c, h, w)

    def forward(self, x1: torch.Tensor, x2: t.Optional[torch.Tensor] = None,
                mask: t.Optional[torch.Tensor] = None,
                fuse: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            x1:  Tensor, shape = [B, C, H, W]  主特征（可见光支路）
            x2:  Tensor, shape = [B, C, H, W]  第二特征（红外支路）；None 时用 x1 自融合
            mask: Tensor, shape = [B, C, H, W] 天气退化掩码；None 时取 0.5（均衡自融合）
            fuse: Tensor, shape = [B, C, H, W] 融合特征（键值来源）；None 时用 x1
        输出:
            out: Tensor, shape = [B, C, H, W]  融合支路输出（原实现三输出中的 fuse）
        """
        if x2 is None:
            x2 = x1                      # 自融合回退
        if fuse is None:
            fuse = x1
        if mask is None:
            mask = torch.full_like(x1, 0.5)
        assert x1.shape == x2.shape, 'The shape of feature maps from image and features are not equal!'
        b, c, h, w = x1.shape

        x = x1 * mask
        y = x2 * (1 - mask)

        q_vi = self.q_dwconv_vi(self.q_vi(x))
        q_ir = self.q_dwconv_ir(self.q_ir(y))

        kv = self.kv_dwconv(self.kv(fuse))
        k, v = kv.chunk(2, dim=1)

        q_vi = self._to_heads(q_vi)      # [B, head, c, N]
        q_ir = self._to_heads(q_ir)
        k = self._to_heads(k)
        v = self._to_heads(v)

        q_vi = F.normalize(q_vi, dim=-1)
        q_ir = F.normalize(q_ir, dim=-1)
        k = F.normalize(k, dim=-1)

        attn_vi = (q_vi @ k.transpose(-2, -1)) * self.temperature
        attn_vi = attn_vi.softmax(dim=-1)

        attn_ir = (q_ir @ k.transpose(-2, -1)) * self.temperature
        attn_ir = attn_ir.softmax(dim=-1)

        out_vi = attn_vi @ v             # [B, head, c, N]
        out_ir = attn_ir @ v

        out_vi = self._from_heads(out_vi, h, w)   # [B, C, H, W]
        out_ir = self._from_heads(out_ir, h, w)

        cat = torch.cat([out_vi, out_ir], dim=1)   # [B, 2C, H, W]
        alpha = self.se(cat)                       # [B, 2, 1, 1]
        a_vi, a_ir = alpha[:, 0:1], alpha[:, 1:2]  # each [B, 1, 1, 1]

        out_vi_F = a_vi * out_vi
        out_ir_F = a_ir * out_ir

        fuse_out = out_vi_F + out_ir_F

        out_vi = self.project_out_vi(out_vi)
        out_ir = self.project_out_ir(out_ir)
        fuse_out = self.project_out_f(fuse_out)
        return fuse_out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = CFA(channels=128)
    output = model(input_tensor)
    print('=== CFA: Channel Fusion Attention ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    # 双分支模式（与原实现 forward(x, y, mask, fuse) 同构）
    x2 = torch.randn(1, 128, 64, 64)
    out2 = model(input_tensor, x2)
    print('two-branch output_size:', out2.size())
