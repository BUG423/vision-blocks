import typing as t
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation (CVPR 2026)
# 论文链接：https://arxiv.org/abs/2603.04733
# 代码来源：https://github.com/eVI-group-SCU/FOZO
# 原始许可证：MIT
# 模块出处：models/vpt.py 的 PromptViT 类（仅可学习 prompt 注入机制；FOZO 的
#          零阶 ZO 优化器与 timm ViT 主干不在提取范围）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原 PromptViT 是 timm ViT 包装器：在 patch embedding 序列的 CLS 之后、
#          patch token 之前插入 num_prompts 个可学习 prompt token，prompt 通过
#          后续 ViT block 的自注意力影响 patch 表示。统一接口为 2D 特征 [B,C,H,W]：
#          1) 可学习参数原样保留：prompts = nn.Parameter(zeros(1, num_prompts, C))，
#             初始化 val = sqrt(6/(3*prod(patch_size)+prompt_dim))、uniform(-val,val)
#             （VPT 论文初始化），patch_size 仅用于该公式（2D 特征默认已是 patch 级，
#             取 1；传入 (16,16) 可复现 ViT 场景的 prod=256）；
#          2) prompt_injection 原样保留三段拼接顺序：第一位 token（原 CLS 槽位）
#             → prompts → 其余 token；2D 特征无 CLS，第一位空间 token 占据 CLS 槽位；
#          3) 2D 适配：无 ViT block 可供 prompt 参与自注意力，故在注入后用无参数
#             缩放点积注意力执行"token 对 prompt 行"的交互（等价于自注意力中
#             patch↔prompt 的交叉项），残差后丢弃 prompt 行、还原空间排布；
#             可学习 prompt 参数与初始化公式完全不变；
#          4) 删除 timm 依赖、ViT 主干、forward_head / layers_cls_features 等
#             主干绑定方法与 FOZO 的 ZO 优化流程。

'''
模块名称：VPT (Visual Prompt Tuning) —— 视觉提示调制

一、模块简介
Visual Prompt Tuning（VPT）在预训练 ViT 的输入序列中插入少量可学习
prompt token：prompt 置于 CLS 之后、patch token 之前，随序列一起经过
Transformer 编码，从而以极小的参数量（仅 prompt 矩阵）适配下游任务。
FOZO 在测试时自适应中沿用 PromptViT 作为可调载体（其零阶优化器不在本
模块范围）。本模块抽取其中"可学习 prompt 参数 + 序列注入"机制，并适配
到 2D 特征：prompt 向量作为可学习的空间/通道调制基，通过与空间 token 的
相似度加权注入，完成视觉提示调制。

核心创新点：
1. 可学习 prompt token：[1, num_prompts, C] 参数，VPT 论文的均匀初始化
   val = sqrt(6/(3*prod(patch_size)+C))，参数量极小；
2. 序列注入：prompt 插入 token 序列（原 CLS 之后、patch 之前），保持
   与 VPT 相同的拼接顺序；
3. 2D prompt 调制：无 ViT block 时，以无参数缩放点积注意力让每个空间
   token 聚合 prompt 行（等价于自注意力中 patch↔prompt 的交叉项），
   残差注入后还原空间排布；
4. 支持外部 prompt（prompts_tensor）传入，便于测试时优化 / 提示重置。

二、结构设计
VPT 由以下子结构组成（形状以输入 [B, C, H, W] 计，N = H*W，P = num_prompts）：
1. 可学习 prompt：nn.Parameter [1, P, C]，uniform(-val, val) 初始化，
   val = sqrt(6 / (3 * prod(patch_size) + C))；
2. token 化：x [B,C,H,W] → flatten(2).transpose(1,2) → tokens [B, N, C]；
3. prompt_injection（与原三段拼接一致）：
   cat([tokens[:, :1], prompts.expand(B,-1,-1), tokens[:, 1:]], dim=1)
   → seq [B, 1+P+(N-1), C]；prompt 行为 seq[:, 1:1+P]，
   token 行按原序还原为 cat([seq[:, :1], seq[:, 1+P:]], dim=1) [B, N, C]；
4. prompt 调制（2D 适配，替代 ViT block 中的自注意力交互）：
   attn = softmax(q @ k^T * C^{-0.5})，q = tokens [B,N,C]，
   k = v = prompt 行 [B,P,C] → attn [B,N,P]；
   out = tokens + attn @ v，[B, N, C]；
5. 还原：out.transpose(1,2).reshape(B, C, H, W)，形状保持。
num_prompts=0 时无 prompt 参数，forward 恒等返回。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 FOZO（CVPR 2026）所用的视觉提示调制 VPT：在特征序列中注入
num_prompts 个可学习 prompt 向量（VPT 论文均匀初始化），prompt 与空间
token 经相似度加权聚合完成提示调制，仅 prompt 参数可学习，开销极小。"
（原论文引用格式：PromptViT 为 VPT（arXiv:2203.12119）式的可学习提示
注入，FOZO 将其作为零阶优化的载体；本模块只保留提示注入与调制。）

四、适用任务
适用于测试时自适应（TTA）、少样本/迁移适配、图像分类、检测、分割等
需要"少参数提示适配"的任务；适合把预训练特征与任务提示解耦的场景。
FOZO 的零阶优化流程不在本模块范围，可将其输出的外部 prompts 经
prompts_tensor 注入。
'''


class VPT(nn.Module):
    """VPT: Visual Prompt Tuning —— 视觉提示调制（原 FOZO 的 PromptViT 提示机制，2D 适配）"""

    def __init__(self, channels: int, num_prompts: int = 10,
                 patch_size: t.Union[int, t.Tuple[int, ...]] = 1):
        super().__init__()
        self.channels = channels
        self.num_prompts = num_prompts
        self.prompt_dim = channels
        self.patch_size = patch_size  # 仅用于 VPT 初始化公式；2D 特征默认已是 patch 级

        if num_prompts > 0:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            # 初始化沿用 VPT（arXiv:2203.12119）：val = sqrt(6 / (3*prod(patch_size)+C))
            val = self._init_val()
            nn.init.uniform_(self.prompts.data, -val, val)

    def _init_val(self) -> float:
        """原 PromptViT：val = sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))"""
        if isinstance(self.patch_size, (tuple, list)):
            patch_prod = int(math.prod(self.patch_size))
        else:
            patch_prod = int(self.patch_size)
        return math.sqrt(6. / float(3 * patch_prod + self.prompt_dim))

    def reset(self) -> None:
        """按 VPT 初始化公式重置 prompt（与原 PromptViT.reset 一致）。"""
        if self.num_prompts > 0:
            val = self._init_val()
            nn.init.uniform_(self.prompts.data, -val, val)

    def prompt_injection(self, tokens: torch.Tensor,
                         prompts_tensor: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """与原 PromptViT.prompt_injection 一致的三段拼接（原 CLS 槽位 → prompts → 其余 token）。

        tokens: [B, N, C]；2D 特征无 CLS，第一位空间 token 占据原 CLS 槽位。
        prompts_tensor: 可选外部 prompt [B, P, C]，缺省用 self.prompts。
        """
        if self.num_prompts > 0:
            if prompts_tensor is None:
                actual_prompts = self.prompts.expand(tokens.shape[0], -1, -1)
            else:
                assert prompts_tensor.shape[0] == tokens.shape[0], \
                    'Batch size of prompts_tensor must match x'
                actual_prompts = prompts_tensor

            tokens = torch.cat((
                tokens[:, :1, :],          # 原 CLS token 槽位
                actual_prompts,            # 可学习 / 外部 prompts
                tokens[:, 1:, :],          # 其余 token（原 patch tokens）
            ), dim=1)
        return tokens

    def forward(self, x: torch.Tensor,
                prompts_tensor: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
            prompts_tensor: 可选外部 prompt [B, num_prompts, C]（缺省用 self.prompts）
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        if self.num_prompts == 0:
            return x

        # [B, C, H, W] -> [B, N, C]（token 化，统一接口包装）
        tokens = x.flatten(2).transpose(1, 2)                    # [B, N, C]

        # 原样注入 prompt（三段拼接），再按原序取出 token 行与 prompt 行
        seq = self.prompt_injection(tokens, prompts_tensor)      # [B, 1+P+(N-1), C]
        P = self.num_prompts
        q_tokens = torch.cat([seq[:, :1, :], seq[:, 1 + P:, :]], dim=1)  # [B, N, C]
        kv_prompts = seq[:, 1:1 + P, :]                                   # [B, P, C]

        # 2D 适配：无 ViT block，用无参数缩放点积注意力执行 patch↔prompt 交互
        attn = (q_tokens @ kv_prompts.transpose(1, 2)) * (self.prompt_dim ** -0.5)  # [B, N, P]
        attn = attn.softmax(dim=-1)
        out_tokens = q_tokens + attn @ kv_prompts                # [B, N, C]

        # [B, N, C] -> [B, C, H, W]
        out = out_tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = VPT(channels=128)
    output = model(input_tensor)
    print('=== VPT: Visual Prompt Tuning ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
