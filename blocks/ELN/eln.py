import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Enhancing Out-of-Distribution Detection with Extended Logit Normalization (CVPR 2026)
# 论文链接：https://arxiv.org/abs/2504.11434
# 代码来源：https://github.com/limchaos/ElogitNorm
# 原始许可证：MIT
# 模块出处：elogitnorm_loss.py 的 elogitnorm_loss 函数（另见
#          openood/trainers/elogitnorm_trainer.py 的 ELogitNormLoss 类，公式相同）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删除 OpenOOD 训练器/配置/registry 体系，仅保留损失公式本身；
#          torch.einsum / 原地填充问题沿用官方实现的 +I 对角规避（注释保留）；
#          损失为 hyperparameter-free（论文摘要原文），alpha 为可选温度系数，
#          默认 1.0 时与原实现逐位一致；补 torch.no_grad 检查外不改任何数值路径。
# 注意：本模块是【损失函数】，不是特征变换模块，输出为标量损失（见下）。

'''
模块名称：ELN (Extended Logit Normalization loss) —— 扩展 Logit 归一化损失
【例外说明】输出为标量 loss，而非 [B, C, H, W] -> [B, C, H, W] 特征变换。

一、模块简介
Out-of-Distribution（OOD）检测要求模型对分布外样本给出低置信度分数。
LogitNorm 通过把 logits 除以特征范数再做交叉熵来校准置信度，但本文发现
它存在 feature collapse 现象：类内特征被过度压缩、类间判别信息受损。
ELogitNorm 在 LogitNorm 基础上引入"特征距离感知"的实例级缩放：对每个
样本，计算其 logit 与最大 logit 的差值（gaps），除以对应分类器权重的
两两距离，再取均值得到实例级缩放因子 scale，最后用 logits / scale 做
交叉熵。该公式 hyperparameter-free，同时改善 OOD 检测与 ID 置信度校准，
且不牺牲 ID 分类精度。

核心创新点：
1. 指出 LogitNorm 的 feature collapse 问题；
2. 特征距离感知缩放：scale 由 logit 间隙 / 分类器权重两两距离 逐实例给出；
3. hyperparameter-free：无温度系数等需要调的超参；
4. 与后处理 OOD 评分函数（MSP / Energy / Mahalanobis 等）广泛兼容。

二、结构设计
ELN 为损失模块（无可学习参数），前向接收：
1. logits: [N, C] —— 分类器输出；
2. fc_weight: [C, D] —— 最后一层线性分类器权重（每类一个 D 维向量）；
3. target: [N,] —— 类别标签。
计算流程：
a) w_diff = fc_weight.unsqueeze(1) - fc_weight.unsqueeze(0)  → [C, C, D]；
b) denom = ||w_diff||_2 → [C, C]；denom = denom + I（对角为 0，+I 规避
   对 fc_weight 的 autograd 原地填充问题）；
c) values, nn_idx = logits.max(dim=1) → 最大 logit 与预测类索引；
d) gaps = |logits - values| → [N, C]；
e) scale = mean_C(gaps / denom[nn_idx]) → [N, 1]；
f) loss = CrossEntropyLoss(logits / scale, target) → 标量。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"训练阶段采用 ELogitNorm（CVPR 2026）损失：在 LogitNorm 的基础上，以
分类器权重的两两距离对 logit 间隙进行归一化，得到逐实例缩放因子，并以
缩放后的 logits 计算交叉熵。该方法为 hyperparameter-free，可显著改善
分布外检测与置信度校准，同时保持分布内分类精度。"

四、适用任务
面向 OOD 检测 / 置信度校准的分类训练（CIFAR-10/100、ImageNet-200 等
OpenOOD 基准），也可用于任何需要 logit 校准的分类任务。通常与后处理
OOD 评分函数（MSP、Energy、Mahalanobis、kNN 等）配合使用。
'''

__all__ = ['ELN', 'elogitnorm_loss']


def elogitnorm_loss(logits: torch.Tensor, fc_weight: torch.Tensor,
                    target: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELogitNorm 损失（与论文附录公式一致），返回标量 loss。

    参数:
        logits:    [N, C] 分类器输出
        fc_weight: [C, D] 最后一层线性分类器权重
        target:    [N,]   类别标签
        alpha:     可选温度系数。论文为 hyperparameter-free；默认 1.0 时
                   与原实现完全一致（scaled_logits = logits / scale / alpha）。
    返回:
        标量 Tensor（shape = []）
    """
    # Pairwise classifier-weight differences
    w_diff = fc_weight.unsqueeze(1) - fc_weight.unsqueeze(0)          # [C, C, D]
    denom = torch.norm(w_diff, dim=2)                                  # [C, C]
    # diag(denom) is 0; +I avoids the in-place fill that breaks autograd through fc_weight.
    denom = denom + torch.eye(denom.size(0), device=denom.device, dtype=denom.dtype)
    # Maximum logit and predicted class
    values, nn_idx = logits.max(dim=1)                                 # [N,], [N,]
    # Logit gaps
    gaps = (logits - values.unsqueeze(1)).abs()                        # [N, C]
    # Instance-wise scaling factor
    scale = (gaps / denom[nn_idx]).mean(dim=1, keepdim=True)           # [N, 1]
    # ELogitNorm objective
    scaled_logits = logits / scale / alpha                             # alpha=1 -> 原实现
    loss = F.cross_entropy(scaled_logits, target)
    return loss


class ELN(nn.Module):
    """ELN: Extended Logit Normalization loss —— 扩展 Logit 归一化损失（输出为标量）"""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        # 论文 hyperparameter-free；alpha=1.0 时与官方实现一致
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, fc_weight: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        输入:
            logits:    Tensor, shape = [N, C]
            fc_weight: Tensor, shape = [C, D]（最后一层线性分类器权重）
            target:    Tensor, shape = [N,]（整型类别标签）
        输出:
            loss: 标量 Tensor（shape = []）——【损失模块，非 [B,C,H,W] 特征变换】
        """
        return elogitnorm_loss(logits, fc_weight, target, alpha=self.alpha)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    N, C, D = 8, 10, 16
    logits = torch.randn(N, C)
    fc_weight = torch.randn(C, D)
    target = torch.randint(0, C, (N,))

    model = ELN()                    # alpha=1.0，与论文/官方实现一致
    loss = model(logits, fc_weight, target)
    print('=== ELN: Extended Logit Normalization loss ===')
    print('logits_size:', logits.size())
    print('fc_weight_size:', fc_weight.size())
    print('target_size:', target.size())
    print('loss:', loss.item(), 'shape:', tuple(loss.shape))
    print('params:', count_parameters(model))
    # 功能式接口
    loss_fn = elogitnorm_loss(logits, fc_weight, target)
    print('functional_loss:', loss_fn.item())
