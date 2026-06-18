import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-18

'''
模块名称：KFM (Kalman Filter Module) —— 卡尔曼滤波模块

一、模块简介
在深度网络的前向传播中，每一层接收到的特征可以被视为对某个"理想干净
信号"的有噪观测——卷积、激活、下采样等操作不可避免地引入了噪声、冗
余与信息退化。传统的残差连接或归一化方法缺乏对"预测"与"观测"两者
可信度的自适应度量：要么一律相信输入，要么一律相信变换后的结果。

卡尔曼滤波（Kalman Filter）是最经典的最优状态估计框架，其精髓在于用
一个可自适应的卡尔曼增益（Kalman Gain）在"模型预测"与"实际观测"之
间做贝叶斯加权融合——预测越不确定，越信任观测；观测噪声越大，越信任
预测。

KFM 的核心思想是：把特征图当作对潜在状态的有噪观测，在网络内部模拟
一步卡尔曼"预测-更新"循环。先用一个转移模型（transition）从当前状态
预测出"干净状态应为何"（预测），再用输入本身作为观测计算新息
（innovation = 观测 - 预测）；随后从预测状态的局部空间方差估计预测
不确定度 P，从新息的强度估计观测噪声 R，进而得到逐通道卡尔曼增益
K = P / (P + R) ∈ [0,1]；最后用 K 对预测与新息做加权融合得到更新后
的状态。整个过程是 1D 逐通道卡尔曼滤波在特征图上的自然推广。

核心创新点：
1. 卡尔曼滤波引入特征处理：将最优状态估计框架作为即插即用模块嵌入
   CNN，区别于动量平滑等启发式方法
2. 自适应卡尔曼增益：增益由预测不确定度（局部方差）与观测噪声（新息
   强度）在线计算，逐通道、逐位置自适应融合预测与观测
3. 转移-新息解耦：用 transition 显式建模"状态先验"，用 innovation 显
   式建模"观测带来的新信息"，二者语义清晰可解释
4. 可学习观测噪声先验：引入可学习的逐通道测量噪声偏置，使滤波器可在
   训练中自动校准对预测/观测的信任倾向

二、结构设计
KFM 由以下子结构组成：
1. 状态编码器（State Encoder）：
   - 1x1 卷积将特征压缩到潜在状态空间 s [B, inner, H, W]
2. 转移模型（Transition Model）：
   - 3x3 深度卷积 + 1x1 卷积建模状态动力学，产生预测状态 s_pred
3. 新息计算（Innovation）：
   - y = s - s_pred，即观测相对预测的残差信息
4. 不确定度估计与卡尔曼增益（Uncertainty & Kalman Gain）：
   - P = 预测状态的局部空间方差（avgpool 二阶矩减一阶矩平方）
   - R = 新息强度 + 可学习噪声偏置
   - K = P / (P + R)，逐通道自适应融合权重
5. 状态更新与解码（Update & Decode）：
   - s_upd = s_pred + K * y
   - 1x1 卷积解码回原通道空间
6. 输出精炼与残差连接

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 KFM（Kalman Filter Module）模块，将卡尔曼最优状态估计框架
引入卷积特征增强。该模块将特征视为对潜在状态的有噪观测，先用转移模型
产生状态预测，再以输入为观测计算新息；通过预测状态的局部空间方差估计
预测不确定度、以新息强度估计观测噪声，进而在线计算逐通道卡尔曼增益，
对预测与新息做自适应贝叶斯加权融合。该模块在模型先验与观测证据之间实
现了内容感知的自适应平衡。"

四、适用任务
适用于图像分类、目标检测、语义分割等视觉任务，可作为即插即用模块嵌入
CNN 或 Transformer 主干网络。特别适合深层网络中存在特征退化/噪声累积
的场景，以及对预测与观测可信度需要自适应权衡的任务（如去噪、图像恢复、
域自适应、低质量输入下的鲁棒识别）。
'''


class KFM(nn.Module):
    """KFM: Kalman Filter Module —— 卡尔曼滤波模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 uncertainty_kernel: int = 3):
        super().__init__()
        inner = max(1, channels // reduction)
        self.inner = inner
        self.uk = uncertainty_kernel
        self.padding = uncertainty_kernel // 2

        # State encoder: project to latent state space
        self.encode = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )

        # Transition model: predict the clean state from current state
        self.transition = nn.Sequential(
            nn.Conv2d(inner, inner, 3, padding=1, groups=inner, bias=False),
            nn.BatchNorm2d(inner),
            nn.Conv2d(inner, inner, 1, bias=False),
        )

        # Learnable per-channel measurement-noise prior (log-scale)
        self.meas_noise_bias = nn.Parameter(torch.zeros(1, inner, 1, 1))

        # State decoder: project updated state back to feature space
        self.decode = nn.Sequential(
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-position local spatial variance via moment difference.
        Var(x) ≈ E[x^2] - E[x]^2 over a local neighborhood.
        Represents prediction uncertainty P.
        """
        mu = F.avg_pool2d(x, self.uk, stride=1, padding=self.padding)
        mu_sq = F.avg_pool2d(x * x, self.uk, stride=1, padding=self.padding)
        var = (mu_sq - mu * mu).clamp_min(0.0)       # [B, inner, H, W]
        return var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        # 1. Encode to latent state space (treat as observation z)
        s = self.encode(x)                            # [B, inner, H, W]

        # 2. Predict: transition model forecasts the clean state
        s_pred = self.transition(s)                   # [B, inner, H, W]

        # 3. Innovation: residual of observation over prediction
        innov = s - s_pred                            # [B, inner, H, W]

        # 4. Uncertainty estimation & Kalman gain
        P = self._local_variance(s_pred)             # prediction uncertainty
        # Measurement-noise estimate from innovation magnitude + learnable bias
        R = innov * innov
        R = F.avg_pool2d(R, self.uk, stride=1, padding=self.padding)
        R = R + F.softplus(self.meas_noise_bias) + 1e-6
        K = P / (P + R)                               # Kalman gain in [0, 1]

        # 5. Update: Bayesian blend of prediction and observation
        s_upd = s_pred + K * innov                   # [B, inner, H, W]

        # 6. Decode, refine and residual
        out = self.decode(s_upd)                      # [B, C, H, W]
        out = self.refine(out)
        return out + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = KFM(channels=128)
    output = model(input_tensor)
    print('=== KFM: Kalman Filter Module ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
