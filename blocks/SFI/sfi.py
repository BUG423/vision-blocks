import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：LaDy: Lagrangian-Dynamic Informed Network via Spatial-Temporal Modulation (CVPR 2026)
# 论文链接：https://github.com/HaoyuJi/LaDy （仓库 README 给出论文全文标题，未提供 arXiv 链接）
# 代码来源：https://github.com/HaoyuJi/LaDy
# 原始许可证：MIT
# 模块出处：libs/models/LaDy.py 的 SFI 类（基准流投影对应 STI 类中的 conv_t 角色）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：原 SFI 是骨架动作分割模块：双输入 feature_s(n,c,t,v)（骨架空间特征）与
#          feature_t(n,n_features,t)（时间/动力学特征），输出 (n,n_features,t) 并乘 mask。
#          统一接口只有一个 4D 输入 x[B,C,H,W]，做如下适配（调制公式不变）：
#          1) (n,c,t,v) 折叠 view(n, v*c, t) 在 2D 下取 v=1、t'=H*W，即 x.flatten(2)
#             得 [B,C,HW]，Conv1d 沿空间 token 轴作用（原"时间轴"→展开后的空间轴）；
#          2) 原为双输入，基准流 feature_t 在原架构中由 STI.conv_t 对 4D 块投影得到，
#             此处同样以 conv_t: Conv1d(C→n_features,1) 从同一输入导出，保持与原
#             "两路 1×1 投影 → 通道维拼接 → 1×1 融合 → MLP+残差"公式一致；
#          3) 删除未使用的 self.softmax（原 einsum 注意力路径整段被注释，属死代码）；
#          4) mask 为时序有效位填充掩码，2D 特征全有效，去掉（等价恒等掩码）；
#          5) n_features≠channels 时补 1×1 out_proj 还原通道（原场景 n_features 即输出
#             通道数；默认 n_features=channels 时为 Identity，公式零改动）。
#          6) DynamicFeatureFusion 与 SFI 无耦合（在 STI 循环中独立做时序门控），不纳入。

'''
模块名称：SFI (Spatial Feature Injection) —— 空间特征注入

一、模块简介
LaDy 面向骨架时序动作分割：把拉格朗日动力学合成的广义力与骨架空间表示
融合，增强类间可分性与边界感知。SFI（Spatial Feature Injection）承担其中
"空间注入"一步：把高维空间/骨架特征压缩投影后，与基准特征（原为动力学
驱动的时间特征）在通道维拼接融合，再经 MLP 与残差细化，使空间语义注入到
逐帧（逐 token）特征中。核心思想是"空间流 → 通道压缩 → 融合 → 残差注入"。

核心创新点：
1. 空间折叠注入：把骨架 (c,t,v) 的关节维 v 折入通道，经 1×1 卷积压缩到
   n_features，实现空间结构向时间（token）序列的注入；
2. 双路融合：空间注入流与基准流在通道维 concat 后由 1×1 卷积融合，
   而非简单相加，保留两路互补信息；
3. MLP 残差细化：融合结果过 Linear-GELU-Dropout(0.3)-Linear 后加基准流
   残差，稳定训练；
4. 轻量即插即用：仅 1×1 卷积与两层 MLP，可嵌入任意序列/特征管线。

二、结构设计
SFI 由以下子结构组成（形状以统一输入 [B, C, H, W] 计，N = H*W，n_features 默认 = C）：
1. 空间折叠：x [B,C,H,W] → flatten(2) → tokens [B,C,N]，
   对应原 (n,c,t,v)→(n,v*c,t) 折叠在 v=1、t=N 时的退化形式；
2. 空间注入流 conv_s：Conv1d(C→n_features, 1×1)，[B,C,N] → [B,n_features,N]；
3. 基准流 conv_t：Conv1d(C→n_features, 1×1)，同形（对应原 STI.conv_t 角色，
   原 SFI 的第二输入 feature_t 在原架构中由此类投影产生）；
4. 融合 conv_fusion：Conv1d(2*n_features→n_features, 1×1) 作用于
   cat([feature_t, feature_s], dim=1)（注意 concat 顺序与原一致：基准在前）；
5. MLP 残差：permute → [B,N,n_features] → Linear(n_features,n_features)
   → GELU → Dropout(0.3) → Linear(n_features,n_features) → permute 回
   [B,n_features,N]，再 + feature_t；
6. 输出：out_proj（n_features=channels 时为 Identity）→ reshape
   [B,C,H,W]，形状保持。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 LaDy（CVPR 2026）的空间特征注入模块 SFI：将空间结构特征经
1×1 卷积压缩后与基准特征在通道维拼接融合，再经 MLP 与残差连接完成空间
语义注入。"（原论文引用格式：SFI 为 LaDy 空间-时间调制 STM 中的
Spatial Feature Injection 子模块。）

四、适用任务
原生适用于骨架时序动作分割（STAS）中的空间-时间调制；统一为 4D 接口后，
适用于图像分类、检测、分割、视频理解等需要"空间上下文注入 + 门控融合"的
任务，可作为通用的双路特征融合即插即用模块。
'''


class SFI(nn.Module):
    """SFI: Spatial Feature Injection —— 空间特征注入（原 LaDy 的 SFI，2D 适配）"""

    def __init__(self, channels: int, n_features: t.Optional[int] = None, drop: float = 0.3):
        super().__init__()
        n_features = channels if n_features is None else n_features  # 原 STI 中 n_features=64
        self.channels = channels
        self.n_features = n_features

        self.conv_s = nn.Conv1d(channels, n_features, 1)          # 空间注入流（原 conv_s）
        self.conv_t = nn.Conv1d(channels, n_features, 1)          # 基准流（原 STI.conv_t 角色）
        self.ff = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.GELU(),
            nn.Dropout(drop),                                     # 原 0.3
            nn.Linear(n_features, n_features),
        )
        self.conv_fusion = nn.Conv1d(2 * n_features, n_features, 1)
        # n_features=channels（默认）时为 Identity，数值公式与原完全一致
        self.out_proj = nn.Identity() if n_features == channels else nn.Conv1d(n_features, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'

        # 原 (n,c,t,v)→(n,v*c,t) 折叠：2D 下 v=1, t=H*W → tokens [B, C, N]
        tokens = x.flatten(2)                                     # [B, C, N]

        feature_s = self.conv_s(tokens)                           # [B, n_features, N] 空间注入流
        feature_t = self.conv_t(tokens)                           # [B, n_features, N] 基准流

        # 通道维拼接（基准在前，与原 cat([feature_t, feature_s]) 一致）→ 1×1 融合
        feature_cross = self.conv_fusion(torch.cat([feature_t, feature_s], dim=1))  # [B, n_features, N]
        feature_cross = feature_cross.transpose(1, 2)             # [B, N, n_features]
        feature_cross = self.ff(feature_cross).transpose(1, 2) + feature_t           # MLP + 残差

        out = self.out_proj(feature_cross)                        # [B, n_features|C, N]
        return out.reshape(B, -1, H, W)                           # [B, C, H, W]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SFI(channels=128)
    output = model(input_tensor)
    print('=== SFI: Spatial Feature Injection ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
