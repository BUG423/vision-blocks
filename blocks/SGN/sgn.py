import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight SR (CVPR 2026)
# 论文链接：https://arxiv.org/abs/2603.11680
# 代码来源：https://github.com/hokiyoshi/UCAN
# 原始许可证：Apache-2.0
# 模块出处：basicsr/archs/ucan_arch.py 的 SGFN 类（含其强耦合子模块 SpatialGate）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删除 timm / einops / basicsr 依赖与注册表；token 接口 [B,H*W,C] 包装为统一
#          4D 接口 [B,C,H,W]（内部 flatten(2).transpose / transpose.reshape 还原，不改数值）；
#          in_features/out_features 统一为 channels（原默认 out=in、hidden=in，保持一致）；
#          删除未使用的 act_layer（原 forward 中 self.act 调用已被注释，属死代码，去掉不影响数值）；
#          SpatialGate 原样保留（split → LayerNorm → 7×7 DW Conv → gelu(x1)*x2）；
#          原 x_size 参数改为从 x.shape 自动推断；hidden_features 为奇数时 chunk(2) 与
#          fc2 通道数会不一致，补 assert。门控融合公式不变。

'''
模块名称：SGN (Spatial-Gate Fusion Network) —— 空间门控特征融合

一、模块简介
UCAN 中的 SGFN（Spatial-Gate Feed-Forward Network）是一种空间门控融合
前馈模块：先把通道经 1×1 线性升/变换，再沿通道对半切分，一半经 LayerNorm
与 7×7 深度卷积编码空间上下文，另一半经 GELU 激活后与之逐元素相乘，
形成"内容 × 空间上下文"的门控，最后由 1×1 线性映射回输出通道。
它把 gMLNets 的 Spatial-Gate 思想引入特征融合：用深度大卷积（7×7）显式
建模空间关系，用乘性门控抑制无关位置，实现轻量而有效的空间调制。

核心创新点：
1. 空间门控：通道对半切分，一半提供内容（GELU），一半提供空间上下文
   （LayerNorm + 7×7 深度卷积），逐元素相乘实现乘性空间调制；
2. 7×7 深度卷积：以 groups=C/2 的大核深度卷积编码中等范围空间依赖，
   参数量仅 49·C/2；
3. 融合前馈：fc1 → SpatialGate → fc2 的三段结构，可替换 Transformer FFN；
4. 即插即用：输入输出同形，可嵌入任意 2D 特征管线。

二、结构设计
SGN 由以下子结构组成（形状以输入 [B, C, H, W] 计，N = H*W，hidden 默认 = C）：
1. fc1：Linear(C → hidden)，token 化后作用，[B, N, C] → [B, N, hidden]；
2. SpatialGate(hidden//2)：
   - 切分：x1, x2 = chunk(2, dim=-1)，各 [B, N, hidden//2]；
   - x2 → LayerNorm(hidden//2) → 还原 [B, hidden//2, H, W] → 7×7 深度卷积
     （padding=3，groups=hidden//2）→ flatten 回 [B, N, hidden//2]；
   - 输出 F.gelu(x1) * x2，[B, N, hidden//2]；
3. fc2：Linear(hidden//2 → C)，[B, N, hidden//2] → [B, N, C]；
4. Dropout(drop) 贯穿各段（默认 0）；
5. 4D 包装：x.flatten(2).transpose(1,2) 进，transpose(1,2).reshape 出，
   形状保持 [B, C, H, W]。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 UCAN（CVPR 2026）的空间门控融合模块 SGN（原 SGFN）进行特征
融合：将特征通道对半切分，一半经 GELU 提供内容，另一半经 LayerNorm 与
7×7 深度卷积编码空间上下文，二者逐元素相乘形成空间门控，再由线性层映射
回输出通道，实现轻量的乘性空间调制。"

四、适用任务
适用于超分辨率、去噪、修复等复原任务，也适用于检测/分割/分类中需要
空间上下文门控的前馈/融合位置。适合作为 Transformer FFN 或普通 MLP 的
空间感知替代品。
'''


class SpatialGate(nn.Module):
    """Spatial-Gate：通道对半切分后的空间门控（UCAN 原子模块，仅 SGN 使用）"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)  # DW Conv

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, N, 2*dim] → 切分后 x1 内容、x2 空间上下文，各 [B, N, dim]
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)) \
            .flatten(2).transpose(-1, -2).contiguous()  # [B, N, dim]
        return F.gelu(x1) * x2


class SGN(nn.Module):
    """SGN: Spatial-Gate Fusion Network —— 空间门控特征融合（原 UCAN 的 SGFN）"""

    def __init__(self, channels: int, hidden_features: t.Optional[int] = None, drop: float = 0.):
        super().__init__()
        hidden = channels if hidden_features is None else hidden_features  # 原 hidden_features or in_features
        assert hidden % 2 == 0, f'hidden_features {hidden} must be even (SpatialGate 对半切分)'
        self.channels = channels
        self.hidden_features = hidden

        self.fc1 = nn.Linear(channels, hidden)
        self.sg = SpatialGate(hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        N = H * W

        # [B, C, H, W] -> [B, N, C]（token 化，统一接口包装）
        tok = x.flatten(2).transpose(1, 2)                  # [B, N, C]

        tok = self.fc1(tok)                                 # [B, N, hidden]
        tok = self.drop(tok)

        tok = self.sg(tok, H, W)                            # [B, N, hidden//2]
        tok = self.drop(tok)

        tok = self.fc2(tok)                                 # [B, N, C]
        tok = self.drop(tok)

        # [B, N, C] -> [B, C, H, W]
        out = tok.transpose(1, 2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SGN(channels=128)
    output = model(input_tensor)
    print('=== SGN: Spatial-Gate Fusion Network ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
