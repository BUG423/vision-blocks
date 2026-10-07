import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：Sparsemax SAE: Improving Sparse Autoencoder with Dynamic Attention (CVPR 2026)
# 论文链接：https://github.com/qyj-bkjx/Sparsemax-SAE （openaccess CVPR 2026，仓库未附论文直链）
# 代码来源：https://github.com/qyj-bkjx/Sparsemax-SAE
# 原始许可证：MIT
# 模块出处：src/sae_training/sparse_autoencoder.py 的 SparseAutoencoder 类
#          （核心为 sparsemax 算子 + forward_standard 的交叉注意力 SAE 路径）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删除 transformer_lens（HookPoint/HookedRootModule）、einops、jaxtyping、
#          geom_median、tqdm 与 cfg 配置体系；删除训练管线（mse/l1 损失、ghost residual、
#          resample_neurons_*、initialize_b_dec_*、save/load、get_test_loss）与评测工具
#          （forward_clamp、forward_gated 空实现）及备用激活（rectangle/StepFunction/
#          JumpReLUFunction/_pow_masked，forward_standard 均不引用）；删除 forward_standard
#          未使用的 W_enc/b_enc/W_dec/b_dec、log_threshold/bandwidth/topk_k、gated 的
#          r_mag/b_mag（死代码，不影响数值）。cfg.d_in→channels、cfg.d_sae→n_features
#          （默认 channels*4，原训练 post_init 硬编码 expansion=64）；硬编码 idf 阈值
#          0.1→参数 idf_threshold；idf_socre 由标量 0.0 改为 zeros(n_features) 缓冲
#          （数值等价：首次 EMA 更新后同为激活频次均值）。1D 逐向量自编码适配为 4D：
#          内部 flatten(2).transpose 成 token 序列 [B,N,C] 逐 token 重构再 reshape 还原，
#          SAE 本为逐向量运算，数值不变。核心公式原样保留：
#          x_n = x/‖x‖；scores = x_n @ (concept@k)^T；idf 频次掩码；
#          sparsemax 动态稀疏支撑；acts = scores * (sparsemax>0)（pre-activation ×
#          支撑掩码，非 sparse_probs 本身）；x_rec = acts @ (concept@v) * ‖x‖；
#          concept 初始化保留 kaiming_uniform_(concept.T)（转置视图初始化）。

'''
模块名称：SAE (Sparsemax Sparse Autoencoder) —— Sparsemax 动态稀疏自编码器

一、模块简介
稀疏自编码器（SAE）用于从基础模型激活中解耦可解释概念特征。经典 SAE 用
固定稀疏度（TopK / L1）约束隐层，稀疏度是超参而非数据驱动。Sparsemax SAE
把 SAE 重构为交叉注意力模块：概念字典 concept 充当 Key/Value，输入激活经
可学习 k、v 两侧投影，打分后用 sparsemax（可微、输出稀疏概率的 softmax
替代）自适应选择每个 token 激活的概念数量，实现"动态稀疏"；配合激活频次
（idf）掩码压制过热特征。重构误差更低、特征更干净，且稀疏度由数据自动决定。

核心创新点：
1. 交叉注意力式 SAE：scores = x̂ @ (concept@k)^T，x_rec = acts @ (concept@v)，
   概念字典与 k/v 投影联合学习，等价 Q=x̂、K=concept@k、V=concept@v 的注意力；
2. Sparsemax 动态稀疏：对打分向量取 sparsemax，支撑集大小（激活概念数）
   由输入自适应决定，替代固定 TopK；
3. 支撑掩码激活：acts = scores ⊙ [sparsemax(scores)>0]，以稀疏支撑门控
   pre-activation，保留打分幅值；
4. idf 频次掩码：对激活频次做运行均值统计，压制高频（过热）概念，
   提供数据驱动的稀疏引导。

二、结构设计
SAE 由以下子结构组成（形状以输入 [B, C, H, W] 计，N = H*W，n_features 默认 = 4C）：
1. 可学习参数：
   - concept: [n_features, C]，kaiming_uniform_(concept.T) 初始化（原样）；
   - k, v: [C, C]，kaiming_uniform_ 初始化（原样）；
   - idf_socre 缓冲: [n_features]，激活频次运行均值；
2. token 化：x [B,C,H,W] → flatten(2).transpose(1,2) → [B,N,C]（逐向量运算）；
3. 编码打分：x_n = x/‖x‖（沿 C 维单位化，无 eps，原样）；
   scores = x_n @ (concept@k)^T，[B,N,C]→[B,N,n_features]；
4. idf 掩码：scores ← scores * (1 - (idf_socre > idf_threshold))；
5. sparsemax 动态稀疏：降序排序 → 累积和 → 支撑阈值 τ →
   sparse_probs = clamp(scores - τ, min=0)；支撑 mask = (sparse_probs > 0)；
   acts = scores * mask，[B, N, n_features]；
6. idf 更新：idf_socre ← EMA(acts>0 的批均值)（随 forward 运行统计）；
7. 解码重构：x_rec = acts @ (concept@v) * ‖x‖，[B,N,n_features]→[B,N,C]；
8. 还原：transpose(1,2).reshape(B, C, H, W)，形状保持（输出为输入的重构）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 Sparsemax SAE（CVPR 2026）的动态稀疏自编码器对特征做概念级
解耦：将 SAE 重构为交叉注意力形式，概念字典经可学习 k/v 投影与单位化输入
打分，并用 sparsemax 以数据驱动方式自适应选择每个 token 激活的概念数，
配合激活频次掩码压制过热特征。"（原论文引用格式：Sparsemax SAE:
Improving Sparse Autoencoder with Dynamic Attention。）

四、适用任务
原生用于基础模型（ViT 等）激活的概念解耦、可解释性分析、特征干预与
下游稀疏特征分类；作为即插即用块，适用于任何需要"过完备稀疏重构 / 概念
分解"的表征任务（探针分析、特征可视化、模型编辑）。训练期的死神经元
重采样与损失计算不在本模块范围。
'''


class SAE(nn.Module):
    """SAE: Sparsemax Sparse Autoencoder —— Sparsemax 动态稀疏自编码器（原 SparseAutoencoder）"""

    def __init__(self, channels: int, n_features: t.Optional[int] = None,
                 idf_threshold: float = 0.1):
        super().__init__()
        d_in = channels
        d_sae = channels * 4 if n_features is None else n_features  # 原训练 cfg 实际用 64×
        self.channels = d_in
        self.n_features = d_sae
        self.idf_threshold = idf_threshold  # 原硬编码 0.1

        # 概念字典与 k/v 投影（交叉注意力式 SAE 的全部可学习参数）
        self.concept = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.kaiming_uniform_(self.concept.T)          # 原样：转置视图上初始化
        self.v = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_in, d_in)))
        self.k = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_in, d_in)))

        # idf 激活频次运行统计（原 idf_socre 标量 0.0，改为缓冲数值等价）
        self.register_buffer('idf_socre', torch.zeros(d_sae))
        self.count_batch = 0

    def sparsemax(self, score: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Sparsemax：可微、输出稀疏概率的 softmax 替代（原方法逐行保留）。

        降序排序 → 累积和 → 选取支撑阈值 τ → clamp(score - τ, min=0)。
        """
        s_sorted, _ = torch.sort(score, descending=True, dim=dim)
        s_cumsum = torch.cumsum(s_sorted, dim=dim)
        k = torch.arange(1, s_sorted.size(dim) + 1, dtype=score.dtype, device=score.device)
        bound = 1 + k * s_sorted > s_cumsum
        k_selected = torch.max(k * bound, dim=dim, keepdim=True)[0]
        tau = (torch.sum(s_sorted * bound, dim=dim, keepdim=True) - 1) / k_selected
        return torch.clamp(score - tau, min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]   # 逐 token 的 SAE 重构
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        N = H * W

        # [B, C, H, W] -> [B, N, C]（token 化；SAE 逐向量运算）
        tokens = x.flatten(2).transpose(1, 2)                          # [B, N, C]

        self.count_batch += 1
        x_n = tokens / torch.norm(tokens, dim=-1, keepdim=True)        # 单位化（原样，无 eps）

        # 交叉注意力式编码：scores = x_n @ (concept@k)^T
        concept_k = self.concept @ self.k                              # [n_features, C]
        hidden_pre = x_n @ concept_k.transpose(0, 1)                   # [B, N, n_features]

        # idf 频次掩码：压制激活频次过高的概念
        idf_mask = (self.idf_socre > self.idf_threshold).to(hidden_pre.dtype)
        hidden_pre = hidden_pre * (1 - idf_mask)

        # sparsemax 动态稀疏：支撑集大小随输入自适应
        sparse_probs = self.sparsemax(hidden_pre)
        mask = (sparse_probs > 0).to(hidden_pre.dtype)
        feature_acts = hidden_pre * mask                               # [B, N, n_features]

        # idf 运行均值更新（EMA，与原公式一致）
        self.idf_socre = (self.idf_socre * (self.count_batch - 1)
                          + (feature_acts.detach().view(-1, self.n_features) > 0).float().mean(0))
        self.idf_socre = self.idf_socre / self.count_batch

        # 解码重构：x_rec = acts @ (concept@v) * ‖x‖
        concept_v = self.concept @ self.v                              # [n_features, C]
        x_rec = feature_acts @ concept_v                               # [B, N, C]
        x_rec_scale = x_rec * torch.norm(x.flatten(2).transpose(1, 2), dim=-1, keepdim=True)

        # [B, N, C] -> [B, C, H, W]
        out = x_rec_scale.transpose(1, 2).reshape(B, C, H, W)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = SAE(channels=128)
    output = model(input_tensor)
    print('=== SAE: Sparsemax Sparse Autoencoder ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
