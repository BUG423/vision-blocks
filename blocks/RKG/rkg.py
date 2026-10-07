import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 论文：RankSEG: a consistent ranking-based framework for segmentation (JMLR 2023, 24(224))
#       + RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment
#         Approximation (NeurIPS 2025)
# 论文链接：https://www.jmlr.org/papers/v24/22-0712.html ；
#          https://arxiv.org/abs/2510.15362（RankSEG-RMA）
# 代码来源：https://github.com/rankseg/rankseg
# 原始许可证：BSD-3-Clause
# 模块出处：rankseg/_rankseg.py 的 RankSEG 类 / rankseg/functional.py 的 rankseg 分发 /
#          rankseg/_rankseg_algo.py 的 rankseg_rma（RMA 求解器）核心排序重标注
#          （compute_opt_tau 的 Dice/IoU 一致性打分 + full_sort_masks 的排序截断 +
#          convert_to_nonoverlap 的增量得分冲突消解）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：仅保留 RMA（Reciprocal Moment Approximation）求解器——它是 Dice/IoU
#          两种度量下唯一被官方接口允许的 multiclass 求解器（BA/TRNA 仅支持
#          multilabel Dice 且依赖 scipy 的 RefinedNormalPB，ACC 走 argmax/TR，
#          均非"排序重标注"核心，已删）；删去 safe_screening 实验分支、CUDA/Triton
#          融合核 (_screening*.py)、unassigned_policy='void'/void_index 弃权出口
#          （固定为默认 'max_score'）；RMA 的排序截断打分公式（Dice：
#          2·S_τ/(μ+τ+1+smooth) + smooth/(μ+τ+smooth)；IoU：(S_τ+smooth)/
#          (μ−S_τ+τ+smooth)；含 τ=0 空掩码项与 smooth>1e6 的仿射缩放稳定写法）、
#          类剪枝、多类冲突时的增量得分 argmax 消解规则逐行保留；
#          掩码重建由 batch×class 双重循环改为等价 scatter_ 向量化（原仓库 CUDA
#          路径同款写法，数值不变）；新增 shape assert。

'''
模块名称：RKG (RankSEG) —— 一致性排序分割重标注

一、模块简介
语义分割模型输出逐像素的类别概率图后，标准做法是逐像素取 argmax 得到标签图。
但 argmax 最大化的是逐像素准确率（Accuracy），并非医学/遥感等领域更关心的
Dice、IoU 等区域重叠度量——两者最优解并不一致：为了最大化 Dice，应当对每个
类的像素按概率排序后选取一个最优截断体积，而不是逐像素独立决策；类与类之间
还可能争抢同一像素，需要全局协调。RankSEG（JMLR 2023）首次给出了与 Dice/IoU
一致的排序式重标注框架：把分割预测形式化为"对每个类选一个前 τ 大概率像素
集合"的组合优化，并给出可计算的近似最优 τ。RankSEG-RMA（NeurIPS 2025）进一步
提出倒数矩近似（Reciprocal Moment Approximation），用封闭形式直接给出最优
截断体积，把逐类搜索的复杂度降到一次排序 + 一次 cumsum。

核心创新点：
1. 度量一致的排序重标注：对每个类按概率降序排列，选前 τ 个像素为正类，
   τ 由 Dice/IoU 的近似期望目标 argmax 决定（而非逐像素 argmax）；
2. 倒数矩近似（RMA）：用 S_τ（前 τ 概率和）与 μ（全图概率和）的倒数矩
   闭式近似 E[Dice] / E[IoU]，一次向量化计算所有 τ 的得分；
3. 类剪枝：最大概率 ≤ pruning_prob 的类整体跳过（无前景证据）；
4. 冲突消解：重叠像素只允许"选中了该像素的类"竞争，按加入该像素对目标
   的增量得分 argmax 归属；无人认领像素按活跃类增量得分兜底（max_score）。

二、结构设计
RKG 为纯推理期后处理（无学习参数），流程如下（形状以输入 [B,K,H,W]、
D = H*W 计）：
1. 预处理：probs 展平为 [B,K,D]；fp16/bfloat16 提升到 fp32；μ = probs.sum(-1)
   为 [B,K]；
2. 排序：sorted_prob, top_index = sort(probs, dim=-1, descending=True)；
   active_mask = sorted_prob[:,:,0] > pruning_prob（[B,K]）；
   cumsum_prob = cumsum(sorted_prob, -1)（前 τ 概率和 S_τ）；
3. 最优截断体积 τ*（compute_opt_tau）：
   - Dice：score(τ) = 2·S_τ/(μ+τ+1+smooth) + smooth/(μ+τ+smooth)；
   - IoU：score(τ) = (S_τ+smooth)/(μ−S_τ+τ+smooth)；
   - τ=0 空掩码项：smooth>0 时为 smooth/(μ+smooth)（Dice/IoU 同式），否则 0；
   - smooth>1e6 时改用仿射缩放写法（数值稳定，argmax 等价）；
   - τ* = argmax over {0,1,...,D}；
4. 掩码重建：rank 位置 < τ* 且类活跃 → 该类二值掩码 overlap_preds [B,K,D]
   （按 top_index 还原到像素序）；
5. 非重叠化（convert_to_nonoverlap）：
   - 单选像素固定给该类；对每类累计已固定像素的概率和 μ_k 与个数 n_k；
   - 候选像素 j 加入类 k 的增量得分（Dice/IoU 各自闭式，含 smooth 分支）；
   - 重叠像素：仅"选中该像素的类"有资格，按增量得分 argmax 归属；
   - 无主像素：活跃类（全剪枝时为全部类）按增量得分 argmax 兜底；
6. 输出 [B,H,W] int64 标签图。

★ 形状例外（非形状保持）★
本模块为评测期后处理：forward(probs: [B,K,H,W]) -> [B,H,W]，输出是类别
标签图而非特征图，K 个通道坍缩为 1 个整数标签维。这与本仓库其余形状保持
块 [B,C,H,W]->[B,C,H,W] 的约定不同，请勿插入特征主干中间使用。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"推理阶段采用 RankSEG（JMLR 2023）/ RankSEG-RMA（NeurIPS 2025）的排序重标注
后处理：对每个类的概率图降序排序后，用倒数矩近似求解与 Dice（或 IoU）一致的
最优截断体积，得到每类二值掩码，再以增量得分消解类间像素冲突，输出与目标
度量一致的标签图；相比逐像素 argmax，在 Dice/IoU 评测上更优。"

原论文引用格式：
Dai, B., & Li, C. (2023). RankSEG: a consistent ranking-based framework for
segmentation. Journal of Machine Learning Research, 24(224), 1-50.
Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm
via Reciprocal Moment Approximation. NeurIPS 2025.

四、适用任务
适用于语义分割（2D/3D）的推理期后处理，凡以 Dice/IoU 为评测指标的场景
（医学影像、遥感、细胞分割等）均可在 softmax/sigmoid 概率输出后套用。
也可用于实例分割的掩码打分重标注。注意：这是离散的评测期后处理，不可微、
无参数、不参与训练；输入应为 [0,1] 概率（softmax/sigmoid 输出），K≥2。
'''


class RKG(nn.Module):
    """RKG: RankSEG 一致性排序重标注 —— Dice/IoU 一致的评测期分割后处理

    ★ 例外：forward(probs: [B,K,H,W]) -> [B,H,W]，非形状保持；eval-only ★
    """

    # smooth 超过该阈值时改用仿射缩放得分写法（原 _SCALED_SCORE_SMOOTH_THRESHOLD）
    _SCALED_SCORE_SMOOTH_THRESHOLD = 1e6

    def __init__(self, metric: str = 'dice', smooth: float = 0.0,
                 pruning_prob: float = 0.5):
        super().__init__()
        if not isinstance(metric, str):
            raise TypeError('metric must be a string')
        metric = metric.strip().lower()
        if metric not in ('dice', 'iou'):
            raise ValueError("metric must be 'dice' or 'iou'")
        if not isinstance(smooth, (int, float)) or isinstance(smooth, bool):
            raise TypeError('smooth must be a real number')
        smooth = float(smooth)
        if not math.isfinite(smooth):
            raise ValueError('smooth must be finite')
        if smooth < 0:
            raise ValueError('smooth must be >= 0')
        if not isinstance(pruning_prob, (int, float)) or isinstance(pruning_prob, bool):
            raise TypeError('pruning_prob must be a real number')
        pruning_prob = float(pruning_prob)
        if not 0.0 <= pruning_prob <= 1.0:
            raise ValueError('pruning_prob must be in [0, 1]')

        self.metric = metric
        self.smooth = smooth
        self.pruning_prob = pruning_prob

    def _compute_opt_tau(self, pb_mean: torch.Tensor, cumsum_prob: torch.Tensor,
                         dim: int) -> torch.Tensor:
        """最优截断体积 τ* ∈ {0,...,dim}（RMA 闭式得分，含 τ=0 空掩码项）。"""
        smooth = self.smooth
        device = pb_mean.device
        taus = torch.arange(1, dim + 1, device=device).view(1, 1, -1)
        use_scaled_scores = smooth > self._SCALED_SCORE_SMOOTH_THRESHOLD

        if self.metric == 'dice':
            denom_offset = pb_mean.unsqueeze(-1) + taus
            if use_scaled_scores:
                metric_values = 2.0 * cumsum_prob / (1.0 + (denom_offset + 1) / smooth)
                metric_values -= denom_offset / (1.0 + denom_offset / smooth)
            else:
                discount = denom_offset + 1.0 + smooth
                metric_values = 2.0 * cumsum_prob / discount
                metric_values += smooth / (discount - 1)
        else:  # iou
            denom_offset = pb_mean.unsqueeze(-1) - cumsum_prob + taus
            if use_scaled_scores:
                metric_values = (2.0 * cumsum_prob - pb_mean.unsqueeze(-1) - taus) / \
                    (1.0 + denom_offset / smooth)
            else:
                metric_values = (cumsum_prob + smooth) / (denom_offset + smooth)

        # τ=0：空掩码得分
        if smooth > 0:
            if use_scaled_scores:
                empty_values = -pb_mean / (1.0 + pb_mean / smooth)
            else:
                empty_values = smooth / (pb_mean + smooth)
        else:
            empty_values = torch.zeros_like(pb_mean)

        metric_values = torch.cat([empty_values.unsqueeze(-1), metric_values], dim=-1)
        return torch.argmax(metric_values, dim=-1)

    def _convert_to_nonoverlap(self, overlap_preds: torch.Tensor,
                               probs: torch.Tensor,
                               active_mask: torch.Tensor,
                               pb_mean: torch.Tensor) -> torch.Tensor:
        """类间冲突消解：增量得分 argmax，输出 [B, D] 标签。"""
        smooth = self.smooth

        class_counts = overlap_preds.sum(dim=1)                       # (B, D)
        unassigned_mask = class_counts == 0
        single_pred_mask = class_counts == 1
        has_selected_mask = class_counts > 0
        # 已确定归属的像素（恰好被一个类选中）
        safe_to_predict = overlap_preds & single_pred_mask.unsqueeze(1)

        mu = (probs * safe_to_predict).sum(dim=2, keepdim=True)       # (B,K,1)
        n_assigned = safe_to_predict.sum(dim=2, keepdim=True).to(probs.dtype)

        use_scaled_scores = smooth > self._SCALED_SCORE_SMOOTH_THRESHOLD

        if self.metric == 'dice':
            denom_offset = n_assigned + pb_mean.unsqueeze(2) + 1
            if smooth == 0:
                # 加入该像素后 2*mu/(tau + E[Y] + 1) 的精确增量
                raw_increment_scores = 2 * (
                    (mu + probs) / (denom_offset + 1) - mu / denom_offset)
            elif use_scaled_scores:
                scaled_base_denom = 1.0 + (denom_offset - 1) / smooth
                scaled_denom = 1.0 + denom_offset / smooth
                scaled_next_denom = 1.0 + (denom_offset + 1) / smooth
                raw_increment_scores = 2 * (probs * scaled_denom - mu / smooth) / (
                    scaled_denom * scaled_next_denom
                ) - 1.0 / (scaled_base_denom * scaled_denom)
            else:
                denom = denom_offset + smooth
                base_denom = denom - 1
                safe_base_denom = torch.where(base_denom == 0,
                                              torch.ones_like(base_denom), base_denom)
                raw_increment_scores = 2 * (probs * denom - mu) / (denom * (denom + 1))
                raw_increment_scores -= smooth / (safe_base_denom * denom)
                # 低精度下 smooth 下溢为 0 时的极限增量 -1
                raw_increment_scores = torch.where(
                    base_denom == 0,
                    torch.full_like(raw_increment_scores, -1.0),
                    raw_increment_scores,
                )
        else:  # iou
            denom_offset = n_assigned + pb_mean.unsqueeze(2) - mu
            if use_scaled_scores:
                scaled_denom = 1.0 + denom_offset / smooth
                scaled_next_denom = 1.0 + (denom_offset - probs + 1) / smooth
                raw_increment_scores = (probs * scaled_denom
                                        - (1.0 + mu / smooth) * (1.0 - probs)) / (
                    scaled_denom * scaled_next_denom)
            else:
                denom = denom_offset + smooth
                # 空掩码 + smooth=0 + 全零类：0/0 约定为 0
                safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
                base_scores = (mu + smooth) / safe_denom
                raw_increment_scores = (mu + probs + smooth) / (denom - probs + 1) - base_scores

        # 资格：重叠像素仅"选中该像素的类"；无主像素由活跃类兜底（全剪枝则全类）
        all_classes_pruned = ~active_mask.any(dim=1, keepdim=True)   # (B,1)
        unassigned_eligible_classes = active_mask | all_classes_pruned

        eligible_mask = torch.where(
            has_selected_mask.unsqueeze(1),
            overlap_preds,
            unassigned_eligible_classes.unsqueeze(2),
        )
        eligible_increment_scores = torch.where(
            eligible_mask, raw_increment_scores, float('-inf'))
        return eligible_increment_scores.argmax(dim=1)               # (B, D)

    @torch.no_grad()
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        输入:
            probs: Tensor, shape = [B, K, H, W]，[0,1] 概率（K ≥ 2）
        输出:
            out: Tensor, shape = [B, H, W]，dtype int64 类别标签图
                 ★ 例外：非形状保持——K 通道坍缩为标签维，评测期后处理 ★
        """
        assert probs.dim() == 4, f'expected [B,K,H,W], got {tuple(probs.shape)}'
        B, K, H, W = probs.shape
        assert K >= 2, f'multiclass label map needs K >= 2, got {K}'
        assert probs.is_floating_point(), 'probs must be a floating-point tensor'

        # 半精度提升到 fp32（原实现策略）
        if probs.dtype in (torch.float16, torch.bfloat16):
            probs = probs.float()

        batch_size, num_class = B, K
        image_shape = (H, W)
        probs = torch.flatten(probs, start_dim=2, end_dim=-1)         # (B,K,D)
        dim = probs.shape[-1]

        # μ = E[|Y|]（各类概率总和）
        pb_mean = probs.sum(dim=-1)                                   # (B,K)

        # 排序 + 类剪枝 + cumsum
        sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)
        active_mask = sorted_prob[:, :, 0] > self.pruning_prob        # (B,K)
        cumsum_prob = torch.cumsum(sorted_prob, dim=-1)

        # 最优截断体积
        opt_tau = self._compute_opt_tau(pb_mean, cumsum_prob, dim)    # (B,K)

        # 掩码重建：rank 位置 < τ* 的像素标记为该类正类（还原到像素序）
        rank_positions = torch.arange(dim, device=probs.device).view(1, 1, -1)
        selected_by_rank = rank_positions < opt_tau.unsqueeze(-1)
        selected_by_rank &= active_mask.unsqueeze(-1)
        overlap_preds = torch.zeros_like(selected_by_rank).scatter_(
            2, top_index, selected_by_rank)                           # (B,K,D)

        # 非重叠化 → 标签图
        nonoverlap_preds = self._convert_to_nonoverlap(
            overlap_preds, probs, active_mask, pb_mean)               # (B,D)
        return nonoverlap_preds.reshape(batch_size, *image_shape).long()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # ★ 形状例外：输入为概率图 [B,K,H,W]，输出为标签图 [B,H,W]（评测期后处理）
    torch.manual_seed(0)
    input_tensor = torch.softmax(torch.randn(2, 5, 32, 32), dim=1)   # [B,K,H,W] 概率
    model = RKG(metric='dice')
    output = model(input_tensor)
    print('=== RKG: RankSEG ===')
    print('input_size:', input_tensor.size(), '(probs, K classes)')
    print('output_size:', output.size(), '(label map — shape NOT preserved)')
    print('params:', count_parameters(model))
    print('labels in range:', bool(output.min() >= 0 and output.max() < 5))
    model_iou = RKG(metric='iou')
    print("metric='iou' output_size:", model_iou(input_tensor).size())
