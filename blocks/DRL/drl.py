import typing as t
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：SAM+D: Parameter-Efficient Dimensional Lifting of SAM via Depth-Routed LoRA and Depth Shifting (ECCV 2026)
# 论文链接：https://arxiv.org/abs/2607.29033
# 代码来源：https://github.com/JerrySongCST/SAM-Plus-D
# 原始许可证：MIT (Copyright (c) 2026 Yu Song)
# 模块出处：modeling/lora.py 的 DRLoRA / LoRAExpert 类（L6-156）；调用方为
#          modeling/drlora_block.py 的 DRLoRABlock（L9-92，对 SAM block 的 Q/V 注入）
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：与 DRLoRABlock 解耦：不再包装冻结 SAM block / 窗口注意力 / DSM 深度移位
#          （DSM 已另提取为 DPS，此处不涉及），仅保留 DRLoRA 的"低秩专家 + 路由混合"
#          核心作为独立低秩旁路适配器。LoRAExpert 的 down/up 低秩更新与初始化
#          （down: kaiming_uniform a=sqrt(5)，up: 零初始化）原样保留；深度路由器
#          （Linear(1,E*16)-GELU-Linear(E*16,E) → softmax）与 content 路由、single
#          旁路、content_top1/top2（Switch/MoLE 基线）路由公式原样保留。
#          布局适配：原 x 为 SAM 的 (B,H,W,C) 通道末排布，统一改为 [B,C,H,W]，
#          线性层始终作用于通道维，数值等价。深度维适配与 DPS 约定一致：特征图的
#          H 视作"深度/切片"维——默认 slice_position = linspace(0,1,H)（与原
#          sam_d.py L257 的归一化切片位置一致），路由权重按行（切片）施加；
#          亦可显式传入 (B,1) 切片位置做整图路由（原 DRLoRA.forward 语义）。
#          原实现把 delta 加到注意力 Q/V 上；解耦后作为恒等映射的低秩旁路
#          out = x + delta（up 零初始化使初始输出=输入），delta 可经 lora_delta 单独取出。
#          统一 forward 只返回 [B,C,H,W]，aux 损失（top1/top2 模式）经 forward_with_aux
#          返回；forward 对 aux 丢弃不改输出数值。

'''
模块名称：DRL (Depth-Routed LoRA) —— 深度路由低秩专家适配

一、模块简介
把二维基础模型（SAM）提升到三维医学体数据时，简单共享一套 LoRA 无法刻画不同
深度切片（解剖位置）上的差异化适应需求。SAM+D 提出 DRLoRA：把 LoRA 扩展为
一组低秩专家（LoRA experts），用极轻量的"深度路由器"（仅以归一化切片深度
z∈R^1 为输入）产生 softmax 混合权重，按深度对专家输出做稠密加权混合。相比
内容路由（对特征做全局池化再路由），深度路由只依赖先验的切片位置，参数量极小
（E=4 时约 140 参数），且能稳定学到沿深度轴渐变的适应模式。

核心创新点：
1. 深度路由 MoLE：router(Linear(1,E*16)→GELU→Linear(E*16,E))，对切片深度
   z 做 softmax 得专家权重，按权重线性混合各低秩专家输出；
2. 低秩专家：down(C→r, kaiming) + up(r→C, 零初始化)，初始时适配增量为 0；
3. 稠密混合（dense softmax）无辅助损失（论文主模式）；另保留 content /
   content_top1（Switch 式）/ content_top2（MoLE 式 CV²）等基线路由便于消融；
4. 极低开销：可作为任意线性投影的旁路适配器（原注入 SAM 注意力 Q/V）。

二、结构设计
以输入 [B, C, H, W]、E = num_experts、r = rank 计（H 视作深度/切片维）：
1. LoRA 专家 ×E：each x → Linear(C→r, bias=False) → Linear(r→C, bias=False)，
   down 权重 kaiming_uniform_(a=sqrt(5))，up 权重全零；
2. 深度路由器（routing_mode='depth'）：slice_position ∈ [H, 1]（默认
   linspace(0,1,H)）→ Linear(1, 16E) → GELU → Linear(16E, E) → softmax(dim=-1)
   → w ∈ [H, E]；第 h 行特征的增量 = Σ_e w[h,e] * Expert_e(x)[:, :, h, :]；
3. content 路由：x.mean(dim=(2,3)) → [B, C] → Linear(C, E) → softmax，整图同权重；
   content_top1/top2：top-k 稀疏化权重并计算 Switch/CV² 辅助损失（forward_with_aux）；
   single（或 E=1）：旁路路由，直接取单专家输出；
4. 输出：out = x + Δ(x)，Δ 为上述路由混合低秩增量（初始 Δ=0）。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文采用 SAM+D（ECCV 2026）提出的深度路由低秩专家适配 DRL：以归一化深度坐标
为唯一路由输入，经两层 MLP 产生各低秩 LoRA 专家的 softmax 混合权重，在几乎不
增加参数的前提下实现沿深度轴变化的参数高效适应；低秩分支 up 层零初始化保证
训练起点等价恒等映射。"
（原论文 BibTeX 见 arXiv:2607.29033）

四、适用任务
适用于三维医学影像（CT/MRI 体数据）的分割/检测/分类、视频逐帧适应、以及任何
沿某一空间轴存在有序先验位置（深度、时间、层序）的 [B,C,H,W] 特征适配场景。
作为即插即用低秩旁路，可挂在任意骨干特征后或替换原注意力投影的 LoRA 注入。
'''

VALID_ROUTING_MODES = ('depth', 'content', 'content_top1', 'content_top2', 'single')


class LoRAExpert(nn.Module):
    """单低秩适配：x -> down(in_dim -> rank) -> up(rank -> in_dim)（原 LoRAExpert）。"""

    def __init__(self, in_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, in_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class DRL(nn.Module):
    """DRL: Depth-Routed LoRA —— 深度路由低秩专家适配"""

    def __init__(self, channels: int, rank: int = 16, num_experts: int = 4,
                 routing_mode: str = 'depth'):
        super().__init__()
        if routing_mode not in VALID_ROUTING_MODES:
            raise ValueError(f'routing_mode must be one of {VALID_ROUTING_MODES}, got {routing_mode!r}')
        self.channels = channels
        self.rank = rank
        self.num_experts = num_experts
        self.routing_mode = 'single' if num_experts == 1 else routing_mode

        self.experts = nn.ModuleList([LoRAExpert(channels, rank) for _ in range(num_experts)])

        if self.routing_mode == 'single':
            self.router = None
        elif self.routing_mode == 'depth':
            # ~140 params for E=4：仅以深度 z∈R^1 为输入的位置路由器（DRLoRA 主模式）
            self.router = nn.Sequential(
                nn.Linear(1, num_experts * 16),
                nn.GELU(),
                nn.Linear(num_experts * 16, num_experts),
            )
        else:
            # content 基线路由共享同一结构：Linear(C, E)
            self.router = nn.Linear(channels, num_experts)

    def lora_delta(self, x: torch.Tensor,
                   slice_position: t.Optional[torch.Tensor] = None) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """路由混合低秩增量 Δ(x)（原 DRLoRA.forward 的输出项）。

        x: [B, C, H, W]；返回 (delta, aux_loss)，delta 与 x 同形。
        slice_position: (B,1)/(B,) 为整图深度（原语义）；None 时取 H 行归一化深度
        linspace(0,1,H)（H 视作深度/切片维，与 DPS 约定一致），路由按行施加。
        """
        B, C, H, W = x.shape
        assert C == self.channels, f'expected C={self.channels}, got {C}'
        zero_aux = x.new_zeros(())
        # 专家线性层作用于通道维：转为原实现的 (B,H,W,C) 通道末布局
        x_cl = x.permute(0, 2, 3, 1)

        def _mix(weights_2d: torch.Tensor, w_layout: str) -> torch.Tensor:
            """按路由权重稠密混合专家输出（权重 → (B,H,W,C) 增量）。"""
            delta_cl = torch.zeros_like(x_cl)
            for i, expert in enumerate(self.experts):
                w = weights_2d[:, i]
                w = w.view(1, -1, 1, 1) if w_layout == 'row' else w.view(-1, 1, 1, 1)
                delta_cl = delta_cl + w * expert(x_cl)
            return delta_cl.permute(0, 3, 1, 2)

        # ---- 路由信号 ----
        if self.routing_mode == 'depth':
            if slice_position is None:
                # 默认：H 行 = H 个切片，深度坐标归一化到 [0,1]（原 sam_d 的 slice_pos）
                sp = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).unsqueeze(-1)  # [H, 1]
                logits = self.router(sp)                       # [H, E]
                w_layout = 'row'
            else:
                sp = slice_position.to(dtype=x.dtype, device=x.device)
                if sp.dim() == 1:
                    sp = sp.unsqueeze(-1)
                logits = self.router(sp)                       # [B, E]
                w_layout = 'batch'
            weights = logits.softmax(dim=-1)
            return _mix(weights, w_layout), zero_aux

        if self.routing_mode == 'single':
            return self.experts[0](x_cl).permute(0, 3, 1, 2), zero_aux

        # Content-based routing: pool 空间维到 (B, C)
        content_feat = x.mean(dim=(2, 3))                      # [B, C]
        logits = self.router(content_feat)                     # [B, E]
        weights = logits.softmax(dim=-1)                       # [B, E]

        if self.routing_mode == 'content':
            return _mix(weights, 'batch'), zero_aux

        if self.routing_mode == 'content_top1':
            # Switch 式 top-1 路由 + 负载均衡辅助损失（Switch Transformer 式 f·P）。
            # 计算仍保持稠密（E 个专家全算）以便逐步公平对比，仅路由结构稀疏。
            top1_idx = logits.argmax(dim=-1)                                   # [B]
            sparse_w = torch.zeros_like(weights)
            sparse_w.scatter_(1, top1_idx.unsqueeze(-1),
                              weights.gather(1, top1_idx.unsqueeze(-1)))
            delta = _mix(sparse_w, 'batch')
            f = torch.zeros(self.num_experts, device=x.device, dtype=weights.dtype)
            f = f.scatter_add(0, top1_idx, torch.ones_like(top1_idx, dtype=weights.dtype)) / max(1, top1_idx.numel())
            P = weights.mean(dim=0)
            aux_loss = self.num_experts * (f * P).sum()
            return delta, aux_loss

        if self.routing_mode == 'content_top2':
            # MoLE 式 top-2 稠密混合 + 重要性 CV² 辅助损失
            k = min(2, self.num_experts)
            topk_vals, topk_idx = weights.topk(k, dim=-1)                      # [B, k]
            topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
            sparse_w = torch.zeros_like(weights).scatter_(1, topk_idx, topk_vals)
            delta = _mix(sparse_w, 'batch')
            importance = weights.sum(dim=0)                                    # [E]
            cv2 = (importance.std() / (importance.mean() + 1e-9)) ** 2
            return delta, cv2

        raise RuntimeError(f'unreachable routing_mode: {self.routing_mode}')

    def forward_with_aux(self, x: torch.Tensor,
                         slice_position: t.Optional[torch.Tensor] = None) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """返回 (out, aux_loss)：out = x + Δ(x)；aux_loss 对 depth/content/single 为 0。"""
        delta, aux_loss = self.lora_delta(x, slice_position)
        return x + delta, aux_loss

    def forward(self, x: torch.Tensor,
                slice_position: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]（恒等 + 路由低秩增量）
        """
        out, _ = self.forward_with_aux(x, slice_position)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(2, 128, 16, 16)
    model = DRL(channels=128, rank=16, num_experts=4)
    output = model(input_tensor)
    print('=== DRL: Depth-Routed LoRA ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
