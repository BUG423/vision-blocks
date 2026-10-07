# 模块提取与接口统一规范（内部执行规范）

本文件是并行提取子 Agent 的唯一接口契约。提取第三方论文代码时必须遵守。

## 目录与命名

- 路径：`blocks/<ABBREV>/<abbrev>.py`
- `<ABBREV>`：模块简称，2–5 个大写字母（如 `PSA`, `MCF`, `CGL`），不得与现有 `blocks/` 目录冲突。
- 类名：与简称完全一致，如 `class PSA(nn.Module)`。
- 一个目录一个主类；若论文有多个强相关子模块且不可分，可放在同一文件中，但对外只导出主类。

## 统一接口

```python
class ABBREV(nn.Module):
    def __init__(self, channels: int, **task_specific_kwargs):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, C, H, W]
        out: [B, C, H, W]   # 形状保持；若不能保持必须在 docstring 与头注释中说明
        """
```

约定：
1. **第一个参数必须是 `channels: int`**（输入/输出通道数；若 in/out 不同则用 `in_channels, out_channels`，并写明）。
2. 默认超参与论文一致；额外超参只加必要的，并给论文默认值。
3. `forward` 默认吃 4D `[B,C,H,W]`。若模块天然 1D/3D，提供同一类内的自动适配（内部 reshape），保持对外仍是 4D。
4. 禁止改变数值逻辑：重构只允许改命名、删死代码、统一 `nn` 用法、把硬编码改成参数、补 shape assert。
5. 不依赖第三方包（除 `torch`/`torch.nn`/`torch.nn.functional`/`typing`/`math`）。CUDA 扩展一律重写为纯 PyTorch。

## 文件头注释（必须）

```python
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：<论文标题> (<Venue> <Year>)
# 论文链接：<arxiv 或 openaccess 链接>
# 代码来源：<GitHub 链接>
# 原始许可证：<MIT / Apache-2.0 / BSD-3-Clause 等>
# 模块出处：<repo 内相对路径，如 models/ops.py 的 XXX 类>
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：<改了什么、为何逻辑不变>
```

## 文档字符串（中文，沿用本仓库既有四段式）

```
'''
模块名称：<ABBREV> (<Full English Name>) —— <中文名>

一、模块简介
<3-8 句：问题、核心思想、创新点>

二、结构设计
<子结构列表，含张量形状流转>

三、论文写法参考
<若在论文中引用，可如何描述；并注明原论文引用格式>

四、适用任务
<分类/检测/分割/...>
'''
```

类的 docstring 一行摘要即可。

## 尾部自检

文件末尾必须包含：

```python
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 128, 64, 64)
    model = ABBREV(channels=128)
    output = model(input_tensor)
    print('=== ABBREV: Full Name ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
```

要求：`python blocks/ABBREV/abbrev.py` 能直接跑通（CPU 即可）。

## 禁止事项

- 禁止保留原仓库的 import 体系 / 配置系统 / registry
- 禁止引入 `timm`、`mmcv`、`detectron2` 等依赖
- 禁止悄悄改公式（如 softmax 维度、归一化方式）
- 禁止把整篇论文的训练管线搬进来
- 禁止在无 license 的代码上进行提取

## 提取后交付清单（每个模块）

1. `blocks/<ABBREV>/<abbrev>.py`（含头注释 + 四段文档 + 自检）
2. 在报告中列出：原始类名、文件路径、重构点、shape 测试结果
