# SOTA - State-of-the-Art Neural Network Modules

[![中文](https://img.shields.io/badge/README-中文-red)](README.md)
[![English](https://img.shields.io/badge/README-English-blue)](README_EN.md)

一个系统整理、复现和开发常用神经网络模块的开源仓库，重点面向计算机视觉任务。

> **提示**：本仓库还有 `bcl` 分支，提供针对时序 BCL 格式数据的适配模块。切换分支即可获取不同输入格式的处理代码：
> ```bash
> git checkout bcl  # 切换到 BCL 时序格式分支
> ```

## 项目简介

本项目收集和实现了计算机视觉领域常见的即插即用神经网络模块，涵盖图像分类、目标检测、语义分割等方向。每个模块都包含清晰的论文背景说明、结构设计说明、PyTorch 代码实现和测试代码。**所有模块均为原创设计。**

## 目录结构

```
SOTA/
├── README.md
├── README_EN.md
├── resnet_insert_example.py
└── blocks/
    ├── SRM/  选择性响应模块
    ├── DFA/  差异性特征放大器
    ├── CIM/  上下文信息调制器
    ├── GFF/  门控特征融合模块
    ├── DRS/  动态感受野选择器
    ├── AFM/  自适应频率调制模块
    ├── PFA/  渐进式特征聚合器
    ├── SAM/  空间亲和力模块
    ├── CRM/  通道重校准模块
    ├── LCR/  局部上下文重构模块
    ├── RIM/  递归推理模块
    ├── PDR/  极化双表示模块
    ├── SSM/  显著性引导抑制模块
    ├── PGM/  渐进式门控模块
    ├── FEM/  特征均衡模块
    ├── IGM/  信息汇聚模块
    ├── RGM/  互惠引导模块
    ├── DSM/  双尺度调制器
    ├── OEM/  序统计增强模块
    ├── MPM/  动量传播模块
    ├── PCM/  相位一致性模块
    ├── CGM/  条件门控模块
    ├── IRM/  信息路由模块
    ├── BSM/  双边相似度模块
    ├── DGM/  多样性引导模块
    ├── SUM/  空间不确定性模块
    ├── AGM/  自适应粒度模块
    ├── RAM/  残差放大模块
    ├── TCM/  张量补全模块
    ├── NLM/  非局部调制模块
    ├── EDM/  熵驱动模块
    ├── FIM/  频率重要性模块
    ├── HTM/  层次变换模块
    ├── WAM/  加权注意力模块
    ├── RCM/  递归卷积模块
    ├── CAM/  对比度感知模块
    ├── SDM/  谱分解模块
    ├── QEM/  分位数增强模块
    ├── EEM/  能量均衡模块
    ├── TSFM/ 时序-空间融合模块
    ├── LVM/  局部方差调制模块
    └── ...
```

## 已实现模块

| 日期 | 模块 | 全称 | 核心思想 | 适用任务 |
|------|------|------|----------|----------|
| 05-27 | SRM | Selective Response Module | 分组统计→位置敏感通道调制→软阈值稀疏化 | 分类/检测/分割 |
| 05-27 | DFA | Differential Feature Amplifier | 局部邻域差异→差异驱动放大→对比度敏感 | 分类/检测/边缘检测 |
| 05-27 | CIM | Contextual Information Modulator | 双路径(局部+上下文)→空间自适应混合比例 | 分类/检测/分割 |
| 05-27 | GFF | Gated Feature Fusion | 三路并行变换→空间-通道联合门控→竞争性融合 | 分类/检测/分割 |
| 05-27 | DRS | Dynamic Receptive Field Selector | 多膨胀率并行分支→空间自适应感受野选择 | 检测/分割(多尺度) |
| 05-27 | AFM | Adaptive Frequency Modulation | 多核并行→频率带分解→空间自适应频率调制 | 分类/检测/图像恢复 |
| 05-27 | PFA | Progressive Feature Aggregator | 两阶段粗调-精调→阶段间信息桥接→残差累积 | 分类/检测/分割 |
| 05-27 | SAM | Spatial Affinity Module | 低秩投影→亲和力矩阵→信息传播→全局上下文 | 分割/检测/生成 |
| 05-27 | CRM | Channel Recalibration Module | 激活熵估计→熵引导通道评估→冗余抑制 | 分类/检测/分割 |
| 05-27 | LCR | Local Context Reconstructor | 逐位置动态邻域权重→专属局部卷积核→自适应聚合 | 分类/检测/分割 |
| 05-28 | RIM | Recursive Inference Module | 权重共享递归变换→迭代嵌入→残差累积精炼 | 分类/检测/分割 |
| 05-28 | PDR | Polarized Dual Representation | 空间/语义双通路→交叉门控→极化特征融合 | 分类/检测/分割 |
| 05-28 | SSM | Saliency-Guided Suppression Module | 显著性检测→自适应阈值→软抑制→信息预算重分配 | 分类/检测/分割 |
| 05-29 | PGM | Progressive Gating Module | 三阶段级联门控(粗→中→细)→通道级自适应融合 | 分类/检测/分割 |
| 05-29 | FEM | Feature Equilibrium Module | 通道统计编码→均衡能量学习→指数平滑调节 | 分类/检测/分割 |
| 05-29 | IGM | Information Gathering Module | 多尺度深度可分离汇聚→空间自适应尺度权重 | 分类/检测/分割 |
| 05-30 | RGM | Reciprocal Guidance Module | 通道-空间双分支互惠引导→双向信息调制 | 分类/检测/分割 |
| 05-30 | DSM | Dual-Scale Modulator | 粗-细双尺度互调→上下文引导+细节回注 | 分类/检测/分割 |
| 05-30 | OEM | Order-Statistic Enhancement Module | 多序统计量提取→空间自适应统计选择→鲁棒增强 | 分类/检测/分割 |
| 05-31 | MPM | Momentum Propagation Module | 大核平滑动量参考→瞬态偏差感知→自适应调制 | 分类/检测/分割 |
| 05-31 | PCM | Phase-Coherence Module | FFT频域解耦→幅度重标定+相位一致性增强 | 分类/检测/分割 |
| 05-31 | CGM | Conditional Gating Module | 可学习条件原型→相似度驱动门控→语义参照增强 | 分类/检测/分割 |
| 06-01 | IRM | Information Routing Module | 多专家内容感知路由→空间自适应专家混合 | 分类/检测/分割 |
| 06-01 | BSM | Bilateral Similarity Module | K×K邻域内容相似度→双边自适应加权聚合 | 分类/检测/分割 |
| 06-01 | DGM | Diversity-Guided Module | 通道Gram矩阵→冗余分数驱动调制→多样性增强 | 分类/检测/分割 |
| 06-03 | SUM | Spatial Uncertainty Module | 局部不确定性估计→不确定引导平滑/保持双路径融合 | 分类/检测/分割 |
| 06-03 | AGM | Adaptive Granularity Module | 粒度偏好图→粗细双分支→空间自适应粒度插值 | 分类/检测/分割 |
| 06-03 | RAM | Residual Amplification Module | 基座-残差分解→残差信息分析→内容感知放大/抑制 | 分类/检测/分割 |
| 06-06 | TCM | Tensor Completion Module | 张量低秩分解→信号子空间补全→结构保持融合 | 分类/检测/分割 |
| 06-06 | NLM | Non-local Modulation Module | QKV投影→非局部亲和力→调制信号生成→残差调制 | 分类/检测/分割 |
| 06-06 | EDM | Entropy-Driven Module | 局部信息熵估计→熵引导增强/压缩→自适应资源分配 | 分类/检测/分割 |
| 06-06 | FIM | Frequency Importance Module | DCT频域分解→频率重要性学习→频谱重标定→IDCT | 分类/检测/分割 |
| 06-06 | HTM | Hierarchical Transformation Module | 三阶段递进变换→信息桥接→自适应阶段融合 | 分类/检测/分割 |
| 06-06 | WAM | Weighted Attention Module | 四模式注意力并行→模式融合权重学习→自适应组合 | 分类/检测/分割 |
| 06-06 | RCM | Recursive Convolution Module | 权重共享递归卷积→终止门→空间自适应处理深度 | 分类/检测/分割 |
| 06-06 | CAM | Contrast-Aware Module | 局部对比度估计→锐化/平滑双路径→对比度引导融合 | 分类/检测/分割 |
| 06-06 | SDM | Spectral Decomposition Module | 通道协方差谱分解→子空间分离→谱域自适应滤波 | 分类/检测/分割 |
| 06-06 | QEM | Quantile Enhancement Module | 分位数估计→鲁棒分位数归一化→分布感知增强 | 分类/检测/分割 |
| 06-18 | EEM | Energy Equalization Module | 通道能量统计→自适应缩放→能量感知双路径均衡 | 分类/检测/分割 |
| 06-18 | TSFM | Temporal-Spatial Fusion Module | 空间/语义双路径编码→交叉注意力融合→双向空间-通道调制 | 分类/检测/分割 |
| 06-18 | LVM | Local Variance Modulator | 多尺度方差估计→方差感知双路径调制→自适应细节/抑制融合 | 分类/检测/分割 |
| 06-19 | CFM | Channel Frequency Mixer | DCT频域变换→可学习频率混合矩阵→跨通道频率信息交换 | 分类/检测/分割 |
| 06-19 | SGM | Spatial Gradient Modulator | Sobel梯度提取→梯度幅值-方向联合编码→方向感知特征调制 | 分类/检测/分割/边缘检测 |
| 06-19 | DEM | Dense Evolution Module | 多尺度变异生成→信息熵适应度评估→Top-k自然选择→进化融合 | 分类/检测/分割 |
| 06-19 | PAM | Phase Alignment Module | Gabor相位估计→自适应相位偏移→相位一致性约束→残差融合 | 图像融合/恢复 |
| 06-19 | FGM | Feature Gating Module | 协作门控→局部上下文感知→双向通道交互→稀疏激活 | 分类/检测/分割 |
| 06-19 | MCM | Multi-Scale Context Module | 多尺度并行提取→自适应尺度权重→尺度间交互→全局-局部融合 | 分割/检测/分类 |
| 06-19 | DWM | Dynamic Weight Module | 内容编码→调制因子预测→渐进式调制→平滑性约束 | 分类/检测/风格迁移 |
| 06-19 | ERM | Edge Response Module | 可学习边缘检测→边缘增强→边缘-非边缘分离→自适应融合 | 边缘检测/分割/检测 |
| 06-19 | STM | Spatial-Channel Transformer Module | 空间-通道联合编码→双向交叉注意力→位置编码增强→多头变换 | 分类/检测/分割 |
| 06-19 | LHM | Local Histogram Module | 可微分soft binning→局部分布统计→分布感知增强→残差融合 | 分类/异常检测/纹理分析 |
| 06-19 | CCM | Channel Correlation Module | 通道相关性矩阵→低秩近似→相关性引导增强→正则化 | 分类/检测/分割 |
| 06-19 | RVM | Random Variation Module | 可控随机注入→温度控制→特征级增强→方差保持归一化 | 分类/检测/鲁棒性 |
| 06-19 | KSM | Kernel Selection Module | 位置感知核选择→可微分软选择→多尺度并行→稀疏正则化 | 分类/检测/分割 |
| 06-20 | OSM | Offset Spatial Mixing | 可变形偏移学习→偏移连续性约束→多尺度偏移融合→偏移感知聚合 | 分类/检测/分割 |
| 06-20 | VGM | Variational Gaussian Mixing | 高斯分布建模→变分推断→不确定性感知混合→采样增强 | 分类/检测/分割 |
| 06-20 | ABM | Adaptive Batch Module | 内容感知统计量→双重调制→轻量级预测→渐进式归一化 | 分类/风格迁移/域自适应 |
| 06-20 | BFM | Batch Fusion Module | 批内统计→样本间注意力→自适应融合→计算高效 | 分类/度量学习/少样本学习 |
| 06-20 | CVM | Channel Variance Module | 通道方差估计→方差引导增强→方差归一化→残差融合 | 分类/特征选择/通道剪枝 |
| 06-20 | DPM | Dense Prediction Module | 全局-局部融合→自适应权重→密集预测头→多任务支持 | 分割/深度估计/检测 |
| 06-20 | FTM | Frequency Transform Module | 轻量级Haar变换→频带选择→频域增强→逆变换融合 | 分类/恢复/去噪 |
| 06-20 | GCM | Gradient Correlation Module | 多方向梯度→梯度相关性计算→相关性引导增强→结构保持 | 边缘检测/分割/纹理分析 |
| 06-20 | JRM | Joint Reasoning Module | 多关系建模→关系交互→联合推理→推理门控 | 场景理解/视觉问答/关系检测 |
| 06-20 | NAM | Neural Attention Module | 多尺度注意力→尺度间交互→注意力融合→神经调制 | 分类/检测/分割 |
| 06-20 | ARM | Attention Refinement Module | 迭代精炼→残差精炼→精炼门控→收敛检测 | 分类/检测/分割 |
| 06-20 | CLM | Context Learning Module | 多类型上下文→上下文选择→上下文交互→学习型融合 | 分割/场景理解/分类 |
| 06-20 | DFM | Dynamic Feature Module | 内容编码→参数预测→动态处理→输出精炼 | 分类/风格迁移/域自适应 |
| 06-20 | ESM | Enhanced Spatial Module | 空间位置编码→内容自适应采样→多尺度空间建模→空间一致性 | 检测/分割/姿态估计 |
| 06-20 | FCM | Feature Correlation Module | 低秩相关性→相关性模式学习→相关性引导增强→输出精炼 | 分类/检测/分割 |
| 06-20 | GFM | Global Fusion Module | 全局语义提取→局部细节增强→自适应融合→语义引导增强 | 分类/分割/场景理解 |
| 06-20 | HPM | Hierarchical Prediction Module | 层次化预测→层次间传递→渐进式精炼→多层次融合 | 分割/检测/深度估计 |
| 06-20 | IPM | Iterative Processing Module | 迭代处理→残差累积→迭代门控→自适应迭代次数 | 恢复/去噪/超分辨率 |
| 06-20 | JSM | Joint Selection Module | 空间选择→通道选择→选择一致性→联合稀疏激活 | 分类/检测/分割 |
| 06-20 | KBM | Knowledge Bridge Module | 知识编码→语义对齐→桥接传递→自适应桥接强度 | 多尺度融合/跨层传递 |

## 使用方法

每个模块均可作为即插即用组件嵌入现有网络：

```python
from blocks.RIM.rim import RIM
import torch

rim = RIM(channels=64, num_iterations=3)
x = torch.randn(1, 64, 32, 32)
out = rim(x)
print(out.shape)  # [1, 64, 32, 32]
```

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.10
- thop（可选，用于 FLOPs 统计）

## 贡献指南

欢迎提交 PR 添加新的神经网络模块。请参考 `blocks/` 下已有模块的格式，确保包含完整的文档说明和测试代码。

## 许可证

MIT License
