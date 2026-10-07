import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 论文：AuGhostmentation: The Eyes Never Stand Still—Why Should CNNs? (NeurIPS 2026)
# 论文链接：https://openreview.net/forum?id=UrYjjK6We7
# 代码来源：https://github.com/emirhaninan/AuGhostmentation
# 原始许可证：MIT
# 模块出处：auGhostmentation.py 的 AuGhostmentation 类
# 提取者：MiMoCode ExtractAgent
# 日期：2026-10-07
# 重构说明：删去 numpy/scipy 依赖（np.random.poisson/uniform/vonmises、scipy.stats.lognorm
#          的 cdf/ppf）改为 math.erf + Acklam-Halley 逆正态 + torch 随机数的等价实现
#          （lognormal CDF/PPF 与 scipy 数值误差 <1e-12，von Mises 用 Best & Fisher 1979
#          拒绝采样，分布与 np.random.vonmises 一致；角度经 cos/sin 使用，2π 周期使其
#          与 numpy 的 [-π,π] 取模行为等价）；微扫视随机位移的数学（arcmin↔像素换算、
#          截断对数正态振幅、von Mises 基向角、运动模糊核光栅化、w 混合）逐行保留；
#          原单图 [C,H,W] 接口适配为统一 [B,C,H,W]（batch 内逐样本独立采样轨迹，
#          卷积混合逻辑与原逐图路径一致）；新增 self.training 门控——eval() 恒等映射，
#          train() 才施加增强；删除模块级预设 tiny_imagenet_aughost / imagenet_aughost
#          （实例化样例，属死代码）。

'''
模块名称：AUG (AuGhostmentation) —— 微扫视幽灵增强

一、模块简介
人类眼球在注视时并非静止，而是持续发生微扫视（micro-saccade）：一种小幅、
快速、共轭的眼动。视网膜上的图像因此在每次微扫视期间产生短暂的运动模糊，
神经系统随后将其重建为稳定感知。CNN 训练时看到的却是完美静止的图像切片，
这构成了生物视觉与机器视觉输入分布之间的一道鸿沟。

AuGhostmentation 的核心思想是：按人眼微扫视的统计规律（方向偏好、振幅分布、
发生率）在训练图像上合成随机微扫视轨迹引起的运动模糊"幽灵"，使 CNN 在训练
阶段就适应这种生物上真实的输入扰动。与普通运动模糊增强不同，其随机轨迹
由生理测量驱动：方向服从 von Mises 分布（偏向水平/垂直四个基向）、振幅服从
截断对数正态分布（中位数 19.4 arcmin）、每张图的微扫视次数服从 Poisson 分布。

核心创新点：
1. 生理先验驱动的随机轨迹：von Mises 基向角 + 截断对数正态振幅，参数来自
   微扫视测量文献；
2. 轨迹光栅化运动模糊核：把每次微扫视的位移轨迹离散采样为归一化卷积核，
   模拟视网膜曝光积分；
3. 幽灵混合：模糊结果以权重 w 与原图凸组合，保留可辨识的"幽灵"重影；
4. 多次微扫视链式叠加：每张图采样 Poisson(expected_ms) 次微扫视并顺序混合。

二、结构设计
AUG 由以下子结构组成（无学习参数）：
1. 微扫视计数：n_ms ~ Poisson(expected_ms)，为 0 时恒等输出；
2. 方向采样：以概率 ms_horizontal_weight 选水平基向角 {0, π}，否则选垂直
   {π/2, 3π/2}，再加 von Mises(κ=ms_cardinal_kappa) 偏差，归一化为单位向量
   (dir_x, dir_y)；
3. 振幅采样：像素域截断对数正态 LogNormal(σ=0.53, median=MEDIAN_ARCMIN)，
   截断区间 [min_arcmin, max_arcmin]（按 fov_deg/image_size 的 arcmin/px 换算
   为像素位移 [min_shift, max_shift]），用逆 CDF 采样；
4. 运动模糊核光栅化：dx=amp*dir_x, dy=amp*dir_y；K=max(2, round(amp))，
   核尺寸 ks=2K+1，沿 (cx,cy)=(K,K) 到 (cx+dx, cy+dy) 均匀取 2K 个采样点并
   累加到最近像素，归一化为核 [ks, ks]；
5. 逐通道分组卷积：kernel.view(1,1,ks,ks).expand(C,1,ks,ks)，padding=ks//2，
   groups=C；
6. 混合：result = (1-w) * result + w * blurred，对 n_ms 次微扫视顺序执行。

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"训练阶段采用 AuGhostmentation（NeurIPS 2026）的微扫视数据增强：按人眼
微扫视统计（von Mises 基向角偏好、截断对数正态振幅、Poisson 发生率）合成
随机微扫视轨迹的运动模糊幽灵，并以权重 w 与原图凸组合，使模型适应生物
视觉中真实的注视期眼动扰动；推理阶段该模块退化为恒等映射。"

原论文引用格式：
AuGhostmentation: The Eyes Never Stand Still—Why Should CNNs? NeurIPS 2026.

四、适用任务
适用于图像分类、目标检测、语义分割等以自然图像为输入的视觉任务，作为训练期
数据增强插件置于网络输入端。特别适合对输入分布偏移敏感、需要鲁棒性的小数据
/细粒度识别场景。注意：本模块仅在 self.training=True 时生效，eval() 时为恒等
映射；对特征图（非自然图像）使用意义有限。
'''


def _norm_cdf(z: float) -> float:
    """标准正态累积分布函数 Φ(z)（math.erf，机器精度）。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# Acklam (2003) 有理逼近系数：标准正态分位数函数初值（相对误差 < 1.15e-9）
_PP_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_PP_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
_PP_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_PP_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]


def _norm_ppf(p: float) -> float:
    """标准正态分位数函数 Φ⁻¹(p)（Acklam 初值 + Halley 迭代，误差 <1e-11）。

    等价于 scipy.stats.norm.ppf，用于替代 scipy.stats.lognorm.ppf 的内层求逆。
    """
    if not 0.0 < p < 1.0:
        raise ValueError('_norm_ppf: p must be in (0, 1)')
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_PP_C[0] * q + _PP_C[1]) * q + _PP_C[2]) * q + _PP_C[3]) * q + _PP_C[4]) * q + _PP_C[5]) / \
            ((((_PP_D[0] * q + _PP_D[1]) * q + _PP_D[2]) * q + _PP_D[3]) * q + 1.0)
    elif p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_PP_C[0] * q + _PP_C[1]) * q + _PP_C[2]) * q + _PP_C[3]) * q + _PP_C[4]) * q + _PP_C[5]) / \
             ((((_PP_D[0] * q + _PP_D[1]) * q + _PP_D[2]) * q + _PP_D[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((_PP_A[0] * r + _PP_A[1]) * r + _PP_A[2]) * r + _PP_A[3]) * r + _PP_A[4]) * r + _PP_A[5]) * q / \
            (((((_PP_B[0] * r + _PP_B[1]) * r + _PP_B[2]) * r + _PP_B[3]) * r + _PP_B[4]) * r + 1.0)
    # Halley 修正：解 Φ(x) = p，把 Acklam 的 ~1e-9 相对误差压到机器精度
    e = _norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _lognorm_cdf(x: float, sigma: float, scale: float) -> float:
    """LogNormal(sigma, scale) 的 CDF，等价于 scipy.stats.lognorm(s=sigma, scale=scale).cdf。"""
    return _norm_cdf((math.log(x) - math.log(scale)) / sigma)


def _lognorm_ppf(u: float, sigma: float, scale: float) -> float:
    """LogNormal(sigma, scale) 的 PPF，等价于 scipy.stats.lognorm(s=sigma, scale=scale).ppf。"""
    return scale * math.exp(sigma * _norm_ppf(u))


class AUG(nn.Module):
    """AUG: AuGhostmentation —— 生理先验驱动的微扫视运动模糊增强（train-only，eval 恒等）"""

    _LN_SIGMA = 0.53
    _MEDIAN_ARCMIN = 19.4

    def __init__(self,
                 channels: int = 3,
                 fov_deg: float = 8.0,
                 image_size: int = 224,
                 expected_ms: float = 1.33,
                 ms_horizontal_weight: float = 0.7193,
                 ms_cardinal_kappa: float = 7.5347,
                 w: float = 0.50,
                 min_arcmin: float = 5.0,
                 max_arcmin: float = 60.0):
        super().__init__()
        self.channels = channels

        # 原实现逐行保留：视野角与图像尺寸决定 arcmin/px 换算，
        # 把像素位移与对数正态尺度统一到像素单位
        ppd = image_size / fov_deg
        arcmin_per_px = 60.0 / ppd

        self.min_shift = min_arcmin / arcmin_per_px
        self.max_shift = max_arcmin / arcmin_per_px

        self._ln_scale = self._MEDIAN_ARCMIN / arcmin_per_px
        self._cdf_lo = _lognorm_cdf(self.min_shift, self._LN_SIGMA, self._ln_scale)
        self._cdf_hi = _lognorm_cdf(self.max_shift, self._LN_SIGMA, self._ln_scale)

        self.expected_ms = expected_ms
        self.ms_horizontal_weight = ms_horizontal_weight
        self.ms_cardinal_kappa = ms_cardinal_kappa
        self.w = w

    def _sample_direction(self) -> t.Tuple[float, float]:
        # 原：random.random() < ms_horizontal_weight 选水平/垂直基向角，
        #     np.random.vonmises(center, kappa) 绕基向角散布
        if torch.rand(1).item() < self.ms_horizontal_weight:
            center = [0.0, math.pi][torch.randint(0, 2, (1,)).item()]
        else:
            center = [math.pi / 2, 3 * math.pi / 2][torch.randint(0, 2, (1,)).item()]

        # Best & Fisher (1979) von Mises 采样（均值 0），加 center 得到目标角。
        # 原 np.random.vonmises 返回 [-π, π] 取模值；此处不取模，因下游仅用
        # cos/sin，2π 周期下数值等价。
        theta = center + self._sample_vonmises_std(self.ms_cardinal_kappa)
        dx = math.cos(theta)
        dy = math.sin(theta)
        mag = math.sqrt(dx ** 2 + dy ** 2) + 1e-8
        return dx / mag, dy / mag

    @staticmethod
    def _sample_vonmises_std(kappa: float) -> float:
        """均值 0 的 von Mises 采样，Best & Fisher (1979) 拒绝采样法。"""
        if kappa < 1e-10:
            return math.pi * (2.0 * torch.rand(1).item() - 1.0)
        a = 1.0 + math.sqrt(1.0 + 4.0 * kappa * kappa)
        b = (a - math.sqrt(2.0 * a)) / (2.0 * kappa)
        r = (1.0 + b * b) / (2.0 * b)
        while True:
            u1 = torch.rand(1).item()
            z = math.cos(math.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)
            # u2 钳到 (0,1]：算法假设均匀连续变量，避免 log(c/0) 数域错误
            u2 = max(torch.rand(1).item(), 1e-12)
            if c * (2.0 - c) - u2 > 0.0:
                break
            # c <= 0 时 log 非正：数学上为 -inf，判定拒绝并重采样
            if c > 0.0 and math.log(c / u2) + 1.0 - c >= 0.0:
                break
        u3 = torch.rand(1).item()
        theta = math.acos(f)
        if u3 < 0.5:
            theta = -theta
        return theta

    def _sample_amplitude(self) -> float:
        # 原：u ~ Uniform(cdf_lo, cdf_hi)；amp = lognorm.ppf(u)（截断对数正态）
        u = self._cdf_lo + (self._cdf_hi - self._cdf_lo) * torch.rand(1).item()
        return _lognorm_ppf(u, self._LN_SIGMA, self._ln_scale)

    def _sample_num_saccades(self) -> int:
        # 原：np.random.poisson(expected_ms)；Knuth 算法（λ 小，精确）
        lam = self.expected_ms
        if lam <= 0.0:
            return 0
        limit = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= torch.rand(1).item()
            if p <= limit:
                return k
            k += 1

    def _apply_one(self, img: torch.Tensor) -> torch.Tensor:
        """对单张 [C,H,W] 图像施加微扫视增强（与原 forward 逐行一致）。"""
        n_ms = self._sample_num_saccades()
        if n_ms == 0:
            return img

        result = img
        for _ in range(n_ms):
            dir_x, dir_y = self._sample_direction()
            amplitude = self._sample_amplitude()

            dx = amplitude * dir_x
            dy = amplitude * dir_y

            # 轨迹光栅化为运动模糊核
            K = max(2, int(round(amplitude)))
            ks = (K * 2) + 1
            kernel = torch.zeros(ks, ks, device=result.device,
                                 dtype=result.dtype)
            cx = K
            cy = K
            num_samples = K * 2
            for k in range(num_samples):
                tt = k / (num_samples - 1)
                px = int(round(cx + tt * dx))
                py = int(round(cy + tt * dy))
                px = max(0, min(ks - 1, px))
                py = max(0, min(ks - 1, py))
                kernel[py, px] += 1.0
            kernel = kernel / (kernel.sum() + 1e-8)

            C = result.shape[0]
            k2d = kernel.view(1, 1, ks, ks).expand(C, 1, ks, ks)
            pad = ks // 2
            blurred = F.conv2d(result.unsqueeze(0), k2d,
                               padding=pad, groups=C).squeeze(0)

            result = result * (1.0 - self.w) + blurred * self.w

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]（eval() 时恒等于 x）
        """
        assert x.dim() == 4, f'expected [B,C,H,W], got {tuple(x.shape)}'
        assert x.shape[1] == self.channels, \
            f'channel mismatch: x has {x.shape[1]}, module configured with {self.channels}'

        # self.training 门控：eval() 恒等映射
        if not self.training:
            return x

        # 原实现作用于单张 [C,H,W]；批处理时逐样本独立采样轨迹，逐行复用同一逻辑
        return torch.stack([self._apply_one(x[b]) for b in range(x.shape[0])], dim=0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(2, 3, 64, 64)
    model = AUG(channels=3, image_size=64)
    model.train()
    output = model(input_tensor)
    print('=== AUG: AuGhostmentation ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    model.eval()
    out_eval = model(input_tensor)
    print('eval identity:', bool(torch.equal(out_eval, input_tensor)))
    print('train max abs diff:', (output - input_tensor).abs().max().item())
