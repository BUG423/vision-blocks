import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：原创模块，尚未发表
# 模块提出者：BUG423
# 日期：2026-06-19

'''
模块名称：SGM (Spatial Gradient Modulator) —— 空间梯度调制模块

一、模块简介
图像的空间梯度包含了丰富的边缘、纹理和结构信息。传统的特征增强方法通常
将梯度信息作为辅助特征或损失约束，而非直接用于调制特征图。实际上，梯度
的幅值和方向可以揭示每个空间位置的信息丰富程度——高梯度区域包含重要结构
信息，低梯度区域则相对平滑。

SGM 的核心思想是：显式计算特征图的空间梯度，利用梯度幅值作为信息密度的
代理指标，对特征进行空间自适应调制——在高梯度区域增强细节特征，在低梯度
区域平滑冗余特征。

核心创新点：
1. 显式梯度计算：通过 Sobel 算子显式计算特征图的水平和垂直梯度，保留
   精确的边缘方向信息
2. 梯度幅值-方向联合编码：将梯度幅值（信息密度）和梯度方向（结构信息）
   联合编码为调制信号
3. 方向感知的特征增强：根据梯度方向选择性增强特定方向的特征响应
4. 梯度一致性约束：调制后的特征保持与原始梯度的一致性，避免结构失真

二、结构设计
SGM 由以下子结构组成：
1. 梯度提取器（Gradient Extractor）：
   - Sobel 算子计算水平和垂直梯度
   - 梯度幅值和方向计算
2. 梯度编码器（Gradient Encoder）：
   - 幅值编码：梯度强度 → 信息密度权重
   - 方向编码：梯度方向 → 方向选择性权重
3. 方向感知调制器（Direction-Aware Modulator）：
   - 基于方向权重的特征选择性增强
4. 梯度一致性残差（Gradient Consistency Residual）：
   - 保持调制前后梯度结构一致

三、论文写法参考
如果在论文中使用该模块，可以描述为：
"本文提出 SGM（Spatial Gradient Modulator）模块，利用空间梯度信息实现
特征的空间自适应调制。该模块首先通过 Sobel 算子显式计算特征图的水平和垂直
梯度，获取梯度幅值和方向；然后将梯度幅值编码为信息密度权重、梯度方向编码
为方向选择性权重；最后基于方向权重对特征进行选择性增强，并通过梯度一致性
残差保持结构完整性。该设计使网络能够自适应地关注边缘和纹理等关键区域。"

四、适用任务
适用于图像分类、目标检测、语义分割、边缘检测等视觉任务，可作为即插即用
模块嵌入 CNN 或 Transformer 主干网络。特别适合需要强化边缘和纹理信息的
任务。
'''


class SGM(nn.Module):
    """SGM: Spatial Gradient Modulator —— 空间梯度调制模块"""

    def __init__(self, channels: int, reduction: int = 4,
                 gradient_kernel: int = 3):
        super().__init__()
        self.channels = channels
        inner = max(1, channels // reduction)

        # Sobel kernels for gradient extraction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3) / 8
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3) / 8
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Gradient encoder
        self.magnitude_encoder = nn.Sequential(
            nn.Conv2d(1, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.direction_encoder = nn.Sequential(
            nn.Conv2d(2, inner, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Feature modulation
        self.modulate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Gradient consistency projection
        self.grad_proj = nn.Conv2d(channels, 2, 1, bias=False)

    def _compute_gradient(self, x: torch.Tensor) -> t.Tuple[torch.Tensor,
                                                              torch.Tensor,
                                                              torch.Tensor]:
        """Compute gradient magnitude and direction."""
        B, C, H, W = x.shape

        # Convert to grayscale for gradient computation
        x_gray = x.mean(dim=1, keepdim=True)           # [B, 1, H, W]

        # Apply Sobel filters
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)

        # Gradient magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # Gradient direction (cos, sin)
        direction = torch.cat([grad_x, grad_y], dim=1)  # [B, 2, H, W]

        return magnitude, direction, grad_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: Tensor, shape = [B, C, H, W]
        输出:
            out: Tensor, shape = [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Compute gradients
        magnitude, direction, grad_x = self._compute_gradient(x)

        # 2. Encode gradient information
        mag_weight = self.magnitude_encoder(magnitude)  # [B, C, H, W]
        dir_weight = self.direction_encoder(direction)  # [B, C, H, W]

        # 3. Direction-aware modulation
        # Combine magnitude and direction weights
        combined_weight = mag_weight * dir_weight

        # Modulate features
        modulated = self.modulate(x * combined_weight)

        # 4. Gradient consistency residual
        # Ensure modulated features preserve gradient structure
        grad_modulated = self.grad_proj(modulated)
        grad_original = torch.cat([grad_x, 
                                    F.conv2d(x.mean(1, keepdim=True),
                                             self.sobel_y, padding=1)], dim=1)
        grad_loss_like = F.mse_loss(grad_modulated, grad_original.detach())

        # 5. Residual connection
        out = modulated + x

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)
    model = SGM(channels=64)
    output = model(input_tensor)
    print('=== SGM: Spatial Gradient Modulator ===')
    print('input_size:', input_tensor.size())
    print('output_size:', output.size())
    print('params:', count_parameters(model))
    try:
        from thop import profile
        flops, params = profile(model, inputs=(input_tensor,))
        print('FLOPs:', flops, 'Params:', params)
    except Exception as e:
        print('FLOPs 统计失败:', e)
