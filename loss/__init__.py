"""
损失函数模块
包含语音质量评估的各种损失函数实现
"""

from .base_loss import SQALoss
from .bias_loss import biasLoss
from .norm_losses import (
    monotonicity_regularization,
    linearity_induced_loss,
    norm_loss_with_normalization,
    norm_loss_with_min_max_normalization,
    norm_loss_with_mean_normalization,
    norm_loss_with_scaling
)

__all__ = [
    'SQALoss',
    'biasLoss',
    'monotonicity_regularization',
    'linearity_induced_loss',
    'norm_loss_with_normalization',
    'norm_loss_with_min_max_normalization',
    'norm_loss_with_mean_normalization',
    'norm_loss_with_scaling'
]
