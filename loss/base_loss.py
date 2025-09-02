"""
基础损失函数模块
包含SQALoss主类和核心损失函数
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from .norm_losses import (
    linearity_induced_loss,
    monotonicity_regularization, 
    norm_loss_with_normalization,
    norm_loss_with_min_max_normalization,
    norm_loss_with_mean_normalization,
    norm_loss_with_scaling
)

# 数值稳定性常数
EPS = 1e-8


class SQALoss(torch.nn.Module):
    """
    语音质量评估损失函数类
    支持多种损失函数类型和单调性正则化
    """
    
    def __init__(self, args: Dict[str, Any]) -> None:
        """
        初始化损失函数
        
        Args:
            args: 配置参数字典
        """
        super(SQALoss, self).__init__()
        self.loss_type = args.get('loss_type', 'mae')
        self.alpha = args.get('alpha', [1, 0])
        self.beta = args.get('beta', [0.1, 0.1, 1])
        self.p = args.get('p', 2)
        self.q = args.get('q', 2)
        self.monotonicity_regularization = args.get('monotonicity_regularization', False)
        self.gamma = args.get('gamma', 0.1)
        self.detach = args.get('detach', False)

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            y_pred: 预测值
            y: 真实值
            
        Returns:
            损失值
        """
        return self._compute_loss(y_pred, y)

    def _compute_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        根据配置的损失类型计算损失
        
        Args:
            y_pred: 预测值
            y: 真实值
            
        Returns:
            损失值
        """
        # 基础损失计算
        loss_fn_map = {
            'mae': lambda: F.l1_loss(y_pred, y),
            'mse': lambda: F.mse_loss(y_pred, y),
            'cmse': lambda: self._conditional_mse_loss(y_pred, y),
            'norm-in-norm': lambda: norm_loss_with_normalization(
                y_pred, y, alpha=self.alpha, p=self.p, q=self.q, detach=self.detach
            ),
            'min-max-norm': lambda: norm_loss_with_min_max_normalization(
                y_pred, y, alpha=self.alpha, detach=self.detach
            ),
            'mean-norm': lambda: norm_loss_with_mean_normalization(
                y_pred, y, alpha=self.alpha, detach=self.detach
            ),
            'scaling': lambda: norm_loss_with_scaling(
                y_pred, y, alpha=self.alpha, p=self.p, detach=self.detach
            )
        }
        
        if self.loss_type in loss_fn_map:
            loss = loss_fn_map[self.loss_type]()
        else:
            # 默认使用线性化诱导损失
            loss = linearity_induced_loss(y_pred, y, self.alpha, detach=self.detach)
        
        # 添加单调性正则化
        if self.monotonicity_regularization:
            loss += self.gamma * monotonicity_regularization(y_pred, y, detach=self.detach)
        
        return loss
    
    def _conditional_mse_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """条件MSE损失：只对误差大于阈值的样本计算损失"""
        mse = F.mse_loss(y_pred, y, reduction='none')
        threshold = torch.abs(y_pred - y) > 0.3
        return torch.mean(threshold.float() * mse)
