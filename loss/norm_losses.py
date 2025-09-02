"""
标准化损失函数模块
包含各种标准化损失函数的实现
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List

# 数值稳定性常数
EPS = 1e-8


def monotonicity_regularization(y_pred: torch.Tensor, y: torch.Tensor, detach: bool = False) -> torch.Tensor:
    """
    单调性正则化损失
    确保预测值保持与真实值相同的相对排序
    
    Args:
        y_pred: 预测值
        y: 真实值  
        detach: 是否分离梯度
        
    Returns:
        单调性正则化损失
    """
    if y_pred.size(0) > 1:
        # 计算排序损失
        ranking_loss = F.relu((y_pred - y_pred.t()) * torch.sign((y.t() - y)))
        scale = 1 + torch.max(ranking_loss.detach()) if detach else 1 + torch.max(ranking_loss)
        return torch.sum(ranking_loss) / y_pred.size(0) / (y_pred.size(0) - 1) / scale
    else:
        # 单样本情况下返回零损失
        return F.l1_loss(y_pred, y_pred.detach())


def linearity_induced_loss(y_pred: torch.Tensor, y: torch.Tensor, alpha: List[float] = [1, 1], detach: bool = False) -> torch.Tensor:
    """
    线性化诱导损失：使用z-score标准化的MSE损失
    
    Args:
        y_pred: 预测值
        y: 真实值
        alpha: 损失权重 [loss0_weight, loss1_weight]
        detach: 是否分离梯度
        
    Returns:
        线性化诱导损失
    """
    if y_pred.size(0) > 1:
        # z-score标准化: (x-m(x))/sigma(x)
        if detach:
            sigma_hat, m_hat = torch.std_mean(y_pred.detach(), unbiased=False)
        else:
            sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
        
        y_pred_norm = (y_pred - m_hat) / (sigma_hat + EPS)
        
        sigma, m = torch.std_mean(y, unbiased=False)
        y_norm = (y - m) / (sigma + EPS)
        
        scale = 4
        loss0 = loss1 = 0
        
        if alpha[0] > 0:
            # 相关性损失 ~ 1 - rho, rho是PLCC
            loss0 = F.mse_loss(y_pred_norm, y_norm) / scale
            
        if alpha[1] > 0:
            # 决定系数损失 1 - rho^2 = 1 - R^2
            rho = torch.mean(y_pred_norm * y_norm)
            loss1 = F.mse_loss(rho * y_pred_norm, y_norm) / scale
        
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        # 单样本情况下返回零损失
        return F.l1_loss(y_pred, y_pred.detach())


def norm_loss_with_normalization(y_pred: torch.Tensor, y: torch.Tensor, alpha: List[float] = [1, 1], 
                               p: int = 2, q: int = 2, detach: bool = False, exponent: bool = True) -> torch.Tensor:
    """
    标准化损失函数：norm-in-norm
    
    Args:
        y_pred: 预测值
        y: 真实值
        alpha: 损失权重
        p: 误差的p范数
        q: 标准化的q范数
        detach: 是否分离梯度
        exponent: 是否应用指数
        
    Returns:
        标准化损失
    """
    N = y_pred.size(0)
    if N > 1:
        # 预测值标准化
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)
        y_pred = y_pred / (EPS + normalization)
        
        # 真实值标准化
        y = y - torch.mean(y)
        y = y / (EPS + torch.norm(y, p=q))
        
        # 缩放因子
        scale = np.power(2, max(1, 1. / q)) * np.power(N, max(0, 1. / p - 1. / q))
        
        loss0 = loss1 = 0
        
        if alpha[0] > 0:
            err = y_pred - y
            if p < 1:  # 避免梯度爆炸/消失
                err += EPS
            loss0 = torch.norm(err, p=p) / scale
            loss0 = torch.pow(loss0, p) if exponent else loss0
            
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())
            err = rho * y_pred - y
            if p < 1:
                err += EPS
            loss1 = torch.norm(err, p=p) / scale
            loss1 = torch.pow(loss1, p) if exponent else loss1
        
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())


def norm_loss_with_min_max_normalization(y_pred: torch.Tensor, y: torch.Tensor, alpha: List[float] = [1, 1], detach: bool = False) -> torch.Tensor:
    """
    最小-最大标准化损失
    
    Args:
        y_pred: 预测值
        y: 真实值
        alpha: 损失权重
        detach: 是否分离梯度
        
    Returns:
        最小-最大标准化损失
    """
    if y_pred.size(0) > 1:
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - m_hat) / (EPS + M_hat - m_hat)
        y = (y - torch.min(y)) / (EPS + torch.max(y) - torch.min(y))
        
        loss0 = loss1 = 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y)
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())
            loss1 = F.mse_loss(rho * y_pred, y)
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())


def norm_loss_with_mean_normalization(y_pred: torch.Tensor, y: torch.Tensor, alpha: List[float] = [1, 1], detach: bool = False) -> torch.Tensor:
    """
    均值标准化损失
    
    Args:
        y_pred: 预测值
        y: 真实值
        alpha: 损失权重
        detach: 是否分离梯度
        
    Returns:
        均值标准化损失
    """
    if y_pred.size(0) > 1:
        mean_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
        M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
        y_pred = (y_pred - mean_hat) / (EPS + M_hat - m_hat)
        y = (y - torch.mean(y)) / (EPS + torch.max(y) - torch.min(y))
        
        loss0 = loss1 = 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())


def norm_loss_with_scaling(y_pred: torch.Tensor, y: torch.Tensor, alpha: List[float] = [1, 1], p: int = 2, detach: bool = False) -> torch.Tensor:
    """
    缩放标准化损失
    
    Args:
        y_pred: 预测值
        y: 真实值
        alpha: 损失权重
        p: 范数参数
        detach: 是否分离梯度
        
    Returns:
        缩放标准化损失
    """
    if y_pred.size(0) > 1:
        normalization = torch.norm(y_pred.detach(), p=p) if detach else torch.norm(y_pred, p=p)
        y_pred = y_pred / (EPS + normalization)
        y = y / (EPS + torch.norm(y, p=p))
        
        loss0 = loss1 = 0
        if alpha[0] > 0:
            loss0 = F.mse_loss(y_pred, y) / 4
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.t(), y.t())
            loss1 = F.mse_loss(rho * y_pred, y) / 4
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())
