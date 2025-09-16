"""
评估指标模块
包含各种评估指标的计算函数
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from typing import Tuple


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    计算评估指标：SRCC, PLCC, RMSE
    
    Args:
        predictions: 预测值数组
        labels: 真实标签数组
        
    Returns:
        (SRCC, PLCC, RMSE) 元组
    """
    pred_flat = np.squeeze(predictions)
    label_flat = np.squeeze(labels)
    
    rho_s, _ = spearmanr(pred_flat, label_flat)
    rho_p, _ = pearsonr(pred_flat, label_flat)
    rmse = metrics.mean_squared_error(pred_flat, label_flat)
    
    return rho_s, rho_p, rmse
