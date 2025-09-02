"""
偏置损失函数模块
用于处理不同数据库间的系统性差异
"""
import torch
import numpy as np
from scipy.stats import pearsonr
from typing import Optional


class biasLoss:
    """
    偏置损失类
    
    在考虑数据库偏置的情况下计算损失，用于处理不同数据库间的系统性差异
    """
    
    def __init__(self, db, anchor_db: Optional[str] = None, mapping: str = 'first_order', 
                 min_r: float = 0.7, loss_weight: float = 0.0, do_print: bool = True) -> None:
        """
        初始化偏置损失
        
        Args:
            db: 数据库标识序列
            anchor_db: 锚点数据库名称
            mapping: 偏置映射类型
            min_r: 最小相关性阈值
            loss_weight: 损失权重
            do_print: 是否打印信息
        """
        self.db = db
        self.mapping = mapping
        self.min_r = min_r
        self.anchor_db = anchor_db
        self.loss_weight = loss_weight
        self.do_print = do_print
        
        # 初始化偏置参数：[a0, a1, a2, a3] 对应多项式 a0 + a1*x + a2*x^2 + a3*x^3
        self.b = np.zeros((len(db), 4))
        self.b[:, 1] = 1  # 默认线性偏置为1（无偏置）
        self.do_update = False
        
        # 检查是否应用偏置损失
        self.apply_bias_loss = (self.min_r is not None) and (self.mapping is not None)

    def get_loss(self, yb: torch.Tensor, yb_hat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        计算偏置感知损失
        
        Args:
            yb: 真实标签
            yb_hat: 预测值
            idx: 样本索引
            
        Returns:
            损失值
        """
        if self.apply_bias_loss:
            # 获取对应样本的偏置参数
            b = torch.tensor(self.b, dtype=torch.float, device=yb_hat.device)
            b = b[idx, :]
    
            # 应用多项式偏置映射：b0 + b1*x + b2*x^2 + b3*x^3
            yb_hat_corrected = (
                b[:, 0] + 
                b[:, 1] * yb_hat[:, 0] + 
                b[:, 2] * yb_hat[:, 0] ** 2 + 
                b[:, 3] * yb_hat[:, 0] ** 3
            ).view(-1, 1)
            
            # 计算偏置感知损失和常规损失
            loss_bias = self._nan_mse(yb_hat_corrected, yb)   
            loss_normal = self._nan_mse(yb_hat, yb)           
            
            loss = loss_bias + self.loss_weight * loss_normal
        else:
            loss = self._nan_mse(yb_hat, yb)

        return loss
    
    def update_bias(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        """
        更新偏置参数
        
        Args:
            y: 真实值数组
            y_hat: 预测值数组
        """
        if not self.apply_bias_loss:
            return
            
        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)
        
        # 检查是否开始更新偏置
        if not self.do_update:
            valid_mask = ~np.isnan(y) & ~np.isnan(y_hat)
            if valid_mask.sum() > 1:
                r = pearsonr(y[valid_mask], y_hat[valid_mask])[0]
                
                if self.do_print:
                    print(f'--> 偏置更新检查: 最小相关性 {self.min_r:.2f}, 当前相关性 {r:.2f}')
                
                if r > self.min_r:
                    self.do_update = True
            
        # 更新各数据库的偏置参数
        if self.do_update:
            if self.do_print:
                print('--> 正在更新偏置参数')
                
            for db_name in self.db.unique():
                db_idx = (self.db == db_name).to_numpy().nonzero()[0]
                y_hat_db = y_hat[db_idx]
                y_db = y[db_idx]
                
                # 跳过包含NaN的数据库
                if not np.isnan(y_db).any() and len(y_db) > 1:
                    if self.mapping == 'first_order':
                        b_db = self._calc_bias_first_order(y_hat_db, y_db)
                    else:
                        raise NotImplementedError(f"偏置映射类型 '{self.mapping}' 未实现")
                    
                    # 更新偏置参数（锚点数据库除外）
                    if db_name != self.anchor_db:
                        self.b[db_idx, :len(b_db)] = b_db
                
    def _calc_bias_first_order(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算一阶偏置参数：y = a0 + a1 * y_hat"""
        A = np.vstack([np.ones(len(y_hat)), y_hat]).T
        btmp = np.linalg.lstsq(A, y, rcond=None)[0]
        b = np.zeros(4)
        b[0:2] = btmp
        return b
    
    def _nan_mse(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """计算忽略NaN值的MSE损失"""
        err = (y - y_hat).view(-1)
        valid_mask = ~torch.isnan(err)
        valid_err = err[valid_mask]
        return torch.mean(valid_err ** 2) if valid_err.numel() > 0 else torch.tensor(0.0, device=err.device)