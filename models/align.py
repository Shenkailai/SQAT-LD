"""
对齐模块实现
包含多种注意力机制用于音频特征对齐
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple


class AttDot(torch.nn.Module):
    """
    点积注意力机制
    计算查询和键之间的点积作为注意力分数
    """
    
    def __init__(self) -> None:
        """初始化点积注意力模块"""
        super().__init__()
    
    def forward(self, query: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [B, N, D]
            y: 键值张量 [B, M, D]
            
        Returns:
            (注意力分数, 最大相似度) 元组
        """
        att = torch.bmm(query, y.transpose(2, 1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    


class AttCosine(torch.nn.Module):
    """
    余弦相似度注意力机制
    使用余弦相似度计算查询和键之间的注意力分数
    """
    
    def __init__(self) -> None:
        """初始化余弦注意力模块"""
        super().__init__()
        self.pdist = nn.CosineSimilarity(dim=3)
    
    def forward(self, query: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [B, N, D]
            y: 键值张量 [B, M, D]
            
        Returns:
            (注意力分数, 最大相似度) 元组
        """
        att = self.pdist(query.unsqueeze(2), y.unsqueeze(1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim    
    


class AttDistance(torch.nn.Module):
    """
    距离注意力机制
    使用距离度量计算查询和键之间的注意力分数
    """
    
    def __init__(self, dist_norm: int = 1, weight_norm: int = 1) -> None:
        """
        初始化距离注意力模块
        
        Args:
            dist_norm: 距离范数阶数
            weight_norm: 权重范数阶数
        """
        super().__init__()
        self.dist_norm = dist_norm
        self.weight_norm = weight_norm
    
    def forward(self, query: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [B, N, D]
            y: 键值张量 [B, M, D]
            
        Returns:
            (注意力分数, 最大相似度) 元组
        """
        att = (query.unsqueeze(1) - y.unsqueeze(2)).abs().pow(self.dist_norm)
        att = att.mean(dim=3).pow(self.weight_norm)
        att = -att.transpose(2, 1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    


class AttBahdanau(torch.nn.Module):
    """
    Bahdanau 注意力机制
    使用加性注意力模型，适用于序列到序列任务
    """
    
    def __init__(self, q_dim: int, y_dim: int, att_dim: int = 128) -> None:
        """
        初始化 Bahdanau 注意力模块
        
        Args:
            q_dim: 查询特征维度
            y_dim: 键值特征维度
            att_dim: 注意力隐藏层维度
        """
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.att_dim = att_dim
        self.Wq = nn.Linear(self.q_dim, self.att_dim)
        self.Wy = nn.Linear(self.y_dim, self.att_dim)
        self.v = nn.Linear(self.att_dim, 1)
    
    def forward(self, query: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [B, N, q_dim]
            y: 键值张量 [B, M, y_dim]
            
        Returns:
            (注意力分数, 最大相似度) 元组
        """
        att = torch.tanh(self.Wq(query).unsqueeze(1) + self.Wy(y).unsqueeze(2))
        att = self.v(att).squeeze(3).transpose(2, 1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim



class AttLuong(torch.nn.Module):
    """
    Luong 注意力机制
    使用乘性注意力模型，计算效率高
    """
    
    def __init__(self, q_dim: int, y_dim: int) -> None:
        """
        初始化 Luong 注意力模块
        
        Args:
            q_dim: 查询特征维度
            y_dim: 键值特征维度
        """
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.W = nn.Linear(self.y_dim, self.q_dim)
    
    def forward(self, query: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [B, N, q_dim]
            y: 键值张量 [B, M, y_dim]
            
        Returns:
            (注意力分数, 最大相似度) 元组
        """
        att = torch.bmm(query, self.W(y).transpose(2, 1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim



class ApplyHardAttention(torch.nn.Module):
    """
    硬注意力应用模块
    使用最大注意力分数位置进行硬对齐
    """
    
    def __init__(self) -> None:
        """初始化硬注意力应用模块"""
        super().__init__()
        self.idx = None  # 存储最大注意力位置索引
    
    def forward(self, y: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        """
        应用硬注意力
        
        Args:
            y: 输入特征 [B, M, D]
            att: 注意力分数 [B, N, M]
            
        Returns:
            对齐后的特征 [B, N, D]
        """
        self.idx = att.argmax(2)
        y = y[torch.arange(y.shape[0]).unsqueeze(-1), self.idx]
        return y    
    


class ApplySoftAttention(torch.nn.Module):
    """
    软注意力应用模块
    使用加权平均进行软对齐
    """
    
    def __init__(self) -> None:
        """初始化软注意力应用模块"""
        super().__init__()
    
    def forward(self, y: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        """
        应用软注意力
        
        Args:
            y: 输入特征 [B, M, D]
            att: 注意力分数 [B, N, M]
            
        Returns:
            对齐后的特征 [B, N, D]
        """
        y = torch.bmm(att, y)
        return y     



class Alignment(torch.nn.Module):
    """
    对齐模块
    支持五种不同的注意力机制用于特征对齐
    
    支持的注意力方法：
    - 'bahd': Bahdanau 注意力
    - 'luong': Luong 注意力
    - 'dot': 点积注意力
    - 'cosine': 余弦相似度注意力
    - 'distance': 距离注意力
    - 'none': 不使用注意力
    
    支持的应用方法：
    - 'soft': 软注意力（加权平均）
    - 'hard': 硬注意力（最大值选择）
    """
    
    def __init__(
        self,
        att_method: str,
        apply_att_method: str,
        q_dim: int = None,
        y_dim: int = None,
    ) -> None:
        """
        初始化对齐模块
        
        Args:
            att_method: 注意力计算方法
            apply_att_method: 注意力应用方法
            q_dim: 查询特征维度（某些注意力方法需要）
            y_dim: 键值特征维度（某些注意力方法需要）
            
        Raises:
            NotImplementedError: 如果提供的方法不支持
        """
        super().__init__()
        
        # 选择注意力计算方法
        if att_method in ('bahd', 'luong'):
            if q_dim is None or y_dim is None:
                raise ValueError(f"{att_method.capitalize()} 注意力需要提供 q_dim 和 y_dim")
        if att_method == 'bahd':
            self.att = AttBahdanau(q_dim=q_dim, y_dim=y_dim)
        elif att_method == 'luong':
            self.att = AttLuong(q_dim=q_dim, y_dim=y_dim)
        elif att_method == 'dot':
            self.att = AttDot()
        elif att_method == 'cosine':
            self.att = AttCosine()
        elif att_method == 'distance':
            self.att = AttDistance()
        elif att_method == 'none' or att_method is None:
            self.att = None
        else:
            raise NotImplementedError(f"不支持的注意力方法: {att_method}")
        
        # 选择注意力应用方法
        if apply_att_method == 'soft':
            self.apply_att = ApplySoftAttention()
        elif apply_att_method == 'hard':
            self.apply_att = ApplyHardAttention()
        else:
            raise NotImplementedError(f"不支持的注意力应用方法: {apply_att_method}")
    
    def forward(self, query: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 对齐两个特征序列
        
        Args:
            query: 查询特征 [B, N, D]
            y: 键值特征 [B, M, D]
            
        Returns:
            对齐后的特征 [B, N, D]
        """
        if self.att is not None:
            att_score, sim = self.att(query, y)
            att_score = F.softmax(att_score, dim=2)
            y = self.apply_att(y, att_score)
        return y        
