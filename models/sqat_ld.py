"""
SQAT-LD模型实现
语音质量评估的Transformer架构，结合注意力机制和评委偏好建模
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from models.align import Alignment
from .ssast_models import SSASTModel
from einops import rearrange


class TABlock(nn.Module):
    """
    时间注意力块 (Temporal Attention Block)
    用于处理音频特征的时序注意力机制
    """
    
    def __init__(self, dim: int, drop: float = 0.1) -> None:
        """
        初始化时间注意力块
        
        Args:
            dim: 特征维度
            drop: Dropout概率
        """
        super().__init__()
        self.c_q = nn.Linear(dim, dim)  # 查询投影
        self.c_k = nn.Linear(dim, dim)  # 键投影
        self.c_v = nn.Linear(dim, dim)  # 值投影
        self.norm_fact = dim ** -0.5    # 缩放因子
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, N]
            
        Returns:
            注意力处理后的特征
        """
        residual = x
        B, C, N = x.shape
        
        # 计算查询、键、值
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        # 计算注意力权重
        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        
        # 残差连接
        return x + residual


class SQAT_LD(nn.Module):
    """
    SQAT-LD: 语音质量评估模型
    结合SSAST特征提取、评委偏好建模和注意力对齐机制
    """
    
    def __init__(self, args: Dict[str, Any], **kwargs) -> None:
        """
        初始化SQAT-LD模型
        
        Args:
            args: 模型配置参数
        """
        super().__init__()

        # 基本参数
        drop = 0.1
        self.f_dim = args['mel_bins'] // args['fshape']  # 频率维度
        self.t_dim = args['target_length'] // args['tshape']  # 时间维度
        self.embed_dim = args.get('embed_dim', 768)
        self.num_tab = args.get('num_tab', 2)
        self.att_method = args.get('att_method', 'luong')
        self.apply_att_method = args.get('apply_att_method', 'hard')
        
        # 评委嵌入
        self.num_judges = args.get("num_judges", 1)
        self.judge_embedding = nn.Embedding(
            num_embeddings=self.num_judges, 
            embedding_dim=self.embed_dim
        )
        
        # SSAST骨干网络
        self.ast = SSASTModel(
            label_dim=1, 
            fshape=args['fshape'], 
            tshape=args['tshape'], 
            fstride=args['fstride'], 
            tstride=args['tstride'],
            input_fdim=args['mel_bins'], 
            input_tdim=args['target_length'], 
            model_size=args.get('model_size', 'base'),
            pretrain_stage=False, 
            load_pretrained_mdl_path=args.get('load_pretrained_mdl_path')
        )
        
        # 时间注意力块
        self.tablock1 = nn.ModuleList([
            TABlock(self.f_dim * self.t_dim) for _ in range(self.num_tab)
        ])
        
        # 特征融合
        self.conv1 = nn.Conv2d(self.embed_dim * 2, self.embed_dim, 1, 1, 0)
        
        # 对齐模块
        self.align = Alignment(
            self.att_method, 
            self.apply_att_method,
            q_dim=self.embed_dim,
            y_dim=self.embed_dim,
        )
        
        # 质量评分网络
        self.fc_score = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.embed_dim, 2),
            nn.ReLU()
        )
        
        # 权重网络
        self.fc_weight = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.embed_dim, 2),
            nn.Sigmoid()
        )
        
        # 最终预测头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim), 
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, judge_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 音频特征 [B, T, F]
            judge_id: 评委ID [B]
            
        Returns:
            (全局质量分, 评委特定质量分)
        """
        # SSAST特征提取
        x = self.ast(x)  # [B, seq_len, embed_dim]
        
        # 评委特征嵌入
        judge_feat = self.judge_embedding(judge_id)  # [B, embed_dim]
        
        # 扩展评委特征到序列长度
        seq_len = self.f_dim * self.t_dim
        judge_feat = judge_feat.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, embed_dim]
        
        # 特征对齐
        judge_feat = self.align(x, judge_feat)
        
        # 融合音频特征和评委特征
        x = torch.cat((x, judge_feat), dim=2)  # [B, seq_len, 2*embed_dim]
        
        # 重排列用于时间注意力处理
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.f_dim, w=self.t_dim)
        
        # 应用时间注意力块
        for tab in self.tablock1:
            x = tab(x)
            
        # 重排列并应用卷积融合
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.f_dim, w=self.t_dim)
        x = self.conv1(x)  # 降维到原始嵌入维度
        
        # 最终重排列用于评分
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.f_dim, w=self.t_dim)
        
        # 加权评分计算（批量化处理）
        scores = self._compute_weighted_score(x)
        judge_scores = self._compute_weighted_score(x)  # 使用相同的网络
        
        return scores.unsqueeze(1), judge_scores.unsqueeze(1)
    
    def _compute_weighted_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算加权质量评分
        
        Args:
            x: 特征张量 [B, seq_len, embed_dim]
            
        Returns:
            质量评分 [B]
        """
        # 计算特征评分和权重
        features = self.fc_score(x)  # [B, seq_len, 2]
        weights = self.fc_weight(x)  # [B, seq_len, 2]
        
        # 加权平均
        weighted_features = features * weights  # [B, seq_len, 2]
        scores = torch.sum(weighted_features, dim=(1, 2)) / torch.sum(weights, dim=(1, 2))
        
        return scores


# 测试代码已移除 - 在实际项目中建议使用单独的测试文件

