"""
模型工具模块
包含模型初始化、训练组件设置等功能
"""
import os
import torch
from torch import nn

from config import Config
from models import SQAT_LD
from loss import SQALoss, biasLoss


def initialize_model(config: Config, device: torch.device) -> nn.Module:
    """初始化模型"""
    model = SQAT_LD(args=config.to_dict())
    
    if config.pretrained_model is not None and os.path.exists(config.pretrained_model):
        print(f'[Info] 加载预训练模型: {config.pretrained_model}')
        model = torch.load(config.pretrained_model)
    
    if config.tr_parallel and device.type == 'cuda':
        model = nn.DataParallel(model)
    
    model.to(device)
    return model


def setup_training_components(model: nn.Module, config: Config, ds_train) -> tuple:
    """设置训练组件：优化器、调度器、损失函数"""
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.tr_lr, 
        weight_decay=config.tr_wd
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=50, 
        eta_min=0
    )
    
    # 损失函数
    if config.use_biasloss:
        criterion = biasLoss(
            ds_train.df.db,
            anchor_db=config.tr_bias_anchor_db,
            mapping=config.tr_bias_mapping,
            min_r=config.tr_bias_min_r,
            do_print=(config.tr_verbose > 0)
        )
    else:
        criterion = SQALoss(args=config.to_dict())
    
    return optimizer, scheduler, criterion
