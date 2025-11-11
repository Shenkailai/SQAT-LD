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
    """
    初始化模型
    
    Args:
        config: 配置对象
        device: 训练设备
        
    Returns:
        初始化后的模型
        
    Raises:
        RuntimeError: 如果模型初始化或加载失败
    """
    try:
        model = SQAT_LD(args=config.to_dict())
    except Exception as e:
        raise RuntimeError(f"模型初始化失败: {str(e)}")
    
    if config.pretrained_model is not None:
        if not os.path.exists(config.pretrained_model):
            print(f'[Warning] 预训练模型文件不存在: {config.pretrained_model}，使用随机初始化')
        else:
            try:
                print(f'[Info] 加载预训练模型: {config.pretrained_model}')
                model = torch.load(config.pretrained_model, map_location=device)
            except Exception as e:
                print(f'[Warning] 加载预训练模型失败: {str(e)}，使用随机初始化')
    
    if config.tr_parallel and device.type == 'cuda':
        if torch.cuda.device_count() > 1:
            print(f'[Info] 使用 {torch.cuda.device_count()} 个GPU进行并行训练')
            model = nn.DataParallel(model)
        else:
            print('[Warning] 只有一个GPU可用，禁用并行训练')
    
    try:
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"模型转移到设备 {device} 失败: {str(e)}")
    
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
