"""
设置工具模块
包含随机种子设置、设备管理等工具函数
"""
import os
import random
import torch
import numpy as np
from config import Config


def setup_seed(seed: int) -> None:
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(config: Config) -> torch.device:
    """获取训练设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if config.tr_parallel:
            config.tr_parallel = False
            print('[Info] 使用CPU -> tr_parallel设置为False')
    
    print(f'[Info] 设备: {device}')
    return device
