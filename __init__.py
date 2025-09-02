"""
SQAT-LD: 语音质量评估模型
Speech Quality Assessment with Transformer and Listener-aware Decomposition

模块化架构包含：
- config: 配置管理
- data: 数据集和数据处理
- loss: 损失函数
- models: 模型架构
- training: 训练和评估
- utils: 工具函数
"""

__version__ = "1.0.0"
__author__ = "SQAT-LD Team"

# 主要组件导入
from config import Config
from data import SpeechQualityDataset
from loss import SQALoss, biasLoss
from models import SQAT_LD
from training import train
from utils import setup_seed, get_device

__all__ = [
    'Config',
    'SpeechQualityDataset',
    'SQALoss',
    'biasLoss',
    'SQAT_LD',
    'train',
    'setup_seed',
    'get_device'
]
