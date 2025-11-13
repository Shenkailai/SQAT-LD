"""
数据处理工具模块
包含数据加载、分割、数据加载器创建等功能
"""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Tuple

from config import Config
from data import SpeechQualityDataset


def load_and_split_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载并分割数据集
    
    Args:
        config: 配置对象
        
    Returns:
        (训练集, 验证集, 测试集) DataFrame元组
        
    Raises:
        FileNotFoundError: 如果CSV文件不存在
        ValueError: 如果数据集为空或缺少必要列
    """
    csv_file_path = os.path.join(config.datapath, config.csv_file)
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
    
    try:
        dfile = pd.read_csv(csv_file_path)
    except Exception as e:
        raise RuntimeError(f"读取CSV文件失败: {str(e)}")
    
    # 验证必要的列是否存在
    required_columns = ['db', 'user_ID']
    missing_columns = [col for col in required_columns if col not in dfile.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
    
    if len(dfile) == 0:
        raise ValueError("CSV文件为空")
    
    # 分割训练、验证和测试集
    df_train = dfile[dfile.db.isin(config.csv_db_train)].reset_index()
    df_val = dfile[dfile.db.isin(config.csv_db_val) & (dfile['user_ID'] == 'mean_listener')].reset_index()
    df_test = dfile[dfile.db.isin(config.csv_db_test) & (dfile['user_ID'] == 'mean_listener')].reset_index()
    
    # 验证数据集不为空
    if len(df_train) == 0:
        raise ValueError("训练集为空，请检查csv_db_train配置")
    
    print(f'[Info] 训练集大小: {len(df_train)}, 验证集大小: {len(df_val)}, 测试集大小: {len(df_test)}')
    
    return df_train, df_val, df_test


def create_dataloader(df: pd.DataFrame, config: Config, norm_mean: float, norm_std: float, 
                     shuffle: bool = True) -> Tuple[DataLoader, object]:
    """
    创建数据加载器
    
    Args:
        df: 数据DataFrame
        config: 配置对象
        norm_mean: 归一化均值
        norm_std: 归一化标准差
        shuffle: 是否打乱数据
        
    Returns:
        (数据加载器, 数据集) 元组
        
    Raises:
        ValueError: 如果参数无效
    """
    # 输入验证
    if df is None or len(df) == 0:
        raise ValueError("数据DataFrame为空")
    
    if norm_std <= 0:
        raise ValueError(f"归一化标准差必须大于0，得到: {norm_std}")
    
    if config.batch_size <= 0:
        raise ValueError(f"批次大小必须大于0，得到: {config.batch_size}")
    
    if config.num_workers < 0:
        raise ValueError(f"工作进程数不能为负数，得到: {config.num_workers}")
    
    dataset = SpeechQualityDataset(df, config.to_dict(), norm_mean=norm_mean, norm_std=norm_std)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=config.num_workers
    )
    
    return dataloader, dataset
