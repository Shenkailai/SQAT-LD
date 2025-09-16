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
    """加载并分割数据集"""
    csv_file_path = os.path.join(config.datapath, config.csv_file)
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
    
    dfile = pd.read_csv(csv_file_path)
    
    # 分割训练、验证和测试集
    df_train = dfile[dfile.db.isin(config.csv_db_train)].reset_index()
    df_val = dfile[dfile.db.isin(config.csv_db_val) & (dfile['user_ID'] == 'mean_listener')].reset_index()
    df_test = dfile[dfile.db.isin(config.csv_db_test) & (dfile['user_ID'] == 'mean_listener')].reset_index()
    
    print(f'[Info] 训练集大小: {len(df_train)}, 验证集大小: {len(df_val)}, 测试集大小: {len(df_test)}')
    
    return df_train, df_val, df_test


def create_dataloader(df: pd.DataFrame, config: Config, norm_mean: float, norm_std: float, 
                     shuffle: bool = True) -> Tuple[DataLoader, object]:
    """创建数据加载器"""
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
