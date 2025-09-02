"""
语音质量评估数据集
支持单端、双端和幻觉增强模式
"""
import multiprocessing
import os
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


class SpeechQualityDataset(Dataset):
    """
    语音质量评估数据集类
    支持单端、双端和幻觉增强模式
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        args: Dict[str, Any],
        double_ended: bool = False,
        filename_column_ref: Optional[str] = None,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
    ) -> None:
        """
        初始化语音质量数据集
        
        Args:
            df: 包含文件路径和标签的DataFrame
            args: 配置参数字典
            double_ended: 是否为双端模式
            filename_column_ref: 参考文件列名（双端模式用）
            norm_mean: 归一化均值
            norm_std: 归一化标准差
        """
        self.df = df
        self.data_dir = args['datapath']
        self.filename_column = args['csv_deg']
        self.user_ID = args['csv_user_ID']
        self.mos_column = args['csv_mos_train']
        self.mean_mos_column = args['csv_mean_train']
        self.to_memory_workers = args.get('to_memory_workers', 0)
        self.target_length = args['target_length']
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.melbins = args['mel_bins']
        self.skip_norm = args.get('skip_norm', False)
        self.hallucinate = args.get('hallucinate', False)
        self.filename_column_ref = filename_column_ref
        self.double_ended = double_ended

        # 内存加载选项
        self.to_memory = False
        if args.get('to_memory', False):
            self._to_memory()

        # 构建用户映射
        self.users = sorted(df['user_ID'].unique())
        self.num_judges = len(self.users)
        self.id_dict = {user_id: index for index, user_id in enumerate(self.users)}

    def _to_memory_multi_helper(self, idx_list: List[int]) -> List[torch.Tensor]:
        """多进程辅助函数：加载一批音频特征"""
        return [self._load_fbank(i) for i in idx_list]
    
    def _to_memory(self) -> None:
        """将所有音频特征加载到内存中"""
        if self.to_memory_workers == 0:
            # 单进程加载
            self.mem_list = [self._load_fbank(idx) for idx in tqdm(range(len(self)), desc="加载音频到内存")]
        else:
            # 多进程加载
            buffer_size = 128
            idx = np.arange(len(self))
            n_bufs = int(len(idx) / buffer_size)
            
            # 分批处理索引
            idx_batches = []
            if n_bufs > 0:
                idx_batches = idx[:buffer_size * n_bufs].reshape(-1, buffer_size).tolist()
            if buffer_size * n_bufs < len(idx):
                remaining = idx[buffer_size * n_bufs:].tolist()
                if remaining:
                    idx_batches.append(remaining)
            
            # 多进程处理
            with multiprocessing.Pool(processes=self.to_memory_workers) as pool:
                mem_list = []
                for batch_result in tqdm(pool.imap(self._to_memory_multi_helper, idx_batches), 
                                       total=len(idx_batches), desc="多进程加载音频"):
                    mem_list.extend(batch_result)
                self.mem_list = mem_list
        
        self.to_memory = True
    
    def _wav2fbank(self, filename: str) -> torch.Tensor:
        """
        将音频文件转换为Mel滤波器组特征
        
        Args:
            filename: 音频文件路径
            
        Returns:
            处理后的Mel特征张量
        """
        # 加载音频并去除直流分量
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        
        # 提取Mel滤波器组特征
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, 
            htk_compat=True, 
            sample_frequency=16000, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=self.melbins, 
            dither=0.0, 
            frame_shift=10
        )

        # 调整时间长度：重复和截断
        n_frames = fbank.shape[0]
        if n_frames < self.target_length:
            # 需要重复帧
            dup_times = self.target_length // n_frames
            remain = self.target_length - n_frames * dup_times
            
            duplicated_frames = [fbank] * dup_times
            if remain > 0:
                duplicated_frames.append(fbank[:remain, :])
            
            fbank = torch.cat(duplicated_frames, dim=0)
        else:
            # 截断到目标长度
            fbank = fbank[:self.target_length, :]
        
        return fbank

    def _load_fbank(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        加载指定索引的音频特征
        
        Args:
            index: 数据索引
            
        Returns:
            音频特征张量，双端或幻觉模式返回元组
        """
        # 主音频文件路径
        file_path = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index])
        
        # 根据模式获取额外的文件路径
        if self.double_ended and self.filename_column_ref:
            file_path_ref = os.path.join(self.data_dir, self.df[self.filename_column_ref].iloc[index])
        elif self.hallucinate:
            # 幻觉模式：将'deg'替换为'est'获得增强音频路径
            file_path_hall = os.path.join(
                self.data_dir, 
                self.df[self.filename_column].iloc[index].replace('deg', 'est', 1)
            )
        
        # 加载主音频特征
        fbank = self._wav2fbank(file_path)
        
        # 处理不同模式
        if self.double_ended and self.filename_column_ref:
            fbank_ref = self._wav2fbank(file_path_ref)
            return (fbank, fbank_ref)
        elif self.hallucinate:
            fbank_hall = self._wav2fbank(file_path_hall)
            return (fbank, fbank_hall)
        else:
            return fbank
            
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, int, int]:
        """
        获取指定索引的数据项
        
        Args:
            index: 数据索引
            
        Returns:
            (fbank, mean_mos, mos, judge_id, index) 元组
        """
        assert isinstance(index, int), '索引必须是整数（不支持切片）'

        # 获取音频特征
        if self.to_memory:
            fbank = self.mem_list[index]
        else:
            fbank = self._load_fbank(index)
        
        # 处理不同模式的特征
        if self.double_ended:
            fbank, fbank_ref = fbank
            # TODO: 双端模式的进一步处理
        elif self.hallucinate:
            fbank, fbank_hall = fbank
            # TODO: 幻觉模式的进一步处理
        
        # 特征预处理：转置以适配模型输入
        fbank = self._preprocess_fbank(fbank)
        
        # 归一化
        if not self.skip_norm and self.norm_mean is not None and self.norm_std is not None:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        
        # 获取标签
        mean_mos = self.df[self.mean_mos_column].iloc[index]
        mos = self.df[self.mos_column].iloc[index]
        
        # 转换为正确的数据类型
        mean_mos = np.array(mean_mos, dtype=np.float32).reshape(-1)
        mos = np.array(mos, dtype=np.float32).reshape(-1)
        
        # 获取评委ID
        user_id = self.df[self.user_ID].iloc[index]
        judge_id = self.id_dict.get(user_id, 0)  # 默认为0如果找不到
        
        return fbank, mean_mos, mos, int(judge_id), index

    def _preprocess_fbank(self, fbank: torch.Tensor) -> torch.Tensor:
        """
        预处理音频特征张量
        
        Args:
            fbank: 原始特征张量
            
        Returns:
            预处理后的特征张量
        """
        # 转置并处理维度以兼容不同的torchaudio版本
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0).squeeze(0)  # 兼容性处理
        fbank = torch.transpose(fbank, 0, 1)
        return fbank

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.df)
