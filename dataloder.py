#%% Dataset
import multiprocessing
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


class SpeechQualityDataset(Dataset):
    '''
    Dataset for Speech Quality Model.
    '''  
    def __init__(
        self,
        df,
        args,
        double_ended=False,
        filename_column_ref=None,
        norm_mean=None,
        norm_std=None,
        ):

        self.df = df
        self.data_dir = args['datapath']
        self.filename_column = args['csv_deg']
        self.user_ID = args['csv_user_ID']
        self.mos_column = args['csv_mos_train']   
        self.mean_mos_column = args['csv_mean_train']     
        self.to_memory_workers = args['to_memory_workers']
        self.target_length = args['target_length']  
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.melbins = args['mel_bins']
        self.skip_norm = args['skip_norm']
        self.hallucinate = args['hallucinate']
        self.filename_column_ref = filename_column_ref
        self.double_ended = double_ended

        # if True load all specs to memory
        self.to_memory = False
        if args['to_memory']:
            self._to_memory()
            
        self.users = sorted(df['user_ID'].unique())
        self.num_judges = len(self.users)
        # 构造字典
        self.id_dict = {id_: index for index, id_ in enumerate(self.users)}

            
        
            
    def _to_memory_multi_helper(self, idx):
        return [self._load_spec(i) for i in idx]
    
    def _to_memory(self):
        if self.to_memory_workers==0:
            self.mem_list = [self._load_spec(idx) for idx in tqdm(range(len(self)))]
        else: 
            buffer_size = 128
            idx = np.arange(len(self))
            n_bufs = int(len(idx)/buffer_size) 
            idx = idx[:buffer_size*n_bufs].reshape(-1,buffer_size).tolist() + idx[buffer_size*n_bufs:].reshape(1,-1).tolist()  
            pool = multiprocessing.Pool(processes=self.to_memory_workers)
            mem_list = []
            for out in tqdm(pool.imap(self._to_memory_multi_helper, idx), total=len(idx)):
                mem_list = mem_list + out
            self.mem_list = mem_list
            pool.terminate()
            pool.join()    
        self.to_memory=True 
    
    def _wav2fbank(self, filename):

        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        
        n_frames = fbank.shape[0]
        dup_times = self.target_length // n_frames
        remain = self.target_length - n_frames * dup_times
        to_dup = [fbank for t in range(dup_times)]
        to_dup.append(fbank[:remain, :])
        fbank = torch.Tensor(np.concatenate(to_dup, axis = 0))
        
        return fbank

    def _load_fbank(self, index):
        
        # Load fbank    
        file_path = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index])

        if self.double_ended:
            file_path_ref = os.path.join(self.data_dir, self.df[self.filename_column_ref].iloc[index])
        elif self.hallucinate:
            file_path_hall = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index].replace('deg', 'est', 1))
            
            
        fbank = self._wav2fbank(file_path)
        
        if self.double_ended:
            fbank_ref = self._wav2fbank(file_path_ref) 
            fbank = (fbank, fbank_ref)
        elif self.hallucinate:
            fbank_hall = self._wav2fbank(file_path_hall)
            fbank = (fbank, fbank_hall)
                
        return fbank
            
    def __getitem__(self, index):
        assert isinstance(index, int), 'index must be integer (no slice)'

        if self.to_memory:
            fbank = self.mem_list[index]
        else:
            fbank = self._load_fbank(index)
            
        if self.double_ended:               
            fbank, fbank_ref = fbank
        elif self.hallucinate:
            fbank, fbank_hall = fbank
        
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass
        
        mean_mos = self.df[self.mean_mos_column].iloc[index].reshape(-1).astype('float32')
        mos = self.df[self.mos_column].iloc[index].reshape(-1).astype('float32')   

        judge_id = self.id_dict.get(self.df[self.user_ID].iloc[index])
        # print("judge_id: {}".format(judge_id))
        return fbank, mean_mos, mos, int(judge_id), index

    def __len__(self):
        return len(self.df)