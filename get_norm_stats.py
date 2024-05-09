# this is a sample code of how to get normalization stats for input spectrogram

import argparse
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import yaml

import dataloder



parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, help='YAML file with config', default=r"./configs/config.yaml")

args = parser.parse_args()
args = vars(args)
 # Load yaml file -----------------------------------------------------------
with open(args['yaml'], "r") as ymlfile:
    args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
args = {**args_yaml, **args}

csv_file_path = os.path.join(args['datapath'], args['csv_file'])
dfile = pd.read_csv(csv_file_path)
df_train = dfile[dfile.db.isin(args['csv_db_train'])].reset_index()
df_val = dfile[dfile.db.isin(args['csv_db_val'])].reset_index()
df_test = dfile[dfile.db.isin(args['csv_db_test'])].reset_index()
print('Training size: {}, Validation size: {}, Test Size: {}'.format(len(df_train), len(df_val), len(df_test)))

# Dataloader    -------------------------------------------------------------
ds_train = dataloder.SpeechQualityDataset(df_train, args)
ds_val = dataloder.SpeechQualityDataset(df_val, args)
df_test = dataloder.SpeechQualityDataset(df_test, args)
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=args['batch_size'],
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=args['num-workers'])
dl_val = torch.utils.data.DataLoader(
    ds_val,
    batch_size=args['batch_size'],
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=args['num-workers'])
dl_test = torch.utils.data.DataLoader(
    df_test,
    batch_size=args['batch_size'],
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=args['num-workers'])

mean=[]
std=[]
for i, (audio_input, labels, index) in enumerate(tqdm(dl_train)):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
print('train')  
print(np.mean(mean), np.mean(std))

mean=[]
std=[]
for i, (audio_input, labels, index) in enumerate(tqdm(dl_val)):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
print('val')
print(np.mean(mean), np.mean(std))

# mean=[]
# std=[]
# for i, (audio_input, labels, index) in enumerate(tqdm(dl_test)):
#     cur_mean = torch.mean(audio_input)
#     cur_std = torch.std(audio_input)
#     mean.append(cur_mean)
#     std.append(cur_std)
# print('test')
# print(np.mean(mean), np.mean(std))
