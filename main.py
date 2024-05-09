import argparse
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from tqdm import tqdm

import tensorboardX
import dataloder
import models
import traintest
from SQAloss import SQALoss, biasLoss

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, help='YAML file with config', default=r"./configs/config.yaml")

args = parser.parse_args()
args = vars(args)


def setup_seed(seed):
    # Set seed for reproducibility
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getDevice(args):
    '''
    Train on GPU if available.
    '''         
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    print('[Info] Device: {}'.format(dev))
    
    if "tr_parallel" in args:
        if (dev==torch.device("cpu")) and args['tr_parallel']==True:
            args['tr_parallel']==False 
            print('[Info] Using CPU -> tr_parallel set to False')
    return dev


if __name__ == "__main__":
    print("[Info] I am process %s, running on %s: starting (%s)" %
          (os.getpid(), platform.uname()[1], time.asctime()))
    
    # Load yaml file -----------------------------------------------------------
    with open(args['yaml'], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}
    
    # Set seed -----------------------------------------------------------------
    setup_seed(args['seed'])
    print("[Info] Seed: {}".format(args['seed']))
    
    # Load csv file ------------------------------------------------------------
    csv_file_path = os.path.join(args['datapath'], args['csv_file'])
    dfile = pd.read_csv(csv_file_path)
    
    # Split train and validation ------------------------------------------------
    df_train = dfile[dfile.db.isin(args['csv_db_train'])].reset_index()
    df_val = dfile[dfile.db.isin(args['csv_db_val']) & (dfile['user_ID'] == 'mean_listener')].reset_index()
    df_test = dfile[dfile.db.isin(args['csv_db_test']) & (dfile['user_ID'] == 'mean_listener')].reset_index()
    print('[Info] Training size: {}, Validation size: {}, Test Size: {}'.format(len(df_train), len(df_val), len(df_test)))

    # Dataloader    -------------------------------------------------------------
    ds_train = dataloder.SpeechQualityDataset(df_train, args, norm_mean=args['train_norm_mean'], norm_std=args['train_norm_std'])
    ds_val = dataloder.SpeechQualityDataset(df_val, args, norm_mean=args['val_norm_mean'], norm_std=args['val_norm_std'])
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
    
    num_judges = ds_train.num_judges
    dict = ds_train.id_dict
    mean_linsener_id = dict.get('mean_listener')
    args["num_judges"] = num_judges
    args["mean_linsener_id"] = mean_linsener_id
    print("[Info] Number of judges: {}".format(num_judges))
    print("[Info] ID of mean_listener: {}".format(mean_linsener_id))
    
    # Initialize  -------------------------------------------------------------
    dev = getDevice(args)
    model = models.SQAT_LD(args=args)
    if args['pretrained_model'] is not None:
        model = torch.load(args['pretrained_model'])
    if args['tr_parallel']:
        model = nn.DataParallel(model)
    model.to(dev)
    
    # Optimizer  -------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=float(args['tr_lr']), weight_decay=float(args['tr_wd']))        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0)
    if args['use_biasloss'] == True:
        criterion = biasLoss(
            ds_train.df.db, 
            anchor_db=args['tr_bias_anchor_db'], 
            mapping=args['tr_bias_mapping'], 
            min_r=args['tr_bias_min_r'],
            do_print=(args['tr_verbose']>0),
            )
        args['loss_type'] = 'biasloss'
    else:
        criterion = SQALoss(args=args)
    
    # Train  -------------------------------------------------------------
    
    traintest.train(model,criterion,opt,scheduler,dl_train,dl_val,df_test,args, mean_linsener_id)
    
        