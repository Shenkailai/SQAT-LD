import logging
import os
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dataloder
from SQAloss import biasLoss

def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader, args):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for batch_x, batch_y, batch_mos, batch_judge_id, index in tqdm(train_loader):
        x_d = batch_x.cuda()
        labels = batch_y
        labels = labels.type(torch.FloatTensor).cuda()
        
        batch_mos = batch_mos.type(torch.FloatTensor).cuda()
        batch_judge_id = batch_judge_id.type(torch.LongTensor).cuda()
        pred_d, judge_mos = net(x_d, batch_judge_id)

        optimizer.zero_grad()
        if args['use_biasloss'] == True:
            loss1 = criterion.get_loss(yb = labels, yb_hat = pred_d, idx = index)
            loss2 = criterion.get_loss(yb = batch_mos, yb_hat = judge_mos, idx = index)
            loss = loss1 + loss2
        else:
            loss1 = criterion(pred_d, labels)
            loss2 = criterion(judge_mos, batch_mos)
            loss = loss1 + loss2
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rmse = metrics.mean_squared_error(np.squeeze(
        pred_epoch), np.squeeze(labels_epoch)) 

    ret_loss = np.mean(losses)
    if args['use_biasloss'] == True:
        criterion.update_bias(np.squeeze(labels_epoch), np.squeeze(pred_epoch))
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}/ MSE:{:.4}'.format(
        epoch + 1, ret_loss, rho_s, rho_p, rmse))

    return ret_loss, rho_s, rho_p, rmse


def eval_epoch(epoch, net, criterion, test_loader, mean_linsener_id):
    with torch.no_grad():

        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for batch_x, batch_y, batch_mos, batch_judge_id, index in tqdm(test_loader):
            pred = 0
           
            x_d = batch_x.cuda()
            labels = batch_y
            labels = labels.type(torch.FloatTensor).cuda()
            batch_judge_id = batch_judge_id.type(torch.LongTensor).cuda()
            batch_judge_id[:] = mean_linsener_id
            pred, judge_mos = net(x_d, batch_judge_id)

            
          
            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = metrics.mean_squared_error(np.squeeze(
            pred_epoch), np.squeeze(labels_epoch))
        return rho_s, rho_p, rmse
    
def test(df_val, epoch, net, criterion, args, mean_linsener_id):
    for db_name in df_val.db.astype("category").cat.categories:
       
        if db_name == 'NISQA_TEST_LIVETALK':
            norm_stats_val = {'nisqa': [-9.051971, 3.7531793]}
        elif db_name == 'NISQA_TEST_FOR':
            norm_stats_val = {'nisqa': [-8.937617, 4.2769117]}
        elif db_name == 'NISQA_TEST_P501':
            norm_stats_val = {'nisqa': [-9.90131, 4.708985]}
        elif db_name == 'NISQA_VAL_LIVE':
            norm_stats_val = {'nisqa': [-9.823734, 3.6818407]}
        elif db_name == 'NISQA_VAL_SIM':
            norm_stats_val = {'nisqa': [-8.027123, 4.3762627]}
        elif db_name == 'NISQA_VAL':
            norm_stats_val = {'nisqa': [-8.185567, 4.3552947]}
        elif db_name == 'tencent_with':
            norm_stats_val = {'tencent': [-8.642287, 4.199733]}
        elif db_name == 'tencent_without':
            norm_stats_val = {'tencent': [-9.084293, 5.4488106]}
        elif db_name == 'TMHINTQI_Valid':
            norm_stats_val = {'voicemos2023': [-4.865034, 4.1673865]}
        elif db_name == 'VoiceMOS2022_mian_Test':
            norm_stats_val = {'voicemos2023': [-8.353344, 4.4906945]}
        elif db_name == 'VoiceMOS2022_OOD_unlabeled1':
            norm_stats_val = {'voicemos2023': [-7.7979817, 4.426462]}
        elif db_name == 'VoiceMOS2022_OOD_Test1':
            norm_stats_val = {'voicemos2023': [-7.64568, 4.315633]}
        
        args['norm_mean'] = norm_stats_val[args['dataset']][0]
        args['norm_std'] = norm_stats_val[args['dataset']][1]
        
        df_db = df_val.loc[df_val.db==db_name]
        
        ds_val = dataloder.SpeechQualityDataset(df_db, args, norm_mean=norm_stats_val[args['dataset']][0], norm_std=norm_stats_val[args['dataset']][1])
        dl_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=args['batch_size'],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=args['num-workers'])
       
        # Eval  -------------------------------------------------------------
        logging.info('Test in {}...'.format(db_name))
        rho_s, rho_p, rmse = eval_epoch(epoch, net, criterion, dl_val, mean_linsener_id)
        print('Eval {}, SRCC:{:.4}, PLCC:{:.4}, MSE:{:.4}'.format(
                db_name, rho_s, rho_p, rmse))
        logging.info('Eval {} - [SRCC]:{:.4}, [PLCC]:{:.4}, [MSE]:{:.4}'.format(
                        db_name, rho_s, rho_p, rmse))
        
    
    
 
def train(net, criterion, optimizer, scheduler, train_loader, val_loader, df_val, args, mean_linsener_id):
    
    model_tag = 'bs_{}_seed_{}_{}_{}_{}_{}'.format(args['batch_size'],args['seed'],args['loss_type'],args['att_method'],args['apply_att_method'],args['comment'])
    
    # create log and tensorboard directory ------------------------------------
    log_path = os.path.join(args['output_dir'], args['dataset'], 'logs')
    tensorboard_path = os.path.join(args['output_dir'], args['dataset'], 'tensorboard')
    if not os.path.exists(log_path):
        print('Creating log directory: {:s}'.format(log_path))
        os.makedirs(log_path)
    if not os.path.exists(tensorboard_path):
        print('Creating tensorboard directory: {:s}'.format(tensorboard_path))
        os.makedirs(tensorboard_path)
    
    # create model directory ---------------------------------------------------
    model_path = os.path.join(args['output_dir'], args['dataset'], 'models/', model_tag)
    if not os.path.exists(model_path):
        print('Creating model directory: {:s}'.format(model_path))
        os.makedirs(model_path)
        
    # set logger ----------------------------------------------------------------
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(
        log_path, model_tag) + '.log', level=logging.INFO, format=LOG_FORMAT)
    logging.info("[Info] Number of judges: {}".format(args["num_judges"]))
    logging.info("[Info] ID of mean_listener: {}".format(args["mean_linsener_id"]))
    # set tensorboard -----------------------------------------------------------
    writer = SummaryWriter(tensorboard_path)
    
    # set best results ---------------------------------------------------------
    best_srocc = 0
    best_plcc = 0
    besst_rmse = 1
    
    # start training ------------------------------------------------------------
    print('Now starting training for {:d} epochs'.format(args['n-epochs']))
    for epoch in range(args['n-epochs']):
        start_time = time.time()
        logging.info('========================================= [ Running training epoch {} ]============================================='.format(epoch + 1))
        
        loss_train, rho_s, rho_p, rmse = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader, args)
        
       
        writer.add_scalar('train/loss', loss_train, epoch)
        writer.add_scalar('train/rho_s', rho_s, epoch)
        writer.add_scalar('train/rho_p', rho_p, epoch)
        writer.add_scalar('train/mse', rmse, epoch)
        
        # Validation ------------------------------------------------------------
        logging.info('Starting val...')
        logging.info('Running val in epoch {}'.format(epoch + 1))
        
        rho_s, rho_p, rmse = eval_epoch(epoch, net, criterion, val_loader, mean_linsener_id)
        
        print('Val of epoch{}, SRCC:{}, PLCC:{}, MSE:{}'.format(
                epoch + 1, rho_s, rho_p, rmse))
        logging.info('Val done...')
        
        # save best results -----------------------------------------------------
        if rho_s > best_srocc or rho_p > best_plcc or rmse < besst_rmse:
            best_srocc = rho_s
            best_plcc = rho_p
            besst_rmse = rmse
            logging.info('Best weights and model of epoch{}, SRCC:{}, PLCC:{}, MSE:{}'.format(
                epoch + 1, best_srocc, best_plcc, besst_rmse))
            
            torch.save(net, os.path.join(model_path, 'epoch_{}.pth'.format(epoch + 1)))
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

        # Test ------------------------------------------------------------
        logging.info('========================================= [ Running testing epoch {} ]==============================================='.format(epoch + 1))
        
        test(df_val, epoch, net, criterion, args, mean_linsener_id)
        
        logging
        
        logging.info('================================================= [ Test done ] =====================================================\n')
        
        
        
        
        
        

    