"""
评估器模块
包含模型评估和测试相关功能
"""
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple
from data import SpeechQualityDataset
from .metrics import compute_metrics


def eval_epoch(epoch: int, net: torch.nn.Module, criterion, test_loader, 
               mean_listener_id: int) -> Tuple[float, float, float]:
    """
    评估单个epoch
    
    Args:
        epoch: 当前轮次
        net: 模型
        criterion: 损失函数
        test_loader: 测试数据加载器
        mean_listener_id: 平均听众ID
        
    Returns:
        (SRCC, PLCC, RMSE) 评估指标
    """
    net.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc=f"评估 Epoch {epoch+1}"):
            batch_x, batch_y, batch_mos, batch_judge_id, index = batch_data
            
            # 数据转移到GPU
            x_d = batch_x.cuda()
            labels_tensor = batch_y.type(torch.FloatTensor).cuda()
            batch_judge_id = batch_judge_id.type(torch.LongTensor).cuda()
            
            # 使用平均听众ID
            batch_judge_id[:] = mean_listener_id
            pred, judge_mos = net(x_d, batch_judge_id)

            # 保存预测结果
            predictions.extend(pred.cpu().numpy())
            labels.extend(labels_tensor.cpu().numpy())

    # 计算指标
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    return compute_metrics(predictions, labels)


def test_on_databases(df_val, epoch: int, net: torch.nn.Module, criterion, 
                     args: Dict[str, Any], mean_listener_id: int) -> None:
    """
    在不同数据库上测试模型性能
    
    Args:
        df_val: 验证数据框
        epoch: 当前轮次
        net: 模型
        criterion: 损失函数
        args: 配置参数
        mean_listener_id: 平均听众ID
    """
    norm_stats = args.get('norm_stats', {})
    dataset_name = args['dataset']
    
    for db_name in df_val.db.astype("category").cat.categories:
        # 从配置获取归一化统计信息
        if db_name in norm_stats and dataset_name in norm_stats[db_name]:
            norm_mean, norm_std = norm_stats[db_name][dataset_name]
        else:
            # 使用默认值作为后备
            logging.warning(f"未找到数据库 {db_name} 的归一化统计信息，使用默认值")
            norm_mean, norm_std = args['val_norm_mean'], args['val_norm_std']
        
        # 更新参数
        args.update({
            'norm_mean': norm_mean,
            'norm_std': norm_std
        })
        
        # 过滤数据
        df_db = df_val.loc[df_val.db == db_name]
        if len(df_db) == 0:
            logging.warning(f"数据库 {db_name} 中没有数据，跳过")
            continue
        
        # 创建数据集和加载器
        ds_val = SpeechQualityDataset(
            df_db, args, norm_mean=norm_mean, norm_std=norm_std
        )
        dl_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=args['batch_size'],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=args.get('num_workers', args.get('num-workers', 4))
        )
       
        # 评估
        logging.info(f'在数据库 {db_name} 上测试...')
        rho_s, rho_p, rmse = eval_epoch(epoch, net, criterion, dl_val, mean_listener_id)
        
        print(f'评估 {db_name}: SRCC={rho_s:.4f}, PLCC={rho_p:.4f}, RMSE={rmse:.4f}')
        logging.info(f'评估 {db_name} - SRCC={rho_s:.4f}, PLCC={rho_p:.4f}, RMSE={rmse:.4f}')
