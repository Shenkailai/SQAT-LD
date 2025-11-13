"""
训练器模块
包含模型训练相关功能
"""
import logging
import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any, Tuple

from .metrics import compute_metrics
from .evaluator import eval_epoch, test_on_databases


def train_epoch(epoch: int, net: torch.nn.Module, criterion, optimizer, scheduler, 
                train_loader, args: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    训练单个epoch
    
    Args:
        epoch: 当前轮次
        net: 模型
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        train_loader: 训练数据加载器
        args: 配置参数
        
    Returns:
        (loss, SRCC, PLCC, RMSE) 训练指标
        
    Raises:
        ValueError: 如果参数无效
    """
    # 输入验证
    if epoch < 0:
        raise ValueError(f"epoch必须为非负整数，得到: {epoch}")
    
    if len(train_loader) == 0:
        raise ValueError("训练数据加载器为空")
    
    net.train()
    losses = []
    predictions, labels = [], []

    for batch_data in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}"):
        batch_x, batch_y, batch_mos, batch_judge_id, index = batch_data
        
        # 数据转移到GPU
        x_d = batch_x.cuda()
        labels_tensor = batch_y.type(torch.FloatTensor).cuda()
        batch_mos = batch_mos.type(torch.FloatTensor).cuda()
        batch_judge_id = batch_judge_id.type(torch.LongTensor).cuda()
        
        # 前向传播
        pred_d, judge_mos = net(x_d, batch_judge_id)

        # 计算损失
        optimizer.zero_grad()
        if args['use_biasloss']:
            loss1 = criterion.get_loss(yb=labels_tensor, yb_hat=pred_d, idx=index)
            loss2 = criterion.get_loss(yb=batch_mos, yb_hat=judge_mos, idx=index)
        else:
            loss1 = criterion(pred_d, labels_tensor)
            loss2 = criterion(judge_mos, batch_mos)
        
        loss = loss1 + loss2
        losses.append(loss.item())

        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 保存预测结果
        predictions.extend(pred_d.cpu().detach().numpy())
        labels.extend(labels_tensor.cpu().detach().numpy())

    # 计算指标
    avg_loss = np.mean(losses)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    rho_s, rho_p, rmse = compute_metrics(predictions, labels)
    
    # 更新bias（如果使用）
    if args['use_biasloss']:
        criterion.update_bias(np.squeeze(labels), np.squeeze(predictions))
    
    logging.info(f'训练 epoch {epoch+1}: loss={avg_loss:.4f}, SRCC={rho_s:.4f}, PLCC={rho_p:.4f}, RMSE={rmse:.4f}')
    
    return avg_loss, rho_s, rho_p, rmse


def _create_model_tag(args: Dict[str, Any]) -> str:
    """
    创建模型标签用于标识训练实验
    
    Args:
        args: 配置参数
        
    Returns:
        模型标签字符串
    """
    components = [
        f"bs_{args['batch_size']}",
        f"seed_{args['seed']}",
        args['loss_type'],
        args['att_method'],
        args['apply_att_method']
    ]
    
    # 添加备注（如果存在）
    if args.get('comment'):
        components.append(args['comment'])
    
    return "_".join(components)


def _ensure_directories(base_path: str, model_tag: str) -> Dict[str, str]:
    """
    创建必要的目录结构
    
    Args:
        base_path: 基础输出路径
        model_tag: 模型标签
        
    Returns:
        包含各类路径的字典
    """
    paths = {
        'log': os.path.join(base_path, 'logs'),
        'tensorboard': os.path.join(base_path, 'tensorboard'),
        'model': os.path.join(base_path, 'models', model_tag)
    }
    
    for path_type, path in paths.items():
        os.makedirs(path, exist_ok=True)
        print(f'[Info] {path_type}目录: {path}')
    
    return paths


def _setup_logging(log_path: str, model_tag: str, args: Dict[str, Any]) -> None:
    """
    配置日志系统
    
    Args:
        log_path: 日志目录路径
        model_tag: 模型标签
        args: 配置参数
    """
    log_file = os.path.join(log_path, f'{model_tag}.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='w'  # 覆盖模式，每次训练生成新日志
    )
    
    # 记录关键配置信息
    logging.info("=" * 80)
    logging.info(f"训练实验标签: {model_tag}")
    logging.info(f"评委数量: {args.get('num_judges', 'N/A')}")
    logging.info(f"平均听众ID: {args.get('mean_listener_id', 'N/A')}")
    logging.info(f"批次大小: {args['batch_size']}")
    logging.info(f"学习率: {args.get('tr_lr', 'N/A')}")
    logging.info(f"损失函数: {args['loss_type']}")
    logging.info("=" * 80)


def setup_directories_and_logging(args: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    设置训练所需的目录和日志
    
    Args:
        args: 配置参数
        
    Returns:
        (model_path, tensorboard_path, model_tag) 路径信息
    """
    # 创建模型标签
    model_tag = _create_model_tag(args)
    
    # 创建目录结构
    base_path = os.path.join(args['output_dir'], args['dataset'])
    paths = _ensure_directories(base_path, model_tag)
    
    # 配置日志
    _setup_logging(paths['log'], model_tag, args)
    
    return paths['model'], paths['tensorboard'], model_tag


def train(net: torch.nn.Module, criterion, optimizer, scheduler, train_loader, 
         val_loader, df_val, args: Dict[str, Any], mean_listener_id: int) -> None:
    """
    主训练函数
    
    Args:
        net: 模型
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        df_val: 验证数据框
        args: 配置参数
        mean_listener_id: 平均听众ID
    """
    # 设置目录和日志
    model_path, tensorboard_path, model_tag = setup_directories_and_logging(args)
    writer = SummaryWriter(tensorboard_path)
    
    # 初始化最佳结果跟踪
    best_metrics = {
        'srocc': 0.0,
        'plcc': 0.0, 
        'rmse': float('inf')
    }
    
    # 开始训练
    n_epochs = args.get('n_epochs', args.get('n-epochs', 100))
    print(f'开始训练，共 {n_epochs} 个epoch')
    
    for epoch in range(n_epochs):
        start_time = time.time()
        logging.info(f'=== 训练 Epoch {epoch + 1} ===')
        
        # 训练阶段
        loss_train, train_srocc, train_plcc, train_rmse = train_epoch(
            epoch, net, criterion, optimizer, scheduler, train_loader, args
        )
        
        # 记录训练指标
        writer.add_scalar('train/loss', loss_train, epoch)
        writer.add_scalar('train/srocc', train_srocc, epoch)
        writer.add_scalar('train/plcc', train_plcc, epoch)
        writer.add_scalar('train/rmse', train_rmse, epoch)
        
        # 验证阶段
        logging.info(f'开始验证 Epoch {epoch + 1}')
        val_srocc, val_plcc, val_rmse = eval_epoch(epoch, net, criterion, val_loader, mean_listener_id)
        
        print(f'验证 Epoch {epoch + 1}: SRCC={val_srocc:.4f}, PLCC={val_plcc:.4f}, RMSE={val_rmse:.4f}')
        
        # 记录验证指标
        writer.add_scalar('val/srocc', val_srocc, epoch)
        writer.add_scalar('val/plcc', val_plcc, epoch)
        writer.add_scalar('val/rmse', val_rmse, epoch)
        
        # 保存最佳模型
        is_best = (val_srocc > best_metrics['srocc'] or 
                  val_plcc > best_metrics['plcc'] or 
                  val_rmse < best_metrics['rmse'])
        
        if is_best:
            best_metrics.update({
                'srocc': val_srocc,
                'plcc': val_plcc,
                'rmse': val_rmse
            })
            
            model_file = os.path.join(model_path, f'best_model_epoch_{epoch + 1}.pth')
            torch.save(net, model_file)
            logging.info(f'保存最佳模型 Epoch {epoch + 1}: SRCC={val_srocc:.4f}, PLCC={val_plcc:.4f}, RMSE={val_rmse:.4f}')
        
        epoch_time = (time.time() - start_time) / 60
        logging.info(f'Epoch {epoch + 1} 完成，用时: {epoch_time:.2f}分钟')

        # 测试阶段
        logging.info(f'=== 测试 Epoch {epoch + 1} ===')
        test_on_databases(df_val, epoch, net, criterion, args, mean_listener_id)
        logging.info(f'=== 测试完成 ===\n')
    
    writer.close()
    print(f'训练完成！最佳结果: SRCC={best_metrics["srocc"]:.4f}, PLCC={best_metrics["plcc"]:.4f}, RMSE={best_metrics["rmse"]:.4f}')
