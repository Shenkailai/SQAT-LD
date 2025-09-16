"""
SQAT-LD主程序（重构版）
语音质量评估模型训练入口
使用重构后的模块化架构
"""
import argparse
import os
import platform
import time

from config import Config
from utils import setup_seed, get_device, load_and_split_data, create_dataloader, initialize_model, setup_training_components
from training import train


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SQAT-LD 语音质量评估训练（重构版）')
    parser.add_argument('--yaml', type=str, default='./configs/config.yaml',
                       help='YAML配置文件路径')
    args = parser.parse_args()
    
    try:
        # 加载配置
        print(f'[Info] 加载配置文件: {args.yaml}')
        config = Config.from_yaml(args.yaml)
        
        # 设置随机种子
        setup_seed(config.seed)
        print(f'[Info] 随机种子: {config.seed}')
        
        # 获取设备
        device = get_device(config)
        
        # 加载并分割数据
        df_train, df_val, df_test = load_and_split_data(config)
        
        # 创建数据加载器
        dl_train, ds_train = create_dataloader(
            df_train, config, config.train_norm_mean, config.train_norm_std, shuffle=True
        )
        dl_val, ds_val = create_dataloader(
            df_val, config, config.val_norm_mean, config.val_norm_std, shuffle=False
        )
        
        # 获取评委信息
        num_judges = ds_train.num_judges
        mean_listener_id = ds_train.id_dict.get('mean_listener')
        
        # 更新配置
        config.num_judges = num_judges
        config.mean_listener_id = mean_listener_id
        
        print(f'[Info] 评委数量: {num_judges}')
        print(f'[Info] 平均听众ID: {mean_listener_id}')
        
        # 初始化模型
        model = initialize_model(config, device)
        
        # 设置训练组件
        optimizer, scheduler, criterion = setup_training_components(model, config, ds_train)
        
        # 开始训练
        print('[Info] 开始训练...')
        train(
            model, criterion, optimizer, scheduler, 
            dl_train, dl_val, df_test, config.to_dict(), mean_listener_id
        )
        
    except Exception as e:
        print(f'[Error] 训练过程中出现错误: {e}')
        raise


if __name__ == "__main__":
    print(f"[Info] 进程ID: {os.getpid()}, 运行在: {platform.uname()[1]}, 开始时间: {time.asctime()}")
    main()
