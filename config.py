"""
配置管理模块
集中管理所有训练和模型参数
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Config:
    """配置管理类，集中管理所有训练参数"""
    
    # 数据集配置
    dataset: str = "voicemos2023"
    datapath: str = ""
    csv_file: str = "voicemos2023_mean.csv"
    csv_deg: str = "filename"
    csv_user_ID: str = "user_ID"
    csv_mos_train: str = "mos"
    csv_mean_train: str = "mean_mos"
    csv_mos_val: str = "mos"
    csv_db_train: List[str] = field(default_factory=list)
    csv_db_val: List[str] = field(default_factory=list)
    csv_db_test: List[str] = field(default_factory=list)
    
    # 数据预处理配置
    target_length: int = 1024
    mel_bins: int = 128
    skip_norm: bool = False
    train_norm_mean: float = -7.0234294
    train_norm_std: float = 4.659489
    val_norm_mean: float = -8.346359
    val_norm_std: float = 4.475686
    test_norm_mean: float = -6.3182464
    test_norm_std: float = 4.0547657
    
    # 训练配置
    batch_size: int = 8
    num_workers: int = 16  # 对应yaml中的num-workers
    n_epochs: int = 100
    seed: int = 20
    tr_parallel: bool = False
    tr_lr: float = 1e-5
    tr_wd: float = 1e-5
    
    # 模型配置
    fstride: int = 16
    tstride: int = 16
    fshape: int = 16
    tshape: int = 16
    load_pretrained_mdl_path: str = "pre_models/SSAST-Base-Patch-400.pth"
    model_size: str = "base"
    num_tab: int = 2
    embed_dim: int = 768  # 添加嵌入维度
    
    # 损失函数配置
    loss_type: str = "mae"
    alpha: List[float] = field(default_factory=lambda: [1, 0])
    beta: List[float] = field(default_factory=lambda: [.1, .1, 1])
    p: int = 2
    q: int = 2
    monotonicity_regularization: bool = False
    gamma: float = 0.1
    detach: bool = False
    
    # Bias loss配置
    use_biasloss: bool = False
    tr_bias_mapping: str = "first_order"
    tr_bias_min_r: float = 0.7
    tr_bias_anchor_db: Optional[str] = None
    tr_verbose: int = 2
    
    # 其他配置
    output_dir: str = "./output"
    pretrained_model: Optional[str] = None
    comment: str = ""
    to_memory: bool = False
    to_memory_workers: int = 0
    
    # 注意力机制配置
    hallucinate: bool = False
    att_method: str = "luong"
    apply_att_method: str = "hard"
    
    # 数据库标准化统计
    norm_stats: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: {
        'NISQA_TEST_LIVETALK': {'nisqa': [-9.051971, 3.7531793]},
        'NISQA_TEST_FOR': {'nisqa': [-8.937617, 4.2769117]},
        'NISQA_TEST_P501': {'nisqa': [-9.90131, 4.708985]},
        'NISQA_VAL_LIVE': {'nisqa': [-9.823734, 3.6818407]},
        'NISQA_VAL_SIM': {'nisqa': [-8.027123, 4.3762627]},
        'NISQA_VAL': {'nisqa': [-8.185567, 4.3552947]},
        'tencent_with': {'tencent': [-8.642287, 4.199733]},
        'tencent_without': {'tencent': [-9.084293, 5.4488106]},
        'TMHINTQI_Valid': {'voicemos2023': [-4.865034, 4.1673865]},
        'VoiceMOS2022_mian_Test': {'voicemos2023': [-8.353344, 4.4906945]},
        'VoiceMOS2022_OOD_unlabeled1': {'voicemos2023': [-7.7979817, 4.426462]},
        'VoiceMOS2022_OOD_Test1': {'voicemos2023': [-7.64568, 4.315633]}
    })
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, "r", encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        # 处理参数名映射
        if 'num-workers' in data:
            data['num_workers'] = data.pop('num-workers')
        if 'n-epochs' in data:
            data['n_epochs'] = data.pop('n-epochs')
        
        config = cls(**data)
        config.validate()
        return config
    
    def validate(self) -> None:
        """验证配置参数的有效性"""
        if not self.datapath:
            raise ValueError("datapath不能为空")
        
        if not os.path.exists(self.datapath):
            print(f"[Warning] 数据路径不存在: {self.datapath}")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        
        if self.n_epochs <= 0:
            raise ValueError("n_epochs必须大于0")
        
        if self.tr_lr <= 0:
            raise ValueError("学习率必须大于0")
        
        if not self.csv_db_train:
            raise ValueError("csv_db_train不能为空")
        
        print("[Info] 配置验证通过")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式以兼容现有代码"""
        result = {}
        for key, value in self.__dict__.items():
            result[key] = value
        return result
