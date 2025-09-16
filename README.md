# SQAT-LD: Speech Quality Assessment Transformer Utilizing Listener Dependent Modeling for Zero-Shot Out-of-Domain MOS Prediction

[![paper](https://img.shields.io/badge/IEEE-Paper-green.svg)](https://ieeexplore.ieee.org/document/10389681)
[![python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)

This is the official PyTorch implementation of SQAT-LD: Speech Quality Assessment Transformer Utilizing Listener Dependent Modeling for Zero-Shot Out-of-Domain MOS Prediction. :fire::fire::fire: We won first place in **VoiceMOS Challenge 2023 Track 2**!

## 🎯 Project Introduction

<p align="center"><img src="./poster/poster-asru.png" alt="SQAT-LD Architecture" width="500"/></p>

SQAT-LD is an advanced speech quality assessment model that leverages self-supervised learning SSAST backbone networks and listener-dependent modeling techniques to achieve excellent zero-shot out-of-domain MOS prediction performance.

### ✨ Key Features

- 🏆 **Award-winning Model**: 1st place in VoiceMOS Challenge 2023 Track 2
- 🔄 **Zero-shot Learning**: High-quality MOS prediction without target domain data
- 🎧 **Listener Modeling**: Advanced modeling approach considering individual listener differences
- 🧠 **Self-supervised Learning**: Powerful feature extraction based on SSAST pre-trained models
- 📦 **Modular Design**: Clear code architecture that's easy to understand and extend

## 🏗️ Code Architecture

This project adopts a modular design with clear code structure for easy maintenance and extension:

```
SQAT-LD/
├── config.py              # Unified configuration management
├── main.py                # Main program entry
├── get_norm_stats.py      # Data normalization statistics
│
├── configs/               # Configuration files
│   └── config.yaml       # Main configuration file
│
├── data/                  # Data processing module
│   ├── __init__.py
│   └── dataset.py        # Speech quality dataset
│
├── loss/                  # Loss function module
│   ├── __init__.py
│   ├── base_loss.py      # SQA loss functions
│   ├── norm_losses.py    # Normalization loss functions
│   └── bias_loss.py      # Bias loss functions
│
├── models/               # Model architecture
│   ├── __init__.py
│   ├── sqat_ld.py       # SQAT-LD main model
│   ├── align.py         # Attention alignment mechanisms
│   └── ssast_models.py  # SSAST backbone network
│
├── training/             # Training module
│   ├── __init__.py
│   ├── trainer.py       # Trainer
│   ├── evaluator.py     # Evaluator
│   └── metrics.py       # Evaluation metrics
│
└── utils/                # Utility functions
    ├── __init__.py
    ├── setup.py         # Environment setup
    ├── data_utils.py    # Data processing utilities
    └── model_utils.py   # Model utilities
```

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA (recommended)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn
pip install tensorboardX tqdm einops
pip install pyyaml
```

### Prepare SSAST Pre-trained Models

| Model Name | Data | Pretrain fshape | Pretrain tshape | #Masked Patches | Model Size | Audio Performance | Speech Performance |
|-----------|------|-----------------|-----------------|----------------|------------|-------------------|-------------------|
| [SSAST-Base-Frame-400](https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1) | AudioSet + Librispeech | 128 | 2 | 400 | Base (89M) | 57.6 | 84.0 |

After downloading, place the model file in an appropriate directory and update the path in the configuration file.

### Prepare Dataset

Download the VoiceMOS Challenge 2023 dataset from the following link:
<a href="https://zenodo.org/record/6572573#.ZCorDy8Rr0o" target="_blank">VoiceMOS 2023 Dataset</a>

### Configuration File

Edit the `configs/config.yaml` file to set data paths, model parameters, etc.:

```yaml
# Data configuration
datapath: "/path/to/your/dataset"
feature_cache_path: "/path/to/cache"

# Model configuration
model_name: "SQAT_LD"
lr: 0.00005
batch_size: 32
n_epochs: 100

# Other configurations...
```

### Run Training

```bash
python main.py --yaml ./configs/config.yaml
```

## 📈 Performance Evaluation

The model is evaluated using the following metrics:

- **SRCC**: Spearman rank correlation coefficient
- **PLCC**: Pearson linear correlation coefficient  
- **RMSE**: Root mean square error

## 🤝 Contributing

Issues and Pull Requests are welcome to improve this project.

## 📄 License

This project follows an open source license. See the LICENSE file for details.

## 📚 Citation

If you use this code, please cite our paper:

```bibtex
@INPROCEEDINGS{10389681,
  author={Shen, Kailai and Yan, Diqun and Dong, Li and Ren, Ying and Wu, Xiaoxun and Hu, Jing},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)}, 
  title={SQAT-LD: SPeech Quality Assessment Transformer Utilizing Listener Dependent Modeling for Zero-Shot Out-of-Domain MOS Prediction}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  keywords={Databases;Conferences;Self-supervised learning;Predictive models;Transformers;Quality assessment;Automatic speech recognition;VoiceMOS Challenge;synthetic speech evaluation;MOS prediction;self-supervised learning;zero-shot},
  doi={10.1109/ASRU57964.2023.10389681}
}
```

## 🙏 Acknowledgments

Thanks to the organizers and all participants of VoiceMOS Challenge 2023. The success of this project is inseparable from the support and contributions of the open source community.

---

<p align="center">
  <strong>🏆 VoiceMOS Challenge 2023 Track 2 Winner 🏆</strong>
</p>