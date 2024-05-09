# SQAT-LD: Speech Quality Assessment Transformer Utilizing Listener Dependent Modeling for Zero-Shot Out-of-Domain MOS Prediction

[![paper](https://img.shields.io/badge/IEEE-Paper-green.svg)](https://ieeexplore.ieee.org/document/10389681)

This repository is the official PyTorch implementation of SQAT-LD: Speech Quality Assessment Transformer Utilizing Listener Dependent Modeling for Zero-Shot Out-of-Domain MOS Prediction. :fire::fire::fire: We won first place in the **VoiceMOS Challenge 2023 Track 2**.




## Introduction ##

<p align="center"><img src="./poster/poster-asru.png" alt="Illustration of SQAT-LD." width="500"/></p>


## Getting Started  
### Prepare the SSAST Pretrained-Models

| Model Name                                                                                        | Data  | Pretrain fshape | Pretrain tshape | #Masked   Patches | Model Size  | Avg Audio  Performance | Avg Speech  Performance |
|---------------------------------------------------------------------------------------------------|-------|-----------------|-----------------|-------------------|-------------|------------------------|-------------------------|
| [SSAST-Base-Frame-400](https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1)   | AudioSet + Librispeech | 128             | 2               | 400               | Base (89M)  | 57.6                   | 84.0                    |

## Dataset ##
You can download the dataset in the following  <a href="https://zenodo.org/record/6572573#.ZCorDy8Rr0o" target="_blank">link</a> 




## Citation ##

Please kindly cite our paper, if you find this code is useful.
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
  doi={10.1109/ASRU57964.2023.10389681}}
```


