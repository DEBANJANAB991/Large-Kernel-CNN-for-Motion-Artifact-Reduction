![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![License](https://img.shields.io/badge/license-MIT-green)
# Large Kernel CNN for Motion Artifact Reduction in CT

## Overview
This repository contains the implementation used in the Master's thesis:
"Large Kernel CNN for Motion Artifact Reduction in CT".

The project investigates whether super large-kernel CNN architectures can improve artifact removal compared to traditional architectures such as U-Net.

## Key Contributions
- Making Sinograms from Volume Domain
- Motion artifact simulation pipeline for CT sinograms
- Large Kernel CNN architecture for artifact reduction
- Baseline comparison (U-Net, SwinIR, Restormer, RepLKNet)
- Evaluation using PSNR, SSIM and FLOPs

## Dataset

The experiments use the CQ500 head CT dataset.

Download:
https://www.kaggle.com/datasets/crawford/qureai-headct

Pipeline:

1. Convert DICOM CT scans to sinograms using diffct.
2. Introduce simulated motion artifacts in the sinogram domain.
3. Train CNN models to reconstruct artifact-free images.

Dataset structure:

dataset/
    clean/
    artifact/
## Installation

Clone repository

git clone https://github.com/DEBANJANAB991/Large-Kernel-CNN-for-Motion-Artifact-Reduction

Create environment

conda create -n artifactcnn python=3.10
conda activate artifactcnn

Install dependencies

pip install -r requirements.txt

## Training

python train.py --config configs/train_config.yaml

## Inference

python inference.py \
    --model checkpoints/best_model.pth \
    --input data/test \
    --output results/

## Evaluation

python run_inferences.py

Metrics computed:
- PSNR
- SSIM
- FLOPs

## Results

| Model | PSNR ↑ | SSIM ↑ |
|------|------|------|
| UNet | 37.11 | 0.9815 |
| RepLKNet | -12.42 | 0.9034 |
| MR-LKV | 39.97 | 0.994 |
| SwinIR | 40.55 | 0.9916 |
| Restormer | 11.22 | 0.0228 |


## Hardware
Experiments were executed on an HPC cluster using an NVIDIA Tesla V100 GPU with 32GB VRAM.

## Configuration

All training and experiment settings are defined in:

config.py

This file contains parameters such as:

- model architecture
- learning rate
- batch size
- number of epochs
- dataset paths
- image resolution

## Citation
If you use this code please cite:

Bhattacharjya, Debanjona.  
Large Kernel CNN for Motion Artifact Reduction in CT Imaging.  
Master's Thesis, 2026.