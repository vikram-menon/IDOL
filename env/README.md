# Environment Setup Guide

## Prerequisites

- Python 3.10
- CUDA 11.8
- PyTorch 2.3.1

## Installation Steps

### 1. Environment Preparation

First, create and activate a conda environment:
```bash
conda create -n idol python=3.10
conda activate idol
```


Install all dependencies:
```bash
bash scripts/pip_install.sh
```

### 2. Download Required Models

Before proceeding, please register on:
- [SMPL-X website](https://smpl-x.is.tue.mpg.de/)
- [FLAME website](https://flame.is.tue.mpg.de/)

Then download the template files:
```bash
bash scripts/fetch_template.sh
```

### 3. Download Pretrained Models and caches with:
```bash
bash scripts/download_files.sh      # download pretrained models
```

Or mannually download the following models from HuggingFace:
- [IDOL Model Checkpoint](https://huggingface.co/yiyuzhuang/IDOL/blob/main/model.ckpt)
- [Sapiens Pretrained Model](https://huggingface.co/yiyuzhuang/IDOL/blob/main/sapiens_1b_epoch_173_torchscript.pt2)

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8 support
- **GPU RAM**: Recommended 24GB+
- **Storage**: At least 15GB free space

## Common Issues & Solutions
**Issue**: 
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
when importing OpenCV (`import cv2`)

**Solution**:
```bash
# For Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx
```

### 2. Gaussian Splatting Antialiasing Issue
**Issue**: Error related to `antialiasing=True` setting in `GaussianRasterizationSettings`

**Solution**:
This issue arises due to updates in the Gaussian Splatting repository. Reinstalling the module from the GitHub repository with the latest version should resolve the problem.