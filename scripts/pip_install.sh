#!/bin/bash

# Complete environment setup process

# Step 0: Ensure you create a Conda environment 
# and Activate the environment
# conda activate idol

# Step 1: Install Pytorch with CUDA:
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
--index-url https://download.pytorch.org/whl/cu118

# Step 2: Use pip to install additional dependencies
pip_packages=(
    "absl-py==2.1.0"
    "accelerate==0.29.1"
    "addict==2.4.0"
    "albumentations==1.4.17"
    "bitsandbytes"
    "deepspeed==0.15.1"
    "diffusers==0.20.2"
    "einops==0.8.0"
    "fastapi==0.111.0"
    "gradio==3.41.2"
    "matplotlib==3.8.4"
    "numpy==1.26.3"
    "opencv-python==4.9.0.80"
    "pandas==2.2.2"
    "pillow==10.3.0"
    "scikit-image==0.23.2"
    "scipy==1.13.0"
    "timm==0.9.16"
    "transformers==4.40.1"
    "pytorch-lightning==2.3.1"
    "omegaconf==2.3.0"
    "av"
    "webdataset"
    "omegaconf"
    "rembg==2.0.57"
    "tensorboard"
)

# Install pip packages in bulk
for package in "${pip_packages[@]}"
do
    pip install "$package"
done


# Create submodule directory if it doesn't exist
mkdir -p submodule
cd submodule

# Step 3: Install PyTorch3D
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout v0.7.7  
pip install -e .
cd ..

# Step 4: Install Simple-KNN
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git 
cd simple-knn
pip install -e .
cd ..

# Step 5: Install Gaussian Splatting
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/submodules/diff-gaussian-rasterization
python setup.py develop
cd ../../..

# Step 6: Install Sapiens
git clone https://github.com/facebookresearch/sapiens
cd sapiens/engine
pip install -e .
cd ../pretrain
pip install -e .
cd ../../..

# Step 7: Install deformation module
python setup.py develop

echo "idol environment setup completed!"
