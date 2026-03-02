#!/bin/bash
set -euo pipefail

# Colab inference-only setup (GPU runtime required).

python3 -m pip install --upgrade pip setuptools wheel ninja

# Tested base stack for extension builds.
python3 -m pip install \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

python3 -m pip install \
  pytorch-lightning==2.3.1 \
  omegaconf==2.3.0 \
  einops==0.8.0 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  imageio \
  pillow==10.3.0 \
  rembg==2.0.57 \
  opencv-python-headless==4.9.0.80 \
  trimesh==4.4.9 \
  pygltflib==1.16.2 \
  open3d==0.18.0 \
  av \
  tqdm

mkdir -p submodule
pushd submodule >/dev/null

if [ ! -d pytorch3d ]; then
  git clone https://github.com/facebookresearch/pytorch3d.git
fi
pushd pytorch3d >/dev/null
git fetch --all --tags
git checkout v0.7.7
python3 -m pip install -e .
popd >/dev/null

if [ ! -d simple-knn ]; then
  git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
fi
pushd simple-knn >/dev/null
python3 -m pip install -e .
popd >/dev/null

if [ ! -d gaussian-splatting ]; then
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
fi
pushd gaussian-splatting/submodules/diff-gaussian-rasterization >/dev/null
python3 setup.py develop
popd >/dev/null

popd >/dev/null

# Build local Fast-SNARF CUDA extension modules.
python3 setup.py develop

echo "Colab inference environment setup completed."
