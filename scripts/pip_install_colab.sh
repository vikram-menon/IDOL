#!/bin/bash
set -euo pipefail

# Colab inference-only setup (GPU runtime required).

# Keep setuptools on a version that still ships pkg_resources, which
# pytorch_lightning/lightning_fabric import on startup.
python3 -m pip install --upgrade pip wheel ninja
python3 -m pip install "setuptools==75.2.0"

# open3d 0.18.0 has no Python 3.12 wheels; use 0.19.0 on 3.12+.
OPEN3D_VERSION="0.18.0"
if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  OPEN3D_VERSION="0.19.0"
fi

# Tested base stack for extension builds.
python3 -m pip install \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

python3 -m pip install \
  pytorch-lightning==2.3.1 \
  torchmetrics==1.8.2 \
  omegaconf==2.3.0 \
  einops==0.8.0 \
  timm==0.9.16 \
  transformers==4.40.1 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  imageio \
  pillow==10.3.0 \
  opencv-python-headless==4.9.0.80 \
  trimesh==4.4.9 \
  pygltflib==1.16.2 \
  "open3d==${OPEN3D_VERSION}" \
  av \
  tqdm

# Optional: background-removal dependency chain can be fragile on Python 3.12.
if [ "${INSTALL_REMBG:-0}" = "1" ]; then
  python3 -m pip install rembg==2.0.57
else
  echo "[INFO] Skipping rembg install (set INSTALL_REMBG=1 to enable)."
fi

mkdir -p submodule
pushd submodule >/dev/null

if [ ! -d pytorch3d ]; then
  git clone https://github.com/facebookresearch/pytorch3d.git
fi
pushd pytorch3d >/dev/null
git fetch --all --tags
git checkout v0.7.7
# Editable/develop installs now route through PEP517 on modern setuptools
# and fail for pytorch3d in Colab Python 3.12. Use a regular install.
python3 -m pip install --no-build-isolation .
popd >/dev/null

if [ ! -d simple-knn ]; then
  git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
fi
pushd simple-knn >/dev/null
python3 -m pip install --no-build-isolation .
popd >/dev/null

if [ ! -d gaussian-splatting ]; then
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
fi
pushd gaussian-splatting/submodules/diff-gaussian-rasterization >/dev/null
python3 -m pip install --no-build-isolation .
popd >/dev/null

popd >/dev/null

# Build local Fast-SNARF CUDA extension modules.
python3 -m pip install --no-build-isolation .

python3 scripts/verify_colab_env.py

echo "Colab inference environment setup completed."
