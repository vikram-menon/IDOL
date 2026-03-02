# IDOL Colab Guide (Image Upload -> Avatar GLB)

This guide runs single-image inference in Google Colab and exports `avatar.glb`.

## 1. Start Colab

- Runtime -> Change runtime type -> **GPU**

## 2. Clone and system packages

```bash
!git clone https://github.com/yiyuzhuang/IDOL.git
%cd IDOL
!apt-get update -y
!apt-get install -y libgl1 libglib2.0-0 ffmpeg ninja-build
```

## 3. Install Python/CUDA dependencies

```bash
!bash scripts/pip_install_colab.sh
```

## 4. Download checkpoints and cache files

```bash
!bash scripts/download_files.sh
```

## 5. Upload licensed SMPL-X model

Upload your licensed `SMPLX_NEUTRAL.pkl` and place it here:

```bash
lib/models/deformers/smplx/SMPLX/SMPLX_NEUTRAL.pkl
```

Example Colab cell:

```python
from google.colab import files
import os, shutil

os.makedirs("lib/models/deformers/smplx/SMPLX", exist_ok=True)
uploaded = files.upload()  # choose SMPLX_NEUTRAL.pkl
src = next(iter(uploaded.keys()))
dst = "lib/models/deformers/smplx/SMPLX/SMPLX_NEUTRAL.pkl"
shutil.move(src, dst)
print("Placed:", dst)
```

## 6. Upload your input image

```python
from google.colab import files
uploaded = files.upload()  # choose your image, e.g. person.jpg
input_image = next(iter(uploaded.keys()))
print("Image:", input_image)
```

## 7. Run inference + GLB export

```python
!python run_colab_infer.py \
  --input_image "{input_image}" \
  --output_dir /content/out \
  --export_glb \
  --render_preview
```

Outputs:
- `/content/out/avatar.glb`
- `/content/out/preview.mp4`
- `/content/out/preview.png`
- `/content/out/input_preprocessed.jpg`

## 8. Download results

```python
from google.colab import files
files.download("/content/out/avatar.glb")
files.download("/content/out/preview.mp4")
```

## Notes

- If GLB reconstruction is slow on a T4 GPU, lower cost:
  - `--mesh_depth 7`
  - `--max_points 80000`
  - `--sigma_percentile 40`
- If you already removed image background externally, add `--no_rembg`.
