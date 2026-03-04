#!/usr/bin/env python3
"""One-command automation: run IDOL inference -> GLB -> headless Blender auto-rig FBX.

Example:
python scripts/auto_unity_avatar.py \
  --input_image ./examples/person.jpg \
  --config configs/idol_v0.yaml \
  --ckpt work_dirs/ckpt/model.ckpt \
  --output_dir outputs_unity \
  --blender_exe blender
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate Unity-ready avatar export.")
    parser.add_argument("--input_image", required=True, help="Input image path.")
    parser.add_argument("--output_dir", default="outputs_unity", help="Output directory.")
    parser.add_argument("--config", default="configs/idol_v0.yaml", help="IDOL config path.")
    parser.add_argument("--ckpt", default="work_dirs/ckpt/model.ckpt", help="IDOL checkpoint path.")
    parser.add_argument("--smplx_json", default=None, help="Optional SMPL-X file (json/npy).")
    parser.add_argument("--sapiens_ckpt", default=None, help="Optional sapiens torchscript checkpoint.")
    parser.add_argument("--blender_exe", default="blender", help="Blender executable.")
    parser.add_argument(
        "--unity_fbx_name",
        default="avatar_unity.fbx",
        help="Output FBX filename inside output_dir.",
    )
    parser.add_argument(
        "--target_faces",
        type=int,
        default=45000,
        help="Target face count used in GLB and Blender decimation.",
    )
    parser.add_argument("--mesh_depth", type=int, default=9, help="Poisson depth for GLB export.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no_rembg",
        action="store_true",
        help="Disable background removal in preprocessing.",
    )
    parser.add_argument(
        "--extra_infer_args",
        default="",
        help="Extra args forwarded to run_colab_infer.py (quoted string).",
    )
    return parser.parse_args()


def run_cmd(cmd):
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_cmd = [
        sys.executable,
        str(repo_root / "run_colab_infer.py"),
        "--input_image",
        args.input_image,
        "--output_dir",
        str(output_dir),
        "--config",
        args.config,
        "--ckpt",
        args.ckpt,
        "--seed",
        str(args.seed),
        "--glb_mode",
        "watertight",
        "--mesh_depth",
        str(args.mesh_depth),
        "--watertight_target_faces",
        str(args.target_faces),
        "--no_export_npz",
        "--no_export_ply",
    ]

    if args.no_rembg:
        infer_cmd.append("--no_rembg")
    if args.smplx_json:
        infer_cmd.extend(["--smplx_json", args.smplx_json])
    if args.sapiens_ckpt:
        infer_cmd.extend(["--sapiens_ckpt", args.sapiens_ckpt])
    if args.extra_infer_args.strip():
        infer_cmd.extend(shlex.split(args.extra_infer_args))

    run_cmd(infer_cmd)

    input_glb = output_dir / "avatar.glb"
    if not input_glb.exists():
        raise FileNotFoundError(f"Expected GLB missing after inference: {input_glb}")

    output_fbx = output_dir / args.unity_fbx_name
    blender_script = repo_root / "scripts" / "blender_autorig_unity.py"
    blender_cmd = [
        args.blender_exe,
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--input_glb",
        str(input_glb),
        "--output_fbx",
        str(output_fbx),
        "--decimate_target_faces",
        str(args.target_faces),
    ]

    run_cmd(blender_cmd)
    print(f"[DONE] Unity avatar FBX: {output_fbx}")


if __name__ == "__main__":
    main()
