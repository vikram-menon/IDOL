import argparse
import json
import os

import torch
import torchvision
import trimesh
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

from lib.utils.glb_export import export_avatar_glb
from lib.utils.infer_util import (
    add_root_rotate_to_smplx,
    construct_camera,
    load_image,
    load_smplify_json,
    load_smplx_from_json,
    load_smplx_from_npy,
    prepare_camera,
    save_video,
)
from lib.utils.train_util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Colab-friendly single-image IDOL inference with GLB export."
    )
    parser.add_argument("--input_image", type=str, required=True, help="Input image path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_colab",
        help="Output directory for preview and GLB.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/idol_v0.yaml",
        help="Model config path.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="work_dirs/ckpt/model.ckpt",
        help="IDOL checkpoint path.",
    )
    parser.add_argument(
        "--sapiens_ckpt",
        type=str,
        default=None,
        help="Optional override for sapiens torchscript checkpoint.",
    )
    parser.add_argument(
        "--smplx_json",
        type=str,
        default=None,
        help="Optional SMPL-X json/npy path. If omitted, uses default neutral A-pose.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--no_rembg",
        action="store_true",
        help="Skip background removal in preprocessing.",
    )

    parser.add_argument(
        "--export_glb",
        dest="export_glb",
        action="store_true",
        default=True,
        help="Export avatar GLB.",
    )
    parser.add_argument(
        "--no_export_glb",
        dest="export_glb",
        action="store_false",
        help="Disable GLB export.",
    )
    parser.add_argument(
        "--mesh_depth",
        type=int,
        default=8,
        help="Poisson depth for mesh reconstruction.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=120000,
        help="Maximum points after filtering/downsampling for mesh reconstruction.",
    )
    parser.add_argument(
        "--sigma_percentile",
        type=float,
        default=30.0,
        help="Drop points below this sigma percentile before meshing.",
    )

    parser.add_argument(
        "--render_preview",
        action="store_true",
        help="Render turntable preview video and first frame PNG.",
    )
    parser.add_argument(
        "--preview_frames",
        type=int,
        default=60,
        help="Number of preview frames for turntable rendering.",
    )
    return parser.parse_args()


def validate_assets(args):
    required = [args.input_image, args.config, args.ckpt]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    smplx_neutral_path = os.path.join(
        "lib", "models", "deformers", "smplx", "SMPLX", "SMPLX_NEUTRAL.pkl"
    )
    if not os.path.exists(smplx_neutral_path):
        raise FileNotFoundError(
            "Missing SMPL-X model file: "
            f"{smplx_neutral_path}. "
            "Please place your licensed SMPL-X file there."
        )

    if args.smplx_json is not None and not os.path.exists(args.smplx_json):
        raise FileNotFoundError(f"SMPL-X input file not found: {args.smplx_json}")


def load_model(args, device):
    config = OmegaConf.load(args.config)
    if args.sapiens_ckpt:
        config.model.params.encoder.params.model_path = args.sapiens_ckpt

    model = instantiate_from_config(config.model)
    model.encoder = model.encoder.to(torch.bfloat16)
    model = model.__class__.load_from_checkpoint(args.ckpt, **config.model.params)
    model = model.to(device)
    model = model.eval()
    return model


def load_smpl_params(args, model, device) -> torch.Tensor:
    if args.smplx_json is None:
        smpl_params = model.get_default_smplx_params().to(device)
        if smpl_params.ndim == 1:
            smpl_params = smpl_params.unsqueeze(0)
        return smpl_params.to(torch.float32)

    if args.smplx_json.endswith(".npy"):
        smpl_params = load_smplx_from_npy(args.smplx_json, device=device)[:1]
        return smpl_params.to(torch.float32)

    if args.smplx_json.endswith(".json"):
        with open(args.smplx_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "annotations" in data:
            smpl_params = load_smplx_from_json(args.smplx_json, device=device)[:1]
        else:
            _, _, smpl_ref = load_smplify_json(args.smplx_json)
            smpl_params = smpl_ref.reshape(1, -1).to(device)
        return smpl_params.to(torch.float32)

    raise ValueError("Unsupported --smplx_json format. Use .json or .npy.")


@torch.no_grad()
def render_preview(model, code, smpl_params, output_dir: str, preview_frames: int = 60):
    preview_frames = max(1, int(preview_frames))
    H, W = 896, 640
    cam_idx = 0

    K, cam_list = prepare_camera(
        resolution_x=H, resolution_y=W, num_views=preview_frames, stides=1
    )
    cameras = construct_camera(K, cam_list, device=code.device)
    intrics = torch.tensor([K[0, 0], K[1, 1], 256, 256], device=code.device)
    cameras[:, :4] = intrics.reshape(1, -1).repeat(cameras.shape[0], 1)
    cameras = cameras[cam_idx : cam_idx + 1].repeat(preview_frames, 1)
    cameras = cameras[:, None, :]

    smpl_seq = add_root_rotate_to_smplx(
        smpl_params[0].to(torch.float32), frames_num=preview_frames, device=code.device
    )

    output_list = []
    num_imgs_batch = 5
    for i in tqdm(range(0, preview_frames, num_imgs_batch), desc="Rendering preview"):
        bt = min(num_imgs_batch, preview_frames - i)
        code_bt = code.expand(bt, -1, -1, -1)
        cameras_bt = cameras[i : i + bt]
        res_uv = model.decoder._decode_feature(code_bt)
        res_points = model.decoder._sample_feature(res_uv)
        res_def_points = model.decoder.deform_pcd(
            res_points,
            smpl_seq[i : i + bt].to(code_bt.dtype),
            zeros_hands_off=True,
            value=0.02,
        )
        output = model.decoder.forward_render(
            res_def_points, cameras_bt.to(code_bt.dtype), num_imgs=1
        )
        image = output["image"][:, 0].cpu().to(torch.float32)
        output_list.append(image)

    output = torch.concatenate(output_list, 0)
    frames = rearrange(output, "b h w c -> b c h w")

    preview_mp4 = os.path.join(output_dir, "preview.mp4")
    preview_png = os.path.join(output_dir, "preview.png")
    save_video(frames[:, :3, ...].to(torch.float32), preview_mp4)
    torchvision.utils.save_image(frames[0, :3, ...], preview_png)
    return preview_mp4, preview_png


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this inference path.")

    validate_assets(args)
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args, device)

    image = load_image(args.input_image, args.output_dir, no_rembg=args.no_rembg)
    sample = image.unsqueeze(0).to(device)
    ref_path = os.path.join(args.output_dir, "input_preprocessed.jpg")
    torchvision.utils.save_image(sample[0], ref_path)

    with torch.no_grad():
        code = model.forward_image_to_uv(sample, is_training=False).to(torch.float32)

    smpl_params = load_smpl_params(args, model, device)

    if args.export_glb:
        with torch.no_grad():
            xyzs, sigmas, rgbs, _, _, _, _ = model.decoder.extract_pcd(
                code, smpl_params.to(code.dtype), init=False, zeros_hands_off=True
            )

        glb_path = os.path.join(args.output_dir, "avatar.glb")
        stats = export_avatar_glb(
            xyzs=xyzs,
            rgbs=rgbs,
            sigmas=sigmas,
            output_path=glb_path,
            mesh_depth=args.mesh_depth,
            max_points=args.max_points,
            sigma_percentile=args.sigma_percentile,
            random_seed=args.seed,
        )

        # Sanity check that the GLB is loadable.
        _ = trimesh.load(glb_path, force="mesh")
        print(f"[OK] GLB exported: {glb_path}")
        print(f"[INFO] GLB stats: {stats}")

    if args.render_preview:
        preview_mp4, preview_png = render_preview(
            model=model,
            code=code,
            smpl_params=smpl_params,
            output_dir=args.output_dir,
            preview_frames=args.preview_frames,
        )
        print(f"[OK] Preview video: {preview_mp4}")
        print(f"[OK] Preview image: {preview_png}")

    print(f"[DONE] Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
