#!/usr/bin/env python3
import importlib
import sys


def check(mod_name, required=True):
    try:
        importlib.import_module(mod_name)
        print(f"[OK] import {mod_name}")
        return True
    except Exception as exc:
        level = "ERROR" if required else "WARN"
        print(f"[{level}] import {mod_name} failed: {exc}")
        return not required


def main():
    # Keep stale Debian pkg_resources from shadowing pip-managed setuptools.
    dist_packages = "/usr/lib/python3/dist-packages"
    if dist_packages in sys.path:
        sys.path.remove(dist_packages)
    sys.modules.pop("pkg_resources", None)

    required_modules = [
        "torch",
        "torchvision",
        "pytorch_lightning",
        "omegaconf",
        "einops",
        "timm",
        "transformers",
        "trimesh",
        "pytorch3d",
        "simple_knn._C",
        "diff_gaussian_rasterization",
        "fuse_cuda",
        "filter_cuda",
        "precompute_cuda",
    ]
    optional_modules = [
        "open3d",
        "rembg",
    ]

    ok = True
    for mod in required_modules:
        ok = check(mod, required=True) and ok
    for mod in optional_modules:
        check(mod, required=False)

    if not ok:
        raise SystemExit(1)
    print("[DONE] Core import checks passed.")


if __name__ == "__main__":
    main()
