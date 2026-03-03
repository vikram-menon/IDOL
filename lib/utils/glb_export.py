import os
from typing import Dict, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_first_pose(
    xyzs, rgbs, sigmas
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_np = _to_numpy(xyzs)
    rgb_np = _to_numpy(rgbs)
    sig_np = _to_numpy(sigmas)

    # xyz can be [B, N, P, 3], [B, N, 3], or [N, 3].
    if xyz_np.ndim == 4:
        xyz_np = xyz_np[0, :, 0, :]
    elif xyz_np.ndim == 3:
        xyz_np = xyz_np[0]
    elif xyz_np.ndim != 2:
        raise ValueError(f"Unexpected xyz shape: {xyz_np.shape}")

    # rgbs can be [B, N, 3] or [N, 3].
    if rgb_np.ndim == 3:
        rgb_np = rgb_np[0]
    elif rgb_np.ndim != 2:
        raise ValueError(f"Unexpected rgb shape: {rgb_np.shape}")

    # sigmas can be [B, N, 1], [N, 1], [B, N], or [N].
    if sig_np.ndim == 3:
        sig_np = sig_np[0, :, 0]
    elif sig_np.ndim == 2:
        if sig_np.shape[0] == 1:
            sig_np = sig_np[0]
        elif sig_np.shape[1] == 1:
            sig_np = sig_np[:, 0]
        else:
            sig_np = sig_np[0]
    elif sig_np.ndim != 1:
        raise ValueError(f"Unexpected sigma shape: {sig_np.shape}")

    return xyz_np.astype(np.float32), rgb_np.astype(np.float32), sig_np.astype(
        np.float32
    )


def export_avatar_glb(
    xyzs,
    rgbs,
    sigmas,
    output_path: str,
    mesh_depth: int = 10,
    max_points: int = 0,
    sigma_percentile: float = 5.0,
    density_quantile: float = 0.0,
    random_seed: int = 42,
) -> Dict[str, int]:
    """Export a static, vertex-colored GLB from IDOL Gaussian points."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for GLB export. Install with `pip install open3d`."
        ) from exc

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    points, colors, sigma = _select_first_pose(xyzs, rgbs, sigmas)

    finite_mask = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(sigma)
    )
    points = points[finite_mask]
    colors = colors[finite_mask]
    sigma = sigma[finite_mask]

    if points.shape[0] == 0:
        raise ValueError("No valid points available for GLB export.")

    sigma_percentile = float(np.clip(sigma_percentile, 0.0, 100.0))
    sigma_threshold = np.percentile(sigma, sigma_percentile)
    keep = sigma >= sigma_threshold
    points = points[keep]
    colors = colors[keep]

    if points.shape[0] == 0:
        raise ValueError("No points left after sigma filtering.")

    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    normal_radius = max(bbox_diag * 0.01, 1e-4)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(mesh_depth)
    )
    densities = np.asarray(densities)
    if densities.size > 0 and density_quantile > 0.0:
        density_quantile = float(np.clip(density_quantile, 0.0, 0.5))
        cutoff = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < cutoff)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("Poisson reconstruction produced an empty mesh.")

    nn = cKDTree(points)
    _, nn_idx = nn.query(vertices, k=1)
    vertex_colors = np.clip(colors[nn_idx], 0.0, 1.0)
    vertex_colors = (vertex_colors * 255.0).astype(np.uint8)

    mesh_tm = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    mesh_tm.export(output_path)

    return {
        "points_after_filter": int(points.shape[0]),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
    }
