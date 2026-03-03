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


def _compact_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    keep_vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop masked vertices and reindex faces."""
    if keep_vertices.dtype != bool:
        keep_vertices = keep_vertices.astype(bool)
    if keep_vertices.shape[0] != vertices.shape[0]:
        return vertices, faces

    kept_idx = np.nonzero(keep_vertices)[0]
    if kept_idx.size == 0:
        return vertices, faces

    old_to_new = np.full(vertices.shape[0], -1, dtype=np.int64)
    old_to_new[kept_idx] = np.arange(kept_idx.size, dtype=np.int64)

    face_keep = keep_vertices[faces[:, 0]] & keep_vertices[faces[:, 1]] & keep_vertices[faces[:, 2]]
    new_faces = old_to_new[faces[face_keep]]
    new_vertices = vertices[kept_idx]
    return new_vertices, new_faces


def _select_first_attr(x, channels: int = 3) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != channels:
        raise ValueError(f"Unexpected attribute shape: {arr.shape}, expected [N, {channels}]")
    return arr.astype(np.float32)


def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles (radians) to quaternions in w,x,y,z order."""
    x = euler_xyz[:, 0] * 0.5
    y = euler_xyz[:, 1] * 0.5
    z = euler_xyz[:, 2] * 0.5
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    q = np.stack([qw, qx, qy, qz], axis=1).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    return q


def _prepare_points(
    xyzs,
    rgbs,
    sigmas,
    sigma_percentile: float = 15.0,
    max_points: int = 0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise ValueError("No valid points available for export.")

    sigma_percentile = float(np.clip(sigma_percentile, 0.0, 100.0))
    sigma_threshold = np.percentile(sigma, sigma_percentile)
    keep = sigma >= sigma_threshold
    points = points[keep]
    colors = colors[keep]
    sigma = sigma[keep]

    if points.shape[0] == 0:
        raise ValueError("No points left after sigma filtering.")

    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        sigma = sigma[idx]

    return points, colors, sigma


def export_avatar_ply(
    xyzs,
    rgbs,
    sigmas,
    radius,
    rot,
    output_path: str,
    max_points: int = 0,
    sigma_percentile: float = 5.0,
    random_seed: int = 42,
    gs_max_points: int = 300000,
    write_rgb_companion: bool = True,
) -> Dict[str, int]:
    """Export Gaussian-splat-compatible PLY.

    Writes a GS-compatible binary PLY for splat viewers and (optionally) a
    standard RGB point-cloud companion PLY for generic mesh/point-cloud viewers.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    points, colors, sigma = _select_first_pose(xyzs, rgbs, sigmas)
    radius_np = _select_first_attr(radius, channels=3)
    rot_np = _select_first_attr(rot, channels=3)

    finite_mask = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(sigma)
        & np.isfinite(radius_np).all(axis=1)
        & np.isfinite(rot_np).all(axis=1)
    )
    points = points[finite_mask]
    colors = colors[finite_mask]
    sigma = sigma[finite_mask]
    radius_np = radius_np[finite_mask]
    rot_np = rot_np[finite_mask]

    sigma_percentile = float(np.clip(sigma_percentile, 0.0, 100.0))
    sigma_threshold = np.percentile(sigma, sigma_percentile)
    keep = sigma >= sigma_threshold
    points = points[keep]
    colors = colors[keep]
    sigma = sigma[keep]
    radius_np = radius_np[keep]
    rot_np = rot_np[keep]

    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        sigma = sigma[idx]
        radius_np = radius_np[idx]
        rot_np = rot_np[idx]

    # Keep GS PLY size manageable for web viewers (superspl.at, etc.).
    if gs_max_points > 0 and points.shape[0] > gs_max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(points.shape[0], size=gs_max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        sigma = sigma[idx]
        radius_np = radius_np[idx]
        rot_np = rot_np[idx]

    if points.shape[0] == 0:
        raise ValueError("No points left for PLY export.")

    c0 = 0.28209479177387814  # SH constant used by GS viewers for f_dc
    f_dc = (np.clip(colors, 0.0, 1.0) - 0.5) / c0
    scale = np.log(np.clip(np.abs(radius_np), 1e-4, None))
    quat = _euler_xyz_to_quat_wxyz(rot_np)
    opacity = np.log(np.clip(sigma, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - sigma, 1e-4, 1.0))
    nx = np.zeros((points.shape[0],), dtype=np.float32)
    ny = np.zeros((points.shape[0],), dtype=np.float32)
    nz = np.zeros((points.shape[0],), dtype=np.float32)

    # GS viewers usually expect 45 f_rest coefficients (15 SH bands * 3 channels).
    f_rest = np.zeros((points.shape[0], 45), dtype=np.float32)
    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ] + [f"f_rest_{i}" for i in range(45)] + [
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = [(n, "<f4") for n in names]
    vert = np.empty(points.shape[0], dtype=dtype)
    vert["x"], vert["y"], vert["z"] = points[:, 0], points[:, 1], points[:, 2]
    vert["nx"], vert["ny"], vert["nz"] = nx, ny, nz
    vert["f_dc_0"], vert["f_dc_1"], vert["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(45):
        vert[f"f_rest_{i}"] = f_rest[:, i]
    vert["opacity"] = opacity
    vert["scale_0"], vert["scale_1"], vert["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    vert["rot_0"], vert["rot_1"], vert["rot_2"], vert["rot_3"] = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    with open(output_path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n".encode("ascii"))
        for p in names:
            f.write(f"property float {p}\n".encode("ascii"))
        f.write(b"end_header\n")
        vert.tofile(f)

    if write_rgb_companion:
        rgb_path = output_path.replace(".ply", "_rgb.ply")
        vertex_colors = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        cloud = trimesh.points.PointCloud(vertices=points, colors=vertex_colors)
        cloud.export(rgb_path)

    return {"points_after_filter": int(points.shape[0])}


def export_avatar_glb(
    xyzs,
    rgbs,
    sigmas,
    output_path: str,
    mesh_depth: int = 10,
    max_points: int = 0,
    sigma_percentile: float = 15.0,
    density_quantile: float = 0.01,
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

    points, colors, _ = _prepare_points(
        xyzs=xyzs,
        rgbs=rgbs,
        sigmas=sigmas,
        sigma_percentile=sigma_percentile,
        max_points=max_points,
        random_seed=random_seed,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    # Remove disconnected/background point clusters (e.g., broad image-plane blobs).
    try:
        eps = max(bbox_diag * 0.02, 1e-3)
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=40, print_progress=False))
        valid = labels >= 0
        if valid.any():
            cluster_ids, counts = np.unique(labels[valid], return_counts=True)
            keep_label = cluster_ids[np.argmax(counts)]
            keep_points = labels == keep_label
            if keep_points.sum() > 1000:
                points = points[keep_points]
                colors = colors[keep_points]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
    except Exception:
        # Best-effort cleanup only.
        pass

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
    nn_dist, nn_idx = nn.query(vertices, k=1)

    # Prune mesh regions not supported by nearby points (common source of planar artifacts).
    point_nn = nn.query(points, k=2)[0][:, 1]
    point_spacing = float(np.median(point_nn[np.isfinite(point_nn)]))
    # Relax support distance to avoid over-pruning thin clothing regions.
    max_support_dist = max(point_spacing * 12.0, bbox_diag * 0.004)
    keep_vertices = nn_dist <= max_support_dist
    if keep_vertices.sum() > 1000:
        pruned_vertices, pruned_faces = _compact_mesh(vertices, faces, keep_vertices)
        if pruned_vertices.shape[0] > 1000 and pruned_faces.shape[0] > 1000:
            vertices, faces = pruned_vertices, pruned_faces
            nn_dist, nn_idx = nn.query(vertices, k=1)

    vertex_colors = np.clip(colors[nn_idx], 0.0, 1.0)
    vertex_colors = (vertex_colors * 255.0).astype(np.uint8)

    mesh_tm = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )

    # Keep the largest connected component to suppress stray fragments.
    try:
        parts = mesh_tm.split(only_watertight=False)
        if len(parts) > 1:
            mesh_tm = max(parts, key=lambda m: len(m.faces))
    except Exception:
        pass

    # Lightweight cleanup to reduce tiny holes/degeneracies.
    try:
        mesh_tm.remove_unreferenced_vertices()
        mesh_tm.remove_degenerate_faces()
        trimesh.repair.fill_holes(mesh_tm)
    except Exception:
        pass

    mesh_tm.export(output_path)

    return {
        "points_after_filter": int(points.shape[0]),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
    }
