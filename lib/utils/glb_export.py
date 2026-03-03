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

    if xyz_np.ndim == 4:
        xyz_np = xyz_np[0, :, 0, :]
    elif xyz_np.ndim == 3:
        xyz_np = xyz_np[0]
    elif xyz_np.ndim != 2:
        raise ValueError(f"Unexpected xyz shape: {xyz_np.shape}")

    if rgb_np.ndim == 3:
        rgb_np = rgb_np[0]
    elif rgb_np.ndim != 2:
        raise ValueError(f"Unexpected rgb shape: {rgb_np.shape}")

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

    return xyz_np.astype(np.float32), rgb_np.astype(np.float32), sig_np.astype(np.float32)


def _select_first_attr(x, channels: int = 3) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != channels:
        raise ValueError(f"Unexpected attribute shape: {arr.shape}, expected [N, {channels}]")
    return arr.astype(np.float32)


def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
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


def _compact_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    keep_vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _compute_bounds(points: np.ndarray, margin_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = np.maximum(pmax - pmin, 1e-3)
    margin = extent * margin_ratio
    return pmin - margin, pmax + margin


def _points_to_indices(points: np.ndarray, pmin: np.ndarray, pmax: np.ndarray, resolution: int) -> np.ndarray:
    scale = (resolution - 1) / np.maximum(pmax - pmin, 1e-8)
    idx = ((points - pmin) * scale[None, :]).astype(np.int32)
    return np.clip(idx, 0, resolution - 1)


def _mesh_boundary_edges(mesh: trimesh.Trimesh) -> int:
    if mesh.faces.shape[0] == 0:
        return 0
    edges = np.sort(np.asarray(mesh.edges), axis=1)
    packed = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * 2)))
    _, counts = np.unique(packed, return_counts=True)
    return int(np.sum(counts == 1))


def _mesh_quality_metrics(mesh: trimesh.Trimesh) -> Dict[str, float]:
    areas = mesh.area_faces if mesh.faces.shape[0] > 0 else np.zeros((0,), dtype=np.float64)
    return {
        "watertight": bool(mesh.is_watertight),
        "boundary_edges": _mesh_boundary_edges(mesh),
        "components": int(len(mesh.split(only_watertight=False))),
        "min_face_area": float(areas.min()) if areas.size > 0 else 0.0,
        "mean_face_area": float(areas.mean()) if areas.size > 0 else 0.0,
    }


def _keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    return max(parts, key=lambda m: len(m.faces))


def _colorize_mesh(mesh: trimesh.Trimesh, points: np.ndarray, colors: np.ndarray) -> trimesh.Trimesh:
    if mesh.vertices.shape[0] == 0:
        return mesh
    nn = cKDTree(points)
    _, nn_idx = nn.query(mesh.vertices, k=1)
    vertex_colors = np.clip(colors[nn_idx], 0.0, 1.0)
    vertex_colors = (vertex_colors * 255.0).astype(np.uint8)
    mesh.visual.vertex_colors = vertex_colors
    return mesh


def _finalize_for_glb(
    mesh: trimesh.Trimesh,
    smooth_iters: int = 0,
    target_faces: int = 0,
) -> trimesh.Trimesh:
    mesh = _keep_largest_component(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except Exception:
        pass
    if smooth_iters > 0 and mesh.faces.shape[0] > 0:
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=int(smooth_iters))
        except Exception:
            pass
    if target_faces > 0 and mesh.faces.shape[0] > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(int(target_faces))
        except Exception:
            pass
    mesh.remove_unreferenced_vertices()
    return mesh


def _candidate_mesh_poisson(
    points: np.ndarray,
    colors: np.ndarray,
    mesh_depth: int,
    density_quantile: float,
) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for GLB export. Install with `pip install open3d`.") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    points_work = points
    colors_work = colors
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    try:
        eps = max(bbox_diag * 0.02, 1e-3)
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=40, print_progress=False))
        valid = labels >= 0
        if valid.any():
            cluster_ids, counts = np.unique(labels[valid], return_counts=True)
            keep_label = cluster_ids[np.argmax(counts)]
            keep_points = labels == keep_label
            if keep_points.sum() > 1000:
                points_work = points_work[keep_points]
                colors_work = colors_work[keep_points]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_work)
    except Exception:
        pass

    bbox_diag = float(np.linalg.norm(points_work.max(axis=0) - points_work.min(axis=0)))
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

    nn = cKDTree(points_work)
    nn_dist, _ = nn.query(vertices, k=1)
    point_nn = nn.query(points_work, k=2)[0][:, 1]
    point_spacing = float(np.median(point_nn[np.isfinite(point_nn)]))
    max_support_dist = max(point_spacing * 12.0, bbox_diag * 0.004)
    keep_vertices = nn_dist <= max_support_dist
    if keep_vertices.sum() > 1000:
        pruned_vertices, pruned_faces = _compact_mesh(vertices, faces, keep_vertices)
        if pruned_vertices.shape[0] > 1000 and pruned_faces.shape[0] > 1000:
            vertices, faces = pruned_vertices, pruned_faces

    mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_tm = _finalize_for_glb(mesh_tm)
    mesh_tm = _colorize_mesh(mesh_tm, points_work, colors_work)
    return mesh_tm, points_work, colors_work


def _largest_component_mask(binary_volume: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, ncomp = ndimage.label(binary_volume)
    if ncomp <= 1:
        return binary_volume
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = np.argmax(counts)
    return labels == keep


def _mesh_from_binary_volume(
    binary_volume: np.ndarray,
    pmin: np.ndarray,
    pmax: np.ndarray,
) -> trimesh.Trimesh:
    from skimage import measure

    resolution = int(binary_volume.shape[0])
    verts, faces, _, _ = measure.marching_cubes(binary_volume.astype(np.float32), level=0.5)
    world = pmin + (verts / max(resolution - 1, 1)) * (pmax - pmin)
    return trimesh.Trimesh(vertices=world.astype(np.float32), faces=faces.astype(np.int64), process=False)


def _watertight_mesh_from_volume(
    points: np.ndarray,
    colors: np.ndarray,
    sigma: np.ndarray,
    resolution: int,
    close_iters: int,
    smooth_iters: int,
    target_faces: int,
) -> trimesh.Trimesh:
    from scipy import ndimage

    resolution = int(max(64, resolution))
    pmin, pmax = _compute_bounds(points, margin_ratio=0.06)
    idx = _points_to_indices(points, pmin, pmax, resolution)

    density = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    weights = np.clip(sigma.astype(np.float32), 1e-3, None)
    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), weights)

    blur_sigma = max(1.0, resolution / 256.0)
    density = ndimage.gaussian_filter(density, sigma=blur_sigma, mode="nearest")
    nonzero = density[density > 0]
    if nonzero.size == 0:
        raise ValueError("Volumetric fallback failed: empty density grid.")
    threshold = float(np.quantile(nonzero, 0.20))
    occ = density >= threshold
    if not occ.any():
        occ = density > 0

    if close_iters > 0:
        occ = ndimage.binary_closing(occ, iterations=int(close_iters))
    occ = ndimage.binary_fill_holes(occ)
    occ = _largest_component_mask(occ)
    if not occ.any():
        raise ValueError("Volumetric fallback failed: empty occupancy after cleanup.")

    mesh = _mesh_from_binary_volume(occ, pmin, pmax)
    mesh = _finalize_for_glb(mesh, smooth_iters=smooth_iters, target_faces=target_faces)
    mesh = _colorize_mesh(mesh, points, colors)
    return mesh


def _force_close_with_mesh_voxel_fill(
    mesh: trimesh.Trimesh,
    points_for_color: np.ndarray,
    colors_for_color: np.ndarray,
    resolution: int,
    close_iters: int,
    smooth_iters: int,
    target_faces: int,
) -> trimesh.Trimesh:
    from scipy import ndimage

    resolution = int(max(64, resolution))
    sample_count = int(min(max(mesh.faces.shape[0] * 2, 50000), 600000))
    try:
        sampled, _ = trimesh.sample.sample_surface(mesh, sample_count)
        pts = np.concatenate([mesh.vertices, sampled], axis=0)
    except Exception:
        pts = mesh.vertices
    if pts.shape[0] == 0:
        raise ValueError("Hard close fallback failed: empty mesh points.")

    pmin, pmax = _compute_bounds(pts, margin_ratio=0.03)
    idx = _points_to_indices(pts, pmin, pmax, resolution)

    occ = np.zeros((resolution, resolution, resolution), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ = ndimage.binary_dilation(occ, iterations=1)
    if close_iters > 0:
        occ = ndimage.binary_closing(occ, iterations=int(close_iters))
    occ = ndimage.binary_fill_holes(occ)
    occ = _largest_component_mask(occ)
    if not occ.any():
        raise ValueError("Hard close fallback failed: empty occupancy.")

    mesh_closed = _mesh_from_binary_volume(occ, pmin, pmax)
    mesh_closed = _finalize_for_glb(
        mesh_closed,
        smooth_iters=max(int(smooth_iters), 1),
        target_faces=target_faces,
    )
    mesh_closed = _colorize_mesh(mesh_closed, points_for_color, colors_for_color)
    return mesh_closed


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

    c0 = 0.28209479177387814
    f_dc = (np.clip(colors, 0.0, 1.0) - 0.5) / c0
    scale = np.log(np.clip(np.abs(radius_np), 1e-4, None))
    quat = _euler_xyz_to_quat_wxyz(rot_np)
    opacity = np.log(np.clip(sigma, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - sigma, 1e-4, 1.0))
    nx = np.zeros((points.shape[0],), dtype=np.float32)
    ny = np.zeros((points.shape[0],), dtype=np.float32)
    nz = np.zeros((points.shape[0],), dtype=np.float32)

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
    glb_mode: str = "watertight",
    watertight_resolution: int = 384,
    watertight_close_iters: int = 2,
    watertight_smooth_iters: int = 5,
    watertight_target_faces: int = 200000,
    fail_if_non_watertight: bool = True,
) -> Dict[str, int]:
    """Export GLB mesh, with optional watertight-first pipeline for Unity."""
    if glb_mode not in {"watertight", "legacy"}:
        raise ValueError(f"Unsupported glb_mode={glb_mode}. Use 'watertight' or 'legacy'.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    points, colors, sigma = _prepare_points(
        xyzs=xyzs,
        rgbs=rgbs,
        sigmas=sigmas,
        sigma_percentile=sigma_percentile,
        max_points=max_points,
        random_seed=random_seed,
    )

    mesh_tm, points_used, colors_used = _candidate_mesh_poisson(
        points=points,
        colors=colors,
        mesh_depth=mesh_depth,
        density_quantile=density_quantile,
    )
    mesh_tm = _finalize_for_glb(mesh_tm)
    mesh_tm = _colorize_mesh(mesh_tm, points_used, colors_used)
    metrics = _mesh_quality_metrics(mesh_tm)
    pipeline = "poisson"

    if glb_mode == "watertight" and not metrics["watertight"]:
        mesh_tm = _watertight_mesh_from_volume(
            points=points,
            colors=colors,
            sigma=sigma,
            resolution=watertight_resolution,
            close_iters=watertight_close_iters,
            smooth_iters=watertight_smooth_iters,
            target_faces=watertight_target_faces,
        )
        metrics = _mesh_quality_metrics(mesh_tm)
        pipeline = "volumetric_fallback"

    if glb_mode == "watertight" and not metrics["watertight"]:
        mesh_tm = _force_close_with_mesh_voxel_fill(
            mesh=mesh_tm,
            points_for_color=points,
            colors_for_color=colors,
            resolution=max(watertight_resolution // 2, 128),
            close_iters=max(watertight_close_iters, 1),
            smooth_iters=watertight_smooth_iters,
            target_faces=watertight_target_faces,
        )
        metrics = _mesh_quality_metrics(mesh_tm)
        pipeline = "volumetric_fallback"

    if glb_mode == "watertight" and not metrics["watertight"]:
        hull = trimesh.points.PointCloud(points).convex_hull
        hull = _finalize_for_glb(
            hull,
            smooth_iters=max(watertight_smooth_iters // 2, 0),
            target_faces=watertight_target_faces,
        )
        hull = _colorize_mesh(hull, points, colors)
        mesh_tm = hull
        metrics = _mesh_quality_metrics(mesh_tm)
        pipeline = "hull_fallback"

    if fail_if_non_watertight and not metrics["watertight"]:
        raise RuntimeError(
            f"GLB export failed watertight requirement. boundary_edges={metrics['boundary_edges']}, "
            f"pipeline={pipeline}."
        )

    mesh_tm.export(output_path)
    return {
        "pipeline": pipeline,
        "watertight": bool(metrics["watertight"]),
        "boundary_edges": int(metrics["boundary_edges"]),
        "points_after_filter": int(points.shape[0]),
        "mesh_vertices": int(mesh_tm.vertices.shape[0]),
        "mesh_faces": int(mesh_tm.faces.shape[0]),
    }
