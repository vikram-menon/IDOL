#!/usr/bin/env python3
"""Headless Blender helper: import GLB, create a basic humanoid armature, and export FBX for Unity.

Usage:
  blender --background --python scripts/blender_autorig_unity.py -- \
      --input_glb outputs/avatar.glb --output_fbx outputs/avatar_unity.fbx
"""

import argparse
import os
import sys

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Auto-rig GLB and export Unity FBX.")
    parser.add_argument("--input_glb", required=True, help="Input GLB path.")
    parser.add_argument("--output_fbx", required=True, help="Output FBX path.")
    parser.add_argument(
        "--decimate_target_faces",
        type=int,
        default=45000,
        help="Optional face budget before rigging (0 disables decimation).",
    )
    parser.add_argument(
        "--name",
        default="Avatar",
        help="Base name for generated mesh/armature objects.",
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_glb(path: str):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not meshes:
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects found after importing: {path}")
    return meshes


def join_meshes(meshes, mesh_name: str):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh_obj = bpy.context.view_layer.objects.active
    mesh_obj.name = mesh_name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return mesh_obj


def estimate_bounds_world(mesh_obj):
    corners = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def add_decimate_if_needed(mesh_obj, target_faces: int) -> None:
    if target_faces <= 0:
        return

    face_count = len(mesh_obj.data.polygons)
    if face_count <= target_faces:
        return

    ratio = max(0.02, min(1.0, float(target_faces) / float(face_count)))
    mod = mesh_obj.modifiers.new(name="AutoDecimate", type="DECIMATE")
    mod.ratio = ratio
    mod.use_collapse_triangulate = True

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def create_humanoid_armature(name: str, bmin: Vector, bmax: Vector):
    center_x = 0.5 * (bmin.x + bmax.x)
    center_y = 0.5 * (bmin.y + bmax.y)
    width = max(1e-4, bmax.x - bmin.x)
    depth = max(1e-4, bmax.y - bmin.y)
    height = max(1e-4, bmax.z - bmin.z)

    z = lambda t: bmin.z + t * height
    x_off = max(width * 0.16, height * 0.05)
    y_off = depth * 0.08

    arm_data = bpy.data.armatures.new(f"{name}_ArmatureData")
    arm_obj = bpy.data.objects.new(f"{name}_Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name_, head, tail, parent=None):
        b = eb.new(name_)
        b.head = head
        b.tail = tail
        if parent is not None:
            b.parent = parent
            b.use_connect = True
        return b

    hips = bone("Hips", (center_x, center_y, z(0.50)), (center_x, center_y, z(0.58)))
    spine = bone("Spine", hips.tail, (center_x, center_y, z(0.68)), hips)
    chest = bone("Chest", spine.tail, (center_x, center_y, z(0.80)), spine)
    neck = bone("Neck", chest.tail, (center_x, center_y, z(0.88)), chest)
    bone("Head", neck.tail, (center_x, center_y, z(0.98)), neck)

    l_sh = bone("LeftShoulder", chest.tail, (center_x + x_off * 0.55, center_y, z(0.81)), chest)
    l_ua = bone("LeftUpperArm", l_sh.tail, (center_x + x_off * 1.35, center_y - y_off, z(0.77)), l_sh)
    l_la = bone("LeftLowerArm", l_ua.tail, (center_x + x_off * 2.0, center_y - y_off, z(0.73)), l_ua)
    bone("LeftHand", l_la.tail, (center_x + x_off * 2.3, center_y - y_off, z(0.72)), l_la)

    r_sh = bone("RightShoulder", chest.tail, (center_x - x_off * 0.55, center_y, z(0.81)), chest)
    r_ua = bone("RightUpperArm", r_sh.tail, (center_x - x_off * 1.35, center_y - y_off, z(0.77)), r_sh)
    r_la = bone("RightLowerArm", r_ua.tail, (center_x - x_off * 2.0, center_y - y_off, z(0.73)), r_ua)
    bone("RightHand", r_la.tail, (center_x - x_off * 2.3, center_y - y_off, z(0.72)), r_la)

    l_ul = bone("LeftUpperLeg", hips.head, (center_x + x_off * 0.55, center_y, z(0.33)), hips)
    l_ll = bone("LeftLowerLeg", l_ul.tail, (center_x + x_off * 0.55, center_y, z(0.12)), l_ul)
    l_foot = bone("LeftFoot", l_ll.tail, (center_x + x_off * 0.55, center_y + depth * 0.22, z(0.05)), l_ll)
    bone("LeftToes", l_foot.tail, (center_x + x_off * 0.55, center_y + depth * 0.34, z(0.05)), l_foot)

    r_ul = bone("RightUpperLeg", hips.head, (center_x - x_off * 0.55, center_y, z(0.33)), hips)
    r_ll = bone("RightLowerLeg", r_ul.tail, (center_x - x_off * 0.55, center_y, z(0.12)), r_ul)
    r_foot = bone("RightFoot", r_ll.tail, (center_x - x_off * 0.55, center_y + depth * 0.22, z(0.05)), r_ll)
    bone("RightToes", r_foot.tail, (center_x - x_off * 0.55, center_y + depth * 0.34, z(0.05)), r_foot)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def bind_mesh_to_armature(mesh_obj, arm_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def export_fbx(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=False,
        apply_unit_scale=True,
        use_armature_deform_only=True,
        path_mode="AUTO",
    )


def main() -> None:
    args = parse_args()
    input_glb = os.path.abspath(args.input_glb)
    output_fbx = os.path.abspath(args.output_fbx)

    if not os.path.exists(input_glb):
        raise FileNotFoundError(f"Input GLB not found: {input_glb}")

    clear_scene()
    meshes = import_glb(input_glb)
    mesh_obj = join_meshes(meshes, f"{args.name}_Mesh")
    add_decimate_if_needed(mesh_obj, target_faces=int(args.decimate_target_faces))
    bmin, bmax = estimate_bounds_world(mesh_obj)

    arm_obj = create_humanoid_armature(args.name, bmin, bmax)
    bind_mesh_to_armature(mesh_obj, arm_obj)
    export_fbx(output_fbx)
    print(f"[OK] Unity FBX exported: {output_fbx}")


if __name__ == "__main__":
    main()
