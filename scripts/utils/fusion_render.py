"""
场景融合与多视角渲染工具
==========================
使用 Open3D 将多个 Mesh 融合到统一场景中, 渲染多视角图像,
再合成为漫游视频。

相机来源:
  1. 优先读取背景场景的 COLMAP 相机位姿 (images.bin)
  2. 若无则生成环绕物体的人造轨迹
"""

import argparse
import numpy as np
import os
import sys
import struct
from collections import namedtuple

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ============================================================
# COLMAP binary reader
# ============================================================

Camera = namedtuple('Camera', ['id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'name'])


def read_colmap_images_bin(path):
    """读取 COLMAP images.bin, 返回相机位姿列表"""
    if not os.path.exists(path):
        return None
    cameras = []
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack('<I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
            tx, ty, tz = struct.unpack('<ddd', f.read(24))
            cam_id = struct.unpack('<I', f.read(4))[0]
            # null-terminated string
            name_bytes = []
            while True:
                b = f.read(1)
                if b == b'\x00':
                    break
                name_bytes.append(b)
            name = b''.join(name_bytes).decode('utf-8', errors='replace')
            cameras.append(Camera(img_id, qw, qx, qy, qz, tx, ty, tz, name))
    return cameras


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """四元数 → 旋转矩阵 (COLMAP convention)"""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ])
    return R


def colmap_camera_to_open3d(cam, up_sign=1.0):
    """
    COLMAP 相机 → Open3D (eye, look_at, up)
    COLMAP: world_point = R * camera_point + t
    → camera center in world = -R^T * t
    → forward direction = R * [0, 0, 1]  (camera looks along +z in COLMAP)
    """
    R = quaternion_to_rotation_matrix(cam.qw, cam.qx, cam.qy, cam.qz)
    t = np.array([cam.tx, cam.ty, cam.tz])

    # 相机在世界坐标中的位置
    eye = -R.T @ t

    # 相机朝向 (COLMAP: camera looks along +z)
    forward = R @ np.array([0.0, 0.0, 1.0])

    # up 方向 (COLMAP: -y is up)
    up = -R @ np.array([0.0, up_sign, 0.0])

    # look_at = eye + forward
    look_at = eye + forward

    return eye, look_at, up


def build_colmap_trajectory(images_bin_path, max_cameras=120):
    """
    从 COLMAP images.bin 构建相机轨迹
    均匀采样 max_cameras 个视角
    """
    cameras = read_colmap_images_bin(images_bin_path)
    if not cameras:
        return None

    n = len(cameras)
    if n > max_cameras:
        indices = np.linspace(0, n - 1, max_cameras, dtype=int)
        cameras = [cameras[i] for i in indices]

    trajectory = []
    for cam in cameras:
        eye, look_at, up = colmap_camera_to_open3d(cam)
        trajectory.append((eye, look_at, up))
    return trajectory


def load_mesh(path: str, default_color=None, normalize=True):
    """加载并预处理Mesh"""
    if not os.path.exists(path):
        print(f"Warning: mesh not found: {path}")
        return None

    mesh = o3d.io.read_triangle_mesh(path)

    if len(mesh.vertices) == 0:
        print(f"Warning: empty mesh: {path}")
        return None

    # 计算法线
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # 设置颜色
    if not mesh.has_vertex_colors() and default_color is not None:
        mesh.paint_uniform_color(default_color)

    # 归一化到单位球（仅对物体，不对背景）
    if normalize:
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = np.max(bbox.get_extent())
        if extent > 0:
            scale = 1.0 / extent
            mesh.translate(-center)
            mesh.scale(scale, center=(0, 0, 0))

    return mesh


def place_object(mesh, position, scale=1.0, rotation_y=0.0):
    """
    放置物体到场景中
    - position: (x, y, z) 放置位置
    - scale: 缩放
    - rotation_y: 绕Y轴旋转 (度)
    """
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.scale(scale, center=(0, 0, 0))
    R = mesh_copy.get_rotation_matrix_from_xyz((0, np.radians(rotation_y), 0))
    mesh_copy.rotate(R, center=(0, 0, 0))
    mesh_copy.translate(position)
    return mesh_copy


def create_camera_trajectory(center, radius, num_views=120, height_variation=True):
    """
    创建环绕相机轨迹

    参数:
    - center: 场景中心点
    - radius: 相机距离
    - num_views: 视角数量
    - height_variation: 是否在高度上做正弦变化
    """
    camera_positions = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        if height_variation:
            y = center[1] + radius * 0.3 * np.sin(angle * 2)
        else:
            y = center[1]

        # 相机看向中心
        look_at = center
        eye = np.array([x, y, z])
        camera_positions.append((eye, look_at))

    return camera_positions


def setup_visualizer_with_camera(vis, eye, look_at, up=(0, 1, 0)):
    """设置可视化器相机"""
    ctr = vis.get_view_control()
    ctr.set_lookat(look_at)
    ctr.set_front(eye - look_at)
    ctr.set_up(up)


def render_scene_open3d(scene_geometries, trajectory, output_dir,
                         width=1920, height=1080):
    """
    使用 Open3D 离屏渲染多视角图像
    trajectory: list of (eye, look_at, up) — up 可选, 默认 (0,1,0)
    """
    os.makedirs(output_dir, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    for geom in scene_geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = False
    render_option.background_color = np.array([0.05, 0.05, 0.1])
    render_option.point_size = 1.0

    for i, item in enumerate(trajectory):
        eye, look_at = item[0], item[1]
        up = item[2] if len(item) >= 3 else np.array([0.0, 1.0, 0.0])

        ctr = vis.get_view_control()
        ctr.set_lookat(look_at)
        ctr.set_front(eye - look_at)
        ctr.set_up(up)

        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = (np.asarray(image) * 255).astype(np.uint8)

        import cv2
        out_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if i % 30 == 0:
            print(f"  Rendered {i}/{len(trajectory)}...")

    vis.destroy_window()
    print(f"Rendered {len(trajectory)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="场景融合与多视角渲染")
    parser.add_argument("--bg_mesh", help="背景Mesh路径")
    parser.add_argument("--mesh_a", help="物体A Mesh路径")
    parser.add_argument("--mesh_b", help="物体B Mesh路径")
    parser.add_argument("--mesh_c", help="物体C Mesh路径")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_views", type=int, default=120)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--bg_source", default="",
                        help="Mip-NeRF 360 场景源路径，用于读取相机参数")
    args = parser.parse_args()

    if not HAS_O3D:
        print("Error: open3d is required")
        print("Install with: pip install open3d")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ---------- 加载所有模型 ----------
    print("=== 加载模型 ===")
    scene_geometries = []

    # 背景 — 保持原始坐标
    bg = load_mesh(args.bg_mesh, default_color=[0.4, 0.5, 0.3], normalize=False)
    if bg is not None:
        scene_geometries.append(bg)
        print(f"Background: {len(bg.vertices)} vertices")

    # 计算放置位置（基于背景包围盒）
    if bg is not None:
        bg_bbox = bg.get_axis_aligned_bounding_box()
        bg_center = bg_bbox.get_center()
        bg_min = bg_bbox.get_min_bound()
        # 物体放在背景中心附近、接近地面高度
        pos_base = bg_center.copy()
        pos_base[1] = bg_min[1] + 1.5  # 离地1.5m（桌面高度）
    else:
        pos_base = np.array([0, 0, 0])

    # 物体A: 毛绒挂件 (归一化后=1.0, scale=0.35 → 房间中约5-7cm)
    obj_a = load_mesh(args.mesh_a, default_color=[0.9, 0.5, 0.3])
    if obj_a is not None:
        obj_a = place_object(obj_a, position=pos_base + [0.8, 0, 0],
                             scale=0.35, rotation_y=-30)
        scene_geometries.append(obj_a)
        print(f"Object A: {len(obj_a.vertices)} vertices")

    # 物体B: 茶壶 (归一化后=1.0, scale=0.60 → 房间中约8-12cm)
    obj_b = load_mesh(args.mesh_b, default_color=[0.3, 0.6, 0.9])
    if obj_b is not None:
        obj_b = place_object(obj_b, position=pos_base + [-0.8, 0, 0],
                             scale=0.60, rotation_y=45)
        scene_geometries.append(obj_b)
        print(f"Object B: {len(obj_b.vertices)} vertices")

    # 物体C: 毛绒挂件 (归一化后=1.0, scale=0.35, 同A)
    obj_c = load_mesh(args.mesh_c, default_color=[0.3, 0.9, 0.4])
    if obj_c is not None:
        obj_c = place_object(obj_c, position=pos_base + [0, 0, 0.8],
                             scale=0.35, rotation_y=180)
        scene_geometries.append(obj_c)
        print(f"Object C: {len(obj_c.vertices)} vertices")

    # ---------- 创建相机轨迹 ----------
    print("\n=== 创建相机轨迹 ===")
    trajectory = None

    # 优先: 读取背景场景的原始 COLMAP 相机位姿
    if args.bg_source:
        colmap_images = os.path.join(args.bg_source, 'sparse', '0', 'images.bin')
        if not os.path.exists(colmap_images):
            colmap_images = os.path.join(args.bg_source, 'sparse', 'images.bin')
        if os.path.exists(colmap_images):
            print(f"  读取 COLMAP 相机: {colmap_images}")
            trajectory = build_colmap_trajectory(colmap_images, max_cameras=args.num_views)

    if trajectory and len(trajectory) > 0:
        print(f"  使用 {len(trajectory)} 个 COLMAP 原始视角")
    else:
        # 备选: 环绕物体放置位置的人造轨迹
        print(f"  COLMAP 相机不可用，使用环绕轨迹")
        scene_center = pos_base.copy()
        scene_center[1] += 0.3
        orbit_radius = 1.5
        print(f"  相机中心: ({scene_center[0]:.1f}, {scene_center[1]:.1f}, {scene_center[2]:.1f})")
        print(f"  绕行半径: {orbit_radius}")
        traj_raw = create_camera_trajectory(
            scene_center, radius=orbit_radius, num_views=args.num_views
        )
        trajectory = [(eye, look_at, np.array([0.0, 1.0, 0.0])) for eye, look_at in traj_raw]

    # ---------- 渲染 ----------
    print("\n=== 开始渲染 ===")
    render_output = os.path.join(args.output, "render_output")
    render_scene_open3d(
        scene_geometries, trajectory, render_output,
        width=args.width, height=args.height
    )

    print(f"\n渲染完成! 输出目录: {render_output}")


if __name__ == "__main__":
    main()
