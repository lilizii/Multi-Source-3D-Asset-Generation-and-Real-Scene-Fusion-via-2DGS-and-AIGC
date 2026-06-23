"""
AIGC 模型转高斯面片 (GS) 工具
===========================================
核心功能: 将 threestudio/Magic123 生成的隐式场/Mesh 转换为高斯面片表示,
实现与 2DGS 背景的统一表达形式, 支持代码级拼接渲染。

技术路线:
1. 从 Mesh 采样点云 (均匀采样 + Poisson Disk)
2. 为每个点计算几何属性 (位置, 法线, 半径, 颜色)
3. 构造 2D Gaussian 面片 (Surfel) 表示
4. 导出为 .ply 格式, 与 2DGS 兼容
5. 合并到背景场景中进行统一渲染

参考文献:
- 2DGS: "2D Gaussian Splatting for Geometrically Accurate Radiance Fields"
- 3DGS: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
"""

import argparse
import numpy as np
import os
import sys

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


def sample_mesh_to_pointcloud(mesh, num_points: int = 50000):
    """
    从 Mesh 采样点云 - 使用均匀采样
    """
    if hasattr(mesh, 'sample'):
        # trimesh
        points, face_indices = mesh.sample(num_points, return_index=True)
        return points, face_indices
    else:
        # open3d
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        return np.asarray(pcd.points), None


def compute_surfel_properties(points, normals, colors, radius_scale: float = 0.4):
    """
    为点云计算 2DGS Surfel 属性:
    - 位置 (xyz)
    - 法线 (用于确定面片方向)
    - 缩放 (scaling): 切线方向用半径, 法线方向用小值
    - 旋转 (rotation): 从法线和切线构建
    - 颜色 (RGB)
    - 不透明度 (opacity): 默认值
    """
    N = len(points)

    # 位置
    xyz = points.astype(np.float32)

    # 法线: 估计 + 统一朝向 (全部指向外侧)
    if normals is None:
        normals = estimate_normals(xyz)
    # 确保法线一致朝外 (以点云中心为参考)
    center = xyz.mean(axis=0)
    to_center = center - xyz
    dot = np.sum(normals * to_center, axis=1)
    normals = np.where(dot[:, np.newaxis] < 0, -normals, normals)
    # 归一化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    # 颜色: 归一化到 [0, 1]
    if colors is None:
        colors = np.ones((N, 3), dtype=np.float32) * 0.7
    else:
        colors = np.asarray(colors, dtype=np.float32)
        # 处理 uint8 存储为 float 的情况 (trimesh 常见)
        if colors.max() > 1.5:
            colors = colors / 255.0
        # 确保最少3通道
        if colors.shape[1] >= 4:
            colors = colors[:, :3]
        # clip
        colors = np.clip(colors, 0.0, 1.0)

    # 完全复制 2DGS create_from_pcd 的初始化逻辑
    # scale = log(NN_dist), opacity = 0.1
    # 注意: 训练时 optimizer 会自动调大 scale, 静态 surfel 需要乘个系数
    if HAS_O3D:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        nn = np.array([kdtree.search_knn_vector_3d(pcd.points[i], 2)[2][1]
                       for i in range(N)], dtype=np.float32)
    else:
        nn = np.ones(N, dtype=np.float32) * 0.01
    # 返回 raw scale (非 log), export_gs_ply 会做 np.log
    raw_scale = np.sqrt(np.maximum(nn, 1e-7)) * radius_scale
    scaling = np.zeros((N, 2), dtype=np.float32)
    scaling[:, 0] = raw_scale
    scaling[:, 1] = raw_scale

    rotations = compute_rotations_from_normals(normals)
    opacities = np.ones((N, 1), dtype=np.float32) * 0.5

    return {
        'xyz': xyz,
        'normals': normals,
        'colors': colors,
        'scaling': scaling,
        'rotations': rotations,
        'opacities': opacities,
    }


def estimate_normals(points, k: int = 30):
    """估计点云法线"""
    if HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        return np.asarray(pcd.normals)
    else:
        # 简单近似: 指向原点
        n = np.zeros_like(points)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        n = points / norms
        return n


def estimate_radii(points, k: int = 20):
    """基于最近邻距离估计半径"""
    if HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        radii = []
        for i in range(len(points)):
            _, idx, dist_sq = kdtree.search_knn_vector_3d(pcd.points[i], k + 1)
            # 跳过自己 (idx[0] 是自己)
            avg_dist = np.sqrt(np.mean(dist_sq[1:]))
            radii.append(avg_dist)
        return np.array(radii).astype(np.float32)
    else:
        return np.ones(len(points), dtype=np.float32) * 0.01


def compute_rotations_from_normals(normals):
    """
    从法线计算四元数旋转
    将世界坐标的 +z 旋转到法线方向

    返回: (N, 4) 四元数 xyzw
    """
    N = len(normals)
    rotations = np.zeros((N, 4), dtype=np.float32)

    # 默认方向: z轴
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for i in range(N):
        n = normals[i]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            rotations[i] = [0, 0, 0, 1]  # 单位四元数
            continue
        n = n / n_norm

        # 从 z_axis 旋转到 n 的四元数
        axis_angle = np.cross(z_axis, n) * np.arccos(np.clip(np.dot(z_axis, n), -1, 1))
        if HAS_O3D:
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(rotation)
            rotations[i] = r.as_quat()
        else:
            v = np.cross(z_axis, n)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, n)
            if s < 1e-8:
                rotations[i] = [0, 0, 0, 1]
            else:
                v = v / s
                angle = np.arctan2(s, c) / 2
                rotations[i, 0:3] = v * np.sin(angle)
                rotations[i, 3] = np.cos(angle)

    return rotations


def export_gs_ply(surfel_data: dict, output_path: str):
    """
    导出为标准 3DGS/2DGS .ply 格式

    PLY 格式包含:
    - x, y, z: 位置
    - nx, ny, nz: 法线
    - f_dc_0, f_dc_1, f_dc_2: DC 颜色 (SH degree 0)
    - f_rest_*: 高阶 SH 系数 (可选, 填充0)
    - opacity: 不透明度
    - scale_0, scale_1, scale_2: 缩放
    - rot_0, rot_1, rot_2, rot_3: 旋转四元数
    """
    from plyfile import PlyData, PlyElement
    import struct

    xyz = surfel_data['xyz']
    N = len(xyz)

    # 构建顶点数据
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ]

    # DC 颜色 (SH degree 0, 共3个分量)
    for i in range(3):
        dtype.append((f'f_dc_{i}', 'f4'))

    # 高阶 SH (degree 1-3, 共 (4^2-1)*3 = 45 个分量, 填充0)
    num_extra_sh = 45
    for i in range(num_extra_sh):
        dtype.append((f'f_rest_{i}', 'f4'))

    # 2DGS 只用 2 个 scale (切向 u, v)。法向厚度由 rasterizer 内部处理。
    # 导出 3 个 scale 会让 dtype 与原始 2DGS 训练出的 PLY 不兼容
    dtype += [
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    data = np.zeros(N, dtype=dtype)
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['nx'], data['ny'], data['nz'] = (surfel_data['normals'][:, 0],
                                           surfel_data['normals'][:, 1],
                                           surfel_data['normals'][:, 2])

    # SH DC (颜色编码)
    # CUDA 渲染路径 (2DGS 默认): display = C0 * SH_DC  (无 +0.5)
    # 所以 SH_DC = color / C0
    colors = surfel_data['colors']
    sh_factor = 0.28209479177387814
    for i in range(3):
        data[f'f_dc_{i}'] = colors[:, i] / sh_factor

    # 高阶 SH 填充 0
    for i in range(num_extra_sh):
        data[f'f_rest_{i}'] = 0.0

    data['scale_0'] = np.log(np.maximum(surfel_data['scaling'][:, 0], 1e-7))
    data['scale_1'] = np.log(np.maximum(surfel_data['scaling'][:, 1], 1e-7))
    op = np.clip(surfel_data['opacities'][:, 0], 0.001, 0.999)
    data['opacity'] = np.log(op / (1.0 - op))  # inverse_sigmoid
    data['rot_0'] = surfel_data['rotations'][:, 0]
    data['rot_1'] = surfel_data['rotations'][:, 1]
    data['rot_2'] = surfel_data['rotations'][:, 2]
    data['rot_3'] = surfel_data['rotations'][:, 3]

    el = PlyElement.describe(data, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Exported GS PLY: {output_path} ({N} surfels)")


def load_mesh(path: str):
    """加载 Mesh, 支持 .ply, .obj, .glb 等格式"""
    if not os.path.exists(path):
        print(f"Warning: mesh file not found: {path}")
        return None

    # 优先使用 trimesh
    if HAS_TRIMESH:
        try:
            mesh = trimesh.load(path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                # 合并所有子mesh
                meshes = [g for g in mesh.geometry.values()
                          if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    return None
            return mesh
        except Exception as e:
            print(f"trimesh failed: {e}")

    # fallback to open3d
    if HAS_O3D:
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if len(mesh.vertices) > 0:
                return mesh
        except Exception as e:
            print(f"open3d failed: {e}")

    return None


def mesh_to_surfel(mesh_path: str, output_path: str, num_points: int = 10000, scale_mult: float = 0.1):
    """
    完整流程: Mesh → 点云 → Surfel (高斯面片)
    """
    print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    if mesh is None:
        print(f"Failed to load mesh: {mesh_path}")
        return False

    # 提取顶点/点
    if hasattr(mesh, 'vertices'):
        if HAS_TRIMESH and isinstance(mesh, trimesh.Trimesh):
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32) if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None else None
            # trimesh 返回 TrackedArray uint8 [N,4] RGBA, 需要转换为 float [0,1]
            try:
                colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)
                if colors.max() > 1.5:
                    colors = colors / 255.0
                if colors.shape[1] >= 4:
                    colors = colors[:, :3]
                colors = np.clip(colors, 0.0, 1.0)
            except Exception:
                colors = None
        elif HAS_O3D and isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32) if mesh.has_vertex_normals() else None
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else None
        else:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            normals = None
            colors = None
    else:
        print("Cannot extract vertices from mesh")
        return False

    # 采样点云 (保留颜色)
    if len(vertices) > num_points:
        indices = np.random.choice(len(vertices), num_points, replace=False)
        vertices = vertices[indices]
        if normals is not None:
            normals = normals[indices]
        if colors is not None:
            colors = colors[indices]
    else:
        # 上采样: 用 trimesh.sample 获取 face_index, 插值顶点色
        print(f"Up-sampling: {len(vertices)} -> {num_points}")
        try:
            if HAS_TRIMESH and isinstance(mesh, trimesh.Trimesh):
                points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
                vertices = np.asarray(points, dtype=np.float32)
                # 插值颜色: 取 face 三个顶点的颜色平均
                if colors is not None and len(colors) > 0:
                    faces_v = np.asarray(mesh.faces, dtype=np.int64)[face_idx]  # TrackedArray → 普通数组
                    colors = (colors[faces_v[:, 0]] +
                              colors[faces_v[:, 1]] +
                              colors[faces_v[:, 2]]) / 3.0
                    colors = np.clip(colors, 0.0, 1.0)
                normals = None
            else:
                points, _ = sample_mesh_to_pointcloud(mesh, num_points)
                vertices = np.asarray(points, dtype=np.float32)
                normals = None
                colors = None
        except Exception:
            pass  # 使用原有顶点

    print(f"Computing surfel properties for {len(vertices)} points...")
    if colors is not None:
        print(f"  mesh colors: min={colors.min(axis=0)}, max={colors.max(axis=0)}, mean={colors.mean(axis=0)}")
    else:
        print(f"  mesh has NO vertex colors, using default gray 0.7")
    surfel_data = compute_surfel_properties(vertices, normals, colors, radius_scale=scale_mult)

    export_gs_ply(surfel_data, output_path)
    return True


def merge_gs_plys(ply_paths: list, output_path: str):
    """
    合并多个 .ply 文件到一个场景中
    用于将物体A, B, C 的高斯面片合并到背景场景
    """
    from plyfile import PlyData, PlyElement
    all_vertices = []
    for path in ply_paths:
        if os.path.exists(path):
            ply = PlyData.read(path)
            verts = ply['vertex'].data
            all_vertices.append(verts)
            print(f"  Loaded {path}: {len(verts)} surfels")
        else:
            print(f"  Skipped (not found): {path}")

    if not all_vertices:
        print("No valid PLY files to merge")
        return

    merged = np.concatenate(all_vertices)
    el = PlyElement.describe(merged, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Merged PLY: {output_path} ({len(merged)} total surfels)")


def main():
    parser = argparse.ArgumentParser(
        description="AIGC模型 → 高斯面片 (GS) 转换与合并"
    )
    parser.add_argument("--mesh_b", help="物体B (文本生成) mesh路径")
    parser.add_argument("--mesh_c", help="物体C (单图生成) mesh路径")
    parser.add_argument("--mesh_a", help="物体A (多视角重建) mesh路径(可选)")
    parser.add_argument("--bg_ply", help="背景GS PLY路径(可选,用于合并)")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--num_points", type=int, default=10000,
                        help="每个mesh采样的点数")
    parser.add_argument("--scale_mult", type=float, default=0.1,
                        help="surfel 尺寸倍数")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ply_files = []

    # 如果提供了背景,加入列表
    if args.bg_ply and os.path.exists(args.bg_ply):
        ply_files.append(args.bg_ply)

    # 物体B
    if args.mesh_b:
        out_b = os.path.join(args.output, "object_b_surfel.ply")
        if mesh_to_surfel(args.mesh_b, out_b, args.num_points, args.scale_mult):
            ply_files.append(out_b)

    # 物体C
    if args.mesh_c:
        out_c = os.path.join(args.output, "object_c_surfel.ply")
        if mesh_to_surfel(args.mesh_c, out_c, args.num_points, args.scale_mult):
            ply_files.append(out_c)

    # 物体A (可选, 如果已有mesh)
    if args.mesh_a and os.path.exists(args.mesh_a):
        out_a = os.path.join(args.output, "object_a_surfel.ply")
        if mesh_to_surfel(args.mesh_a, out_a, args.num_points, args.scale_mult):
            ply_files.append(out_a)

    # 合并所有 GS PLY
    if len(ply_files) > 1:
        merged_path = os.path.join(args.output, "merged_scene.ply")
        merge_gs_plys(ply_files, merged_path)
        print(f"\n合并场景已保存: {merged_path}")
        print("可用于 2DGS viewer 或后续渲染")


if __name__ == "__main__":
    main()
