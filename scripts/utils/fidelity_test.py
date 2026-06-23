"""
转换保真度实测 — Mesh → Surfel 在不同采样密度下的几何误差
============================================================
用 Point-to-Mesh 距离 (Chamfer Distance) 评估不同 num_points 下
GS Surfel 和原始 Mesh 之间的几何偏差。
输出 fidelity.json 供 comprehensive_comparison.py 读取。

用法:
  python scripts/utils/fidelity_test.py --mesh_path output/object_b_mesh.obj --output output/comparison/fidelity_test
"""

import numpy as np, os, sys, json, argparse

# 把 utils 目录加入 path 以便导入 aigc_to_gs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aigc_to_gs import mesh_to_surfel


def chamfer_distance_mesh_to_pcd(mesh, pcd_points, n_samples=20000):
    """Mesh 到 点云的 Chamfer Distance"""
    import open3d as o3d
    try:
        mesh_pts = mesh.sample_points_uniformly(n_samples)
        mesh_pts_np = np.asarray(mesh_pts.points)
    except Exception:
        # fallback: 用 mesh vertices
        mesh_pts_np = np.asarray(mesh.vertices)
        if len(mesh_pts_np) > n_samples:
            idx = np.random.choice(len(mesh_pts_np), n_samples, replace=False)
            mesh_pts_np = mesh_pts_np[idx]

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_points)

    mesh_pts_o3d = o3d.geometry.PointCloud()
    mesh_pts_o3d.points = o3d.utility.Vector3dVector(mesh_pts_np)

    dist_pcd2mesh = np.asarray(pcd_o3d.compute_point_cloud_distance(mesh_pts_o3d))

    try:
        from scipy.spatial import KDTree
        tree = KDTree(pcd_points)
        dist_mesh2pcd, _ = tree.query(mesh_pts_np, k=1)
    except ImportError:
        dist_mesh2pcd = np.zeros_like(dist_pcd2mesh)

    cd = (np.mean(dist_pcd2mesh) + np.mean(dist_mesh2pcd)) / 2
    return cd * 1000  # 转换为毫米


def main():
    parser = argparse.ArgumentParser(description="Mesh→Surfel 转换保真度测试")
    parser.add_argument("--mesh_path", default="output/object_b_mesh.obj",
                        help="输入 Mesh 路径")
    parser.add_argument("--output", default="output/comparison/fidelity_test",
                        help="输出目录")
    args = parser.parse_args()

    import open3d as o3d
    mesh_path = args.mesh_path
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(mesh_path):
        print(f'ERROR: Mesh 文件不存在: {mesh_path}')
        return

    # 加载原 Mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        print(f'ERROR: {mesh_path} 是空Mesh'); return
    mesh.compute_vertex_normals()
    print(f'Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces')

    densities = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    cd_vals, size_vals = [], []

    for n in densities:
        ply_path = os.path.join(out_dir, f'fidelity_{n}.ply')
        if not os.path.exists(ply_path):
            success = mesh_to_surfel(mesh_path, ply_path, n)
            if not success:
                print(f'  {n:6d} pts: FAILED (mesh_to_surfel 转换失败)')
                continue
        size_mb = os.path.getsize(ply_path) / (1024*1024)
        size_vals.append(round(size_mb, 2))

        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            pcd_points = np.asarray(pcd.points)
            cd = chamfer_distance_mesh_to_pcd(mesh, pcd_points)
            cd_vals.append(round(cd, 3))
            print(f'  {n:6d} pts: CD={cd:.3f} mm, {size_mb:.1f} MB')
        except Exception as e:
            print(f'  {n:6d} pts: ERROR ({e})')

    with open(os.path.join(out_dir, 'fidelity.json'), 'w') as f:
        json.dump({'densities': densities[:len(cd_vals)],
                   'cd_mm': cd_vals, 'size_mb': size_vals}, f, indent=2)
    print(f'Done → {out_dir}/fidelity.json')


if __name__ == '__main__':
    main()
