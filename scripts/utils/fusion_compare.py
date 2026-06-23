"""
场景融合 — GS 合并 (代码级 Surfel 拼接)
==========================================
将物体A(原始2DGS Surfel) + B/C(转换Surfel) + 背景 合并为统一场景。

原理:
  1. 加载各物体的 GS PLY
  2. 对每个物体: 归一化 → 缩放 → 旋转 → 平移 (对齐到背景坐标)
  3. 面片大小从变换后的 NN 距离重算 (避免 log-space 不一致)
  4. 合并所有 Surfel → 导出 full_scene.ply

用法:
  # 自动检测所有路径 (仅需指定背景场景名)
  python scripts/utils/fusion_compare.py --bg_scene room --output output/fusion

  # 完整指定所有路径
  python scripts/utils/fusion_compare.py \
      --gs_bg output/background_room/point_cloud/iteration_30000/point_cloud.ply \
      --gs_a  output/object_a/point_cloud/iteration_30000/point_cloud.ply \
      --gs_b  output/fusion/gs_converted/object_b_surfel.ply \
      --gs_c  output/fusion/gs_converted/object_c_surfel.ply \
      --output output/fusion
"""

import argparse, os, sys, json
import numpy as np
from scipy.spatial import KDTree


def quaternion_multiply(q1, q2):
    """
    四元数乘法 (Hamilton product): q1 * q2
    输入均为 (x, y, z, w) 格式
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def apply_rotations(result, rot_x_deg, rot_y_deg, rot_z_deg):
    """按 X→Y→Z 顺序旋转位置、法线、四元数"""
    has_quats = all(f in result.dtype.names for f in ['rot_0', 'rot_1', 'rot_2', 'rot_3'])
    for axis, deg in [('x', rot_x_deg), ('y', rot_y_deg), ('z', rot_z_deg)]:
        if abs(deg) < 0.001:
            continue
        angle = np.radians(deg)
        ca, sa = np.cos(angle), np.sin(angle)
        if axis == 'x':
            yn = result['y'] * ca - result['z'] * sa
            zn = result['y'] * sa + result['z'] * ca
            result['y'], result['z'] = yn, zn
            nyn = result['ny'] * ca - result['nz'] * sa
            nzn = result['ny'] * sa + result['nz'] * ca
            result['ny'], result['nz'] = nyn, nzn
            q_rot = np.array([np.sin(angle/2), 0, 0, np.cos(angle/2)], dtype=np.float32)
        elif axis == 'y':
            xn = result['x'] * ca + result['z'] * sa
            zn = -result['x'] * sa + result['z'] * ca
            result['x'], result['z'] = xn, zn
            nxn = result['nx'] * ca + result['nz'] * sa
            nzn = -result['nx'] * sa + result['nz'] * ca
            result['nx'], result['nz'] = nxn, nzn
            q_rot = np.array([0, np.sin(angle/2), 0, np.cos(angle/2)], dtype=np.float32)
        else:  # z
            xn = result['x'] * ca - result['y'] * sa
            yn = result['x'] * sa + result['y'] * ca
            result['x'], result['y'] = xn, yn
            nxn = result['nx'] * ca - result['ny'] * sa
            nyn = result['nx'] * sa + result['ny'] * ca
            result['nx'], result['ny'] = nxn, nyn
            q_rot = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)], dtype=np.float32)
        if has_quats:
            q_old = np.stack([result['rot_0'], result['rot_1'], result['rot_2'], result['rot_3']], axis=-1)
            q_new = quaternion_multiply(q_rot[np.newaxis, :], q_old)
            result['rot_0'], result['rot_1'], result['rot_2'], result['rot_3'] = q_new[:,0], q_new[:,1], q_new[:,2], q_new[:,3]


def transform_gs_vertices(verts, target_center, scale=1.0,
                          rot_x_deg=0, rot_y_deg=0, rot_z_deg=0,
                          preserve_geometry=True):
    result = np.copy(verts)
    has_scales = 'scale_0' in result.dtype.names and 'scale_1' in result.dtype.names
    cur_center = np.array([np.mean(result['x']), np.mean(result['y']), np.mean(result['z'])])
    cur_extent = max(result['x'].max()-result['x'].min(), result['y'].max()-result['y'].min(), result['z'].max()-result['z'].min())

    if preserve_geometry:
        result['x'] -= cur_center[0]; result['y'] -= cur_center[1]; result['z'] -= cur_center[2]
        result['x'] *= scale; result['y'] *= scale; result['z'] *= scale
        if has_scales:
            log_S = np.log(max(scale, 1e-7))
            result['scale_0'] += log_S; result['scale_1'] += log_S
        apply_rotations(result, rot_x_deg, rot_y_deg, rot_z_deg)
        result['x'] += target_center[0]; result['y'] += target_center[1]; result['z'] += target_center[2]

    else:
        # === 转换 surfel: 归一化确保一致的放置尺寸 ===
        # 1. 移到原点
        result['x'] -= cur_center[0]
        result['y'] -= cur_center[1]
        result['z'] -= cur_center[2]

        # 2. 归一化
        norm_factor = 1.0 / max(cur_extent, 1e-6)
        result['x'] *= norm_factor
        result['y'] *= norm_factor
        result['z'] *= norm_factor

        # 3. 缩放
        result['x'] *= scale; result['y'] *= scale; result['z'] *= scale

        # 4. 旋转 (X→Y→Z)
        apply_rotations(result, rot_x_deg, rot_y_deg, rot_z_deg)

        # 5. 平移
        result['x'] += target_center[0]
        result['y'] += target_center[1]
        result['z'] += target_center[2]

        # 6. 从变换后坐标重算 surfel 尺寸
        if has_scales:
            pts = np.column_stack([result['x'], result['y'], result['z']])
            k = min(10, len(pts) - 1)
            if k > 0:
                tree = KDTree(pts)
                dist, _ = tree.query(pts, k=k + 1)
                r = np.mean(dist[:, 1:], axis=1) * 0.5
                result['scale_0'] = np.log(np.maximum(r, 1e-7))
                result['scale_1'] = np.log(np.maximum(r, 1e-7))

    return result


def _deduce_bg_scene():
    """从 output/ 目录自动检测背景场景名"""
    import glob as gb
    for d in sorted(gb.glob('output/background_*')):
        if os.path.isdir(d):
            name = os.path.basename(d).replace('background_', '')
            ply = f'output/background_{name}/point_cloud/iteration_30000/point_cloud.ply'
            if os.path.exists(ply):
                return name
    return 'room'


def merge_all_gs(gs_bg_path, gs_a_path, gs_b_path, gs_c_path,
                 output_dir,
                 scale_a=1.5, scale_b=0.4, scale_c=0.4,
                 rot_x_a=0, rot_y_a=0, rot_z_a=0,
                 rot_x_b=0, rot_y_b=0, rot_z_b=0,
                 rot_x_c=0, rot_y_c=0, rot_z_c=0,
                 offset_a=(0.8, 0, 0), offset_b=(-0.8, 0, 0), offset_c=(0, 0, 0.8),
                 place_y=None):
    """
    合并背景 + A + B + C 为 full_scene.ply

    参数:
      gs_bg_path, gs_a_path, gs_b_path, gs_c_path: 各资产的 GS PLY 路径 (可为 None)
      output_dir: 输出目录
      scale_*: 各物体的世界空间缩放
      rot_y_*: 绕Y轴旋转角度 (度)
      offset_*: 相对于放置基准点的偏移
      place_y: 放置高度, 若为 None 则从背景中心获取
    """
    from plyfile import PlyData, PlyElement

    # 验证至少有一个输入
    all_paths = [gs_bg_path, gs_a_path, gs_b_path, gs_c_path]
    all_labels = ['background', 'object_a', 'object_b', 'object_c']
    available = [(l, p) for l, p in zip(all_labels, all_paths) if p and os.path.exists(p)]
    if not available:
        print("ERROR: 没有任何可用的 GS PLY 文件")
        print(f"  期望路径: bg={gs_bg_path}, a={gs_a_path}, b={gs_b_path}, c={gs_c_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # --- 加载背景并确定放置基准 ---
    bg_v = None
    bg_center = np.array([0., 0., 0.])
    if gs_bg_path and os.path.exists(gs_bg_path):
        bg_ply = PlyData.read(gs_bg_path)
        bg_v = bg_ply['vertex'].data
        bg_center = np.array([np.mean(bg_v['x']), np.mean(bg_v['y']), np.mean(bg_v['z'])])
        bg_extent_x = bg_v['x'].max() - bg_v['x'].min()
        bg_extent_z = bg_v['z'].max() - bg_v['z'].min()
        print(f"  背景: {len(bg_v)} surfels, "
              f"中心=({bg_center[0]:.2f}, {bg_center[1]:.2f}, {bg_center[2]:.2f}), "
              f"范围≈{bg_extent_x:.1f}×{bg_extent_z:.1f}")
    else:
        print(f"  WARNING: 背景 PLY 不存在: {gs_bg_path}, 将以物体A中心为基准")
        if gs_a_path and os.path.exists(gs_a_path):
            a_ply = PlyData.read(gs_a_path)
            bg_center = np.array([np.mean(a_ply['vertex']['x']),
                                  np.mean(a_ply['vertex']['y']),
                                  np.mean(a_ply['vertex']['z'])])

    # 放置基准点 (背景中心 xz, 指定高度 y)
    if place_y is None:
        place_y = bg_center[1]  # 默认用背景中心高度
    pos_base = np.array([bg_center[0], place_y, bg_center[2]])
    print(f"  放置基准: ({pos_base[0]:.2f}, {pos_base[1]:.2f}, {pos_base[2]:.2f})")

    verts = []
    stats = {}

    # --- 背景 (保持原位) ---
    if bg_v is not None:
        verts.append(bg_v)
        stats['background'] = len(bg_v)

    # --- 物体 A (原始训练 GS, 保留 surfel 参数) ---
    if gs_a_path and os.path.exists(gs_a_path):
        a_ply = PlyData.read(gs_a_path)
        target = pos_base + np.array(offset_a)
        a_v = transform_gs_vertices(a_ply['vertex'].data,
                                    target, scale=scale_a,
                                    rot_x_deg=rot_x_a, rot_y_deg=rot_y_a, rot_z_deg=rot_z_a,
                                    preserve_geometry=True)
        verts.append(a_v)
        stats['object_a'] = len(a_v)
        print(f"  A: {len(a_v)} surfels, scale={scale_a}, "
              f"pos=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f}), "
              f"rot=({rot_x_a},{rot_y_a},{rot_z_a})°")
    else:
        print(f"  A: SKIP ({gs_a_path} not found)")
        stats['object_a'] = 0

    # --- 物体 B (文本生成, Mesh→Surfel — 转换surfel无原始几何) ---
    if gs_b_path and os.path.exists(gs_b_path):
        b_ply = PlyData.read(gs_b_path)
        target = pos_base + np.array(offset_b)
        b_v = transform_gs_vertices(b_ply['vertex'].data,
                                    target, scale=scale_b,
                                    rot_x_deg=rot_x_b, rot_y_deg=rot_y_b, rot_z_deg=rot_z_b,
                                    preserve_geometry=False)
        verts.append(b_v)
        stats['object_b'] = len(b_v)
        print(f"  B: {len(b_v)} surfels, scale={scale_b}, "
              f"pos=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f}), "
              f"rot=({rot_x_b},{rot_y_b},{rot_z_b})°")
    else:
        print(f"  B: SKIP ({gs_b_path} not found)")
        stats['object_b'] = 0

    # --- 物体 C (单图生成, Mesh→Surfel — 转换surfel无原始几何) ---
    if gs_c_path and os.path.exists(gs_c_path):
        c_ply = PlyData.read(gs_c_path)
        target = pos_base + np.array(offset_c)
        c_v = transform_gs_vertices(c_ply['vertex'].data,
                                    target, scale=scale_c,
                                    rot_x_deg=rot_x_c, rot_y_deg=rot_y_c, rot_z_deg=rot_z_c,
                                    preserve_geometry=False)
        verts.append(c_v)
        stats['object_c'] = len(c_v)
        print(f"  C: {len(c_v)} surfels, scale={scale_c}, "
              f"pos=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f}), "
              f"rot=({rot_x_c},{rot_y_c},{rot_z_c})°")
    else:
        print(f"  C: SKIP ({gs_c_path} not found)")
        stats['object_c'] = 0

    if len(verts) == 0:
        print("ERROR: 没有任何 verts 可合并")
        return None

    # --- dtype 统一 (不同来源 PLY 字段可能不一致) ---
    ref_dtype = verts[0].dtype
    unified = []
    for v in verts:
        if v.dtype != ref_dtype:
            u = np.zeros(len(v), dtype=ref_dtype)
            for name in ref_dtype.names:
                if name in v.dtype.names:
                    u[name] = v[name]
            unified.append(u)
        else:
            unified.append(v)

    merged = np.concatenate(unified)
    out = os.path.join(output_dir, 'full_scene.ply')
    PlyData([PlyElement.describe(merged, 'vertex')]).write(out)
    stats['total'] = len(merged)
    print(f"  Total: {len(merged)} surfels → {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="GS Surfel 场景合并")

    # --- 核心输入路径 ---
    parser.add_argument("--gs_bg", default=None,
                        help="背景 GS PLY 路径 (如未指定则自动检测)")
    parser.add_argument("--gs_a", default=None,
                        help="物体A GS PLY 路径")
    parser.add_argument("--gs_b", default=None,
                        help="物体B 转换后 GS PLY 路径")
    parser.add_argument("--gs_c", default=None,
                        help="物体C 转换后 GS PLY 路径")

    # --- 快捷模式: 只指定背景场景名 ---
    parser.add_argument("--bg_scene", default=None,
                        help="背景场景名 (如 room), 自动推导所有路径")

    # --- 输出 ---
    parser.add_argument("--output", default="output/fusion",
                        help="输出目录")

    # --- 放置参数 ---
    parser.add_argument("--scale_a", type=float, default=1.5)
    parser.add_argument("--scale_b", type=float, default=0.4)
    parser.add_argument("--scale_c", type=float, default=0.4)
    parser.add_argument("--rot_x_a", type=float, default=0)
    parser.add_argument("--rot_y_a", type=float, default=0)
    parser.add_argument("--rot_z_a", type=float, default=0)
    parser.add_argument("--rot_x_b", type=float, default=0)
    parser.add_argument("--rot_y_b", type=float, default=0)
    parser.add_argument("--rot_z_b", type=float, default=0)
    parser.add_argument("--rot_x_c", type=float, default=0)
    parser.add_argument("--rot_y_c", type=float, default=0)
    parser.add_argument("--rot_z_c", type=float, default=0)
    parser.add_argument("--offset_a_x", type=float, default=0.8)
    parser.add_argument("--offset_a_y", type=float, default=0.0)
    parser.add_argument("--offset_a_z", type=float, default=0.0)
    parser.add_argument("--offset_b_x", type=float, default=-0.8)
    parser.add_argument("--offset_b_y", type=float, default=0.0)
    parser.add_argument("--offset_b_z", type=float, default=0.0)
    parser.add_argument("--offset_c_x", type=float, default=0.0)
    parser.add_argument("--offset_c_y", type=float, default=0.0)
    parser.add_argument("--offset_c_z", type=float, default=0.8)
    parser.add_argument("--place_y", type=float, default=None,
                        help="物体放置高度 (None=从背景自动获取)")

    args = parser.parse_args()

    # --- 路径推断 ---
    if args.bg_scene:
        # 快捷模式: 根据场景名推导
        bg_scene = args.bg_scene
        gs_bg = args.gs_bg or f'output/background_{bg_scene}/point_cloud/iteration_30000/point_cloud.ply'
        gs_a = args.gs_a or 'output/object_a/point_cloud/iteration_30000/point_cloud.ply'
        gs_b = args.gs_b or 'output/fusion/gs_converted/object_b_surfel.ply'
        gs_c = args.gs_c or 'output/fusion/gs_converted/object_c_surfel.ply'
    else:
        # 完整参数模式: 使用传入路径或自动检测
        bg_scene = _deduce_bg_scene()
        gs_bg = args.gs_bg or f'output/background_{bg_scene}/point_cloud/iteration_30000/point_cloud.ply'
        gs_a = args.gs_a or 'output/object_a/point_cloud/iteration_30000/point_cloud.ply'
        gs_b = args.gs_b or 'output/fusion/gs_converted/object_b_surfel.ply'
        gs_c = args.gs_c or 'output/fusion/gs_converted/object_c_surfel.ply'

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print(f"GS Surfel 场景合并 (背景: {bg_scene})")
    print(f"  背景 GS: {gs_bg}")
    print(f"  物体A GS: {gs_a}")
    print(f"  物体B GS: {gs_b}")
    print(f"  物体C GS: {gs_c}")
    print(f"  输出:    {args.output}/")
    print("=" * 60)

    stats = merge_all_gs(
        gs_bg, gs_a, gs_b, gs_c,
        args.output,
        scale_a=args.scale_a, scale_b=args.scale_b, scale_c=args.scale_c,
        rot_x_a=args.rot_x_a, rot_y_a=args.rot_y_a, rot_z_a=args.rot_z_a,
        rot_x_b=args.rot_x_b, rot_y_b=args.rot_y_b, rot_z_b=args.rot_z_b,
        rot_x_c=args.rot_x_c, rot_y_c=args.rot_y_c, rot_z_c=args.rot_z_c,
        offset_a=(args.offset_a_x, args.offset_a_y, args.offset_a_z),
        offset_b=(args.offset_b_x, args.offset_b_y, args.offset_b_z),
        offset_c=(args.offset_c_x, args.offset_c_y, args.offset_c_z),
        place_y=args.place_y,
    )

    if stats is None:
        print("\nFATAL: 合并失败")
        sys.exit(1)

    # --- 输出合并统计 ---
    out_path = os.path.join(args.output, 'full_scene.ply')
    stats["file_size_mb"] = round(os.path.getsize(out_path) / (1024 * 1024), 1)
    print(f"\n合并完成:")
    print(f"  总 surfel 数: {stats['total']}")
    print(f"  文件大小:     {stats['file_size_mb']} MB")
    print(f"  输出:         {out_path}")


if __name__ == "__main__":
    main()
