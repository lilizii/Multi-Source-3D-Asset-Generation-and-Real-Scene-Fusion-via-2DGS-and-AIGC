"""
简化评估: 三方法对比 (几何准确度 + 纹理细节 + 计算效率)
==========================================================
纯 numpy/scipy 实现, 无第三方依赖 (包含简易 PLY/OBJ 解析器)

输出: output/evaluation/evaluation.json + summary_table.md

用法:
  python scripts/utils/evaluate.py [--skip_clip]
"""

import os, sys, csv, json, glob, argparse, struct, re
import numpy as np


# ============================================================
# 简易 PLY 解析器 (只读顶点 x,y,z — 兼容 2DGS binary PLY)
# ============================================================

def parse_ply_header(path):
    """读取 PLY header，返回 (format, vertex_count, properties, header_bytes)"""
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('end_header'):
                break
            if not line:
                raise ValueError("PLY header not terminated")

    header_text = '\n'.join(header_lines)
    fmt = 'ascii'
    if 'format binary_little_endian' in header_text:
        fmt = 'binary_le'
    elif 'format binary_big_endian' in header_text:
        fmt = 'binary_be'

    # 统计顶点数
    v_match = re.search(r'element vertex (\d+)', header_text)
    v_count = int(v_match.group(1)) if v_match else 0

    # 解析所有 property
    props = []
    for line in header_lines:
        if line.startswith('property '):
            parts = line.split()
            props.append({'type': parts[1], 'name': parts[2]})

    header_len = sum(len(l.encode()) + 1 for l in header_lines)  # +1 for \n
    return fmt, v_count, props, header_len


def load_ply_xyz(path, n_max=50000):
    """从 PLY 读取 x,y,z 坐标"""
    return _load_ply_fields(path, ['x','y','z'], n_max)


def load_ply_with_normals(path, n_max=50000):
    """从 PLY 读取 x,y,z,nx,ny,nz"""
    return _load_ply_fields(path, ['x','y','z','nx','ny','nz'], n_max)


def _load_ply_fields(path, fields, n_max=50000):
    """从 PLY 读取指定字段"""
    if not os.path.exists(path):
        return None
    try:
        fmt, n_verts, props, header_len = parse_ply_header(path)
        n_read = min(n_verts, n_max)

        idx_map = {p['name']: i for i, p in enumerate(props)}
        for f in fields:
            if f not in idx_map:
                return None  # 缺少必需字段

        type_sizes = {'float': 4, 'int': 4, 'uint': 4, 'double': 8, 'uchar': 1}
        record_size = sum(type_sizes.get(p['type'], 4) for p in props)
        byte_order = '<' if fmt == 'binary_le' else '>'

        result = {}
        for f in fields:
            result[f] = np.zeros(n_read, dtype=np.float32)

        with open(path, 'rb') as fh:
            fh.seek(header_len)
            buf = fh.read(record_size * n_read)

        for i in range(n_read):
            off = i * record_size
            for f in fields:
                result[f][i] = struct.unpack_from(f'{byte_order}f', buf, off + idx_map[f] * 4)[0]

        arr = np.stack([result[f] for f in fields], axis=-1).astype(np.float32)
        return arr
    except Exception as e:
        print(f"  [WARN] PLY parse failed for {path}: {e}")
        return None


# ============================================================
# 简易 OBJ 解析器
# ============================================================

def load_obj_xyz(path, n_max=50000):
    """从 OBJ 读取顶点坐标"""
    if not os.path.exists(path):
        return None
    try:
        verts = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not verts:
            return None
        pts = np.array(verts, dtype=np.float32)
        if len(pts) > n_max:
            pts = pts[np.random.choice(len(pts), n_max, replace=False)]
        return pts
    except Exception as e:
        print(f"  [WARN] OBJ parse failed for {path}: {e}")
        return None


def load_points(path):
    if not os.path.exists(path):
        return None
    if path.endswith('.ply'):
        return load_ply_xyz(path)
    return load_obj_xyz(path)


# ============================================================
# Chamfer Distance (scipy KDTree)
# ============================================================

def chamfer_distance(pc1, pc2):
    from scipy.spatial import KDTree
    d1, _ = KDTree(pc1).query(pc2, k=1)
    d2, _ = KDTree(pc2).query(pc1, k=1)
    return float(np.mean(d1) + np.mean(d2))


# ============================================================
# NeRF → Mesh 几何统计 (无需渲染)
# ============================================================

def obj_mesh_stats(path):
    """从 OBJ 提取: 顶点数, 面数, 包围盒, 是否有法线"""
    if not os.path.exists(path):
        return None
    try:
        verts = []
        has_normals = False
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vn '):
                    has_normals = True

        if not verts:
            return None
        v = np.array(verts, dtype=np.float32)
        bmin = v.min(axis=0)
        bmax = v.max(axis=0)
        extent = bmax - bmin

        # 面数: 单独扫描
        face_count = 0
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('f '):
                    face_count += 1

        return {
            'num_vertices': len(verts),
            'num_faces': face_count,
            'bbox_x': round(float(extent[0]), 3),
            'bbox_y': round(float(extent[1]), 3),
            'bbox_z': round(float(extent[2]), 3),
            'has_vertex_normals': has_normals,
        }
    except Exception as e:
        print(f"  [WARN] OBJ stats failed for {path}: {e}")
        return None


# ============================================================
# Mesh → GS 转换保真度
# ============================================================

def estimate_normals_pca(pts, k=30):
    """PCA 估计点云法线 (无 open3d 依赖)"""
    from scipy.spatial import KDTree
    tree = KDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    # 每个点的 k 近邻 (排除自身)
    neighbors = pts[idx[:, 1:]]

    # 中心化
    centered = neighbors - pts[:, None, :]
    # 协方差矩阵 (N, 3, 3) — 向量化
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    # 特征分解, 最小特征值对应的特征向量 = 法线
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]  # 最小特征值对应的向量

    # 归一化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    return normals.astype(np.float32)


def compute_mesh_to_gs_fidelity(mesh_path, gs_path):
    """
    计算 Mesh→GS 转换保真度:
      - CD: mesh顶点 ↔ surfel中心
      - 法线偏差: mesh顶点法线 vs 最近surfel法线
      - 压缩比: mesh顶点数 / surfel数
    返回 dict 或 None
    """
    mesh_pts = load_obj_xyz(mesh_path)
    gs_data = load_ply_with_normals(gs_path)

    if mesh_pts is None or gs_data is None:
        return None

    gs_xyz = gs_data[:, :3]
    gs_normals = gs_data[:, 3:6]

    # 统一采样量
    n_max = 20000
    if len(mesh_pts) > n_max:
        mesh_pts = mesh_pts[np.random.choice(len(mesh_pts), n_max, replace=False)]
    if len(gs_xyz) > n_max:
        idx = np.random.choice(len(gs_xyz), n_max, replace=False)
        gs_xyz = gs_xyz[idx]
        gs_normals = gs_normals[idx]

    # CD: mesh ↔ gs
    from scipy.spatial import KDTree
    tree_m = KDTree(mesh_pts)
    tree_g = KDTree(gs_xyz)
    d_m2g, _ = tree_g.query(mesh_pts, k=1)
    d_g2m, _ = tree_m.query(gs_xyz, k=1)
    cd = float(np.mean(d_m2g) + np.mean(d_g2m))

    # 法线偏差: 为mesh顶点估计法线, 找最近surfel, 比夹角
    mesh_normals = estimate_normals_pca(mesh_pts, k=30)
    # 对每个mesh顶点找最近surfel
    _, g_idx = tree_g.query(mesh_pts, k=1)
    g_nn = gs_normals[g_idx]
    # 夹角 (度)
    dot = np.abs(np.sum(mesh_normals * g_nn, axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    mean_angle = float(np.mean(angles))

    # 压缩比
    mesh_vert_count = len(load_obj_xyz(mesh_path))  # 原始全部顶点数
    ratio = round(mesh_vert_count / len(gs_data), 2)

    return {
        'chamfer_distance': round(cd, 4),
        'normal_angle_mean_deg': round(mean_angle, 1),
        'mesh_vertices': mesh_vert_count,
        'gs_surfels': len(gs_data),
        'compression_ratio': ratio,
    }


# ============================================================
# PSNR / SSIM (纯 numpy)
# ============================================================

def psnr_np(img1, img2):
    """PSNR (peak signal-to-noise ratio)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def compute_psnr_dir(render_dir, gt_dir):
    """对目录中的渲染图和GT图计算 PSNR"""
    from PIL import Image
    renders = sorted(glob.glob(os.path.join(render_dir, '*.png')))
    gts = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    if not renders or not gts:
        return None

    psnr_list = []
    for rp, gp in zip(renders, gts):
        r = np.array(Image.open(rp).convert('RGB'), dtype=np.float32) / 255.0
        g = np.array(Image.open(gp).convert('RGB'), dtype=np.float32) / 255.0
        h, w = min(r.shape[0], g.shape[0]), min(r.shape[1], g.shape[1])
        psnr_list.append(psnr_np(g[:h, :w], r[:h, :w]))

    return {
        'PSNR': round(np.mean(psnr_list), 2),
        'num_pairs': len(psnr_list),
    }


# ============================================================
# 路径配置
# ============================================================

PATHS = {
    'pc_a': 'output/object_a/point_cloud/iteration_30000/point_cloud.ply',
    'pc_b': 'output/object_b_mesh.obj',
    'pc_c': 'output/object_c_mesh.obj',
    'a_renders': 'output/object_a/test/ours_30000/renders',
    'a_gt': 'output/object_a/test/ours_30000/gt',
    'b_trial': 'output/object_b/dreamfusion-sd',
    'c_trial': 'output/object_c/magic123-refine-sd',
    'c_input_img': 'data/object_c/object_c_rgba.png',
    'csv_b': 'output/object_b/dreamfusion-sd',
    'csv_c_coarse': 'output/object_c/magic123-coarse-sd',
    'csv_c_refine': 'output/object_c/magic123-refine-sd',
    'gs_a': 'output/object_a/point_cloud/iteration_30000/point_cloud.ply',
    'gs_b': 'output/fusion/gs_converted/object_b_surfel.ply',
    'gs_c': 'output/fusion/gs_converted/object_c_surfel.ply',
    'mesh_a': 'output/object_a_mesh.ply',
    'mesh_b': 'output/object_b_mesh.obj',
    'mesh_c': 'output/object_c_mesh.obj',
    'mesh_b_for_gs': 'output/object_b_mesh.obj',
    'gs_b_surfel': 'output/fusion/gs_converted/object_b_surfel.ply',
    'mesh_c_for_gs': 'output/object_c_mesh.obj',
    'gs_c_surfel': 'output/fusion/gs_converted/object_c_surfel.ply',
    'out_dir': 'output/evaluation',
    'out_json': 'output/evaluation/evaluation.json',
    'out_md': 'output/evaluation/summary_table.md',
}


# ============================================================
# 从 threestudio CSV 读取步数
# ============================================================

def find_trial_dir(search_dir):
    if not os.path.isdir(search_dir):
        return None
    for item in sorted(os.listdir(search_dir)):
        item_path = os.path.join(search_dir, item)
        if os.path.isdir(item_path) and '@' in item:
            if os.path.isdir(os.path.join(item_path, 'ckpts')):
                return item_path
    return None


def find_csv(trial_dir):
    """在 trial 目录下递归查找 metrics.csv"""
    if not trial_dir:
        return None
    for root, dirs, files in os.walk(trial_dir):
        for f in files:
            if f == 'metrics.csv':
                return os.path.join(root, f)
    return None


def read_csv_steps(search_dir):
    trial = find_trial_dir(search_dir)
    if not trial:
        return None
    for root, dirs, files in os.walk(trial):
        for f in files:
            if f == 'metrics.csv':
                csv_path = os.path.join(root, f)
                try:
                    with open(csv_path, 'r') as cf:
                        reader = csv.DictReader(cf)
                        steps = []
                        for row in reader:
                            s = int(row.get('step', 0))
                            if s > 0:
                                steps.append(s)
                    if steps:
                        return {
                            'total_steps': len(steps),
                            'max_step': max(steps),
                            'estimated_min': round(len(steps) * 0.4 / 60, 1),
                        }
                except Exception:
                    pass
    return None


def read_2dgs_train_time(logdir):
    """从 2DGS TB 读取 iter_time, 计算总训练时间 (分钟)"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(logdir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        iter_times = None
        max_step = 0
        for tag in tags:
            events = ea.Scalars(tag)
            if not events:
                continue
            max_step = max(max_step, events[-1].step)
            if tag == 'iter_time':
                iter_times = np.array([e.value for e in events])
        if iter_times is not None and len(iter_times) > 0 and max_step > 0:
            avg_ms = np.mean(iter_times)
            total_min = round(avg_ms * max_step / 60000, 1)
            return total_min
    except Exception:
        pass
    return None


def read_threestudio_train_time(trial_dir):
    """从 trial 目录名中的时间戳和 CSV 修改时间估算训练时间"""
    trial = find_trial_dir(trial_dir)
    if not trial:
        return None
    # 从目录名提取开始时间: ...@20260615-194833
    basename = os.path.basename(trial)
    import re as re_mod
    m = re_mod.search(r'@(\d{8})-(\d{6})', basename)
    if not m:
        return None

    # 找 CSV 文件修改时间作为结束时间
    csv_path = find_csv(trial)
    if not csv_path:
        return None

    try:
        import datetime
        start_str = f"{m.group(1)}T{m.group(2)}"
        start = datetime.datetime.strptime(start_str, "%Y%m%dT%H%M%S")
        end_ts = os.path.getmtime(csv_path)
        end = datetime.datetime.fromtimestamp(end_ts)
        delta = (end - start).total_seconds() / 60
        return round(delta, 1) if delta > 0 else None
    except Exception:
        return None


def file_size_mb(path):
    try:
        return round(os.path.getsize(path) / 1048576, 2)
    except Exception:
        return None


# ============================================================
# CLIP (需要 torch + open_clip_torch)
# ============================================================

def find_test_views(trial_dir):
    if not trial_dir or not os.path.isdir(trial_dir):
        return []
    for root, dirs, files in os.walk(trial_dir):
        for d in sorted(dirs, reverse=True):
            if d.endswith('-test'):
                views = sorted(glob.glob(os.path.join(root, d, '*.png')))
                if views:
                    step = max(1, len(views) // 8)
                    return views[::step][:8]
    return []


def _load_clip():
    """尝试加载 CLIP 模型 (OpenAI clip → open_clip fallback)"""
    # 优先 OpenAI clip (autodl 通常有缓存)
    try:
        import clip
        import torch
        model, preprocess = clip.load("ViT-B/32")
        return model, preprocess, 'openai'
    except Exception:
        pass
    # fallback: open_clip (先试 openai 权重, 通常有 HF 缓存; 再试 laion2b)
    try:
        import open_clip
        for pretrained in ['openai', 'laion2b_s34b_b79k']:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained=pretrained)
                print(f"  CLIP: open_clip ViT-B-32 ({pretrained}) loaded")
                return model, preprocess, 'open_clip'
            except Exception:
                continue
    except Exception:
        pass
    return None, None, None


def compute_clip_text_image(image_paths, text_prompt):
    try:
        import torch
        from PIL import Image
        import clip as clip_module
        model, preprocess, backend = _load_clip()
        if model is None:
            return None
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            if backend == 'openai':
                text_tokens = clip_module.tokenize([text_prompt]).to(device)
                text_feat = model.encode_text(text_tokens)
            else:
                import open_clip
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
                text_tokens = tokenizer([text_prompt]).to(device)
                text_feat = model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            scores = []
            for ip in image_paths:
                img_tensor = preprocess(Image.open(ip).convert('RGB')).unsqueeze(0).to(device)
                img_feat = model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                scores.append((img_feat @ text_feat.T).item())
        return {'mean': round(np.mean(scores), 4), 'std': round(np.std(scores), 4),
                'num_views': len(scores)}
    except Exception as e:
        print(f"  [WARN] CLIP text-image failed: {e}")
        return None


def compute_clip_image_image(image_paths, ref_path):
    if not os.path.exists(ref_path):
        return None
    try:
        import torch
        from PIL import Image
        model, preprocess, backend = _load_clip()
        if model is None:
            return None
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            ref_img = preprocess(Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
            ref_feat = model.encode_image(ref_img)
            ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
            scores = []
            for ip in image_paths:
                img_tensor = preprocess(Image.open(ip).convert('RGB')).unsqueeze(0).to(device)
                img_feat = model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                scores.append((img_feat @ ref_feat.T).item())
        return {'mean': round(np.mean(scores), 4), 'std': round(np.std(scores), 4),
                'num_views': len(scores)}
    except Exception as e:
        print(f"  [WARN] CLIP image-image failed: {e}")
        return None


# ============================================================
# Markdown 报告
# ============================================================

def generate_md(R):
    geo = R.get('geometry', {})
    nerf2mesh = R.get('nerf_to_mesh_stats', {})
    mesh2gs = R.get('mesh_to_gs_fidelity', {})
    psnr_a = R.get('psnr_a', {})
    clip_b = R.get('clip_b', {})
    clip_c = R.get('clip_c', {})
    eff = R.get('efficiency', {})
    sizes = R.get('file_sizes', {})

    def v(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]       # 逐层深入
            else:
                return 'N/A'
        return d

    lines = [
        "# 三种方法对比: 多视角重建 vs 文本生成 vs 单图生成\n",
        "## 几何准确度\n",
        "| 指标 | A (多视角重建) | B (文本→3D) | C (单图→3D) |",
        "|------|:---:|:---:|:---:|",
        f"| CD A↔B (↓) | — | {v(geo,'A↔B')} | — |",
        f"| CD A↔C (↓) | — | — | {v(geo,'A↔C')} |",
        f"| CD B↔C (↓) | — | — | {v(geo,'B↔C')} |",
        f"| GS文件 (MB) | {v(sizes,'gs_a','size_mb')} | {v(sizes,'gs_b','size_mb')} | {v(sizes,'gs_c','size_mb')} |",
        "",
        "## 转换保真度: NeRF → Mesh\n",
        "| 指标 | B (文本→3D) | C (单图→3D) |",
        "|------|:---:|:---:|",
        f"| Mesh 顶点数 | {v(nerf2mesh,'b','num_vertices')} | {v(nerf2mesh,'c','num_vertices')} |",
        f"| Mesh 三角面数 | {v(nerf2mesh,'b','num_faces')} | {v(nerf2mesh,'c','num_faces')} |",
        f"| 包围盒 (x×y×z) | {v(nerf2mesh,'b','bbox_x')}×{v(nerf2mesh,'b','bbox_y')}×{v(nerf2mesh,'b','bbox_z')} | {v(nerf2mesh,'c','bbox_x')}×{v(nerf2mesh,'c','bbox_y')}×{v(nerf2mesh,'c','bbox_z')} |",
        f"| 顶点法线 | {v(nerf2mesh,'b','has_vertex_normals')} | {v(nerf2mesh,'c','has_vertex_normals')} |",
        "",
        "## 转换保真度: Mesh → GS\n",
        "| 指标 | B (文本→3D) | C (单图→3D) |",
        "|------|:---:|:---:|",
        f"| CD Mesh↔GS (↓) | {v(mesh2gs,'b','chamfer_distance')} | {v(mesh2gs,'c','chamfer_distance')} |",
        f"| 法线偏差 (°, ↓) | {v(mesh2gs,'b','normal_angle_mean_deg')} | {v(mesh2gs,'c','normal_angle_mean_deg')} |",
        f"| 顶点 → Surfel 压缩比 | {v(mesh2gs,'b','compression_ratio')} | {v(mesh2gs,'c','compression_ratio')} |",
        "",
        "## 纹理细节\n",
        "| 指标 | A (多视角重建) | B (文本→3D) | C (单图→3D) |",
        "|------|:---:|:---:|:---:|",
        f"| PSNR (dB ↑) | {v(psnr_a,'PSNR')} | N/A ¹ | N/A ¹ |",
        f"| CLIP Text→Image (↑) | N/A ² | {v(clip_b,'mean')} | N/A |",
        f"| CLIP Image→Image (↑) | N/A | N/A | {v(clip_c,'mean')} |",
        "",
        "## 计算效率\n",
        "| 指标 | A (多视角重建) | B (文本→3D) | C (单图→3D) |",
        "|------|:---:|:---:|:---:|",
    ]
    t_a = v(eff, 'a_2dgs', 'actual_time_min') if 'actual_time_min' in eff.get('a_2dgs', {}) else v(eff, 'a_2dgs', 'estimated_min')
    t_b = v(eff, 'b_dreamfusion', 'actual_time_min') if 'actual_time_min' in eff.get('b_dreamfusion', {}) else v(eff, 'b_dreamfusion', 'estimated_min')
    cc_raw = eff.get('c_coarse', {})
    cr_raw = eff.get('c_refine', {})
    cc = cc_raw.get('actual_time_min') or cc_raw.get('estimated_min', 0) or 0
    cr = cr_raw.get('actual_time_min') or cr_raw.get('estimated_min', 0) or 0
    t_c = round(cc + cr, 1) if isinstance(cc, (int,float)) and isinstance(cr, (int,float)) and (cc or cr) else 'N/A'
    lines.append(f"| 训练时间 (min ↓) | {t_a} | {t_b} | {t_c} |")
    lines.append(f"| 输入需求 | ~100张多视角照片 | 1句文本描述 | 1张RGBA照片 + 文本 |")
    lines.append(f"| Mesh文件 (MB) | {v(sizes,'mesh_a','size_mb')} | {v(sizes,'mesh_b','size_mb')} | {v(sizes,'mesh_c','size_mb')} |")
    lines.append("")
    lines.append("> ¹ B/C 无真实GT图像，PSNR不适用。")
    lines.append("> ² A 无文本Prompt，CLIP Text→Image不适用；CLIP 需要 `pip install open_clip_torch`。")
    lines.append("> CD = Chamfer Distance，越小越好。")
    lines.append("> NeRF→Mesh 为隐式场提取mesh的几何统计（无渲染）。Mesh→GS 为mesh转surfel的几何信息损失。")
    lines.append("> A 训练时间来自 TB iter_time 总和；B/C 来自 trial 目录名(@timestamp)到 CSV 修改时间的时间差。")

    with open(PATHS['out_md'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_clip', action='store_true')
    args = parser.parse_args()

    os.makedirs(PATHS['out_dir'], exist_ok=True)
    R = {}

    # 1. 几何准确度 — Chamfer Distance
    print("=" * 60)
    print("1. Geometry (Chamfer Distance)")
    pcs = {}
    for name in ['a', 'b', 'c']:
        pcs[name] = load_points(PATHS[f'pc_{name}'])
        if pcs[name] is not None:
            print(f"  {name}: {len(pcs[name])} points ({PATHS[f'pc_{name}']})")
        else:
            print(f"  {name}: NOT FOUND ({PATHS[f'pc_{name}']})")

    geo = {}
    for n1, n2, label in [('a','b','A↔B'), ('a','c','A↔C'), ('b','c','B↔C')]:
        if pcs[n1] is not None and pcs[n2] is not None:
            cd = chamfer_distance(pcs[n1], pcs[n2])
            geo[label] = round(cd, 4)
            print(f"  CD({label}): {cd:.4f}")
    R['geometry'] = geo

    # 2. 纹理细节
    print("\n" + "=" * 60)
    print("2. Texture")

    # A: PSNR
    if os.path.isdir(PATHS['a_renders']) and os.path.isdir(PATHS['a_gt']):
        psnr_res = compute_psnr_dir(PATHS['a_renders'], PATHS['a_gt'])
        if psnr_res:
            R['psnr_a'] = psnr_res
            print(f"  A PSNR: {psnr_res['PSNR']} dB ({psnr_res['num_pairs']} pairs)")
        else:
            print("  A PSNR: no images found")
    else:
        print("  A PSNR: no test renders dir")

    # B: CLIP
    if args.skip_clip:
        print("  CLIP: skipped")
        R['clip_b'] = {'note': 'skipped'}
        R['clip_c'] = {'note': 'skipped'}
    else:
        b_trial = find_trial_dir(PATHS['b_trial'])
        b_views = find_test_views(b_trial) if b_trial else []
        if b_views:
            clip_b = compute_clip_text_image(b_views, 'a detailed ceramic teapot with gold trim, photorealistic')
            if clip_b:
                R['clip_b'] = clip_b
                print(f"  B CLIP T→I: {clip_b['mean']} ({clip_b['num_views']} views)")
            else:
                R['clip_b'] = {'note': 'open_clip_torch not available'}
                print("  B CLIP: open_clip_torch not available")

        c_trial = find_trial_dir(PATHS['c_trial'])
        c_views = find_test_views(c_trial) if c_trial else []
        if c_views and os.path.exists(PATHS['c_input_img']):
            clip_c = compute_clip_image_image(c_views, PATHS['c_input_img'])
            if clip_c:
                R['clip_c'] = clip_c
                print(f"  C CLIP I→I: {clip_c['mean']} ({clip_c['num_views']} views)")
            else:
                R['clip_c'] = {'note': 'open_clip_torch not available'}
                print("  C CLIP: open_clip_torch not available")

    # 3. 计算效率
    print("\n" + "=" * 60)
    print("3. Efficiency")
    eff = {}

    # A: 从 TB iter_time 算实际训练时间
    a_time = read_2dgs_train_time('output/object_a')
    if a_time:
        eff['a_2dgs'] = {'actual_time_min': a_time, 'source': 'TB iter_time'}
        print(f"  A (2DGS): {a_time} min (TB iter_time)")
    else:
        eff['a_2dgs'] = {'estimated_min': 200, 'source': 'hardcoded estimate'}

    # B/C: 从目录时间戳 + CSV 修改时间算实际训练时间
    for name, d in [('b_dreamfusion', PATHS['csv_b']),
                     ('c_coarse', PATHS['csv_c_coarse']),
                     ('c_refine', PATHS['csv_c_refine'])]:
        steps = read_csv_steps(d)
        actual = read_threestudio_train_time(d) if steps else None
        entry = {}
        if steps:
            entry['total_steps'] = steps['total_steps']
        if actual:
            entry['actual_time_min'] = actual
            print(f"  {name}: {steps['total_steps']} steps, {actual} min (trial timestamp)")
        elif steps:
            entry['estimated_min'] = steps['estimated_min']
            print(f"  {name}: {steps['total_steps']} steps, ~{steps['estimated_min']} min (estimated)")
        if entry:
            eff[name] = entry
    R['efficiency'] = eff

    sizes = {}
    for name, p in [('gs_a', PATHS['gs_a']), ('gs_b', PATHS['gs_b']), ('gs_c', PATHS['gs_c']),
                     ('mesh_a', PATHS['mesh_a']), ('mesh_b', PATHS['mesh_b']), ('mesh_c', PATHS['mesh_c'])]:
        sz = file_size_mb(p)
        if sz is not None:
            sizes[name] = {'size_mb': sz}
            print(f"  {name}: {sz} MB")
    R['file_sizes'] = sizes

    # 4. Mesh→GS 转换保真度
    print("\n" + "=" * 60)
    print("4. Mesh→GS Fidelity")
    fidelity = {}
    for name, mesh_k, gs_k in [('b', 'mesh_b_for_gs', 'gs_b_surfel'),
                                ('c', 'mesh_c_for_gs', 'gs_c_surfel')]:
        r = compute_mesh_to_gs_fidelity(PATHS[mesh_k], PATHS[gs_k])
        if r:
            fidelity[name] = r
            print(f"  {name}: CD={r['chamfer_distance']}, normal_angle={r['normal_angle_mean_deg']}°, "
                  f"verts={r['mesh_vertices']}→surfels={r['gs_surfels']}, ratio={r['compression_ratio']}")
        else:
            print(f"  {name}: SKIP")
    R['mesh_to_gs_fidelity'] = fidelity

    # 5. NeRF→Mesh 几何统计
    print("\n" + "=" * 60)
    print("5. NeRF→Mesh Stats")
    mesh_stats = {}
    for name, path in [('b', PATHS['mesh_b_for_gs']), ('c', PATHS['mesh_c_for_gs'])]:
        s = obj_mesh_stats(path)
        if s:
            mesh_stats[name] = s
            print(f"  {name}: {s['num_vertices']} verts, {s['num_faces']} faces, "
                  f"bbox=({s['bbox_x']},{s['bbox_y']},{s['bbox_z']}), "
                  f"normals={'yes' if s['has_vertex_normals'] else 'no'}")
        else:
            print(f"  {name}: SKIP")
    R['nerf_to_mesh_stats'] = mesh_stats

    # 输出
    with open(PATHS['out_json'], 'w') as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {PATHS['out_json']}")

    generate_md(R)
    print(f"MD: {PATHS['out_md']}")

    # 终端摘要
    ta_v = eff.get('a_2dgs', {}).get('actual_time_min') or eff.get('a_2dgs', {}).get('estimated_min', '?')
    tb_v = eff.get('b_dreamfusion', {}).get('actual_time_min') or eff.get('b_dreamfusion', {}).get('estimated_min', '?')
    cc_v = eff.get('c_coarse', {}).get('actual_time_min') or eff.get('c_coarse', {}).get('estimated_min', 0) or 0
    cr_v = eff.get('c_refine', {}).get('actual_time_min') or eff.get('c_refine', {}).get('estimated_min', 0) or 0
    tc_v = round(cc_v + cr_v, 1) if (cc_v or cr_v) else 'N/A'

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Metric':<35} {'A (Multi-view)':<18} {'B (Text→3D)':<18} {'C (Image→3D)':<18}")
    print("-" * 89)
    print(f"{'CD vs A':<35} {'—':<18} {str(geo.get('A↔B','?')):<18} {str(geo.get('A↔C','?')):<18}")
    print(f"{'PSNR (dB)':<35} {str(R.get('psnr_a',{}).get('PSNR','N/A')):<18} {'N/A':<18} {'N/A':<18}")
    print(f"{'CLIP Score':<35} {'N/A':<18} {str(R.get('clip_b',{}).get('mean','N/A')):<18} {str(R.get('clip_c',{}).get('mean','N/A')):<18}")
    print(f"{'Train Time (min)':<35} {str(eff.get('a_2dgs',{}).get('estimated_min','?')):<18} {str(eff.get('b_dreamfusion',{}).get('estimated_min','?')):<18} {str(tc_v):<18}")


if __name__ == "__main__":
    main()
