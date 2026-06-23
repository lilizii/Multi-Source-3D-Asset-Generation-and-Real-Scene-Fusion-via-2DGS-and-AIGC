"""
清理 2DGS PLY — 去除背景碎片，只保留中央物体
用法:
  python scripts/utils/filter_ply.py --input <in.ply> --output <out.ply> --method percentile --keep 0.9
  python scripts/utils/filter_ply.py --input <in.ply> --output <out.ply> --method radius --radius 0.5
"""
import numpy as np, os, sys, argparse
from plyfile import PlyData, PlyElement

def filter_by_percentile(pts, keep_ratio=0.9):
    """保留距离中心最近的 keep_ratio 比例的点"""
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    threshold = np.percentile(dist, keep_ratio * 100)
    return dist <= threshold

def filter_by_sigma(pts, n_sigma=3.0):
    """保留中心 n_sigma 标准差内的点"""
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    mean_d = dist.mean()
    std_d = dist.std()
    return dist <= mean_d + n_sigma * std_d

def filter_by_radius(pts, radius=1.0):
    """保留指定半径内的点"""
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    return dist <= radius

def filter_by_clusters(pts, keep_largest=True, threshold=5.0):
    """保留最大的连通分量 (基于 KNN 图)"""
    try:
        from scipy.spatial import KDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        print("  scipy 未安装, 回退到 percentile")
        return filter_by_percentile(pts, 0.9)

    k = min(20, len(pts) - 1)
    tree = KDTree(pts)
    dist, idx = tree.query(pts, k=k + 1)
    avg_dist = np.mean(dist[:, 1:])

    threshold_dist = avg_dist * threshold
    print(f"  avg NN dist={avg_dist:.4f}, threshold={threshold_dist:.4f}")

    # 稀疏构建: 只记录 close neighbors
    edges = []
    for i in range(len(pts)):
        neighbors = idx[i, 1:]
        d = dist[i, 1:]
        close = d < threshold_dist
        for j in neighbors[close]:
            if i < j:  # 避免重复边
                edges.append((i, j))

    if not edges:
        print("  WARNING: 无边, 返回全部点")
        return np.ones(len(pts), dtype=bool)

    edges = np.array(edges)
    row, col = edges[:, 0], edges[:, 1]
    data = np.ones(len(row))
    n = len(pts)
    graph = csr_matrix((data, (row, col)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)

    counts = np.bincount(labels)
    print(f"  分量数: {n_components}, 大小: {sorted(counts)[-5:]}")
    largest_label = np.argmax(counts)
    return labels == largest_label


def main():
    parser = argparse.ArgumentParser(description="清理 PLY 碎片")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="percentile",
                        choices=["percentile", "sigma", "radius", "clusters"])
    parser.add_argument("--keep", type=float, default=0.95, help="percentile 保留比例")
    parser.add_argument("--sigma", type=float, default=3.0, help="sigma 倍数")
    parser.add_argument("--radius", type=float, default=1.0, help="绝对半径")
    parser.add_argument("--cluster-threshold", type=float, default=5.0,
                        help="clusters: NN距离倍数阈值 (越大保留越多)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found"); return

    ply = PlyData.read(args.input)
    verts = ply['vertex'].data
    pts = np.column_stack([verts['x'], verts['y'], verts['z']])
    print(f"原始: {len(pts)} surfels")

    if args.method == "percentile":
        mask = filter_by_percentile(pts, args.keep)
    elif args.method == "sigma":
        mask = filter_by_sigma(pts, args.sigma)
    elif args.method == "radius":
        mask = filter_by_radius(pts, args.radius)
    elif args.method == "clusters":
        mask = filter_by_clusters(pts, threshold=args.cluster_threshold)
    else:
        mask = np.ones(len(pts), dtype=bool)

    filtered = verts[mask]
    print(f"过滤后: {len(filtered)} surfels ({len(filtered)/len(pts)*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    PlyData([PlyElement.describe(filtered, 'vertex')]).write(args.output)
    print(f"已保存: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
