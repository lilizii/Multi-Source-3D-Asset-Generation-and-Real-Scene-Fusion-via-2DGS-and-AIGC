"""
2DGS CUDA 原生渲染 GS 合并场景
================================
用法:
  python scripts/utils/gs_native_render.py \
      --gs_ply output/fusion/full_scene.ply \
      --bg_source data/mipnerf360/room \
      --output output/fusion --num_views 120
"""
import os, sys, torch, numpy as np, cv2, argparse, copy, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '2d-gaussian-splatting'))
from gaussian_renderer import GaussianModel, render
from utils.render_utils import generate_path


def build_synthetic_cameras(gs_ply_path, num_views=120, max_res=600):
    """基于 PLY 包围盒构建轨道相机"""
    from plyfile import PlyData
    ply = PlyData.read(gs_ply_path)
    v = ply['vertex'].data
    pts = np.column_stack([v['x'], v['y'], v['z']])
    center = np.array([pts[:,0].mean(), pts[:,1].mean(), pts[:,2].mean()])
    extent = max(pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min())
    radius = extent * 1.5
    print(f"  合成轨迹: center=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}) radius={radius:.1f}")

    hfov = 60 * np.pi / 180
    fx = (max_res / 2) / math.tan(hfov / 2)
    proj = torch.zeros(4, 4, device='cuda')
    proj[0, 0] = 2 * fx / max_res
    proj[1, 1] = 2 * fx / max_res
    proj[2, 2] = 0.0
    proj[3, 2] = 1.0

    class Cam:
        pass

    seed_cams = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * 0.1 * np.sin(angle * 2)
        z = center[2] + radius * np.sin(angle)
        look_dir = center - np.array([x, y, z])
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)
        up = np.array([0., 1., 0.])
        right = np.cross(up, look_dir)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(look_dir, right)
        c2w = np.eye(4)
        c2w[0, :3] = right
        c2w[1, :3] = up
        c2w[2, :3] = look_dir
        c2w[:3, 3] = [x, y, z]
        cam = Cam()
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.image_height = max_res
        cam.image_width = max_res
        cam.FoVx = hfov
        cam.FoVy = hfov
        cam.projection_matrix = proj
        seed_cams.append(cam)
    return seed_cams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_ply", default="output/fusion/full_scene.ply")
    parser.add_argument("--bg_source", default="", help="Mip-NeRF 360 场景路径")
    parser.add_argument("--output", default="output/fusion")
    parser.add_argument("--num_views", type=int, default=120)
    parser.add_argument("--max_res", type=int, default=600)
    args = parser.parse_args()

    if not os.path.exists(args.gs_ply):
        print(f"ERROR: {args.gs_ply} not found"); return

    out_dir = os.path.join(args.output, 'gs_native_frames')
    os.makedirs(out_dir, exist_ok=True)

    gaussians = GaussianModel(3)
    gaussians.load_ply(args.gs_ply)
    print(f"加载: {len(gaussians.get_xyz)} surfels, sh_degree={gaussians.active_sh_degree}")

    bg = torch.tensor([0., 0., 0.], dtype=torch.float32, device='cuda')

    class Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        depth_ratio = 0.0
        debug = False

    # --- 获取相机 ---
    train_cams = None

    if args.bg_source and os.path.isdir(args.bg_source):
        try:
            from scene import Scene
            ds = type('Args', (), {})()
            ds.source_path = os.path.abspath(args.bg_source)
            ds.model_path = ''
            ds.sh_degree = 3
            ds.images = 'images'
            ds.resolution = -1
            ds.white_background = False
            ds.data_device = 'cuda'
            ds.eval = False
            scene = Scene(ds, GaussianModel(3), shuffle=False, resolution_scales=[1])
            train_cams = scene.getTrainCameras()
            print(f"  COLMAP 相机: {len(train_cams)} 个")
        except Exception as e:
            print(f"  COLMAP 加载失败: {e}")

    if train_cams is None:
        print("  改用合成轨道相机")
        train_cams = build_synthetic_cameras(args.gs_ply, args.num_views, args.max_res)

    cameras = generate_path(train_cams, n_frames=args.num_views)
    print(f"渲染 {len(cameras)} 帧...")

    for i, cam in enumerate(cameras):
        oh, ow = cam.image_height, cam.image_width
        s = min(args.max_res / max(oh, ow), 1.0)
        cam.image_height, cam.image_width = int(oh*s), int(ow*s)
        with torch.no_grad():
            pkg = render(cam, gaussians, Pipe(), bg)
            img = torch.clamp(pkg['render'], 0., 1.)
            img_np = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        cam.image_height, cam.image_width = oh, ow
        cv2.imwrite(os.path.join(out_dir, f'frame_{i:04d}.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if i % 30 == 0:
            print(f'  {i}/{len(cameras)}')

    video_path = os.path.join(args.output, 'gs_native_video.mp4')
    import subprocess
    r = subprocess.run(['ffmpeg','-y','-framerate','24',
        '-i',os.path.join(out_dir,'frame_%04d.png'),
        '-c:v','libx264','-pix_fmt','yuv420p','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2',
        video_path], capture_output=True, text=True)
    if r.returncode == 0:
        print(f'Done: {video_path}')
    else:
        print(f'帧已保存 {out_dir}/')


if __name__ == '__main__':
    main()
