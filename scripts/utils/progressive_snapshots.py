"""
训练过程渐进快照 + 360° 环绕视频生成
======================================
对每个物体和背景场景：
  1. 从不同训练阶段的 checkpoint 渲染同一组视角 → 产生"训练渐进图"
  2. 从最终模型渲染 360° 环绕视频
  3. 合成多画面对比视频

2DGS: 从 checkpoint 渲染 (render.py 复用)
threestudio: 从 save/ 目录收集验证图 + 复制已有 360° 视频

用法:
  # 生成所有物体的渐进快照和对比视频
  python scripts/utils/progressive_snapshots.py --mode all --type all --output output/snapshots
"""

import argparse
import os
import sys
import glob
import numpy as np
import subprocess
import shutil


# ========== 查找 threestudio trial 目录的辅助函数 ==========

def find_trial_dir(search_dir):
    """在 search_dir 下找到 threestudio trial 目录"""
    if not os.path.isdir(search_dir):
        return None
    for item in sorted(os.listdir(search_dir)):
        item_path = os.path.join(search_dir, item)
        if os.path.isdir(item_path) and ('@' in item):
            # 检查是否有 ckpts/ 子目录
            if os.path.isdir(os.path.join(item_path, 'ckpts')):
                return item_path
    return None


# ========== 2DGS: 从多阶段 checkpoint 渲染渐进图 ==========

def render_2dgs_progression(source_path, model_path, output_dir, name,
                            checkpoints=(1000, 3000, 7000, 15000, 30000)):
    """从不同 checkpoint 渲染同一固定视角组"""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                     '2d-gaussian-splatting'))
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.render_utils import save_img_u8

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集 (load_iteration=-1 加载全部视角信息)
    from types import SimpleNamespace
    dataset = SimpleNamespace(
        source_path=os.path.abspath(source_path),
        model_path=os.path.abspath(model_path),
        sh_degree=3, images="images", resolution=-1,
        white_background=False, data_device="cuda", eval=False)

    class PipeArgs:
        convert_SHs_python = False
        compute_cov3D_python = False
        depth_ratio = 0.0
        debug = False
    pipe = PipeArgs()

    # 加载完整模型以获取相机
    gaussians_full = GaussianModel(3)
    try:
        scene_full = Scene(dataset, gaussians_full, load_iteration=-1, shuffle=False)
    except Exception as e:
        print(f"  ERROR loading scene: {e}")
        print(f"  source_path={source_path}, model_path={model_path}")
        return

    train_cams = scene_full.getTrainCameras()
    if len(train_cams) == 0:
        print(f"  No train cameras found")
        return

    # 均匀采样 8 个视角
    snapshot_indices = np.linspace(0, len(train_cams) - 1, 8, dtype=int)
    snapshot_cams = [train_cams[i] for i in snapshot_indices]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_rows = []

    for ckpt_iter in checkpoints:
        gaussians = GaussianModel(3)
        try:
            scene = Scene(dataset, gaussians, load_iteration=ckpt_iter, shuffle=False)
        except Exception:
            print(f"  [skip] checkpoint {ckpt_iter} not found")
            continue

        row_images = []
        for j, cam in enumerate(snapshot_cams):
            try:
                render_pkg = render(cam, gaussians, pipe, background)
                img = torch.clamp(render_pkg["render"], 0.0, 1.0)
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                save_path = os.path.join(output_dir,
                                         f"{name}_iter{ckpt_iter:05d}_view{j:02d}.png")
                save_img_u8(img_np, save_path)
                row_images.append(img_np)
            except Exception as e:
                print(f"  ERROR rendering iter={ckpt_iter} view={j}: {e}")
                # 用黑色填充
                if row_images:
                    row_images.append(np.zeros_like(row_images[0]))
                continue

            if j == 0:
                print(f"  [{name}] iter={ckpt_iter:05d} rendered {len(snapshot_cams)} views")

        if row_images:
            # 确保所有图片尺寸一致
            target_h = min(img.shape[0] for img in row_images)
            target_w = min(img.shape[1] for img in row_images)
            row_images = [img[:target_h, :target_w] for img in row_images]
            all_rows.append(np.concatenate(row_images, axis=1))

    # 生成"训练渐进图"
    if all_rows:
        progression_img = np.concatenate(all_rows, axis=0)
        prog_path = os.path.join(output_dir, f"{name}_training_progression.png")
        save_img_u8(progression_img, prog_path)
        print(f"  渐进图已保存: {prog_path}")

    del gaussians_full, scene_full
    torch.cuda.empty_cache()


def render_2dgs_360_video(source_path, model_path, output_dir, name):
    """从最终模型渲染 360° 环绕视频（调用 2DGS render.py）"""
    render_script = os.path.join(os.path.dirname(__file__), '..', '..',
                                 '2d-gaussian-splatting', 'render.py')

    cmd = [
        sys.executable, render_script,
        "-s", os.path.abspath(source_path),
        "-m", os.path.abspath(model_path),
        "--render_path",
        "--skip_train",
        "--skip_test",
        "--skip_mesh",
    ]
    print(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(render_script),
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: render.py 返回非零: {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")

    # 复制视频
    traj_src = os.path.join(model_path, "traj", "ours_30000")
    if os.path.exists(traj_src):
        for f in os.listdir(traj_src):
            if f.endswith('.mp4'):
                src = os.path.join(traj_src, f)
                dst = os.path.join(output_dir, f"{name}_{f}")
                shutil.copy2(src, dst)
                print(f"  环绕视频: {dst}")


# ========== threestudio: 收集渐进图 + 360° 视频 ==========

def collect_threestudio_progression(trial_dir, output_dir, name):
    """threestudio: 从 save/ 目录收集 it*-0.png 作为训练渐进图"""
    os.makedirs(output_dir, exist_ok=True)

    val_pattern = os.path.join(trial_dir, "save", "it*-0.png")
    val_files = sorted(glob.glob(val_pattern))
    if not val_files:
        # 也尝试在 save/it*-test/ 下找
        for d in sorted(glob.glob(os.path.join(trial_dir, "save", "it*-test"))):
            pngs = sorted(glob.glob(os.path.join(d, "*.png")))
            if pngs:
                val_files.append(pngs[0])

    if len(val_files) > 8:
        indices = np.linspace(0, len(val_files) - 1, 8, dtype=int)
        val_files = [val_files[i] for i in indices]

    copied = []
    for vf in val_files:
        basename = os.path.basename(vf)
        dst = os.path.join(output_dir, f"{name}_{basename}")
        shutil.copy2(vf, dst)
        copied.append(dst)

    print(f"  [{name}] 收集了 {len(copied)} 张渐进验证图")
    return copied


def collect_threestudio_360_video(trial_dir, output_dir, name):
    """threestudio: 复制已有的 360° 测试视频"""
    os.makedirs(output_dir, exist_ok=True)

    # 找 it*-test.mp4
    mp4_pattern = os.path.join(trial_dir, "save", "it*-test.mp4")
    mp4_files = sorted(glob.glob(mp4_pattern))
    for mp4f in mp4_files:
        basename = os.path.basename(mp4f)
        dst = os.path.join(output_dir, f"{name}_{basename}")
        shutil.copy2(mp4f, dst)
        print(f"  [{name}] 环绕视频: {dst}")
        return dst

    # fallback: 从 test 帧合成
    test_dirs = sorted(glob.glob(os.path.join(trial_dir, "save", "it*-test")))
    if test_dirs:
        test_dir = test_dirs[-1]
        frames = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        if frames:
            try:
                import imageio
                video_path = os.path.join(output_dir, f"{name}_360video.mp4")
                writer = imageio.get_writer(video_path, fps=30)
                for f in frames[:180]:  # 最多180帧
                    writer.append_data(imageio.imread(f))
                writer.close()
                print(f"  [{name}] 合成环绕视频: {video_path}")
                return video_path
            except ImportError:
                print(f"  [{name}] imageio 未安装，跳过视频合成")

    print(f"  [{name}] 未找到 360° 视频")
    return None


# ========== 合成对比视频 ==========

def create_comparison_video(video_paths, output_path, layout="2x2"):
    """将多个 360° 视频合成为多画面并排视频"""
    try:
        import cv2
    except ImportError:
        print("  opencv-python 未安装，跳过合成视频")
        return

    readers = {}
    for name, path in video_paths.items():
        if path and os.path.exists(path):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                readers[name] = cap

    if len(readers) < 2:
        print(f"  至少需要 2 个视频，目前只有 {len(readers)} 个")
        return

    min_frames = min(int(r.get(cv2.CAP_PROP_FRAME_COUNT)) for r in readers.values())
    fps = 30
    frame_w, frame_h = 640, 480

    names = list(readers.keys())
    n = len(names)
    if n <= 2:
        cols, rows = n, 1
    elif n <= 4:
        cols, rows = 2, 2
    else:
        cols, rows = 3, 2

    out_w, out_h = frame_w * cols, frame_h * rows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.abspath(output_path)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    for frame_idx in range(min(int(min_frames), 360)):  # 最多360帧=12秒
        row_frames_all = []
        for r in range(rows):
            col_frames = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(names):
                    ret, frame = readers[names[idx]].read()
                    if ret:
                        frame = cv2.resize(frame, (frame_w, frame_h))
                    else:
                        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                    cv2.putText(frame, names[idx], (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                col_frames.append(frame)
            row_frames_all.append(np.concatenate(col_frames, axis=1))
        combined = np.concatenate(row_frames_all, axis=0)
        writer.write(combined)

    writer.release()
    for r in readers.values():
        r.release()
    print(f"  对比视频已保存: {out_path}")


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="渐进快照 + 环绕视频生成")
    parser.add_argument("--mode", default="all",
                        choices=["progression", "360video", "all"],
                        help="生成模式")
    parser.add_argument("--type", default="all",
                        choices=["2dgs", "threestudio", "all"],
                        help="模型类型")

    # 2DGS 参数
    parser.add_argument("--source", default="")
    parser.add_argument("--model", default="")

    # threestudio 参数
    parser.add_argument("--trial_dir", default="")

    # 通用
    parser.add_argument("--name", default="object", help="输出文件前缀")
    parser.add_argument("--output", default="output/snapshots",
                        help="输出目录")
    parser.add_argument("--compose", action="store_true",
                        help="将所有视频合成为多画面对比视频")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ========== 预定义的 2DGS 物体和背景 ==========
    objects_2dgs = [
        {
            'source': 'data/object_a',
            'model': 'output/object_a',
            'name': 'object_a',
        },
        # 背景：自动检测场景
    ]

    # 检测背景场景
    bg_scenes = []
    for d in sorted(os.listdir('output')):
        bg_match = d.startswith('background_')
        if bg_match and os.path.isdir(os.path.join('output', d)):
            scene_name = d.replace('background_', '')
            m360_path = f'data/mipnerf360/{scene_name}'
            if os.path.isdir(m360_path):
                bg_scenes.append({
                    'source': m360_path,
                    'model': f'output/{d}',
                    'name': f'background_{scene_name}',
                })

    # ========== 预定义的 threestudio 物体 ==========
    objects_ts = [
        {
            'search_dir': 'output/object_b/dreamfusion-sd',
            'name': 'object_b',
        },
        {
            'search_dir': 'output/object_c/magic123-refine-sd',
            'name': 'object_c',
        },
    ]

    all_videos = {}

    if args.type in ("2dgs", "all"):
        # 处理 2DGS 物体
        for obj in objects_2dgs:
            if not os.path.isdir(obj['source']) or not os.path.isdir(obj['model']):
                print(f"  SKIP {obj['name']}: source or model dir not found")
                continue

            if args.mode in ("progression", "all"):
                print(f"\n=== {obj['name']}: 2DGS 训练渐进图 ===")
                render_2dgs_progression(obj['source'], obj['model'],
                                       args.output, obj['name'])

            if args.mode in ("360video", "all"):
                print(f"\n=== {obj['name']}: 2DGS 360° 环绕视频 ===")
                # 先检查是否已有视频
                existing_mp4 = os.path.join(args.output, f"{obj['name']}_render_traj_color.mp4")
                if not os.path.exists(existing_mp4):
                    # 检查 model 目录里是否有
                    traj_mp4 = os.path.join(obj['model'], 'traj', 'ours_30000',
                                           'render_traj_color.mp4')
                    if os.path.exists(traj_mp4):
                        shutil.copy2(traj_mp4, existing_mp4)
                        print(f"  复用已有: {traj_mp4}")
                    else:
                        render_2dgs_360_video(obj['source'], obj['model'],
                                             args.output, obj['name'])
                else:
                    print(f"  视频已存在: {existing_mp4}")
                all_videos[obj['name']] = existing_mp4

        # 处理背景场景
        for bg in bg_scenes:
            if args.mode in ("progression", "all"):
                print(f"\n=== {bg['name']}: 2DGS 训练渐进图 ===")
                render_2dgs_progression(bg['source'], bg['model'],
                                       args.output, bg['name'])

            if args.mode in ("360video", "all"):
                print(f"\n=== {bg['name']}: 2DGS 360° 环绕视频 ===")
                existing_mp4 = os.path.join(args.output,
                                           f"{bg['name']}_render_traj_color.mp4")
                if not os.path.exists(existing_mp4):
                    traj_mp4 = os.path.join(bg['model'], 'traj', 'ours_30000',
                                           'render_traj_color.mp4')
                    if os.path.exists(traj_mp4):
                        shutil.copy2(traj_mp4, existing_mp4)
                        print(f"  复用已有: {traj_mp4}")
                    else:
                        render_2dgs_360_video(bg['source'], bg['model'],
                                             args.output, bg['name'])
                else:
                    print(f"  视频已存在: {existing_mp4}")
                all_videos[bg['name']] = existing_mp4

    if args.type in ("threestudio", "all"):
        for obj in objects_ts:
            trial_dir = find_trial_dir(obj['search_dir'])
            if not trial_dir:
                print(f"  SKIP {obj['name']}: trial dir not found in {obj['search_dir']}")
                continue
            print(f"  {obj['name']} trial: {trial_dir}")

            if args.mode in ("progression", "all"):
                print(f"\n=== {obj['name']}: threestudio 训练渐进图 ===")
                collect_threestudio_progression(trial_dir, args.output, obj['name'])

            if args.mode in ("360video", "all"):
                print(f"\n=== {obj['name']}: threestudio 360° 环绕视频 ===")
                video_path = collect_threestudio_360_video(trial_dir, args.output, obj['name'])
                if video_path:
                    all_videos[obj['name']] = video_path

    # ========== 合成对比视频 ==========
    if args.compose and len(all_videos) >= 2:
        print(f"\n=== 合成多画面对比视频 ({len(all_videos)} 个) ===")
        create_comparison_video(
            all_videos,
            os.path.join(args.output, "all_objects_comparison.mp4"),
            layout="2x2"
        )

    print(f"\n全部完成! 输出目录: {args.output}/")
    for f in sorted(os.listdir(args.output)):
        size = os.path.getsize(os.path.join(args.output, f))
        print(f"  {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
