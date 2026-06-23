"""
将 threestudio 验证图 (彩色+深度+alpha三栏拼接) 裁剪重组
==========================================================
输入: it{step}-{view}.png  (1536×512, 三栏各512)
输出: it{step}.png          (2048×512, 四视角从左到右拼接)

用法:
  python scripts/utils/stitch_validation.py
  python scripts/utils/stitch_validation.py --steps 200,1000,2000,5000,10000
"""
import os, sys, argparse, re
from collections import defaultdict
from PIL import Image


def find_trial_dir(search_dir):
    """找到 threestudio trial 目录"""
    if not os.path.isdir(search_dir):
        return None
    for item in sorted(os.listdir(search_dir)):
        item_path = os.path.join(search_dir, item)
        if os.path.isdir(item_path) and '@' in item:
            if os.path.isdir(os.path.join(item_path, 'save')):
                return item_path
    return None


def crop_color(img):
    """裁出左边 1/3 的彩色图"""
    w = img.width // 3
    return img.crop((0, 0, w, img.height))


def process_trial(trial_dir, output_dir, tag, target_steps=None):
    """处理单个 trial 的 save/ 目录"""
    save_dir = os.path.join(trial_dir, 'save')
    if not os.path.isdir(save_dir):
        print(f"  {tag}: save/ not found")
        return

    # 收集按迭代分组: {step: {view: path}}
    groups = defaultdict(dict)
    for fname in os.listdir(save_dir):
        m = re.match(r'it(\d+)-(\d)\.png$', fname)
        if not m:
            continue
        step = int(m.group(1))
        view = int(m.group(2))
        if view not in (0, 1, 2, 3):
            continue
        if target_steps and step not in target_steps:
            continue
        groups[step][view] = os.path.join(save_dir, fname)

    if not groups:
        print(f"  {tag}: no images found")
        return

    os.makedirs(output_dir, exist_ok=True)
    for step in sorted(groups):
        views = groups[step]
        if len(views) < 4:
            # 补缺失视角提示
            missing = {0,1,2,3} - set(views.keys())
            if missing:
                print(f"  {tag} it{step}: missing views {missing}, skipped")
            continue

        # 裁出每个视角的彩色部分
        strips = []
        for v in range(4):
            img = Image.open(views[v]).convert('RGB')
            strips.append(crop_color(img))

        # 水平拼接
        total_w = sum(s.width for s in strips)
        h = strips[0].height
        result = Image.new('RGB', (total_w, h))
        x = 0
        for s in strips:
            result.paste(s, (x, 0))
            x += s.width

        out_path = os.path.join(output_dir, f'it{step}.png')
        result.save(out_path)
        print(f"  {tag} it{step}: {strips[0].width}×{strips[0].height} ×4 → {total_w}×{h}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=str, default='',
                        help='逗号分隔的 step 列表, 默认处理全部')
    args = parser.parse_args()

    target_steps = None
    if args.steps:
        target_steps = {int(s.strip()) for s in args.steps.split(',') if s.strip()}

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    tasks = [
        ('object_c_coarse', os.path.join(base, 'output', 'object_c', 'magic123-coarse-sd')),
        ('object_c_refine', os.path.join(base, 'output', 'object_c', 'magic123-refine-sd')),
    ]

    for tag, search_dir in tasks:
        trial = find_trial_dir(search_dir)
        if not trial:
            print(f"{tag}: trial dir not found in {search_dir}")
            continue
        out_dir = os.path.join(base, 'output', 'validation_strips', tag)
        process_trial(trial, out_dir, tag, target_steps)

    print("\nDone. Output: output/validation_strips/")


if __name__ == '__main__':
    main()
