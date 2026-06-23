"""
2DGS 训练 + SwanLab/WandB 日志记录
=====================================
解决 2DGS 原生只支持 TensorBoard 的问题。
在训练过程中同步将 Loss 曲线、PSNR 等指标和渲染图推送到 SwanLab/WandB。

用法:
  # 使用 SwanLab
  python scripts/utils/train_2dgs_with_swanlab.py \
      -s data/object_a -m output/object_a --use_swanlab

  # 使用 WandB
  python scripts/utils/train_2dgs_with_swanlab.py \
      -s data/object_a -m output/object_a --use_wandb

同时保留原生 TensorBoard 日志。
"""

import os
import sys
import torch
import uuid
import time
import numpy as np
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

# 将 2DGS 加入 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "2d-gaussian-splatting"))

# 尝试导入 TensorBoard (兼容)
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F811
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    SummaryWriter = None

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr, render_net_image


def setup_logger(args, model_path):
    """初始化 SwanLab 或 WandB logger"""
    logger = None
    logger_type = None

    if getattr(args, 'use_swanlab', False):
        try:
            import swanlab
            exp_name = getattr(args, 'exp_name', None) or os.path.basename(model_path)
            swanlab.init(
                project="cv-hw3-task1",
                experiment_name=exp_name,
                config={
                    "iterations": args.iterations,
                    "source_path": args.source_path,
                    "model_path": model_path,
                }
            )
            logger = swanlab
            logger_type = "swanlab"
            print(f"[SwanLab] 已初始化, 实验: {exp_name}")
        except ImportError:
            print("[WARNING] swanlab 未安装, 回退到 TensorBoard only")
        except Exception as e:
            print(f"[WARNING] SwanLab 初始化失败: {e}")

    if getattr(args, 'use_wandb', False) and logger is None:
        try:
            import wandb
            exp_name = getattr(args, 'exp_name', None) or os.path.basename(model_path)
            wandb.init(
                project="cv-hw3-task1",
                name=exp_name,
                config={
                    "iterations": args.iterations,
                    "source_path": args.source_path,
                    "model_path": model_path,
                }
            )
            logger = wandb
            logger_type = "wandb"
            print(f"[WandB] 已初始化, 实验: {exp_name}")
        except ImportError:
            print("[WARNING] wandb 未安装, 回退到 TensorBoard only")
        except Exception as e:
            print(f"[WARNING] WandB 初始化失败: {e}")

    return logger, logger_type


def log_scalar(logger, logger_type, name, value, step):
    """向 logger 记录标量"""
    if logger is None:
        return
    try:
        if logger_type == "swanlab":
            logger.log({name: value}, step=step)
        elif logger_type == "wandb":
            logger.log({name: value}, step=step)
    except Exception:
        pass


def log_image(logger, logger_type, name, image_np, step):
    """
    向 logger 记录图像
    image_np: numpy array (H, W, C) 或 (C, H, W), float or uint8
    """
    if logger is None:
        return
    try:
        # 统一为 (H, W, C), uint8
        if image_np.ndim == 3 and image_np.shape[0] <= 4:
            image_np = image_np.transpose(1, 2, 0)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        if logger_type == "swanlab":
            logger.log({name: image_np}, step=step)
        elif logger_type == "wandb":
            import wandb
            logger.log({name: wandb.Image(image_np)}, step=step)
    except Exception:
        pass


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, logger, logger_type):
    """修改版训练循环, 增加 SwanLab/WandB 日志"""
    first_iter = 0
    tb_writer = SummaryWriter(dataset.model_path) if TENSORBOARD_FOUND else None

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            # ======== SwanLab/WandB 记录训练指标 ========
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "distort": f"{ema_dist_for_log:.5f}",
                    "normal": f"{ema_normal_for_log:.5f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

                # 记录到 SwanLab/WandB
                log_scalar(logger, logger_type, "train/loss_l1", ema_loss_for_log, iteration)
                log_scalar(logger, logger_type, "train/loss_distortion", ema_dist_for_log, iteration)
                log_scalar(logger, logger_type, "train/loss_normal", ema_normal_for_log, iteration)
                log_scalar(logger, logger_type, "train/num_points", len(gaussians.get_xyz), iteration)

            if iteration == opt.iterations:
                progress_bar.close()

            # TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            # ======== 验证集评估与日志 ========
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                validation_configs = (
                    {'name': 'test', 'cameras': scene.getTestCameras()},
                    {'name': 'train',
                     'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                 for idx in range(5, 30, 5)]}
                )

                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0.0
                        psnr_test = 0.0
                        for idx, viewpoint in enumerate(config['cameras']):
                            render_pkg_val = render(viewpoint, scene.gaussians, pipe, background)
                            image_val = torch.clamp(render_pkg_val["render"], 0.0, 1.0).to("cuda")
                            gt_image_val = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                            l1_test += l1_loss(image_val, gt_image_val).mean().double()
                            psnr_test += _psnr(image_val, gt_image_val).mean().double()

                            # 记录前5张渲染图到 SwanLab/WandB
                            if idx < 5 and iteration == testing_iterations[-1]:
                                img_np = image_val.permute(1, 2, 0).cpu().numpy()
                                log_image(logger, logger_type,
                                          f"{config['name']}/render_view_{idx}", img_np, iteration)

                            if tb_writer and idx < 5:
                                from utils.general_utils import colormap
                                depth = render_pkg_val["surf_depth"]
                                depth = colormap(depth.cpu().numpy()[0], cmap='turbo') / 255.0
                                tb_writer.add_images(
                                    f"{config['name']}_view_{idx}/depth", depth[None], iteration)
                                tb_writer.add_images(
                                    f"{config['name']}_view_{idx}/render", image_val[None], iteration)

                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])

                        # ======== SwanLab/WandB 记录验证指标 ========
                        log_scalar(logger, logger_type,
                                   f"val/{config['name']}_psnr", psnr_test.item(), iteration)
                        log_scalar(logger, logger_type,
                                   f"val/{config['name']}_l1", l1_test.item(), iteration)

                        print(f"\n[ITER {iteration}] {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.2f}")

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull,
                        scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)

    # 训练结束, 关闭 logger
    if logger_type == "wandb" and logger is not None:
        import wandb
        wandb.finish()

    print("\n训练完成.")


if __name__ == "__main__":
    parser = ArgumentParser(description="2DGS + SwanLab/WandB 训练")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--use_swanlab", action="store_true", default=False,
                        help="启用 SwanLab 日志")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="启用 WandB 日志")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名称 (SwanLab/WandB)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)

    # 初始化远程 logger
    logger, logger_type = None, None
    if args.use_swanlab or args.use_wandb:
        # 确保 model_path 存在
        if not args.model_path:
            args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
        logger, logger_type = setup_logger(args, args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint,
             logger, logger_type)
