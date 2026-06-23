#!/bin/bash
# =============================================================================
# 02_prepare_object_b.sh - 物体B: 文本到3D生成 (threestudio DreamFusion/SDI)
#
# 使用说明:
#   修改下方 TEXT_PROMPT 为你想要的物体描述
#   运行: bash scripts/02_prepare_object_b.sh
#
# 推荐硬件: >= 12GB VRAM (使用 SD); >= 24GB VRAM (使用 DeepFloyd IF)
# =============================================================================
set -e

# ========== 配置 ==========
TEXT_PROMPT="a detailed ceramic teapot with gold trim, photorealistic"
USE_HIGH_QUALITY=false  # true=DeepFloyd IF (需要>15GB VRAM), false=Stable Diffusion (6GB)

OUTPUT_DIR="../output/object_b"
CONFIG_SD="configs/dreamfusion-sd.yaml"
CONFIG_IF="configs/dreamfusion-if.yaml"

echo "============================================"
echo "Step 2: 物体B - 文本到3D生成 (threestudio)"
echo "提示词: ${TEXT_PROMPT}"
echo "============================================"

cd threestudio
mkdir -p ../output/object_b

# HF 统一走数据盘 + 镜像
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# ---------- Step 2.1: 训练 (Coarse Stage - NeRF) ----------
echo "[2.1] 开始文本到3D训练 (NeRF stage)..."
if [ "$USE_HIGH_QUALITY" = true ]; then
    echo "使用 DeepFloyd IF (高质量模式)"
    python launch.py \
        --config "${CONFIG_IF}" \
        --train \
        --gpu 0 \
        system.prompt_processor.prompt="${TEXT_PROMPT}" \

else
    echo "使用 Stable Diffusion 1.5 (标准模式)"
    python launch.py \
        --config "${CONFIG_SD}" \
        --train \
        --gpu 0 \
        system.prompt_processor.prompt="${TEXT_PROMPT}" \
        system.prompt_processor.pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
        system.guidance.pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"

fi

echo "训练完成! 输出在 outputs/dreamfusion-sd/ 或 outputs/dreamfusion-if/"

# ---------- Step 2.2: 导出 Mesh ----------
echo "[2.2] 导出Mesh..."
# 找到最新训练的 trial 目录
TRIAL_DIR=$(ls -td outputs/dreamfusion-sd/*/ 2>/dev/null | head -1)
if [ -z "$TRIAL_DIR" ]; then
    TRIAL_DIR=$(ls -td outputs/dreamfusion-if/*/ 2>/dev/null | head -1)
fi

if [ -n "$TRIAL_DIR" ]; then
    CKPT_PATH="${TRIAL_DIR}ckpts/last.ckpt"
    CONFIG_PATH="${TRIAL_DIR}configs/parsed.yaml"

    python launch.py \
        --config "${CONFIG_PATH}" \
        --export \
        --gpu 0 \
        resume="${CKPT_PATH}" \
        system.exporter_type=mesh-exporter \
        system.exporter.fmt=obj \
        system.geometry.isosurface_resolution=128 \
        system.exporter.save_uv=false \
        system.exporter.save_texture=true

    # 复制导出的 mesh (threestudio 输出到 trial_dir/save/model.obj 或 outputs/...)
    OBJ_FILE=$(find "${TRIAL_DIR}/save" -name "model.obj" 2>/dev/null | head -1)
    if [ -z "${OBJ_FILE}" ]; then
        OBJ_FILE=$(find outputs/dreamfusion-sd -name "model.obj" 2>/dev/null | tail -1)
    fi
    if [ -f "${OBJ_FILE}" ]; then
        cp "${OBJ_FILE}" ../output/object_b_mesh.obj
        echo "Mesh 已导出到 output/object_b_mesh.obj"
    else
        echo "警告: Mesh 导出可能失败"
    fi
    # 复制 360° 视频
    mkdir -p ../output/snapshots
    cp "${TRIAL_DIR}save/"*test*.mp4 ../output/snapshots/object_b_360video.mp4 2>/dev/null || true
else
    echo "错误: 未找到训练输出目录"
fi

# ---------- Step 2.3: 生成训练渐进快照 + 360° 环绕视频 ----------
echo "[2.3] 生成渐进快照与环绕视频..."
# TRIAL_DIR 已在上一步设置
if [ -n "${TRIAL_DIR}" ]; then
    python ../scripts/utils/progressive_snapshots.py \
        --mode all \
        --type threestudio \
        --trial_dir "${TRIAL_DIR}" \
        --name "object_b" \
        --output ../output/snapshots
fi

cd ..
echo "物体B (文本到3D) 完成!"
