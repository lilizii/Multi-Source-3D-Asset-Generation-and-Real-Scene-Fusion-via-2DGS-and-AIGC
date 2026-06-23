#!/bin/bash
# =============================================================================
# 03_prepare_object_c.sh - 物体C: 单图到3D生成 (Magic123 / Stable Zero123)
#
# 使用说明:
#   1. 用手机拍摄一张真实物体的照片
#   2. 将照片放到 data/object_c/object_c_raw.jpg
#   3. 运行脚本自动去背景并生成3D模型
#
# 推荐硬件: >= 12GB VRAM
# =============================================================================
set -e

INPUT_IMAGE="data/object_c/object_c_raw.jpg"
INPUT_RGBA="data/object_c/object_c_rgba.png"
TEXT_PROMPT="a photorealistic 3D model of the object"  # 修改为你的物体描述

echo "============================================"
echo "Step 3: 物体C - 单图到3D生成 (Magic123)"
echo "============================================"

mkdir -p data/object_c output/object_c

# ---------- Step 3.1: 图片去背景 ----------
echo "[3.1] 去除图片背景..."
python scripts/utils/remove_bg.py \
    --input "${INPUT_IMAGE}" \
    --output "${INPUT_RGBA}" \
    --model u2net_cloth

if [ ! -f "${INPUT_RGBA}" ]; then
    echo "错误: 去背景失败, 请手动处理图片"
    exit 1
fi
echo "去背景完成: ${INPUT_RGBA}"

# ---------- Step 3.2: 复制图片到 threestudio 目录 ----------
cp "${INPUT_RGBA}" threestudio/load/images/object_c_rgba.png

# ---------- Step 3.3: Magic123 粗阶段 (NeRF + Zero123 + SD) ----------
echo "[3.2] Magic123 Coarse Stage (NeRF)..."
cd threestudio

# HF 统一走数据盘 + 镜像
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

python launch.py \
    --config configs/magic123-coarse-sd.yaml \
    --train \
    --gpu 0 \
    data.image_path=load/images/object_c_rgba.png \
    system.prompt_processor.prompt="${TEXT_PROMPT}" \


echo "Magic123 粗阶段完成"

# ---------- Step 3.4: Magic123 细化阶段 (DMTet) ----------
echo "[3.3] Magic123 Refine Stage (DMTet)..."
# 找到粗阶段的trial目录
COARSE_DIR=$(ls -td outputs/magic123-coarse-sd/*/ 2>/dev/null | head -1)

if [ -n "$COARSE_DIR" ]; then
    CKPT_PATH="${COARSE_DIR}ckpts/last.ckpt"

    python launch.py \
        --config configs/magic123-refine-sd.yaml \
        --train \
        --gpu 0 \
        data.image_path=load/images/object_c_rgba.png \
        system.prompt_processor.prompt="${TEXT_PROMPT}" \
        system.geometry_convert_from="${CKPT_PATH}" \


    echo "Magic123 细化阶段完成"

    # ---------- Step 3.5: 导出 Mesh ----------
    echo "[3.4] 导出Mesh..."
    REFINE_DIR=$(ls -td outputs/magic123-refine-sd/*/ 2>/dev/null | head -1)
    if [ -n "$REFINE_DIR" ]; then
        REFINE_CKPT="${REFINE_DIR}ckpts/last.ckpt"
        REFINE_CONFIG="${REFINE_DIR}configs/parsed.yaml"

        python launch.py \
            --config "${REFINE_CONFIG}" \
            --export \
            --gpu 0 \
            resume="${REFINE_CKPT}" \
            system.exporter_type=mesh-exporter \
            system.exporter.fmt=obj \
            system.geometry.isosurface_resolution=128 \
            system.exporter.save_uv=false \
            system.exporter.save_texture=true

        OBJ_FILE=$(find "${REFINE_DIR}/save" -name "model.obj" 2>/dev/null | head -1)
        if [ -z "${OBJ_FILE}" ]; then
            OBJ_FILE=$(find outputs/magic123-refine-sd -name "model.obj" 2>/dev/null | tail -1)
        fi
        if [ -f "${OBJ_FILE}" ]; then
            cp "${OBJ_FILE}" ../output/object_c_mesh.obj
            echo "Mesh 已导出到 output/object_c_mesh.obj"
        else
            echo "警告: Mesh 导出可能失败"
        fi
        # 复制 360° 视频
        mkdir -p ../output/snapshots
        cp "${REFINE_DIR}save/"*test*.mp4 ../output/snapshots/object_c_360video.mp4 2>/dev/null || true
    fi
else
    echo "警告: 粗阶段未找到输出, 尝试使用 Stable Zero123 替代方案..."
    # 备选方案: Stable Zero123 (更快, 更稳定)
    echo "使用 Stable Zero123..."
    python launch.py \
        --config configs/stable-zero123.yaml \
        --train \
        --gpu 0 \
        data.image_path=load/images/object_c_rgba.png \

fi

# ---------- Step 3.6: 生成训练渐进快照 + 360° 环绕视频 ----------
echo "[3.6] 生成渐进快照与环绕视频..."
# 使用 refine 阶段的 trial（如果有），否则 coarse
SNAPSHOT_TRIAL="${REFINE_DIR:-${COARSE_DIR}}"
if [ -n "${SNAPSHOT_TRIAL}" ]; then
    python ../scripts/utils/progressive_snapshots.py \
        --mode all \
        --type threestudio \
        --trial_dir "${SNAPSHOT_TRIAL}" \
        --name "object_c" \
        --output ../output/snapshots
fi

cd ..
echo "物体C (单图到3D) 完成!"
