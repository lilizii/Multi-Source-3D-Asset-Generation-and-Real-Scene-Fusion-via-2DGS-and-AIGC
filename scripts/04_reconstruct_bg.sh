#!/bin/bash
# =============================================================================
# 04_reconstruct_bg.sh - 背景场景重建 (Mip-NeRF 360 + 2DGS)
#
# 使用说明:
#   1. 从 https://jonbarron.info/mipnerf360/ 下载 Mip-NeRF 360 数据集
#   2. 解压到 data/mipnerf360/
#   3. 选择场景: garden, bicycle, counter, bonsai, kitchen, room
#   4. 运行: SCENE=room bash scripts/04_reconstruct_bg.sh
#
# 推荐硬件: >= 8GB VRAM
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "============================================"
echo "Step 4: 背景场景重建 - ${SCENE}"
echo "============================================"

# ---------- 检查数据集 ----------
if [ ! -d "${M360_PATH}/${SCENE}" ]; then
    echo "错误: 找不到场景 ${M360_PATH}/${SCENE}"
    echo "请从 https://jonbarron.info/mipnerf360/ 下载数据集"
    echo "解压后应有: data/mipnerf360/garden/, data/mipnerf360/bicycle/ 等"
    exit 1
fi

# ---------- 修复 CUDA 库链接 ----------
ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvToolsExt.so.1 \
       /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvToolsExt-847d78f2.so.1 2>/dev/null || true
ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so \
       /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc-847d78f2.so.1 2>/dev/null || true

# ---------- [4.1] 2DGS 训练 ----------
echo "[4.1] 2DGS 训练背景场景..."
cd 2d-gaussian-splatting
if [ -f "../${GS_BG}" ]; then
    echo "  ✅ 2DGS 已训练完成，跳过"
else
    python train.py \
        -s "../${M360_PATH}/${SCENE}" \
        -m "../${OUTPUT_BG}" \
        --iterations 30000 \
        --lambda_normal 0.05 \
        --lambda_dist 0.0 \
        --depth_ratio 0
    echo "  背景场景训练完成"
fi

# ---------- [4.2] 导出 Mesh ----------
if [ -f "../${OUTPUT_BG}/train/ours_30000/fuse_unbounded_post.ply" ]; then
    echo "[4.2] Mesh 已导出，跳过"
else
    echo "[4.2] 导出无界Mesh..."
    python render.py \
        -s "../${M360_PATH}/${SCENE}" \
        -m "../${OUTPUT_BG}" \
        --unbounded --mesh_res 1024 \
        --skip_test --skip_train
fi
cd ..

# 复制 Mesh 到统一位置
cp "${OUTPUT_BG}/train/ours_30000/fuse_unbounded_post.ply" "${MESH_BG}" 2>/dev/null || true
cp "${OUTPUT_BG}/train/ours_30000/fuse_post.ply" "${MESH_BG}" 2>/dev/null || true

# ---------- [4.3] 360° 环绕视频 ----------
mkdir -p "${OUTPUT_SNAPSHOTS}"
if [ -f "${OUTPUT_SNAPSHOTS}/background_${SCENE}_360video.mp4" ]; then
    echo "[4.3] 360° 视频已存在，跳过"
else
    echo "[4.3] 生成 360° 环绕视频..."
    cd 2d-gaussian-splatting
    python render.py \
        -s "../${M360_PATH}/${SCENE}" \
        -m "../${OUTPUT_BG}" \
        --render_path --skip_train --skip_test --skip_mesh
    cd ..
    cp "${OUTPUT_BG}/traj/ours_30000/render_traj_color.mp4" \
       "${OUTPUT_SNAPSHOTS}/background_${SCENE}_360video.mp4" 2>/dev/null || true
fi

# ---------- [4.4] 测试视图 + PSNR/SSIM/LPIPS ----------
mkdir -p "${OUTPUT_EVAL}"
if [ -f "${OUTPUT_BG}/test/ours_30000/renders/00000.png" ]; then
    echo "[4.4] 测试视图已存在，跳过渲染"
else
    echo "[4.4] 生成测试视图 (用于 PSNR/SSIM/LPIPS)..."
    cd 2d-gaussian-splatting
    python render.py \
        -s "../${M360_PATH}/${SCENE}" \
        -m "../${OUTPUT_BG}" \
        --skip_train --skip_mesh --eval --quiet
    cd ..
fi

echo "[4.5] 运行 PSNR/SSIM/LPIPS 评估..."
cd 2d-gaussian-splatting
if [ -d "../${OUTPUT_BG}/test/ours_30000/renders" ] && [ -d "../${OUTPUT_BG}/test/ours_30000/gt" ]; then
    python metrics.py -m "../${OUTPUT_BG}" 2>&1 | tee "../${OUTPUT_EVAL}/background_${SCENE}_metrics.txt" || true
else
    echo "  WARNING: test views 不存在，无法计算 PSNR/SSIM/LPIPS"
fi
cd ..

echo ""
echo "背景场景重建完成: ${OUTPUT_BG}/"
