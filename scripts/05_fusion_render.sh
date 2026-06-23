#!/bin/bash
# =============================================================================
# 05_fusion_render.sh — 场景融合 (GS Surfel 代码级拼接 + 2DGS CUDA 渲染)
#
# A: 原始训练 GS → filter_ply 去背景碎片 → 直接用 (保留训练参数)
# B: threestudio mesh → aigc_to_gs → surfel
# C: threestudio mesh → aigc_to_gs → surfel
# BG: 原始训练 GS, 不动
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

NUM_POINTS=10000
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)       SCENE="$2"; shift 2 ;;
        --num-points)  NUM_POINTS="$2"; shift 2 ;;
        --bg-source)   BG_SOURCE_OVERRIDE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

export SCENE
BG_SOURCE="${BG_SOURCE_OVERRIDE:-${M360_PATH}/${SCENE}}"

echo "============================================"
echo "Step 5: 场景融合"
echo "  A: GS 直接过滤 → 保留训练参数"
echo "  B/C: Mesh → Surfel (${NUM_POINTS} 点)"
echo "============================================"

# --- 前置诊断 ---
echo ""
echo "--- 前置诊断 ---"
ok=true
for p in "${GS_A_RAW}" "${GS_BG}" "${MESH_B}" "${MESH_C}"; do
    if [ -f "$p" ]; then
        echo "  [ok] $p ($(du -h "$p" | cut -f1))"
    else
        echo "  [MISSING] $p"
        ok=false
    fi
done
[ "$ok" = false ] && echo "FATAL: 输入文件缺失" && exit 1

mkdir -p "${OUTPUT_FUSION}/gs_converted" "${OUTPUT_FUSION}"

# ================================================================
# [5.1a] 过滤 A 的背景碎片 (保留训练好的 surfel 参数)
# ================================================================
echo ""
echo "--- [5.1a] 过滤物体A背景碎片 ---"

if [ -f "${GS_A}" ]; then
    echo "  [skip] 已存在: ${GS_A}"
else
    python scripts/utils/filter_ply.py \
        --input "${GS_A_RAW}" \
        --output "${GS_A}" \
        --method clusters --cluster-threshold 3.0
    echo "  A 过滤后: $(du -h "${GS_A}" | cut -f1)"
fi

# ================================================================
# [5.1b] B/C Mesh → Surfel
# ================================================================
echo ""
echo "--- [5.1b] B/C Mesh → Surfel ---"

if [ -f "${GS_B_CONVERTED}" ] && [ -f "${GS_C_CONVERTED}" ]; then
    echo "  [skip] 已存在:"
    ls -lh "${GS_B_CONVERTED}" "${GS_C_CONVERTED}"
else
    python scripts/utils/aigc_to_gs.py \
        --mesh_b "${MESH_B}" \
        --mesh_c "${MESH_C}" \
        --output "${OUTPUT_FUSION}/gs_converted" \
        --num_points "${NUM_POINTS}"
    echo "  完成:"
    ls -lh "${OUTPUT_FUSION}/gs_converted/"
fi

# ================================================================
# [5.2] GS 合并
# ================================================================
echo ""
echo "--- [5.2] GS Surfel 合并 ---"

# A 是训练 GS (preserve_geometry=True), B/C 是转换 surfel (False)
python scripts/utils/fusion_compare.py \
    --gs_bg "${GS_BG}" \
    --gs_a  "${GS_A}" \
    --gs_b  "${GS_B_CONVERTED}" \
    --gs_c  "${GS_C_CONVERTED}" \
    --output "${OUTPUT_FUSION}" \
    --bg_scene "${SCENE}" \
    --scale_a 0.15 --scale_b 0.3 --scale_c 0.3 \
    --rot_y_a 0 --rot_y_b 0 --rot_y_c 0

FULL_SCENE="${OUTPUT_FUSION}/full_scene.ply"
[ ! -f "${FULL_SCENE}" ] && echo "FATAL: full_scene.ply 未生成!" && exit 1
echo "  合并场景: ${FULL_SCENE} ($(du -h "${FULL_SCENE}" | cut -f1))"

# ================================================================
# [5.3] 2DGS CUDA 渲染
# ================================================================
echo ""
echo "--- [5.3] 2DGS CUDA 渲染 ---"

rm -rf "${OUTPUT_FUSION}/gs_native_frames"

python scripts/utils/gs_native_render.py \
    --gs_ply "${FULL_SCENE}" \
    --bg_source "${BG_SOURCE}" \
    --output "${OUTPUT_FUSION}" \
    --num_views 120

NATIVE_VIDEO="${OUTPUT_FUSION}/gs_native_video.mp4"
if [ -f "${NATIVE_VIDEO}" ]; then
    echo "  [ok] 视频: ${NATIVE_VIDEO} ($(du -h "${NATIVE_VIDEO}" | cut -f1))"
else
    echo "  [warn] 帧文件: ${OUTPUT_FUSION}/gs_native_frames/"
fi

echo ""
echo "============================================"
echo "Step 5 完成!"
echo "  full_scene.ply: ${FULL_SCENE}"
echo "  视频:           ${NATIVE_VIDEO}"
echo "============================================"
