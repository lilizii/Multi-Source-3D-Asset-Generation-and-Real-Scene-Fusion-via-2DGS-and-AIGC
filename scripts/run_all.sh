#!/bin/bash
# =============================================================================
# run_all.sh - 一键运行完整流水线
#
# 用法:
#   bash scripts/run_all.sh                           # 全部
#   bash scripts/run_all.sh --skip-a --skip-bg         # 跳过物体A和背景
#
# 前提: 已完成环境安装 (bash scripts/00_setup_env.sh) 并准备数据
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

SKIP_A=false; SKIP_B=false; SKIP_C=false; SKIP_BG=false
for arg in "$@"; do
    case $arg in
        --skip-a) SKIP_A=true ;;
        --skip-b) SKIP_B=true ;;
        --skip-c) SKIP_C=true ;;
        --skip-bg) SKIP_BG=true ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="output/logs_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "全链路流水线启动 (SCENE=${SCENE})"
echo "时间: $(date)"
echo "日志: ${LOG_DIR}/"
echo "============================================"

START_TIME=$(date +%s)

# ---------- [1/6] 物体A: 多视角重建 ----------
if [ "$SKIP_A" = false ]; then
    echo ""; echo ">>> [1/6] 物体A: 多视角重建"
    bash scripts/01_prepare_object_a.sh 2>&1 | tee "${LOG_DIR}/01_object_a.log"
else
    echo ">>> [1/6] 跳过物体A"
fi

# ---------- [2/6] 物体B: 文本→3D ----------
if [ "$SKIP_B" = false ]; then
    echo ""; echo ">>> [2/6] 物体B: 文本到3D"
    bash scripts/02_prepare_object_b.sh 2>&1 | tee "${LOG_DIR}/02_object_b.log"
else
    echo ">>> [2/6] 跳过物体B"
fi

# ---------- [3/6] 物体C: 单图→3D ----------
if [ "$SKIP_C" = false ]; then
    echo ""; echo ">>> [3/6] 物体C: 单图到3D"
    bash scripts/03_prepare_object_c.sh 2>&1 | tee "${LOG_DIR}/03_object_c.log"
else
    echo ">>> [3/6] 跳过物体C"
fi

# ---------- [4/6] 背景场景重建 ----------
if [ "$SKIP_BG" = false ]; then
    echo ""; echo ">>> [4/6] 背景场景重建 (${SCENE})"
    bash scripts/04_reconstruct_bg.sh 2>&1 | tee "${LOG_DIR}/04_background.log"
else
    echo ">>> [4/6] 跳过背景重建"
fi

# ---------- [5/6] 场景融合 ----------
echo ""; echo ">>> [5/6] 场景融合与渲染"
bash scripts/05_fusion_render.sh 2>&1 | tee "${LOG_DIR}/05_fusion.log"

# ---------- [6/6] 评估与可视化 ----------
echo ""; echo ">>> [6/6] 评估与可视化"
bash scripts/06_evaluate_visualize.sh 2>&1 | tee "${LOG_DIR}/06_evaluate_visualize.log"

# ---------- 完成 ----------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "全链路流水线完成!"
echo "总耗时: $((DURATION / 3600))h $(((DURATION % 3600) / 60))min"
echo "============================================"

# 生成摘要
cat > "${LOG_DIR}/summary.txt" << EOF
========================================
任务完成摘要
========================================
完成时间: $(date)
总耗时: $((DURATION / 3600))h $(((DURATION % 3600) / 60))min
背景场景: ${SCENE}

输出目录:
  物体A Mesh:    ${MESH_A}
  物体B Mesh:    ${MESH_B}
  物体C Mesh:    ${MESH_C}
  背景 Mesh:     ${MESH_BG}
  融合视频:      ${OUTPUT_FUSION}/fusion_open3d_video.mp4
  合并场景:      ${OUTPUT_FUSION}/full_scene.ply
  评估报告:      ${OUTPUT_EVAL}/evaluation.json
  对比表:        ${OUTPUT_EVAL}/summary_table.md
  训练曲线:      ${OUTPUT_CHARTS}/
  渐进快照:      ${OUTPUT_SNAPSHOTS}/
  综合对比:      ${OUTPUT_COMPARISON}/comprehensive_report.json
========================================
EOF

echo "摘要: ${LOG_DIR}/summary.txt"
