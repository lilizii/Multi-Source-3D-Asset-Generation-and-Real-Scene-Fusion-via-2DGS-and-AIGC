#!/bin/bash
# =============================================================================
# 06_evaluate_visualize.sh — 评估与可视化
#
# 生成:
#   output/evaluation/
#     evaluation.json           几何 + 纹理 + 效率 + 转换保真度 全部量化数据
#     summary_table.md          五张 Markdown 对比表
#   output/charts/
#     2dgs_training_curves.png  A: L1/PSNR/Point数 三子图
#     object_b_loss_*.png       B: 每个 loss 指标一张曲线图
#     object_c_coarse_loss_*.png C coarse: 每个 loss 指标一张
#     object_c_refine_loss_*.png C refine: 每个 loss 指标一张
#     method_comparison.png     几何/纹理 综合横评图
#
# 用法:
#   bash scripts/06_evaluate_visualize.sh              # 全部
#   bash scripts/06_evaluate_visualize.sh --skip-clip  # 跳过 CLIP
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

SKIP_CLIP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-clip) SKIP_CLIP="--skip_clip"; shift ;;
        *) shift ;;
    esac
done

echo "============================================"
echo "Step 6: 评估与可视化"
echo "============================================"

mkdir -p "${OUTPUT_EVAL}" "${OUTPUT_CHARTS}"

# ================================================================
# [6.1] 质量评估
# ================================================================
echo ""
echo "--- [6.1] 质量评估 ---"

# 补 A test views (如果尚未生成)
if [ ! -f "${OUTPUT_OBJ_A}/test/ours_30000/renders/00000.png" ]; then
    echo "  生成物体 A 测试视图..."
    cd 2d-gaussian-splatting
    python render.py -s "../${OBJECT_A_DIR}" -m "../${OUTPUT_OBJ_A}" \
        --skip_train --skip_mesh --eval --quiet
    cd ..
fi

# 三方法综合评估 (几何CD + NeRF→Mesh + Mesh→GS + PSNR + CLIP + 效率)
python scripts/utils/evaluate.py ${SKIP_CLIP}

# ================================================================
# [6.2] 训练曲线图表
# ================================================================
echo ""
echo "--- [6.2] 训练曲线图表 ---"
python scripts/utils/export_charts.py --output "${OUTPUT_CHARTS}"

# ================================================================
# 完成
# ================================================================
echo ""
echo "============================================"
echo "评估完成!"
echo "  ${OUTPUT_EVAL}/evaluation.json"
echo "  ${OUTPUT_EVAL}/summary_table.md"
echo "  ${OUTPUT_CHARTS}/"
echo "============================================"
