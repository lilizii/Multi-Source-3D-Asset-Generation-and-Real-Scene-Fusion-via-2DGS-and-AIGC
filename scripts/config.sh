# =============================================================================
# config.sh — 全项目统一配置
# 所有脚本通过 source scripts/config.sh 引用
# =============================================================================

# -------- 场景配置 --------
SCENE="${SCENE:-room}"                       # 背景场景名 (garden/bicycle/counter/bonsai/kitchen/room)
M360_PATH="data/mipnerf360"                  # Mip-NeRF 360 数据集路径
OBJECT_A_DIR="data/object_a"                 # 物体A 数据目录
OBJECT_B_DIR="output/object_b"              # 物体B 输出目录
OBJECT_C_DIR="output/object_c"              # 物体C 输出目录

# -------- 输出路径 (所有脚本统一) --------
OUTPUT_BG="output/background_${SCENE}"       # 背景2DGS模型
OUTPUT_OBJ_A="output/object_a"               # 物体A 2DGS模型
OUTPUT_FUSION="output/fusion"                # 融合渲染输出 (全部放这里)
OUTPUT_EVAL="output/evaluation"              # 评估数据
OUTPUT_CHARTS="output/charts"                # 训练曲线图表
OUTPUT_SNAPSHOTS="output/snapshots"          # 渐进快照 + 视频
OUTPUT_COMPARISON="output/comparison"        # 综合对比分析

# -------- Mesh/PLY 文件路径 --------
MESH_A="output/object_a_mesh.ply"
MESH_B="output/object_b_mesh.obj"
MESH_C="output/object_c_mesh.obj"
MESH_BG="output/background_mesh.ply"

GS_A_RAW="${OUTPUT_OBJ_A}/point_cloud/iteration_30000/point_cloud.ply"  # 原始训练 GS
GS_A="${OUTPUT_FUSION}/object_a_clean.ply"  # 过滤碎片后的 GS (保留训练参数)
GS_BG="${OUTPUT_BG}/point_cloud/iteration_30000/point_cloud.ply"
GS_B_CONVERTED="${OUTPUT_FUSION}/gs_converted/object_b_surfel.ply"
GS_C_CONVERTED="${OUTPUT_FUSION}/gs_converted/object_c_surfel.ply"

# -------- threestudio trial 搜索目录 --------
TRIAL_B_DIR="${OBJECT_B_DIR}/dreamfusion-sd"
TRIAL_C_COARSE_DIR="${OBJECT_C_DIR}/magic123-coarse-sd"
TRIAL_C_REFINE_DIR="${OBJECT_C_DIR}/magic123-refine-sd"

# -------- 物体C 输入图 --------
OBJECT_C_INPUT_IMG="data/object_c/object_c_rgba.png"

# -------- 硬件 --------
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

echo "[config] SCENE=${SCENE}  BG_OUTPUT=${OUTPUT_BG}"
