#!/bin/bash
# =============================================================================
# 01_prepare_object_a.sh - 物体A: 真实多视角重建 (COLMAP + 2DGS)
#
# 使用说明:
#   1. 用手机拍摄一个真实物体的环绕视频(约30秒, 缓慢绕物体一圈)
#   2. 将视频放到 data/object_a/object_a.mp4
#   3. 或者将多视角照片放到 data/object_a/input/ 目录下
#   4. 运行: bash scripts/01_prepare_object_a.sh
# =============================================================================
set -e

OBJECT_DIR="data/object_a"
INPUT_VIDEO="${OBJECT_DIR}/object_a.mp4"
IMAGE_DIR="${OBJECT_DIR}/input"
OUTPUT_DIR="output/object_a"

echo "============================================"
echo "Step 1: 物体A - 多视角重建 (COLMAP + 2DGS)"
echo "============================================"

mkdir -p "${OBJECT_DIR}" "${IMAGE_DIR}" "${OUTPUT_DIR}"

# ---------- Step 1.1: 视频提取帧 ----------
if [ "$(ls -A ${IMAGE_DIR} 2>/dev/null | head -1)" ]; then
    echo "[1.1] 使用已有图片: $(ls ${IMAGE_DIR} | wc -l) 张 (跳过提取)"
elif [ -f "${INPUT_VIDEO}" ]; then
    echo "[1.1] 从视频提取帧..."
    python scripts/utils/extract_frames.py \
        --video "${INPUT_VIDEO}" \
        --output "${IMAGE_DIR}" \
        --fps 2
    echo "帧提取完成: $(ls ${IMAGE_DIR} | wc -l) 张图片"
else
    echo "错误: 请将视频放在 ${INPUT_VIDEO} 或将图片放在 ${IMAGE_DIR}/"
    exit 1
fi

# ---------- Step 1.2: COLMAP 位姿提取 ----------
echo "[1.2] COLMAP 稀疏重建 (SfM)..."
SFM_START=$(date +%s)
export QT_QPA_PLATFORM=offscreen
mkdir -p ${OBJECT_DIR}/distorted/sparse

# 最终产物：sparse/0/ 或 sparse/ (colmap 3.6)
FINAL_SPARSE="${OBJECT_DIR}/sparse/0"
DB="${OBJECT_DIR}/distorted/database.db"
MAPPER_OUT="${OBJECT_DIR}/distorted/sparse/0"

if [ -f "${FINAL_SPARSE}/cameras.bin" ] || [ -f "${OBJECT_DIR}/sparse/cameras.bin" ]; then
    echo "  ✅ SfM 已完整完成，跳过全部 COLMAP 步骤"
    # colmap 3.6 输出在 sparse/，标准化到 sparse/0/
    if [ ! -d "${FINAL_SPARSE}" ]; then
        mkdir -p ${FINAL_SPARSE}
        mv ${OBJECT_DIR}/sparse/*.bin ${FINAL_SPARSE}/ 2>/dev/null
    fi
else
    # --- 特征提取 ---
    if [ -f "${DB}" ]; then
        echo "  ✅ 特征已提取，跳过 feature_extraction"
    else
        echo "  → 特征提取..."
        colmap feature_extractor \
            --database_path ${DB} \
            --image_path ${IMAGE_DIR} \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model OPENCV \
            --SiftExtraction.use_gpu 0
    fi

    # --- 特征匹配 ---（检查 two_view_geometries 表是否有数据）
    if sqlite3 ${DB} "SELECT count(*) FROM two_view_geometries" 2>/dev/null | grep -q '[1-9]'; then
        echo "  ✅ 匹配已完成，跳过 matching"
    else
        echo "  → exhaustive matching..."
        colmap exhaustive_matcher \
            --database_path ${DB} \
            --SiftMatching.use_gpu 0
    fi

    # --- Mapper ---
    if [ -d "${MAPPER_OUT}" ]; then
        echo "  ✅ Mapper 已完成，跳过"
    else
        echo "  → SfM mapper..."
        rm -rf ${OBJECT_DIR}/distorted/sparse
        mkdir -p ${OBJECT_DIR}/distorted/sparse
        colmap mapper \
            --database_path ${DB} \
            --image_path ${IMAGE_DIR} \
            --output_path ${OBJECT_DIR}/distorted/sparse \
            --Mapper.min_num_matches 5 \
            --Mapper.multiple_models 0 \
            --Mapper.ba_local_num_images 30 \
            --Mapper.init_num_trials 500
    fi

    # --- 去畸变 ---
    if [ -f "${OBJECT_DIR}/sparse/cameras.bin" ] || [ -f "${FINAL_SPARSE}/cameras.bin" ]; then
        echo "  ✅ 去畸变已完成，跳过"
    else
        echo "  → 去畸变..."
        colmap image_undistorter \
            --image_path ${IMAGE_DIR} \
            --input_path ${MAPPER_OUT} \
            --output_path ${OBJECT_DIR} \
            --output_type COLMAP
    fi

    # colmap 3.6 输出到 sparse/ 而非 sparse/0/，2DGS 需要稀疏的 0/ 格式
    if [ ! -d "${FINAL_SPARSE}" ] && [ -d "${OBJECT_DIR}/sparse" ]; then
        mkdir -p ${FINAL_SPARSE}
        mv ${OBJECT_DIR}/sparse/*.bin ${FINAL_SPARSE}/ 2>/dev/null
        echo "  → 已移动 .bin 文件到 sparse/0/"
    fi
fi

SFM_END=$(date +%s)
SFM_MIN=$(( (SFM_END - SFM_START) / 60 ))
echo "COLMAP SfM 完成 (耗时 ${SFM_MIN} min)"

# 修复 torch CUDA 库链接
ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvToolsExt.so.1 \
       /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvToolsExt-847d78f2.so.1 2>/dev/null
ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so \
       /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc-847d78f2.so.1 2>/dev/null

# ---------- Step 1.3: 2DGS 训练 ----------
echo "[1.3] 2DGS 训练..."
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 如果模型已训练完成，跳过
if [ -f "${OUTPUT_DIR}/point_cloud/iteration_30000/point_cloud.ply" ]; then
    echo "2DGS 已训练完成，跳过训练"
else
    cd 2d-gaussian-splatting
    python train.py \
        -s ../${OBJECT_DIR} \
        -m ../${OUTPUT_DIR} \
        --iterations 30000 \
        --lambda_normal 0.05 \
        --lambda_dist 0.0 \
        --depth_ratio 0 \
        --port 6010
    cd ..
fi

TRAIN_END=$(date +%s)
TRAIN_MIN=$(( (TRAIN_END - SFM_END) / 60 ))
echo "2DGS 训练完成 (耗时 ${TRAIN_MIN} min)"

# 保存时间到文件
echo "{\"sfm_min\": ${SFM_MIN}, \"train_min\": ${TRAIN_MIN}}" > ${OUTPUT_DIR}/training_time.json

# ---------- Step 1.4: 导出渲染图和 Mesh ----------
echo "[1.4] 导出测试视图和Mesh..."
cd 2d-gaussian-splatting
python render.py \
    -s ../${OBJECT_DIR} \
    -m ../${OUTPUT_DIR} \
    --skip_test \
    --mesh_res 1024 \
    --num_cluster 1
cd ..

# 复制 mesh 到统一输出目录（只保留最大连通分量=物体本身）
cp ${OUTPUT_DIR}/train/ours_30000/fuse_post.ply output/object_a_mesh.ply 2>/dev/null || true
cp ${OUTPUT_DIR}/train/ours_30000/fuse_unbounded_post.ply output/object_a_mesh.ply 2>/dev/null || true

echo "物体A重建完成!"
echo "输出文件:"
echo "  - 模型: ${OUTPUT_DIR}/"
echo "  - Mesh: output/object_a_mesh.ply"
echo "  - 点云: ${OUTPUT_DIR}/point_cloud/iteration_30000/point_cloud.ply"

# ---------- Step 1.5: 生成 360° 环绕视频（完整场景） ----------
echo "[1.5] 生成 360° 环绕视频..."
cd 2d-gaussian-splatting
python render.py \
    -s ../${OBJECT_DIR} \
    -m ../${OUTPUT_DIR} \
    --render_path \
    --skip_train \
    --skip_test \
    --skip_mesh
cd ..

mkdir -p output/snapshots
cp ${OUTPUT_DIR}/traj/ours_30000/render_traj_color.mp4 output/snapshots/object_a_scene_360.mp4 2>/dev/null || true
echo "完整场景视频: output/snapshots/object_a_scene_360.mp4"

# ---------- Step 1.6: 纯物体 Mesh 360° 视频（无背景） ----------
echo "[1.6] 生成纯物体 360° 视频..."
MESH_FILE="${OUTPUT_DIR}/train/ours_30000/fuse_post.ply"
if [ -f "${MESH_FILE}" ]; then
    python -c "
import os, sys, numpy as np, cv2
from pyvirtualdisplay import Display
disp = Display(visible=False, size=(1024, 768))
disp.start()
import open3d as o3d

mesh_path = '${MESH_FILE}'
out_dir = 'output/snapshots/object_a_mesh_frames'
os.makedirs(out_dir, exist_ok=True)

mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
radius = bbox.get_max_extent() * 2.5

vis = o3d.visualization.Visualizer()
vis.create_window(width=1024, height=768, visible=False)
vis.add_geometry(mesh)
opt = vis.get_render_option()
opt.background_color = np.array([0.12, 0.12, 0.15])
opt.mesh_show_back_face = False

for i in range(180):
    angle = 2 * np.pi * i / 180
    eye = [center[0] + radius * np.cos(angle), center[1], center[2] + radius * np.sin(angle)]
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front(np.array(eye) - center)
    ctr.set_up([0, 1, 0])
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f'frame_{i:04d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if i % 30 == 0:
        print(f'{i}/180')
vis.destroy_window()
disp.stop()
print('Frames done')
"
    ffmpeg -y -framerate 30 -i output/snapshots/object_a_mesh_frames/frame_%04d.png \
        -c:v libx264 -pix_fmt yuv420p output/snapshots/object_a_mesh_only.mp4 2>/dev/null
    echo "纯物体视频: output/snapshots/object_a_mesh_only.mp4"
else
    echo "Mesh 文件未找到，跳过纯物体视频"
fi
