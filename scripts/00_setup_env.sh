#!/bin/bash
# =============================================================================
# 00_setup_env.sh - 环境安装脚本
# 在 AutoDL 上运行: bash scripts/00_setup_env.sh
# =============================================================================
set -e

echo "============================================"
echo "Step 0: 环境配置与依赖安装"
echo "============================================"

# ---------- 1. 检测 GPU ----------
echo "[1/7] 检测 GPU 环境..."
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# ---------- 2. 安装 COLMAP ----------
echo "[2/7] 安装 COLMAP..."
# AutoDL 通常预装了 conda
conda install -c conda-forge colmap -y 2>/dev/null || {
    echo "conda install failed, trying pip..."
    pip install pycolmap 2>/dev/null || echo "WARNING: COLMAP installation failed. Please install manually."
}

# ---------- 3. 2DGS 依赖 ----------
echo "[3/7] 安装 2DGS 依赖..."
pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 \
    scikit-image==0.21.0 tqdm trimesh plyfile opencv-python \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install 2d-gaussian-splatting/submodules/diff-surfel-rasterization
pip install 2d-gaussian-splatting/submodules/simple-knn
echo "2DGS deps done."

# ---------- 4. 安装 threestudio 依赖 ----------
echo "[4/7] 安装 threestudio 依赖..."
cd threestudio

# 安装 PyTorch (AutoDL 可能已安装)
python -c "import torch" 2>/dev/null || {
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
}

pip install ninja 2>/dev/null || true
pip install -r requirements.txt

# 创建必要的目录
mkdir -p load/images load/zero123

cd ..
echo "threestudio setup done."

# ---------- 5. 下载预训练模型 ----------
echo "[5/7] 下载预训练模型..."

# Stable Zero123 checkpoint (用于 Magic123 / Stable Zero123)
cd threestudio/load/zero123
if [ ! -f "stable_zero123.ckpt" ]; then
    echo "Downloading Stable Zero123 checkpoint..."
    # 从 HuggingFace 下载
    wget https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable-zero123.ckpt -O stable_zero123.ckpt 2>/dev/null || {
        echo "WARNING: Could not download stable_zero123.ckpt. Please download manually from:"
        echo "  https://huggingface.co/stabilityai/stable-zero123"
    }
fi
cd ../../..

# MipNeRF360 数据集提示
echo "[6/7] Mip-NeRF 360 数据集..."
echo "请手动从 https://jonbarron.info/mipnerf360/ 下载 Mip-NeRF 360 数据集"
echo "并将其解压到 ./data/mipnerf360/ 目录下"
mkdir -p data/mipnerf360

# ---------- 6. Python 工具依赖 ----------
echo "[7/7] 安装 Python 工具依赖..."
pip install rembg pillow numpy opencv-python open3d trimesh imageio wandb swanlab matplotlib 2>/dev/null || true

echo "============================================"
echo "环境安装完成！"
echo ""
echo "请确保以下模型已下载:"
echo "  1. Stable Zero123: threestudio/load/zero123/stable_zero123.ckpt"
echo "  2. MipNeRF360 数据集: data/mipnerf360/"
echo ""
echo "下一步:"
echo "  bash scripts/01_prepare_object_a.sh   # 多视角重建物体A"
echo "  bash scripts/02_prepare_object_b.sh   # 文本生成物体B"
echo "  bash scripts/03_prepare_object_c.sh   # 单图生成物体C"
echo "  bash scripts/04_reconstruct_bg.sh     # 重建背景场景"
echo "  bash scripts/05_fusion_render.sh      # 融合渲染"
echo "============================================"
