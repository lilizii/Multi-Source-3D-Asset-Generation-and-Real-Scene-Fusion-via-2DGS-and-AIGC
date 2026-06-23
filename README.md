# Multi-Source-3D-Asset-Generation-and-Real-Scene-Fusion-via-2DGS-and-AIGC

> 全链路 3D 视觉项目：真实世界重建 + AIGC 虚拟资产生成 + 场景融合渲染
> 一个详细的AutoDL上运行指南

## 项目结构

```
.
├── scripts/                    # 全部项目脚本
│   ├── config.sh               # 全项目统一配置
│   ├── 00_setup_env.sh         # 环境安装
│   ├── 01_prepare_object_a.sh  # 物体A: 多视角重建 (COLMAP+2DGS)
│   ├── 02_prepare_object_b.sh  # 物体B: 文本→3D (DreamFusion)
│   ├── 03_prepare_object_c.sh  # 物体C: 单图→3D (Magic123)
│   ├── 04_reconstruct_bg.sh    # 背景: Mip-NeRF 360 + 2DGS
│   ├── 05_fusion_render.sh     # 场景融合 (GS Surfel 代码级拼接)
│   ├── 06_evaluate_visualize.sh # 评估 + 训练曲线图表
│   ├── run_all.sh              # 一键全链路运行
│   └── utils/                  # Python 工具
│       ├── extract_frames.py           # 视频帧提取
│       ├── remove_bg.py                # 图片去背景
│       ├── aigc_to_gs.py              # AIGC Mesh→高斯面片转换
│       ├── filter_ply.py              # GS PLY 背景碎片过滤
│       ├── fusion_compare.py          # GS 面片场景合并 + 变换
│       ├── gs_native_render.py        # 2DGS CUDA 场景渲染 → 视频
│       ├── evaluate.py                # 质量评估 (几何+纹理+转换+效率)
│       └── export_charts.py           # 训练曲线 + 综合对比图
│
├── configs/                    # 自定义配置
│   ├── object_b_custom.yaml    # DreamFusion SD 自定义配置
│   └── object_c_custom.yaml    # Magic123 自定义配置
│
├── data/                       # 数据目录 (需自行准备)
│   ├── object_a/               # 物体A 环绕视频或照片
│   ├── object_c/               # 物体C 单张照片
│   └── mipnerf360/             # Mip-NeRF 360 背景场景
│
├── output/                     # 全部输出 (自动生成)
└── README.md
```

---

## 一、租用实例

1. 打开 [AutoDL](https://www.autodl.com/)，登录
2. **GPU 选择**：RTX 3090 或 4090（24GB VRAM），按量计费即可
3. **镜像选择**：`PyTorch 2.0.1 + Python 3.8 + CUDA 11.8`
4. **数据盘**：建议 50GB 以上
5. 创建实例，等待启动

---

## 二、上传项目

> **关键**：AutoDL 大陆实例访问 GitHub 很慢或被阻断。
> 所有代码**已在本地准备好**，直接上传即可，**不需要在 AutoDL 上 git clone 任何东西**。

AutoDL 启动后，打开 **JupyterLab**。

### 上传（二选一）

**方式1 — JupyterLab 拖拽**（简单）：
在左侧文件管理器中，将整个 `task1/` 文件夹拖拽到 `/root/autodl-tmp/`。

**方式2 — scp**（更快）：
```bash
# 在本地终端执行
scp -rP [端口号] task1/ root@[AutoDL_IP]:/root/autodl-tmp/
```

上传完成后，项目在 `/root/autodl-tmp/task1/`。

---

## 三、环境安装

### 3.0 克隆项目与依赖仓库

```bash
cd /root/autodl-tmp

# 克隆项目本体
git clone https://github.com/lilizii/Multi-Source-3D-Asset-Generation-and-Real-Scene-Fusion-via-2DGS-and-AIGC.git task1
cd task1

# 克隆三个依赖仓库（如果 GitHub 慢可改用镜像）
git clone https://github.com/colmap/colmap.git
git clone https://github.com/hbb1/2d-gaussian-splatting.git
git clone https://github.com/threestudio-project/threestudio.git

# 初始化 2DGS 子模块
cd 2d-gaussian-splatting
git submodule update --init --recursive
cd ..
```

> 如果 GitHub 网络不通，也可以本地下载 zip 后上传到 `/root/autodl-tmp/task1/`。

### 3.1 安装依赖

```bash
# 0. 验证镜像
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"

# 1. 系统工具 + COLMAP
apt-get update
apt-get install -y colmap xvfb ffmpeg sqlite3

# 清华源加速
PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

# 2. 2DGS 环境
pip install ninja ${PIP_MIRROR}
pip install open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 \
    scikit-image==0.21.0 tqdm trimesh plyfile opencv-python ${PIP_MIRROR}
pip install /root/autodl-tmp/task1/2d-gaussian-splatting/submodules/diff-surfel-rasterization
pip install /root/autodl-tmp/task1/2d-gaussian-splatting/submodules/simple-knn

# 3. threestudio 依赖（git 包从 local_deps/ 本地安装）
cd /root/autodl-tmp/task1/threestudio
pip install local_deps/nerfacc
cd local_deps/tiny-cuda-nn && pip install bindings/torch && cd /root/autodl-tmp/task1/threestudio
pip install --no-build-isolation ./local_deps/nvdiffrast
pip install local_deps/envlight
pip install local_deps/CLIP
pip install xatlas plotly comm accelerate pytorch-lightning==2.0.0 ${PIP_MIRROR}

# 锁定关键版本（避免 HF/transformers 兼容问题）
pip install "diffusers<0.20" "huggingface_hub==0.14.1" "transformers==4.28.1" "imageio==2.28.1" imageio-ffmpeg ${PIP_MIRROR}

# 4. 其他工具
pip install pyvirtualdisplay faiss-cpu ${PIP_MIRROR}
```

---

## 四、准备数据

### 物体A（多视角重建）

用手机拍一段环绕物体的视频（约30秒，缓慢绕一圈），上传到：

```
/root/autodl-tmp/task1/data/object_a/object_a.mp4
```

**如果没有拍摄条件**，在 `data/object_a/input/` 下放 30-100 张环绕拍摄的照片。

### 物体C（单图到3D）

拍一张物体照片，上传到：

```
/root/autodl-tmp/task1/data/object_c/object_c_raw.jpg
```

### 背景场景（Mip-NeRF 360）

```bash
cd /root/autodl-tmp/task1
mkdir -p data

# 下载（AutoDL 直连 Google 很快）
wget -P data http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip data/360_v2.zip -d data/mipnerf360/

# 删掉不用的场景省空间
cd data/mipnerf360
rm -rf bicycle flowers garden kitchen room stump treehill
cd /root/autodl-tmp/task1
rm data/360_v2.zip
```

修改 `scripts/04_reconstruct_bg.sh` 中的场景名：
```bash
SCENE="counter"  # 可选: counter, bonsai
```

---

## 五、一键运行

```bash
# 先开 screen 防断连
screen -S task1

cd /root/autodl-tmp/task1
bash scripts/run_all.sh 2>&1 | tee output/full_log.txt
```

跳过某些步骤：
```bash
cd /root/autodl-tmp/task1
bash scripts/run_all.sh --skip-a --skip-c
bash scripts/run_all.sh --skip-b --skip-c
```

---

## 六、分步运行（如需单独控制）

### Step 1：物体A — 多视角重建

```bash
cd /root/autodl-tmp/task1
bash scripts/01_prepare_object_a.sh
```
输出：`output/object_a/` (2DGS 模型 + checkpoint) + `output/object_a_mesh.ply`

### Step 2：物体B — 文本→3D

```bash
cd /root/autodl-tmp/task1
vim scripts/02_prepare_object_b.sh  # 改 TEXT_PROMPT
bash scripts/02_prepare_object_b.sh
```
输出：`output/object_b_mesh.obj` + `output/object_b/`

### Step 3：物体C — 单图→3D

```bash
cd /root/autodl-tmp/task1
bash scripts/03_prepare_object_c.sh
```
输出：`output/object_c_mesh.obj` + `output/object_c/` 

### Step 4：背景场景重建

```bash
cd /root/autodl-tmp/task1
bash scripts/04_reconstruct_bg.sh
```
输出：`output/background_*/` (2DGS 模型 + checkpoint) + `output/background_mesh.ply`

### Step 5：融合渲染

```bash
cd /root/autodl-tmp/task1
bash scripts/05_fusion_render.sh
```

**融合流程**：
1. A 的 GS 用 cluster 方法过滤背景碎片
2. B/C Mesh → GS Surfel 转换 (`aigc_to_gs.py`)
3. 四个 PLY 缩放/旋转/平移对齐后合并 (`fusion_compare.py`)
4. 单 pass CUDA 渲染 → ffmpeg 合成 360° 环绕视频

输出：`output/fusion/gs_native_video.mp4` + `full_scene.ply`

### Step 6：评估与可视化

```bash
cd /root/autodl-tmp/task1
bash scripts/06_evaluate_visualize.sh
```

生成：
- `output/evaluation/evaluation.json` — 几何 + 纹理 + 转换保真度 + 效率 全部量化数据
- `output/evaluation/summary_table.md` — 五张 Markdown 对比表
- `output/charts/*.png` — 训练曲线 + 综合对比图

---

## 七、进度监控

| 任务 | 监控方式 |
|------|---------|
| 2DGS (物体A/背景) | 终端 tqdm 进度条 (Loss/Points 实时刷新) |
| threestudio (物体B/C) | 终端 PyTorch Lightning 进度条 (Loss/iter)，每 200 步保存验证图 |
| 评估/对比 | 终端日志，约 2 分钟完成 |

---

## 八、查看输出

训练完成后，输出目录结构：

```
output/
├── object_a_mesh.ply                    # 物体A Mesh
├── object_b_mesh.obj                    # 物体B 生成 Mesh
├── object_c_mesh.obj                    # 物体C 生成 Mesh
├── background_mesh.ply                  # 背景 Mesh
│
├── object_a/                            # 2DGS 完整模型 (checkpoint+ply)
├── object_b/                            # threestudio DreamFusion 输出
├── object_c/                            # threestudio Magic123 输出
├── background_*/                        # 背景 2DGS 完整模型
│
├── fusion/                              # === 融合渲染 ===
│   ├── gs_native_video.mp4             # 四物体融合 360° 环绕视频 
│   ├── full_scene.ply                  # 融合后的完整 GS 场景
│   ├── show_A.png / show_B.png / show_C.png  # 单物体多视角预览
│   └── gs_converted/                   # Mesh→GS 转换结果
│
├── evaluation/                          # === 质量评估 ===
│   ├── evaluation.json                 # 几何+纹理+转换+效率 全部量化数据
│   └── summary_table.md               # 五张 Markdown 对比表
│
└── charts/                              # === 训练曲线 ===
    ├── 2dgs_training_curves.png        # A: L1/PSNR/Points
    ├── object_b_loss_*.png             # B: 每个 loss 指标一张
    ├── object_c_coarse_loss_*.png      # C coarse 阶段
    ├── object_c_refine_loss_*.png      # C refine 阶段
    └── method_comparison.png           # 三方法综合横评
```

### 下载到本地

在 JupyterLab 中右键 `output/` → Download as zip，或用 scp：

```bash
# 在本地终端执行
scp -rP [端口号] root@[AutoDL_IP]:/root/autodl-tmp/task1/output/ ./
```

---

## 九、常见问题

| 问题 | 解决 |
|------|------|
| COLMAP `libOpenImageIO.so` 找不到 | `apt-get install -y libopenimageio2.1`。若 conda 装的 colmap 有依赖问题，换成 `apt-get install -y colmap` |
| COLMAP Qt/OpenGL 报错 | 脚本已设 `export QT_QPA_PLATFORM=offscreen` + `--SiftExtraction.use_gpu 0` |
| torch `libnvToolsExt` / `libnvrtc` 找不到 | 环境安装第 5 步已创建软链 + `LD_LIBRARY_PATH` |
| SfM 注册图片太少 (`Could not register`) | 拍摄时物体下面垫花纹布，每张图同时看到物体和布料纹理 |
| threestudio `ModuleNotFoundError: tinycudann` | `cd local_deps/tiny-cuda-nn && pip install bindings/torch`（必须从根目录安装） |
| threestudio `ImportError: cached_download` | `pip install "huggingface_hub==0.14.1"` 锁定版本 |
| threestudio `ModuleNotFoundError: xatlas` | `pip install xatlas`；或加 `system.exporter.save_uv=false` 跳过 UV 展开 |
| threestudio `FileNotFoundError: 256_tets.npz` | 改用 `system.geometry.isosurface_resolution=128`（已预置模板） |
| threestudio 导出 Mesh 文件路径 | Mesh 实际在 `save/it*-export/model.obj`，不是 `save/mesh.obj` |
| HF 缓存撑满系统盘 | 环境安装第 6 步已建软链到数据盘。若已有重复缓存，先 `rm -rf /root/.cache/huggingface` 再链 |
| CUDA Out of Memory | 减小分辨率：2DGS 加 `-r 2`，threestudio 设 `data.width=64` |
| 断连后训练中断 | **一定先开 `screen -S task1`**，重连后 `screen -r task1` |
| threestudio `imageio` 视频合成报错 | `pip install "imageio==2.28.1" imageio-ffmpeg` |

---
