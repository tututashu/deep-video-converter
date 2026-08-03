# 深度视频转换器 · Deep Video Converter

本地、私有、网页版视频特效工具。上传一段短视频，浏览器内选择效果并下载结果（**保留原声**）。

## 支持的效果模式
| 标识 | 模式 | 技术 |
|------|------|------|
| `depth` | 灰度深度图（近亮远暗） | Depth-Anything-V2 |
| `pose`  | 人体姿态骨架叠加 | PyTorch KeypointRCNN (COCO 17 点) |
| `combo` | 深度 + 姿态组合 | 深度为底 + 骨架叠加 |
| `face`  | 面部 478 点云 | MediaPipe FaceLandmarker |
| `all`   | 以上全部叠加 | 深度 + 姿态 + 面部 |

## 关键架构约束（已实现）
> Apple Silicon 上 torch(MPS) 与 MediaPipe(Metal) 推理不能混用；面部相关库需独立 Python 环境。

- **双虚拟环境**：`venvs/env-main`(Python 3.12：torch / torchvision / transformers / opencv / flask) 与 `venvs/env-face`(Python 3.11：mediapipe)。
- **进程隔离**：面部捕捉运行在独立的 `env-face` 子进程中，**绝不**与主进程的 torch 共享地址空间。
- **执行顺序**：编排器**先**完成面部点云（MediaPipe 子进程），**再**在本进程执行深度估计 + 姿态（torch）。
- **保留原声**：最终经 `ffmpeg` 将原视频音轨封装回结果，无声轨时自动降级。

## 目录结构
```
deep-video-converter/
├── backend/
│   ├── app.py            # Flask 服务：上传 / 进度 / 下载
│   ├── pipeline.py       # 深度+姿态+合成+封装（仅 torch，绝不 import mediapipe）
│   ├── face_worker.py    # 面部子进程（仅 mediapipe）
│   ├── config.py         # 路径与参数
│   ├── requirements_main.txt
│   └── requirements_face.txt
├── frontend/index.html  # 单页：拖拽 + 模式卡片 + 进度条
├── models/               # 模型权重（自包含）
│   ├── face_landmarker/face_landmarker.task       # 面部 478 点 landmark 模型
│   ├── face_detector/blaze_face_short_range.tflite # 整帧人脸检测器(两阶段用)
│   ├── depth_anything_v2/   (Depth-Anything-V2-Small-hf)
│   └── keypointrcnn/keypointrcnn_resnet50_fpn.pth
├── venvs/env-main  venvs/env-face
├── jobs/                 # 每个任务的处理中间产物与结果
└── scripts/              # 跨平台启动 / 搭建脚本
```

## 运行
需要先安装 **ffmpeg**（`Mac: brew install ffmpeg`；`Win` 从 ffmpeg.org 安装并加入 PATH）。

```bash
# Mac / Linux
bash scripts/setup.sh     # 首次：建环境 + 下模型（幂等）
bash scripts/run.sh       # 启动，打开 http://localhost:8000

# Windows
scripts\setup.bat
scripts\run.bat
```

自定义端口：`PORT=9000 bash scripts/run.sh`

## 使用
1. 拖拽（或点击）上传短视频（建议 ≤ 15s，处理分辨率上限 480px）。
2. 选择一种效果模式（A–E）。
3. 点击「开始转换」，实时查看进度。
4. 完成后点击「下载结果视频」。

## 实现细节
- 处理分辨率上限在 `backend/config.py` 的 `MAX_DIM`（默认 480）调整。
- 深度图标准化后反转亮度，使「近亮远暗」。
- **面部点云（两阶段检测）**：FaceLandmarker 自带的是严格的自拍距离检测器，对走动/中远景/侧脸召回很低。本工具先用 BlazeFace 检测器（`blaze_face_short_range.tflite`，低置信度阈值 `FACE_DET_CONF=0.3`）在整帧上定位人脸框，再把人脸框**裁剪放大**后交给 FaceLandmarker 提取 478 点并映射回整帧——在任意普通视频上也能稳定出点云。小分辨率帧会先放大再检测。
- 面部点云按 z 深度做蓝→红渐变着色，输出透明叠加层后与底图 alpha 混合。
- 姿态骨架按 COCO 17 点连接绘制，置信度阈值 `CONF_POSE=0.7`。

## 端到端验证
已用一段测试视频（`tmp/test_clip.mp4`，4s/48 帧/含人声）端到端验证：
- **五种模式**（depth / pose / combo / face / all）均产出 `h264 + aac` 的结果视频，**原声完整保留**。
- 面部点云在测试视频上命中 24/48 帧（人脸进入画面并被两阶段检测捕获的时段）。
- Web 全流程（上传 → 处理 → 进度轮询 → 下载）通过 `/api/process` `/api/status` `/api/download` 实测可用。
- 复跑命令：`bash scripts/run.sh` 后在浏览器操作；或命令行 `PYTHONPATH=backend venvs/env-main/bin/python scripts/run_e2e_all.py`（mode=all），`scripts/recomposite_test.py` 可复用已算出的各层帧快速重合成其余四种模式。

## 性能说明
- 当前测试机为 Intel Mac（CPU 推理）。深度/姿态在 CPU 上逐帧处理，短视频（数秒）可在数十秒内完成；Apple Silicon 上可启用 MPS 加速（面部仍独立在 py3.11 子进程）。
- 如需更高分辨率/更长视频，提高 `MAX_DIM` 或减小帧率。

## 故障排查
- `env-face` 在 macOS Sequoia + Homebrew Python 3.11 下可能出现 `pyexpat` 加载失败：
  `setup.sh` 会自动 `brew install expat` 并用 `install_name_tool` 重新指向新库。
- 模型下载走 `HF_ENDPOINT`（默认 `https://hf-mirror.com` 国内镜像），如网络受限可改回官方 `https://huggingface.co`。
