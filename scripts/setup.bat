@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
set "HF_ENDPOINT=https://hf-mirror.com"

where py >nul 2>nul || (
  echo [!] 未找到 py 启动器, 请安装 Python 3.11 与 3.12 并勾选 "Add to PATH"
  pause
  exit /b 1
)

if not exist "venvs\env-main\Scripts\python.exe" py -3.12 -m venv venvs\env-main
if not exist "venvs\env-face\Scripts\python.exe" py -3.11 -m venv venvs\env-face

echo [*] 安装 env-main 依赖 …
venvs\env-main\Scripts\pip install -r backend\requirements_main.txt
echo [*] 安装 env-face 依赖 …
venvs\env-face\Scripts\pip install -r backend\requirements_face.txt

mkdir models\face_landmarker models\face_detector models\depth_anything_v2 models\keypointrcnn 2>nul

if not exist "models\face_landmarker\face_landmarker.task" (
  echo [*] 下载 face_landmarker.task …
  curl -sSL -o models\face_landmarker\face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
if not exist "models\face_detector\blaze_face_short_range.tflite" (
  echo [*] 下载 blaze_face_short_range.tflite (两阶段人脸检测用) …
  curl -sSL -o models\face_detector\blaze_face_short_range.tflite "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
if not exist "models\depth_anything_v2\config.json" (
  echo [*] 下载 Depth-Anything-V2-Small-hf …
  venvs\env-main\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Small-hf', local_dir='models/depth_anything_v2')"
)
if not exist "models\keypointrcnn\keypointrcnn_resnet50_fpn.pth" (
  echo [*] 下载 KeypointRCNN 权重 (237MB) …
  curl -sSL -o models\keypointrcnn\keypointrcnn_resnet50_fpn.pth "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth"
)

echo.
echo =====================================================
echo  环境就绪! 运行: scripts\run.bat
echo =====================================================
pause
