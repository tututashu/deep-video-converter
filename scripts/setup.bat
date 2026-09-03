@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0\.."
set "HF_ENDPOINT=https://hf-mirror.com"
if not defined RELEASE_BASE set "RELEASE_BASE=https://github.com/tututashu/deep-video-converter/releases/download/models-v1"
echo [*] HF 模型源: %HF_ENDPOINT%

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

rem ---------- 模型下载: 官方源优先 → MIRROR 镜像 → GitHub Releases 分卷兜底 ----------
rem MIRROR_MEDIAPIPE / MIRROR_PYTORCH 可挂自建镜像; RELEASE_BASE 可覆盖分卷地址。

if not exist "models\face_landmarker\face_landmarker.task" (
  echo [*] 下载 face_landmarker.task …
  set "OK=0"
  curl -f -sSL --connect-timeout 10 --retry 2 -o "models\face_landmarker\face_landmarker.task" "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" && set "OK=1"
  if not "!OK!"=="1" if defined MIRROR_MEDIAPIPE curl -f -sSL --connect-timeout 10 -o "models\face_landmarker\face_landmarker.task" "%MIRROR_MEDIAPIPE%/face_landmarker/face_landmarker/float16/1/face_landmarker.task" && set "OK=1"
  if not "!OK!"=="1" (
    echo [*] 官方源不可达, 尝试 GitHub Releases 分卷 models_face.zip …
    curl -f -sSL --connect-timeout 15 --retry 3 -o "%TEMP%\models_face.zip" "%RELEASE_BASE%/models_face.zip" && tar -xf "%TEMP%\models_face.zip" -C . && del "%TEMP%\models_face.zip"
    if not exist "models\face_landmarker\face_landmarker.task" echo [!] 手动方案: 设 MIRROR_MEDIAPIPE 镜像重跑, 或从可达机器拷贝 models\
  )
)

if not exist "models\face_detector\blaze_face_short_range.tflite" (
  echo [*] 下载 blaze_face_short_range.tflite (两阶段人脸检测用) …
  set "OK=0"
  curl -f -sSL --connect-timeout 10 --retry 2 -o "models\face_detector\blaze_face_short_range.tflite" "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" && set "OK=1"
  if not "!OK!"=="1" if defined MIRROR_MEDIAPIPE curl -f -sSL --connect-timeout 10 -o "models\face_detector\blaze_face_short_range.tflite" "%MIRROR_MEDIAPIPE%/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" && set "OK=1"
  if not "!OK!"=="1" (
    echo [*] 官方源不可达, 尝试 GitHub Releases 分卷 models_face.zip …
    curl -f -sSL --connect-timeout 15 --retry 3 -o "%TEMP%\models_face.zip" "%RELEASE_BASE%/models_face.zip" && tar -xf "%TEMP%\models_face.zip" -C . && del "%TEMP%\models_face.zip"
    if not exist "models\face_detector\blaze_face_short_range.tflite" echo [!] 手动方案: 设 MIRROR_MEDIAPIPE 镜像重跑, 或从可达机器拷贝 models\
  )
)

if not exist "models\depth_anything_v2\config.json" (
  echo [*] 下载 Depth-Anything-V2-Small-hf (源: %HF_ENDPOINT%) …
  set "HF_ENDPOINT=%HF_ENDPOINT%" && venvs\env-main\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Small-hf', local_dir='models/depth_anything_v2')"
  if errorlevel 1 (
    echo [*] HF 源失败, 尝试 GitHub Releases 分卷 models_depth.zip …
    curl -f -sSL --connect-timeout 15 --retry 3 -o "%TEMP%\models_depth.zip" "%RELEASE_BASE%/models_depth.zip" && tar -xf "%TEMP%\models_depth.zip" -C . && del "%TEMP%\models_depth.zip"
    if not exist "models\depth_anything_v2\config.json" echo [!] 手动方案: 换 HF_ENDPOINT=https://huggingface.co 重跑, 或从可达机器拷贝 models\
  )
)

if not exist "models\keypointrcnn\keypointrcnn_resnet50_fpn.pth" (
  echo [*] 下载 KeypointRCNN 权重 (237MB) …
  set "OK=0"
  curl -f -sSL --connect-timeout 10 --retry 2 -o "models\keypointrcnn\keypointrcnn_resnet50_fpn.pth" "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth" && set "OK=1"
  if not "!OK!"=="1" if defined MIRROR_PYTORCH curl -f -sSL --connect-timeout 10 -o "models\keypointrcnn\keypointrcnn_resnet50_fpn.pth" "%MIRROR_PYTORCH%/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth" && set "OK=1"
  if not "!OK!"=="1" (
    echo [*] 官方源不可达, 尝试 GitHub Releases 分卷 models_keypointrcnn.zip …
    curl -f -sSL --connect-timeout 15 --retry 3 -o "%TEMP%\models_keypointrcnn.zip" "%RELEASE_BASE%/models_keypointrcnn.zip" && tar -xf "%TEMP%\models_keypointrcnn.zip" -C . && del "%TEMP%\models_keypointrcnn.zip"
    if not exist "models\keypointrcnn\keypointrcnn_resnet50_fpn.pth" echo [!] 手动方案: 设 MIRROR_PYTORCH 镜像重跑, 或从可达机器拷贝 models\
  )
)

echo.
echo =====================================================
echo  环境就绪! 运行: scripts\run.bat
echo =====================================================
pause
