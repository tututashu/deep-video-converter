#!/usr/bin/env bash
# 深度视频转换器 — 一键环境搭建 (Mac / Linux)
# 幂等: 已存在的 venv / 权重会被跳过。
set -e
cd "$(cd "$(dirname "$0")" && pwd)/.."
ROOT="$(pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[*] 根目录: $ROOT"

# ---------- 1. 解释器检查 ----------
PYMAIN="$(command -v python3.12 || command -v python3)"
PYFACE="$(command -v python3.11 || true)"
if [ -z "$PYFACE" ]; then
  echo "[!] 未找到 python3.11 (面部环境必需)。"
  echo "    Mac:  brew install python@3.11"
  echo "    Win:  https://www.python.org/downloads/ (勾选 3.11)"
  exit 1
fi
echo "[*] env-main 解释器: $PYMAIN"
echo "[*] env-face  解释器: $PYFACE"

# ---------- 2. 创建虚拟环境 ----------
if [ ! -x "venvs/env-main/bin/python" ]; then
  echo "[*] 创建 env-main (py3.12) …"
  "$PYMAIN" -m venv venvs/env-main
fi
if [ ! -x "venvs/env-face/bin/python" ]; then
  echo "[*] 创建 env-face (py3.11) …"
  "$PYFACE" -m venv venvs/env-face
fi

# ---------- 3. macOS 上对 brewed python3.11 的 libexpat 链接修复 ----------
# (Sequoia 上 /usr/lib/libexpat.1.dylib 过旧, 导致 pyexpat 无法加载)
if [ "$(uname)" = "Darwin" ]; then
  SO=venvs/env-face/lib/python3.11/site-packages/../../../../Frameworks/Python.framework/Versions/3.11/lib/python3.11/lib-dynload/pyexpat.cpython-311-darwin.so
  # 定位真实 so
  SO="$($PYFACE -c "import pyexpat,os;print(os.path.join(os.path.dirname(pyexpat.__file__),'pyexpat.cpython-311-darwin.so'))" 2>/dev/null || true)"
  if [ -n "$SO" ] && command -v brew >/dev/null 2>&1; then
    if otool -L "$SO" 2>/dev/null | grep -q "/usr/lib/libexpat.1.dylib"; then
      echo "[*] 修复 env-face 的 libexpat 链接 …"
      brew install expat >/dev/null 2>&1 || true
      EXPAT="$(brew --prefix expat)/lib/libexpat.1.dylib"
      install_name_tool -change /usr/lib/libexpat.1.dylib "$EXPAT" "$SO" 2>/dev/null || true
    fi
  fi
fi

# ---------- 4. 安装依赖 ----------
echo "[*] 安装 env-main 依赖 (torch/transformers/opencv/flask) …"
venvs/env-main/bin/pip install -r backend/requirements_main.txt
echo "[*] 安装 env-face 依赖 (mediapipe) …"
# 若默认 pip 在 macOS 触发 truststore bug, 退回 23.3.2 引导
if ! venvs/env-face/bin/python -c "import mediapipe" >/dev/null 2>&1; then
  venvs/env-face/bin/pip install -r backend/requirements_face.txt
fi

# ---------- 5. 下载模型权重 ----------
mkdir -p models/face_landmarker models/face_detector models/depth_anything_v2 models/keypointrcnn
if [ ! -f models/face_landmarker/face_landmarker.task ]; then
  echo "[*] 下载 face_landmarker.task …"
  curl -sSL -o models/face_landmarker/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
fi
if [ ! -f models/face_detector/blaze_face_short_range.tflite ]; then
  echo "[*] 下载 blaze_face_short_range.tflite (两阶段人脸检测用) …"
  curl -sSL -o models/face_detector/blaze_face_short_range.tflite \
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
fi
if [ ! -f models/depth_anything_v2/config.json ]; then
  echo "[*] 下载 Depth-Anything-V2-Small-hf …"
  venvs/env-main/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("depth-anything/Depth-Anything-V2-Small-hf", local_dir="models/depth_anything_v2")
PY
fi
if [ ! -f models/keypointrcnn/keypointrcnn_resnet50_fpn.pth ]; then
  echo "[*] 下载 KeypointRCNN 权重 (237MB) …"
  curl -sSL -o models/keypointrcnn/keypointrcnn_resnet50_fpn.pth \
    "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth"
fi

echo ""
echo "====================================================="
echo " 环境就绪! 运行启动脚本:"
echo "   Mac/Linux : bash scripts/run.sh"
echo "   Windows   : scripts\\run.bat"
echo "====================================================="
