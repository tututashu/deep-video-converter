#!/usr/bin/env bash
# 深度视频转换器 — 一键环境搭建 (Mac / Linux)
# 幂等: 已存在的 venv / 权重会被跳过。
# 模型下载: 逐个候选源自动降级(官方源优先, 可用镜像自动切换, 见各 MIRROR_* 环境变量)。
set -e
cd "$(cd "$(dirname "$0")" && pwd)/.."
ROOT="$(pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[*] 根目录: $ROOT"
echo "[*] HF 模型源: $HF_ENDPOINT"

# ---------- 0. 源候选与降级下载器 ----------
# 用法: fetch_fallback <输出文件> <url1> [url2 ...]
# 依次尝试: 官方源 → (若设置了镜像环境变量) 镜像 → 全部失败给出指引。
# 通过 MIRROR_MEDIAPIPE / MIRROR_PYTORCH 环境变量可挂载自建/社区镜像
# (把值替换为目标文件 URL 的目录前缀即可)。
fetch_fallback() {
  local out="$1"; shift
  local tried=0
  for url in "$@"; do
    tried=$((tried+1))
    echo "[*] 下载($tried/$#): $url"
    if curl -sSL --connect-timeout 10 --retry 3 --retry-delay 2 -o "$out" "$url" 2>/dev/null && [ -s "$out" ]; then
      echo "[✓] 成功: $(basename "$out") ($(du -h "$out" | cut -f1))"
      return 0
    fi
    echo "[!] 该源不可用, 尝试下一个…"
    rm -f "$out"
  done
  echo "[✗] 所有候选源均下载失败: $out"
  return 1
}

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
  SO="$(PYTHONPATH= "$PYFACE" -c "import pyexpat,os;print(os.path.join(os.path.dirname(pyexpat.__file__),'pyexpat.cpython-311-darwin.so'))" 2>/dev/null || true)"
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

# ---------- 5. 下载模型权重 (多源自动降级) ----------
mkdir -p models/face_landmarker models/face_detector models/depth_anything_v2 models/keypointrcnn

if [ ! -f models/face_landmarker/face_landmarker.task ]; then
  echo "[*] 下载 face_landmarker.task …"
  fetch_fallback models/face_landmarker/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
    "${MIRROR_MEDIAPIPE:+${MIRROR_MEDIAPIPE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task}" \
    || echo "[!] 提示: 国内网络可设置 MIRROR_MEDIAPIPE=<镜像目录> 后重跑; 或从可达机器拷贝 models/ 目录"
fi

if [ ! -f models/face_detector/blaze_face_short_range.tflite ]; then
  echo "[*] 下载 blaze_face_short_range.tflite (两阶段人脸检测用) …"
  fetch_fallback models/face_detector/blaze_face_short_range.tflite \
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" \
    "${MIRROR_MEDIAPIPE:+${MIRROR_MEDIAPIPE}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite}" \
    || echo "[!] 提示: 同上, 可设置 MIRROR_MEDIAPIPE 或拷贝 models/"
fi

if [ ! -f models/depth_anything_v2/config.json ]; then
  echo "[*] 下载 Depth-Anything-V2-Small-hf (源: $HF_ENDPOINT) …"
  if ! HF_ENDPOINT="$HF_ENDPOINT" venvs/env-main/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("depth-anything/Depth-Anything-V2-Small-hf", local_dir="models/depth_anything_v2")
PY
  then
    echo "[!] Depth 模型下载失败: 若 $HF_ENDPOINT 不可达, 可设 HF_ENDPOINT=https://huggingface.co 重跑, 或拷贝 models/ 目录"
  fi
fi

if [ ! -f models/keypointrcnn/keypointrcnn_resnet50_fpn.pth ]; then
  echo "[*] 下载 KeypointRCNN 权重 (237MB) …"
  fetch_fallback models/keypointrcnn/keypointrcnn_resnet50_fpn.pth \
    "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth" \
    "${MIRROR_PYTORCH:+${MIRROR_PYTORCH}/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth}" \
    || echo "[!] 提示: 可设置 MIRROR_PYTORCH=<镜像目录> 后重跑; 或从可达机器拷贝 models/ 目录"
fi

echo ""
echo "====================================================="
echo " 环境就绪! 运行启动脚本:"
echo "   Mac/Linux : bash scripts/run.sh"
echo "   Windows   : scripts\\run.bat"
echo "====================================================="
