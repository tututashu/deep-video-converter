#!/usr/bin/env bash
# 启动深度视频转换器 (Mac / Linux)
set -e
cd "$(cd "$(dirname "$0")" && pwd)/.."
PORT="${PORT:-8000}"
# 模型权重已随工具自带, 禁止联网校验, 保证离线/内网可用
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="venvs/env-main/bin/python"
if [ ! -x "$PY" ]; then
  echo "[!] env-main 尚未就绪, 请先运行: bash scripts/setup.sh"
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[!] 未检测到 ffmpeg, 请先安装 (Mac: brew install ffmpeg)"
  exit 1
fi
echo "====================================================="
echo " 深度视频转换器已启动"
echo " 打开浏览器: http://localhost:$PORT"
echo " 按 Ctrl+C 停止"
echo "====================================================="
exec "$PY" backend/app.py
