@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
set "PORT=8000"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTORCH_ENABLE_MPS_FALLBACK=1"
if not exist "venvs\env-main\Scripts\python.exe" (
  echo [!] env-main 尚未就绪, 请先运行 scripts\setup.bat
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul || (
  echo [!] 未检测到 ffmpeg, 请先安装 (https://ffmpeg.org/download.html)
  pause
  exit /b 1
)
echo =====================================================
echo  深度视频转换器已启动
echo  打开浏览器: http://localhost:%PORT%
echo  按 Ctrl+C 停止
echo =====================================================
venvs\env-main\Scripts\python.exe backend\app.py
pause
