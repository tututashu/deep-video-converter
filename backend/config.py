"""全局配置与路径。所有路径基于仓库根目录解析，便于自包含分发。"""
import os
import sys

# 仓库根目录: deep-video-converter/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
VENVS_DIR = os.path.join(BASE_DIR, "venvs")

# 模型权重路径
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker", "face_landmarker.task")
FACE_DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detector", "blaze_face_short_range.tflite")
DEPTH_MODEL_DIR = os.path.join(MODELS_DIR, "depth_anything_v2")
KP_DIR = os.path.join(MODELS_DIR, "keypointrcnn")
KP_WEIGHTS = os.path.join(KP_DIR, "keypointrcnn_resnet50_fpn.pth")

# 处理参数
MAX_DIM = 480          # 处理分辨率上限(长边), 兼顾速度与可用性; 导出视频以此分辨率
CONF_POSE = 0.7        # KeypointRCNN 置信度阈值
FACE_NUM = 1           # 同时检测人脸数
FACE_DET_CONF = 0.3    # 整帧人脸检测(BlazeFace)置信度, 低值提升中远景/侧脸召回
FACE_PRESENCE_CONF = 0.5  # 裁剪后 FaceLandmarker 人脸存在置信度
FACE_CROP_PAD = 0.25   # 人脸框裁剪 padding 比例

# 模式定义
MODES = {
    "depth":  "灰度深度图",
    "pose":   "人体姿态骨架叠加",
    "combo":  "深度+姿态组合",
    "face":   "面部478点云",
    "all":    "以上全部叠加",
}


def _is_windows():
    return sys.platform.startswith("win")


def env_python(env_name: str) -> str:
    """返回某个虚拟环境的可执行 python 路径(自动适配 Mac/Linux 与 Windows)。"""
    if _is_windows():
        return os.path.join(VENVS_DIR, env_name, "Scripts", "python.exe")
    return os.path.join(VENVS_DIR, env_name, "bin", "python")


ENV_FACE_PY = env_python("env-face")
ENV_MAIN_PY = env_python("env-main")


def ensure_dirs():
    for d in (MODELS_DIR, JOBS_DIR, TMP_DIR, VENVS_DIR):
        os.makedirs(d, exist_ok=True)
