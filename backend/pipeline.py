"""核心处理流水线 (env-main, Python 3.12): 深度估计 + 姿态 + 合成 + 音视频封装。

执行顺序强约束(满足用户要求, 且与 Apple Silicon MPS/Metal 隔离原则一致):
  1) 先调用 face_worker 子进程(独立 py3.11 + MediaPipe)完成面部点云;
  2) 再在本进程用 torch 完成深度估计与姿态(绝不早于面部);
  3) 按所选模式合成;
  4) ffmpeg 封装, 保留原声。

本文件**绝不** import mediapipe。
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

# 在 Apple Silicon 上, 部分算子(如 upsample_bicubic2d)尚未实现 MPS 后端,
# 开启回退到 CPU 以保证 Depth-Anything-V2(DINOv2 骨干)等模型可运行。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np

import config

# ---- 延迟加载的重模型(全局缓存) ----
_DEPTH_PROC = None
_DEPTH_MODEL = None
_POSE_MODEL = None
_DEVICE = None

# COCO 17 关键点骨架连接
_KP_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # 头/脸
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 肩-臂-手
    (5, 11), (6, 12), (11, 12),            # 肩-髋
    (11, 13), (13, 15), (12, 14), (14, 16),  # 髋-膝-踝
]


def get_device():
    global _DEVICE
    if _DEVICE is None:
        # 仅在真正的 Apple Silicon(arm64)上使用 MPS; Intel Mac 即使 torch 报告
        # MPS 可用, 实际无 GPU, 且部分算子未在 MPS 实现, 故强制 CPU。
        is_apple_silicon = (platform.system() == "Darwin" and platform.machine() == "arm64")
        if is_apple_silicon and hasattr(__import__("torch").backends, "mps") \
                and __import__("torch").backends.mps.is_available():
            _DEVICE = "mps"
        else:
            _DEVICE = "cpu"
    return _DEVICE


def load_depth():
    global _DEPTH_PROC, _DEPTH_MODEL
    if _DEPTH_MODEL is None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        _DEPTH_PROC = AutoImageProcessor.from_pretrained(config.DEPTH_MODEL_DIR, local_files_only=True)
        _DEPTH_MODEL = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL_DIR, local_files_only=True)
        _DEPTH_MODEL = _DEPTH_MODEL.to(get_device()).eval()
    return _DEPTH_PROC, _DEPTH_MODEL


def load_pose():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        import torch
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        sd = torch.load(config.KP_WEIGHTS, map_location="cpu", weights_only=True)
        # weights_backbone=None 避免 torchvision 联网下载 ResNet50 主干, 完全使用本地权重
        _POSE_MODEL = keypointrcnn_resnet50_fpn(num_classes=2, weights=None, weights_backbone=None)
        _POSE_MODEL.load_state_dict(sd)
        _POSE_MODEL = _POSE_MODEL.to(get_device()).eval()
    return _POSE_MODEL


# ---------- 单帧推理 ----------
def infer_depth(rgb):
    """返回近亮远暗的 uint8 灰度图。"""
    import torch
    from PIL import Image
    proc, model = load_depth()
    pil = Image.fromarray(rgb)
    inputs = proc(images=pil, return_tensors="pt").to(get_device())
    with torch.no_grad():
        out = model(**inputs)
        depth = out.predicted_depth.squeeze().cpu().numpy()
    dmin, dmax = depth.min(), depth.max()
    dn = (depth - dmin) / (dmax - dmin + 1e-8)
    gray = ((1.0 - dn) * 255.0).astype(np.uint8)  # 近(值小)->亮
    return gray


def infer_pose(rgb):
    """返回透明 BGRA 姿态骨架叠加层。"""
    import torch
    from torchvision.transforms import functional as TF
    model = load_pose()
    t = TF.to_tensor(rgb).to(get_device())
    with torch.no_grad():
        preds = model([t])
    pred = preds[0]
    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    kps = pred["keypoints"].cpu().numpy()
    kp_scores = pred.get("keypoints_scores", np.ones_like(kps[:, :, 0])) if "keypoints_scores" in pred else np.ones_like(kps[:, :, 0])
    if isinstance(kp_scores, list):
        kp_scores = np.array(kp_scores)
    kp_scores = kp_scores if hasattr(kp_scores, "shape") else np.ones_like(kps[:, :, 0])
    for i, sc in enumerate(scores):
        if sc < config.CONF_POSE:
            continue
        k = kps[i]
        ks = kp_scores[i] if kp_scores.ndim == 2 else None
        pts = {}
        for j, (x, y, v) in enumerate(k):
            pts[j] = (int(round(x)), int(round(y)), (ks[j] if ks is not None else 1.0))
        for (a, b) in _KP_EDGES:
            pa, pb = pts.get(a), pts.get(b)
            if pa is None or pb is None:
                continue
            if pa[2] < 0.3 or pb[2] < 0.3:
                continue
            cv2.line(overlay, (pa[0], pa[1]), (pb[0], pb[1]), (0, 255, 255, 255), 2)
        for j, (x, y, v) in enumerate(k):
            if v >= 0.3:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 3, (0, 255, 0, 255), -1)
    return overlay


# ---------- 合成 ----------
def alpha_blend(base_bgr, overlay_bgra):
    ov = overlay_bgra[:, :, :3].astype(np.float32)
    al = (overlay_bgra[:, :, 3:4].astype(np.float32)) / 255.0
    out = base_bgr.astype(np.float32) * (1 - al) + ov * al
    return out.astype(np.uint8)


def composite_frame(mode, original_bgr, depth_gray, pose_bgra, face_bgra):
    H, W = original_bgr.shape[:2]
    if mode == "depth":
        d3 = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
        return d3
    if mode == "pose":
        return alpha_blend(original_bgr, pose_bgra)
    if mode == "combo":
        d3 = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
        return alpha_blend(d3, pose_bgra)
    if mode == "face":
        return alpha_blend(original_bgr, face_bgra)
    if mode == "all":
        d3 = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
        d3 = alpha_blend(d3, pose_bgra)
        return alpha_blend(d3, face_bgra)
    # 默认原图
    return original_bgr


# ---------- 视频探测 ----------
def probe_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # 探测是否有音轨
    has_audio = False
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=index", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=30)
        has_audio = bool(r.stdout.strip())
    except Exception:
        has_audio = False
    return {"fps": fps, "frames": n, "w": w, "h": h, "has_audio": has_audio}


def target_size(w, h, max_dim):
    if max(w, h) <= max_dim:
        return w, h
    if w >= h:
        return max_dim, int(round(h * max_dim / w))
    return int(round(w * max_dim / h)), max_dim


# ---------- 主流程 ----------
def run_pipeline(job_id, video_path, mode, progress):
    """progress(pct:int, stage:str, msg:str) 回调用于上报进度。"""
    import torch  # 仅需确保可用
    config.ensure_dirs()
    job_dir = os.path.join(config.JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    face_dir = os.path.join(job_dir, "face_frames")
    depth_dir = os.path.join(job_dir, "depth_frames")
    pose_dir = os.path.join(job_dir, "pose_frames")
    orig_dir = os.path.join(job_dir, "orig_frames")
    out_dir = os.path.join(job_dir, "out_frames")
    for d in (face_dir, depth_dir, pose_dir, orig_dir, out_dir):
        os.makedirs(d, exist_ok=True)

    info = probe_video(video_path)
    tw, th = target_size(info["w"], info["h"], config.MAX_DIM)
    total = max(info["frames"], 1)
    fps = info["fps"]

    # ---- 阶段1: 面部(必须先于深度, 独立子进程) ----
    progress(2, "face", f"面部点云检测中 (py3.11/MediaPipe)…")
    face_meta_path = os.path.join(job_dir, "face_meta.json")
    face_prog = os.path.join(job_dir, "face_progress.json")
    face_cmd = [
        config.ENV_FACE_PY, os.path.join(config.BACKEND_DIR, "face_worker.py"),
        "--video", video_path, "--outdir", face_dir, "--meta", face_meta_path,
        "--w", str(tw), "--h", str(th), "--model", config.FACE_MODEL_PATH,
        "--detector-model", config.FACE_DETECTOR_PATH,
        "--progress-file", face_prog, "--num-faces", str(config.FACE_NUM),
        "--det-conf", str(config.FACE_DET_CONF),
        "--presence-conf", str(config.FACE_PRESENCE_CONF),
        "--crop-pad", str(config.FACE_CROP_PAD),
    ]
    fproc = subprocess.Popen(face_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    face_done = 0
    while fproc.poll() is None:
        try:
            with open(face_prog) as f:
                pj = json.load(f)
            face_done = pj.get("done", 0)
        except Exception:
            face_done = 0
        progress(int(2 + 38 * face_done / total), "face",
                 f"面部点云检测中 {face_done}/{total}")
        time.sleep(0.3)
    frc, frout = fproc.communicate()
    if fproc.returncode != 0:
        err = (frout or b"").decode(errors="ignore")[-2000:]
        raise RuntimeError(f"面部子进程失败 (code {fproc.returncode}): {err}")
    progress(40, "depth", "面部完成, 开始深度估计 + 姿态…")

    # ---- 阶段2: 深度 + 姿态(本进程 torch) ----
    import torch
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (tw, th))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(orig_dir, f"orig_{idx:06d}.png"), frame)
        depth_gray = infer_depth(rgb)
        # Depth-Anything-V2 的图像处理器会把输入缩放到模型内部分辨率,
        # 这里缩回 (tw, th) 以便与各层对齐合成。
        depth_gray = cv2.resize(depth_gray, (tw, th))
        pose_bgra = infer_pose(rgb)
        cv2.imwrite(os.path.join(depth_dir, f"depth_{idx:06d}.png"), depth_gray)
        cv2.imwrite(os.path.join(pose_dir, f"pose_{idx:06d}.png"), pose_bgra)
        idx += 1
        if idx % 3 == 0 or idx == total:
            progress(int(40 + 40 * idx / total), "depth", f"深度+姿态 {idx}/{total}")
    cap.release()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    progress(80, "composite", "合成输出帧…")

    # ---- 阶段3: 合成 ----
    n = idx
    produced = 0
    while produced < n:
        frame = cv2.imread(os.path.join(orig_dir, f"orig_{produced:06d}.png"))
        depth_gray = cv2.imread(os.path.join(depth_dir, f"depth_{produced:06d}.png"), cv2.IMREAD_GRAYSCALE)
        pose_bgra = cv2.imread(os.path.join(pose_dir, f"pose_{produced:06d}.png"), cv2.IMREAD_UNCHANGED)
        face_bgra = cv2.imread(os.path.join(face_dir, f"face_{produced:06d}.png"), cv2.IMREAD_UNCHANGED)
        if frame is None:
            break
        if depth_gray is None:
            depth_gray = np.zeros((th, tw), np.uint8)
        if pose_bgra is None:
            pose_bgra = np.zeros((th, tw, 4), np.uint8)
        if face_bgra is None:
            face_bgra = np.zeros((th, tw, 4), np.uint8)
        out = composite_frame(mode, frame, depth_gray, pose_bgra, face_bgra)
        cv2.imwrite(os.path.join(out_dir, f"out_{produced:06d}.png"), out)
        produced += 1
        if produced % 5 == 0 or produced == n:
            progress(int(80 + 15 * produced / n), "composite", f"合成 {produced}/{n}")
    progress(95, "mux", "封装视频并保留原声…")

    # ---- 阶段4: 编码 + 混音 ----
    silent = os.path.join(job_dir, "silent.mp4")
    final = os.path.join(job_dir, "result.mp4")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    enc = [ffmpeg, "-y", "-framerate", f"{fps:.3f}", "-i",
           os.path.join(out_dir, "out_%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", silent]
    subprocess.run(enc, check=True, capture_output=True)
    if info["has_audio"]:
        mux = [ffmpeg, "-y", "-i", silent, "-i", video_path,
               "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", final]
        r = subprocess.run(mux, capture_output=True)
        if r.returncode != 0:
            shutil.copy(silent, final)
    else:
        shutil.copy(silent, final)

    progress(100, "done", "完成", result=os.path.relpath(final, config.BASE_DIR))
    return final
