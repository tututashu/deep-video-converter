"""面部点云子进程 (env-face, Python 3.11, MediaPipe only)。

关键约束实现:
- 本进程**只**加载 MediaPipe, 绝不 import torch, 与深度估计(torch)进程隔离。
- 由编排器在深度估计之前调用, 满足「面部捕获须在深度估计之前执行」。
- 输出: 每帧一张 BGRA 透明 PNG 叠加层(face_XXXXXX.png) + meta.json(每帧是否有人脸)。

两阶段检测(提升任意视频上的人脸召回):
  1) FaceDetector(BlazeFace short-range, 低置信度) 先在整帧上定位人脸框;
  2) 将人脸框裁剪(带 padding)后交给 FaceLandmarker 提取 478 点,
     再把点坐标映射回整帧。
FaceLandmarker 自带检测器是严格的自拍距离检测, 对走动/中远景人脸召回低,
单独裁剪放大后喂给 landmark 模型可显著改善。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


def write_progress(path, done, total):
    try:
        with open(path, "w") as f:
            json.dump({"done": done, "total": total}, f)
    except Exception:
        pass


def lerp_color(t):
    """t in [0,1]: 0=近(红) -> 1=远(蓝)。返回 BGRA 四元组(含不透明 alpha)。"""
    b = int(0 + (255 - 0) * t)
    g = 0
    r = int(255 + (0 - 255) * t)
    return (b, g, r, 255)


def draw_points(overlay, pts, out_w, out_h):
    """pts: [(nx, ny, z)] 整帧归一化坐标 + z 深度。按 z 渐变绘制点云。"""
    if not pts:
        return
    zs = [p[2] for p in pts]
    zmin, zmax = min(zs), max(zs)
    zr = (zmax - zmin) or 1.0
    for (nx, ny, z) in pts:
        px = int(nx * out_w)
        py = int(ny * out_h)
        if 0 <= px < out_w and 0 <= py < out_h:
            t = (z - zmin) / zr  # 越小越近
            cv2.circle(overlay, (px, py), 2, lerp_color(t), -1)


def detect_size(w, h, out_w, out_h, min_dim=640):
    """为检测选择内部分辨率(小帧放大以提升小目标 recall), 再缩回输出分辨率。"""
    scale = max(1.0, min_dim / max(w, h))
    sw, sh = int(round(w * scale)), int(round(h * scale))
    sw, sh = max(sw, out_w), max(sh, out_h)
    return sw, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--model", required=True, help="FaceLandmarker .task 模型")
    ap.add_argument("--detector-model", required=True, help="FaceDetector BlazeFace .tflite 模型")
    ap.add_argument("--progress-file", required=True)
    ap.add_argument("--num-faces", type=int, default=1)
    ap.add_argument("--det-conf", type=float, default=0.3, help="整帧人脸检测置信度")
    ap.add_argument("--presence-conf", type=float, default=0.5, help="裁剪后 landmark 存在置信度")
    ap.add_argument("--crop-pad", type=float, default=0.25, help="人脸框 padding 比例")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    write_progress(args.progress_file, 0, 1)

    # 阶段1: 整帧人脸检测器(召回优先, 低置信度)
    detector = FaceDetector.create_from_options(FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=args.detector_model),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=args.det_conf,
    ))
    # 阶段2: 裁剪后人脸 landmark(每裁剪内 1 张脸)
    landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=args.presence_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    ))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("ERROR: cannot open video", file=sys.stderr)
        sys.exit(2)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    write_progress(args.progress_file, 0, max(total, 1))

    meta = {"fps": fps, "width": args.w, "height": args.h,
            "det_conf": args.det_conf, "frames": []}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_h, raw_w = frame.shape[:2]
        dw, dh = detect_size(raw_w, raw_h, args.w, args.h)
        detect_frame = cv2.resize(frame, (dw, dh)) if (dw, dh) != (raw_w, raw_h) else frame
        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det_res = detector.detect(mp_img)

        overlay = np.zeros((args.h, args.w, 4), dtype=np.uint8)  # BGRA 输出分辨率
        has_face = False

        if det_res.detections:
            # 按分数排序, 取前 num_faces 个
            dets = sorted(det_res.detections,
                          key=lambda d: (d.categories[0].score if d.categories else 0.0),
                          reverse=True)[: args.num_faces]
            for d in dets:
                bb = d.bounding_box
                bx, by, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
                if bw <= 0 or bh <= 0:
                    continue
                pad = int(args.crop_pad * max(bw, bh))
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(dw, bx + bw + pad)
                y1 = min(dh, by + bh + pad)
                crop = detect_frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                cw, ch = x1 - x0, y1 - y0
                crgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                cimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=crgb)
                lm_res = landmarker.detect(cimg)
                if not (lm_res.face_landmarks and len(lm_res.face_landmarks) > 0):
                    continue
                has_face = True
                lms = lm_res.face_landmarks[0]
                # 裁剪内归一化坐标 -> 检测帧像素 -> 检测帧归一化(=整帧归一化)
                pts = []
                for lm in lms:
                    px_d = x0 + lm.x * cw
                    py_d = y0 + lm.y * ch
                    pts.append((px_d / dw, py_d / dh, lm.z))
                draw_points(overlay, pts, args.w, args.h)

        out_path = os.path.join(args.outdir, f"face_{idx:06d}.png")
        cv2.imwrite(out_path, overlay)
        meta["frames"].append({"idx": idx, "face": has_face})
        idx += 1
        if idx % 5 == 0 or idx == total:
            write_progress(args.progress_file, idx, max(total, 1))

    cap.release()
    detector.close()
    landmarker.close()
    with open(args.meta, "w") as f:
        json.dump(meta, f)
    write_progress(args.progress_file, idx, max(total, 1))
    print(f"FACE_DONE frames={idx} faces={sum(1 for fr in meta['frames'] if fr['face'])} out={args.outdir}")


if __name__ == "__main__":
    main()
