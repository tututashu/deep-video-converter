"""复用 e2e_all 任务已保存的各层帧(orig/depth/pose/face), 重新合成 depth/pose/combo/face
四种模式并封装(保留原声), 以验证这四种模式的合成与音视频封装均可用。
用法(在 env-main 中):
  venvs/env-main/bin/python scripts/recomposite_test.py --job e2e_all --video tmp/test_clip.mp4 --out tmp/e2e_recomp
"""
import os
import sys
import shutil
import subprocess
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import config
import pipeline  # 复用 composite_frame / probe_video, 避免与主流水线逻辑漂移


def has_audio(video):
    return pipeline.probe_video(video)["has_audio"]


def encode_mux(frames_dir, video, out, fps):
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    silent = out + ".silent.mp4"
    subprocess.run([ffmpeg, "-y", "-framerate", f"{fps:.3f}", "-i",
                   os.path.join(frames_dir, "out_%06d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", silent],
                  check=True, capture_output=True)
    if has_audio(video):
        mux = [ffmpeg, "-y", "-i", silent, "-i", video, "-map", "0:v:0",
               "-map", "1:a:0", "-c", "copy", "-shortest", out]
        if subprocess.run(mux, capture_output=True).returncode != 0:
            shutil.copy(silent, out)
    else:
        shutil.copy(silent, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="e2e_all")
    ap.add_argument("--video", default="tmp/test_clip.mp4")
    ap.add_argument("--out", default="tmp/e2e_recomp")
    a = ap.parse_args()

    job_dir = os.path.join(config.JOBS_DIR, a.job)
    orig = os.path.join(job_dir, "orig_frames")
    depth = os.path.join(job_dir, "depth_frames")
    pose = os.path.join(job_dir, "pose_frames")
    face = os.path.join(job_dir, "face_frames")
    info = pipeline.probe_video(a.video)
    fps = info["fps"]
    os.makedirs(a.out, exist_ok=True)

    frames = sorted(f for f in os.listdir(orig) if f.startswith("orig_"))
    for mode in ["depth", "pose", "combo", "face"]:
        od = os.path.join(a.out, mode)
        os.makedirs(od, exist_ok=True)
        for f in frames:
            idx = f.split("_")[1].split(".")[0]
            o = cv2.imread(os.path.join(orig, f))
            if o is None:
                continue
            dg = cv2.imread(os.path.join(depth, f"depth_{idx}.png"), cv2.IMREAD_GRAYSCALE)
            pb = cv2.imread(os.path.join(pose, f"pose_{idx}.png"), cv2.IMREAD_UNCHANGED)
            fb = cv2.imread(os.path.join(face, f"face_{idx}.png"), cv2.IMREAD_UNCHANGED)
            if dg is None:
                dg = np.zeros((o.shape[0], o.shape[1]), np.uint8)
            if pb is None:
                pb = np.zeros((o.shape[0], o.shape[1], 4), np.uint8)
            if fb is None:
                fb = np.zeros((o.shape[0], o.shape[1], 4), np.uint8)
            out = pipeline.composite_frame(mode, o, dg, pb, fb)
            cv2.imwrite(os.path.join(od, f"out_{idx}.png"), out)
        out_path = os.path.join(a.out, f"result_{mode}.mp4")
        encode_mux(od, a.video, out_path, fps)
        sz = os.path.getsize(out_path)
        print(f"MODE {mode:6s} -> {out_path}  ({sz/1024:.0f} KB)", flush=True)
    print("RECOMPOSITE_DONE")


if __name__ == "__main__":
    main()
