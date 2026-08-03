"""Flask Web 服务 (env-main, Python 3.12)。

提供:
- 单页前端 (frontend/index.html)
- /api/process : 上传视频 + 选择模式, 后台启动流水线, 返回 job_id
- /api/status  : 轮询进度
- /api/download: 下载结果(保留原声)
"""
import json
import os
import threading
import time
import uuid

from flask import Flask, request, jsonify, send_file, send_from_directory
import config
import pipeline as pipe

BASE = config.BASE_DIR
FRONTEND = os.path.join(BASE, "frontend")
app = Flask(__name__, static_folder=None)

JOBS = {}


def write_status(job_id, pct, stage, msg, result=None, error=None):
    path = os.path.join(config.JOBS_DIR, job_id, "status.json")
    state = {
        "job_id": job_id, "progress": pct, "stage": stage,
        "message": msg, "done": stage == "done",
        "error": error, "result": result,
    }
    with open(path, "w") as f:
        json.dump(state, f)
    return state


def progress_cb(job_id):
    def cb(pct, stage, msg, result=None):
        write_status(job_id, pct, stage, msg, result=result)
    return cb


def worker(job_id, video_path, mode):
    try:
        pipe.run_pipeline(job_id, video_path, mode, progress_cb(job_id))
    except Exception as e:
        write_status(job_id, 0, "error", f"处理失败: {e}", error=str(e))


@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND, "index.html"))


@app.route("/api/modes")
def modes():
    return jsonify(config.MODES)


@app.route("/api/process", methods=["POST"])
def process():
    if "video" not in request.files:
        return jsonify({"error": "缺少视频文件"}), 400
    mode = request.form.get("mode", "depth")
    if mode not in config.MODES:
        return jsonify({"error": "未知模式"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "空文件"}), 400
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(config.JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    ext = os.path.splitext(f.filename)[1] or ".mp4"
    video_path = os.path.join(job_dir, "input" + ext)
    f.save(video_path)
    write_status(job_id, 1, "queued", "已接收, 准备处理")
    t = threading.Thread(target=worker, args=(job_id, video_path, mode), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    path = os.path.join(config.JOBS_DIR, job_id, "status.json")
    if not os.path.exists(path):
        return jsonify({"error": "任务不存在"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/download/<job_id>")
def download(job_id):
    final = os.path.join(config.JOBS_DIR, job_id, "result.mp4")
    if not os.path.exists(final):
        return jsonify({"error": "结果尚未生成"}), 404
    return send_file(final, as_attachment=True,
                     download_name=f"deep_video_{job_id}.mp4",
                     mimetype="video/mp4")


@app.route("/healthz")
def health():
    return "ok"


if __name__ == "__main__":
    config.ensure_dirs()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
