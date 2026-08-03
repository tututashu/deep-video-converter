"""端到端驱动: 以较低分辨率跑通 mode='all' 的完整流水线(面部子进程 + 深度 + 姿态 + 合成 + 封装)。
用法(在 env-main 中, 且同一时刻不要运行其它重进程):
  venvs/env-main/bin/python scripts/run_e2e_all.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import config
import pipeline

config.MAX_DIM = 320  # 测试用低分辨率, 降低显存/内存与耗时
JOB = "e2e_all"
VIDEO = "tmp/test_clip.mp4"


def cb(p, s, m, result=None):
    print(f"[{s}] {p:3d}%  {m}", flush=True)


if __name__ == "__main__":
    out = pipeline.run_pipeline(JOB, VIDEO, "all", cb)
    print("E2E_ALL_DONE ->", out)
