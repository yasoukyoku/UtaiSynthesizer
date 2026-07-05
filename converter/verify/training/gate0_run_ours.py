"""关卡0（我方侧）：用 utai_train 的 RVC 预处理链跑 gate 数据集。

对照物 = 原版 RVC 脚本在其自带 runtime 里的产物（gate0 的原版侧命令见
README）。运行（必须用 training/.venv 的 python，cwd 任意）：

    training/.venv/Scripts/python.exe converter/verify/training/gate0_run_ours.py
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
DATASET = os.path.join(TESTING, "gate_dataset")
EXP = os.path.join(TESTING, "rvc_ours")
RVC_REPO = r"D:\MyDev\RVC\RVC20240604Nvidia"

from utai_train.protocol import Reporter
from utai_train.stopfile import StopFlag
from utai_train.rvc.preprocess import preprocess_trainset
from utai_train.rvc.extract_f0 import extract_f0
from utai_train.rvc.extract_feature import extract_features
from utai_train.rvc.filelist import build_filelist_and_config
from utai_train.rvc.index_npy import build_index
from utai_train.rvc import train_utils

def main():
    os.makedirs(EXP, exist_ok=True)
    train_utils.get_logger(EXP)
    reporter = Reporter(throttle_secs=1.0)
    stop = StopFlag(os.path.join(EXP, "stop.flag.never"))
    ffmpeg = os.path.join(RVC_REPO, "ffmpeg.exe")  # same decoder as the original run

    preprocess_trainset(DATASET, 48000, EXP, 3.7, ffmpeg, reporter, stop)
    extract_f0(
        EXP,
        os.path.join(REPO, "data", "models", "aux", "rmvpe.pt"),
        "cuda",
        True,  # original passes a truthy string -> always half on NVIDIA
        ffmpeg,
        reporter,
        stop,
    )
    extract_features(
        EXP,
        "v2",
        os.path.join(REPO, "data", "models", "aux", "contentvec_768l12.onnx"),
        reporter,
        stop,
    )
    build_index(EXP, "v2", 1234, reporter)
    build_filelist_and_config(
        EXP,
        "48k",
        "v2",
        0,
        os.path.join(REPO, "training", "assets", "configs", "rvc"),
        os.path.join(REPO, "training", "assets", "mute"),
        1234,
        True,
        reporter,
    )
    print("OURS-DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
