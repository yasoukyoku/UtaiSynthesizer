"""SoVITS 关卡0（我方侧）：用 utai_train 的 SoVITS 预处理链跑 gate 数据集。
与原版侧同参数：loudnorm ON（原版 resample.py 默认）、vec768l12、vol_embedding ON、
全 CPU fp32（CUDA_VISIBLE_DEVICES=-1 在 torch import 之前）。

    training/.venv/Scripts/python.exe converter/verify/training/gate0_sovits_run_ours.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
DATASET = os.path.join(TESTING, "gate_dataset")
EXP = os.path.join(TESTING, "sovits_ours")

from utai_train.protocol import Reporter
from utai_train.stopfile import StopFlag
from utai_train.rvc import train_utils
from utai_train.sovits import utils
from utai_train.sovits.cluster import build_retrieval
from utai_train.sovits.extract import extract_all
from utai_train.sovits.flist import build_flist_and_config
from utai_train.sovits.preprocess import slice_and_resample


def main():
    os.makedirs(EXP, exist_ok=True)
    train_utils.get_logger(EXP)
    reporter = Reporter(throttle_secs=1.0)
    stop = StopFlag(os.path.join(EXP, "stop.flag.never"))
    ffmpeg = r"D:\MyDev\RVC\RVC20240604Nvidia\ffmpeg.exe"

    d44k = os.path.join(EXP, "dataset_44k")
    spk_dir = os.path.join(d44k, "gate")
    slice_and_resample(DATASET, spk_dir, True, ffmpeg, reporter, stop)  # loudnorm ON

    build_flist_and_config(
        EXP,
        "gate",
        d44k,
        "vec768l12",
        True,   # vol_embedding
        False,  # fp16
        2,      # total_epoch
        4,      # batch_size
        800,    # save_every_steps
        3,      # keep_ckpts
        True,   # all_in_mem (gate1 reuses this workspace: num_workers=0 both sides)
        1234,
        os.path.join(REPO, "training", "assets", "configs", "sovits"),
        reporter,
    )

    hps = utils.get_hparams_from_file(os.path.join(EXP, "config.json"))
    extract_all(
        d44k,
        hps,
        os.path.join(REPO, "data", "models", "auxiliary", "contentvec_768l12.onnx"),
        os.path.join(REPO, "data", "models", "training", "sovits", "rmvpe.pt"),
        "cpu",
        reporter,
        stop,
    )

    build_retrieval(EXP, spk_dir, 1234, reporter, stop)
    print("GATE0 SOVITS OURS SIDE DONE")


if __name__ == "__main__":
    main()
