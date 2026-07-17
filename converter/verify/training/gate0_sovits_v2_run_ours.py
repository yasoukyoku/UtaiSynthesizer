"""SoVITS 4.0-v2 关卡0（我方侧）：用 utai_train 的 sovits_v2 预处理链跑 gate
数据集。与原版侧同参数：loudnorm ON（v2 resample.py 无条件峰值归一）、
trim top_db=20（v2 值）、f0=dio（对拍上游训练血统；产品默认 RMVPE 在
extract.py 登记）、全 CPU fp32（CUDA_VISIBLE_DEVICES=-1 在 torch import 之前）。
产物工作区 = TESTING\\sovits_v2_ours（gate1 直接复用：config 里 num_workers=0）。

    training/.venv/Scripts/python.exe converter/verify/training/gate0_sovits_v2_run_ours.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
DATASET = os.path.join(TESTING, "gate_dataset")
EXP = os.path.join(TESTING, "sovits_v2_ours")

from utai_train.protocol import Reporter
from utai_train.stopfile import StopFlag
from utai_train.rvc import train_utils
from utai_train.sovits.cluster import build_retrieval
from utai_train.sovits.flist import build_filelists
from utai_train.sovits.preprocess import slice_and_resample
from utai_train.sovits_v2 import utils
from utai_train.sovits_v2.extract import extract_all
from utai_train.sovits_v2.flist import build_config


def main():
    os.makedirs(EXP, exist_ok=True)
    train_utils.get_logger(EXP)
    reporter = Reporter(throttle_secs=1.0)
    stop = StopFlag(os.path.join(EXP, "stop.flag.never"))
    ffmpeg = r"D:\MyDev\RVC\RVC20240604Nvidia\ffmpeg.exe"

    d44k = os.path.join(EXP, "dataset_44k")
    spk_dir = os.path.join(d44k, "gate")
    # loudnorm ON + v2 trim(top_db=20): both match upstream resample.py for the
    # comparison (product defaults: loudnorm OFF, top_db=20 via the pipeline)
    slice_and_resample(DATASET, spk_dir, True, ffmpeg, reporter, stop, trim_top_db=20)

    build_config(
        EXP,
        "gate",
        2,      # total_epoch
        4,      # batch_size
        1000,   # save_every_steps (only the step-0 eval boundary fires in gate1)
        3,      # keep_ckpts
        1234,   # seed
        os.path.join(REPO, "training", "assets", "configs", "sovits_v2"),
        num_workers=0,  # gate1 reuses this workspace: num_workers=0 both sides
    )

    hps = utils.get_hparams_from_file(os.path.join(EXP, "config.json"))
    extract_all(
        d44k,
        hps,
        os.path.join(REPO, "data", "models", "auxiliary", "contentvec_256l9.onnx"),
        os.path.join(REPO, "data", "models", "training", "sovits", "rmvpe.pt"),
        "cpu",
        reporter,
        stop,
        f0_method="dio",  # gate parity vs upstream (product default: rmvpe)
    )

    build_filelists(EXP, "gate", d44k, 1234, reporter)

    build_retrieval(EXP, spk_dir, 1234, reporter, stop)
    print("GATE0 SOVITS_V2 OURS SIDE DONE")


if __name__ == "__main__":
    main()
