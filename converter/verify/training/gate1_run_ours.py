"""关卡1（我方侧）：在 gate1_ours 工作区上直接跑 utai_train 的 train()（跳过预处理，
产物已由 gate1_prepare.py 布置）。stdout 的 JSONL step 流重定向进文件供 compare 用。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_run_ours.py > gate1_ours_steps.jsonl
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU 确定性（注意 Windows 空值=删除变量）

from utai_train.protocol import Reporter
from utai_train.stopfile import StopFlag
from utai_train.rvc import train_utils
from utai_train.rvc.train import train

EXP = r"D:\MyDev\TESTING\utai-v2-testing\gate1_ours"
RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"

CFG = {
    "workspace": EXP,
    "model_slug": "gate1",
    "sample_rate": "48k",
    "version": "v2",
    "total_epoch": 2,
    "batch_size": 4,
    "save_every_epoch": 1,
    "save_every_weights": True,
    "keep_only_latest": True,
    "cache_gpu": False,
    "fp16": False,
    "seed": 1234,
    "pretrain_g": os.path.join(RVC, "assets", "pretrained_v2", "f0G48k.pth"),
    "pretrain_d": os.path.join(RVC, "assets", "pretrained_v2", "f0D48k.pth"),
}


def main():
    train_utils.get_logger(EXP)
    reporter = Reporter(throttle_secs=0.0)  # 每 step 全量 emit（关卡用）
    stop = StopFlag(os.path.join(EXP, "stop.flag.never"))
    summary = train(CFG, EXP, reporter, stop)
    print("SUMMARY:", summary, file=sys.stderr)


if __name__ == "__main__":
    main()
