"""SoVITS 关卡1（我方侧）：vendored 训练循环，CPU fp32，协议 JSONL 全量输出。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_run_ours.py ^
        > D:\\MyDev\\TESTING\\utai-v2-testing\\gate1_sovits_ours_steps.jsonl
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⛔ S139: **运行期**确认 CPU-only 真的生效 —— 十个 gate1 跑器里此前只有 `gate1_run_orig.py`
#    有这条硬拒绝，而那是唯一一个**不是被测对象**的臂。只写 stderr（stdout 是协议流）。
import gate1_guard as _G1  # noqa: E402
_G1.assert_cpu_only(__file__)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

EXP = r"D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_ours"

from utai_train.protocol import Reporter
from utai_train.stopfile import StopFlag
from utai_train.rvc import train_utils
from utai_train.sovits.train import train


def main():
    train_utils.get_logger(EXP)
    reporter = Reporter(throttle_secs=0.0)  # every step, no throttle
    stop = StopFlag(os.path.join(EXP, "stop.flag.never"))
    cfg = {
        "model_slug": "gate1_sovits",
        "model_name": "gate1_sovits",
        "workspace": EXP,
        "dataset_dir": "",  # resolve_speakers' single-speaker fallback reads it (flist.py:86,
        # hard-subscript); _write_release_config runs at train.py:306, BEFORE the loop at :378 ⇒
        # without this key the run dies with KeyError before step 0. Same key the v2 driver
        # already carries — 4.1's driver was simply never updated alongside it.
    }
    # S134: 3rd positional `pool_dir` (sovits/train.py:188). Position matters — see gate1_run_ours.
    summary = train(cfg, EXP, EXP, reporter, stop)
    print("SUMMARY", summary, file=sys.stderr)


if __name__ == "__main__":
    main()
