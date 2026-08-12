"""SoVITS 4.0-v2 关卡1（我方侧）：vendored sovits_v2 训练循环，协议 JSONL 打
stdout —— 调用方重定向 > TESTING\\gate1_sovits_v2_ours_steps.jsonl。
skip_optimizer=True = 上游原生 cpurun 语义（train.py 头注登记的 gate 旗）。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_v2_run_ours.py ^
        > D:\\MyDev\\TESTING\\utai-v2-testing\\gate1_sovits_v2_ours_steps.jsonl
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

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
EXP = os.path.join(TESTING, "gate1_sovits_v2_ours")


def main():
    from utai_train.protocol import Reporter
    from utai_train.stopfile import StopFlag
    from utai_train.rvc import train_utils
    from utai_train.sovits_v2.train import train

    train_utils.get_logger(EXP)
    reporter = Reporter(throttle_secs=0.0)  # every step, unthrottled
    stop = StopFlag(os.path.join(EXP, "stop.flag.never"))
    cfg = {
        "model_slug": "gate",
        "model_name": "gate",
        "workspace": EXP,
        "dataset_dir": "",  # resolve_speakers' single-speaker fallback reads it
        "skip_optimizer": True,  # upstream cpurun semantics (see train.py header)
    }
    # S134: 3rd positional `pool_dir` (sovits_v2/train.py:197). Position matters — appending would
    # keep the arity right and misbind every later parameter silently.
    summary = train(cfg, EXP, EXP, reporter, stop)
    print("SUMMARY %s" % summary, file=sys.stderr)


if __name__ == "__main__":
    main()
