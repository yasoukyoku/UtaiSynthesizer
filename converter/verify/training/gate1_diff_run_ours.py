"""gate1_diff（我方侧）：走真实驱动 utai_train.sovits.diff_pipeline._train_diff
（内部自播种 random/torch = 与原版侧 harness 同点同种子），total_steps=24 =
原版 3 epochs 的自然长度 —— 完成判定与自然结束重合，训练步序列一致。

运行（our venv）：
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate1_diff_run_ours.py ^
        > D:\\MyDev\\TESTING\\utai-v2-testing\\gate1_diff_ours_steps.jsonl
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⛔ S139: **运行期**确认 CPU-only 真的生效 —— 十个 gate1 跑器里此前只有 `gate1_run_orig.py`
#    有这条硬拒绝，而那是唯一一个**不是被测对象**的臂。只写 stderr（stdout 是协议流）。
import gate1_guard as _G1  # noqa: E402
_G1.assert_cpu_only(__file__)

UTAI = r"D:\MyDev\Utai_v2-dev"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
WS = os.path.join(TESTING, "gate1_diff_ours_ws")

sys.path.insert(0, os.path.join(UTAI, "training"))


class _Stop:
    def requested(self):
        return False

    def check(self):
        pass


def main():
    from utai_train.protocol import Reporter
    from utai_train.sovits.diff_pipeline import _train_diff

    reporter = Reporter(throttle_secs=0.0)  # every step on the wire for the gate
    cfg = {"seed": 1234, "total_steps": 24}
    # S134: 3rd positional `pool_dir` (diff_pipeline.py:630 `_train_diff(cfg, run_dir, pool_dir,
    # reporter, stop)`). Position matters — appending keeps the arity right and misbinds silently.
    summary = _train_diff(cfg, WS, WS, reporter, _Stop())
    reporter.done("completed", {k: v for k, v in summary.items() if k != "stopped"})
    sys.stderr.write("GATE1 DIFF OURS SIDE DONE\n")


if __name__ == "__main__":
    main()
