"""SoVITS 4.0-v2 关卡1 对拍：原版 tensorboard events vs 我方 JSONL 步流。
九分量（v2 的 TB tag → 我方 losses 键），无 clamp（上游 TB 写原始值）。
结构性移植错误 = O(0.1~1) 的 rel；期望 ~1e-6 级（同 torch/CPU/fp32/seed/RNG 流）。
对齐步数 < 10 直接 FAIL（防空交集假 PASS，红队 A11）。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_v2_compare.py
"""
import json
import os
import sys

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SOVITS_V2 = r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc"
ORIG_TB_DIR = os.path.join(SOVITS_V2, "logs", "gate1_sovits_v2")
OURS_JSONL = os.path.join(TESTING, "gate1_sovits_v2_ours_steps.jsonl")

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

PAIRS = [
    ("loss/total", "g_total"),
    ("loss/mel", "mel"),
    ("loss/adv", "adv"),
    ("loss/fm", "fm"),
    ("loss/mel_ddsp", "mel_ddsp"),
    ("loss/spec_ddsp", "spec_ddsp"),
    ("loss/mel_am", "mel_am"),
    ("loss/kl_div", "kl"),
    ("loss/lf0", "lf0"),
]
MAX_REL = 1e-3


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(ORIG_TB_DIR, size_guidance={"scalars": 0})
    acc.Reload()
    orig = {}
    for tag, _ in PAIRS:
        for ev in acc.Scalars(tag):
            orig.setdefault(ev.step, {})[tag] = ev.value

    ours = {}
    with open(OURS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") == "step" and msg.get("losses", {}).get("g_total") is not None:
                ours[msg["step"]] = msg["losses"]

    steps = sorted(set(orig) & set(ours))
    print("orig steps: %d, ours steps: %d, aligned: %d" % (len(orig), len(ours), len(steps)))
    if len(steps) < 10:
        print("[FAIL] 对齐步数 < 10 —— 空交集假 PASS 防线")
        sys.exit(1)

    worst = (0.0, "", -1)
    for s in steps:
        for tag, key in PAIRS:
            a = orig[s].get(tag)
            b = ours[s].get(key)
            if a is None or b is None:
                print("[FAIL] step %d 缺分量 %s/%s" % (s, tag, key))
                sys.exit(1)
            rel = abs(a - b) / max(abs(a), 1e-6)
            if rel > worst[0]:
                worst = (rel, tag, s)
    ok = worst[0] <= MAX_REL
    print(
        "[%s] GATE1 SOVITS_V2: %d steps x %d 分量, max_rel=%.3e (%s @ step %d), 线=%.0e"
        % ("PASS" if ok else "FAIL", len(steps), len(PAIRS), worst[0], worst[1], worst[2], MAX_REL)
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
