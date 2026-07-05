"""关卡1 对拍：逐 step loss 轨迹 —— 原版 train.py vs 我们 vendored train()。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_compare.py

两侧同 torch(2.5.1)/同数据/同序(1234)/同底模/fp32 CPU（确定性）。原版侧取
tensorboard events（全精度；stdout 只有 3 位小数），我方侧取协议 JSONL。
注意原版把 mel>75 / kl>9 夹到上限后才写 TB —— 比较时对我方值施加同一夹取。
结构性移植错误（损失权重/数据顺序/模型接线）会造成 O(0.1~1) 的相对差；
通过线设在 max 相对差 ≤1e-3（实测期望 ~1e-6 级）。
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ORIG_TB_DIR = r"D:\MyDev\RVC\RVC20240604Nvidia\logs\gate1"
OURS_JSONL = r"D:\MyDev\TESTING\utai-v2-testing\gate1_ours_steps.jsonl"

PAIRS = [  # (TB tag, ours key, clamp)
    ("loss/g/total", "g_total", None),
    ("loss/d/total", "d_total", None),
    ("loss/g/fm", "fm", None),
    ("loss/g/mel", "mel", 75.0),
    ("loss/g/kl", "kl", 9.0),
]


def load_orig():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(ORIG_TB_DIR, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag, _, _ in PAIRS:
        out[tag] = {e.step: e.value for e in acc.Scalars(tag)}
    return out


def load_ours():
    steps = {}
    with open(OURS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "step" and "g_total" in obj.get("losses", {}):
                steps[obj["step"]] = obj["losses"]
    return steps


def main():
    orig = load_orig()
    ours = load_ours()
    common = sorted(set(orig["loss/g/total"]) & set(ours))
    print(f"orig steps={len(orig['loss/g/total'])} ours steps={len(ours)} common={len(common)}")
    if len(common) < 10:
        print("GATE1: FAIL — 对齐步数不足")
        sys.exit(1)

    failures = []
    for tag, key, clamp in PAIRS:
        rels = []
        for s in common:
            a = orig[tag][s]
            b = ours[s][key]
            if clamp is not None:
                b = min(b, clamp)
            denom = max(abs(a), 1e-6)
            rels.append(abs(a - b) / denom)
        rels = np.array(rels)
        worst = common[int(rels.argmax())]
        ok = rels.max() <= 1e-3
        print(
            f"[{'PASS' if ok else 'FAIL'}] {tag:>14} vs {key:>7}: max_rel={rels.max():.3e} @step {worst}, mean_rel={rels.mean():.3e}"
        )
        if not ok:
            failures.append(tag)

    print()
    if failures:
        print("GATE1: FAIL —", ", ".join(failures))
        sys.exit(1)
    print(f"GATE1: ALL PASS ({len(common)} steps compared)")


if __name__ == "__main__":
    main()
