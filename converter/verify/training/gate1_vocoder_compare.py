# -*- coding: utf-8 -*-
"""gate1_vocoder_compare — 双侧 TB 标量逐步对拍（S40）。

判定（红队 A11：空/短交集一律 FAIL——compare 必须先断言点数符合预期）：
  - training/* 每分量点数 == 12（log_interval=1，12 个 train batch 全在场）
  - validation/total_loss 点数 >= 3（sanity@0 + val@5 + val@10）
  - 双侧 key 集合一致；每 (key, step) 值 max_rel <= 1e-7（TB f32 序列化噪声轴；
    期望逐位 0.0——同库同版同 RNG 流）
"""
import math
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

GATE = pathlib.Path(r"D:\MyDev\TESTING\gate1_vocoder")
TOL = 1e-7
EXPECT_TRAIN_POINTS = 15
EXPECT_VAL_MIN = 4

ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        ok = False


ORIG_WORK = pathlib.Path(r"D:\MyDev\SingingVocoders\experiments\gate1_voc")


def load_scalars(side):
    # orig runs inside the repo tree (its DsModelCheckpoint asserts cwd is an
    # ancestor of work_dir — see run_orig header); ours runs in the gate dir
    base = ORIG_WORK if side == "orig" else GATE / "ours" / "gate1_voc"
    logdir = base / "lightning_logs" / "lastest"
    acc = EventAccumulator(str(logdir))
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        out[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    return out


def main():
    a = load_scalars("orig")
    b = load_scalars("ours")
    tags_a = {t for t in a if t.startswith(("training/", "validation/"))}
    tags_b = {t for t in b if t.startswith(("training/", "validation/"))}
    check(tags_a == tags_b, f"tag sets identical ({len(tags_a)} tags)")
    if tags_a != tags_b:
        print("  only orig:", sorted(tags_a - tags_b))
        print("  only ours:", sorted(tags_b - tags_a))

    for tag in sorted(tags_a & tags_b):
        pa, pb = a[tag], b[tag]
        if tag.startswith("training/"):
            check(len(pa) == EXPECT_TRAIN_POINTS and len(pb) == EXPECT_TRAIN_POINTS,
                  f"{tag}: point count {len(pa)}/{len(pb)} == {EXPECT_TRAIN_POINTS}")
        else:
            check(len(pa) >= EXPECT_VAL_MIN and len(pa) == len(pb),
                  f"{tag}: point count {len(pa)}/{len(pb)} >= {EXPECT_VAL_MIN}")
        steps_a = [s for s, _ in pa]
        steps_b = [s for s, _ in pb]
        check(steps_a == steps_b, f"{tag}: step axes identical {steps_a[:6]}...")
        worst = 0.0
        for (sa, va), (sb, vb) in zip(pa, pb):
            # 审查修复: python max() silently drops NaN operands — a NaN loss on
            # either side must FAIL, never vanish into a green max_rel
            if not (math.isfinite(va) and math.isfinite(vb)):
                check(False, f"{tag}@{sa}: non-finite value ({va} vs {vb})")
                continue
            denom = max(abs(va), abs(vb), 1e-12)
            worst = max(worst, abs(va - vb) / denom)
        check(worst <= TOL, f"{tag}: max_rel {worst:.3e} <= {TOL:.0e}")

    print("\n=== gate1_vocoder:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
