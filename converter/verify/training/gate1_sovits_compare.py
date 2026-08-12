# -*- coding: utf-8 -*-
"""SoVITS 4.1 关卡1 对拍:逐 step loss 轨迹 —— 原版 so-vits train.py vs 我们 vendored
sovits/train.py。

    training/.venv/Scripts/python.exe converter/verify/training/run_gate1_chain.py sovits

⛔ **别再直接手敲这个脚本**:它现在要求跑器钉的 `GATE1_T0`(新鲜度),没有它会
   响亮地判 exit 3(不可归因),而不是给你一个没有意义的绿 —— S139 实测:
   拿一份 2026-07-07 的 jsonl 喂给八月的参照,这条链打出 `ALL PASS (16 steps compared)`。

两侧同数据/同 filelist 行序/同 seed(1234)/同底模(G_0/D_0 vec768)/fp32 CPU(确定性)。
原版侧取 tensorboard events(全精度),我方侧取协议 JSONL。
⚠ **torch 轴**:此前写的「双方同 torch(2.5.1)」是**陈货** —— 经跑器跑时这条链用的是
   `envs\\s42_staging_nv_cu130`(torch 2.11.0+cu130),两侧同一个。实际版本打进转录。

与 RVC 关卡1 的差异:so-vits **没有** mel>75/kl>9 的显示夹取(TB 写的就是原始值)
⇒ 这条链**没有夹取致盲**,每一步都真的可证伪;且多一个 loss/g/lf0 分量
(自动 f0 预测,模板默认开)。
通过线 max 相对差 ≤1e-3(结构性移植错误 = O(0.1~1);实测期望 ~1e-6 级)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402

CHAIN = "sovits"
GATE = "GATE1 SOVITS"
ORIG_TB_DIR = r"D:\MyDev\so-vits-svc\so-vits-svc\logs\gate1_sovits"
OURS_JSONL = r"D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_ours_steps.jsonl"

PAIRS = [  # (TB tag, ours key) — NO clamps (upstream writes raw values)
    ("loss/g/total", "g_total"),
    ("loss/d/total", "d_total"),
    ("loss/g/fm", "fm"),
    ("loss/g/mel", "mel"),
    ("loss/g/kl", "kl"),
    ("loss/g/lf0", "lf0"),
]
MAX_REL = 1e-3


def main():
    allow_uncovered = "--allow-uncovered" in sys.argv
    t0 = G1.read_t0(GATE)
    orig_frozen = "orig" in G1.skipped_stages()
    G1.header(GATE, CHAIN, [("orig TB", ORIG_TB_DIR), ("ours JSONL", OURS_JSONL)])
    G1.say_input_identity([("orig", ORIG_TB_DIR),
                           ("ours", r"D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_ours")])

    orig = G1.tb_scalars(
        "orig/TB", ORIG_TB_DIR, [t for t, _k in PAIRS], t0,
        frozen_why=("--skip-orig:参照侧本轮**故意没有重跑**,按冻结参照记账"
                    if orig_frozen else None))
    if orig_frozen:
        G1.note_uncovered("参照侧未重跑(--skip-orig)",
                          "这一轮只证明了我方侧与**上一次**跑出来的参照一致")

    ours, _nones = G1.jsonl_steps("ours/JSONL", OURS_JSONL, "g_total", t0)
    exp = G1.EXPECT[CHAIN]["steps"]
    steps = G1.require_exact_steps("ours/JSONL", CHAIN, ours, exp,
                                   other=orig["loss/g/total"], other_label="orig/TB")
    # ⛔ 发现 None(= 非有限 = 发散)立刻收尾,不许往下走进算术 —— 否则抛 TypeError
    #    而被归到「闸自己炸了」(exit 3),掩盖掉「被测的东西发散了」(exit 1)。
    failures = G1.require_no_none("ours/JSONL", ours, [k for _t, k in PAIRS])
    if failures:
        G1.finish(GATE, failures, allow_uncovered=allow_uncovered)

    for tag, key in PAIRS:
        rels = [abs(orig[tag][s] - ours[s][key]) / max(abs(orig[tag][s]), 1e-6) for s in steps]
        arr = np.array(rels)
        worst = steps[int(arr.argmax())]
        ok = arr.max() <= MAX_REL
        G1._say("[%s] %14s vs %7s: max_rel=%.3e @step %d, mean_rel=%.3e  (%d/%d 步可比)"
                % ("PASS" if ok else "FAIL", tag, key, arr.max(), worst, arr.mean(),
                   len(rels), len(steps)))
        if not ok:
            failures.append(tag)

    G1.finish(GATE, failures, allow_uncovered=allow_uncovered)


if __name__ == "__main__":
    G1.run(GATE, main)
