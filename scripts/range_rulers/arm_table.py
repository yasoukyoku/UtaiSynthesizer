# -*- coding: utf-8 -*-
"""S159za —— **消融臂比较台**:拿两把**在用户 ground truth 上验过阳性**的尺子给每条臂打分。

## ⛔ 为什么只用这两把

用户 2026-08-22 给了 12 个亲耳确认的坐标。我在这一场里提过四条机理假说
(周期倍化 / 半整数子谐波 / donor 基频泄漏 / 不连续度),**四条全部被自己的阴性对照打掉**。
⇒ 所以这里**不用任何机理量**,只用两把**排序上被 ground truth 背书过**的尺子:

* `grain_glitch`  —— 逐周期峰值 / 邻居中位。6/6 咔哒进前 **0.7%**(130505 个候选)。
* `donor_leak`    —— 输出基频以下 0.25-0.80·f0 的能量占比。4/4 面状进前 **1.9%**(4081 个窗)。

⚠ 两把尺子**都还没有可闻性刻度**(不知道降多少 dB 才听得出来)。
   ⇒ 它们只用来**排序候选臂**;最终仍然由耳朵拍板(S146 协议:盲测在前,翻默认在后)。

## ⛔ 比较纪律
* 两条臂只许差**一个**自由度(S158 血训)。
* 必须带**阴性对照段**(不在救援窗的唱音):若某条臂在对照段上也动了,
  那它动的不是这个缺陷,是全局的东西。
* 出厂臂必须在表里,而且**它的读数要与用户听到的那一版对得上**。

用法:`python arm_table.py <目录> [<目录2> ...] --model ya|ak`
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

TRUTH = {
    # 模型 → (咔哒坐标, 面状段, 阴性对照坐标, 阴性对照段)
    "ya": (
        [42.675, 54.302, 115.398, 119.873, 126.420, 127.039],
        ["268.857-269.000"],
        [22.7, 82.1, 240.0],
        ["22.60-22.85", "82.00-82.25", "240.0-240.25"],
    ),
    "ak": (
        [129.376],
        ["268.888-269.075", "130.825-131.094", "131.349-131.618", "132.688-132.856"],
        [22.7, 82.1, 240.0],
        ["22.60-22.85", "82.00-82.25", "240.0-240.25"],
    ),
}


def run(script, wav, flag, arg):
    out = subprocess.run(
        [PY, os.path.join(HERE, script), wav, flag, arg],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "(空)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--model", default="ya", choices=["ya", "ak"])
    a = ap.parse_args()
    clicks, sheets, cctrl, sctrl = TRUTH[a.model]

    wavs = []
    for d in a.dirs:
        wavs += sorted(glob.glob(os.path.join(d, "*.wav")))
    if not wavs:
        sys.exit("没有找到 wav")

    print(f"### 咔哒族 —— grain_glitch(逐周期跳变 dB,**越小越好**)· 模型 {a.model}")
    print(f"{'臂':<46}" + "".join(f"{c:>8.1f}" for c in clicks) + "   | 中位/合计")
    for w in wavs:
        print("  " + run("grain_glitch.py", w, "--at", ",".join(str(c) for c in clicks)))
    print(f"\n  ⛔ 阴性对照(不在救援窗):")
    for w in wavs:
        print("  " + run("grain_glitch.py", w, "--at", ",".join(str(c) for c in cctrl)))

    print(f"\n### 面状族 —— donor_leak(基频以下占比 dB,**越负越好**)")
    print(f"{'臂':<44}" + "".join(f"{s.split('-')[0][:7]:>8}" for s in sheets) + "   | 中位")
    for w in wavs:
        print("  " + run("donor_leak.py", w, "--at", ",".join(sheets)))
    print(f"\n  ⛔ 阴性对照:")
    for w in wavs:
        print("  " + run("donor_leak.py", w, "--at", ",".join(sctrl)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
