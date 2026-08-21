# -*- coding: utf-8 -*-
"""S159za —— **消融臂比较台**:两把在用户 ground truth 上验过阳性的尺子 + 群体统计 + 阴性对照。

## ⛔ 为什么只用这两把、而且只当【排序】用

用户 2026-08-22 给了 12 个亲耳确认的坐标。我在这一场里提过**五条**机理假说
(周期倍化 / 半整数子谐波 / donor 基频泄漏 / 不连续度 / 鼻音次基频残留),
**五条全部被自己的阴性对照打掉**。⇒ 这里**不用任何机理量**,只用排序上被 ground truth 背书的:

* `grain_glitch`  —— 逐周期峰值 / 邻居中位。6/6 咔哒进前 **0.7%**(130505 候选,偶然 ~1e-13)。
* `donor_leak`    —— 输出基频以下 0.25-0.80·f0 的能量占比。4/4 面状进前 **1.9%**(4081 窗,~1e-7)。

⚠ 两把都**没有可闻性刻度** ⇒ 只排序候选臂,最终由耳朵拍板(S146 协议)。

## 三层读数(缺一层都可能读错)
1. **点读**:用户点名的那几处 —— 直接、但 n 只有 6-7,噪声大。
2. **群体**:深窗(|位移| ≥12)内**全部**候选的分位数 —— n 上百,功效高。
3. ⛔ **阴性对照**:救援窗**之外**的同一批统计。**某条臂若在对照上也动了,
   它动的就不是这个缺陷,是全局的东西** —— 那种「改善」不算数(S159za 的 VALLEY=0 就是这样)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from donor_leak import leak_at  # noqa: E402
from grain_glitch import glitch_scan, lead_silence, load  # noqa: E402

SCORE = r"D:\MyDev\TESTING\s145_range_color\lch\lch_score.json"

TRUTH = {
    "ya": ([42.675, 54.302, 115.398, 119.873, 126.420, 127.039], ["268.857-269.000"]),
    "ak": ([129.376], ["268.888-269.075", "130.825-131.094", "131.349-131.618", "132.688-132.856"]),
}


def windows(plan_path, score_path):
    tri = json.load(open(score_path, encoding="utf-8"))["triples"]
    T = np.concatenate([[0.0], np.cumsum([max(0, x["frames"]) / 50.0 for x in tri])])
    d = json.load(open(plan_path, encoding="utf-8"))
    g = d.get("groups")
    if g is None:
        g = next(x for x in d["arms"] if "出厂" in x["label"])["groups"]
    return [(T[a], T[min(b + 1, len(T) - 1)], s) for a, b, s in g]


def stats(wav, wins, clicks, sheets):
    y, sr = load(wav)
    off = lead_silence(y, sr)
    deep = [(a, b) for a, b, s in wins if abs(s) >= 12]

    def in_deep(t):
        return any(a <= t - off < b for a, b in deep)

    def in_any(t):
        return any(a <= t - off < b for a, b, _ in wins)

    g = glitch_scan(y, sr)
    gd = np.array([h["score"] for h in g if in_deep(h["t"])])
    go = np.array([h["score"] for h in g if not in_any(h["t"])])
    pt = []
    for c in clicks:
        near = [h["score"] for h in g if abs(h["t"] - off - c) <= 0.080]
        pt.append(max(near) if near else 0.0)

    lk_d, lk_o = [], []
    t = 0.0
    while t + 0.15 < len(y) / sr:
        v, _ = leak_at(y, sr, t, t + 0.15)
        if not np.isnan(v):
            (lk_d if in_deep(t + 0.075) else (lk_o if not in_any(t + 0.075) else [])).append(v)
        t += 0.10
    sh = []
    for s in sheets:
        a, b = [float(x) for x in s.split("-")]
        v, _ = leak_at(y, sr, a + off, b + off)
        sh.append(v)
    # ⛔⛔ S159za —— **电平轴,缺它这张表会骗人**。
    # 实测:`UTAI_PSOLA_ENVFIX=0` 让两把尺子的读数全线变好(点读 92.6→76.6、泄漏 −22.98→−27.12),
    # 而真相是那几处**掉了 16-22 dB 几乎消失了** —— 安静的区域当然没有跳变、也没有次基频。
    # ⇒ 任何一条臂在缺陷处的电平相对出厂动了 >3 dB,它的「改善」都必须先打问号。
    lv = []
    for c in clicks:
        i0, i1 = int((c + off - 0.025) * sr), int((c + off + 0.025) * sr)
        seg = y[max(0, i0):i1]
        if len(seg) > 64:
            lv.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12))
    return {
        "level_pts": float(np.median(lv)) if lv else float("nan"),
        "click_pts": pt,
        "click_deep_p90": float(np.percentile(gd, 90)) if len(gd) else float("nan"),
        "click_deep_n6": int((gd > 6).sum()),
        "click_out_p90": float(np.percentile(go, 90)) if len(go) else float("nan"),
        "sheet_pts": sh,
        "leak_deep_p90": float(np.percentile(lk_d, 90)) if lk_d else float("nan"),
        "leak_out_p90": float(np.percentile(lk_o, 90)) if lk_o else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--model", default="ya", choices=["ya", "ak"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--score", default=SCORE)
    a = ap.parse_args()
    clicks, sheets = TRUTH[a.model]
    wins = windows(a.plan, a.score)

    wavs = []
    for d in a.dirs:
        wavs += sorted(glob.glob(os.path.join(d, "*.wav")))
    if not wavs:
        sys.exit("没有找到 wav")

    print(f"救援窗 {len(wins)} 个(深窗 {sum(1 for _,_,s in wins if abs(s)>=12)} 个)· 模型 {a.model}")
    print(f"\n{'臂':<34}{'缺陷处电平':>11}{'点读合计':>10}{'深窗p90':>9}{'深窗>6':>8}{'窗外p90':>9}"
          f"{'泄漏点读':>10}{'泄漏深p90':>11}{'泄漏窗外':>10}")
    print("  " + "-" * 100)
    rows = []
    for w in wavs:
        s = stats(w, wins, clicks, sheets)
        name = os.path.basename(w).replace("mg_deadonly_", "").replace(".wav", "")
        name = name.replace("_yachiyo_runami", "").replace("_akiko_320000", "")
        rows.append((name, s))
        print(
            f"  {name:<32}{s['level_pts']:>11.2f}{np.sum(s['click_pts']):>10.1f}{s['click_deep_p90']:>9.2f}"
            f"{s['click_deep_n6']:>8d}{s['click_out_p90']:>9.2f}"
            f"{np.median(s['sheet_pts']):>10.2f}{s['leak_deep_p90']:>11.2f}{s['leak_out_p90']:>10.2f}"
        )
    print("")
    print("  ⛔⛔ **先看『缺陷处电平』**：相对出厂动了 >3 dB 的臂，它的『改善』很可能只是**变安静了**")
    print("     (实测：ENVFIX=0 让两把尺子全线变好，而真相是那几处掉了 16-22 dB 几乎消失)。")
    print("\n  ⛔ 判读:『点读合计』『深窗p90』『深窗>6』越小越好;『泄漏』越负越好。")
    print("     ⛔⛔ 但凡『窗外p90』或『泄漏窗外p90』也跟着动了 ⇒ 那条臂动的不是这个缺陷,**不算数**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
