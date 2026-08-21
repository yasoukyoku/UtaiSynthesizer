# -*- coding: utf-8 -*-
"""S159zb —— **咬字轴的验收**:辅音谷在成品上【实际】刻了多深。

## 为什么需要它

S159zb 把辅音谷从「donor 渲染的 chunk 循环里」挪到「逆变换之后」。
⛔ 那一刀只有在**咬字收益不丢**的前提下才成立 —— 而「收益」就是这个谷的深度
(S84:按音素类实测的 mix−render 差,鼻/边 11.4 dB、塞/浊塞 11.7、擦 5.1、通音 1.4)。

## 预期(可证伪)
* **挪之后**:救援段里的谷 ≈ **设计值**(11.4 / 11.7 dB),因为 PSOLA 再也看不到它。
* **挪之前(今天)**:同一处的谷**更深** —— PSOLA 把它放大了 4-20 dB(S159zb 的 M2)。
⇒ 所以「挪之后谷变浅」**不是退化**,是**回到设计值**;真正要盯的是它有没有**浅于设计值**。
⛔ 若挪之后读到 < 8 dB,那才是咬字丢了,这一刀就不成立。

## 量法
对每个「链内辅音」窗(与 `boundary_valley_depths` 同一批位置,这里按音素类近似):
`谷深 = 该音前 40% 的最低 5 ms 包络 − 该音后 60% 的中位 5 ms 包络`,dB。
⚠ 它比引擎内部的定义粗,但**跨臂比较**只需要它是同一把尺子。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

NASAL = set("なにぬねのまみむめもん")
STOP = set("たちつてとかきくけこぱぴぷぺぽばびぶべぼだぢづでどがぎぐげご")
FRIC = set("さしすせそはひふへほざじずぜぞ")


def cls(ly):
    c = (ly or "?")[0]
    if c in NASAL:
        return "鼻音", 11.4
    if c in STOP:
        return "塞音", 11.7
    if c in FRIC:
        return "擦音", 5.1
    return "其他", 0.0


def env(y, sr, a, b, ms=5.0):
    i0, i1 = int(a * sr), int(b * sr)
    seg = y[max(0, i0):i1]
    h = max(1, int(sr * ms / 1000.0))
    k = len(seg) // h
    if k < 2:
        return None
    return 20 * np.log10(np.sqrt(np.mean(seg[: k * h].reshape(k, h) ** 2, 1)) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--score", required=True)
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()

    tri = json.load(open(a.score, encoding="utf-8"))["triples"]
    T = np.concatenate([[0.0], np.cumsum([max(0, x["frames"]) / 50.0 for x in tri])])
    d = json.load(open(a.plan, encoding="utf-8"))
    g = d.get("groups") or next(x for x in d["arms"] if "出厂" in x["label"])["groups"]
    deep = [(s, e) for s, e, sh in g if abs(sh) >= 12]
    inwin = set()
    for s, e in deep:
        inwin.update(range(s, e + 1))

    wavs = []
    for dd in a.dirs:
        wavs += sorted(glob.glob(os.path.join(dd, "*.wav")))
    print(f"深窗内音符 {len(inwin)} 个 · 臂 {len(wavs)} 条")
    print(f"\n{'臂':<32}{'鼻音谷 p50':>12}{'塞音谷 p50':>12}{'窗外鼻音':>11}{'窗外塞音':>11}")
    print("  " + "-" * 78)
    for w in wavs:
        y, sr = sf.read(w, dtype="float64", always_2d=True)
        y = y.mean(1)
        buckets = {("鼻音", True): [], ("塞音", True): [], ("鼻音", False): [], ("塞音", False): []}
        for k, t in enumerate(tri):
            if t["note_num"] <= 0:
                continue
            c, want = cls(t.get("lyric", ""))
            if want <= 0:
                continue
            t0, t1 = T[k], T[k + 1]
            if t1 - t0 < 0.08:
                continue
            e1 = env(y, sr, t0, t0 + (t1 - t0) * 0.4)
            e2 = env(y, sr, t0 + (t1 - t0) * 0.5, t1)
            if e1 is None or e2 is None:
                continue
            key = (c, k in inwin)
            if key in buckets:
                buckets[key].append(float(np.median(e2) - e1.min()))
        nm = os.path.basename(w).replace("mg_deadonly_", "").replace(".wav", "")
        nm = nm.replace("_yachiyo_runami", "").replace("_akiko_320000", "")
        v = [np.median(buckets[k]) if buckets[k] else float("nan")
             for k in (("鼻音", True), ("塞音", True), ("鼻音", False), ("塞音", False))]
        print(f"  {nm:<30}{v[0]:>12.2f}{v[1]:>12.2f}{v[2]:>11.2f}{v[3]:>11.2f}")
    print("\n  设计值:鼻音 **11.4 dB** · 塞音 **11.7 dB**(S84 的实测 mix−render 差)。")
    print("  ⇒ 深窗里读到**远深于**设计值 = PSOLA 在放大它;读到 **< 8 dB** = 咬字真的丢了。")
    print("  ⛔ 窗外两列是**阴性对照**:它们不该随这一刀变(窗外不过 PSOLA)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
