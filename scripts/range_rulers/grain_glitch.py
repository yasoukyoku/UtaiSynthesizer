# -*- coding: utf-8 -*-
"""S159za —— **颗粒错位尺**:单个基音周期的幅度相对邻居跳变。

## ⛔ 为什么是这个量(不是又一把随手造的尺子)

用户 2026-08-22 点名的 12 处,放到样本级看,形状是一致的:
**一个孤立的周期冲到周围的 2-3 倍,紧接着塌陷**(见 TESTING\\s159za_zoom.png)。
而仓里早有一条登记结论说明它是什么(vocal_range.rs 的 `wsola_frac` doc):
「电平丢在**颗粒与颗粒之间的相位抵消**上 …… 合成那一遍是**盲加**每一颗颗粒的」。
⇒ 相消是一半,**相长**是另一半,而相长表现为一个 2-3 倍的孤立周期 = 咔哒。

⇒ 这把尺子直接量那个:**逐周期峰值 / 邻居中位**。
* 它对人声本身的动态免疫(渐强渐弱是平滑的,邻居中位跟着走);
* 它对整体电平免疫(是比值);
* 它对「深坑」与「巨脉冲」**同时**有反应(比值 >1 与 <1 两侧)。

## ⛔ 使用纪律
* 这把尺子**必须**先在 `--truth` 上自检:用户确认过的坐标要能被它排进前列,
  否则当场作废(这条线上已经废掉 11 把没这么做的尺子)。
* 它**不判好坏**,只给一个可比的数;跨臂比较时两条臂必须只差一个自由度。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sys.exit("需要 soundfile:pip install soundfile")


def load(path):
    y, sr = sf.read(path, dtype="float64", always_2d=True)
    return y.mean(1), sr


def lead_silence(y, sr):
    nz = np.nonzero(np.abs(y) > 1e-4)[0]
    return float(nz[0]) / sr if len(nz) else 0.0


def notes_from_score(path):
    tri = json.load(open(path, encoding="utf-8"))["triples"]
    t, out = 0.0, []
    for k, n in enumerate(tri):
        d = max(0, n["frames"]) / 50.0
        out.append((k, t, t + d, n.get("lyric", ""), n.get("note_num", 0)))
        t += d
    return out


def f0_track(y, sr, hop_ms=10.0, win_ms=40.0):
    """自相关 f0,只在够响且够周期的帧上给值。"""
    h = int(sr * hop_ms / 1000.0)
    w = int(sr * win_ms / 1000.0)
    out = np.zeros(len(y) // h + 1)
    lo, hi = int(sr / 1400), int(sr / 70)
    for j in range(len(out)):
        c = j * h
        a, b = max(0, c - w // 2), min(len(y), c + w // 2)
        x = y[a:b]
        if len(x) < lo * 2 or np.sqrt(np.mean(x**2)) < 1e-3:
            continue
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[len(x) - 1 :]
        ac = ac / (ac[0] + 1e-30)
        seg = ac[lo : min(hi, len(ac))]
        if len(seg) < 3:
            continue
        k = int(np.argmax(seg)) + lo
        if ac[k] > 0.45:
            out[j] = sr / k
    return out, h


def period_peaks(y, sr, f0, hop):
    """按当地周期切段,取每段的峰值绝对值。返回 (中心样本, 峰值)。"""
    peaks = []
    i = 0
    n = len(y)
    while i < n:
        j = min(int(i / hop), len(f0) - 1)
        p = f0[j]
        if p <= 0:
            i += hop
            continue
        T = int(sr / p)
        if T < 4:
            i += hop
            continue
        b = min(n, i + T)
        seg = y[i:b]
        if len(seg) >= 4:
            peaks.append((i + int(np.argmax(np.abs(seg))), float(np.abs(seg).max())))
        i = b
    return peaks


def glitch_scan(y, sr, neigh=3):
    f0, hop = f0_track(y, sr)
    pk = period_peaks(y, sr, f0, hop)
    if len(pk) < 2 * neigh + 3:
        return []
    pos = np.array([p[0] for p in pk])
    amp = np.array([p[1] for p in pk])
    out = []
    for i in range(neigh, len(amp) - neigh):
        nb = np.concatenate([amp[i - neigh : i], amp[i + 1 : i + 1 + neigh]])
        m = np.median(nb)
        # ⛔ 电平地板:休止里「零除以零」会读出 120 dB 并霸占榜首(第一版就是这样,
        #    前 15 名全是休止)。地板取 −60 dBFS —— 比它安静的地方本来就听不见跳变。
        if m <= 1e-3:
            continue
        r = amp[i] / m
        # 「跳变」= 相长(r 大)或相消(r 小)里更极端的那一侧,取对数对称
        score = abs(20.0 * np.log10(max(r, 1e-6)))
        out.append({"t": pos[i] / sr, "ratio": float(r), "score": float(score)})
    out.sort(key=lambda q: -q["score"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render")
    ap.add_argument("--score", default=None)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--at", default=None, help="只在这些秒数附近 ±80 ms 报最大跳变(跨臂比较用)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    y, sr = load(a.render)
    off = lead_silence(y, sr)
    hits = glitch_scan(y, sr)
    notes = notes_from_score(a.score) if a.score else None

    def note_of(t):
        if not notes:
            return ""
        for k, na, nb, ly, nn in notes:
            if na <= t - off < nb and nn > 0:
                return f"音[{k}]{ly}"
        return "(休止)"

    if a.at:
        print(f"{os.path.basename(a.render):<46}", end="")
        vals = []
        for s in [float(x) for x in a.at.split(",") if x.strip()]:
            near = [h for h in hits if abs(h["t"] - off - s) <= 0.080]
            v = max((h["score"] for h in near), default=0.0)
            vals.append(v)
            print(f"{v:>8.2f}", end="")
        print(f"   | 中位 {np.median(vals):6.2f}  合计 {np.sum(vals):8.2f}")
        return 0

    print(f"素材 {os.path.basename(a.render)} · {len(y)/sr:.2f}s @{sr} · 前导静音 {off:.3f}s")
    print(f"逐周期跳变候选 {len(hits)} 个(全曲)")
    if not a.quiet:
        print(f"\n{'名次':>4}{'时刻':>10}{'比值':>8}{'dB':>8}  音")
        for i, h in enumerate(hits[: a.top], 1):
            print(f"{i:>4}{h['t']-off:>10.3f}{h['ratio']:>8.2f}{h['score']:>8.1f}  {note_of(h['t'])}")

    if a.truth:
        tv = [float(x) for x in a.truth.split(",") if x.strip()]
        print(f"\n⛔ **自检:用户确认过的 {len(tv)} 个坐标在这把尺子的排序里排第几**")
        ranks = []
        for t in tv:
            cand = [(i, h) for i, h in enumerate(hits, 1) if abs(h["t"] - off - t) <= 0.080]
            if cand:
                i, h = min(cand, key=lambda q: q[0])
                ranks.append(i)
                print(f"   {t:9.3f}s ⇒ 第 **{i}** 名 / {len(hits)}(比值 {h['ratio']:.2f} = {h['score']:.1f} dB)")
            else:
                ranks.append(None)
                print(f"   {t:9.3f}s ⇒ ⛔ **±80 ms 内没有候选**")
        got = [r for r in ranks if r]
        if got:
            print(f"   ⇒ 命中 **{len(got)}/{len(tv)}**;名次中位 **{int(np.median(got))}**,最差 **{max(got)}**")
            print(f"   ⇒ 前 1% 分位是第 {max(1,len(hits)//100)} 名;前 5% 是第 {max(1,len(hits)//20)} 名")
    return 0


if __name__ == "__main__":
    sys.exit(main())
