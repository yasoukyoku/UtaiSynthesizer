# -*- coding: utf-8 -*-
"""S159za —— **donor 基频泄漏尺**:救援段里,输出基频【以下】那条本不该存在的能量带。

## ⛔ 它量的是一个在频谱图上【看见】的东西

用户 2026-08-22 点名的 5 处「面状伪影」,并排放到同一色标的频谱图上
(TESTING\\s159za_sheet.png),四处形状完全一样:**青线之间、在基频以下 300-900 Hz
有一条明亮的能量带,而同一文件里不在救援窗的对照在同一位置是暗的。**

对上数(akiko,位移 −14):输出 f0 ≈ 980 Hz ⇒ donor 唱的是 980 / 2^(14/12) = **437 Hz**
—— 正是那条亮带。note[502] 输出 897 ⇒ donor 400;note[500] 输出 1004 ⇒ donor 447。全部对上。

⇒ **面状伪影 = donor 基频没被搬干净的残留。**
⇒ 「为什么只在深窗」也跟着解释了:位移浅时 donor 基频紧挨输出基频、被掩蔽;
   深到 −12 以下,它掉进输出基频**以下**那片空区,直接暴露。

三方对照(S159z)给过同一个结论的数:150-300 Hz 上 **PSOLA 压掉了 42.48 dB,
但残留仍比原生高 20.1 dB**。

## 量法
`leak = 10·log10( E(0.25·f0 … 0.80·f0) / E(0.85·f0 … 8000) )`,dB。
* 下界 0.25·f0 —— 再往下是次声/直流,那是另一把刀(`Infrasonic`)的地盘;
* 上界 0.80·f0 —— 留出基频主瓣的裙边;
* 分母用主体,所以它对整体电平免疫。
⚠ f0 用自相关 + 抛物线细化,只在够周期的段上给值;取不到 f0 的段跳过(不猜)。

## ⛔ 使用纪律
必须先在 `--truth` 上自检。这条线上已经废掉 11 把没这么做的尺子。
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


def f0_of(x, sr):
    d = x - x.mean()
    if np.sqrt(np.mean(d**2)) < 2e-3:
        return 0.0
    ac = np.correlate(d, d, "full")[len(d) - 1 :]
    ac = ac / (ac[0] + 1e-30)
    lo, hi = int(sr / 1400), int(sr / 70)
    seg = ac[lo : min(hi, len(ac))]
    if len(seg) < 3:
        return 0.0
    k = int(np.argmax(seg)) + lo
    if ac[k] < 0.40:
        return 0.0
    if 0 < k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
    return sr / k


def leak_at(y, sr, a, b):
    x = y[int(a * sr) : int(b * sr)]
    if len(x) < 512:
        return float("nan"), 0.0
    f0 = f0_of(x, sr)
    if f0 <= 0:
        return float("nan"), 0.0
    n = len(x) // 2 * 2
    P = np.abs(np.fft.rfft(x[:n] * np.hanning(n), 1 << 16)) ** 2
    f = np.fft.rfftfreq(1 << 16, 1 / sr)
    lo = (f >= 0.25 * f0) & (f < 0.80 * f0)
    hi = (f >= 0.85 * f0) & (f < 8000)
    if not lo.any() or not hi.any():
        return float("nan"), f0
    return 10 * np.log10((P[lo].sum() + 1e-30) / (P[hi].sum() + 1e-30)), f0


def scan(y, sr, win=0.150, hop=0.050):
    out = []
    t = 0.0
    while t + win < len(y) / sr:
        v, f0 = leak_at(y, sr, t, t + win)
        if not np.isnan(v):
            out.append({"t": t + win / 2, "leak": v, "f0": f0})
        t += hop
    out.sort(key=lambda q: -q["leak"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render")
    ap.add_argument("--score", default=None)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--at", default=None, help="逗号分隔 起-止 秒段,只报这些段(跨臂比较用)")
    a = ap.parse_args()

    y, sr = load(a.render)
    off = lead_silence(y, sr)

    if a.at:
        print(f"{os.path.basename(a.render):<44}", end="")
        vals = []
        for part in a.at.split(","):
            s, e = part.split("-")
            v, f0 = leak_at(y, sr, float(s) + off, float(e) + off)
            vals.append(v)
            print(f"{v:>8.2f}", end="")
        v = [x for x in vals if not np.isnan(x)]
        print(f"   | 中位 {np.median(v):7.2f}" if v else "   | (无)")
        return 0

    hits = scan(y, sr)
    notes = notes_from_score(a.score) if a.score else None

    def note_of(t):
        if not notes:
            return ""
        for k, na, nb, ly, nn in notes:
            if na <= t - off < nb and nn > 0:
                return f"音[{k}]{ly}"
        return "(休止)"

    print(f"素材 {os.path.basename(a.render)} · {len(y)/sr:.2f}s @{sr}")
    print(f"有基频的窗 {len(hits)} 个(150 ms 窗,步进 50 ms)")
    v = np.array([h["leak"] for h in hits])
    print(f"泄漏 p50 {np.median(v):.2f} · p90 {np.percentile(v,90):.2f} · p99 {np.percentile(v,99):.2f} dB")
    print(f"\n{'名次':>4}{'时刻':>10}{'泄漏dB':>9}{'f0':>8}  音")
    for i, h in enumerate(hits[: a.top], 1):
        print(f"{i:>4}{h['t']-off:>10.3f}{h['leak']:>9.2f}{h['f0']:>8.0f}  {note_of(h['t'])}")

    if a.truth:
        tv = [float(x) for x in a.truth.split(",") if x.strip()]
        print(f"\n⛔ **自检:用户确认过的 {len(tv)} 个坐标排第几**")
        ranks = []
        for t in tv:
            cand = [(i, h) for i, h in enumerate(hits, 1) if abs(h["t"] - off - t) <= 0.100]
            if cand:
                i, h = min(cand, key=lambda q: q[0])
                ranks.append(i)
                print(f"   {t:9.3f}s ⇒ 第 **{i}** 名 / {len(hits)}(泄漏 {h['leak']:+.2f} dB,f0 {h['f0']:.0f})")
            else:
                ranks.append(None)
                print(f"   {t:9.3f}s ⇒ ⛔ **±100 ms 内没有窗**")
        got = [r for r in ranks if r]
        if got:
            print(f"   ⇒ 命中 **{len(got)}/{len(tv)}**;名次中位 **{int(np.median(got))}**,最差 **{max(got)}**")
            print(f"   ⇒ 前 1% 是第 {max(1,len(hits)//100)} 名;前 5% 是第 {max(1,len(hits)//20)} 名")
    return 0


if __name__ == "__main__":
    sys.exit(main())
