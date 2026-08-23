# -*- coding: utf-8 -*-
"""⑥ `rescue_bench` —— 音域扩展这条线的**验收台**(S159zzl 立)。

## ⛔ 它为什么存在

S159 这一场做了三把刀,**两把撤回**,原因都一样:**先做刀、后找靶子**。
⇒ 这个台子把靶子、对照与护栏**在动刀之前**固定下来,任何候选都必须整台跑过。

## 台面(每一项都有出处,别现场发明)

| | 量 | 判据 | 出处 |
|---|---|---|---|
| ⛔ **门** | **音高闸** | `inverse_probe` 报「目标周期赢」≥ 95 % | S159zze:`WSOLA=0.15` 把移调整个抵消而两把老尺子都没看见 |
| ⭐ **靶** | **H1−H3 / H1−H4 相对原生的偏离** | 越接近 0 越好 | S159zzh→zzk:两模型 × 两谱面 × 三母音,单调剂量曲线 |
| ⭐ 副靶 | `envmod` 2-4 kHz | 用户耳判背书的那把(S159zy/zzd) | ⚠ 但它在配对数据集上只有 4-6/12 分开 ⇒ **只当副的** |
| ⛔ 护栏 | **次基频(0.25-0.75 f0)能量** | 相对原生不许涨 | S159za→zc 的「面状伪影」:`ENV_RESTORE` 那次就是这么回来的,用户原话「动不动就会回来」 |
| ⛔ 护栏 | **谐波梳深度** | 不许比原生浅太多 | S159zzc:摊开那把刀就是在这里炸的(−12.7 dB) |
| ⛔ 护栏 | **电平 / 8-12 kHz 倾斜** | 相对原生 ±1 dB | 区分「修好了」与「只是变安静 / 变亮」 |
| ⛔ 对照 | **原生唱得动的音,不救那一档** | 必须读 ~0 | 没有它,任何非零读数都不算数 |

## ⛔ 硬规矩

* **靶子只用同一个【输出音高】的比较**(`base` vs `donor_post`)——
  这一场有六把尺子死在 f0 / 元音混杂上。
* **长尾分布上同时报 均值 + 超阈占比 + p90**,别只报中位(S159zzk:
  中位数在 −12 上造出过一个假的非单调)。
* **两个谱面都要跑**(鹅妈妈 + 炉心融解),两个模型至少一个交叉点。
* ⛔ **不许在这里复刻生产逻辑**:计划、窗、donor 全部读生产转储。
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

SR, HOP, N = 44100, 882, 8192
W = np.hanning(N)
FR = SR / N

SCORES = {
    "goose": (r"D:\MyDev\TESTING\不为人所知的鹅妈妈童谣\probe\mg_score.json", 7),
    "lch": (r"D:\MyDev\TESTING\s145_range_color\lch\lch_score.json", 7),
}


def note_spans(score_path, transpose, usable=(36, 79), min_samples=N):
    """音表:从**谱面自己的** `frames` 累加。⛔ 别拿别的谱面那份(S159zi 那族的错法)。"""
    tri = json.load(open(score_path, encoding="utf-8"))["triples"]
    out, f = [], 0
    for k, t in enumerate(tri):
        if t["note_num"] > 0:
            midi = t["note_num"] + transpose
            a, b = f * HOP, (f + t["frames"]) * HOP
            if usable[0] <= midi <= usable[1] - 1 and b - a >= min_samples:
                out.append((k, midi, a, b, t["lyric"]))
        f += t["frames"]
    return out


def harmonics(x, a, b, f0, K=6):
    """前 K 次谐波的幅度(dB)。⛔ 门必须是 `>= N`:`np.mean` 在空列表上返回标量 nan。"""
    seg = x[a:b]
    if len(seg) < N:
        return None
    P = 10 * np.log10(np.mean([np.abs(np.fft.rfft(seg[o:o + N] * W)) ** 2
                               for o in range(0, len(seg) - N + 1, N // 4)], axis=0) + 1e-30)
    return [P[max(0, int(round(j * f0 / FR)) - 2):int(round(j * f0 / FR)) + 3].max()
            for j in range(1, K + 1)]


def _mod(x, lo, hi):
    if len(x) < 2048:
        return None
    sos = butter(4, [lo / (SR / 2), min(hi, 0.98 * SR / 2) / (SR / 2)], btype="band", output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, x)))
    if e.mean() <= 1e-9:
        return None
    e = e - e.mean()
    f, p = welch(e, fs=SR, nperseg=min(4096, len(e)))
    num, den = p[(f >= 20) & (f <= 200)].sum(), p[(f >= 1) & (f <= 1000)].sum()
    return 10 * np.log10(num / den) if num > 0 and den > 0 else None


def _band_db(x, lo, hi):
    sos = butter(4, [lo / (SR / 2), min(hi, 0.98 * SR / 2) / (SR / 2)], btype="band", output="sos")
    e = (sosfiltfilt(sos, x) ** 2).mean()
    return 10 * np.log10(e) if e > 1e-30 else np.nan


def comb_depth(x, a, b, f0, lo=1500, hi=6000):
    seg = x[a:b]
    if len(seg) < N:
        return None
    P = np.mean([np.abs(np.fft.rfft(seg[o:o + N] * W)) ** 2
                 for o in range(0, len(seg) - N + 1, N // 4)], axis=0)
    pk, vl, k = [], [], int(np.ceil(lo / f0))
    while k * f0 < hi:
        for t, acc in ((k * f0, pk), ((k + 0.5) * f0, vl)):
            i = int(round(t / FR))
            if 0 < i < len(P) - 1:
                acc.append(P[i - 1:i + 2].max())
        k += 1
    if len(pk) < 4 or len(vl) < 4:
        return None
    a_, b_ = np.median(pk), np.median(vl)
    return 10 * np.log10(a_ / b_) if a_ > 0 and b_ > 0 else None


def score_arm(base, cand, spans):
    """`base` = 原生@目标音高;`cand` = 被救回目标音高的那一版。⛔ 两者必须同音高。"""
    rows = []
    n = min(len(base), len(cand))
    for _k, midi, a, b, _ly in spans:
        if b > n:
            continue
        f0 = 440 * 2 ** ((midi - 69) / 12.0)
        ha, hb = harmonics(base, a, b, f0), harmonics(cand, a, b, f0)
        if not (ha and hb):
            continue
        r = {
            "H1-H3": (hb[0] - hb[2]) - (ha[0] - ha[2]),
            "H1-H4": (hb[0] - hb[3]) - (ha[0] - ha[3]),
        }
        for nm, lo, hi in (("envmod", 2000, 4000),):
            u, v = _mod(base[a:b], lo, hi), _mod(cand[a:b], lo, hi)
            r[nm] = (v - u) if (u is not None and v is not None) else np.nan
        # ⛔ 面状护栏:次基频那一层不许涨
        r["subf0"] = _band_db(cand[a:b], 0.25 * f0, 0.75 * f0) - _band_db(base[a:b], 0.25 * f0, 0.75 * f0)
        cu, cv = comb_depth(base, a, b, f0), comb_depth(cand, a, b, f0)
        r["comb"] = (cv - cu) if (cu is not None and cv is not None) else np.nan
        r["level"] = 10 * np.log10(((cand[a:b] ** 2).mean() + 1e-30) / ((base[a:b] ** 2).mean() + 1e-30))
        r["tilt"] = (_band_db(cand[a:b], 8000, 12000) - _band_db(cand[a:b], 300, 1000)) - \
                    (_band_db(base[a:b], 8000, 12000) - _band_db(base[a:b], 300, 1000))
        rows.append(r)
    return rows


def summarise(rows, keys=("H1-H3", "H1-H4", "envmod", "subf0", "comb", "level", "tilt")):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) == r.get(k)])
        if len(v):
            out[k] = (float(v.mean()), float(np.median(v)), float(np.percentile(v, 90)),
                      float((np.abs(v) > 8).mean() * 100), len(v))
    return out


def report(title, rows):
    s = summarise(rows)
    print(f"  {title}")
    print(f"    {'量':<8}{'均值':>9}{'中位':>9}{'p90':>9}{'|Δ|>8 的%':>11}{'n':>6}")
    for k, (m, md, p9, big, n) in s.items():
        print(f"    {k:<8}{m:>+9.2f}{md:>+9.2f}{p9:>+9.2f}{big:>10.0f}%{n:>6}")


if __name__ == "__main__":
    print(__doc__.split("\n")[1])
    print("⇒ 这是库,不是脚本;由候选刀的 sweep 脚本 import 它。")
