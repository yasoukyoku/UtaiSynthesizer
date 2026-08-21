# -*- coding: utf-8 -*-
"""S159za —— **周期倍化尺**:逐周期峰值的「强弱强弱」交替(= 用户看到的「面状伪影」)。

## ⛔ 它量的是一个【已经在波形上看见】的东西,不是又一个假说

用户 2026-08-22 点名的面状伪影处,逐周期数出来是这样的
(yachiyo 音[783]ん,268.930 s,位移 −17;donor 转储 pre/post 逐样本):

```
PRE  周期 ms  2.75 2.75 2.73 2.75 2.73 2.73 2.73 2.71     ← 完美规则
POST 周期 ms  0.60 0.42 0.60 0.42 0.60 0.42 0.62 0.40     ← 严格交替
POST 峰值     0.318 0.177 0.315 0.171 0.305 0.186 0.316   ← 强弱强弱
```

⇒ 输出每两个周期重复一次 = **周期倍化** ⇒ 谱上在 f0/2 及其奇数倍处长出一整套边带,
   看起来就是「基频以下/谐波之间的一片」。

机理在 psola.rs 的设计注里早有登记:`k = round(u)` 让相邻若干颗输出颗粒**读同一个源标记**
却放在不同相位上,而被复制的不是一个脉冲、是「脉冲 ⊛ 声道冲激响应」的一整条长尾
⇒ 尾巴之间非相干叠加。`xgrain` 是为这条造的,但在 ratio > 2 的深窗上显然没压住。

## 量法
把逐周期峰值序列取出来,去趋势,算**序列自身的奈奎斯特分量**(= 一上一下的强度):
`alt = |Σ (−1)^k · a_k| / Σ a_k`。纯交替 = 1.0,完全无交替 ≈ 0。
⚠ 对整体渐强渐弱免疫(去趋势);对 f0 变化免疫(按当地周期切段)。

## ⛔ 使用纪律
必须先在 `--truth` 上自检,排不进前列就作废。这条线上已经废掉 11 把没这么做的尺子。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grain_glitch import f0_track, lead_silence, load, notes_from_score, period_peaks  # noqa: E402


def alternation(amp: np.ndarray) -> float:
    """峰值序列的「一上一下」强度,0..1。"""
    if len(amp) < 6:
        return 0.0
    a = amp.astype(float)
    # 去趋势:减掉 5 点滑动中位
    k = 5
    pad = np.pad(a, (k // 2, k // 2), mode="edge")
    med = np.array([np.median(pad[i : i + k]) for i in range(len(a))])
    d = a - med
    sign = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(a))])
    num = abs(float(np.sum(sign * d)))
    den = float(np.sum(np.abs(d))) + 1e-12
    return num / den


def scan(y, sr, win_periods=24, hop_periods=8):
    f0, hop = f0_track(y, sr)
    pk = period_peaks(y, sr, f0, hop)
    if len(pk) < win_periods + 2:
        return []
    pos = np.array([p[0] for p in pk])
    amp = np.array([p[1] for p in pk])
    out = []
    for i in range(0, len(amp) - win_periods, hop_periods):
        seg = amp[i : i + win_periods]
        if np.median(seg) < 1e-3:            # 电平地板,见 grain_glitch 的同名注释
            continue
        out.append(
            {
                "t": float(pos[i + win_periods // 2]) / sr,
                "alt": alternation(seg),
                "level": float(20 * np.log10(np.median(seg) + 1e-12)),
            }
        )
    out.sort(key=lambda q: -q["alt"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render")
    ap.add_argument("--score", default=None)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--at", default=None)
    a = ap.parse_args()

    y, sr = load(a.render)
    off = lead_silence(y, sr)
    hits = scan(y, sr)
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
            near = [h for h in hits if abs(h["t"] - off - s) <= 0.120]
            v = max((h["alt"] for h in near), default=0.0)
            vals.append(v)
            print(f"{v:>8.3f}", end="")
        print(f"   | 中位 {np.median(vals):6.3f}  合计 {np.sum(vals):7.3f}")
        return 0

    print(f"素材 {os.path.basename(a.render)} · {len(y)/sr:.2f}s @{sr}")
    print(f"窗数 {len(hits)}(24 周期一窗,步进 8)")
    print(f"\n{'名次':>4}{'时刻':>10}{'交替度':>9}{'电平dB':>9}  音")
    for i, h in enumerate(hits[: a.top], 1):
        print(f"{i:>4}{h['t']-off:>10.3f}{h['alt']:>9.3f}{h['level']:>9.1f}  {note_of(h['t'])}")

    if a.truth:
        tv = [float(x) for x in a.truth.split(",") if x.strip()]
        print(f"\n⛔ **自检:用户确认过的 {len(tv)} 个面状坐标排第几**")
        ranks = []
        for t in tv:
            cand = [(i, h) for i, h in enumerate(hits, 1) if abs(h["t"] - off - t) <= 0.120]
            if cand:
                i, h = min(cand, key=lambda q: q[0])
                ranks.append(i)
                print(f"   {t:9.3f}s ⇒ 第 **{i}** 名 / {len(hits)}(交替度 {h['alt']:.3f})")
            else:
                ranks.append(None)
                print(f"   {t:9.3f}s ⇒ ⛔ **±120 ms 内没有窗**")
        got = [r for r in ranks if r]
        if got:
            print(f"   ⇒ 命中 **{len(got)}/{len(tv)}**;名次中位 **{int(np.median(got))}**,最差 **{max(got)}**")
            print(f"   ⇒ 前 1% 是第 {max(1,len(hits)//100)} 名;前 5% 是第 {max(1,len(hits)//20)} 名")
    return 0


if __name__ == "__main__":
    sys.exit(main())
