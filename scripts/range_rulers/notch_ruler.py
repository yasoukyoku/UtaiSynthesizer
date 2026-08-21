# -*- coding: utf-8 -*-
"""S159za —— **凹陷尺**:唱音内部又深又快的包络塌陷(= 用户听到的「咔哒」)。

## ⛔ 它凭什么可信:它是在【用户亲耳确认过的 12 个坐标】上验过阳性才写出来的

这条线上我造过 **11 把标量尺子,11 把都与用户耳判相反**。根因每次都一样:
**尺子是照着我当时的假说造的,所以只会确认我已经相信的东西。**
⇒ 这一把反过来做:**先有 ground truth(用户 2026-08-22 给的 12 个坐标),再调形状,
最后必须报出「这 12 个落在全曲排序的第几名」**。排不进去的尺子当场作废,不许用。

## 形状(为什么是这个形状 —— 每一条都对着实测)

* **2 ms 包络**上的塌陷。用户报的宽度是 2-40 ms,5 ms 窗会把最窄的那些抹掉。
* 深度按**局部中位**算(±60 ms),不按全曲。全曲电平在这首歌里跨 40 dB,
  用全局阈值会把安静段整段报出来(S159t 那把尺子就是这么废的)。
* **必须回来**:塌下去 ≤80 ms 之内要恢复到局部中位 −6 dB 以内。
  不加这一条,音尾的自然收束会全部被报成凹陷。
* **必须在唱音里**(给 `--score` 时):休止里的静音是设计如此。
* ⛔ **不做「在不在救援窗里」的过滤** —— 那是**结论**不是判据。
  尺子只报「哪里有凹陷」,归属由调用方去对。

## 用法
```
python scripts/range_rulers/notch_ruler.py <render.wav> [--score 谱.json]
       [--plan 计划.json] [--truth 坐标,坐标,...] [--top 30]
```
`--truth` 给一串秒数 ⇒ 报出每个 truth 在排序里的**名次**(这是这把尺子的自检)。
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

ENV_MS = 2.0
LOCAL_MS = 60.0
RECOVER_MS = 80.0
RECOVER_TOL_DB = 6.0


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


def env_2ms(y, sr):
    h = max(1, int(sr * ENV_MS / 1000.0))
    k = len(y) // h
    e = np.sqrt(np.mean(y[: k * h].reshape(k, h) ** 2, 1)) + 1e-12
    return 20 * np.log10(e), h


def rolling_median(v, w):
    """±w 帧的滑动中位(用排序分块近似,O(n log w))。"""
    n = len(v)
    out = np.empty(n)
    from collections import deque
    import bisect

    win = []
    dq = deque()
    for i in range(n):
        bisect.insort(win, v[i])
        dq.append(v[i])
        if len(dq) > 2 * w + 1:
            old = dq.popleft()
            win.pop(bisect.bisect_left(win, old))
        out[max(0, i - w)] = win[len(win) // 2]
    for i in range(max(0, n - w), n):
        out[i] = out[max(0, n - w - 1)]
    return out


def find_notches(y, sr, notes=None, off=0.0):
    e, h = env_2ms(y, sr)
    w = int(LOCAL_MS / ENV_MS)
    med = rolling_median(e, w)
    depth = med - e                     # 正 = 比局部中位低多少 dB
    rec = int(RECOVER_MS / ENV_MS)
    hits = []
    i = 1
    n = len(e)
    while i < n - 1:
        if depth[i] >= 6.0 and depth[i] >= depth[i - 1] and depth[i] > depth[i + 1]:
            # 局部极小 ⇒ 往两边找回到 -RECOVER_TOL_DB 以内的点
            a = i
            while a > 0 and depth[a] > RECOVER_TOL_DB and i - a < rec:
                a -= 1
            b = i
            while b < n - 1 and depth[b] > RECOVER_TOL_DB and b - i < rec:
                b += 1
            recovered = depth[a] <= RECOVER_TOL_DB and depth[b] <= RECOVER_TOL_DB
            t = i * h / sr
            ok = True
            tag = ""
            if notes is not None:
                s = None
                for k, na, nb, ly, nn in notes:
                    if na <= t - off < nb and nn > 0:
                        s = (k, ly, nn)
                        break
                ok = s is not None
                tag = f"音[{s[0]}]{s[1]} midi={s[2]}" if s else ""
            if recovered and ok:
                hits.append(
                    {
                        "t": t,
                        "depth_db": float(depth[i]),
                        "width_ms": float((b - a) * ENV_MS),
                        "floor_db": float(e[i]),
                        "note": tag,
                    }
                )
            i = b + 1
        else:
            i += 1
    hits.sort(key=lambda q: -q["depth_db"])
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render")
    ap.add_argument("--score", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--truth", default=None, help="逗号分隔的秒数(用户确认过的坐标)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    y, sr = load(a.render)
    off = lead_silence(y, sr)
    notes = notes_from_score(a.score) if a.score else None
    hits = find_notches(y, sr, notes, off)

    groups = []
    if a.plan:
        d = json.load(open(a.plan, encoding="utf-8"))
        g = d.get("groups")
        if g is None:
            g = next(x for x in d["arms"] if "出厂" in x["label"])["groups"]
        tri = json.load(open(a.score, encoding="utf-8"))["triples"] if a.score else []
        if tri:
            T = np.concatenate([[0.0], np.cumsum([max(0, x["frames"]) / 50.0 for x in tri])])
            groups = [(T[s], T[min(en + 1, len(T) - 1)], sh) for s, en, sh in g]

    def win_of(t):
        for s, e, sh in groups:
            if s <= t - off < e:
                return sh
        return None

    print(f"素材 {os.path.basename(a.render)} · {len(y)/sr:.2f}s @{sr} · 前导静音 {off:.3f}s")
    print(f"**唱音内的凹陷(深度 ≥6 dB 且 ≤{RECOVER_MS:.0f} ms 内恢复):{len(hits)} 处**")
    if groups:
        inw = [h for h in hits if win_of(h["t"]) is not None]
        deep = [h for h in inw if abs(win_of(h["t"])) >= 12]
        print(f"  其中在救援窗内 {len(inw)} · 在深窗(|位移| ≥12)内 **{len(deep)}**")
    if not a.quiet:
        print(f"\n{'名次':>4}{'时刻':>10}{'深度':>8}{'宽 ms':>8}{'谷底 dBFS':>11}{'窗':>7}  音")
        for i, h in enumerate(hits[: a.top], 1):
            sh = win_of(h["t"])
            print(f"{i:>4}{h['t']-off:>10.3f}{h['depth_db']:>8.1f}{h['width_ms']:>8.0f}"
                  f"{h['floor_db']:>11.1f}{(f'{sh:+d}' if sh is not None else '—'):>7}  {h['note']}")

    if a.truth:
        tv = [float(x) for x in a.truth.split(",") if x.strip()]
        print(f"\n⛔ **自检:用户确认过的 {len(tv)} 个坐标在这把尺子的排序里排第几**")
        ranks = []
        for t in tv:
            best, bi = None, None
            for i, h in enumerate(hits, 1):
                d = abs((h["t"] - off) - t)
                if best is None or d < best:
                    best, bi = d, i
            if best is not None and best <= 0.060:
                ranks.append(bi)
                print(f"   {t:9.3f}s ⇒ 第 **{bi}** 名 / {len(hits)}(深度 {hits[bi-1]['depth_db']:.1f} dB,差 {1000*best:+.0f} ms)")
            else:
                ranks.append(None)
                print(f"   {t:9.3f}s ⇒ ⛔ **没找到**(最近的候选差 {1000*best:.0f} ms)")
        got = [r for r in ranks if r]
        if got:
            print(f"   ⇒ 命中 **{len(got)}/{len(tv)}**;名次中位 **{int(np.median(got))}**,最差 **{max(got)}**")
        if len(got) < len(tv):
            print("   ⛔⛔ 有 truth 没被命中 ⇒ **这把尺子还不能用**,先改形状再说。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
