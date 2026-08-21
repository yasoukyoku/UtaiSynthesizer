# -*- coding: utf-8 -*-
"""S159zb —— **自适应辅音谷的最终验收表**(成对臂,四根轴 + 两条阴性对照)。

## 四根轴
1. **凹陷深度**(用户点名坐标上;⭐ 这是唯一在空臂上验过「噪声地板 = 0.0 dB」的物理量);
2. **同处电平**(⛔ 防「变安静冒充修好」—— ENVFIX=0 就是那样骗过两把尺子的);
3. **咬字轴**:辅音谷在成品上实际刻了多深(设计值 鼻 11.4 / 塞 11.7 dB);
4. **群体**:深窗内 `grain_glitch` 候选数 + `donor_leak` p90。

## 两条阴性对照
* **窗外**的同一批统计(这一刀不该碰窗外);
* **原 key**(用户说它是干净的 ⇒ **不许退化**)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from donor_leak import leak_at  # noqa: E402
from grain_glitch import glitch_scan  # noqa: E402

TRUTH_YA = [42.675, 54.302, 115.398, 119.873, 126.420, 127.039]
TRUTH_AK = [129.376, 130.96, 131.48, 132.77, 268.98]
NASAL = set("なにぬねのまみむめもん")
STOP = set("たちつてとかきくけこぱぴぷぺぽばびぶべぼだぢづでどがぎぐげご")


def load(p):
    y, sr = sf.read(p, dtype="float64", always_2d=True)
    return y.mean(1), sr


def dip(y, sr, c, half=0.10):
    a, b = int((c - half) * sr), int((c + half) * sr)
    seg = y[max(0, a):b]
    h = max(1, int(sr * 0.002))
    k = len(seg) // h
    if k < 4:
        return float("nan")
    e = 20 * np.log10(np.sqrt(np.mean(seg[: k * h].reshape(k, h) ** 2, 1)) + 1e-12)
    return float(np.median(e) - e.min())


def lvl(y, sr, c, half=0.025):
    a, b = int((c - half) * sr), int((c + half) * sr)
    seg = y[max(0, a):b]
    return float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)) if len(seg) > 64 else float("nan")


def windows(plan, score):
    tri = json.load(open(score, encoding="utf-8"))["triples"]
    T = np.concatenate([[0.0], np.cumsum([max(0, x["frames"]) / 50.0 for x in tri])])
    d = json.load(open(plan, encoding="utf-8"))
    g = d.get("groups") or next(x for x in d["arms"] if "出厂" in x["label"])["groups"]
    return tri, T, [(T[s], T[min(e + 1, len(T) - 1)], sh) for s, e, sh in g], g


def report(pairs, score, truth):
    for lab, hp, fp, plan in pairs:
        if not (os.path.exists(hp) and os.path.exists(fp)):
            print(f"\n### {lab} —— ⛔ 产物缺失({'出厂' if not os.path.exists(hp) else '修法'})")
            continue
        H, F = load(hp), load(fp)
        tri, T, W, g = windows(plan, score)
        deep = [(a, b) for a, b, sh in W if abs(sh) >= 12]
        print(f"\n### {lab}   (救援窗 {len(W)},其中深窗 {len(deep)})")
        if truth:
            print(f"  {'坐标':<10}{'出厂谷深':>10}{'修法谷深':>10}{'Δ':>8}{'出厂电平':>10}{'修法电平':>10}{'Δ电平':>8}")
            r = []
            for c in truth:
                d0, d1, l0, l1 = dip(*H, c), dip(*F, c), lvl(*H, c), lvl(*F, c)
                r.append((d0, d1, l0, l1))
                print(f"  {c:<10.3f}{d0:>10.1f}{d1:>10.1f}{d1-d0:>+8.1f}{l0:>10.1f}{l1:>10.1f}{l1-l0:>+8.1f}")
            r = np.array(r)
            print(f"  {'中位':<10}{np.median(r[:,0]):>10.1f}{np.median(r[:,1]):>10.1f}"
                  f"{np.median(r[:,1])-np.median(r[:,0]):>+8.1f}"
                  f"{np.median(r[:,2]):>10.1f}{np.median(r[:,3]):>10.1f}{np.median(r[:,3])-np.median(r[:,2]):>+8.1f}")

        # 群体 + 窗外对照
        def pop(Y):
            y, sr = Y
            gg = glitch_scan(y, sr)
            ind = lambda t: any(a <= t < b for a, b in deep)
            outw = lambda t: not any(a <= t < b for a, b, _ in W)
            gd = sum(1 for h in gg if ind(h["t"]) and h["score"] > 6)
            go = sum(1 for h in gg if outw(h["t"]) and h["score"] > 6)
            lk_d, lk_o = [], []
            t = 0.0
            while t + 0.15 < len(y) / sr:
                v, _ = leak_at(y, sr, t, t + 0.15)
                if not np.isnan(v):
                    (lk_d if ind(t + 0.075) else (lk_o if outw(t + 0.075) else [])).append(v)
                t += 0.20
            return gd, go, (np.percentile(lk_d, 90) if lk_d else float("nan")), \
                   (np.percentile(lk_o, 90) if lk_o else float("nan"))
        a1, a2, a3, a4 = pop(H)
        b1, b2, b3, b4 = pop(F)
        print(f"  {'':<10}{'深窗>6dB':>10}{'⛔窗外>6dB':>12}{'泄漏深p90':>11}{'⛔泄漏窗外':>11}")
        print(f"  {'出厂':<10}{a1:>10d}{a2:>12d}{a3:>11.2f}{a4:>11.2f}")
        print(f"  {'修法':<10}{b1:>10d}{b2:>12d}{b3:>11.2f}{b4:>11.2f}")
        print(f"  {'Δ':<10}{b1-a1:>+10d}{b2-a2:>+12d}{b3-a3:>+11.2f}{b4-a4:>+11.2f}")

        # 咬字轴
        inwin = set()
        for s, e, sh in g:
            if abs(sh) >= 12:
                inwin.update(range(s, e + 1))
        def diction(Y):
            y, sr = Y
            out = {"鼻音": [], "塞音": []}
            for k, t in enumerate(tri):
                if t["note_num"] <= 0 or k not in inwin:
                    continue
                c = (t.get("lyric", "") or "?")[0]
                cl = "鼻音" if c in NASAL else ("塞音" if c in STOP else None)
                if not cl:
                    continue
                t0, t1 = T[k], T[k + 1]
                if t1 - t0 < 0.08:
                    continue
                out[cl].append(dip(y, sr, (t0 + t1) / 2, half=(t1 - t0) / 2))
            return {k: (np.nanmedian(v) if v else float("nan")) for k, v in out.items()}
        dh, df = diction(H), diction(F)
        print(f"  咬字轴(深窗内谷深,设计值 鼻 11.4 / 塞 11.7):"
              f" 出厂 鼻 {dh['鼻音']:.1f} 塞 {dh['塞音']:.1f} → 修法 鼻 {df['鼻音']:.1f} 塞 {df['塞音']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\MyDev\TESTING\s159zb_final")
    a = ap.parse_args()
    D = a.dir
    LCH = r"D:\MyDev\TESTING\s145_range_color\lch\lch_score.json"
    GOOSE = r"D:\MyDev\TESTING\不为人所知的鹅妈妈童谣\probe\mg_score.json"
    p = lambda t, m: os.path.join(D, f"mg_deadonly_{t}_{m}.wav")
    pl = lambda t, m: os.path.join(D, f"mg_deadonly_{t}_{m}.plan.json")

    print("=" * 96)
    print("① ground truth —— 炉心融解 +7(用户点名的 12 处就在这里)")
    print("=" * 96)
    report([("yachiyo +7", p("f_null", "yachiyo_runami"), p("f_ya", "yachiyo_runami"), pl("f_ya", "yachiyo_runami"))],
           LCH, TRUTH_YA)
    report([("akiko +7", p("f_null", "yachiyo_runami"), p("f_ak", "akiko_320000"), pl("f_ak", "akiko_320000"))],
           LCH, None)
    print("\n" + "=" * 96)
    print("② 原 key(用户说干净 ⇒ 不许退化)")
    print("=" * 96)
    report([("yachiyo 原key", p("f_k_h", "yachiyo_runami"), p("f_k_f", "yachiyo_runami"), pl("f_k_f", "yachiyo_runami"))],
           LCH, None)
    print("\n" + "=" * 96)
    print("③ 鹅妈妈 +7(死音率 51.9%,比炉心融解还狠)")
    print("=" * 96)
    for m, t in (("yachiyo_runami", "g_ya"), ("yuyuko", "g_yu")):
        report([(m, p(f"{t}_h", m), p(f"{t}_f", m), pl(f"{t}_f", m))], GOOSE, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
