# -*- coding: utf-8 -*-
"""S159u —— **看得见**:一份渲染的「缺陷接触印相」(contact sheet)。

## ⛔ 它为什么存在(用户 2026-08-21 的原话:「更棘手的其实是这些问题你到底能不能看得见」)

同一场里我为这条线造了**八把标量尺子**,**八把都与用户耳判相反**:
边界电平台阶 · 谐波间噪声 · 谱平坦度 · 周期性丢失 · 八度错误 · donor 基频泄漏 ·
donor 音高透出量 · 宽带咔哒探测。而每一次真正找到东西,都是因为**去看了图**:
* 「竖直条纹」是在频谱图上先看见的;
* 那 10 个洞是看见竖线之后去**波形**上确认 `|x| < 1e-7` 才找到的,
  而同一份音频上我那把「全带同时跳 >6 dB」的探测器报 **0 个**。

⇒ 根因不是「尺子调得不好」,是**每把标量尺子都编码了我当时那个假说**,
所以它只会确认我已经相信的东西。**图不编码假说。**
⇒ 这份工具的产物是**给人看的图**,外加一小撮**形状**判据 —— 而每条形状判据都必须
先在「用户独立确认过的病例」上验过阳性,才允许写进来(见 `SHAPES` 每条的出处)。

## 用法
```
python scripts/range_rulers/artifact_sheet.py <render.wav> [--ref 源.wav]
       [--score 谱.json] [--plan 计划.txt] [--out 目录] [--top 12]
```
`--plan` = 每行 `起 止 位移`(音符下标),即审计行 `notes[a..=b] renders at s st` 的抄本。

## ⛔ 硬规矩
* **它不判定「好/坏」**,只**排序候选并把图画出来**。判定要靠眼睛和耳朵。
* 每条形状判据都带 `evidence`(它在哪个真实病例上被验过)。**没有出处的判据不许加进来。**
* 阴性对照写在同一张表里(哪个模型/哪条臂上它读到 0),否则「报了 N 个」什么也不说明。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - 环境缺件时给人话
    sys.exit("需要 soundfile:pip install soundfile")


# ─────────────────────────── 形状判据 ───────────────────────────
# ⛔ 每条都必须写清楚:它长什么样 · 它在哪个**用户独立确认过**的病例上验过阳性 ·
#    以及它的**阴性对照**(哪条臂上读到 0)。没有这三样的判据不许进这张表。
SHAPES = [
    {
        "id": "digital_silence_in_note",
        "what": "唱音内部的绝对数字静音(|x| < 1e-7,>8 ms)—— 两端硬边 ⇒ 频谱上贯穿全频的竖线",
        "evidence": "S159t:用户报「咔哒声/竖直条纹又回来了」。实机导出 yachiyo +7 **10 个**,"
                    "akiko +7 **0 个**(阴性对照)。含用户点名的 `[685]「ぴゃ」`。"
                    "⚠ 它是先在频谱图上看见竖线、再去波形上确认的;"
                    "同一份音频上「全带同时跳 >6 dB」的探测器报 **0 个**。",
    },
    {
        "id": "sub_fundamental_sheet",
        "what": "基频**以下**的成片能量(60 Hz..0.85·f0 的能量占比异常高)—— donor 的基频漏出来",
        "evidence": "S159t:用户点名 yachiyo 43.90/44.05 s 的低频竖线与 akiko 269.7-270.4 s 的面状伪影;"
                    "两处都在 0-600 Hz 而基频在 630/1000 Hz。深位移(−12/−15)时 donor 基频正落在那一段。"
                    "同族即 S157b 修过的「基频附近的合唱感」。",
    },
    {
        "id": "hard_level_step",
        "what": "5 ms 包络上的硬台阶(>12 dB),且不是正常起音(前后都在唱)",
        "evidence": "S159t:yachiyo +7 上 4 处(50.45 / 128.75 / 175.21 / 188.25 s),akiko 0 处。"
                    "⚠ 它只抓到了 `digital_silence_in_note` 的一个子集 —— 留着是因为它能抓**没到绝对零**的那一类。",
    },
]


def load(path: str):
    y, sr = sf.read(path, dtype="float64", always_2d=True)
    return y.mean(1), sr


def lead_silence(y: np.ndarray, sr: int) -> float:
    nz = np.nonzero(np.abs(y) > 1e-4)[0]
    return float(nz[0]) / sr if len(nz) else 0.0


def notes_from_score(path: str):
    tri = json.load(open(path, encoding="utf-8"))["triples"]
    t, out = 0.0, []
    for k, n in enumerate(tri):
        d = max(0, n["frames"]) / 50.0
        out.append((k, t, t + d, n.get("lyric", ""), n.get("note_num", 0)))
        t += d
    return out


def sung_at(notes, x: float):
    for k, a, b, ly, nn in notes:
        if a <= x < b and nn > 0:
            return k, a, b, ly, nn
    return None


def find_digital_silence(y, sr, notes, off):
    """绝对零的连段;若给了谱,只留**落在唱音里**的(休止被静音是设计如此)。"""
    z = np.abs(y) < 1e-7
    hits, i, n = [], 0, len(z)
    while i < n:
        if z[i]:
            j = i
            while j + 1 < n and z[j + 1]:
                j += 1
            a, b = i / sr, (j + 1) / sr
            if b - a > 0.008 and a > 1.0 and b < n / sr - 1.0:
                if notes is None:
                    hits.append((a, b, "静音段"))
                else:
                    s = sung_at(notes, (a + b) / 2 - off)
                    if s:
                        k, na, nb, ly, nn = s
                        ov = min(b - off, nb) - max(a - off, na)
                        if ov > 0.6 * (b - a):
                            hits.append((a, b, f"音[{k}]{ly} midi={nn}"))
            i = j + 1
        else:
            i += 1
    return hits


def find_sub_fundamental(y, sr, win=0.150, hop=0.075):
    """基频以下的能量占比;返回 (时刻, dB, f0)。⛔ 只在**找得到基频**的帧上算。"""
    w = int(sr * win)
    out = []
    for i in range(w, len(y) - w, int(sr * hop)):
        x = y[i - w // 2:i + w // 2]
        if np.sqrt(np.mean(x ** 2)) < 2e-3:
            continue
        d = x - x.mean()
        ac = np.correlate(d, d, "full")[len(d) - 1:]
        ac = ac / (ac[0] + 1e-30)
        lo, hi = int(sr / 1400), int(sr / 80)
        seg = ac[lo:hi]
        if len(seg) < 3:
            continue
        k = int(np.argmax(seg)) + lo
        if ac[k] < 0.40:
            continue
        f0 = sr / k
        xw = x * np.hanning(len(x))
        P = np.abs(np.fft.rfft(xw)) ** 2
        f = np.fft.rfftfreq(len(xw), 1 / sr)
        below = (f > 60) & (f < 0.85 * f0)
        above = (f >= 0.85 * f0) & (f < 8000)
        if below.any() and above.any():
            out.append((i / sr, 10 * np.log10((P[below].sum() + 1e-30) / (P[above].sum() + 1e-30)), f0))
    return out


def find_level_steps(y, sr, thr=12.0):
    h = int(sr * 0.005)
    K = len(y) // h
    e = 20 * np.log10(np.sqrt(np.mean(y[:K * h].reshape(K, h) ** 2, 1)) + 1e-12)
    d = np.diff(e)
    loud = e[:-1] > np.percentile(e, 55)
    idx = np.nonzero(loud & (np.abs(d) > thr))[0]
    return [(i * 0.005, float(d[i])) for i in idx]


def contact_sheet(path_out, arms, spots, title, ylim=(0, 8000), half=0.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(spots)
    if n == 0:
        return None
    fig, ax = plt.subplots(len(arms), n, figsize=(3.4 * n, 3.2 * len(arms)), squeeze=False)
    for j, (t, lab) in enumerate(spots):
        for i, (tag, y, sr) in enumerate(arms):
            a, b = int((t - half) * sr), int((t + half) * sr)
            a, b = max(0, a), min(len(y), b)
            if b - a < int(sr * 0.05):
                continue
            ax[i][j].specgram(y[a:b], NFFT=2048, Fs=sr, noverlap=1920, cmap="magma", vmin=-110, vmax=-25)
            ax[i][j].set_ylim(*ylim)
            ax[i][j].axvline(half, color="cyan", lw=1.0, ls="--")
            ax[i][j].set_title(f"{tag} {t:.2f}s\n{lab}", fontsize=7)
            ax[i][j].tick_params(labelsize=6)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path_out, dpi=100)
    plt.close(fig)
    return path_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("render")
    ap.add_argument("--ref", default=None, help="参照(源音频或另一条臂)")
    ap.add_argument("--score", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    out_dir = a.out or os.path.dirname(os.path.abspath(a.render))
    os.makedirs(out_dir, exist_ok=True)

    y, sr = load(a.render)
    off = lead_silence(y, sr)
    notes = notes_from_score(a.score) if a.score else None
    arms = [("render", y, sr)]
    if a.ref:
        ry, rsr = load(a.ref)
        arms.append(("ref", ry, rsr))

    print(f"素材 {a.render}\n  {len(y)/sr:.2f}s @{sr} · 峰值 {np.abs(y).max():.4f} · 前导静音 {off:.3f}s")
    print("\n⛔ 这份工具**不判定好坏**,只排序候选并画图。判定靠眼睛和耳朵。\n")

    sil = find_digital_silence(y, sr, notes, off)
    steps = find_level_steps(y, sr)
    sub = find_sub_fundamental(y, sr)
    subv = np.array([s[1] for s in sub]) if sub else np.array([])

    for sh in SHAPES:
        print(f"◆ {sh['id']}\n    形状:{sh['what']}\n    出处:{sh['evidence']}")
    print()
    print(f"digital_silence_in_note : **{len(sil)}** 处" + ("(⚠ 没给 --score ⇒ 未区分休止)" if notes is None else ""))
    for t0, t1, tag in sil[:a.top]:
        print(f"    {t0-off:8.3f}s  {(t1-t0)*1000:5.0f} ms  {tag}")
    print(f"hard_level_step         : **{len(steps)}** 处;前几个 " +
          " ".join(f"{t-off:.2f}s({d:+.0f}dB)" for t, d in steps[:8]))
    if len(subv):
        print(f"sub_fundamental_sheet   : p50 {np.median(subv):.1f} dB · p90 {np.percentile(subv,90):.1f} · max {subv.max():.1f}")
        worst = sorted(sub, key=lambda s: -s[1])[:a.top]
        print("    最高的几处:" + " ".join(f"{t-off:.2f}s({v:.0f}dB,f0={f:.0f})" for t, v, f in worst[:8]))

    spots = [(t0, tag) for t0, t1, tag in sil[:a.top]]
    for t, v, f in sorted(sub, key=lambda s: -s[1])[:max(0, a.top - len(spots))]:
        spots.append((t, f"基频以下 {v:.0f} dB"))
    p1 = contact_sheet(os.path.join(out_dir, "sheet_wide.png"), arms, spots,
                       "候选点 0-8 kHz(青线 = 候选时刻)")
    p2 = contact_sheet(os.path.join(out_dir, "sheet_low.png"), arms, spots,
                       "同一批点 0-1.5 kHz(基频以下看这张)", ylim=(0, 1500), half=0.6)
    print(f"\n⇒ 图:{p1}\n   {p2}\n**下一步是【看这两张图】,不是再读一个数。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
