# -*- coding: utf-8 -*-
"""⑤ `envmod` —— 「油/晃」那条线的尺子(S159zy 立,S159zz 定价)。

## 它答什么

**同一个音、同一个目标音高**下,2-4 kHz 那一段的**能量包络**里有多少能量落在
**20-200 Hz** 的调制上。读数越高(越接近 0)越「油」。

    带通 2-4 kHz → Hilbert 包络 → 去均值 → Welch → 10·log10( P[20..200] / P[1..1000] )

## ⛔ 它答不了(已登记的盲区)

* **它不回答「这个音本身脏不脏」** —— 它量的是「相对同一个音的另一条臂多了多少调制」。
  ⚠ 这正是前面八把标量尺子全军覆没的地方:那些尺子把**根本没被救**的音排成全曲最脏。
* **跨音高比较要小心**:原生渲染上实测 **+0.11 dB/半音**(r = 0.118,n = 137,
  MIDI 68-79)⇒ 混杂小但非零,而且**只在那 11 个半音上采过样**
  (S158 血训:窄区间上的单调性只是那个区间的性质)。⇒ 承重的比较一律**同音同目标音高**。
* **它不回答「哪个更好听」**(README 第 8 条)。今天它有一对用户耳判背书的对照
  (音[1035] 在 lb3 与 lb5 上),没有盲测。

## ⭐ 为什么是「调制占比」而不是「有没有噪声」

用户 2026-08-23 把方向拧对了:「**不完全是共振峰之间【有没有】噪声** —— 3:03.055 也有噪声
却不奇怪,**录音的人声也有**」。⇒ 判别量是噪声的**性质**。实测那个形状**不是一条谱线**,
而是 20-200 Hz **整段抬 12-18 dB**(峰 32 Hz,Q ≈ 0.5)⇒ 包络在**慢速上整体不稳**,
听感是「油/晃」而不是「沙」。⛔ 半整数谐波(周期加倍)那个假说**判负**:
lb5 在 2-4 kHz 上反而比 lb3 低 21 dB。

## ⛔ 台面噪声先量,再读任何差(S159zx 立的规矩)

* **整曲两遍渲染**(同一条命令、同一份计划、同一个 binary):331 个 ≥0.2 s 的浊音上
  |Δ| **中位 1.01 dB · p90 3.00 dB**。⇒ 整曲 A/B 上,**小于 1 dB 的差不算差**。
* **探针**(同一份输入喂 `inverse_probe`,只翻一个旋钮):**没有渲染噪声**,差就是差。
  ⇒ 要分辨小刀一律走探针。
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

# ── 口径(⛔ 动任何一个都要同步 registry.json 的 envmod.caliber)────────────
SR = 44100
BAND_LO, BAND_HI = 2000.0, 4000.0   # 被测的「高次共振峰」带
MOD_LO, MOD_HI = 20.0, 200.0        # 「油/晃」那一段调制
REF_LO, REF_HI = 1.0, 1000.0        # 归一用的整个调制带
NPERSEG = 4096
MIN_SAMPLES = NPERSEG // 2          # 短于这个不给读数(而不是给一个坏读数)


def caliber() -> dict:
    """判据自己会走的那条路径算出来的口径 —— 登记值必须由它产生(README 第 5 条)。"""
    return {
        "SR": SR, "BAND_LO": BAND_LO, "BAND_HI": BAND_HI,
        "MOD_LO": MOD_LO, "MOD_HI": MOD_HI, "REF_LO": REF_LO, "REF_HI": REF_HI,
        "NPERSEG": NPERSEG, "MIN_SAMPLES": MIN_SAMPLES,
    }


_SOS = butter(4, [BAND_LO / (SR / 2), BAND_HI / (SR / 2)], btype="band", output="sos")


def modulation_db(x, band=None, mod=None):
    """一段单声道音频的调制占比(dB)。读不出来返回 `None`,⛔ 不返回 0。"""
    if len(x) < MIN_SAMPLES:
        return None
    sos = _SOS if band is None else butter(
        4, [band[0] / (SR / 2), band[1] / (SR / 2)], btype="band", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, np.asarray(x, dtype="float64"))))
    env = env - env.mean()
    f, p = welch(env, fs=SR, nperseg=min(NPERSEG, len(env)))
    m0, m1 = mod or (MOD_LO, MOD_HI)
    num = p[(f >= m0) & (f <= m1)].sum()
    den = p[(f >= REF_LO) & (f <= REF_HI)].sum()
    if num <= 0 or den <= 0:
        return None
    return float(10.0 * np.log10(num / den))


# ── 取样规则 ⛔ 里面不许出现被测变量(README 第 2 条)────────────────────
def note_spans(dump_json, hop=882, min_frames=10):
    """从 `mg_dump_plan_arms` 的谱面转储里读音表 ⇒ [(k, note, lo, hi)]。

    ⛔ 这里**不复刻任何生产逻辑**:`frames` 是生产产物,累加就是帧起点。
    ⛔⛔ S159zi 血训:**用户给时间戳时,必须程序化列出区间覆盖到的【所有】音**再动尺子 ——
    「最近的音头」那种读法把 `3:16.060-3:16.461` 读成了 [1033],真正覆盖的是 [1034][1035],
    于是五把尺子在一个**真干净**的音上全读「最干净」,连报五次「找不到」。用 `cover()`。
    """
    d = json.load(open(dump_json, encoding="utf-8"))
    out, f = Spans(), 0
    for r in d["notes"]:
        if r["note"] > 0 and r["frames"] >= min_frames:
            out.append((r["k"], r["note"], f * hop, (f + r["frames"]) * hop))
        f += r["frames"]
    out.min_frames = min_frames
    return out


class Spans(list):
    """音表 + 它是**怎么过滤出来的**。见 [`cover`]:过滤过的表不许拿去查覆盖。"""

    min_frames = 0


def cover(spans, t0, t1, sr=SR):
    """一个时间区间**覆盖到的全部音**(⛔ 不是「最近的音头」)。

    ⛔⛔ 必须喂 `note_spans(..., min_frames=0)`。用打分用的那张(默认 ≥10 帧)会**静默漏掉短音** ——
    写这个 helper 的当天我就在它上面复现了一次 S159zi:同一个 `3:16.060-3:16.461`,
    过滤过的表答 `[1035]`,完整的表答 `[1034, 1035]`。⇒ 这里退一条**响亮的**错,
    不是给一个少一项的答案(S129:「跑不起来」不许被读成「通过」)。
    """
    if getattr(spans, "min_frames", 0) != 0:
        raise ValueError(
            f"cover() 拿到的是过滤过的音表(min_frames={spans.min_frames})—— "
            "它会静默漏掉短音。请用 note_spans(dump, min_frames=0)。")
    a, b = int(t0 * sr), int(t1 * sr)
    return [s for s in spans if s[2] < b and s[3] > a]


def score(path_or_array, spans):
    """逐音打分 ⇒ {k: dB}。`.f32` 走原始浮点(⛔ 别用 PCM_16:量化地板骗过我们一次)。"""
    if isinstance(path_or_array, str):
        if path_or_array.endswith(".f32"):
            x = np.fromfile(path_or_array, dtype="<f4").astype("float64")
        else:
            import soundfile as sf
            x, sr = sf.read(path_or_array, always_2d=False)
            assert sr == SR, (path_or_array, sr)
            x = x.mean(axis=1) if x.ndim > 1 else x
    else:
        x = np.asarray(path_or_array, dtype="float64")
    out = {}
    for k, _n, lo, hi in spans:
        if hi <= len(x):
            v = modulation_db(x[lo:hi])
            if v is not None:
                out[k] = v
    return out


if __name__ == "__main__":  # 手工用:envmod.py <dump.json> <a.wav> [b.wav]
    sp = note_spans(sys.argv[1])
    for p in sys.argv[2:]:
        s = score(p, sp)
        print(f"{p}: n={len(s)} 中位 {np.median(list(s.values())):+.2f} dB")
