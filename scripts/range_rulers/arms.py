# -*- coding: utf-8 -*-
"""参照臂 —— 把同一段音频升 N 个半音的两种「不重估包络」的做法。

这两条臂**不是候选实现**,是**参照系**:笔 1 的 Rust TD-PSOLA 要和 `praat_psola` 对拍。
⚠ 二者都跑在 python 侧、离线,永远不会进产品;它们的价值就是「一个已知透明的答案」。

* `praat_psola` —— praat 的 TD-PSOLA(Manipulation + PitchTier ×ratio + overlap-add 重合成)。
  共振峰**由构造保持**(时域上只是把周期排密,从不重新估计包络)。S145 在真素材上读到
  包络位移只漏 +0.30 半音(现行 Signalsmith κ=0 漏 +2.40)。
* `world_resynth` —— WORLD 源-滤波重合成(f0 换掉、`sp`/`ap` 原样)。
  ⛔ **它已经在 S145 出局**:塌清塞音(用户两次耳判「つつまれてきれい 的 き」),
  而本工装的**两把仪器都看不见这个失效**。留着它只有一个用途:
  **当 HNR/瞬态尺的已知阴性样本** —— 一个「仪器说没事、耳朵说完蛋」的实例,
  提醒读数的人这套判据有一个已登记的盲区。⛔ 不许把它当候选。
"""

from __future__ import annotations

import numpy as np

# praat "To Manipulation" 的三个参数(S145 原样;改动会改变参照臂 ⇒ 判据跟着变)
PRAAT_TIME_STEP = 0.01
PRAAT_PITCH_FLOOR = 75.0
PRAAT_PITCH_CEILING = 1400.0


def _fit(y, n):
    """长度对齐到 n(参照臂也要守逐样本长度契约)。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def praat_psola(x, sr, ratio):
    """praat TD-PSOLA:音高 ×ratio,共振峰保持,时长不变。"""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=sr)
    man = call(snd, "To Manipulation", PRAAT_TIME_STEP, PRAAT_PITCH_FLOOR, PRAAT_PITCH_CEILING)
    pt = call(man, "Extract pitch tier")
    call(pt, "Multiply frequencies", snd.xmin, snd.xmax, ratio)
    call([pt, man], "Replace pitch tier")
    out = call(man, "Get resynthesis (overlap-add)")
    return _fit(np.asarray(out.values), len(x))


def world_resynth(x, sr, ratio):
    """WORLD 源-滤波重合成:f0 ×ratio,谱包络与非周期性原样。⛔ 已出局,只作盲区样本。"""
    import pyworld as pw

    x = np.asarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=80.0, f0_ceil=1500.0, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    y = pw.synthesize(f0 * ratio, sp, ap, sr, frame_period=5.0)
    return _fit(y, len(x))


def peak_to(y, peak=0.92):
    """S145 的臂全部归一到 0.92 峰值再落盘 —— 复现读数必须照做。

    ⚠ 包络位移尺逐帧去均值 ⇒ 对电平免疫;但 **HNR 与瞬态尺不是**,而 16-bit 落盘还会
       clamp。⇒ 生成臂时统一归一,别让电平差混进读数。
    """
    y = np.asarray(y, dtype=np.float64)
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return y * (peak / m) if m > 1e-9 else y
