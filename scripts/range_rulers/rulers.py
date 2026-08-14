# -*- coding: utf-8 -*-
"""音域扩展逆变换的四把尺子 —— 单一实现,selfcheck 与 compare 都从这里进。

⛔ 为什么是【四把】而不是队列 §E13 笔 0 写的两把:
    2026-07-26 我们把 PSOLA 换成 Signalsmith,理由是「PSOLA 有些地方脏」。守那份实现的闸
    `psola_ab.rs` **只量音高** ⇒ 一个实现缺陷结构上溜了过去,而仓里同时躺着一条实测:
    praat 的 TD-PSOLA 透明(ΔHNR −0.12~+2.68 dB),**我们那份 Rust 实现掉 5.15~8.57 dB**。
    ⇒ **分开 praat 与我们那份坏实现的那把尺子是 HNR**,而包络位移尺与瞬态尺**都看不见它**:
       * 包络位移尺:PSOLA 由构造保持共振峰 —— 哪怕 OLA 的相位全错、嗡嗡作响,它照样读 ~0。
       * 瞬态尺:量的是 2-11 kHz 的正向谱通量,对稳态段的嗡嗡声零分辨力。
    所以本工装 = 包络位移 + 瞬态 + **HNR** + **f0**,而 f0 那把**明确登记为「必要不充分」**——
    它就是当年那把放行了坏实现的尺子,留着是为了让「只看它」这件事在代码里显形。

## 每把尺子答什么 / 不答什么(边界即判据能力)

| 尺子 | 答 | ⛔ 答不了 |
|---|---|---|
| `envelope_shift` | 共振峰包络被搬了几个半音 | 哪个更好听;OLA 相位错没错 |
| `transient_flux` | 2-11 kHz 瞬态(塞音爆发)有没有被抹平 | **WORLD 塌清塞音它看不见**(S145 实测只读 −0.37 dB,而对人为抹平能读 −33 dB)⇒ 该轴唯一仪器是耳朵 |
| `hnr` | 谐波/噪声比掉了多少(= 重合成脏不脏) | 共振峰位置对不对 |
| `f0_error` | 音高准不准 | **音色、纹理、响度全都不管** —— 2026-07 就是它放行的 |

⚠ 全部四把都只能在**真实素材**上下结论:合成周期信号系统性冤枉 PSOLA 类算法
   (S81 被误导三次;[[project_v2_range_extend_quality]] §7-1)。
   `selfcheck.py` 里的合成信号**只用来标定尺子本身**,不许拿去判算法好坏。
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------- 共用

FRAME_PERIOD_MS = 5.0  # pyworld 逐帧步长;所有基于 WORLD 的尺子共用这一个网格
F0_FLOOR, F0_CEIL = 80.0, 1500.0


def load_mono(path):
    """读成 float64 单声道。返回 (x, sr)。"""
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def samples_to_world_frames(spans, sr, n_frames):
    """样本区间 -> WORLD 帧下标集合(升序去重)。spans 为 None ⇒ 全曲。"""
    if not spans:
        return np.arange(n_frames)
    hop = sr * FRAME_PERIOD_MS / 1000.0
    idx = set()
    for a, b in spans:
        fa = int(max(a, 0) / hop)
        fb = int(max(b, 0) / hop)
        idx.update(range(max(fa, 0), min(fb, n_frames)))
    return np.array(sorted(idx), dtype=int)


# ---------------------------------------------------------------- ① 包络位移尺

# 对数频率轴:200..9000 Hz,每格 0.1 半音(S145 原样)
FLO, FHI, CENTS_PER_BIN = 200.0, 9000.0, 10.0
NAX = int(np.round(1200.0 * np.log2(FHI / FLO) / CENTS_PER_BIN))
LOGAX = FLO * 2 ** (np.arange(NAX) * CENTS_PER_BIN / 1200.0)
MAXLAG = int(np.round(100.0 * 9.0 / CENTS_PER_BIN))  # ±9 半音搜索域


def world_analyse(x, sr):
    """pyworld harvest->stonemask->cheaptrick(->d4c) + 对数轴包络。"""
    import pyworld as pw

    f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL,
                       frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    fftsize = (sp.shape[1] - 1) * 2
    freqs = np.arange(sp.shape[1]) * sr / fftsize
    logenv = np.empty((sp.shape[0], NAX), dtype=np.float64)
    db = 10.0 * np.log10(sp + 1e-30)
    for i in range(sp.shape[0]):
        logenv[i] = np.interp(LOGAX, freqs, db[i])
    return dict(sr=sr, f0=f0, sp=sp, logenv=logenv, n=sp.shape[0])


def envelope_shift(ref, cand, idx):
    """把 ref 的对数包络平移多少格才最像 cand。

    ⛔ 选帧规则里**不许出现被测变量**(g2p_rulers/README.md 第 5 条):`idx` 只由
       **参照臂**的 f0 决定。候选臂丢掉的浊音**不许被静默丢弃** —— 它单独作为
       `voiced_survival` 报出来(一个把高音唱没了的候选会得到漂亮的位移读数,
       S145 的 base f0/4、f0/8 干预臂就是这样:残余 +0.20 而浊帧 8/51)。
    ⛔ `peak_corr` 必须与位移一起看:包络被抹平时位移会读成一个漂亮的 0。
    返回 dict;`shifts` 为逐帧半音位移。
    """
    shifts, peak, zero = [], [], []
    survived = 0
    for i in idx:
        if i >= ref["n"] or i >= cand["n"]:
            continue
        if cand["f0"][i] > 0:
            survived += 1
        a = ref["logenv"][i]
        b = cand["logenv"][i]
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        best_lag, best_c = 0, -2.0
        for lag in range(-MAXLAG, MAXLAG + 1):  # 正 lag = cand 相对 ref 向高频移动
            if lag >= 0:
                aa, bb = a[: NAX - lag], b[lag:]
            else:
                aa, bb = a[-lag:], b[: NAX + lag]
            c = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-30))
            if c > best_c:
                best_c, best_lag = c, lag
        shifts.append(best_lag * CENTS_PER_BIN / 100.0)
        peak.append(best_c)
        zero.append(float(np.dot(a, b) / (na * nb)))
    if not shifts:
        return dict(n=0, median_st=float("nan"), p25=float("nan"), p75=float("nan"),
                    peak_corr=float("nan"), zero_corr=float("nan"),
                    voiced_survival=0.0, shifts=np.zeros(0))
    sh = np.array(shifts)
    return dict(
        n=len(sh),
        median_st=float(np.median(sh)),
        p25=float(np.percentile(sh, 25)),
        p75=float(np.percentile(sh, 75)),
        peak_corr=float(np.median(peak)),
        zero_corr=float(np.median(zero)),
        voiced_survival=survived / max(len(idx), 1),
        shifts=sh,
    )


# ---------------------------------------------------------------- ② 瞬态尺

NFFT, HOP = 1024, 220  # 23 ms 窗 / 5 ms 跳 @44.1k
FLUX_LO, FLUX_HI = 2000.0, 11000.0
# ⚠ 阳性对照的平滑核 = 2 ms,**不是** S145 脚本注释里写的 20 ms(注释是错的,代码是
#   `int(sr * 0.002)`)。登记的 −33 dB 是 2 ms 那一档;改成 20 ms 就再也复现不出那个数。
SMEAR_SECONDS = 0.002


def transient_flux(x, sr):
    """2-11 kHz 正向谱通量,逐帧。塞音爆发 = 这个频带上的一次陡升。"""
    w = np.hanning(NFFT)
    n = 1 + max(len(x) - NFFT, 0) // HOP
    f = np.fft.rfftfreq(NFFT, 1.0 / sr)
    m = (f >= FLUX_LO) & (f <= FLUX_HI)
    prev, out = None, np.zeros(n)
    for i in range(n):
        seg = x[i * HOP:i * HOP + NFFT] * w
        mag = np.abs(np.fft.rfft(seg))[m]
        if prev is not None:
            d = mag - prev
            out[i] = float(np.sum(d[d > 0]))
        prev = mag
    return out


def flux_frames(spans, sr, n_frames):
    if not spans:
        return np.arange(n_frames)
    idx = set()
    for a, b in spans:
        idx.update(range(max(int(a) // HOP, 0), min(int(b) // HOP, n_frames)))
    return np.array(sorted(idx), dtype=int)


def flux_ratio_db(cand_flux, ref_flux, idx):
    """候选相对参照的通量比(dB)。负 = 瞬态被抹平。"""
    n = min(len(cand_flux), len(ref_flux))
    sel = idx[idx < n]
    if len(sel) == 0:
        return dict(median_db=float("nan"), peak_db=float("nan"), n=0)
    a = max(float(np.median(cand_flux[sel])), 1e-9)
    b = max(float(np.median(ref_flux[sel])), 1e-9)
    pa = max(float(np.max(cand_flux[sel])), 1e-9)
    pb = max(float(np.max(ref_flux[sel])), 1e-9)
    return dict(median_db=20.0 * np.log10(a / b),
                peak_db=20.0 * np.log10(pa / pb), n=len(sel))


def smear(x, sr):
    """阳性对照:人为抹平瞬态(移动平均)。"""
    k = max(int(sr * SMEAR_SECONDS), 1)
    return np.convolve(x, np.ones(k) / k, mode="same")


# ---------------------------------------------------------------- ③ HNR 尺

HNR_TIME_STEP = 0.01
HNR_MIN_PITCH = 75.0
HNR_UNDEFINED = -200.0


def hnr_track(x, sr):
    """praat harmonicity (cc),逐帧 dB。未定义帧 = -200 ⇒ 转成 NaN。"""
    import parselmouth

    snd = parselmouth.Sound(x, sampling_frequency=sr)
    h = snd.to_harmonicity_cc(time_step=HNR_TIME_STEP, minimum_pitch=HNR_MIN_PITCH)
    v = np.asarray(h.values).ravel().astype(np.float64)
    v[v <= HNR_UNDEFINED + 1e-6] = np.nan
    return v, float(h.x1), float(h.dx)


def hnr_median(x, sr, spans=None):
    """窗内 HNR 中位数(dB)。spans = 样本区间。"""
    v, x1, dx = hnr_track(x, sr)
    if spans:
        idx = set()
        for a, b in spans:
            fa = int(round((a / sr - x1) / dx))
            fb = int(round((b / sr - x1) / dx))
            idx.update(range(max(fa, 0), min(fb, len(v))))
        sel = np.array(sorted(idx), dtype=int)
        v = v[sel] if len(sel) else np.zeros(0)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


# ---------------------------------------------------------------- ④ f0 尺(必要不充分)

def f0_error_cents(ref, cand, idx, ratio):
    """候选相对「参照 × ratio」的音高误差(cents)。

    ⛔ **必要不充分**。这就是 2026-07 那道 `psola_ab.rs` 唯一量过的东西,而它放行了
       一份 HNR 掉 5-8 dB 的实现。⇒ 任何验收里它都不许单独出现。
    """
    errs = []
    lost = 0
    for i in idx:
        if i >= ref["n"] or i >= cand["n"]:
            continue
        fr = ref["f0"][i]
        fc = cand["f0"][i]
        if fr <= 0:
            continue
        if fc <= 0:
            lost += 1
            continue
        errs.append(1200.0 * np.log2(fc / (fr * ratio)))
    if not errs:
        return dict(n=0, median_cents=float("nan"), p90_abs_cents=float("nan"), lost_voiced=lost)
    e = np.array(errs)
    return dict(n=len(e), median_cents=float(np.median(e)),
                p90_abs_cents=float(np.percentile(np.abs(e), 90)), lost_voiced=lost)
