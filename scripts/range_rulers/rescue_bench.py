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

# ⛔⛔ S162 —— 这三个**不是常量**,是「当前被测那条臂的参数」。默认值是 SoVITS 44.1 kHz 的,
#    因为 S159zz 那一批读数都在那上面量的;**换模型必须先 `configure_from_plan()`**。
#    盘上实测的三种 hop(= `samples / total_frames`):882.02(sovits 44.1 k)·
#    960.0(yachiyo rvc 48 k)· 800.0(yuyuko rvc 40 k)。
#    拿 882 去量 48 k 的臂:末端错位 23.6 s,而且**一条 span 都不会被丢掉**(方向相反)
#    ⇒ 输出一整张看起来正常、每个数都无意义的表。这就是 `_alignment_gate` 存在的全部理由。
SR, HOP, N = 44100, 882, 8192
W = np.hanning(N)
FR = SR / N


def configure(sample_rate, hop, n=None):
    """按这条臂重设采样率 / 帧长 / FFT 长度。⛔ 每换一条臂都要调。"""
    global SR, HOP, N, W, FR
    SR = int(sample_rate)
    HOP = float(hop)
    if n:
        N = int(n)
        W = np.hanning(N)
    FR = SR / N
    return SR, HOP, N


def configure_from_plan(plan_path):
    """从生产落的 `*.plan.json` 读 —— `sample_rate` 与 `samples / total_frames` 都是现成字段。

    ⛔ 别写字面量:S159g 那次拿 48 k 去读 44.1 k 的转储,整批读数全废而且长得很像真读数。
    """
    p = json.load(open(plan_path, encoding="utf-8"))
    return configure(p["sample_rate"], p["samples"] / p["total_frames"])


class Spans(list):
    """`note_spans` 的返回值 —— 多带一个 `total_frames`,好让对齐闸有东西可对。"""

    total_frames = 0
    hop = 0.0


def _alignment_gate(spans, n_samples):
    """⛔ 响亮失败:谱面总帧数 × HOP 必须与缓冲长度对得上(容差 1%)。

    这条闸挡的是**两个方向**的错:HOP 偏小 ⇒ 末端错位但一条都不丢(48 k 那种,
    `if b > n: continue` 结构上看不见);HOP 偏大 ⇒ 末端被静静丢掉(40 k 那种)。
    """
    tf = getattr(spans, "total_frames", 0)
    if not tf:
        return
    want = tf * HOP
    if abs(want - n_samples) > 0.01 * max(want, n_samples):
        raise ValueError(
            "⛔ 帧→样本对不上:谱面 %d 帧 × HOP %.2f = %.0f 样本,而缓冲是 %d 样本(差 %.1f%%)。"
            "\n   多半是这条臂不是 44.1 kHz —— 先 `rescue_bench.configure_from_plan(<那条臂的 plan.json>)`。"
            % (tf, HOP, want, n_samples, 100.0 * abs(want - n_samples) / max(want, n_samples))
        )

SCORES = {
    "goose": (r"D:\MyDev\TESTING\不为人所知的鹅妈妈童谣\probe\mg_score.json", 7),
    "lch": (r"D:\MyDev\TESTING\s145_range_color\lch\lch_score.json", 7),
}


def note_spans(score_path, transpose, usable=(36, 79), min_samples=N):
    """音表:从**谱面自己的** `frames` 累加。⛔ 别拿别的谱面那份(S159zi 那族的错法)。"""
    tri = json.load(open(score_path, encoding="utf-8"))["triples"]
    out, f = Spans(), 0
    for k, t in enumerate(tri):
        if t["note_num"] > 0:
            midi = t["note_num"] + transpose
            a, b = int(f * HOP), int((f + t["frames"]) * HOP)
            if usable[0] <= midi <= usable[1] - 1 and b - a >= min_samples:
                out.append((k, midi, a, b, t["lyric"]))
        f += t["frames"]
    out.total_frames = f
    out.hop = HOP
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
    """`base` = 原生@目标音高;`cand` = 被救回目标音高的那一版。⛔ 两者必须同音高。

    ⛔ S162 —— 进来先过 `_alignment_gate`:`HOP` 与这条臂对不上就**响亮失败**,
    不许再出现「一整张看起来正常、每个数都无意义」的表。
    """
    rows = []
    n = min(len(base), len(cand))
    _alignment_gate(spans, n)
    dropped = 0
    for _k, midi, a, b, _ly in spans:
        if b > n:
            dropped += 1
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
        # ⛔⛔ S162 —— 面状护栏必须是【形状】量，不是绝对带能量差。
        #    原实现 = `band(cand,sub) - band(base,sub)`，而 donor 转储是【峰值归一之前】的
        #    (钩子区：比成品低 9-11 dB）⇒ 整体变安静会被它读成「面状变好了」。
        #    实测：S159zzb 上它逐档读 -10…-12 dB（看着很安全），而同一批的 `level`
        #    是 -10…-13.6 ⇒ 扣掉电平之后是 **+0.2 → +3.0 dB 随深度单调涨**。
        #    ⇒ 改成与 `tilt` 同款的差之差（各自减自己的广带），增益不影响它。
        #    ⚠ 旧口径保留成 `subf0_abs`，因为 S159zzl 那一批读数是在那个口径上量的。
        r["subf0_abs"] = _band_db(cand[a:b], 0.25 * f0, 0.75 * f0) - _band_db(base[a:b], 0.25 * f0, 0.75 * f0)
        r["subf0"] = (_band_db(cand[a:b], 0.25 * f0, 0.75 * f0) - _band_db(cand[a:b], 600, 6000)) -                      (_band_db(base[a:b], 0.25 * f0, 0.75 * f0) - _band_db(base[a:b], 600, 6000))
        cu, cv = comb_depth(base, a, b, f0), comb_depth(cand, a, b, f0)
        r["comb"] = (cv - cu) if (cu is not None and cv is not None) else np.nan
        r["level"] = 10 * np.log10(((cand[a:b] ** 2).mean() + 1e-30) / ((base[a:b] ** 2).mean() + 1e-30))
        r["tilt"] = (_band_db(cand[a:b], 8000, 12000) - _band_db(cand[a:b], 300, 1000)) - \
                    (_band_db(base[a:b], 8000, 12000) - _band_db(base[a:b], 300, 1000))
        rows.append(r)
    if spans and dropped > 0.02 * len(spans):
        raise ValueError("⛔ %d/%d 条 span 超出缓冲被丢掉(>2%%)—— 先 configure_from_plan()"
                         % (dropped, len(spans)))
    return rows


def summarise(rows, keys=("H1-H3", "H1-H4", "envmod", "subf0", "subf0_abs", "comb", "level", "tilt")):
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


def selfcheck():
    """⛔ 含**变异**:把 HOP 调错必须当场 raise,否则这条闸是空的。"""
    ok = True
    spans = Spans([(0, 70, 0, 8192, "a")])
    spans.total_frames = 1000
    spans.hop = HOP
    good = int(1000 * HOP)
    try:
        _alignment_gate(spans, good)
        print("① 对齐闸在正确长度上放行 : PASS")
    except ValueError as e:  # noqa: BLE001
        print("① 对齐闸在正确长度上放行 : FAIL", e)
        ok = False
    # 变异:同一份 spans 拿 48 kHz 的缓冲长度(hop 960)⇒ 必须红
    try:
        _alignment_gate(spans, int(1000 * 960))
        print("② 变异(48 k 缓冲)必须红 : FAIL —— 闸没响,它是空的")
        ok = False
    except ValueError:
        print("② 变异(48 k 缓冲)必须红 : PASS")
    # 变异:40 kHz(hop 800)⇒ 也必须红
    try:
        _alignment_gate(spans, int(1000 * 800))
        print("③ 变异(40 k 缓冲)必须红 : FAIL")
        ok = False
    except ValueError:
        print("③ 变异(40 k 缓冲)必须红 : PASS")
    # configure 之后必须放行
    sr0, hop0, n0 = SR, HOP, N
    configure(48000, 960)
    spans2 = Spans([(0, 70, 0, 8192, "a")])
    spans2.total_frames = 1000
    try:
        _alignment_gate(spans2, int(1000 * 960))
        print("④ configure(48000,960) 之后放行 : PASS")
    except ValueError as e:  # noqa: BLE001
        print("④ configure(48000,960) 之后放行 : FAIL", e)
        ok = False
    configure(sr0, hop0, n0)
    print("SELFCHECK:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    print(__doc__.split("\n")[1])
    print("⇒ 这是库,不是脚本;由候选刀的 sweep 脚本 import 它。")
    import sys as _sys
    if "--selfcheck" in _sys.argv:
        _sys.exit(0 if selfcheck() else 1)
