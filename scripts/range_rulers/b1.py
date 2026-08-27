# -*- coding: utf-8 -*-
"""⑥ `b1` —— 噪声残差的**相邻周期相关**(S165 立,起因是一条被推翻的 workflow 结论)。

## 答什么
一段浊音里,**噪声(气声/湍流)有多少是「上一个周期的拷贝」**。
b1 ≈ 拷贝比例:独立湍流 ≈ 0,半数周期是拷贝 ≈ 0.43(本实现,见 registry 的指纹)。
这是本目录**第一把量「能量周期化」**的尺子 —— ①②③④⑤ 一把都够不着这条轴。

## ⛔ 答不了(已登记的盲区)
* **好不好听**。深救援让 b1 掉(打散),浅救援让它涨(锁紧)—— 哪个更难听没有耳判背书。
* **谐波**那一半。它按构造把谐波剥掉了(见下),所以「谐波结构坏没坏」要问 ①③。
* **绝对可比性**:`refine` 开/关是**两把尺子**(选择偏置 ~+0.08),不许混着引用。

## 口径
`r[n] = x[n] − mean_{m=−M..M} x[n+m·T]`(周期同步平均当谐波模板)剥谐波
⇒ 残差相邻两周期的归一化互相关。
* 所有周期都一致的成分(= 谐波)减干净;
* **相邻**周期的拷贝在 2M+1 个周期的平均里只占 2/9 ⇒ 留得下来,正是要量的。

两个**确定性**偏置,都解析算得出来、都在读数里扣掉了:
* 陷波器自身在 lag=T 的偏置 `comb_baseline(M)`,M=4 ⇒ **−0.138889**(实测 −0.1431)
* `refine=True`(±6% 精搜真周期)带 **~+0.08 选择偏置** ⇒ 结论要在开/关两种口径下都成立

## ⛔⛔ 两条血训(这把尺子是踩着它们造出来的)
1. **第一版用「高通 3 kHz」当剥谐波,直接塌了** —— f0=200 Hz 的元音在 3 kHz 以上还有几十次
   谐波,一根都剥不掉 ⇒ 真实 `donor_pre` 读出 0.62-0.94,量的是**谐波**的周期性。
   而第一版的合成台谐波太弱(24 次 · 峰值 −6 dB vs 噪声 0.15 std)⇒ **结构上看不见这个失效**。
2. ⇒ **要判定一把尺子在不在量它自称量的东西,就把它自称免疫的那个量拉大 100 倍。**
   这一版的合成台把谐波铺到 60 次,并加 `harm_amp ×1/×10/×100` 的阴性对照 ——
   三个读数的**极差必须 < 1e-6** 才算过(实测 −0.003726,极差 3.7e-8 = 浮点舍入)。

## 跑
    python b1.py            # 自检 + 与 registry.json 预注册值对拍。0=过 / 1=读数不对 / 3=跑不起来
用法(在 donor 转储上):见 `TESTING\s163_scratch\b1_pairs2.py`,那里写着 f0 口径的四条纪律。
⛔ f0 口径:`donor_f0` = 喂给逆变换的 FED f0 = donor **自己的(低唱的)**音高
   ⇒ `pre` 用 `f0`、`post` 用 `f0×2^(−sh/12)`。**反过来会造出一个漂亮的假效应**
   (S165 §48.2:workflow 就是这么把 −0.19 读成 +0.34 的,而且 6/6 同号)。
"""
import io, json, os, sys, math
import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 48000

def _hp(x, sr, fc):
    sos = butter(4, fc / (sr / 2.0), btype="highpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float64)

def comb_baseline(M):
    """⛔ 陷波器**自己**在 lag=T 上的确定性偏置 —— 独立噪声进去也读不出 0。

    `r[n] = Σ_m a_m·w[n+mT]`,`a_0 = 1 − 1/N`、`a_{m≠0} = −1/N`,`N = 2M+1`。
    对白噪声 `w`:
      `E(r[n]·r[n+T]) = Σ_{m=-M+1}^{M} a_m·a_{m-1} = −2(1−1/N)/N + (2M−2)/N²`
      `var(r) = a_0² + 2M/N² = (1−1/N)² + 2M/N²`
    M=4 ⇒ −10/72 = −0.1389(实测 −0.1431,吻合)。⇒ 归一化 `(raw − base)/(1 − base)`。
    """
    N = 2 * M + 1
    cov = -2.0 * (1.0 - 1.0 / N) / N + (2.0 * M - 2.0) / (N * N)
    var = (1.0 - 1.0 / N) ** 2 + 2.0 * M / (N * N)
    return cov / var

def b1_frames(x, f0, sr=SR, hop=None, hp_hz=2000.0, M=4, min_f0=60.0, level_floor=1e-6,
              debias=True, with_index=False, refine=True):
    """逐帧:剥谐波 ⇒ 残差 ⇒ 相邻两周期的归一化互相关(扣掉陷波器自身的偏置)。"""
    x = np.asarray(x, dtype=np.float64)
    if hp_hz:
        x = _hp(x, sr, hp_hz)
    n = len(x)
    if hop is None:
        hop = sr // 50
    out = []
    keep_i = []
    for i, f in enumerate(f0):
        if not np.isfinite(f) or f < min_f0:
            continue
        T = int(round(sr / float(f)))
        if T < 8:
            continue
        t = i * hop
        if t - M * T < 0 or t + (2 + M) * T > n:
            continue
        if refine:
            # ⛔ 用**参数 f0** 量 PSOLA 的输出会系统性低估:实测 `post` 只有 24-66% 的帧
            # 真的落在参数 f0 上(`f0_check.py`)⇒ 周期对不准,相关自然掉。
            # ⇒ 两边都在 ±6% 内按各自波形精搜真周期,尺子才是公平的。
            lo_T, hi_T = max(8, int(T * 0.94)), int(T * 1.06) + 1
            if t + 2 * hi_T + M * hi_T <= n and t - M * hi_T >= 0:
                seg = x[t:t + 3 * T]
                seg = seg - seg.mean()
                best, bT = -2.0, T
                e0 = float(seg[:T] @ seg[:T])
                for TT in range(lo_T, hi_T):
                    u, v2 = x[t:t + TT], x[t + TT:t + 2 * TT]
                    u = u - u.mean(); v2 = v2 - v2.mean()
                    dd = (float(u @ u) * float(v2 @ v2)) ** 0.5
                    if dd <= 0:
                        continue
                    c = float(u @ v2) / dd
                    if c > best:
                        best, bT = c, TT
                T = bT
                if t - M * T < 0 or t + (2 + M) * T > n:
                    continue
        idx = np.arange(t, t + 2 * T)
        stack = np.empty((2 * M + 1, 2 * T))
        for j, m in enumerate(range(-M, M + 1)):
            stack[j] = x[idx + m * T]
        r = x[idx] - stack.mean(axis=0)
        a, b = r[:T], r[T:]
        if a.std() < level_floor or b.std() < level_floor:
            continue
        a = a - a.mean(); b = b - b.mean()
        d = math.sqrt(float(a @ a) * float(b @ b))
        if d <= 0:
            continue
        out.append(float(a @ b) / d)
        keep_i.append(i)
    v = np.asarray(out)
    if debias and len(v):
        base = comb_baseline(M)
        v = (v - base) / (1.0 - base)
    return (np.asarray(keep_i, dtype=np.int64), v) if with_index else v

def b1(x, f0, **kw):
    v = b1_frames(x, f0, **kw)
    return float(np.mean(v)) if len(v) else float("nan")

# ── 合成台 ────────────────────────────────────────────────────────────────────
def synth(copy_frac, f0_hz=220.0, dur=2.0, sr=SR, seed=0, harm_amp=1.0, noise_amp=0.15, nharm=60):
    """周期性谐波 + 湍流噪声;`copy_frac` 比例的周期,其噪声是**上一周期的拷贝**。

    ⚠ `nharm=60` 是故意的:f0=220 ⇒ 谐波一直铺到 13 kHz,高通剥不掉 —— 这就是
       第一版塌掉的那一族。`harm_amp` 拉大就是**谐波主导**的阴性对照。
    """
    rng = np.random.default_rng(seed)
    T = int(round(sr / f0_hz))
    npd = int(dur * sr) // T
    tt = np.arange(T) / sr
    per = np.zeros(T)
    for k in range(1, nharm + 1):
        per += (1.0 / k) * np.sin(2 * np.pi * k * f0_hz * tt + rng.uniform(0, 2 * np.pi))
    per *= harm_amp / (np.abs(per).max() + 1e-12)
    noi = [rng.standard_normal(T) * noise_amp for _ in range(npd)]
    isc = rng.random(npd) < copy_frac
    for j in range(1, npd):
        if isc[j]:
            noi[j] = noi[j - 1].copy()
    x = np.concatenate([per + nj for nj in noi])
    f0 = np.full(len(x) // (sr // 50) + 1, f0_hz)
    return x, f0

def selftest():
    ok = True
    print("=== ⑴ 剂量:b1 该 ≈ 拷贝比例,且单调 ===")
    print("   refine=True 带 ~+0.08 的选择偏置(精搜总能挑到最相关的那个周期);对 pre/post 一致但不完全抵消")
    print("   => 结论必须在 refine 开/关两种口径下都成立。")
    ms = []
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9):
        vs = [b1(*synth(frac, seed=s)) for s in range(3)]
        ms.append(float(np.mean(vs)))
        print("   copy=%.2f  b1=%.4f" % (frac, ms[-1]))
    mono = all(ms[i] < ms[i + 1] for i in range(len(ms) - 1))
    print("   单调 %s" % ("OK" if mono else "!! 非单调"))
    ok &= mono
    ok &= abs(ms[0]) < 0.10 and abs(ms[2] - 0.5) < 0.15 and ms[4] > 0.65

    print("=== ⑵ ⛔ 阴性对照:**谐波主导**(第一版正是死在这里)===")
    for amp in (1.0, 10.0, 100.0):
        v = float(np.mean([b1(*synth(0.0, seed=s, harm_amp=amp)) for s in range(3)]))
        flag = "OK " if abs(v) < 0.12 else "!! "
        if abs(v) >= 0.12:
            ok = False
        print("   %s谐波幅度 ×%-5.0f 无拷贝 ⇒ b1=%+.4f  (该 ≈ 0;谐波再强也不许把它顶起来)" % (flag, amp, v))

    print("=== ⑶ 阴性对照:纯谐波、零噪声 ⇒ 残差只剩数值噪声 ===")
    v = b1(*synth(0.0, noise_amp=0.0, harm_amp=1.0))
    # 残差恒零 ⇒ 一帧都过不了 `level_floor` ⇒ nan。**这就是对的行为**(不是缺陷):
    # 尺子拒绝在没有噪声的地方给读数,而不是编一个出来。
    good = (not np.isfinite(v)) or abs(v) < 0.2
    print("   b1=%+.4f  %s" % (v, "OK(nan = 拒绝给读数)" if good else "!! "))
    ok_local = good

    ok &= ok_local
    print("=== ⑷ 电平不变性(尺子不许是电平的影子)===")
    x, f = synth(0.5, seed=7)
    v1, v2 = b1(x, f), b1(x * 0.01, f)
    print("   ×1 %.4f  ×0.01 %.4f  Δ=%+.5f  %s" % (v1, v2, v2 - v1, "OK" if abs(v2 - v1) < 0.02 else "!!"))
    ok &= abs(v2 - v1) < 0.02
    return ok

def registry_check():
    """⛔ 预注册对拍(g2p_rulers 第 3 条:期望答案在跑之前写死)。

    两类**不许混着引用**:
      · `predictions`  —— 与本实现无关就能知道的真值(尺子错了才会不过)
      · `fingerprints` —— 本实现在合成台上量出来的数,只用来查漂移
    """
    reg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.json")
    try:
        reg = json.load(io.open(reg_path, encoding="utf-8"))["b1"]
    except Exception as e:
        print("!! 读不到 registry.json 的 b1 段:%s" % e)
        return 3
    bad = 0

    print("=== 预注册 · predictions(真值)===")
    P = reg["predictions"]
    # ⑴ 谐波免疫必须是**逐位**的
    vs = [b1(*synth(0.0, seed=s, harm_amp=a), refine=False) for a in (1.0, 10.0, 100.0) for s in range(3)]
    g1 = [float(np.mean(vs[0:3])), float(np.mean(vs[3:6])), float(np.mean(vs[6:9]))]
    # ⚠ 不是**逐位**相同:谐波幅度变了,加法的舍入顺序也变 ⇒ 实测差 3.7e-8。
    #    那是浮点噪声(相对 1e-5),不是尺子在跟着谐波动 —— 容差按它定,别写成 0。
    spread = max(g1) - min(g1)
    same = spread < P["harmonic_immunity_spread_max"]
    print("   谐波 ×1/×10/×100 一致     : %s  (%.6f, 极差 %.1e)"
          % ("OK" if same else "!! " + repr(g1), g1[0], spread))
    bad += 0 if same else 1
    # ⑵ 独立湍流 ≈ 0
    ok2 = abs(g1[0]) < P["independent_abs_max"]
    print("   独立湍流 |b1| < %.3f      : %s (%.6f)" % (P["independent_abs_max"], "OK" if ok2 else "!!", g1[0]))
    bad += 0 if ok2 else 1
    # ⑶ 剂量单调
    ms = [float(np.mean([b1(*synth(f, seed=s), refine=False) for s in range(3)])) for f in (0.0, 0.25, 0.5, 0.75, 0.9)]
    mono = all(ms[i] < ms[i + 1] for i in range(len(ms) - 1))
    print("   剂量单调                  : %s  (%s)" % ("OK" if mono else "!!", " < ".join("%.3f" % v for v in ms)))
    bad += 0 if mono else 1
    # ⑷ 电平不变(逐位)
    x, f = synth(0.5, seed=7)
    l1, l2 = b1(x, f, refine=False), b1(x * 0.01, f, refine=False)
    ok4 = (l1 == l2)
    print("   电平 ×0.01 逐位不变       : %s  (%.6f / %.6f)" % ("OK" if ok4 else "!!", l1, l2))
    bad += 0 if ok4 else 1
    # ⑸ 陷波器基线是解析值,不是拟合值
    cb = comb_baseline(4)
    ok5 = abs(cb - P["comb_baseline_M4"]) < 1e-12
    print("   comb_baseline(4) 解析     : %s  (%.6f,登记 %.6f)" % ("OK" if ok5 else "!!", cb, P["comb_baseline_M4"]))
    bad += 0 if ok5 else 1

    print("=== 预注册 · fingerprints(只查漂移,不是真值)===")
    F = reg["fingerprints"]
    tol = reg["fingerprint_tol"]
    for k, want in sorted(F.items()):
        if k.startswith("dose_"):
            frac = float(k.split("_")[1]) / 100.0
            ref = k.endswith("refine1")
            got = float(np.mean([b1(*synth(frac, seed=s), refine=ref) for s in range(3)]))
        elif k.startswith("harm_x"):
            got = float(np.mean([b1(*synth(0.0, seed=s, harm_amp=float(k[6:])), refine=False) for s in range(3)]))
        else:
            continue
        d = abs(got - want)
        f_ok = d < tol
        print("   %-22s %+.6f  登记 %+.6f  Δ%.2e  %s" % (k, got, want, d, "OK" if f_ok else "!! 漂移"))
        bad += 0 if f_ok else 1
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    rc_self = 0 if selftest() else 1
    print()
    rc_reg = registry_check()
    rc = rc_reg if rc_reg == 3 else max(rc_self, rc_reg)
    print("\n=== b1 ruler: %s (exit %d) ===" % ("PASS" if rc == 0 else "FAIL", rc))
    sys.exit(rc)
