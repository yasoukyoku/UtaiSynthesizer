# -*- coding: utf-8 -*-
"""工装自检 —— 「这四把尺子今天还是不是那四把尺子」。

退出码(⛔ 三者必须分得开 —— 一条闸的红必须能被归因,S129 铁律):
    0  ALL PASS
    1  **读数不符**:尺子跑起来了,但量出来的东西不对。⇒ 被测的东西变了,或者尺子坏了。
    3  **跑不起来**:缺 python 依赖 / 缺夹具 / 夹具指纹不符。
       ⛔ 3 不许被读成通过,也不许被读成「被测的东西不对」——
       换过夹具的读数与登记值**不构成比较**(g2p_rulers/README.md 第 8 条)。

两档:
  * **A 档 合成标定**(不需要夹具,永远能跑):用**已知答案**的合成信号量四把尺子。
    ⛔ 合成信号**只用来标定尺子本身**;合成周期信号系统性冤枉 PSOLA 类算法
       ([[project_v2_range_extend_quality]] §7-1,S81 被误导三次)⇒ 永远不许拿 A 档
       去判任何算法的好坏。
  * **B 档 真素材**(需要 S145 夹具):先复现 registry 的 predictions(真值,尺子错了才不过),
    再复现 fingerprints(指纹,查漂移)。

    python selfcheck.py [--no-fixtures]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

EXIT_OK, EXIT_READING, EXIT_UNRUNNABLE = 0, 1, 3

FAILS: list[str] = []
UNRUNNABLE: list[str] = []


def ok(name, got, lo, hi, unit=""):
    good = (got == got) and lo <= got <= hi  # NaN 不许通过(否定式写法,S140)
    print(f"  [{'PASS' if good else 'FAIL'}] {name:<46} {got:+8.3f}{unit}  期望 [{lo:+.3f}, {hi:+.3f}]")
    if not good:
        FAILS.append(f"{name}: 量到 {got:+.3f}{unit},期望 [{lo:+.3f}, {hi:+.3f}]")
    return good


# ------------------------------------------------------------------ A 档

def synth_vowel(sr, seconds, f0, formants, env_shift_st=0.0, seed=0):
    """合成一段元音样的周期信号:固定共振峰包络 × 谐波列。

    `env_shift_st` 把**包络**整体上移这么多半音(f0 不动)⇒ 尺子应当恰好读出这个数。
    """
    n = int(sr * seconds)
    t = np.arange(n) / sr
    s = 2.0 ** (env_shift_st / 12.0)
    y = np.zeros(n)
    rng = np.random.RandomState(seed)
    for k in range(1, int(sr / 2 / f0)):
        f = k * f0
        if f > 9000.0:
            break
        # 共振峰包络在 f/s 处取值 ⇒ 峰值出现在 f = s·F(即上移 env_shift_st 半音)
        e = 1e-4
        for fc, bw, amp in formants:
            e += amp / (1.0 + ((f / s - fc) / bw) ** 2)
        y += e * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return y / max(np.max(np.abs(y)), 1e-9) * 0.9


def tier_calibration(rulers, arms, reg):
    """口径闸 —— 「尺子还是不是同一把尺子」。

    ⛔ 与 A 档是两件事:变异测试实测,把包络尺的对数轴改窄后 A 档**零红**,而真素材上
       k0 从 +2.40 漂到 +3.15。⇒ 『还能量出正确答案』盖不住『口径被人动过』。
    ⚠ 这不是「常量等于常量」的空断言:登记值是**读数所属的那个口径**,动了旋钮就必须
       连着重登记 fingerprints,否则两个口径的数会被混着引用。
    """
    print("\n=== 口径闸:尺子的旋钮还是不是登记的那一套 ===")
    mods = {"rulers": rulers, "arms": arms}
    bad = 0
    for key, want in reg["calibration"].items():
        if key.startswith("_"):
            continue
        mod, attr = key.split(".", 1)
        got = getattr(mods[mod], attr, None)
        if got is None:
            UNRUNNABLE.append(f"口径闸:{key} 在模块里不存在(改名了?)")
            continue
        good = float(got) == float(want)
        bad += 0 if good else 1
        if not good:
            FAILS.append(f"口径 {key}: 现在 {got},登记 {want}")
            print(f"  [FAIL] {key:<28} 现在 {got}  登记 {want}")
    print(f"  [{'PASS' if bad == 0 else 'FAIL'}] {len(reg['calibration']) - 1} 个旋钮"
          f"{'全部与登记一致' if bad == 0 else f',{bad} 个被动过'}")


def tier_a(rulers):
    print("\n=== A 档:合成标定(⛔ 只标定尺子,不判算法)===")
    sr, sec, f0 = 44100, 2.0, 220.0
    F = [(700.0, 90.0, 1.0), (1200.0, 110.0, 0.6), (2600.0, 160.0, 0.35), (3800.0, 220.0, 0.2)]
    a = synth_vowel(sr, sec, f0, F, 0.0)
    b = synth_vowel(sr, sec, f0, F, 3.0)  # 只把包络上移 3 半音
    an_a, an_b = rulers.world_analyse(a, sr), rulers.world_analyse(b, sr)
    idx = np.array([i for i in range(min(an_a["n"], an_b["n"])) if an_a["f0"][i] > 0], dtype=int)
    if len(idx) < 50:
        UNRUNNABLE.append(f"A 档:合成信号只拿到 {len(idx)} 个浊帧,标定不构成判定")
        return

    print(f"  (合成 {sec:g}s @ f0 {f0:g} Hz,浊帧 {len(idx)})")
    ok("① 包络尺 · 阴性 同一信号自比", rulers.envelope_shift(an_a, an_a, idx)["median_st"], -0.05, 0.05, " st")
    r = rulers.envelope_shift(an_a, an_b, idx)
    ok("① 包络尺 · 阳性 包络上移 3.00 st", r["median_st"], 2.5, 3.5, " st")
    ok("① 包络尺 · 阳性的峰值相关", r["peak_corr"], 0.90, 1.001, "")

    fa = rulers.transient_flux(a, sr)
    fi = np.arange(len(fa))
    ok("② 瞬态尺 · 阴性 同一信号自比", rulers.flux_ratio_db(fa, fa, fi)["median_db"], -0.01, 0.01, " dB")
    click = a.copy()
    for p in range(int(0.1 * sr), len(click), int(0.2 * sr)):  # 每 200 ms 一个脉冲
        click[p:p + 8] += 0.9
    fc = rulers.transient_flux(click, sr)
    ok("② 瞬态尺 · 阳性 插入脉冲串", rulers.flux_ratio_db(fc, fa, fi)["median_db"], 0.5, 1000.0, " dB")
    ok("② 瞬态尺 · 阳性 抹平脉冲串", rulers.flux_ratio_db(
        rulers.transient_flux(rulers.smear(click, sr), sr), fc, fi)["median_db"], -1000.0, -20.0, " dB")

    h0 = rulers.hnr_median(a, sr)
    rng = np.random.RandomState(146)
    d20 = rulers.hnr_median(a + rng.randn(len(a)) * np.sqrt(np.mean(a ** 2) / 100.0), sr) - h0
    d10 = rulers.hnr_median(a + rng.randn(len(a)) * np.sqrt(np.mean(a ** 2) / 10.0), sr) - h0
    ok("③ HNR 尺 · 阴性 同一信号自比", rulers.hnr_median(a, sr) - h0, -0.01, 0.01, " dB")
    ok("③ HNR 尺 · 阳性 注入 20 dB SNR 噪", d20, -1000.0, -0.5, " dB")
    ok("③ HNR 尺 · 阳性 注入 10 dB SNR 噪", d10, -1000.0, -0.5, " dB")
    ok("③ HNR 尺 · 阳性 单调(10dB 掉得更多)", d10 - d20, -1000.0, -0.1, " dB")

    c = synth_vowel(sr, sec, f0 * 2 ** (6 / 12.0), F, 0.0)
    an_c = rulers.world_analyse(c, sr)
    ok("④ f0 尺 · 阴性 未移调却期望 +6 st", rulers.f0_error_cents(an_a, an_a, idx, 2 ** 0.5)["median_cents"],
       -620.0, -580.0, " ¢")
    ok("④ f0 尺 · 阳性 真的移了 +6 st", rulers.f0_error_cents(an_a, an_c, idx, 2 ** 0.5)["median_cents"],
       -25.0, 25.0, " ¢")


# ------------------------------------------------------------------ B 档

def tier_b(rulers, reg):
    print("\n=== B 档:S145 真素材(预注册真值 + 指纹)===")
    d = pathlib.Path(os.environ.get(reg["fixtures"]["dir_env"], reg["fixtures"]["dir_default"]))
    if not d.is_dir():
        UNRUNNABLE.append(f"夹具目录不存在:{d}(设 {reg['fixtures']['dir_env']} 指过去)")
        return
    for name, want in reg["fixtures"]["sha256"].items():
        p = d / name
        if not p.is_file():
            UNRUNNABLE.append(f"缺夹具 {p}")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            UNRUNNABLE.append(f"夹具指纹不符 {name}\n      登记 {want}\n      实际 {got}")
    if UNRUNNABLE:
        return

    import compare  # 同一份 measure(),不许在这里再写一遍
    spans = [tuple(s) for s in reg["window"]["spans"]]
    shift = reg["window"]["shift_st"]
    arms = [(n.replace(".wav", ""), str(d / n)) for n in reg["fingerprints"]["arms"]]
    res = compare.measure(str(d / "arm_raw.wav"), arms, shift, spans)
    print(f"  (窗内浊帧 {res['_ref']['voiced_frames_in_window']},ref f0 中位 "
          f"{res['_ref']['ref_f0_median_hz']:.1f} Hz,ref HNR {res['_ref']['ref_hnr_db']:.2f} dB)")

    x, sr = rulers.load_mono(str(d / "arm_raw.wav"))

    print("\n  -- 预注册真值(尺子错了才会不过)--")
    ok("env-neg  raw→raw 包络位移", res["arm_raw"]["envelope_shift_st"], -0.05, 0.05, " st")
    ok("env-neg  raw→raw 峰值相关", res["arm_raw"]["peak_corr"], 0.999, 1.001, "")
    ok("env-pos  κ=1 = 纯移调(真值 +6.00)", res["arm_k1"]["envelope_shift_st"], 5.5, 6.5, " st")
    ok("flux-neg raw→raw", res["arm_raw"]["flux_median_db"], -0.01, 0.01, " dB")
    fr = rulers.transient_flux(x, sr)
    fi = rulers.flux_frames(spans, sr, len(fr))
    ok(f"flux-pos 人为抹平 {rulers.SMEAR_SECONDS * 1000:.0f} ms",
       rulers.flux_ratio_db(rulers.transient_flux(rulers.smear(x, sr), sr), fr, fi)["median_db"],
       -1000.0, -30.0, " dB")
    ok("hnr-neg  raw→raw", res["arm_raw"]["hnr_delta_db"], -0.01, 0.01, " dB")
    h0 = rulers.hnr_median(x, sr, spans)
    rng = np.random.RandomState(145)
    p = np.mean(x ** 2)
    d20 = rulers.hnr_median(x + rng.randn(len(x)) * np.sqrt(p / 100.0), sr, spans) - h0
    d10 = rulers.hnr_median(x + rng.randn(len(x)) * np.sqrt(p / 10.0), sr, spans) - h0
    ok("hnr-pos  注入 20 dB SNR 白噪", d20, -6.0, -1.5, " dB")
    ok("hnr-pos  注入 10 dB SNR 白噪", d10, -16.0, -6.0, " dB")
    ok("hnr-pos  单调(10dB 掉得更多)", d10 - d20, -1000.0, -0.1, " dB")
    ok("f0-neg   raw 未移调而期望 +6 st", res["arm_raw"]["f0_median_cents"], -620.0, -580.0, " ¢")

    print("\n  -- 指纹(不是真值;只查夹具/尺子有没有从 2026-08-14 漂开)--")
    tol = reg["fingerprints"]["tolerance"]
    for arm, want in reg["fingerprints"]["arms"].items():
        k = arm.replace(".wav", "")
        got = res[k]
        for field, t in (("envelope_shift_st", tol["envelope_st"]), ("peak_corr", tol["peak_corr"]),
                         ("flux_median_db", tol["flux_db"]), ("hnr_delta_db", tol["hnr_db"])):
            ok(f"{k:<10} {field}", got[field], want[field] - t, want[field] + t, "")


def tier_envmod(reg):
    """⑤ envmod —— 「油/晃」那条线的尺子(S159zy 立,S159zz 定价)。

    ⛔ 三件分开报(S129:一条闸的红必须能被归因):
    ⑴ **口径闸** —— 尺子的带宽还是不是登记的那一套。这里它**不是空断言**:同一批音在
       300-800 Hz 上读 −0.47 dB(没动),在 2000-4000 Hz 上读 +5.13 dB(最重)⇒
       带一改结论就换一个,而读数照样漂亮。
    ⑵ **预注册真值** —— 用户耳判背书过的那一对必须被分开,而且方向要对。
    ⑶ **指纹** —— 只回答有没有从 2026-08-23 漂开。
    """
    print("\n=== envmod 档:「油/晃」那条线的尺子 ===")
    try:
        import envmod as EM
    except Exception as e:
        UNRUNNABLE.append(f"envmod 导不进来:{e!r}")
        return
    reg_e = reg.get("envmod")
    if not reg_e:
        UNRUNNABLE.append("registry.json 里没有 envmod 段")
        return

    # ⑴ 口径闸 —— 登记值由判据自己那条路径算出来(README 第 5 条)
    got, want = EM.caliber(), reg_e["caliber"]
    for k, v in want.items():
        if k.startswith("_"):
            continue
        if k not in got:
            FAILS.append(f"envmod 口径缺 {k}")
        elif got[k] != v:
            FAILS.append(f"envmod 口径 {k} 被动过:登记 {v},实际 {got[k]}")
    for k in got:
        if k not in want:
            FAILS.append(f"envmod 多了一个没登记的口径 {k}")
    print(f"  [{'PASS' if not FAILS else 'FAIL'}] 口径闸({len(got)} 个)")

    d = pathlib.Path(os.environ.get(reg_e["fixtures"]["dir_env"],
                                    reg_e["fixtures"]["dir_default"]))
    if not d.is_dir():
        UNRUNNABLE.append(f"envmod 夹具目录不存在:{d}")
        return
    for name, w in reg_e["fixtures"]["sha256"].items():
        p = d / name
        if not p.is_file():
            UNRUNNABLE.append(f"缺 envmod 夹具 {p}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != w:
            UNRUNNABLE.append(f"envmod 夹具指纹不符 {name}")
    if UNRUNNABLE:
        return

    sp = EM.note_spans(str(d / reg_e["fixtures"]["notes_json"]))
    seg = {k: (lo, hi) for k, _n, lo, hi in sp}
    pr = reg_e["predictions"]
    import soundfile as sf
    read = {}
    for arm in ("lb3", "lb5"):
        x, _sr = sf.read(str(d / f"s159zu\\mg_deadonly_{arm}_dxl41.wav"))
        x = x.mean(axis=1) if x.ndim > 1 else x
        read[arm] = EM.score(x, sp)
        for k in (1035, 971):
            key = f"note_{k}_{arm}"
            if key in pr:
                lo, hi = seg[k]
                ok(f"envmod {arm} 音[{k}]", EM.modulation_db(x[lo:hi]),
                   pr[key] - 0.10, pr[key] + 0.10, " dB")

    # ⑵ 承重的是「分得开」,不是那个具体的数
    ok("envmod 用户那一对必须分得开", abs(read["lb5"][1035] - read["lb3"][1035]),
       pr["min_separation_db"], 1e3, " dB")
    # ⛔ 阴性对照:从没被救过的音,两版之间不许出现同量级的差
    ok("envmod 阴性对照 音[971] 两版之差", abs(read["lb5"][971] - read["lb3"][971]),
       -1e3, pr["min_separation_db"] / 2.0, " dB")

    # ⑶ 指纹
    fp = reg_e["fingerprints"]
    ok("envmod 打分音数", float(len(read["lb3"])),
       fp["n_notes_ge_0p2s"] - 0.5, fp["n_notes_ge_0p2s"] + 0.5, "")
    for arm in ("lb3", "lb5"):
        ok(f"envmod {arm} 全曲中位", float(np.median(list(read[arm].values()))),
           fp[f"{arm}_median_db"] - 0.05, fp[f"{arm}_median_db"] + 0.05, " dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fixtures", action="store_true", help="只跑 A 档")
    a = ap.parse_args()

    try:
        import rulers  # noqa: F401
        import pyworld  # noqa: F401
        import parselmouth  # noqa: F401
        import soundfile  # noqa: F401
    except Exception as e:  # 缺依赖 = 跑不起来,不是读数不符
        print(f"UNRUNNABLE: 缺 python 依赖 —— {e!r}")
        print("解释器:用 training/.venv 那个(pyworld / praat-parselmouth / soundfile / scipy 都在里面)")
        return EXIT_UNRUNNABLE
    import rulers as R
    import arms as A

    reg = json.loads((HERE / "registry.json").read_text(encoding="utf-8"))
    print(f"range_rulers 自检 · 夹具口径 {reg['_provenance']}")
    tier_calibration(R, A, reg)
    tier_a(R)
    if not a.no_fixtures:
        tier_b(R, reg)
        tier_envmod(reg)
    else:
        print("\n=== B 档:按 --no-fixtures 跳过 ===")
        print("  ⚠ 只跑 A 档 = 只证明了尺子自洽,**没有**证明它还能复现真素材上的读数。")

    print("\n" + "=" * 74)
    if UNRUNNABLE:
        print("EXIT=3 跑不起来(⛔ 这不是通过,也不是『被测的东西不对』):")
        for m in UNRUNNABLE:
            print("  - " + m)
        return EXIT_UNRUNNABLE
    if FAILS:
        print(f"EXIT=1 读数不符 —— {len(FAILS)} 条:")
        for m in FAILS:
            print("  - " + m)
        return EXIT_READING
    print("EXIT=0 ALL PASS")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
