# -*- coding: utf-8 -*-
"""把若干条候选臂放到同一把(四把)尺子下,与同一条参照臂对拍。

    python compare.py --ref arm_raw.wav --shift 6 \
        --cand k0=arm_k0.wav --cand psola=arm_psola.wav \
        [--window 396924:458668,910279:972023] [--json out.json]

`--ref` = 逆变换的**输入**(模型在移调位唱出来的那一段);`--shift` = 音频要上移的半音数。
不给 `--window` ⇒ 全曲;给了 ⇒ 只量那些**样本区间**(通常 = 被救的那几条乐句窗)。

⛔ 选帧只由 `--ref` 的 f0 决定(g2p_rulers 第 5 条:取样规则里不许出现被测变量)。
   候选丢掉的浊音在 `浊帧存活` 一栏单独报,**不许被静默丢弃**。
⛔ 四把尺子必须一起读:任何一把单独拿来当判据都有已登记的盲区(见 rulers.py 顶部的表)。
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import rulers  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - 老解释器
    pass


def parse_window(s):
    if not s:
        return None
    spans = []
    for part in s.split(","):
        a, b = part.split(":")
        spans.append((int(a), int(b)))
    return spans


def measure(ref_path, cand_paths, shift_st, spans):
    """返回 {name: readings}。ref 只分析一次。"""
    ratio = 2.0 ** (shift_st / 12.0)
    xr, sr = rulers.load_mono(ref_path)
    ref_an = rulers.world_analyse(xr, sr)
    ref_flux = rulers.transient_flux(xr, sr)
    ref_hnr = rulers.hnr_median(xr, sr, spans)

    w_idx_all = rulers.samples_to_world_frames(spans, sr, ref_an["n"])
    idx = np.array([i for i in w_idx_all if ref_an["f0"][i] > 0], dtype=int)
    f_idx = rulers.flux_frames(spans, sr, len(ref_flux))

    out = {
        "_ref": {
            "path": str(ref_path), "sr": sr, "samples": len(xr),
            "world_frames": ref_an["n"], "voiced_frames_in_window": int(len(idx)),
            "ref_f0_median_hz": float(np.median(ref_an["f0"][idx])) if len(idx) else float("nan"),
            "ref_hnr_db": ref_hnr,
            "shift_st": shift_st,
        }
    }
    for name, path in cand_paths:
        xc, src = rulers.load_mono(path)
        if src != sr:
            raise SystemExit(f"UNRUNNABLE: 采样率不一致 {name} {src} != ref {sr}")
        if len(xc) != len(xr):
            raise SystemExit(
                f"UNRUNNABLE: 长度不一致 {name} {len(xc)} != ref {len(xr)}"
                " —— 逐样本对齐是这把尺子的前提(apply_inverse 保长契约)")
        an = rulers.world_analyse(xc, sr)
        env = rulers.envelope_shift(ref_an, an, idx)
        fx = rulers.flux_ratio_db(rulers.transient_flux(xc, sr), ref_flux, f_idx)
        hn = rulers.hnr_median(xc, sr, spans)
        f0e = rulers.f0_error_cents(ref_an, an, idx, ratio)
        out[name] = {
            "envelope_shift_st": env["median_st"],
            "envelope_p25": env["p25"], "envelope_p75": env["p75"],
            "peak_corr": env["peak_corr"], "zero_corr": env["zero_corr"],
            "voiced_survival": env["voiced_survival"],
            "flux_median_db": fx["median_db"], "flux_peak_db": fx["peak_db"],
            "hnr_db": hn, "hnr_delta_db": hn - ref_hnr,
            "f0_median_cents": f0e["median_cents"], "f0_p90_abs_cents": f0e["p90_abs_cents"],
            "f0_lost_voiced": f0e["lost_voiced"],
            "n_frames": env["n"],
        }
    return out


def render(res):
    r = res["_ref"]
    print(f"参照臂 {r['path']}")
    print(f"  {r['sr']} Hz · {r['samples']} 样本 · 窗内浊帧 {r['voiced_frames_in_window']}"
          f" · ref f0 中位 {r['ref_f0_median_hz']:.1f} Hz ⇒ 升 {r['shift_st']:+g} st 后"
          f" {r['ref_f0_median_hz'] * 2 ** (r['shift_st'] / 12.0):.1f} Hz"
          f" · ref HNR {r['ref_hnr_db']:.2f} dB")
    print()
    print(f"{'臂':<12}{'包络位移':>10}{'峰值相关':>10}{'浊帧存活':>10}"
          f"{'瞬态 dB':>10}{'ΔHNR dB':>10}{'f0 cents':>10}")
    print("-" * 72)
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:<12}{v['envelope_shift_st']:>+10.2f}{v['peak_corr']:>10.3f}"
              f"{v['voiced_survival']:>10.2%}{v['flux_median_db']:>+10.2f}"
              f"{v['hnr_delta_db']:>+10.2f}{v['f0_median_cents']:>+10.1f}")
    print()
    print("读法:包络位移越接近 0 = 共振峰越没被搬走(纯移调 = +shift);峰值相关低 = 包络形状变了")
    print("      (⛔ 包络被抹平时位移会读成漂亮的 0,必须两个一起看);浊帧存活 < 100% = 把音唱没了;")
    print("      ΔHNR 负得多 = 重合成脏(⭐ 这是分开 praat 与我们 2026-07 那份坏实现的那把尺子);")
    print("      f0 cents 只说明音高准 —— ⛔ 必要不充分,不许单独当判据。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", action="append", required=True,
                    help="name=path,可重复")
    ap.add_argument("--shift", type=float, required=True, help="音频上移的半音数")
    ap.add_argument("--window", default=None, help="a:b,a:b(样本);缺省 = 全曲")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cands = []
    for c in a.cand:
        if "=" not in c:
            raise SystemExit("--cand 要写成 name=path")
        n, p = c.split("=", 1)
        cands.append((n, p))
    res = measure(a.ref, cands, a.shift, parse_window(a.window))
    render(res)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
