# -*- coding: utf-8 -*-
"""gate0b_parselmouth_xenv — parselmouth 跨版本 f0 一次性交叉定审（S40，红队 A13）。

背景：gate0 双侧同 venv 只隔离代码轴;librosa/torch 轴有 S38/S39 既有逐位证据,
唯 parselmouth(to_pitch_ac,内嵌不同 Praat 内核)是本链全新依赖、项目内零跨版本
证据。本脚本在第二环境(praat-parselmouth 旧版)对同一组切片跑【原版 wav2F0 真码】,
与主 venv(0.4.7)结果按 S37 f0 定审口径比对(超 0.5Hz 帧数 / 清浊翻转数)。

两阶段(同一脚本,--dump 在任一环境产出 npz;主环境再 --compare):
  1) <second_venv_python> gate0b_parselmouth_xenv.py --dump out_old.npz
  2) <training venv python> gate0b_parselmouth_xenv.py --dump out_new.npz
  3) <任意 python+numpy>    gate0b_parselmouth_xenv.py --compare out_old.npz out_new.npz
"""
import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

ORIG = pathlib.Path(r"D:\MyDev\SingingVocoders")
SLICES = pathlib.Path(r"D:\MyDev\TESTING\smoke_vocoder\ws\slices")
HPARAMS = {"hop_size": 512, "audio_sample_rate": 44100, "f0_min": 65, "f0_max": 1100}


def dump(out_path):
    import importlib.util

    import parselmouth  # noqa: F401  (version report)
    import soundfile as sf

    # load the ORIGINAL wav2F0.py file directly (verbatim code) — a package
    # import would execute utils/__init__.py, which drags in the full
    # lightning stack this minimal second venv deliberately lacks
    spec = importlib.util.spec_from_file_location(
        "wav2F0_solo", str(ORIG / "utils" / "wav2F0.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    get_pitch = mod.get_pitch

    arrs = {}
    for wav in sorted(SLICES.glob("*.wav")):
        data, sr = sf.read(str(wav), dtype="float32")
        assert sr == 44100, wav
        length = (len(data) + HPARAMS["hop_size"] - 1) // HPARAMS["hop_size"]
        f0, uv = get_pitch("parselmouth", data, length=length, hparams=HPARAMS,
                           interp_uv=True)
        arrs[wav.stem + ".f0"] = f0
        arrs[wav.stem + ".uv"] = uv.astype(np.uint8)
    np.savez(out_path, **arrs)
    print(f"dumped {len(arrs)//2} slices with parselmouth {parselmouth.VERSION} "
          f"(Praat {parselmouth.PRAAT_VERSION}) -> {out_path}")


def compare(a_path, b_path):
    za, zb = np.load(a_path), np.load(b_path)
    assert set(za.files) == set(zb.files)
    stems = sorted({k[:-3] for k in za.files if k.endswith(".f0")})
    total = bad = flips = 0
    worst = 0.0
    for s in stems:
        fa, fb = za[s + ".f0"], zb[s + ".f0"]
        ua, ub = za[s + ".uv"], zb[s + ".uv"]
        voiced = (~ua.astype(bool)) & (~ub.astype(bool))
        d = np.abs(fa - fb)[voiced]
        total += int(voiced.sum())
        bad += int((d > 0.5).sum())
        flips += int((ua != ub).sum())
        if d.size:
            worst = max(worst, float(d.max()))
    print(f"voiced frames {total} | >0.5Hz: {bad} | uv flips: {flips} | max {worst:.4f}Hz")
    ok = bad == 0 and flips <= max(2, total // 1000)  # S37/S38 axis: ~0.1% edge frames
    print("=== gate0b_parselmouth_xenv:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.dump:
        dump(args.dump)
    elif args.compare:
        compare(*args.compare)
