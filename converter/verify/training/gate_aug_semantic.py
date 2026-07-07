# -*- coding: utf-8 -*-
"""S41 gate_aug_semantic — semantic verification of the PSOLA augmentation
engine + quality gate (design doc B5; run inside training/.venv, cwd=training).

    ..\\..\\..\\training\\.venv\\Scripts\\python.exe ^
        ..\\converter\\verify\\training\\gate_aug_semantic.py   (from training/)

Parts (all must PASS):
  1  clean-source semantics: production-parity slices (sovits slicer chain) ->
     augment_slices copies=2 -> independent rmvpe recheck of every aug wav
     (median<=30 / p90<=100 cents vs source*2^(k/12)), duration conservation,
     cross-run bitwise determinism
  2  formant preservation + metric anchor: synthetic vowel with KNOWN formants
     -> PSOLA +/-3 must measure |shift|<=0.5 st; a resample-shifted twin must
     measure ~= its true shift (proves the metric itself can see shifts —
     anti-self-certification, red-team V11)
  3  real dirty material, two failure arms (red-team V8):
     human (OpenUtau render, PSOLA p90 arm) and kazane dataset1@30s (median
     arm) — sliced with the production slicer, gate must reject >= 1 slice
     per arm; per-slice distributions archived to gate_aug_semantic_dist.json
  4  parselmouth x rmvpe cross-check on the dirty rejects (red-team V9) —
     VERDICT 2026-07-07: praat is BLIND to the PSOLA glitch tail (measured:
     rmvpe p90 323/245 cents where praat reads 12/17 with the SAME voiced
     coverage — the continuity prior smooths right over the glitches; not an
     unvoiced-dropout effect). Design consequence: the production quality
     gate is rmvpe-blooded on ALL backends (the vocoder chain must NOT gate
     on its own parselmouth npz f0). This part now asserts the blind spot
     stays on record: rmvpe rejects >=1 copy per arm that praat misses — if
     a future parselmouth starts seeing them, this trips and we re-evaluate.
  5  gate unit semantics via run_f0_gate with stub loaders: voiced-ratio
     rejection (noise source), mask polarity (all-unvoiced -> reject), and
     high-pitch headroom exemption (sweep source: kept, headroom counted)
"""
import json
import os
import shutil
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(APP, "training"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from utai_train.augment import (  # noqa: E402
    GATE_MEDIAN_CENTS,
    GATE_P90_CENTS,
    augment_slices,
    draw_keyshift,
    list_aug_entries,
    psola_shift,
    run_f0_gate,
)
from utai_train.sovits.f0.RMVPEF0Predictor import RMVPEF0Predictor  # noqa: E402

SR = 44100
HOP = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RMVPE_PT = os.path.join(APP, "data", "models", "training", "sovits", "rmvpe.pt")
CLEAN_SRC = r"D:\MyDev\TESTING\Kazano_Sayo\dataset\20260704-004726.mp3"
DIRTY_HUMAN = r"D:\MyDev\TESTING\utai-v2-testing\aug_engine_ab\human_00_orig.wav"
DIRTY_KAZANE = r"D:\MyDev\TESTING\Kazano_Sayo\dataset\dataset1.wav"
OUT_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_aug_semantic_dist.json")

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  [%s] %s %s" % (tag, name, detail))
    if not ok:
        FAILURES.append(name)


class StubReporter:
    def stage(self, *a, **k):
        pass


class StubStop:
    def check(self):
        pass


predictor = RMVPEF0Predictor(
    hop_length=HOP, sampling_rate=SR, dtype=torch.float32, device=DEV,
    threshold=0.05, model_path=RMVPE_PT,
)


def rmvpe(x):
    f0, uv = predictor.compute_f0_uv(x)
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1)
    n = min(len(f0), len(uv))
    return f0[:n], uv[:n] > 0.5


def praat_f0(x):
    """Independent Praat-lineage detector (vocoder-chain blood line, V9)."""
    import parselmouth

    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch_ac(time_step=HOP / SR, pitch_floor=65, pitch_ceiling=1100)
    f0 = pitch.selected_array["frequency"].astype(np.float64)
    return f0, f0 > 0


def cents_stats(f0_s, v_s, f0_a, v_a, keyshift):
    n = min(len(f0_s), len(f0_a))
    mask = v_s[:n] & v_a[:n]
    target = f0_s[:n] * (2.0 ** (keyshift / 12.0))
    mask = mask & (target <= 1100.0)
    if int(mask.sum()) < 10:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(f0_a[:n][mask] / target[mask])
    cents = np.abs(cents[np.isfinite(cents)])
    if len(cents) < 10:
        return None
    return float(np.median(cents)), float(np.percentile(cents, 90))


def read_wav(path):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def write_int16(tmp_path, samples, sr):
    from scipy.io import wavfile

    pcm = (np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(tmp_path, sr, pcm)


def production_slices(src_path, out_dir, offset=None, duration=None):
    """Slice with the PRODUCTION chain (sovits slicer + resample math)."""
    import librosa

    from utai_train.rvc.slicer2 import Slicer
    from utai_train.sovits.preprocess import _resample_chain

    os.makedirs(out_dir, exist_ok=True)
    wav, sr = librosa.load(src_path, sr=None, mono=True, offset=offset or 0.0,
                           duration=duration)
    slicer = Slicer(sr=sr)
    n = 0
    for idx, chunk in enumerate(slicer.slice(wav)):
        out = _resample_chain(chunk, sr, False)
        if out is None:
            continue
        from scipy.io import wavfile

        wavfile.write(os.path.join(out_dir, "000_%03d.wav" % idx), SR, out)
        n += 1
    return n


def remove_products(slice_dir, meta_dir, stem):
    for name in os.listdir(slice_dir):
        if name.split(".")[0] == stem:
            os.remove(os.path.join(slice_dir, name))
    mp = os.path.join(meta_dir, stem + ".json")
    if os.path.exists(mp):
        os.remove(mp)


def run_aug(slice_dir, meta_dir, copies, seed=1234):
    return augment_slices(
        slice_dir, copies, seed, meta_dir, read_wav, write_int16,
        lambda stem: remove_products(slice_dir, meta_dir, stem),
        StubReporter(), StubStop(),
    )


# ---------------------------------------------------------------- part 1
def part1():
    print("[1] clean-source semantics (production-parity slices)")
    root = tempfile.mkdtemp(prefix="gate_aug1_")
    try:
        sdir = os.path.join(root, "slices")
        mdir = os.path.join(root, "aug_meta")
        n = production_slices(CLEAN_SRC, sdir, offset=60.0, duration=45.0)
        check("clean slices produced", n >= 3, "(%d)" % n)
        gen, reused, _ = run_aug(sdir, mdir, copies=2)
        check("aug generated", gen == 2 * n and reused == 0, "(%d gen)" % gen)

        meds, p90s = [], []
        f0_cache = {}
        for src_stem, aug_stem, k in list_aug_entries(sdir, mdir):
            if src_stem not in f0_cache:
                x, _ = read_wav(os.path.join(sdir, src_stem + ".wav"))
                f0_cache[src_stem] = (rmvpe(x), len(x))
            (f0_s, v_s), src_len = f0_cache[src_stem]
            y, _ = read_wav(os.path.join(sdir, aug_stem + ".wav"))
            check("duration conserved %s" % aug_stem, abs(len(y) - src_len) <= HOP)
            st = cents_stats(f0_s, v_s, *rmvpe(y), k)
            if st is None:
                continue  # low-voicing slice; gate would reject, not a semantics fail
            meds.append(st[0])
            p90s.append(st[1])
        check("f0 target median", meds and max(meds) <= GATE_MEDIAN_CENTS,
              "(worst %.1f cents over %d slices)" % (max(meds), len(meds)))
        check("f0 target p90", p90s and max(p90s) <= GATE_P90_CENTS,
              "(worst %.1f cents)" % max(p90s))

        # cross-run bitwise determinism: regenerate into a fresh dir
        sdir2 = os.path.join(root, "slices2")
        mdir2 = os.path.join(root, "aug_meta2")
        os.makedirs(sdir2)
        for f in os.listdir(sdir):
            if f.endswith(".wav") and "_aug" not in f:
                shutil.copy2(os.path.join(sdir, f), os.path.join(sdir2, f))
        run_aug(sdir2, mdir2, copies=2)
        same = all(
            open(os.path.join(sdir, f), "rb").read()
            == open(os.path.join(sdir2, f), "rb").read()
            for f in os.listdir(sdir)
            if "_aug" in f and f.endswith(".wav")
        )
        check("cross-run bitwise determinism", same)

        # 审查修复 PY-1: sub-0.3s sources are skipped (praat's Manipulation
        # throws below ~50ms; <0.3s aug could never enter train.txt anyway)
        from scipy.io import wavfile as _wf

        short = os.path.join(sdir2, "999_000.wav")
        _wf.write(short, 44100, (np.zeros(int(0.2 * 44100)) + 0.1).astype(np.float32))
        gen2, _, _ = run_aug(sdir2, mdir2, copies=2)
        check("short source skipped (no aug, no crash)",
              gen2 == 0 and not os.path.exists(os.path.join(sdir2, "999_000_aug1.wav")))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- part 2
def synth_vowel(f0_hz=180.0, dur=3.0, formants=((700, 80), (1200, 90), (2600, 120))):
    """Impulse train through parallel resonators — a vowel with KNOWN formants."""
    from scipy.signal import lfilter

    n = int(dur * SR)
    src = np.zeros(n, dtype=np.float64)
    period = int(round(SR / f0_hz))
    src[::period] = 1.0
    out = np.zeros(n)
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / SR)
        theta = 2 * np.pi * fc / SR
        a = [1.0, -2 * r * np.cos(theta), r * r]
        out += lfilter([1.0], a, src)
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.6
    return out.astype(np.float32)


def formant_shift_estimate(x_ref, y):
    """cheaptrick envelope + warp search (the S41 selection-phase metric)."""
    import pyworld

    def env(x):
        f0, v = rmvpe(x)
        t = np.arange(len(f0)) * (HOP / SR)
        f0w = np.where(v, f0, 0.0)
        sp = pyworld.cheaptrick(x.astype(np.float64), f0w, t, SR)
        return np.log(np.maximum(sp, 1e-12)), v

    e_ref, v_ref = env(x_ref)
    e_y, v_y = env(y)
    n = min(e_ref.shape[0], e_y.shape[0])
    mask = v_ref[:n] & v_y[:n]
    bins = e_ref.shape[1]
    freqs = np.linspace(0, SR / 2, bins)
    sel = (freqs >= 300) & (freqs <= 9000)
    best = None
    for st in np.arange(-6, 6.01, 0.25):
        w = 2.0 ** (st / 12.0)
        ref_i = np.array([np.interp(freqs[sel] / w, freqs, row) for row in e_ref[:n][mask]])
        a = ref_i - ref_i.mean(axis=1, keepdims=True)
        b = e_y[:n][mask][:, sel]
        b = b - b.mean(axis=1, keepdims=True)
        c = (a * b).sum(axis=1) / (
            np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-12
        )
        med = float(np.median(c))
        if best is None or med > best[1]:
            best = (float(st), med)
    return best[0]


def part2():
    print("[2] formant preservation + metric anchor (synthetic vowel)")
    vowel = synth_vowel()
    for k in (3.0, -3.0):
        y = psola_shift(vowel, SR, k)
        est = formant_shift_estimate(vowel, y)
        check("PSOLA %+.0fst formant shift ~0" % k, abs(est) <= 0.5, "(est %+.2f st)" % est)
    # anchor: a resample-shift REALLY moves formants; the metric must see it
    import librosa

    k_true = 3.0
    factor = 2.0 ** (k_true / 12.0)
    y_rs = librosa.resample(vowel, orig_sr=SR, target_sr=int(round(SR / factor)))
    est = formant_shift_estimate(vowel, y_rs.astype(np.float32))
    check("metric anchor (resample +3st)", abs(est - k_true) <= 0.5, "(est %+.2f st)" % est)


# ---------------------------------------------------------------- parts 3+4
def dirty_arm(tag, src_path, offset, duration, dist):
    root = tempfile.mkdtemp(prefix="gate_aug3_")
    try:
        sdir = os.path.join(root, "slices")
        mdir = os.path.join(root, "aug_meta")
        n = production_slices(src_path, sdir, offset=offset, duration=duration)
        if n == 0:
            check("%s produced slices" % tag, False)
            return
        run_aug(sdir, mdir, copies=1)
        rejects = []
        praat_agree = 0
        for src_stem, aug_stem, k in list_aug_entries(sdir, mdir):
            x, _ = read_wav(os.path.join(sdir, src_stem + ".wav"))
            y, _ = read_wav(os.path.join(sdir, aug_stem + ".wav"))
            f0_s, v_s = rmvpe(x)
            if float(np.mean(v_s)) < 0.30:
                rejects.append((aug_stem, "lowvoiced", None, None))
                continue
            st = cents_stats(f0_s, v_s, *rmvpe(y), k)
            dist.setdefault(tag, []).append(
                {"slice": aug_stem, "k": round(k, 3),
                 "median": None if st is None else round(st[0], 2),
                 "p90": None if st is None else round(st[1], 2)}
            )
            bad = st is None or st[0] > GATE_MEDIAN_CENTS or st[1] > GATE_P90_CENTS
            if bad:
                rejects.append((aug_stem, "cents", x, (y, k)))
        check("%s: gate rejects >=1" % tag, len(rejects) >= 1, "(%d/%d)" % (len(rejects), n))
        # V9 blind-spot record: praat must MISS at least one copy rmvpe caught
        # (measured property of these fixed materials/seeds; a future
        # parselmouth that starts seeing them should trip this for re-eval)
        crossed = [r for r in rejects if r[1] == "cents"]
        for _, _, x, (y, k) in crossed:
            ps = cents_stats(*praat_f0(x), *praat_f0(y), k)
            if ps is None or ps[0] > GATE_MEDIAN_CENTS or ps[1] > GATE_P90_CENTS:
                praat_agree += 1
        if crossed:
            check("%s: praat blind spot on record (misses >=1)" % tag,
                  praat_agree < len(crossed), "(praat flagged %d/%d)" % (praat_agree, len(crossed)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- part 5
def part5():
    print("[5] gate unit semantics (run_f0_gate with stub loaders)")
    frames = 400

    def gate_with(f0bank, entries):
        removed = []
        run_f0_gate(entries, lambda s: f0bank.get(s),
                    lambda s: removed.append(s), StubReporter(), StubStop())
        return removed

    rng = np.random.RandomState(0)
    # noise source: voiced ratio ~0 -> reject
    bank = {
        "noise": (rng.uniform(80, 90, frames), np.zeros(frames, dtype=bool)),
        "noise_aug1": (rng.uniform(80, 90, frames), np.zeros(frames, dtype=bool)),
    }
    removed = gate_with(bank, [("noise", "noise_aug1", 2.0)])
    check("all-unvoiced source rejected", removed == ["noise_aug1"])

    # good pair: 200Hz source, perfect +2st copy -> kept
    f0 = np.full(frames, 200.0)
    v = np.ones(frames, dtype=bool)
    bank = {"good": (f0, v), "good_aug1": (f0 * 2 ** (2 / 12), v)}
    removed = gate_with(bank, [("good", "good_aug1", 2.0)])
    check("faithful copy kept", removed == [])

    # broken pair: copy stuck at source pitch (the PSOLA no-shift failure) -> reject
    bank = {"bad": (f0, v), "bad_aug1": (f0.copy(), v)}
    removed = gate_with(bank, [("bad", "bad_aug1", 3.0)])
    check("no-shift failure rejected", removed == ["bad_aug1"])

    # headroom: sweep 700->1000Hz +3st -> frames above 1100 exempted, rest verify -> kept
    f0_sweep = np.linspace(700, 1000, frames)
    bank = {
        "hi": (f0_sweep, v),
        "hi_aug1": (f0_sweep * 2 ** (3 / 12), v),
    }
    removed = gate_with(bank, [("hi", "hi_aug1", 3.0)])
    check("high-pitch headroom kept", removed == [])


def main():
    print("gate_aug_semantic (device=%s)" % DEV)
    dist = {}
    part1()
    part2()
    print("[3/4] dirty-material arms + parselmouth cross-check")
    dirty_arm("human(OpenUtau)", DIRTY_HUMAN, None, None, dist)
    dirty_arm("kazane@30s", DIRTY_KAZANE, 30.0, 15.0, dist)
    part5()
    with open(OUT_DIST, "w", encoding="utf-8") as f:
        json.dump(dist, f, ensure_ascii=False, indent=1)
    print("per-slice distributions -> %s" % OUT_DIST)
    if FAILURES:
        print("RESULT: FAIL (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
