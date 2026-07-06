"""gate0_diff step 4 — compare the --use_diff products, C layer (same 44k
input, both sides fp32 CPU, same aug seed/draw order).

Per slice:
  .vol.npy      expect BIT-EXACT (Volume_Extractor was bit-exact torch 2.0<->2.5
                in the S38 C5 gate)
  .mel.npy      nsf_hifigan-recipe mel — axis: torch 2.0 vs 2.5 stft/matmul
                (librosa mel filterbank 0.9.1 vs 0.11 measured bit-identical);
                PASS line: max_abs <= 1e-4 in ln-mel domain
  .aug_mel.npy  keyshift must match EXACTLY (same random draw — proves the RNG
                alignment). PASS = loud bins (ln-mel > -10) within 1e-5 and
                near-floor bins within 5e-4: the keyshift path runs a
                non-power-of-two FFT (n_fft*2^(k/12)) where the torch 2.0<->2.5
                kernel axis leaves slightly more fp noise, and ln amplifies it
                near the 1e-5 clamp floor (the S36-documented effect; measured
                S39: 12/21632 entries >1e-5, ALL at ln values < -10.1,
                linear-domain rel 1.1e-4) — a real code difference would show
                on loud bins, which must stay at fp-noise level
  .aug_vol.npy  same loudness shift -> expect bit-exact-or-fp-noise (<=1e-6)
Also verifies the (aug_mel, aug_vol) PAIR consistency file-by-file on our side
(both present — the pair-atomic rule).

Run (our venv):
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate0_diff_compare.py
"""
import os

import numpy as np

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
ORIG = os.path.join(TESTING, "diff_orig", "gate")
OURS = os.path.join(TESTING, "diff_ours", "gate")

MEL_LINE = 1e-4
VOL_LINE = 0.0
AUG_VOL_LINE = 1e-6
AUG_MEL_LOUD_LINE = 1e-5   # bins with ln-mel > NEAR_FLOOR (the meaningful ones)
AUG_MEL_FLOOR_LINE = 5e-4  # near-clamp bins (ln amplification, see header)
NEAR_FLOOR = -10.0


def main():
    wavs = sorted(n for n in os.listdir(ORIG) if n.endswith(".wav"))
    assert wavs, "orig side empty — run gate0_diff_orig.py first"
    stats = {"vol": 0.0, "mel": 0.0, "aug_mel": 0.0, "aug_vol": 0.0}
    keyshift_mismatch = 0
    pair_missing = 0
    for n in wavs:
        vol_a = np.load(os.path.join(ORIG, n + ".vol.npy"))
        vol_b = np.load(os.path.join(OURS, n + ".vol.npy"))
        stats["vol"] = max(stats["vol"], float(np.abs(vol_a - vol_b).max()))

        mel_a = np.load(os.path.join(ORIG, n + ".mel.npy"))
        mel_b = np.load(os.path.join(OURS, n + ".mel.npy"))
        assert mel_a.shape == mel_b.shape, (n, mel_a.shape, mel_b.shape)
        stats["mel"] = max(stats["mel"], float(np.abs(mel_a - mel_b).max()))

        am_a, ks_a = np.load(os.path.join(ORIG, n + ".aug_mel.npy"), allow_pickle=True)
        am_b, ks_b = np.load(os.path.join(OURS, n + ".aug_mel.npy"), allow_pickle=True)
        if float(ks_a) != float(ks_b):
            keyshift_mismatch += 1
            print(f"KEYSHIFT MISMATCH {n}: orig={ks_a} ours={ks_b}")
        am_a = np.array(am_a, dtype=np.float64)
        am_b = np.array(am_b, dtype=np.float64)
        assert am_a.shape == am_b.shape, (n, am_a.shape, am_b.shape)
        d = np.abs(am_a - am_b)
        stats["aug_mel"] = max(stats["aug_mel"], float(d.max()))
        loud = am_a > NEAR_FLOOR
        if loud.any():
            stats["aug_mel_loud"] = max(
                stats.get("aug_mel_loud", 0.0), float(d[loud].max())
            )

        av_a = np.load(os.path.join(ORIG, n + ".aug_vol.npy"))
        av_b = np.load(os.path.join(OURS, n + ".aug_vol.npy"))
        stats["aug_vol"] = max(stats["aug_vol"], float(np.abs(av_a - av_b).max()))

        # pair-atomic rule sanity on our side
        if not (
            os.path.exists(os.path.join(OURS, n + ".aug_mel.npy"))
            and os.path.exists(os.path.join(OURS, n + ".aug_vol.npy"))
        ):
            pair_missing += 1

    print(f"files: {len(wavs)}")
    for k, v in stats.items():
        print(f"max_abs {k}: {v:.3e}")
    print(f"keyshift mismatches: {keyshift_mismatch}")
    print(f"aug pair missing: {pair_missing}")

    ok = (
        keyshift_mismatch == 0
        and pair_missing == 0
        and stats["vol"] <= VOL_LINE
        and stats["mel"] <= MEL_LINE
        and stats.get("aug_mel_loud", 0.0) <= AUG_MEL_LOUD_LINE
        and stats["aug_mel"] <= AUG_MEL_FLOOR_LINE
        and stats["aug_vol"] <= AUG_VOL_LINE
    )
    print("GATE0 DIFF:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
