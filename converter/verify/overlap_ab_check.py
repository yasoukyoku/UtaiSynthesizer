# S32 htdemucs ov4-vs-ov2 gate (descriptive, per converter/verify/README.md rules):
# different chunk geometries CANNOT be judged by naive SNR (split-jitter dominates) —
# we check (a) corr ~0.99+ / no gross energy change per stem, (b) sum(stems)-vs-mix
# residual is statistically identical between variants, (c) no periodic residual/diff
# spikes at step-boundary instants (boxcar averaging seam artifact would land there).
# Usage: python ov2_check.py <mix.wav> <ov4_dir> <ov2_dir> <step_samples_ov2>
import sys, glob, os
import numpy as np
import soundfile as sf

mix_path, d4, d2, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
mix, sr = sf.read(mix_path, dtype="float64")

def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.wav")):
        name = os.path.basename(p).split(".wav.")[-1].replace(".wav", "")
        out[name] = sf.read(p, dtype="float64")[0]
    return out

s4, s2 = load(d4), load(d2)
assert set(s4) == set(s2), (set(s4), set(s2))

def snr(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    err = np.sum((a - b) ** 2)
    return 10 * np.log10(np.sum(a ** 2) / err) if err > 0 else float("inf")

print(f"{'stem':<14}{'corr':>10}{'SNR dB':>10}{'rms_ov4':>12}{'rms_ov2':>12}{'level dB':>10}")
for name in sorted(s4):
    a, b = s4[name], s2[name]
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    af, bf = a.ravel(), b.ravel()
    corr = float(np.corrcoef(af, bf)[0, 1])
    ra, rb = np.sqrt(np.mean(af**2)), np.sqrt(np.mean(bf**2))
    lvl = 20 * np.log10(rb / ra) if ra > 0 else 0.0
    print(f"{name:<14}{corr:>10.6f}{snr(af, bf):>10.2f}{ra:>12.6f}{rb:>12.6f}{lvl:>+10.3f}")

# (b) additivity residual: mix - sum(stems), per variant
for tag, stems in (("ov4", s4), ("ov2", s2)):
    n = min(len(mix), *(len(v) for v in stems.values()))
    total = np.zeros((n, mix.shape[1] if mix.ndim > 1 else 1))
    for v in stems.values():
        vv = v[:n] if v.ndim > 1 else v[:n, None]
        total += vv
    m = mix[:n] if mix.ndim > 1 else mix[:n, None]
    resid = m - total
    rms = np.sqrt(np.mean(resid**2))
    print(f"additivity[{tag}]: residual rms={rms:.6e} ({20*np.log10(rms/np.sqrt(np.mean(m**2))):.1f} dB vs mix)")

# (c) boundary-seam probe on the ov4-vs-ov2 DIFFERENCE signal: windowed RMS around
# each ov2 step boundary vs mid-step windows — a seam artifact concentrates there.
name0 = "vocals" if "vocals" in s4 else sorted(s4)[0]
a, b = s4[name0], s2[name0]
n = min(len(a), len(b))
diff = (a[:n] - b[:n]); diff = diff.ravel() if diff.ndim == 1 else diff.mean(axis=1)
w = 2048
bound_rms, mid_rms = [], []
for pos in range(step, n - w, step):
    bound_rms.append(np.sqrt(np.mean(diff[pos - w//2 : pos + w//2] ** 2)))
    mid = pos + step // 2
    if mid + w//2 < n:
        mid_rms.append(np.sqrt(np.mean(diff[mid - w//2 : mid + w//2] ** 2)))
br, mr = np.mean(bound_rms), np.mean(mid_rms)
print(f"seam probe [{name0}]: boundary-window diff rms={br:.6e}, mid-step={mr:.6e}, ratio={br/mr:.3f}")
print("ratio ~1.0 = no seam concentration; >2 = investigate boundaries")
