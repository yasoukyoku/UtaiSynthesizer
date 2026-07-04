"""Gate 1 — NSF-HiFiGAN vocoder equivalence: ORIGINAL (so-vits-svc 4.1-Stable
vdecoder/nsf_hifigan) vs our port (architectures/nsf_hifigan_gen.py).

a.  REAL weights (pretrain/nsf_hifigan/model, 44.1k/hop512/128mel): original
    Generator built per load_model() (config.json + strict load +
    remove_weight_norm) vs ours, driven by a REAL mel (original nvSTFT
    get_mel on real singing audio) + an f0 ramp 100->400 Hz with an unvoiced
    (0 Hz) span. Zero-noise tier (original monkeypatched, ours
    deterministic=True — exercises the shipped gating mechanism) AND
    fixed-noise tier (both sides monkeypatched to identical shape-keyed
    noise). The SineGen stable-phase reformulation is a gated DEVIATION
    (sin-invariant identity, see rvc_v2.SineGen) so audio is NOT bit-equal:
    PASS line = max_abs_diff < 5e-4 AND corr > 0.9999; the sine output is
    always compared directly (< 1e-5) so a failure bisects itself.
b.  ours torch vs ORT deterministic export: < 1e-4 at the export T (200),
    dynamic-T sweep down to 6 frames < 5e-4 (same rationale as gate1_sovits:
    ORT/torch fp32 conv rounding through the upsample/resblock stack).
c.  nvSTFT get_mel DSP: original torch vs an independent numpy float64
    reimplementation of the contract math (pad (win-hop)//2 with the
    reflect/constant switch, hann periodic, sqrt(re^2+im^2+1e-9), slaney mel
    filters, ln(clamp 1e-5)) < 1e-5 on deterministic chirp+LCG-noise signals
    (the noise floor keeps every mel band well above the 1e-5 clamp — near
    the clamp ln() amplifies fp rounding by up to 1e5 and no cross-
    implementation comparison is meaningful). Covers BOTH the reflect branch
    (N=5120) and the constant-pad branch (N=700 < pad_left). Also DUMPS Rust
    unit-test reference vectors (gen_refs const-array style, %.9e, values
    from the imported ORIGINAL nvSTFT) to test_output/nsf_hifigan_gate/
    rust_mel_refs.txt for src-tauri/src/inference/mel.rs.
d.  shipping export via the export_nsf_hifigan.py CLI into
    data/models/aux/ (the REAL deployment target): sidecar schema, mel
    filterbank npy == librosa exactly, ORT sanity at two T (bytes-load per
    house rule), and live-noise proof (two identical runs must differ).

Run:  converter/.venv/Scripts/python.exe converter/verify/voice/gate1_nsf_hifigan.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Windows console default codepage (cp932/936) chokes on the report glyphs.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOVITS_REPO = r"D:\MyDev\so-vits-svc\so-vits-svc"
CONVERTER = r"D:\MyDev\Utai_v2-dev\converter"
MODEL = os.path.join(SOVITS_REPO, "pretrain", "nsf_hifigan", "model")
REAL_WAV = r"D:\MyDev\TESTING\ikanaiteyo\vocal.wav"
AUX_DIR = r"D:\MyDev\Utai_v2-dev\data\models\aux"
GATE_OUT = os.path.join(CONVERTER, "test_output", "nsf_hifigan_gate")

TOL_SINE = 1e-5    # SineGen stable-phase reformulation vs original, direct
TOL_AUDIO = 5e-4   # torch-vs-torch audio through the full generator (the
                   # gated deviation grows through conv stacks; contract line)
MIN_CORR = 0.9999
TOL_ONNX = 1e-4    # ours torch vs ORT fp32 at the export T
TOL_ONNX_SWEEP = 5e-4
TOL_MEL = 1e-5     # original torch get_mel vs independent numpy f64 reimpl
EXPORT_T = 200

SR = 44100
N_FFT = 2048
WIN = 2048
HOP = 512
N_MELS = 128
FMIN = 40
FMAX = 16000

sys.path.insert(0, SOVITS_REPO)
sys.path.insert(0, CONVERTER)

import soundfile as sf                                    # noqa: E402
from librosa.filters import mel as librosa_mel_fn         # noqa: E402
from vdecoder.nsf_hifigan import models as orig_nsf       # noqa: E402
from vdecoder.nsf_hifigan.nvSTFT import STFT as OrigSTFT  # noqa: E402

from architectures import nsf_hifigan_gen                 # noqa: E402
import export_nsf_hifigan as export_mod                   # noqa: E402

os.makedirs(GATE_OUT, exist_ok=True)
torch.set_grad_enabled(False)

failures = []


def check(cond, label):
    tag = "PASS" if cond else "FAIL"
    print(f"    [{tag}] {label}")
    if not cond:
        failures.append(label)


def mad(a, b):
    return (a - b).abs().max().item()


def corr(a, b):
    x = a.reshape(-1).numpy().astype(np.float64)
    y = b.reshape(-1).numpy().astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1])


# --- noise patching --------------------------------------------------------
# original SineGen randomness: rand_ini = torch.rand([B, 9]) and
# noise = noise_amp * torch.randn_like(sine_waves). Ours uses the same two
# calls when deterministic=False (zeroed internally when True).

# pristine references captured BEFORE any patching (FixedNoise._get must not
# call the patched torch.rand — that recurses)
_TRUE_RAND = torch.rand
_TRUE_RANDN = torch.randn

class ZeroNoise:
    """torch.rand -> zeros, torch.randn_like -> zeros (for the ORIGINAL side
    of the zero-noise tier; ours runs deterministic=True instead, so the
    shipped gating mechanism itself is what gets exercised)."""

    def __enter__(self):
        self._rand, self._randn_like = torch.rand, torch.randn_like

        def zrand(*size, **kw):
            if len(size) == 1 and isinstance(size[0], (tuple, list)):
                size = tuple(size[0])
            return torch.zeros(*size, dtype=kw.get("dtype"))

        torch.rand = zrand
        torch.randn_like = lambda t, **kw: torch.zeros_like(t)
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like = self._rand, self._randn_like


class FixedNoise:
    """torch.rand / torch.randn_like -> deterministic tensors cached by
    (kind, shape): the ORIGINAL and OUR forwards make the same calls with the
    same shapes in the same order, so both receive identical noise. Clones
    are returned because both SineGens mutate rand_ini in place
    (rand_ini[:, 0] = 0)."""

    def __init__(self, seed):
        self.seed = seed
        self.cache = {}

    def _get(self, kind, shape):
        key = (kind, tuple(int(s) for s in shape))
        if key not in self.cache:
            g = torch.Generator().manual_seed(self.seed + len(self.cache))
            fn = _TRUE_RAND if kind == "rand" else _TRUE_RANDN
            self.cache[key] = fn(*key[1], generator=g)
        return self.cache[key].clone()

    def __enter__(self):
        self._rand, self._randn_like = torch.rand, torch.randn_like
        outer = self

        def frand(*size, **kw):
            if len(size) == 1 and isinstance(size[0], (tuple, list)):
                size = tuple(size[0])
            return outer._get("rand", size)

        torch.rand = frand
        torch.randn_like = lambda t, **kw: outer._get("randn", t.shape)
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like = self._rand, self._randn_like


# --- drive signals ----------------------------------------------------------

def load_real_audio():
    data, sr = sf.read(REAL_WAV, always_2d=True)
    mono = data[:, 0].astype(np.float32)
    if sr != SR:
        import librosa
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    assert np.isfinite(mono).all()
    return mono


REAL_AUDIO = load_real_audio()
ORIG_STFT = OrigSTFT(SR, N_MELS, N_FFT, WIN, HOP, FMIN, FMAX)


def real_slice(n):
    start = len(REAL_AUDIO) // 3
    assert start + n <= len(REAL_AUDIO), (start, n, len(REAL_AUDIO))
    return REAL_AUDIO[start:start + n]


def mel_for(t):
    """REAL mel [1, 128, t] via the ORIGINAL nvSTFT get_mel."""
    y = torch.from_numpy(real_slice(t * HOP).copy())[None, :]
    m = ORIG_STFT.get_mel(y)
    assert m.shape == (1, N_MELS, t), m.shape
    return m


def make_f0(t):
    """f0 ramp 100->400 Hz with an unvoiced (0 Hz) span (uv gating path)."""
    f0 = torch.linspace(100.0, 400.0, t).unsqueeze(0)
    u0 = t // 3
    f0[:, u0:u0 + max(t // 6, 1)] = 0.0
    return f0


# ===========================================================================
print("=== gate (a) — real weights: original Generator vs our port ===")
h_orig = orig_nsf.load_config(MODEL)  # AttrDict from config.json next to model
orig = orig_nsf.Generator(h_orig)
cp = torch.load(MODEL, map_location="cpu", weights_only=False)
orig.load_state_dict(cp["generator"], strict=True)  # load_model() semantics
orig.eval()
orig.remove_weight_norm()
del cp
print("    orig strict=True load + remove_weight_norm OK")

ours = nsf_hifigan_gen.build_from_checkpoint(MODEL, dict(h_orig))
nsf_hifigan_gen.set_deterministic(ours, True)
print(f"    ours strict=True load OK (upp={ours.upp}, "
      f"{sum(p.numel() for p in ours.parameters())} params)")
check(ours.upp == 512 and h_orig["hop_size"] == 512,
      f"upp == hop_size == 512 (got {ours.upp})")

mel200 = mel_for(EXPORT_T)
f0_200 = make_f0(EXPORT_T)

# sine bisect first — the ONE deviation lives here (always printed)
with ZeroNoise():
    s_orig, uv_o, _ = orig.m_source.l_sin_gen(f0_200, 512)
s_ours, uv_m, _ = ours.m_source.l_sin_gen(f0_200, 512)  # deterministic=True
d_sine = mad(s_orig, s_ours)
d_uv = mad(uv_o, uv_m)
check(d_sine < TOL_SINE,
      f"SineGen zero-noise sine max_abs_diff {d_sine:.3e} < {TOL_SINE:.0e} "
      f"(stable-phase deviation, uv diff {d_uv:.1e})")

with ZeroNoise():
    a_orig = orig(mel200, f0_200)
a_ours = ours(mel200, f0_200)
d_z = mad(a_orig, a_ours)
c_z = corr(a_orig, a_ours)
check(d_z < TOL_AUDIO and c_z > MIN_CORR,
      f"zero-noise audio max_abs_diff {d_z:.3e} < {TOL_AUDIO:.0e}, "
      f"corr {c_z:.6f} > {MIN_CORR}")

nsf_hifigan_gen.set_deterministic(ours, False)
fixed = FixedNoise(20260704)
with fixed:
    a_orig_f = orig(mel200, f0_200)
with fixed:
    a_ours_f = ours(mel200, f0_200)
nsf_hifigan_gen.set_deterministic(ours, True)
d_f = mad(a_orig_f, a_ours_f)
c_f = corr(a_orig_f, a_ours_f)
check(len(fixed.cache) == 2,
      f"fixed-noise patch saw exactly rand+randn_like shapes ({list(fixed.cache)})")
check(d_f < TOL_AUDIO and c_f > MIN_CORR,
      f"fixed-noise audio max_abs_diff {d_f:.3e} < {TOL_AUDIO:.0e}, "
      f"corr {c_f:.6f} > {MIN_CORR}")

# ===========================================================================
print("=== gate (b) — ours torch vs ORT (deterministic export) ===")
import onnxruntime as ort  # noqa: E402

DET_DIR = os.path.join(GATE_OUT, "det")
export_mod.run_export(MODEL, None, DET_DIR, deterministic=True)
sess_det = ort.InferenceSession(
    Path(DET_DIR, export_mod.ONNX_NAME).read_bytes(),
    providers=["CPUExecutionProvider"])
check([i.name for i in sess_det.get_inputs()] == ["mel", "f0"],
      "det graph inputs == [mel, f0]")


def ort_vs_torch(sess, t):
    m = mel_for(t)
    f0 = make_f0(t)
    ref = ours(m, f0)
    got = sess.run(None, {"mel": m.numpy(), "f0": f0.numpy()})[0]
    assert got.shape == (1, 1, t * HOP), got.shape
    return mad(ref, torch.from_numpy(got))


d_main = ort_vs_torch(sess_det, EXPORT_T)
check(d_main < TOL_ONNX,
      f"T={EXPORT_T}: torch vs ORT max_abs_diff {d_main:.3e} < {TOL_ONNX:.0e}")
print(f"    dynamic-T sweep (graph exported at T={export_mod.EXPORT_T}):")
for t in (311, 137, 57, 20, 10, 6):
    d = ort_vs_torch(sess_det, t)
    check(d < TOL_ONNX_SWEEP,
          f"T={t}: max_abs_diff {d:.3e} < {TOL_ONNX_SWEEP:.0e}")

# ===========================================================================
print("=== gate (c) — nvSTFT get_mel: original vs independent numpy reimpl ===")

MEL_BASIS_F64 = librosa_mel_fn(
    sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).astype(np.float64)


def numpy_get_mel(y):
    """Independent float64 reimplementation of nvSTFT.get_mel (keyshift=0,
    speed=1, center=False — the contract math, coded from nvSTFT.py:71-125
    without torch)."""
    y = y.astype(np.float64)
    pad_left = (WIN - HOP) // 2
    pad_right = max((WIN - HOP + 1) // 2, WIN - len(y) - pad_left)
    mode = "reflect" if pad_right < len(y) else "constant"
    yp = np.pad(y, (pad_left, pad_right), mode=mode)
    n = np.arange(WIN, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / WIN)  # periodic (torch default)
    n_frames = 1 + (len(yp) - WIN) // HOP
    mags = np.empty((N_FFT // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        seg = yp[i * HOP:i * HOP + WIN] * hann
        sp = np.fft.rfft(seg)
        mags[:, i] = np.sqrt(sp.real ** 2 + sp.imag ** 2 + 1e-9)
    m = MEL_BASIS_F64 @ mags
    return np.log(np.clip(m, 1e-5, None)), mode


def lcg_noise(n):
    """Deterministic broadband noise, exactly reproducible in Rust with u32
    integer arithmetic (Numerical Recipes LCG). Keeps every mel band well
    above the ln clamp so the comparison is well-conditioned."""
    out = np.empty(n, dtype=np.float64)
    s = 1
    for i in range(n):
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        out[i] = s / 4294967296.0 * 2.0 - 1.0
    return out


def make_chirp(n, amp_noise=0.1):
    """0.5*sin(2*pi*(220*t + (880-220)/(2*dur)*t^2)) + amp_noise*lcg, f64
    math, cast f32 at the end (the Rust test must generate the identical
    signal — formula documented in the ref dump)."""
    t = np.arange(n, dtype=np.float64) / SR
    dur = n / SR
    phase = 2.0 * np.pi * (220.0 * t + (880.0 - 220.0) / (2.0 * dur) * t * t)
    return (0.5 * np.sin(phase) + amp_noise * lcg_noise(n)).astype(np.float32)


def orig_get_mel_np(y_f32):
    return ORIG_STFT.get_mel(torch.from_numpy(y_f32)[None, :])[0].numpy()


chirp_a = make_chirp(4096 + 512 * 2)   # 5120 samples -> reflect pad branch
chirp_b = make_chirp(700)              # 700 < pad_left -> constant pad branch

mel_a_orig = orig_get_mel_np(chirp_a)
mel_a_np, mode_a = numpy_get_mel(chirp_a)
check(mode_a == "reflect", f"N=5120 pad mode == reflect (got {mode_a})")
check(mel_a_orig.shape == (N_MELS, 10) and mel_a_np.shape == (N_MELS, 10),
      f"N=5120 n_frames == 10 == N//512 (got {mel_a_orig.shape})")
d_a = np.abs(mel_a_orig - mel_a_np).max()
check(d_a < TOL_MEL, f"N=5120 orig vs numpy ln-mel max_abs_diff {d_a:.3e} < {TOL_MEL:.0e}")

mel_b_orig = orig_get_mel_np(chirp_b)
mel_b_np, mode_b = numpy_get_mel(chirp_b)
check(mode_b == "constant", f"N=700 pad mode == constant (got {mode_b})")
check(mel_b_orig.shape == (N_MELS, 1),
      f"N=700 n_frames == 1 (got {mel_b_orig.shape})")
d_b = np.abs(mel_b_orig - mel_b_np).max()
check(d_b < TOL_MEL, f"N=700 orig vs numpy ln-mel max_abs_diff {d_b:.3e} < {TOL_MEL:.0e}")

# informational only: real audio has near-clamp bands where ln() amplifies fp
# rounding — no meaningful cross-implementation line exists there.
real_y = real_slice(EXPORT_T * HOP)
mel_r_orig = orig_get_mel_np(real_y)
mel_r_np, mode_r = numpy_get_mel(real_y)
d_r = np.abs(mel_r_orig - mel_r_np).max()
n_hot = int((np.abs(mel_r_orig - mel_r_np) > TOL_MEL).sum())
print(f"    [info] real audio T={EXPORT_T} ({mode_r}): max_abs_diff {d_r:.3e} "
      f"({n_hot}/{mel_r_orig.size} bins > {TOL_MEL:.0e}; near-clamp bins only)")
check(np.isfinite(mel_r_np).all() and mel_r_orig.shape == mel_r_np.shape,
      "real audio reimpl finite, shapes match")

# --- Rust unit-test reference vectors (gen_refs const-array style) ---------
REF_FRAMES = [0, 2, 4, 5, 7, 9]
REF_BINS = [0, 31, 64, 127]
SHORT_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 16, 32, 48, 64, 80, 96, 112, 127]


def rust_f32(name, vals):
    return (f"const {name}: &[f32] = &["
            + ", ".join(f"{v:.9e}" for v in np.asarray(vals, dtype=np.float32))
            + "];")


def rust_usize(name, vals):
    return (f"const {name}: &[usize] = &["
            + ", ".join(str(int(v)) for v in vals) + "];")


ref_lines = [
    "// nvSTFT get_mel reference vectors for src-tauri/src/inference/mel.rs",
    "// unit tests. Values computed by the imported ORIGINAL so-vits-svc",
    "// vdecoder/nsf_hifigan/nvSTFT.py get_mel (torch fp32) in",
    "// converter/verify/voice/gate1_nsf_hifigan.py gate (c).",
    "//",
    "// Input signal (generate identically in Rust, f64 math, cast f32):",
    "//   t = n / 44100.0; dur = N / 44100.0",
    "//   x[n] = (0.5*sin(2*pi*(220*t + (880-220)/(2*dur)*t*t))",
    "//           + 0.1*lcg[n]) as f32",
    "//   lcg: u32 state s = 1; per sample: s = s.wrapping_mul(1664525)",
    "//        .wrapping_add(1013904223); lcg[n] = (s as f64)/4294967296.0*2-1",
    "//   (exact integer arithmetic; the 0.1 noise floor keeps all mel bands",
    "//    well above the ln clamp 1e-5 so the comparison is well-conditioned)",
    "// mel filters: shipped aux/nsf_hifigan_mel.npy ([128,1025] f32, librosa",
    "//   slaney/slaney, sr=44100 n_fft=2048 n_mels=128 fmin=40 fmax=16000).",
    "// Suggested assert tolerance: 1e-4 absolute on ln-mel (fp32 STFT vs the",
    "//   torch fp32 oracle; measured orig-vs-f64-numpy max diff is in the",
    f"//   {max(d_a, d_b):.1e} range on these signals).",
    "",
    "// ---- signal A: N = 4096 + 512*2 = 5120 samples (reflect-pad branch) ----",
    f"const NSF_MEL_REF_N: usize = {len(chirp_a)};",
    f"const NSF_MEL_REF_N_FRAMES: usize = {mel_a_orig.shape[1]};",
    "// full first frame, mel bins 0..8 (frame 0):",
    rust_f32("NSF_MEL_REF_FRAME0_BINS0_8", mel_a_orig[0:8, 0]),
    "// 24 spot values at (REF_FRAMES x REF_BINS), row-major over frames:",
    rust_usize("NSF_MEL_REF_POINT_FRAMES",
               [f for f in REF_FRAMES for _ in REF_BINS]),
    rust_usize("NSF_MEL_REF_POINT_BINS",
               [b for _ in REF_FRAMES for b in REF_BINS]),
    rust_f32("NSF_MEL_REF_POINT_VALUES",
             [mel_a_orig[b, f] for f in REF_FRAMES for b in REF_BINS]),
    "",
    "// ---- signal B: N = 700 samples (< pad_left 768 -> constant-pad branch,",
    "//      same formula with N=700) ----",
    f"const NSF_MEL_SHORT_N: usize = {len(chirp_b)};",
    f"const NSF_MEL_SHORT_N_FRAMES: usize = {mel_b_orig.shape[1]};",
    rust_usize("NSF_MEL_SHORT_BINS", SHORT_BINS),
    rust_f32("NSF_MEL_SHORT_VALUES", [mel_b_orig[b, 0] for b in SHORT_BINS]),
]
ref_path = os.path.join(GATE_OUT, "rust_mel_refs.txt")
Path(ref_path).write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
print(f"    rust reference vectors -> {ref_path}")
print("\n".join("    | " + ln for ln in ref_lines))

# ===========================================================================
print("=== gate (d) — shipping export via export_nsf_hifigan.py CLI ===")
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
proc = subprocess.run(
    [sys.executable, "export_nsf_hifigan.py",
     "--model", MODEL, "--outdir", AUX_DIR],
    cwd=CONVERTER, capture_output=True, env=env)
stdout_txt = proc.stdout.decode("utf-8", errors="replace")
print("    " + "\n    ".join(stdout_txt.strip().splitlines()[-3:]))
check(proc.returncode == 0, f"export_nsf_hifigan.py CLI exit {proc.returncode}")
if proc.returncode != 0:
    print(proc.stderr.decode("utf-8", errors="replace"))
else:
    ship_onnx = os.path.join(AUX_DIR, export_mod.ONNX_NAME)
    ship_json = os.path.join(AUX_DIR, export_mod.JSON_NAME)
    ship_npy = os.path.join(AUX_DIR, export_mod.MEL_NPY_NAME)

    sidecar = json.loads(Path(ship_json).read_text(encoding="utf-8"))
    expect = {"type": "nsf_hifigan", "sample_rate": 44100, "hop_size": 512,
              "num_mels": 128, "n_fft": 2048, "win_size": 2048,
              "fmin": 40, "fmax": 16000,
              "mel_filters": "nsf_hifigan_mel.npy"}
    check(sidecar == expect, f"sidecar schema exact (got {sidecar})")

    shipped_basis = np.load(ship_npy)
    fresh_basis = librosa_mel_fn(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                                 fmin=FMIN, fmax=FMAX).astype(np.float32)
    check(shipped_basis.shape == (128, 1025)
          and shipped_basis.dtype == np.float32
          and np.array_equal(shipped_basis, fresh_basis),
          f"mel filterbank npy [128,1025] f32 == librosa EXACT "
          f"({shipped_basis.shape} {shipped_basis.dtype})")

    # bytes-load per house rule (python ORT cannot open Chinese paths; the
    # shipping loader in Rust is unaffected, but the gate must not depend on
    # the path being ASCII).
    sess_ship = ort.InferenceSession(Path(ship_onnx).read_bytes(),
                                     providers=["CPUExecutionProvider"])
    for t in (EXPORT_T, 333):
        m = mel_for(t).numpy()
        f0 = make_f0(t).numpy()
        audio = sess_ship.run(None, {"mel": m, "f0": f0})[0]
        ok = (audio.shape == (1, 1, t * HOP) and np.isfinite(audio).all()
              and np.abs(audio).max() <= 1.0)
        check(ok, f"shipped T={t}: shape {audio.shape} == (1,1,{t * HOP}), "
                  f"finite, |max| {np.abs(audio).max():.3f} <= 1")
    # the shipping graph must keep SineGen noise LIVE (RandomUniformLike /
    # RandomNormalLike): two identical runs must differ.
    a1 = sess_ship.run(None, {"mel": m, "f0": f0})[0]
    a2 = sess_ship.run(None, {"mel": m, "f0": f0})[0]
    check(np.abs(a1 - a2).max() > 0,
          f"shipped: in-graph noise is live (two runs differ by "
          f"{np.abs(a1 - a2).max():.1e})")
    # and the live output must still track the deterministic math (the noise
    # is small: voiced 0.003 additive + random harmonic init phases).
    det_ref = ours(torch.from_numpy(m), torch.from_numpy(f0))
    c_live = corr(det_ref, torch.from_numpy(a1))
    print(f"    [info] live vs det-torch corr {c_live:.4f} "
          f"(rand_ini shifts harmonic phases; informational)")

# ===========================================================================
print()
if failures:
    print(f"GATE FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL NSF-HIFIGAN GATES PASSED")
