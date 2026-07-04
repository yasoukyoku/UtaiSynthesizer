"""Gate — SoVITS auto-f0 predictor (<stem>.f0.onnx) equivalence: ORIGINAL
(so-vits-svc 4.1-Stable, models.py infer(predict_f0=True)) vs our standalone
F0PredictorWrapper export (architectures/sovits_v4.py) + the two-graph chain.

t0. normalize_f0 rewrite tier: so-vits utils.normalize_f0 (in-place
    `uv_sum[uv_sum==0]=9999`) vs our torch.where rewrite (random_scale=False)
    must be EXACTLY equal (diff == 0.0), incl. the all-unvoiced guard.
a.  torch tier, BOTH real checkpoints (akiko 4.0 vec256l9/gin256, 东雪莲 4.1
    vec768l12/vol_embedding/gin768/compressed): orig.infer(predict_f0=True)'s
    returned f0 (the predicted contour, models.py :523-527) vs our wrapper
    torch forward. Cases: normal singing contour with unvoiced gaps,
    interpolated-f0 (f0>0 everywhere, uv gaps — the runtime shape after
    RMVPE post_process), non-zero speaker id (akiko), ALL-UNVOICED uv=0 with
    non-zero f0 AND with f0=0 (the 9999 guard). PASS: max_abs_diff < 1e-5.
b.  torch wrapper vs ORT .f0.onnx (deterministic convert.convert_sovits — the
    f0 graph itself has no randomness): export-T (200) < 1e-4, dynamic-T
    sweep (311..6) < 5e-4, judged in the NETWORK OUTPUT DOMAIN (lf0, O(0.5))
    AND as relative Hz — the contract tolerances were written for O(1)
    outputs, while f0_pred is O(400 Hz): the Hz->lf0 slope is ~440 Hz per
    lf0 unit around 300 Hz, so the SAME ORT fp32 conv rounding that lands at
    ~1e-6 in the network's output reads as ~4e-4 in raw Hz (measured run 1:
    lf0 7.2e-7..1.2e-6, Hz 1.8e-4..6.1e-4). Absolute Hz is still REPORTED.
    Downstream the Hz perturbation is harmless at this size: 0 f0_to_coarse
    bin flips (tier c prints them) and sub-mHz NSF source drift.
c.  CHAIN tier (wiring-order proof): orig full infer(predict_f0=True) audio
    (PatchedNoise zero/fixed) vs [f0 graph -> f0_pred -> main det graph with
    f0=f0_pred, SOURCE uv unchanged].
      c-torch: all-torch chain (our wrapper -> our torch synthesizer) —
               isolates the wiring from ORT fp32 noise. PASS: f0_pred
               < 1e-5 (measured: 0.0 bitwise) AND audio < 5e-5. The audio
               line is above gate1_sovits' 1e-5 deliberately: with f0_pred
               bit-identical the residue is the S35-gated SineGen
               stable-phase deviation, and a PREDICTED contour is voiced
               end-to-end (no zero-Hz phase resets), so the sin-identity
               fp drift accumulates over the whole 2.3 s segment
               (measured 1.3e-5..2.1e-5 vs ~1e-6 on gappy contours).
      c-ort:   full ORT chain — the number is REPORTED; hard cap 1e-2 (a
               wiring error is O(0.1-1)); measured 1.1e-4 (akiko) /
               1.2e-3 (东雪莲): the ~5e-4 Hz ORT f0 perturbation phase-
               integrates through SineGen over the segment (2pi*df*t).
               f0-tensor diff + f0_to_coarse bin flips printed first so a
               failure bisects itself (a coarse-bin flip -> audio jumps).
d.  shipping-export tier via the convert.py CLI (subprocess, utf-8 env) for
    BOTH models into scratch: .f0.onnx exists, sidecar auto_f0 object
    (available/file/inputs) correct, every OTHER sidecar key byte-equal to
    the INSTALLED data/models/sovits/<stem>.json (auto_f0 stripped from both
    sides — idempotent after the in-place refresh), shipping .f0.onnx ORT vs
    torch wrapper < 1e-4 (no randomness -> shipping == det for this graph),
    main .onnx ORT sanity re-run at two T.

Run:  converter/.venv/Scripts/python.exe converter/verify/voice/gate_autof0.py
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
V40_PTH = r"D:\MyDev\TESTING\Sovits-SVC\MinamiyaAkiko-Sovits4.0\akiko_320000.pth"
V41_PTH = r"D:\MyDev\TESTING\Sovits-SVC\东雪莲\Sovits4.1东雪莲主模型.pth"
INSTALLED = r"D:\MyDev\Utai_v2-dev\data\models\sovits"
SCRATCH = (r"C:\Users\admin\AppData\Local\Temp\claude\D--MyDev-Utai-v2-dev"
           r"\86733fc5-aabc-40de-8c29-9c294ceb0706\scratchpad\autof0")

TOL_F0_TORCH = 1e-5      # tier a: orig torch vs wrapper torch (Hz)
# tier b/d: torch vs ORT judged in the lf0 output domain AND relative Hz
# (see the docstring — raw Hz is ~440x scale-amplified); abs Hz reported.
TOL_F0_ONNX = 1e-4       # at the export T
TOL_F0_ONNX_SWEEP = 5e-4  # dynamic-T sweep
TOL_CHAIN_TORCH_F0 = 1e-5  # tier c-torch: wrapper f0 == in-graph pred (bitwise)
TOL_CHAIN_TORCH = 5e-5   # tier c-torch: audio (SineGen phase drift on
                         # end-to-end-voiced predicted contours, docstring)
CHAIN_ORT_CAP = 1e-2     # tier c-ort: hard cap (wiring errors are O(0.1-1));
                         # actual reported, measured 1.1e-4 .. 1.2e-3
EXPORT_T = 200
MIN_FRAMES = 6
NOISE_SCALE = 0.4

sys.path.insert(0, SOVITS_REPO)
sys.path.insert(0, CONVERTER)

import models as orig_models          # noqa: E402 — so-vits models.py
import utils as so_utils              # noqa: E402 — so-vits utils.py
from architectures import sovits_v4   # noqa: E402
import convert as convert_mod         # noqa: E402
import onnxruntime as ort             # noqa: E402

os.makedirs(SCRATCH, exist_ok=True)
torch.set_grad_enabled(False)

failures = []


def check(cond, label):
    tag = "PASS" if cond else "FAIL"
    print(f"    [{tag}] {label}")
    if not cond:
        failures.append(label)


def mad(a, b):
    return (a - b).abs().max().item()


def hz_to_lf0(f0_hz):
    """Back to the network's output domain (monotonic inverse of :527) so the
    ORT-vs-torch diff can be judged without the ~440x Hz-scale amplification."""
    return 2595. * torch.log10(1. + f0_hz.clamp(min=-699.0) / 700.) / 500


def f0_diff(tf, of):
    """(abs Hz, lf0-domain, relative Hz) — tier b/d judge lf0 + rel, report Hz."""
    d_hz = mad(tf, of)
    d_lf0 = mad(hz_to_lf0(tf), hz_to_lf0(of))
    d_rel = d_hz / max(tf.abs().max().item(), 1.0)
    return d_hz, d_lf0, d_rel


class PatchedNoise:
    """torch.rand -> zeros; torch.randn_like -> `fixed` for tensors matching
    fixed.shape (the z_p noise), zeros otherwise (the SineGen noise).
    Template: gate1_sovits.py (that script executes at import — not importable)."""

    def __init__(self, fixed=None):
        self.fixed = fixed

    def __enter__(self):
        self._rand, self._randn_like = torch.rand, torch.randn_like

        def zrand(*size, **kw):
            return torch.zeros(*size, device=kw.get("device"),
                               dtype=kw.get("dtype"))

        def zrandn_like(t, **kw):
            if self.fixed is not None and tuple(t.shape) == tuple(self.fixed.shape):
                return self.fixed
            return torch.zeros_like(t)

        torch.rand, torch.randn_like = zrand, zrandn_like
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like = self._rand, self._randn_like


def build_orig(cfg, strict_state, allow_missing_enc_q=False):
    orig = orig_models.SynthesizerTrn(
        cfg["data"]["filter_length"] // 2 + 1,
        cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
        **cfg["model"])
    if allow_missing_enc_q:
        missing, unexpected = orig.load_state_dict(strict_state, strict=False)
        check(not unexpected, f"orig load: no unexpected keys ({len(unexpected)})")
        check(all(k.startswith("enc_q.") for k in missing),
              f"orig load: missing keys are enc_q-only ({len(missing)})")
    else:
        orig.load_state_dict(strict_state, strict=True)
        print("    orig strict=True load OK")
    orig.eval()
    return orig


def make_session(path):
    """ORT session from BYTES: the python onnxruntime build cannot open
    session paths with Chinese characters on Windows (locale ACP issue)."""
    return ort.InferenceSession(Path(path).read_bytes(),
                                providers=["CPUExecutionProvider"])


def make_inputs(t, ssl_dim, seed, sid_val=0, interp_f0=False):
    """interp_f0=True mimics the runtime shape (RMVPE post_process fills the
    unvoiced stretches by interpolation: f0 > 0 everywhere, uv marks them)."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(1, t, ssl_dim, generator=g)
    f0 = 150.0 + 250.0 * torch.rand(1, t, generator=g)
    uv = torch.ones(1, t)
    if t >= 20:
        u0 = t // 5
        uv[:, u0:u0 + 10] = 0.0
        if not interp_f0:
            f0 = f0 * uv  # true zeros in the unvoiced stretch
    else:
        uv = (f0 > 0).float()
    sid = torch.full((1,), sid_val, dtype=torch.int64)
    return c, f0, uv, sid


def orig_pred_f0(orig, c, f0, uv, sid, vol=None):
    """The ORIGINAL model's predicted f0: infer(predict_f0=True) returns
    (audio, f0) — the returned f0 IS pred (models.py :527, :532)."""
    with PatchedNoise():
        _, f0_pred = orig.infer(c.transpose(1, 2), f0, uv, g=sid,
                                noice_scale=NOISE_SCALE, predict_f0=True,
                                vol=vol)
    return f0_pred


# ===========================================================================
print("=== tier (t0) — normalize_f0 torch.where rewrite: EXACT equality ===")
gen = torch.Generator().manual_seed(7)
for label, uv_case, f0_case in (
        ("normal (uv gaps)", "gaps", "rand"),
        ("all-voiced", "ones", "rand"),
        ("ALL-UNVOICED, f0 nonzero (9999 guard)", "zeros", "rand"),
        ("ALL-UNVOICED, f0 zero", "zeros", "zeros")):
    T = 64
    lf0 = (torch.rand(1, 1, T, generator=gen) * 0.6
           if f0_case == "rand" else torch.zeros(1, 1, T))
    uv = {"gaps": torch.cat([torch.ones(1, 40), torch.zeros(1, 24)], dim=1),
          "ones": torch.ones(1, T),
          "zeros": torch.zeros(1, T)}[uv_case]
    x_mask = torch.ones(1, 1, T)
    ref = so_utils.normalize_f0(lf0.clone(), x_mask, uv.clone(),
                                random_scale=False)
    got = sovits_v4.normalize_f0(lf0, x_mask, uv, random_scale=False)
    d = mad(ref, got)
    check(d == 0.0, f"normalize_f0 {label}: diff {d:.1e} == 0.0")

# ===========================================================================
MODELS = [
    ("4.0 akiko", V40_PTH, False),
    ("4.1 东雪莲", V41_PTH, True),
]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
built = {}

for tag, pth, compressed in MODELS:
    print(f"=== tier (a) — {tag}: orig infer(predict_f0=True) vs wrapper torch ===")
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    cfg, cfg_path = sovits_v4.load_sovits_config(pth)
    print(f"  config: {cfg_path}")
    state = {k: v.float() for k, v in ck["model"].items()}
    orig = build_orig(cfg, state, allow_missing_enc_q=compressed)
    ours, meta = sovits_v4.build_from_checkpoint(ck, cfg)
    sovits_v4.set_deterministic(ours, True)
    check(meta["has_f0_decoder"] is True,
          f"{tag} meta has_f0_decoder == True (weights are the truth)")
    wrapper = sovits_v4.F0PredictorWrapper(ours)
    wrapper.eval()
    ssl_dim, inter = meta["features_dim"], meta["inter_channels"]
    with_vol = meta["vol_embedding"]

    def vol_for(t, seed=41):
        if not with_vol:
            return None
        g = torch.Generator().manual_seed(seed)
        audio_syn = 0.2 * torch.randn(1, t * 512, generator=g)
        return so_utils.Volume_Extractor(512).extract(audio_syn)[None, :]

    cases = [
        ("normal singing (uv gaps, f0=0 in gaps)",
         make_inputs(EXPORT_T, ssl_dim, 1234)),
        ("interpolated f0 (f0>0 everywhere, uv gaps)",
         make_inputs(EXPORT_T, ssl_dim, 4321, interp_f0=True)),
    ]
    if meta["n_speakers"] > 1:
        cases.append(("non-zero speaker id (sid=3)",
                      make_inputs(EXPORT_T, ssl_dim, 9876, sid_val=3)))
    # guard tier: ALL-UNVOICED (uv_sum==0 -> 9999) with non-zero and zero f0
    c_u, _, _, sid_u = make_inputs(EXPORT_T, ssl_dim, 555)
    cases.append(("ALL-UNVOICED uv=0, f0=220 Hz",
                  (c_u, torch.full((1, EXPORT_T), 220.0),
                   torch.zeros(1, EXPORT_T), sid_u)))
    cases.append(("ALL-UNVOICED uv=0, f0=0",
                  (c_u, torch.zeros(1, EXPORT_T),
                   torch.zeros(1, EXPORT_T), sid_u)))

    for label, (c, f0, uv, sid) in cases:
        vol = vol_for(f0.shape[1])
        ref = orig_pred_f0(orig, c, f0, uv, sid, vol=vol)
        got = wrapper(c, f0, uv, sid, vol) if with_vol else wrapper(c, f0, uv, sid)
        d = mad(ref, got)
        check(d < TOL_F0_TORCH,
              f"{tag} {label}: f0_pred max_abs_diff {d:.3e} < {TOL_F0_TORCH:.0e}")

    # =======================================================================
    print(f"=== tier (b) — {tag}: wrapper torch vs ORT .f0.onnx ===")
    det_onnx = Path(SCRATCH) / (Path(pth).stem + ".det.onnx")
    convert_mod.convert_sovits(Path(pth), det_onnx, deterministic=True)
    det_f0_onnx = det_onnx.with_name(det_onnx.stem + ".f0.onnx")
    check(det_f0_onnx.exists(), f"{tag}: {det_f0_onnx.name} exported")
    f0_sess = make_session(det_f0_onnx)
    f0_in_names = [i.name for i in f0_sess.get_inputs()]
    check(f0_in_names == (["c", "f0", "uv", "sid"] + (["vol"] if with_vol else [])),
          f"{tag}: f0 graph inputs {f0_in_names}")

    def run_f0_ort(c, f0, uv, sid, vol=None):
        feeds = {"c": c.numpy(), "f0": f0.numpy(), "uv": uv.numpy(),
                 "sid": sid.numpy()}
        if vol is not None:
            feeds["vol"] = vol.numpy()
        return torch.from_numpy(f0_sess.run(None, feeds)[0])

    for t, tol in [(EXPORT_T, TOL_F0_ONNX)] + \
                  [(tt, TOL_F0_ONNX_SWEEP) for tt in (311, 137, 57, 20, 10, MIN_FRAMES)]:
        c, f0, uv, sid = make_inputs(t, ssl_dim, 100 + t)
        vol = vol_for(t, seed=200 + t)
        tf = wrapper(c, f0, uv, sid, vol) if with_vol else wrapper(c, f0, uv, sid)
        of = run_f0_ort(c, f0, uv, sid, vol)
        d_hz, d_lf0, d_rel = f0_diff(tf, of)
        check(d_lf0 < tol and d_rel < tol,
              f"{tag} T={t}: f0_pred lf0-domain {d_lf0:.3e} & rel-Hz "
              f"{d_rel:.3e} < {tol:.0e} (abs {d_hz:.3e} Hz)")

    # =======================================================================
    print(f"=== tier (c) — {tag}: CHAIN orig full infer vs f0 graph -> main graph ===")
    main_sess = make_session(det_onnx)
    c, f0, uv, sid = make_inputs(EXPORT_T, ssl_dim, 2468, interp_f0=True)
    vol = vol_for(EXPORT_T, seed=99)

    for mode in ("zero-noise", "fixed-noise"):
        if mode == "zero-noise":
            fixed = None
            noise = torch.zeros(1, inter, EXPORT_T)
        else:
            g = torch.Generator().manual_seed(13579)
            fixed = torch.randn(1, inter, EXPORT_T, generator=g)
            noise = fixed * NOISE_SCALE

        with PatchedNoise(fixed):
            o_orig, f0_pred_orig = orig.infer(
                c.transpose(1, 2), f0, uv, g=sid, noice_scale=NOISE_SCALE,
                predict_f0=True, vol=vol)

        # c-torch: all-torch chain — the pure wiring-order proof
        f0p_t = wrapper(c, f0, uv, sid, vol) if with_vol else wrapper(c, f0, uv, sid)
        audio_t = (ours(c, f0p_t, uv, noise, sid, vol) if with_vol
                   else ours(c, f0p_t, uv, noise, sid))
        d_f0_t = mad(f0_pred_orig, f0p_t)
        d_at = mad(o_orig, audio_t)
        print(f"      {mode:>11} c-torch: f0_pred diff {d_f0_t:.3e}, "
              f"audio diff {d_at:.3e}")
        check(d_f0_t < TOL_CHAIN_TORCH_F0,
              f"{tag} {mode} c-torch chain f0_pred {d_f0_t:.3e} "
              f"< {TOL_CHAIN_TORCH_F0:.0e}")
        check(d_at < TOL_CHAIN_TORCH,
              f"{tag} {mode} c-torch chain audio {d_at:.3e} < {TOL_CHAIN_TORCH:.0e}")

        # c-ort: full ORT chain (f0.onnx -> main det onnx, SOURCE uv kept)
        f0p_o = run_f0_ort(c, f0, uv, sid, vol)
        feeds = {"c": c.numpy(), "f0": f0p_o.numpy(), "uv": uv.numpy(),
                 "noise": noise.numpy(), "sid": sid.numpy()}
        if with_vol:
            feeds["vol"] = vol.numpy()
        audio_o = torch.from_numpy(main_sess.run(None, feeds)[0])
        d_f0_o = mad(f0_pred_orig, f0p_o)
        flips = int((sovits_v4.f0_to_coarse(f0_pred_orig)
                     != sovits_v4.f0_to_coarse(f0p_o)).sum().item())
        d_ao = mad(o_orig, audio_o)
        print(f"      {mode:>11} c-ort:   f0_pred diff {d_f0_o:.3e}, "
              f"coarse-bin flips {flips}/{EXPORT_T}, audio diff {d_ao:.3e}")
        check(d_ao < CHAIN_ORT_CAP,
              f"{tag} {mode} c-ort chain audio {d_ao:.3e} < {CHAIN_ORT_CAP:.0e} "
              f"(reported; SineGen phase-integrates the sub-mHz ORT f0 diff)")

    built[tag] = (wrapper, meta, with_vol)
    del orig, ours, ck, state  # keep peak RSS down before the next model

# ===========================================================================
print("=== tier (d) — shipping exports via convert.py CLI vs INSTALLED sidecars ===")
SHIP_DIR = Path(SCRATCH) / "ship"
SHIP_DIR.mkdir(exist_ok=True)

for tag, pth, _ in MODELS:
    stem = Path(pth).stem
    ship_onnx = SHIP_DIR / f"{stem}.onnx"
    proc = subprocess.run(
        [sys.executable, "convert.py", "--input", pth,
         "--output", str(ship_onnx), "--type", "sovits"],
        cwd=CONVERTER, capture_output=True, env=env)
    stdout_txt = proc.stdout.decode("utf-8", errors="replace")
    print("    " + "\n    ".join(stdout_txt.strip().splitlines()[-4:]))
    check(proc.returncode == 0, f"convert.py CLI exit code {proc.returncode} ({stem})")
    if proc.returncode != 0:
        print(proc.stderr.decode("utf-8", errors="replace"))
        continue

    ship_f0 = SHIP_DIR / f"{stem}.f0.onnx"
    check(ship_f0.exists(), f"{stem}.f0.onnx exists ({ship_f0.stat().st_size} bytes)")

    wrapper, meta, with_vol = built[tag]
    sidecar = json.loads((SHIP_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    expect_auto = {"available": True, "file": f"{stem}.f0.onnx",
                   "inputs": ["c", "f0", "uv", "sid"] + (["vol"] if with_vol else [])}
    check(sidecar.get("auto_f0") == expect_auto,
          f"{stem} sidecar auto_f0 == {expect_auto} (got {sidecar.get('auto_f0')})")

    # every OTHER key must match the INSTALLED sidecar exactly (strip auto_f0
    # from both sides so the check stays idempotent after the in-place refresh)
    installed = json.loads(Path(INSTALLED, f"{stem}.json").read_text(encoding="utf-8"))
    a = {k: v for k, v in sidecar.items() if k != "auto_f0"}
    b = {k: v for k, v in installed.items() if k != "auto_f0"}
    check(a == b, f"{stem} sidecar: all non-auto_f0 keys == installed json "
                  f"(keys {sorted(a.keys())})")

    # shipping f0 graph == torch wrapper (no randomness in this graph, so the
    # shipping export must equal the deterministic one semantically)
    ship_sess = make_session(ship_f0)
    ssl_dim = meta["features_dim"]
    c, f0, uv, sid = make_inputs(EXPORT_T, ssl_dim, 31415)
    vol = (0.05 + 0.1 * torch.rand(1, EXPORT_T,
                                   generator=torch.Generator().manual_seed(6)))
    feeds = {"c": c.numpy(), "f0": f0.numpy(), "uv": uv.numpy(), "sid": sid.numpy()}
    tf = wrapper(c, f0, uv, sid, vol) if with_vol else wrapper(c, f0, uv, sid)
    if with_vol:
        feeds["vol"] = vol.numpy()
    of = torch.from_numpy(ship_sess.run(None, feeds)[0])
    d_hz, d_lf0, d_rel = f0_diff(tf, of)
    check(d_lf0 < TOL_F0_ONNX and d_rel < TOL_F0_ONNX,
          f"{stem} shipping .f0.onnx vs torch wrapper: lf0-domain {d_lf0:.3e} "
          f"& rel-Hz {d_rel:.3e} < {TOL_F0_ONNX:.0e} (abs {d_hz:.3e} Hz)")

    # main shipping onnx still loads + runs (dynamic T) — auto-f0 export must
    # not have disturbed the main graph path
    main_sess = make_session(ship_onnx)
    for t in (EXPORT_T, 333):
        c, f0, uv, sid = make_inputs(t, ssl_dim, 7000 + t)
        feeds = {"c": c.numpy(), "f0": f0.numpy(), "uv": uv.numpy(),
                 "noise": (torch.randn(1, 192, t) * NOISE_SCALE).numpy(),
                 "sid": sid.numpy()}
        if with_vol:
            feeds["vol"] = (0.05 + 0.1 * torch.rand(1, t)).numpy()
        audio = main_sess.run(None, feeds)[0]
        ok = (audio.shape == (1, 1, t * 512) and np.isfinite(audio).all()
              and np.abs(audio).max() <= 1.0)
        check(ok, f"{stem} main onnx T={t}: shape {audio.shape} == (1,1,{t * 512}), "
                  f"finite, |max| {np.abs(audio).max():.3f} <= 1")

# ===========================================================================
print()
if failures:
    print(f"GATE FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL AUTO-F0 GATES PASSED")
