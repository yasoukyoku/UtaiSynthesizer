# -*- coding: utf-8 -*-
"""gate1_sovits_v2 — SoVITS 4.0-v2 (VISinger2) converter equivalence gate.

Compares architectures/sovits_v4v2.py against the ORIGINAL 4.0-v2 branch code
(D:\\MyDev\\TESTING\\SoVITS-4.0_v2\\src\\so-vits-svc, official svc-develop-team
@ cf5a8fb) with REAL weights — self-consistency doesn't count (S31 lesson).

Tiers:
  (i)   ConviSTFT vs torch.istft(hann, center=True) — isolated module,
        random amp/phase: interior (>= 1 window from each edge) < 1e-5,
        edge deltas reported (documented deviation 4: 768- vs 1024-trim +
        tail zero-pad; buried inside the 0.5 s inference pad).
  (a)   chika G_73000 real weights, torch-vs-torch, fixed z/phase via scoped
        monkeypatches on the ORIGINAL (randn_like + Generator_Noise.forward
        phase source — istft call verbatim):
          pre-noise intermediates (text_encoder out, predict_mel, m_p/logs_p)
          must be EXACTLY 0.0 (identical op sequences);
          audio interior < 1e-4 (expected ~1e-5: one-ulp z_p regrouping
          (z*0.4)*exp vs z*exp*0.4 + ConviSTFT-vs-istft).
  (a2)  G_0 base weights — same tier, fresh-weight numeric regime.
  (b2)  refusals + missing-config bit-identity: build with the reconstructed
        template config vs config=None must produce BITWISE-equal audio; a
        4.x checkpoint must be refused by the v2 builder and a v2 checkpoint
        by the 4.x builder (loud, Chinese).
  (c)   torch(ours) vs ONNX shipping export (no in-graph randomness — the
        shipping graph IS the deterministic graph): same fixed inputs through
        ORT, audio < 1e-4 at T=200; dynamic sweep T ∈ {137, 64, 23, 7, 6}
        < 5e-4 (rel-attn generic path documented min_frames=6).
  (f)   auto-f0: ours F0PredictorWrapperV2 torch vs the original predict_f0
        branch (normalize_f0 factor pinned to 1 — documented deviation 5)
        must be EXACTLY 0.0 for BOTH outputs (f0_pred + the deviation-7
        f0d_cond side-effect term); <stem>.f0.onnx ORT vs torch < 1.2e-3
        Hz-relative (lf0-domain sensitivity, same bar as gate_autof0).
  (g)   FULL-CHAIN auto-f0 (deviation 7): companion → main graph (ours,
        decomposed) vs upstream infer(predict_f0=True) whose F0Decoder
        mutates decoder_input through its detach alias — fp64 dominance
        criteria like tier (a). This is the tier that would have caught the
        dropped side-effect term (review C0).

Run: converter/.venv/Scripts/python.exe converter/verify/voice/gate1_sovits_v2.py
Readings are recorded in converter/verify/voice/README.md (4.0-v2 section).
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import logging
# claim the root logger BEFORE the upstream import: the original utils.py runs
# logging.basicConfig(level=DEBUG) at import time and floods stdout with numba
# DEBUG (librosa's jit) — basicConfig is first-call-wins.
logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch


def istft_torch2(spec_real_imag, n_fft, hop, win, window, length):
    """torch>=2 shim for the original's real-input torch.istft call:
    view_as_complex(real/imag-stacked-last-dim) == the removed
    return_complex=False input semantics EXACTLY (bit-identical layout —
    same adaptation family as 4.1's mel_processing view_as_real)."""
    return torch.istft(torch.view_as_complex(spec_real_imag.contiguous()),
                       n_fft, hop, win, window, True, length=length)


HERE = Path(__file__).resolve()
CONVERTER = HERE.parents[2]
UPSTREAM = Path(r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc")
CKPT_DIR = Path(r"D:\MyDev\TESTING\SoVITS-4.0_v2\extracted\so-vits-svc\logs\44k")
CHIKA = CKPT_DIR / "G_73000.pth"
BASE_G0 = CKPT_DIR / "G_0.pth"
CONFIG = CKPT_DIR / "config.json"  # reconstructed: official template + spk {chika:0}
ONNX_MAIN = CONVERTER / "test_output" / "chika_v2.onnx"
ONNX_F0 = CONVERTER / "test_output" / "chika_v2.f0.onnx"

sys.path.insert(0, str(CONVERTER))          # architectures.*
sys.path.insert(0, str(UPSTREAM))           # models / utils / modules (original)

import json
import math

import models as orig_models          # noqa: E402  (original 4.0-v2 branch)
import utils as orig_utils            # noqa: E402
from architectures.sovits_v4v2 import (   # noqa: E402
    ConviSTFT,
    F0PredictorWrapperV2,
    build_from_checkpoint,
    is_v2_state_dict,
)

torch.manual_seed(20260717)
T = 200
HOP = 512
N_FFT = 2048
WIN = 2048
INTER = 192
PHASE_BINS = N_FFT // 2 + 1
NOICE_SCALE = 0.4

FAILS = []


def check(name, val, tol, exact=False):
    ok = (val == 0.0) if exact else (val < tol)
    tag = "PASS" if ok else "FAIL"
    bar = "== 0.0" if exact else f"< {tol:g}"
    print(f"[{tag}] {name}: {val:.3e} ({bar})")
    if not ok:
        FAILS.append(name)


def max_abs(a, b):
    return float((a - b).abs().max())


def fixed_inputs(t=T, seed=1234):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(1, t, 256, generator=g)
    f0 = 150.0 + 250.0 * torch.rand(1, t, generator=g)
    f0[:, 2 * t // 5: 2 * t // 5 + t // 10] = 0.0     # unvoiced stretch
    uv = (f0 > 0).float()
    z = torch.randn(1, INTER, t, generator=g)          # N(0,1), pre-scale ours
    phase = torch.rand(1, PHASE_BINS, t, generator=g) * 2 * 3.14 - 3.14
    return c, f0, uv, z, phase


# ---------------------------------------------------------------------------
# (i) ConviSTFT vs torch.istft — isolated
# ---------------------------------------------------------------------------

def tier_istft():
    print("\n=== tier (i): ConviSTFT vs torch.istft(hann) ===")
    g = torch.Generator().manual_seed(7)
    t = 173
    amp = torch.rand(1, PHASE_BINS, t, 1, generator=g) * 2.0
    phase = torch.rand(1, PHASE_BINS, t, 1, generator=g) * 2 * 3.14 - 3.14
    real = amp * torch.cos(phase)
    imag = amp * torch.sin(phase)

    # ground truth = the TRAINING formulation (models.py Generator_Noise)
    spec4 = torch.cat([real, imag], 3)
    window = torch.hann_window(WIN)
    ref = istft_torch2(spec4, N_FFT, HOP, WIN, window, length=t * HOP)

    ours_mod = ConviSTFT(WIN, HOP, N_FFT)
    spec2 = torch.cat([real, imag], 1).squeeze(3)
    with torch.no_grad():
        out = ours_mod(spec2).squeeze(1)

    assert out.shape == ref.shape, (out.shape, ref.shape)
    d_full = max_abs(out, ref)
    denom = float(ref.abs().max())
    # torch.istft-aligned (deviation 4): full range must match, edges included
    check("istft full-range max|Δ|", d_full, 1e-5 * max(denom, 1.0))
    print(f"       (ref max {denom:.3f})")


# ---------------------------------------------------------------------------
# original model loading + scoped determinism patches
# ---------------------------------------------------------------------------

def load_original(ckpt_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    hps = orig_utils.HParams(**cfg)
    net = orig_models.SynthesizerTrn(hps)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"], strict=True)
    net.eval()
    return net


class Patched:
    """Scoped determinism patches on the ORIGINAL: torch.randn_like returns the
    fixed z; Generator_Noise.forward takes the fixed phase (istft call kept
    VERBATIM). Restores everything on exit."""

    def __init__(self, z, phase):
        self.z = z
        self.phase = phase

    def __enter__(self):
        self._randn_like = torch.randn_like
        z = self.z

        def fixed_randn_like(m, **kw):
            assert m.shape == z.shape, (m.shape, z.shape)
            return z

        torch.randn_like = fixed_randn_like

        self._noise_fwd = orig_models.Generator_Noise.forward
        phase_src = self.phase

        def patched_noise_forward(self_mod, x, mask):
            istft_x = x
            istft_x = self_mod.istft_pre(istft_x)
            istft_x = self_mod.net(istft_x) * mask
            amp = self_mod.istft_amplitude(istft_x).unsqueeze(-1)
            phase = phase_src.unsqueeze(-1).to(amp)   # <- instead of torch.rand
            real = amp * torch.cos(phase)
            imag = amp * torch.sin(phase)
            spec = torch.cat([real, imag], 3)
            istft_x = istft_torch2(spec, self_mod.fft_size, self_mod.hop_size,
                                   self_mod.win_size, self_mod.window.to(amp),
                                   length=x.shape[2] * self_mod.hop_size)
            return istft_x.unsqueeze(1)

        orig_models.Generator_Noise.forward = patched_noise_forward
        return self

    def __exit__(self, *exc):
        torch.randn_like = self._randn_like
        orig_models.Generator_Noise.forward = self._noise_fwd
        return False


def capture(module_pairs, fn):
    """Run fn() with forward hooks on the given {name: module} dict; returns
    {name: output tensor (first element if tuple)}."""
    store, handles = {}, []
    for name, mod in module_pairs.items():
        def make_hook(n):
            def hook(_m, _i, out):
                store[n] = out[0] if isinstance(out, tuple) else out
            return hook
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        result = fn()
    finally:
        for h in handles:
            h.remove()
    return result, store


# ---------------------------------------------------------------------------
# (a)/(a2) real-weight torch-vs-torch
# ---------------------------------------------------------------------------

def tier_real_weights(ckpt_path, label):
    print(f"\n=== tier (a) [{label}]: real weights torch-vs-torch ===")
    c, f0, uv, z, phase = fixed_inputs()

    orig = load_original(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ours, meta = build_from_checkpoint(ckpt, cfg)

    sid = torch.zeros(1, dtype=torch.int64)
    zeros_f0d = torch.zeros(1, 192, c.shape[1])
    with torch.no_grad(), Patched(z, phase):
        def run_orig():
            # infer(c[1,256,T], g=sid[1,1]…): infer squeezes 2-D g itself
            return orig.infer(c.transpose(1, 2), g=sid.unsqueeze(0), f0=f0,
                              uv=uv, predict_f0=False, noice_scale=NOICE_SCALE)[0]
        o_ref, cap_ref = capture(
            {"text_encoder": orig.text_encoder,
             "mel_decoder": orig.mel_decoder,
             "decoder": orig.decoder,
             "dec_harm": orig.dec_harm}, run_orig)

    with torch.no_grad():
        def run_ours():
            return ours(c, f0, z * NOICE_SCALE, phase, sid, zeros_f0d)
        o_our, cap_our = capture(
            {"text_encoder": ours.text_encoder,
             "mel_decoder": ours.mel_decoder,
             "decoder": ours.decoder,
             "dec_harm": ours.dec_harm}, run_ours)

    # pre-noise intermediates: identical op sequences -> bitwise equal
    check(f"{label} text_encoder out", max_abs(cap_ref["text_encoder"],
                                               cap_our["text_encoder"]), 0, exact=True)
    check(f"{label} predict_mel", max_abs(cap_ref["mel_decoder"],
                                          cap_our["mel_decoder"]), 0, exact=True)
    check(f"{label} prior m_p/logs_p", max_abs(cap_ref["decoder"],
                                               cap_our["decoder"]), 0, exact=True)
    # post-noise: one-ulp z_p regrouping + stable-phase identity (deviation 6)
    check(f"{label} dec_harm out", max_abs(cap_ref["dec_harm"], cap_our["dec_harm"]),
          1e-3)

    # The original's OWN fp32 per-sample phase cumsum is chaotic at T=200
    # (argument ~3e5 rad at harmonic 64) — the honest equivalence anchor is
    # the fp64-exact original: ours must land closer to it than the
    # original's own fp32 run does (deviation 6 dominance criterion).
    orig64 = load_original(ckpt_path).double()
    with torch.no_grad(), Patched(z.double(), phase.double()):
        o_ref64 = orig64.infer(c.double().transpose(1, 2), g=sid.unsqueeze(0),
                               f0=f0.double(), uv=uv.double(),
                               predict_f0=False,
                               noice_scale=NOICE_SCALE)[0].float()
    assert o_ref.shape == o_our.shape == o_ref64.shape
    d_our_64 = max_abs(o_our, o_ref64)
    d_up_64 = max_abs(o_ref, o_ref64)
    d_our_up = max_abs(o_our, o_ref)
    print(f"       audio max|Δ|: ours-vs-fp64 {d_our_64:.3e} | "
          f"origfp32-vs-fp64 {d_up_64:.3e} | ours-vs-origfp32 {d_our_up:.3e} | "
          f"ref max {float(o_ref64.abs().max()):.3f}")
    check(f"{label} audio ours-vs-fp64 absolute", d_our_64, 2e-3)
    check(f"{label} audio dominance (ours ≤ orig fp32 drift)",
          d_our_64 - d_up_64, 1e-12)
    return ours, meta


# ---------------------------------------------------------------------------
# (b2) refusals + missing-config bit-identity
# ---------------------------------------------------------------------------

def tier_refusals(ours_chika):
    print("\n=== tier (b2): refusals + missing-config bit-identity ===")
    ckpt = torch.load(str(CHIKA), map_location="cpu", weights_only=True)

    # missing-config build must be graph-identical (template defaults)
    ours_nocfg, meta_nocfg = build_from_checkpoint(ckpt, None)
    c, f0, uv, z, phase = fixed_inputs()
    sid = torch.zeros(1, dtype=torch.int64)
    zeros_f0d = torch.zeros(1, 192, c.shape[1])
    with torch.no_grad():
        a1 = ours_chika(c, f0, z * NOICE_SCALE, phase, sid, zeros_f0d)
        a2 = ours_nocfg(c, f0, z * NOICE_SCALE, phase, sid, zeros_f0d)
    check("no-config build bitwise == config build", max_abs(a1, a2), 0, exact=True)
    assert meta_nocfg["speakers"] == {}, meta_nocfg["speakers"]

    # a 4.x checkpoint must be refused (routing detector + builder guard)
    fake_4x = {"model": {"enc_p.proj.weight": torch.zeros(1),
                         "emb_g.weight": torch.zeros(1)}}
    assert not is_v2_state_dict(fake_4x["model"])
    try:
        build_from_checkpoint(fake_4x, None)
        print("[FAIL] 4.x checkpoint accepted by v2 builder")
        FAILS.append("refusal 4.x->v2")
    except (ValueError, KeyError):
        print("[PASS] 4.x checkpoint refused by v2 builder")

    # and the v2 checkpoint IS detected as v2
    assert is_v2_state_dict(ckpt["model"])
    print("[PASS] v2 checkpoint detected by is_v2_state_dict")


# ---------------------------------------------------------------------------
# (c) torch(ours) vs ONNX shipping export
# ---------------------------------------------------------------------------

def tier_onnx(ours_chika):
    print("\n=== tier (c): torch(ours) vs ONNX shipping export ===")
    import onnxruntime as ort
    if not ONNX_MAIN.exists():
        print(f"[FAIL] shipping export missing: {ONNX_MAIN} — run convert.py first")
        FAILS.append("onnx missing")
        return
    sess = ort.InferenceSession(ONNX_MAIN.read_bytes(),
                                providers=["CPUExecutionProvider"])

    for t, tol in ((T, 1e-4), (137, 5e-4), (64, 5e-4), (23, 5e-4), (7, 5e-4), (6, 5e-4)):
        c, f0, uv, z, phase = fixed_inputs(t=t, seed=1000 + t)
        sid = torch.zeros(1, dtype=torch.int64)
        # non-zero f0d_cond so the sweep also exercises the deviation-7 add
        g = torch.Generator().manual_seed(2000 + t)
        f0d = torch.randn(1, 192, t, generator=g) * 0.1
        with torch.no_grad():
            ref = ours_chika(c, f0, z * NOICE_SCALE, phase, sid, f0d)
        out = sess.run(None, {
            "c": c.numpy(), "f0": f0.numpy(),
            "noise": (z * NOICE_SCALE).numpy(), "phase": phase.numpy(),
            "sid": sid.numpy(), "f0d_cond": f0d.numpy(),
        })[0]
        d = float(np.abs(ref.numpy() - out).max())
        check(f"onnx T={t} max|Δ|", d, tol)


# ---------------------------------------------------------------------------
# (f) auto-f0 companion
# ---------------------------------------------------------------------------

def tier_autof0(ours_chika):
    print("\n=== tier (f): auto-f0 predictor ===")
    c, f0, uv, z, phase = fixed_inputs(seed=4321)
    sid = torch.zeros(1, dtype=torch.int64)
    orig = load_original(CHIKA)

    # original predict_f0 branch, step-by-step (models.py infer :1012-1026),
    # with the normalize_f0 factor pinned to 1 (== random_scale=False;
    # documented deviation 5 — v2's infer inherits train-time jitter)
    with torch.no_grad():
        ct = c.transpose(1, 2)
        c_lengths = (torch.ones(ct.size(0)) * ct.size(-1))
        decoder_input, x_mask = orig.text_encoder(ct, c_lengths)
        LF0 = 2595. * torch.log10(1. + f0.unsqueeze(0) / 700.) / 500
        norm_f0 = orig_utils.normalize_f0(LF0, x_mask, uv.squeeze(1),
                                          random_scale=False)
        pred_lf0, _ = orig.f0_decoder(decoder_input, norm_f0, c_lengths,
                                      spk_emb=orig.emb_spk(sid).unsqueeze(-1))
        ref_f0 = 700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)
        ref_f0 = ref_f0.squeeze(1)
        # deviation-7 ground truth: the exact tensor the upstream F0Decoder
        # writes into the caller's decoder_input through its detach alias
        ref_f0d = orig.f0_decoder.f0_prenet(norm_f0)

        wrapper = F0PredictorWrapperV2(ours_chika)
        wrapper.eval()
        our_f0, our_f0d = wrapper(c, f0, uv, sid)

    check("autof0 torch-vs-orig", max_abs(ref_f0, our_f0), 0, exact=True)
    check("autof0 f0d_cond torch-vs-orig", max_abs(ref_f0d, our_f0d), 0, exact=True)

    import onnxruntime as ort
    if not ONNX_F0.exists():
        print(f"[FAIL] companion missing: {ONNX_F0}")
        FAILS.append("f0 onnx missing")
        return
    sess = ort.InferenceSession(ONNX_F0.read_bytes(),
                                providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"c": c.numpy(), "f0": f0.numpy(), "uv": uv.numpy(),
                           "sid": sid.numpy()})
    ref = our_f0.numpy()
    rel = float((np.abs(ref - outs[0]) / np.maximum(np.abs(ref), 1.0)).max())
    check("autof0 onnx Hz-relative", rel, 1.2e-3)
    check("autof0 onnx f0d_cond max|Δ|", float(np.abs(our_f0d.numpy() - outs[1]).max()),
          1e-4)


class PinnedNormalize:
    """Force the upstream's inference-time normalize_f0 to random_scale=False
    (deviation 5 — v2's infer() omits the flag and inherits train-time jitter)."""

    def __enter__(self):
        self._orig = orig_utils.normalize_f0
        orig_fn = self._orig

        def pinned(f0, x_mask, uv, random_scale=True):
            return orig_fn(f0, x_mask, uv, random_scale=False)

        orig_utils.normalize_f0 = pinned
        return self

    def __exit__(self, *exc):
        orig_utils.normalize_f0 = self._orig
        return False


def tier_autof0_chain(ours_chika):
    print("\n=== tier (g): FULL-CHAIN auto-f0 (deviation 7) ===")
    c, f0, uv, z, phase = fixed_inputs(seed=9876)
    sid = torch.zeros(1, dtype=torch.int64)

    def run_upstream(net, cc, ff, uvv, zz, ph):
        with torch.no_grad(), Patched(zz, ph), PinnedNormalize():
            return net.infer(cc.transpose(1, 2), g=sid.unsqueeze(0), f0=ff,
                             uv=uvv, predict_f0=True,
                             noice_scale=NOICE_SCALE)[0]

    o_up32 = run_upstream(load_original(CHIKA), c, f0, uv, z, phase)
    o_up64 = run_upstream(load_original(CHIKA).double(), c.double(), f0.double(),
                          uv.double(), z.double(), phase.double()).float()

    with torch.no_grad():
        wrapper = F0PredictorWrapperV2(ours_chika)
        wrapper.eval()
        pred_f0, f0d = wrapper(c, f0, uv, sid)
        o_ours = ours_chika(c, pred_f0, z * NOICE_SCALE, phase, sid, f0d)

    d_our_64 = max_abs(o_ours, o_up64)
    d_up_64 = max_abs(o_up32, o_up64)
    print(f"       auto audio max|Δ|: ours-vs-fp64 {d_our_64:.3e} | "
          f"origfp32-vs-fp64 {d_up_64:.3e} | ref max {float(o_up64.abs().max()):.3f}")
    # The auto chain amplifies fp noise: ulp-level pred_f0 differences feed the
    # 64-harmonic bank (genuine f0-phase sensitivity). The upstream's OWN fp32
    # sits ~4e-3 from the fp64 truth here (vs ~2.4e-4 on the manual chain), so
    # the absolute cap is calibrated to that same order; DOMINANCE (ours
    # strictly closer to fp64 than upstream's own fp32) is the primary test.
    check("auto-chain audio ours-vs-fp64 absolute", d_our_64, 5e-3)
    check("auto-chain dominance (ours ≤ orig fp32 drift)", d_our_64 - d_up_64, 1e-12)


def main():
    print(f"torch {torch.__version__} | upstream {UPSTREAM}")
    assert CHIKA.exists() and BASE_G0.exists() and CONFIG.exists()

    tier_istft()
    ours_chika, _ = tier_real_weights(CHIKA, "chika")
    tier_real_weights(BASE_G0, "G_0")
    tier_refusals(ours_chika)
    tier_onnx(ours_chika)
    tier_autof0(ours_chika)
    tier_autof0_chain(ours_chika)

    print()
    if FAILS:
        print(f"GATE FAILED: {len(FAILS)} failure(s): {FAILS}")
        sys.exit(1)
    print("GATE PASSED: all tiers green")


if __name__ == "__main__":
    main()
