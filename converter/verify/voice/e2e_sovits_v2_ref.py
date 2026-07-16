# -*- coding: utf-8 -*-
"""关卡 2 — SoVITS 4.0-v2 (VISinger2) E2E python reference (ORIGINAL single-segment
semantics vs Rust).

Reproduces the ORIGINAL 4.0-v2 single-segment path (infer_tool.slice_inference around
ONE non-silent segment + Svc.infer/get_unit_f0), transcribed faithfully, driving the
ORIGINAL torch models.SynthesizerTrn (real .pth weights, reconstructed template
config.json). Determinism: torch.rand / randn / randn_like monkeypatched to zeros —
kills z_p noise AND Generator_Noise's random phase == the Rust run with
{"noise_scale": 0.0, "debug_zero_noise": true}.

DOCUMENTED substitutions (same convention as e2e_sovits_ref.py — each certified
against the true ORIGINAL elsewhere):
  * ContentVec: inject contentvec_256l9.onnx (gate_contentvec.py) — v2 uses the same
    layer-9 + final_proj 256-d extraction as 4.0 (utils.get_hubert_content).
  * f0: the ORIGINAL v2 uses parselmouth; the app's house standard (like 4.0/4.1)
    is RMVPE — inject rmvpe_e2e.onnx + the 4.1 RMVPEF0Predictor post_process, which
    is exactly what src\\inference\\sovits.rs runs. The comparison therefore
    exercises the ARCHITECTURE + pipeline with identical f0 on both sides.
  * pipeline-internal 44100→16000 resample: scipy resample_poly, ONE wav16k for
    both f0 and content — the Rust choice (variant B of e2e_sovits_ref.py).
  * torch>=2 shim: the original's real-input torch.istft call needs
    view_as_complex (bit-identical layout; same family as 4.1's view_as_real).

NOTE the helpers RmvpeFront / sovits_post_process / torch_interp_nearest /
pad_array_center are COPIED VERBATIM from e2e_sovits_ref.py — importing that module
would collide on the upstream module names ('models'/'utils' resolve to whichever
repo got sys.path'd first). Both copies are pinned by their respective gates.

Run: converter\\.venv\\Scripts\\python.exe converter\\verify\\voice\\e2e_sovits_v2_ref.py \
        --input <44k mono wav> --pth <...\\G_73000.pth> --out <ref.wav>
Compare: e2e_compare_voice.py <ref.wav> <rust.wav>  (gate line: SNR > 40 dB)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # claim root before upstream's DEBUG basicConfig

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

V2_REPO = r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc"
CONVERTER = r"D:\MyDev\Utai_v2-dev\converter"
REPO_ROOT = r"D:\MyDev\Utai_v2-dev"

AUX = os.path.join(REPO_ROOT, "data", "models", "auxiliary")
RMVPE_ONNX = os.path.join(AUX, "rmvpe_e2e.onnx")
MEL_FILTERS = os.path.join(AUX, "rmvpe_mel_filters.npy")

sys.path.insert(0, CONVERTER)
sys.path.insert(0, V2_REPO)

import models as orig_models          # noqa: E402  4.0-v2 models.py
import utils as so_utils              # noqa: E402  4.0-v2 utils.py
from architectures import sovits_v4v2  # noqa: E402  converter port (config loader + meta)

torch.set_grad_enabled(False)
PAD_SECONDS = 0.5
NOISE_SCALE = 0.4
RMVPE_THRED = 0.05


def istft_torch2(spec_real_imag, n_fft, hop, win, window, length):
    """torch>=2 shim for the original's real-input torch.istft (bit-identical)."""
    return torch.istft(torch.view_as_complex(spec_real_imag.contiguous()),
                       n_fft, hop, win, window, True, length=length)


def _patch_noise_forward():
    """Replace Generator_Noise.forward with the verbatim body, istft shimmed for
    torch>=2. torch.rand is separately zeroed by ZeroNoise (phase = 0)."""
    def forward(self, x, mask):
        istft_x = x
        istft_x = self.istft_pre(istft_x)
        istft_x = self.net(istft_x) * mask
        amp = self.istft_amplitude(istft_x).unsqueeze(-1)
        phase = (torch.rand(amp.shape) * 2 * 3.14 - 3.14).to(amp)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        spec = torch.cat([real, imag], 3)
        istft_x = istft_torch2(spec, self.fft_size, self.hop_size, self.win_size,
                               self.window.to(amp), length=x.shape[2] * self.hop_size)
        return istft_x.unsqueeze(1)

    orig_models.Generator_Noise.forward = forward


class ZeroNoise:
    def __enter__(self):
        self._r, self._rl, self._rn = torch.rand, torch.randn_like, torch.randn

        def zrand(*size, **kw):
            return torch.zeros(*size, device=kw.get("device"), dtype=kw.get("dtype"))

        torch.rand = zrand
        torch.randn_like = lambda t, **kw: torch.zeros_like(t)
        torch.randn = zrand
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like, torch.randn = self._r, self._rl, self._rn


# ── copied verbatim from e2e_sovits_ref.py (see header NOTE) ──
class RmvpeFront:
    def __init__(self):
        self.sess = ort.InferenceSession(RMVPE_ONNX, providers=["CPUExecutionProvider"])
        self.mel = np.load(MEL_FILTERS).astype(np.float32)
        k = np.arange(1024)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * k / 1024.0)).astype(np.float32)

    def f0_100fps(self, wav16k, thred):
        x = np.asarray(wav16k, dtype=np.float32)
        n = len(x)
        if n < 513:
            x = np.pad(x, (0, 513 - n)); n = 513
        t_frames = 1 + n // 160
        padded = np.pad(x, (512, 512), mode="reflect")
        mag = np.empty((513, t_frames), dtype=np.float32)
        for t in range(t_frames):
            mag[:, t] = np.abs(np.fft.rfft(padded[t*160:t*160+1024] * self.window, n=1024))
        logmel = np.log(np.clip(self.mel @ mag, 1e-5, None)).astype(np.float32)
        f0 = self.sess.run(None, {"mel": logmel[None],
                                  "threshold": np.array([thred], np.float32)})[0]
        return np.asarray(f0, dtype=np.float32).reshape(-1)


def torch_interp_nearest(src, dst_len):
    src = np.asarray(src, dtype=np.float64)
    n = len(src)
    idx = np.minimum((np.arange(dst_len) * (n / dst_len)).astype(np.int64), n - 1)
    return src[idx]


def sovits_post_process(f0_100, pad_to, hop, sr):
    if np.all(f0_100 == 0):
        return np.zeros(pad_to, np.float32), np.zeros(pad_to, np.float32)
    f0 = torch_interp_nearest(f0_100, pad_to)
    uv = (f0 > 0.0).astype(np.float32)
    nz = np.nonzero(f0)[0]
    if nz.size == 0:
        return np.zeros(pad_to, np.float32), uv
    if nz.size == 1:
        return np.full(pad_to, f0[nz[0]], np.float32), uv
    time_org = hop / sr * nz
    time_frame = np.arange(pad_to) * hop / sr
    vals = f0[nz]
    f0i = np.interp(time_frame, time_org, vals, left=vals[0], right=vals[-1])
    return f0i.astype(np.float32), uv


def pad_array_center(arr, target_length):
    cur = len(arr)
    if cur >= target_length:
        return arr
    pad = target_length - cur
    return np.concatenate([np.zeros(pad // 2, arr.dtype), arr,
                           np.zeros(pad - pad // 2, arr.dtype)])
# ── end copied helpers ──


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    ck = torch.load(args.pth, map_location="cpu", weights_only=False)
    cfg, cfg_src = sovits_v4v2.load_sovits_v2_config(
        Path(args.pth), Path(args.config) if args.config else None)
    assert cfg is not None, "v2 E2E 需要 config.json（官方模板重建件）"
    _, meta = sovits_v4v2.build_from_checkpoint(ck, cfg)
    model_sr = meta["sample_rate"]
    assert sr == model_sr, f"input must be {model_sr} Hz mono (got {sr})"
    dim = meta["features_dim"]; hop = meta["hop_size"]
    print(f"[ref] {Path(args.pth).name}: v{meta['version']} dim={dim} hop={hop} "
          f"sr={model_sr} (config {cfg_src})")
    print(f"[ref] input n={len(audio)} rms={np.sqrt(np.mean(audio**2)):.4f}")

    # ORIGINAL v2 synthesizer (hps-driven ctor), strict load
    hps = so_utils.HParams(**cfg)
    orig = orig_models.SynthesizerTrn(hps)
    orig.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
    orig.eval()
    _patch_noise_forward()

    rmvpe = RmvpeFront()

    # single-segment slice_inference semantics: 0.5 s pad, infer, trim, length fit
    pad_native = int(model_sr * PAD_SECONDS)
    wav_m = np.concatenate([np.zeros(pad_native, np.float32),
                            audio.astype(np.float32),
                            np.zeros(pad_native, np.float32)])
    n_frames = len(wav_m) // hop
    wav16k = resample_poly(wav_m, 16000, model_sr).astype(np.float32)

    f0_100 = rmvpe.f0_100fps(wav16k, RMVPE_THRED)
    f0, uv = sovits_post_process(f0_100, n_frames, hop, model_sr)

    cv_sess = ort.InferenceSession(
        os.path.join(AUX, "contentvec_256l9.onnx"), providers=["CPUExecutionProvider"])
    c_raw = cv_sess.run(None, {"waveform": wav16k[None]})[0]          # [1,T,256]
    c_hub = torch.from_numpy(c_raw[0].T.copy())                        # [256,T]
    # v2 utils.repeat_expand_2d = the 'left' variant, no mode argument
    c = so_utils.repeat_expand_2d(c_hub, n_frames).unsqueeze(0)        # [1,256,nf]

    f0t = torch.from_numpy(f0)[None].float()
    uvt = torch.from_numpy(uv)[None].float()
    sid = torch.LongTensor([0]).unsqueeze(0)

    with ZeroNoise():
        out = orig.infer(c, g=sid, f0=f0t, uv=uvt, predict_f0=False,
                         noice_scale=NOISE_SCALE)[0][0, 0].cpu().numpy()

    trimmed = out[pad_native:-pad_native]
    per_length = int(np.ceil(len(audio)))
    y = pad_array_center(trimmed.astype(np.float32), per_length)
    print(f"[ref] out n={len(y)} peak={np.abs(y).max():.4f} "
          f"rms={np.sqrt(np.mean(y**2)):.4f}")
    sf.write(args.out, y, model_sr, subtype="FLOAT")
    print("[ref] done")


if __name__ == "__main__":
    main()
