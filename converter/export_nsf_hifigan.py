# export_nsf_hifigan.py — pretrain NSF-HiFiGAN vocoder -> "nsf_hifigan.onnx"
#
# One-time aux export (same pattern as export_rmvpe.py): the SAME pretrained
# checkpoint serves both the 4.1 enhancer (modules/enhancer.py) and the
# diffusion vocoder (diffusion/vocoder.py), so this one graph covers both.
#
# Input : "mel" f32 [1, num_mels=128, T]  ln-mel, nvSTFT semantics (natural
#                                         log of clamp(mel_filters @ |STFT|,
#                                         1e-5); Rust side: inference/mel.rs)
#         "f0"  f32 [1, T]                Hz at frame rate (hop 512 @ 44100)
# Output: "audio" f32 [1, 1, T*512]
#
# Also writes next to the onnx:
#   nsf_hifigan.json      — sidecar (type/sample_rate/hop_size/num_mels/n_fft/
#                           win_size/fmin/fmax/mel_filters)
#   nsf_hifigan_mel.npy   — [128, 1025] f32 librosa slaney mel filterbank
#                           (librosa.filters.mel defaults htk=False,
#                           norm='slaney' — exactly nvSTFT.py:93)
#
# SineGen randomness (rand_ini + uv-gated noise) stays in-graph;
# --deterministic zeroes both (numerical gate builds only — gate1 uses it).
#
# Usage:
#   .venv\Scripts\python.exe export_nsf_hifigan.py
#       --model <pretrain model file> [--config <config.json>]
#       --outdir <dir> [--deterministic]
#
# torch.onnx.export: dynamo=False, opset 17 (project-wide rule).

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from librosa.filters import mel as librosa_mel_fn  # noqa: E402

from architectures import nsf_hifigan_gen  # noqa: E402

DEFAULT_MODEL = r"D:\MyDev\so-vits-svc\so-vits-svc\pretrain\nsf_hifigan\model"

ONNX_NAME = "nsf_hifigan.onnx"
JSON_NAME = "nsf_hifigan.json"
MEL_NPY_NAME = "nsf_hifigan_mel.npy"

EXPORT_T = 100  # tracing length (frames); T is a dynamic axis


def load_config(model_path, config_path=None):
    """Original load_config() semantics (vdecoder/nsf_hifigan/models.py:29-36):
    config.json read from the checkpoint's directory unless given explicitly."""
    if config_path is None:
        config_path = os.path.join(os.path.split(model_path)[0], "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"找不到 NSF-HiFiGAN 配置文件: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def make_trace_inputs(num_mels, t):
    """Realistic tracing inputs: ln-mel range + an f0 ramp with an unvoiced
    span so both uv branches carry real values through the trace."""
    g = torch.Generator().manual_seed(20260704)
    mel = torch.rand(1, num_mels, t, generator=g) * 8.0 - 6.0  # ~ln-mel range
    f0 = torch.linspace(100.0, 400.0, t).unsqueeze(0)
    f0[:, t // 3:t // 3 + max(t // 10, 1)] = 0.0
    return mel, f0


def run_export(model_path, config_path, outdir, deterministic=False):
    h = load_config(model_path, config_path)
    model = nsf_hifigan_gen.build_from_checkpoint(model_path, h)
    nsf_hifigan_gen.set_deterministic(model, deterministic)

    os.makedirs(outdir, exist_ok=True)

    # mel filterbank — nvSTFT.py:93 verbatim call (librosa defaults:
    # htk=False slaney mel scale, norm='slaney')
    mel_basis = librosa_mel_fn(
        sr=h["sampling_rate"], n_fft=h["n_fft"], n_mels=h["num_mels"],
        fmin=h["fmin"], fmax=h["fmax"],
    ).astype(np.float32)
    assert mel_basis.shape == (h["num_mels"], h["n_fft"] // 2 + 1), mel_basis.shape
    mel_npy_path = os.path.join(outdir, MEL_NPY_NAME)
    np.save(mel_npy_path, mel_basis)
    print(f"mel filterbank {mel_basis.shape} {mel_basis.dtype} -> {mel_npy_path}")

    mel_in, f0_in = make_trace_inputs(h["num_mels"], EXPORT_T)
    with torch.no_grad():
        audio = model(mel_in, f0_in)
    assert audio.shape == (1, 1, EXPORT_T * h["hop_size"]), audio.shape

    onnx_path = os.path.join(outdir, ONNX_NAME)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (mel_in, f0_in),
            onnx_path,
            input_names=["mel", "f0"],
            output_names=["audio"],
            dynamic_axes={"mel": {2: "T"}, "f0": {1: "T"}, "audio": {2: "T_samples"}},
            opset_version=17,
            dynamo=False,
        )
    print(f"exported{' (deterministic)' if deterministic else ''} -> {onnx_path}")

    sidecar = {
        "type": "nsf_hifigan",
        "sample_rate": h["sampling_rate"],
        "hop_size": h["hop_size"],
        "num_mels": h["num_mels"],
        "n_fft": h["n_fft"],
        "win_size": h["win_size"],
        "fmin": h["fmin"],
        "fmax": h["fmax"],
        "mel_filters": MEL_NPY_NAME,
    }
    json_path = os.path.join(outdir, JSON_NAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    print(f"sidecar -> {json_path}")
    return onnx_path, json_path, mel_npy_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="pretrain NSF-HiFiGAN model file (torch ckpt with 'generator')")
    ap.add_argument("--config", default=None,
                    help="config.json (default: next to --model)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="zero SineGen rand_ini + noise (gate builds only)")
    args = ap.parse_args()

    torch.manual_seed(0)
    run_export(args.model, args.config, args.outdir, args.deterministic)


if __name__ == "__main__":
    main()
