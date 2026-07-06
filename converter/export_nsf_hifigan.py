# export_nsf_hifigan.py — NSF-HiFiGAN vocoder checkpoint -> "<stem>.onnx"
#
# Two callers (S40):
#   - one-time aux export (original S36 role, default stem "nsf_hifigan"):
#     the SAME pretrained checkpoint serves both the 4.1 enhancer
#     (modules/enhancer.py) and the diffusion vocoder (diffusion/vocoder.py)
#   - the vocoder RESOURCE import/convert chain (src-tauri models/convert.rs::
#     convert_vocoder_to_onnx): user-trained / community checkpoints ->
#     data/models/nsf_hifigan/<stem>.{onnx,json} + <stem>_mel.npy
#     ("_mel" not ".mel": registry scan probes `<stem>.npy` as an RVC-style
#     index — a dotted suffix would collide with that namespace)
#
# Input : "mel" f32 [1, num_mels=128, T]  ln-mel, nvSTFT semantics (natural
#                                         log of clamp(mel_filters @ |STFT|,
#                                         1e-5); Rust side: inference/mel.rs)
#         "f0"  f32 [1, T]                Hz at frame rate (hop 512 @ 44100)
# Output: "audio" f32 [1, 1, T*512]
#
# Checkpoint layouts accepted (architectures/nsf_hifigan_gen.load_generator_state):
#   {'generator': sd} deploy format | SingingVocoders lightning training ckpt
#   ({'state_dict': {'generator.*'}}). mini_nsf (PC-NSF) configs are REJECTED
#   (一期 classic-only — the converter architecture has no source_conv branch).
#
# Also writes next to the onnx:
#   <stem>.json      — sidecar (type/sample_rate/hop_size/num_mels/n_fft/
#                      win_size/fmin/fmax/mel_filters)
#   <stem>_mel.npy   — [num_mels, n_fft//2+1] f32 librosa slaney mel filterbank
#                      (librosa.filters.mel defaults htk=False, norm='slaney'
#                      — exactly nvSTFT.py:93)
#
# SineGen randomness (rand_ini + uv-gated noise) stays in-graph;
# --deterministic zeroes both (numerical gate builds only — gate1 uses it).
#
# Self-check (default ON — this script runs on user machines at import time):
# a deterministic twin export is compared torch-vs-ORT (<=1e-4, corr>0.9999,
# finite) at two T, then discarded; the live graph additionally proves its
# in-graph noise is alive (two runs must differ). export_diffusion.py
# _check_and_sanity precedent; ORT sessions are bytes-loaded (house rule —
# Windows CJK paths).
#
# Usage:
#   .venv\Scripts\python.exe export_nsf_hifigan.py
#       --model <ckpt file> [--config <config.json>]
#       --outdir <dir> [--stem <name>] [--deterministic]
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

# default-stem artifact names — the aux deployment contract; gate1_nsf_hifigan.py
# references these constants and the no---stem CLI must keep producing them
DEFAULT_STEM = "nsf_hifigan"
ONNX_NAME = "nsf_hifigan.onnx"
JSON_NAME = "nsf_hifigan.json"
MEL_NPY_NAME = "nsf_hifigan_mel.npy"

EXPORT_T = 100  # tracing length (frames); T is a dynamic axis

SELFCHECK_TOL = 1e-4     # torch-vs-ORT max_abs (gate1_nsf_hifigan TOL_ONNX axis)
SELFCHECK_MIN_CORR = 0.9999


def _names(stem):
    return f"{stem}.onnx", f"{stem}.json", f"{stem}_mel.npy"


def load_config(model_path, config_path=None):
    """Original load_config() semantics (vdecoder/nsf_hifigan/models.py:29-36):
    config.json read from the checkpoint's directory unless given explicitly.
    Rejects the mini_nsf (PC-NSF) architecture variant — 一期 classic-only."""
    if config_path is None:
        config_path = os.path.join(os.path.split(model_path)[0], "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"找不到 NSF-HiFiGAN 配置文件: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        h = json.load(f)
    if h.get("mini_nsf"):
        raise ValueError(
            "暂不支持 PC-NSF（mini_nsf）架构的声码器——目前仅支持经典 NSF-HiFiGAN"
            "（如 2022.12 / 2024.02 社区声码器及其微调产物）"
        )
    if float(h.get("noise_sigma") or 0) > 0:
        raise ValueError("暂不支持 noise_sigma > 0 的声码器配置（经典 NSF-HiFiGAN 恒为 0）")
    return h


def make_trace_inputs(num_mels, t, seed=20260704):
    """Realistic tracing inputs: ln-mel range + an f0 ramp with an unvoiced
    span so both uv branches carry real values through the trace."""
    g = torch.Generator().manual_seed(seed)
    mel = torch.rand(1, num_mels, t, generator=g) * 8.0 - 6.0  # ~ln-mel range
    f0 = torch.linspace(100.0, 400.0, t).unsqueeze(0)
    f0[:, t // 3:t // 3 + max(t // 10, 1)] = 0.0
    return mel, f0


def _export_graph(model, num_mels, hop_size, onnx_path):
    mel_in, f0_in = make_trace_inputs(num_mels, EXPORT_T)
    with torch.no_grad():
        audio = model(mel_in, f0_in)
    assert audio.shape == (1, 1, EXPORT_T * hop_size), audio.shape
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


def _ort_session(onnx_path):
    import onnxruntime as ort

    # bytes-load per house rule (Windows CJK paths break the path overload)
    with open(onnx_path, "rb") as f:
        return ort.InferenceSession(f.read(), providers=["CPUExecutionProvider"])


def _selfcheck(model, h, onnx_path, outdir, stem, deterministic):
    """torch-vs-ORT numeric parity on a deterministic graph + liveness of the
    shipped graph's in-graph noise. Raises ValueError (Chinese) on any failure
    — the Rust import chain surfaces it verbatim."""
    num_mels, hop = h["num_mels"], h["hop_size"]

    # (1) numeric parity — on a deterministic graph (the live graph's noise
    # makes a direct compare meaningless). deterministic export = the shipped
    # graph itself; live export = a twin that is deleted afterwards.
    if deterministic:
        det_path = onnx_path
    else:
        det_path = os.path.join(outdir, f"{stem}.selfcheck.onnx")
        nsf_hifigan_gen.set_deterministic(model, True)
        try:
            _export_graph(model, num_mels, hop, det_path)
        finally:
            nsf_hifigan_gen.set_deterministic(model, False)
    try:
        sess = _ort_session(det_path)
        nsf_hifigan_gen.set_deterministic(model, True)
        try:
            for t in (EXPORT_T, 57):
                mel_in, f0_in = make_trace_inputs(num_mels, t, seed=926 + t)
                with torch.no_grad():
                    ref = model(mel_in, f0_in).numpy()
                got = sess.run(None, {"mel": mel_in.numpy(), "f0": f0_in.numpy()})[0]
                if not np.all(np.isfinite(got)):
                    raise ValueError(f"导出自检失败: ORT 输出含非有限值 (T={t})")
                diff = float(np.abs(got - ref).max())
                corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
                if diff > SELFCHECK_TOL or corr < SELFCHECK_MIN_CORR:
                    raise ValueError(
                        f"导出自检失败: torch/ORT 波形不一致 "
                        f"(T={t}, max_abs={diff:.3e}, corr={corr:.6f})"
                    )
                print(f"selfcheck T={t}: max_abs {diff:.3e} corr {corr:.6f} OK")
        finally:
            nsf_hifigan_gen.set_deterministic(model, deterministic)
        del sess
    finally:
        if det_path != onnx_path:
            try:
                os.remove(det_path)
            except OSError:
                pass

    # (2) the SHIPPED graph: dynamic-T sanity + (live graphs) noise liveness
    sess = _ort_session(onnx_path)
    mel_in, f0_in = make_trace_inputs(num_mels, 20, seed=1042)
    feeds = {"mel": mel_in.numpy(), "f0": f0_in.numpy()}
    a = sess.run(None, feeds)[0]
    if a.shape != (1, 1, 20 * hop) or not np.all(np.isfinite(a)):
        raise ValueError(f"导出自检失败: 动态长度输出异常 (shape={a.shape})")
    if not deterministic:
        b = sess.run(None, feeds)[0]
        if np.array_equal(a, b):
            raise ValueError("导出自检失败: 图内噪声未生效（两次运行输出完全相同）")


def run_export(model_path, config_path, outdir, deterministic=False,
               stem=DEFAULT_STEM, selfcheck=True):
    h = load_config(model_path, config_path)
    model = nsf_hifigan_gen.build_from_checkpoint(model_path, h)
    nsf_hifigan_gen.set_deterministic(model, deterministic)

    os.makedirs(outdir, exist_ok=True)
    onnx_name, json_name, mel_npy_name = _names(stem)

    # mel filterbank — nvSTFT.py:93 verbatim call (librosa defaults:
    # htk=False slaney mel scale, norm='slaney')
    mel_basis = librosa_mel_fn(
        sr=h["sampling_rate"], n_fft=h["n_fft"], n_mels=h["num_mels"],
        fmin=h["fmin"], fmax=h["fmax"],
    ).astype(np.float32)
    assert mel_basis.shape == (h["num_mels"], h["n_fft"] // 2 + 1), mel_basis.shape
    mel_npy_path = os.path.join(outdir, mel_npy_name)
    np.save(mel_npy_path, mel_basis)
    print(f"mel filterbank {mel_basis.shape} {mel_basis.dtype} -> {mel_npy_path}")

    onnx_path = os.path.join(outdir, onnx_name)
    _export_graph(model, h["num_mels"], h["hop_size"], onnx_path)
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
        "mel_filters": mel_npy_name,
    }
    json_path = os.path.join(outdir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    print(f"sidecar -> {json_path}")

    if selfcheck:
        _selfcheck(model, h, onnx_path, outdir, stem, deterministic)
        print("selfcheck OK")
    return onnx_path, json_path, mel_npy_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="NSF-HiFiGAN ckpt ({'generator':sd} deploy format or a "
                         "SingingVocoders lightning training ckpt)")
    ap.add_argument("--config", default=None,
                    help="config.json (default: next to --model)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stem", default=DEFAULT_STEM,
                    help="output artifact stem: <stem>.onnx/<stem>.json/"
                         "<stem>_mel.npy (default keeps the aux names)")
    ap.add_argument("--deterministic", action="store_true",
                    help="zero SineGen rand_ini + noise (gate builds only)")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip the torch-vs-ORT self-check (gate scripts run "
                         "their own deeper comparisons)")
    args = ap.parse_args()

    torch.manual_seed(0)
    run_export(args.model, args.config, args.outdir, args.deterministic,
               stem=args.stem, selfcheck=not args.no_selfcheck)


if __name__ == "__main__":
    main()
