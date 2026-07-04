"""so-vits-svc companion-asset converter: shallow-diffusion model (.pt+.yaml).

Converts a so-vits-svc 4.1 diffusion checkpoint (trained by train_diff.py,
paired with a diffusion config.yaml) into the Utai runtime layout
(quality-path contract §2.1-2.3):

    <outdir>/encoder.onnx    condition embedding (units/f0/volume[/spk_mix]
                             -> cond [1,n_hidden,T], already transposed)
    <outdir>/denoiser.onnx   WaveNet eps-net single step (x/time f32/cond
                             -> noise_pred); the sampling loop lives in Rust
    <outdir>/diffusion.json  sidecar — schedule facts (timesteps/max_beta/
                             spec_min/spec_max/k_step_max), encoder dims,
                             sampler defaults, speaker map

<outdir> is conventionally `<model-stem>.diffusion` next to the main SoVITS
onnx. The architecture port + all validation/refusal logic live in
architectures/sovits_diffusion.py; equivalence against the ORIGINAL repo is
gated by verify/voice/gate1_diffusion.py.

Usage:
    python export_diffusion.py --input diff.pt [--config diff.yaml] \
        --outdir "<models>/sovits/<stem>.diffusion" [--out-dims 128]

yaml resolution when --config is omitted: <stem>.yaml > the single *.yaml in
the directory > config.yaml > Chinese refusal (contract §3.5).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Windows console default codepage chokes on Chinese model paths/names.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from architectures import sovits_diffusion  # noqa: E402

EXPORT_T = 100  # dummy frame count; both graphs are dynamic in T (gate sweeps)

# contract §2.3 — the sidecar carries EXACTLY these keys (order preserved).
SIDECAR_KEYS = (
    "type", "encoder", "encoder_out_channels", "sample_rate", "block_size",
    "n_layers", "n_chans", "n_hidden", "timesteps", "k_step_max",
    "schedule", "max_beta", "spec_min", "spec_max", "n_spk", "speakers",
    "use_pitch_aug", "infer_method", "infer_speedup", "unit_interpolate_mode",
)


def _ort_session(path):
    """ORT session from BYTES — python onnxruntime cannot open Chinese
    session paths on Windows (Rust's ort uses wide strings, unaffected)."""
    import onnxruntime as ort
    return ort.InferenceSession(Path(path).read_bytes(),
                                providers=["CPUExecutionProvider"])


def _check_and_sanity(onnx_path, torch_fn, feeds_fn, out_name):
    """onnx.checker + ORT-vs-torch catastrophe net at the export T AND at a
    different T (proves the trace stayed dynamic — a baked graph fails here
    at export time instead of at runtime in Rust). The rigorous 1e-4/5e-4
    parity gates live in gate1_diffusion.py."""
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {onnx_path}", file=sys.stderr)

    sess = _ort_session(onnx_path)
    for t_frames in (EXPORT_T, 57):
        feeds = feeds_fn(t_frames)
        with torch.no_grad():
            want = torch_fn(**{k: torch.from_numpy(v) for k, v in feeds.items()})
        got = sess.run(None, feeds)[0]
        d = float(np.abs(got - want.numpy()).max())
        if not np.isfinite(got).all() or d > 1e-3:
            raise ValueError(
                f"导出自检失败：{Path(onnx_path).name} T={t_frames} 的 ORT 输出与 "
                f"torch 差 {d:.3e}（阈值 1e-3）或含非有限值")
        print(f"  ORT sanity {Path(onnx_path).name} T={t_frames}: "
              f"{out_name} {got.shape}, max_abs_diff vs torch = {d:.3e}")


def export_diffusion_assets(input_path, outdir, config_path=None, out_dims=128):
    """Full conversion: load + validate + export both graphs + write the
    sidecar. Returns the sidecar dict."""
    input_path = Path(input_path)
    outdir = Path(outdir)

    cfg, cfg_path = sovits_diffusion.load_diffusion_config(input_path, config_path)
    print(f"config: {cfg_path}")

    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)
    model, meta = sovits_diffusion.build_from_checkpoint(checkpoint, cfg, out_dims=out_dims)
    print(f"model: encoder={meta['encoder']} ({meta['encoder_out_channels']}d), "
          f"n_layers={meta['n_layers']}, n_chans={meta['n_chans']}, "
          f"n_hidden={meta['n_hidden']}, timesteps={meta['timesteps']}, "
          f"k_step_max={meta['k_step_max']}, n_spk={meta['n_spk']}, "
          f"use_pitch_aug={meta['use_pitch_aug']}, out_dims={meta['out_dims']}")

    outdir.mkdir(parents=True, exist_ok=True)
    encoder_path = outdir / "encoder.onnx"
    denoiser_path = outdir / "denoiser.onnx"

    n_hidden, m_dims, n_spk = meta["n_hidden"], meta["out_dims"], meta["n_spk"]
    dim = meta["encoder_out_channels"]

    # --- encoder.onnx (contract §2.1) ---
    enc = sovits_diffusion.EncoderExport(model)
    gen = torch.Generator().manual_seed(20260704)

    def enc_feeds(t):
        feeds = {
            "units": torch.randn(1, t, dim, generator=gen).numpy(),
            "f0": (150.0 + 250.0 * torch.rand(1, t, generator=gen)).numpy(),
            "volume": (0.5 * torch.rand(1, t, generator=gen)).numpy(),
        }
        if n_spk > 1:
            mix = torch.rand(t, n_spk, generator=gen)
            feeds["spk_mix"] = (mix / mix.sum(dim=1, keepdim=True)).numpy()
        return feeds

    dummy = {k: torch.from_numpy(v) for k, v in enc_feeds(EXPORT_T).items()}
    input_names = ["units", "f0", "volume"] + (["spk_mix"] if n_spk > 1 else [])
    dynamic_axes = {"units": {1: "n_frames"}, "f0": {1: "n_frames"},
                    "volume": {1: "n_frames"}, "cond": {2: "n_frames"}}
    if n_spk > 1:
        dynamic_axes["spk_mix"] = {0: "n_frames"}
    torch.onnx.export(
        enc,
        tuple(dummy[k] for k in input_names),
        str(encoder_path),
        input_names=input_names,
        output_names=["cond"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    _check_and_sanity(encoder_path, enc, enc_feeds, "cond")

    # --- denoiser.onnx (contract §2.2) ---
    den = sovits_diffusion.DenoiserExport(model)

    def den_feeds(t):
        return {
            "x": torch.randn(1, 1, m_dims, t, generator=gen).numpy(),
            # FLOAT time — DPM/UniPC feed non-integer t
            "time": np.array([meta["timesteps"] / 2.0], dtype=np.float32),
            "cond": torch.randn(1, n_hidden, t, generator=gen).numpy(),
        }

    dummy_d = {k: torch.from_numpy(v) for k, v in den_feeds(EXPORT_T).items()}
    torch.onnx.export(
        den,
        (dummy_d["x"], dummy_d["time"], dummy_d["cond"]),
        str(denoiser_path),
        input_names=["x", "time", "cond"],
        output_names=["noise_pred"],
        dynamic_axes={"x": {3: "n_frames"}, "cond": {2: "n_frames"},
                      "noise_pred": {3: "n_frames"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    _check_and_sanity(denoiser_path, den, den_feeds, "noise_pred")

    # --- diffusion.json (contract §2.3, EXACT schema) ---
    sidecar = {k: meta[k] for k in SIDECAR_KEYS}
    sidecar["files"] = {"encoder": "encoder.onnx", "denoiser": "denoiser.onnx"}
    sidecar_path = outdir / "diffusion.json"
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return sidecar


def main():
    parser = argparse.ArgumentParser(
        description="Convert a so-vits-svc diffusion .pt (+config yaml) to "
                    "encoder.onnx + denoiser.onnx + diffusion.json")
    parser.add_argument("--input", type=str, required=True,
                        help="diffusion model .pt (train_diff.py output)")
    parser.add_argument("--config", type=str, default=None,
                        help="diffusion config .yaml (default: resolve next to the .pt)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="output directory (conventionally <stem>.diffusion)")
    parser.add_argument("--out-dims", type=int, default=128,
                        help="mel bins = vocoder num_mels (nsf-hifigan family: 128)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 找不到输入文件: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        export_diffusion_assets(input_path, args.outdir,
                                config_path=args.config, out_dims=args.out_dims)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Converted diffusion model: {input_path} -> {args.outdir}")


if __name__ == "__main__":
    main()
