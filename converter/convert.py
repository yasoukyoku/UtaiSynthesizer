"""Model converter: .pth/.ckpt → .onnx

Converts RVC, SoVITS, BSRoformer, MelBandRoformer, and MDX23C checkpoints
to ONNX format for inference via ONNX Runtime in the Rust backend.

Requires PyTorch (CPU is sufficient for conversion).

Usage:
    python convert.py --input model.pth  --output model.onnx --type rvc
    python convert.py --input model.pth  --output model.onnx --type sovits
    python convert.py --input model.ckpt --output model.onnx --type bs_roformer
    python convert.py --input model.ckpt --output model.onnx --type mel_band_roformer [--config model.yaml]
    python convert.py --input model.ckpt --output model.onnx --type mdx23c --config model.yaml
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from architectures.rvc_v2 import build_from_checkpoint as build_rvc_v2
from architectures.sovits_v4 import build_from_checkpoint as build_sovits_v4
from architectures.bs_roformer import load_from_checkpoint as load_bs_roformer, detect_config
from architectures.mel_band_roformer import (
    load_from_checkpoint as load_mel_band_roformer,
    detect_config as detect_mel_config,
)
from architectures.mdx23c import (
    load_from_checkpoint as load_mdx23c,
    detect_config as detect_mdx23c_config,
)
from architectures.htdemucs import (
    load_from_checkpoint as load_htdemucs,
    detect_config as detect_htdemucs_config,
)


def convert_rvc(input_path: Path, output_path: Path):
    """Convert RVC v2 .pth model to ONNX."""
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        print("ERROR: checkpoint has no 'config' key", file=sys.stderr)
        sys.exit(1)

    if isinstance(config, (list, tuple)):
        sample_rate = config[17] if len(config) > 17 else 40000
        n_speakers = config[15] if len(config) > 15 else 1
    else:
        sample_rate = config.get("sample_rate", 40000)
        n_speakers = config.get("n_speakers", 1)

    model = build_rvc_v2(checkpoint)

    seq_len = 50
    phone = torch.randn(1, seq_len, 768)
    phone_lengths = torch.tensor([seq_len], dtype=torch.long)
    pitch = torch.randint(0, 256, (1, seq_len))
    pitchf = torch.randn(1, seq_len)
    sid = torch.zeros(1, dtype=torch.long)
    noise_scale = torch.tensor([0.667], dtype=torch.float32)

    torch.onnx.export(
        model,
        (phone, phone_lengths, pitch, pitchf, sid, noise_scale),
        str(output_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid", "noise_scale"],
        output_names=["audio"],
        dynamic_axes={
            "phone": {1: "seq_len"},
            "pitch": {1: "seq_len"},
            "pitchf": {1: "seq_len"},
            "audio": {2: "audio_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}", file=sys.stderr)

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "rvc",
        "version": "v2",
        "sample_rate": sample_rate,
        "features_dim": 768,
        "n_speakers": n_speakers,
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted RVC: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Sample rate: {sample_rate}, Speakers: {n_speakers}")


def convert_sovits(input_path: Path, output_path: Path):
    """Convert SoVITS 4.0 .pth model to ONNX."""
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    if "model" not in checkpoint:
        print("ERROR: checkpoint has no 'model' key", file=sys.stderr)
        sys.exit(1)

    model = build_sovits_v4(checkpoint)

    seq_len = 50
    c = torch.randn(1, seq_len, 256)
    f0 = torch.abs(torch.randn(1, seq_len)) * 200 + 100
    uv = (f0 > 150).long()
    sid = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        model,
        (c, f0, uv, sid),
        str(output_path),
        input_names=["c", "f0", "uv", "sid"],
        output_names=["audio"],
        dynamic_axes={
            "c": {1: "seq_len"},
            "f0": {1: "seq_len"},
            "uv": {1: "seq_len"},
            "audio": {2: "audio_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}", file=sys.stderr)

    # Infer config from model
    ssl_dim = checkpoint["model"]["pre.weight"].shape[1]
    n_speakers = checkpoint["model"]["emb_g.weight"].shape[0]

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "sovits",
        "version": "4.0",
        "sample_rate": 44100,
        "features_dim": ssl_dim,
        "n_speakers": int(n_speakers),
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted SoVITS: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Sample rate: 44100, SSL dim: {ssl_dim}, Speakers: {n_speakers}")


def convert_bs_roformer(input_path: Path, output_path: Path):
    """Convert BSRoformer .ckpt model to ONNX."""
    config = detect_config(str(input_path))
    model = load_bs_roformer(str(input_path), config)

    stereo = config.get("stereo", False)
    audio_channels = 2 if stereo else 1
    n_fft = config.get("stft_n_fft", 2048)
    freq_bins = n_fft // 2 + 1
    fs = freq_bins * audio_channels

    dummy = torch.randn(1, fs, 256, 2)

    with torch.no_grad():
        test_out = model(dummy)
    print(f"Test inference OK: input {list(dummy.shape)} -> output {list(test_out.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["stft_repr"],
        output_names=["mask"],
        dynamic_axes={
            "stft_repr": {0: "batch", 2: "time_frames"},
            "mask": {0: "batch", 3: "time_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"stft_repr": dummy.numpy()})
    diff = abs(test_out.numpy() - ort_out[0]).max()
    print(f"ORT verification: max diff = {diff:.6e}")
    # Batch-axis self-check: a B=2 batch must equal two independent B=1 runs, item-for-item. Proves the
    # dynamic batch axis exported correctly (constant-folding didn't bake batch=1) AND that the Rust
    # unbind offset (j*N + stem) matches the real [B, N, ...] output layout. If this asserts, the batched
    # Rust path would silently corrupt output — fail loudly here at export time instead.
    dummy2 = torch.cat([dummy, dummy + 0.01], dim=0).numpy()
    ort_b2 = sess.run(None, {"stft_repr": dummy2})[0]
    s0 = sess.run(None, {"stft_repr": dummy.numpy()})[0]
    s1 = sess.run(None, {"stft_repr": (dummy + 0.01).numpy()})[0]
    b2_diff = max(abs(ort_b2[0] - s0[0]).max(), abs(ort_b2[1] - s1[0]).max())
    assert b2_diff < 1e-3, f"BATCH-AXIS EXPORT BROKEN: B=2 != 2x(B=1) (diff {b2_diff:.3e})"
    print(f"Batch-axis check passed: B=2 == 2x(B=1), diff = {b2_diff:.3e}")

    hop_length = config.get("stft_hop_length", 441)
    chunk_size = 131584  # ~3s at 44100Hz — keeps T≈299 frames, avoids O(T²) attention blowup

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "bs_roformer",
        "sample_rate": config.get("sample_rate", 44100),
        "stereo": stereo,
        "num_stems": config.get("num_stems", 1),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": config.get("stft_win_length", 2048),
        "freq_bins": freq_bins,
        "chunk_size": chunk_size,
        "num_overlap": 2,
        "batch_size": 1,
        "dynamic_batch": True,
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted BSRoformer: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stereo: {stereo}, Stems: {config.get('num_stems', 1)}, "
          f"FFT: {n_fft}, Hop: {hop_length}")


def convert_mel_band_roformer(input_path: Path, output_path: Path,
                              config_yaml: Path = None):
    """Convert MelBandRoformer .ckpt model to ONNX."""
    yaml_str = str(config_yaml) if config_yaml else None
    config = detect_mel_config(str(input_path), yaml_str)
    model = load_mel_band_roformer(str(input_path), config=config, yaml_path=yaml_str)

    stereo = config.get("stereo", False)
    audio_channels = 2 if stereo else 1
    n_fft = config.get("stft_n_fft", 2048)
    freq_bins = n_fft // 2 + 1
    fs = freq_bins * audio_channels

    dummy = torch.randn(1, fs, 256, 2)

    with torch.no_grad():
        test_out = model(dummy)
    print(f"Test inference OK: input {list(dummy.shape)} -> output {list(test_out.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["stft_repr"],
        output_names=["mask"],
        dynamic_axes={
            "stft_repr": {0: "batch", 2: "time_frames"},
            "mask": {0: "batch", 3: "time_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"stft_repr": dummy.numpy()})
    diff = abs(test_out.numpy() - ort_out[0]).max()
    print(f"ORT verification: max diff = {diff:.6e}")
    # Batch-axis self-check: a B=2 batch must equal two independent B=1 runs, item-for-item. Proves the
    # dynamic batch axis exported correctly (constant-folding didn't bake batch=1) AND that the Rust
    # unbind offset (j*N + stem) matches the real [B, N, ...] output layout. If this asserts, the batched
    # Rust path would silently corrupt output — fail loudly here at export time instead.
    dummy2 = torch.cat([dummy, dummy + 0.01], dim=0).numpy()
    ort_b2 = sess.run(None, {"stft_repr": dummy2})[0]
    s0 = sess.run(None, {"stft_repr": dummy.numpy()})[0]
    s1 = sess.run(None, {"stft_repr": (dummy + 0.01).numpy()})[0]
    b2_diff = max(abs(ort_b2[0] - s0[0]).max(), abs(ort_b2[1] - s1[0]).max())
    assert b2_diff < 1e-3, f"BATCH-AXIS EXPORT BROKEN: B=2 != 2x(B=1) (diff {b2_diff:.3e})"
    print(f"Batch-axis check passed: B=2 == 2x(B=1), diff = {b2_diff:.3e}")

    hop_length = config.get("stft_hop_length", 441)
    chunk_size = 131584  # ~3s at 44100Hz — keeps T≈299 frames, avoids O(T²) attention blowup

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "mel_band_roformer",
        "sample_rate": config.get("sample_rate", 44100),
        "stereo": stereo,
        "num_stems": config.get("num_stems", 1),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": config.get("stft_win_length", n_fft),
        "freq_bins": freq_bins,
        "chunk_size": chunk_size,
        "num_overlap": 2,
        "batch_size": 1,
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted MelBandRoformer: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stereo: {stereo}, Stems: {config.get('num_stems', 1)}, "
          f"FFT: {n_fft}, Hop: {hop_length}")


def convert_mdx23c(input_path: Path, output_path: Path, config_yaml: Path = None):
    """Convert MDX23C (TFC-TDF-v3) .ckpt model to ONNX."""
    yaml_str = str(config_yaml) if config_yaml else None
    config = detect_mdx23c_config(str(input_path), yaml_str)
    model = load_mdx23c(str(input_path), config=config, yaml_path=yaml_str)

    dim_f = config["dim_f"]
    num_stems = config["num_target_instruments"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    sample_rate = config.get("sample_rate", 44100)

    # CaC input: [B, 4, dim_f, T]
    dummy = torch.randn(1, 4, dim_f, 256)

    with torch.no_grad():
        test_out = model(dummy)
    print(f"Test inference OK: input {list(dummy.shape)} -> output {list(test_out.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["stft_repr"],
        output_names=["separated"],
        dynamic_axes={
            "stft_repr": {3: "time_frames"},
            "separated": {4: "time_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"stft_repr": dummy.numpy()})
    diff = abs(test_out.numpy() - ort_out[0]).max()
    print(f"ORT verification: max diff = {diff:.6e}")

    chunk_size = 261120  # matches MSST default for MDX23C

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "mdx23c",
        "sample_rate": sample_rate,
        "stereo": True,
        "num_stems": num_stems,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "dim_f": dim_f,
        "num_subbands": config["num_subbands"],
        "chunk_size": chunk_size,
        "num_overlap": 4,
        "batch_size": 1,
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted MDX23C: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stems: {num_stems}, dim_f: {dim_f}, FFT: {n_fft}, Hop: {hop_length}")


def convert_htdemucs(input_path: Path, output_path: Path):
    """Convert HTDemucs .th/.ckpt model to ONNX."""
    config = detect_htdemucs_config(str(input_path))
    model = load_htdemucs(str(input_path), config=config)

    nfft = config.get("nfft", 4096)
    samplerate = config.get("samplerate", 44100)
    segment = config.get("segment", 10)
    sources = config.get("sources", 4)
    freq_bins = nfft // 2

    # Fixed segment size (HTDemucs requires fixed input for ONNX)
    segment_samples = int(segment * samplerate)
    hop = nfft // 4
    # Spec frames: ceil(segment_samples / hop)
    le = int(math.ceil(segment_samples / hop))

    dummy_spec = torch.randn(1, 4, freq_bins, le)
    dummy_mix = torch.randn(1, 2, segment_samples)

    with torch.no_grad():
        freq_out, time_out = model(dummy_spec, dummy_mix)
    print(f"Test inference OK:")
    print(f"  CaC input: {list(dummy_spec.shape)}, Mix input: {list(dummy_mix.shape)}")
    print(f"  Freq out: {list(freq_out.shape)}, Time out: {list(time_out.shape)}")

    torch.onnx.export(
        model,
        (dummy_spec, dummy_mix),
        str(output_path),
        input_names=["cac_spec", "mix"],
        output_names=["freq_out", "time_out"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"cac_spec": dummy_spec.numpy(), "mix": dummy_mix.numpy()})
    diff_f = abs(freq_out.numpy() - ort_out[0]).max()
    diff_t = abs(time_out.numpy() - ort_out[1]).max()
    print(f"ORT verification: freq max diff = {diff_f:.6e}, time max diff = {diff_t:.6e}")

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "htdemucs",
        "sample_rate": samplerate,
        "stereo": True,
        "num_stems": sources,
        "n_fft": nfft,
        "hop_length": hop,
        "win_length": nfft,
        "segment_samples": segment_samples,
        "processing_mode": "hybrid",
        "num_overlap": 4,
        "batch_size": 1,
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted HTDemucs: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Sources: {sources}, FFT: {nfft}, Segment: {segment}s ({segment_samples} samples)")


def main():
    parser = argparse.ArgumentParser(description="Convert .pth/.ckpt models to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Input .pth/.ckpt file")
    parser.add_argument("--output", type=str, required=True, help="Output .onnx file")
    parser.add_argument("--type", type=str, required=True,
                        choices=["rvc", "sovits", "bs_roformer", "mel_band_roformer", "mdx23c", "htdemucs"],
                        help="Model type")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config (for mel_band_roformer / mdx23c STFT params)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.type == "rvc":
        convert_rvc(input_path, output_path)
    elif args.type == "sovits":
        convert_sovits(input_path, output_path)
    elif args.type == "bs_roformer":
        convert_bs_roformer(input_path, output_path)
    elif args.type == "mel_band_roformer":
        config_yaml = Path(args.config) if args.config else None
        convert_mel_band_roformer(input_path, output_path, config_yaml)
    elif args.type == "mdx23c":
        config_yaml = Path(args.config) if args.config else None
        convert_mdx23c(input_path, output_path, config_yaml)
    elif args.type == "htdemucs":
        convert_htdemucs(input_path, output_path)


if __name__ == "__main__":
    main()
