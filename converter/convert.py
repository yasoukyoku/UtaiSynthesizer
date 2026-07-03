"""Model converter: .pth/.ckpt → .onnx

Converts RVC, SoVITS, BSRoformer, MelBandRoformer, and MDX23C checkpoints
to ONNX format for inference via ONNX Runtime in the Rust backend.

Requires PyTorch (CPU is sufficient for conversion).

Usage:
    python convert.py --input model.pth  --output model.onnx --type rvc
    python convert.py --input model.pth  --output model.onnx --type sovits
    python convert.py --input model.ckpt --output model.onnx --type bs_roformer [--config model.yaml]
    python convert.py --input model.ckpt --output model.onnx --type mel_band_roformer [--config model.yaml]
    python convert.py --input model.ckpt --output model.onnx --type mdx23c --config model.yaml
    python convert.py --input model.th   --output model.onnx --type htdemucs [--config model.yaml]

Separation types accept --precision fp32|fp16|both (default fp32). fp16 keeps
the shared <stem>.json and writes <stem>.fp16.onnx (deleting the fp32 .onnx);
both keeps both files. rvc/sovits ignore --precision. fp16 is numerically
verified (45 dB gate) for all four separation types (FP16_VERIFIED_TYPES).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from architectures.rvc_v2 import build_from_checkpoint as build_rvc_v2
from architectures.sovits_v4 import build_from_checkpoint as build_sovits_v4
from architectures.bs_roformer import (
    load_from_checkpoint as load_bs_roformer,
    detect_config,
    MAX_ROPE_SEQ,
)
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
from architectures.msst_yaml import load_msst_yaml, stem_fields, stem_fields_from_yaml

# --precision applies to these (rvc/sovits ignore it); fp16 is numerically
# verified (>45 dB vs fp32) for the roformer archs + mdx23c (71.0/75.5 dB)
# ONLY — refuse the rest.
SEPARATION_TYPES = {"bs_roformer", "mel_band_roformer", "mdx23c", "htdemucs"}
FP16_VERIFIED_TYPES = {"bs_roformer", "mel_band_roformer", "mdx23c", "htdemucs"}


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


def _inference_params_from_yaml(config_yaml, fallback_chunk: int, fallback_overlap: int):
    """Original MSST inference recipe: audio.chunk_size + inference.num_overlap.
    Roformer separation quality depends on the inference chunk length matching the
    training dim_t — do NOT shrink these for speed (S16 did: 131584/2 vs the trained
    352800/4, a real audible SDR hit). The app's overlap slider is the speed knob.
    Hard-fails on unparseable yaml (load_msst_yaml) — silent fallback would bake
    wrong params."""
    y = load_msst_yaml(config_yaml)
    chunk = (y.get("audio") or {}).get("chunk_size", fallback_chunk)
    overlap = (y.get("inference") or {}).get("num_overlap", fallback_overlap)
    return chunk, overlap


def _write_model_json(output_path: Path, config_data: dict) -> Path:
    config_path = output_path.with_suffix(".json")
    config_path.write_text(json.dumps(config_data, indent=2))
    return config_path


def _roformer_chunk_params(config: dict, config_yaml,
                           fallback_chunk: int, fallback_overlap: int):
    """Resolve chunk/overlap BEFORE the expensive export and hard-fail if the
    chunk's STFT frame count exceeds the exported RoPE cos/sin cache
    (MAX_ROPE_SEQ) — the model would export cleanly and then die at runtime
    with a cryptic out-of-range read inside com.microsoft::RotaryEmbedding."""
    chunk_size, num_overlap = _inference_params_from_yaml(
        config_yaml, fallback_chunk, fallback_overlap)
    hop = config.get("stft_hop_length", 441)
    frames = chunk_size // hop + 1
    if frames > MAX_ROPE_SEQ:
        raise ValueError(
            f"chunk_size {chunk_size} @ hop {hop} gives {frames} STFT frames, "
            f"exceeding the exported rotary cache MAX_ROPE_SEQ={MAX_ROPE_SEQ} "
            f"(architectures/bs_roformer.py). Raise MAX_ROPE_SEQ and re-export."
        )
    return chunk_size, num_overlap


def _write_roformer_json(output_path: Path, model_type: str, config: dict,
                         config_yaml, chunk_size: int, num_overlap: int):
    """Shared bs_roformer/mel_band_roformer .json tail (ONE source of truth —
    the melband copy had drifted and lost dynamic_batch)."""
    n_fft = config.get("stft_n_fft", 2048)
    num_stems = config.get("num_stems", 1)
    config_data = {
        "type": model_type,
        "sample_rate": config.get("sample_rate", 44100),
        "stereo": config.get("stereo", False),
        "num_stems": num_stems,
        "n_fft": n_fft,
        "hop_length": config.get("stft_hop_length", 441),
        "win_length": config.get("stft_win_length", n_fft),
        "freq_bins": n_fft // 2 + 1,
        "chunk_size": chunk_size,
        "num_overlap": num_overlap,
        "batch_size": 1,
        "dynamic_batch": True,
    }
    config_data.update(stem_fields_from_yaml(load_msst_yaml(config_yaml), num_stems))
    config_path = _write_model_json(output_path, config_data)
    return config_path


def convert_bs_roformer(input_path: Path, output_path: Path, config_yaml: Path = None):
    """Convert BSRoformer .ckpt model to ONNX."""
    config = detect_config(str(input_path))
    # Fallback = viperx BSRoformer family (ep_317/ep_368 yamls both say 352800/4).
    chunk_size, num_overlap = _roformer_chunk_params(config, config_yaml, 352800, 4)
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

    config_path = _write_roformer_json(
        output_path, "bs_roformer", config, config_yaml, chunk_size, num_overlap)

    print(f"Converted BSRoformer: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stereo: {stereo}, Stems: {config.get('num_stems', 1)}, "
          f"FFT: {n_fft}, Hop: {config.get('stft_hop_length', 441)}")


def convert_mel_band_roformer(input_path: Path, output_path: Path,
                              config_yaml: Path = None):
    """Convert MelBandRoformer .ckpt model to ONNX."""
    yaml_str = str(config_yaml) if config_yaml else None
    config = detect_mel_config(str(input_path), yaml_str)
    # Fallback = melband_roformer_inst_v2 yaml (485100/4); pass --config for other melband models.
    chunk_size, num_overlap = _roformer_chunk_params(config, config_yaml, 485100, 4)
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

    config_path = _write_roformer_json(
        output_path, "mel_band_roformer", config, config_yaml, chunk_size, num_overlap)

    print(f"Converted MelBandRoformer: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stereo: {stereo}, Stems: {config.get('num_stems', 1)}, "
          f"FFT: {n_fft}, Hop: {config.get('stft_hop_length', 441)}")


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

    # Chunk/overlap from the model's own yaml — the old hardcoded 261120 broke
    # models trained with other geometries (e.g. chunk 130560 @ hop 512).
    chunk_size, num_overlap = _inference_params_from_yaml(config_yaml, 261120, 4)

    # The U-Net downsamples time by 2 per scale; the decoder skip-cat hard-fails
    # at runtime unless the frame count divides 2^num_scales. Check BEFORE export.
    num_scales = config["num_scales"]
    frames = chunk_size // hop_length + 1
    if frames % (2 ** num_scales) != 0:
        raise ValueError(
            f"MDX23C chunk/hop geometry is invalid for this model: chunk_size "
            f"{chunk_size} @ hop {hop_length} gives {frames} STFT frames, which is "
            f"not divisible by 2^{num_scales}={2 ** num_scales} (the U-Net's total "
            f"time downsampling). Inference WOULD hard-fail in the decoder skip-cat. "
            f"Pass the model's training yaml via --config so the trained chunk_size "
            f"is used instead of the 261120 default."
        )

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
        "num_overlap": num_overlap,
        "batch_size": 1,
    }
    config_data.update(stem_fields_from_yaml(load_msst_yaml(config_yaml), num_stems))
    config_path = _write_model_json(output_path, config_data)

    print(f"Converted MDX23C: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stems: {num_stems}, dim_f: {dim_f}, FFT: {n_fft}, Hop: {hop_length}")


def convert_htdemucs(input_path: Path, output_path: Path, config_yaml: Path = None):
    """Convert HTDemucs .th/.ckpt model to ONNX."""
    yaml_str = str(config_yaml) if config_yaml else None
    yaml_config = load_msst_yaml(yaml_str)
    config = detect_htdemucs_config(str(input_path), yaml_str)
    model = load_htdemucs(str(input_path), config=config)

    nfft = config.get("nfft", 4096)
    samplerate = config.get("samplerate", 44100)
    # segment: ckpt kwargs > yaml training.segment > 11 (resolved by detect_config;
    # MSST runs its htdemucs vocal models at 11s, not the demucs default 10s)
    segment = config.get("segment", 11)
    sources = config.get("sources", 4)
    num_stems = len(sources) if isinstance(sources, (list, tuple)) else int(sources)
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

    # Fallback 2 (NOT 4): official demucs weights ship signature-only yamls (no inference
    # section), and the model authors' own tooling (demucs apply_model) runs overlap=0.25
    # ≈ effective coverage 1.33 with triangular crossfades — ov4 is 2.95x that compute for
    # no proven quality gain (MSST's blanket 4 is an app-wide convention, not htdemucs
    # tuning). An MSST yaml that explicitly says num_overlap still wins (faithful).
    # Mirror: msst-catalog.ts MSST_DEFAULT_NUM_OVERLAP.htdemucs.
    num_overlap = (yaml_config.get("inference") or {}).get("num_overlap", 2)

    config_data = {
        "type": "htdemucs",
        "sample_rate": samplerate,
        "stereo": True,
        "num_stems": num_stems,
        "n_fft": nfft,
        "hop_length": hop,
        "win_length": nfft,
        "segment_samples": segment_samples,
        "processing_mode": "hybrid",
        "num_overlap": num_overlap,
        "batch_size": 1,
    }
    # Stem labels: ckpt kwargs['sources'] (official demucs) beats yaml training.
    if isinstance(sources, (list, tuple)):
        config_data.update(stem_fields(sources, None, num_stems))
    else:
        config_data.update(stem_fields_from_yaml(yaml_config, num_stems))
    config_path = _write_model_json(output_path, config_data)

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
                        help="Optional MSST training YAML (STFT params for mel_band_roformer / "
                             "mdx23c; inference chunk_size/num_overlap; hyperparams + segment "
                             "for htdemucs; stem labels for all separation models)")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "both"],
                        help="Separation types only (rvc/sovits ignore it). fp32: <stem>.onnx "
                             "as today. fp16: convert to <stem>.fp16.onnx and delete the fp32 "
                             ".onnx (the shared <stem>.json is kept). both: keep both files.")
    args = parser.parse_args()

    # Refuse unverified archs BEFORE touching any file — the torch export is
    # expensive and the result would be numerically unvalidated.
    if args.precision != "fp32" and args.type in SEPARATION_TYPES - FP16_VERIFIED_TYPES:
        print(f"Error: fp16 not yet numerically verified for {args.type}; use fp32",
              file=sys.stderr)
        sys.exit(1)

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
        config_yaml = Path(args.config) if args.config else None
        convert_bs_roformer(input_path, output_path, config_yaml)
    elif args.type == "mel_band_roformer":
        config_yaml = Path(args.config) if args.config else None
        convert_mel_band_roformer(input_path, output_path, config_yaml)
    elif args.type == "mdx23c":
        config_yaml = Path(args.config) if args.config else None
        convert_mdx23c(input_path, output_path, config_yaml)
    elif args.type == "htdemucs":
        config_yaml = Path(args.config) if args.config else None
        convert_htdemucs(input_path, output_path, config_yaml)

    if args.precision != "fp32" and args.type in SEPARATION_TYPES:
        # The torch export above MUST stay fp32 — it is the intermediate the
        # fp16 conversion consumes. Lazy import: rvc/sovits-only environments
        # don't need onnxconverter-common.
        from onnx_fp16 import convert_onnx_to_fp16, default_fp16_path
        fp16_path = default_fp16_path(output_path)
        convert_onnx_to_fp16(output_path, fp16_path)
        print(f"Converted to fp16: {fp16_path}")
        if args.precision == "fp16":
            # Keep the .json — it is shared by both precisions.
            output_path.unlink()
            print(f"Removed fp32 intermediate: {output_path}")


if __name__ == "__main__":
    main()
