"""Model converter: .pth → .onnx

Converts RVC and SoVITS PyTorch checkpoints to ONNX format for inference
via ONNX Runtime in the Rust backend.

Requires PyTorch (CPU is sufficient for conversion).

Usage:
    python convert.py --input model.pth --output model.onnx --type rvc
    python convert.py --input model.pth --output model.onnx --type sovits
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from architectures.rvc_v2 import build_from_checkpoint as build_rvc_v2
from architectures.sovits_v4 import build_from_checkpoint as build_sovits_v4


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

    torch.onnx.export(
        model,
        (phone, phone_lengths, pitch, pitchf, sid),
        str(output_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "sid"],
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


def main():
    parser = argparse.ArgumentParser(description="Convert .pth models to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Input .pth file")
    parser.add_argument("--output", type=str, required=True, help="Output .onnx file")
    parser.add_argument("--type", type=str, required=True, choices=["rvc", "sovits"],
                        help="Model type")
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


if __name__ == "__main__":
    main()
