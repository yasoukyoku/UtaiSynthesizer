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
import numpy as np


def convert_rvc(input_path: Path, output_path: Path):
    """Convert RVC .pth model to ONNX."""
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    # Detect model version and config
    config = checkpoint.get("config", {})
    if isinstance(config, (list, tuple)):
        # RVC format: config is a list [sr, model_version, ...]
        sample_rate = config[0] if len(config) > 0 else 40000
    else:
        sample_rate = config.get("sample_rate", 40000)

    # Extract model weights
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    # Build minimal RVC inference model
    model = build_rvc_model(state_dict, config)
    model.eval()

    # Prepare dummy inputs
    seq_len = 100
    phone = torch.randn(1, seq_len, 768)
    phone_lengths = torch.tensor([seq_len], dtype=torch.long)
    pitch = torch.randint(0, 256, (1, seq_len)).float()
    pitchf = torch.randn(1, seq_len)

    # Export to ONNX
    torch.onnx.export(
        model,
        (phone, phone_lengths, pitch, pitchf),
        str(output_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf"],
        output_names=["audio"],
        dynamic_axes={
            "phone": {1: "seq_len"},
            "phone_lengths": {0: "batch"},
            "pitch": {1: "seq_len"},
            "pitchf": {1: "seq_len"},
            "audio": {1: "audio_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Save config alongside
    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "rvc",
        "sample_rate": sample_rate,
        "version": "v2",
        "features_dim": 768,
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted RVC: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")


def convert_sovits(input_path: Path, output_path: Path):
    """Convert SoVITS .pth model to ONNX."""
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    model = build_sovits_model(state_dict, config)
    model.eval()

    # Dummy inputs for SoVITS
    seq_len = 100
    c = torch.randn(1, seq_len, 768)  # ContentVec features
    f0 = torch.randn(1, seq_len)
    sid = torch.tensor([0], dtype=torch.long)

    torch.onnx.export(
        model,
        (c, f0, sid),
        str(output_path),
        input_names=["c", "f0", "sid"],
        output_names=["audio"],
        dynamic_axes={
            "c": {1: "seq_len"},
            "f0": {1: "seq_len"},
            "audio": {1: "audio_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "sovits",
        "sample_rate": config.get("sample_rate", 44100),
        "version": "4.1",
        "features_dim": 768,
        "speakers": config.get("speakers", ["default"]),
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    print(f"Converted SoVITS: {input_path} -> {output_path}")


def build_rvc_model(state_dict: dict, config) -> torch.nn.Module:
    """Build minimal RVC inference model from state dict.

    TODO: Import actual RVC SynthesizerTrn architecture and load weights.
    This requires the model architecture definition from RVC.
    """
    # Placeholder — in production, this imports and builds the actual VITS model
    return PlaceholderModel(768, 1)


def build_sovits_model(state_dict: dict, config: dict) -> torch.nn.Module:
    """Build minimal SoVITS inference model from state dict.

    TODO: Import actual SoVITS architecture and load weights.
    """
    return PlaceholderModel(768, 1)


class PlaceholderModel(torch.nn.Module):
    """Placeholder for actual model architecture during development."""

    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 256)
        self.out = torch.nn.Linear(256, out_channels)

    def forward(self, *args):
        # Take first positional arg as features
        x = args[0]
        if x.dim() == 3:
            x = x.squeeze(0)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.out(x)
        return x.unsqueeze(0)


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
