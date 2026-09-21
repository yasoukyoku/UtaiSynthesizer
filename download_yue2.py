#!/usr/bin/env python3
"""
Download YuE-2 model files from HuggingFace
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Model directory
MODELS_DIR = Path(__file__).parent / "src-tauri" / "song_models" / "yue2-3b"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# VAE directory
VAE_DIR = MODELS_DIR / "vae"
VAE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading YuE-2 model to: {MODELS_DIR}")

# Files to download from YuE2-3B repo
yue2_3b_files = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "yue2_generation_config.json",
    "weights_manifest.json",
    "modeling_yue2.py",
    "qwen.tiktoken",
    "yue2_infer-0.1.5-py3-none-any.whl",
]

# Files to download from YuE2-Vae repo
yue2_vae_files = [
    "model.safetensors",
    "config.json",
    "weights_manifest.json",
    "modeling_vae.py",
]

print("\n=== Downloading YuE2-3B main model ===")
for file in yue2_3b_files:
    try:
        print(f"Downloading {file}...")
        local_path = hf_hub_download(
            repo_id="m-a-p/YuE2-3B",
            filename=file,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Saved to {local_path}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n=== Downloading YuE2-Vae ===")
for file in yue2_vae_files:
    try:
        print(f"Downloading vae/{file}...")
        local_path = hf_hub_download(
            repo_id="m-a-p/YuE2-Vae",
            filename=file,
            local_dir=str(VAE_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Saved to {local_path}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n=== YuE-2 Download Complete ===")
print(f"Model location: {MODELS_DIR}")
print("Files downloaded:")
for f in MODELS_DIR.rglob("*"):
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.relative_to(MODELS_DIR)} ({size_mb:.1f} MB)")
