"""Download ACE-Step v1.5 BASE model (enables lego/extract/complete multitrack tasks)."""
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_CACHE"] = r"e:\软件开发\UtaiSynthesizer-main\UtaiSynthesizer-main\data\.hf_cache"

REPO = "ACE-Step/acestep-v15-base"
DEST = Path(r"e:\软件开发\UtaiSynthesizer-main\UtaiSynthesizer-main\data\models\song\acestep-v1.5\checkpoints\acestep-v15-base")
FILES = {
    "model.safetensors": 4787825604,
    "silence_latent.pt": 3841215,
    "config.json": 1940,
    "configuration_acestep_v15.py": 13130,
    "modeling_acestep_v15_base.py": 95545,
}


def main():
    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    for fname, size in FILES.items():
        dest = DEST / fname
        if dest.exists() and dest.stat().st_size == size:
            print(f"[skip] {fname} already OK ({size})")
            continue
        if dest.exists():
            dest.unlink()
        print(f"[down] {fname} ({size:,} bytes) ...", flush=True)
        p = hf_hub_download(repo_id=REPO, filename=fname)
        src = Path(p)
        assert src.stat().st_size == size, f"cache size mismatch: {src} {src.stat().st_size} != {size}"
        shutil.copyfile(src, dest)
        assert dest.stat().st_size == size, f"dest size mismatch: {dest}"
        print(f"[ok] {fname} -> {dest}")
    print("ALL DONE")


if __name__ == "__main__":
    sys.exit(main())
