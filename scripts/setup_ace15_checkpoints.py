# -*- coding: utf-8 -*-
"""Reorganize ACE-Step v1.5 model dir into official checkpoints layout.

- Reuse local XL DiT (verified identical to ACE-Step/acestep-v15-xl-turbo, BF16)
- Remap local Qwen weights (strip `model.` prefix) to official key layout
- Download missing small config/tokenizer files + official VAE from hf-mirror
"""
import json
import os
import shutil
import struct
import urllib.request

from safetensors.torch import load_file, save_file

ROOT = r"e:\软件开发\UtaiSynthesizer-main\UtaiSynthesizer-main"
SRC = os.path.join(ROOT, "data", "models", "song", "acestep-v1.5")
CKPT = os.path.join(SRC, "checkpoints")
MIRROR = "https://hf-mirror.com/"
UA = {"User-Agent": "Mozilla/5.0"}

MAIN = "ACE-Step/Ace-Step1.5"
XL = "ACE-Step/acestep-v15-xl-turbo"

DIRS = {
    "dit": os.path.join(CKPT, "acestep-v15-xl-turbo"),
    "vae": os.path.join(CKPT, "vae"),
    "qwen06": os.path.join(CKPT, "Qwen3-Embedding-0.6B"),
    "lm17": os.path.join(CKPT, "acestep-5Hz-lm-1.7B"),
}

# (repo, remote_path, local_dest_dir)
DOWNLOADS = [
    (XL, "config.json", "dit"),
    (XL, "configuration_acestep_v15.py", "dit"),
    (XL, "modeling_acestep_v15_xl_turbo.py", "dit"),
    (XL, "silence_latent.pt", "dit"),
    (MAIN, "vae/config.json", "vae"),
    (MAIN, "vae/diffusion_pytorch_model.safetensors", "vae"),
    (MAIN, "Qwen3-Embedding-0.6B/config.json", "qwen06"),
    (MAIN, "Qwen3-Embedding-0.6B/tokenizer.json", "qwen06"),
    (MAIN, "Qwen3-Embedding-0.6B/tokenizer_config.json", "qwen06"),
    (MAIN, "Qwen3-Embedding-0.6B/special_tokens_map.json", "qwen06"),
    (MAIN, "Qwen3-Embedding-0.6B/added_tokens.json", "qwen06"),
    (MAIN, "Qwen3-Embedding-0.6B/chat_template.jinja", "qwen06"),
    (MAIN, "Qwen3-Embedding-0.6B/merges.txt", "qwen06"),
    (MAIN, "Qwen3-Embedding-0.6B/vocab.json", "qwen06"),
    (MAIN, "acestep-5Hz-lm-1.7B/config.json", "lm17"),
    (MAIN, "acestep-5Hz-lm-1.7B/tokenizer.json", "lm17"),
    (MAIN, "acestep-5Hz-lm-1.7B/tokenizer_config.json", "lm17"),
    (MAIN, "acestep-5Hz-lm-1.7B/special_tokens_map.json", "lm17"),
    (MAIN, "acestep-5Hz-lm-1.7B/added_tokens.json", "lm17"),
    (MAIN, "acestep-5Hz-lm-1.7B/chat_template.jinja", "lm17"),
    (MAIN, "acestep-5Hz-lm-1.7B/merges.txt", "lm17"),
    (MAIN, "acestep-5Hz-lm-1.7B/vocab.json", "lm17"),
]


def download(repo, remote, dest_dir):
    url = MIRROR + repo + "/resolve/main/" + remote
    dest = os.path.join(dest_dir, os.path.basename(remote))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  skip (exists): {dest}")
        return
    tmp = dest + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={done}-"})
    mode = "ab" if done else "wb"
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, mode) as f:
        total = r.headers.get("Content-Length")
        total = int(total) + done if total else None
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {remote}: {pct:3d}% ({done:,}/{total:,})", end="", flush=True)
    print()
    if os.path.exists(dest):
        os.remove(dest)
    os.rename(tmp, dest)
    print(f"  done: {dest}")


def remap_qwen(src_path, dest_path):
    """Strip `model.` prefix from tensor keys to match official layout."""
    print(f"remap: {src_path}")
    tensors = load_file(src_path)
    remapped = {}
    for k, v in tensors.items():
        nk = k[6:] if k.startswith("model.") else k
        remapped[nk] = v.contiguous()
    save_file(remapped, dest_path, metadata={"format": "pt"})
    print(f"  -> {dest_path} ({len(remapped)} tensors)")


def main():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    # 1. Move local XL DiT (fast rename on same drive)
    src_xl = os.path.join(SRC, "unet", "acestep_v1.5_xl_turbo_bf16.safetensors")
    dst_dit = os.path.join(DIRS["dit"], "model.safetensors")
    if os.path.exists(src_xl) and not os.path.exists(dst_dit):
        print(f"move DiT XL -> {dst_dit}")
        shutil.move(src_xl, dst_dit)

    # 2. Remap Qwen weights
    q06_src = os.path.join(SRC, "text_encoders", "qwen_0.6b_ace15.safetensors")
    q06_dst = os.path.join(DIRS["qwen06"], "model.safetensors")
    if os.path.exists(q06_src) and not os.path.exists(q06_dst):
        remap_qwen(q06_src, q06_dst)
    lm_src = os.path.join(SRC, "text_encoders", "qwen_1.7b_ace15.safetensors")
    lm_dst = os.path.join(DIRS["lm17"], "model.safetensors")
    if os.path.exists(lm_src) and not os.path.exists(lm_dst):
        remap_qwen(lm_src, lm_dst)

    # 3. Download all missing files (with resume)
    for repo, remote, key in DOWNLOADS:
        print(f"download: {repo}/{remote}")
        download(repo, remote, DIRS[key])

    print("\nAll done. Final tree:")
    for name, d in DIRS.items():
        print(f"[{name}] {d}")
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            print(f"   {f}  {os.path.getsize(p):,}")


if __name__ == "__main__":
    main()
