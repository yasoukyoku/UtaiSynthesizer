# -*- coding: utf-8 -*-
"""Finish downloading remaining files via huggingface_hub (hf-mirror endpoint)."""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

ROOT = r"e:\软件开发\UtaiSynthesizer-main\UtaiSynthesizer-main"
CKPT = os.path.join(ROOT, "data", "models", "song", "acestep-v1.5", "checkpoints")

JOBS = [
    ("ACE-Step/Ace-Step1.5", "vae/diffusion_pytorch_model.safetensors", os.path.join(CKPT, "vae")),
]
# plus any remaining small files that are missing
SMALL = [
    ("ACE-Step/acestep-v15-xl-turbo", "config.json", os.path.join(CKPT, "acestep-v15-xl-turbo")),
    ("ACE-Step/acestep-v15-xl-turbo", "configuration_acestep_v15.py", os.path.join(CKPT, "acestep-v15-xl-turbo")),
    ("ACE-Step/acestep-v15-xl-turbo", "modeling_acestep_v15_xl_turbo.py", os.path.join(CKPT, "acestep-v15-xl-turbo")),
    ("ACE-Step/acestep-v15-xl-turbo", "silence_latent.pt", os.path.join(CKPT, "acestep-v15-xl-turbo")),
    ("ACE-Step/Ace-Step1.5", "vae/config.json", os.path.join(CKPT, "vae")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/config.json", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/tokenizer.json", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/tokenizer_config.json", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/special_tokens_map.json", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/added_tokens.json", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/chat_template.jinja", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/merges.txt", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/vocab.json", os.path.join(CKPT, "Qwen3-Embedding-0.6B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/config.json", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/tokenizer.json", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/tokenizer_config.json", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/special_tokens_map.json", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/added_tokens.json", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/chat_template.jinja", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/merges.txt", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
    ("ACE-Step/Ace-Step1.5", "acestep-5Hz-lm-1.7B/vocab.json", os.path.join(CKPT, "acestep-5Hz-lm-1.7B")),
]


def ensure(repo, fname, dest_dir):
    dest = os.path.join(dest_dir, os.path.basename(fname))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"skip: {dest}")
        return
    path = hf_hub_download(repo_id=repo, filename=fname)
    os.makedirs(dest_dir, exist_ok=True)
    # copy (keep cache) or move — move to save disk space for big files
    size = os.path.getsize(path)
    if size > 100_000_000:
        if os.path.exists(dest):
            os.remove(dest)
        os.replace(path, dest)
    else:
        import shutil
        shutil.copyfile(path, dest)
    print(f"ok: {dest} ({size:,})")


for repo, fname, dest in SMALL:
    ensure(repo, fname, dest)
for repo, fname, dest in JOBS:
    ensure(repo, fname, dest)
print("ALL DONE")
