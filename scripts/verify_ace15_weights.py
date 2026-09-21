# -*- coding: utf-8 -*-
"""Verify whether locally downloaded safetensors match official ACE-Step1.5 repo weights by comparing tensor keys and dtypes."""
import json, struct, os, sys, urllib.request

BASE = r"e:\软件开发\UtaiSynthesizer-main\UtaiSynthesizer-main\data\models\song\acestep-v1.5"
REPO = "ACE-Step/Ace-Step1.5"
MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]
HDRS = {"User-Agent": "Mozilla/5.0"}

LOCAL = {
    "acestep-v15-turbo/model.safetensors": os.path.join(BASE, "unet", "acestep_v1.5_xl_turbo_bf16.safetensors"),
    "vae/diffusion_pytorch_model.safetensors": os.path.join(BASE, "vae", "ace_1.5_vae.safetensors"),
    "Qwen3-Embedding-0.6B/model.safetensors": os.path.join(BASE, "text_encoders", "qwen_0.6b_ace15.safetensors"),
    "acestep-5Hz-lm-1.7B/model.safetensors": os.path.join(BASE, "text_encoders", "qwen_1.7b_ace15.safetensors"),
}

def read_local_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def remote_header(repo_file, max_fetch=64 * 1024 * 1024):
    last_err = None
    for m in MIRRORS:
        url = f"{m}/{REPO}/resolve/main/{repo_file}"
        try:
            req = urllib.request.Request(url, headers={**HDRS, "Range": f"bytes=0-{max_fetch-1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(8)
                n = struct.unpack("<Q", data)[0]
                buf = data
                while len(buf) < 8 + n:
                    chunk = r.read(min(8 + n - len(buf), 4 * 1024 * 1024))
                    if not chunk:
                        break
                    buf += chunk
                return json.loads(buf[8:8 + n])
        except Exception as e:
            last_err = e
    raise last_err

def summarize(header):
    dtypes = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        dtypes.setdefault(v["dtype"], 0)
        dtypes[v["dtype"]] += 1
    return set(header.keys()) - {"__metadata__"}, dtypes

for remote_name, local_path in LOCAL.items():
    print("=" * 60)
    print("REMOTE:", remote_name)
    rh = remote_header(remote_name)
    rkeys, rdt = summarize(rh)
    print("  remote tensors:", len(rkeys), "dtypes:", rdt)
    lh = read_local_header(local_path)
    lkeys, ldt = summarize(lh)
    print("  local  tensors:", len(lkeys), "dtypes:", ldt)
    inter = rkeys & lkeys
    only_r = rkeys - lkeys
    only_l = lkeys - rkeys
    print("  common:", len(inter), " remote-only:", len(only_r), " local-only:", len(only_l))
    # check dtype consistency on common keys
    mism = [k for k in list(inter)[:5000] if rh[k]["dtype"] != lh[k]["dtype"]]
    print("  dtype mismatches (sampled):", len(mism))
    if only_r:
        print("  sample remote-only:", sorted(only_r)[:5])
    if only_l:
        print("  sample local-only:", sorted(only_l)[:5])
    # check shapes on common keys
    shape_mism = [k for k in list(inter)[:5000] if rh[k]["shape"] != lh[k]["shape"]]
    print("  shape mismatches (sampled):", len(shape_mism))
