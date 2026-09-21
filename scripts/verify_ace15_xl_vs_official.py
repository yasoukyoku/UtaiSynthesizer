# -*- coding: utf-8 -*-
"""Compare local XL unet safetensors header with remote ACE-Step/acestep-v15-xl-turbo (4 shards)."""
import json
import struct
import urllib.request

LOCAL_XL = r"e:\软件开发\UtaiSynthesizer-main\UtaiSynthesizer-main\data\models\song\acestep-v1.5\unet\acestep_v1.5_xl_turbo_bf16.safetensors"
REMOTE_BASE = "https://hf-mirror.com/ACE-Step/acestep-v15-xl-turbo/resolve/main/"
SHARDS = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
HDR_UA = {"User-Agent": "Mozilla/5.0"}


def read_local_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def read_remote_header(url):
    req = urllib.request.Request(url, headers={**HDR_UA, "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=60) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={**HDR_UA, "Range": f"bytes=8-{8 + n - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read(n))


def main():
    local = read_local_header(LOCAL_XL)
    local_tensors = {k: v for k, v in local.items() if k != "__metadata__"}
    print(f"local tensors: {len(local_tensors)}")

    remote_tensors = {}
    for s in SHARDS:
        h = read_remote_header(REMOTE_BASE + s)
        remote_tensors.update({k: v for k, v in h.items() if k != "__metadata__"})
        print(f"  {s}: cumulative {len(remote_tensors)}")
    print(f"remote tensors: {len(remote_tensors)}")

    lk, rk = set(local_tensors), set(remote_tensors)
    common = lk & rk
    only_local = lk - rk
    only_remote = rk - lk
    shape_mismatch = [
        (k, local_tensors[k].get("shape"), remote_tensors[k].get("shape"))
        for k in common
        if list(local_tensors[k].get("shape", [])) != list(remote_tensors[k].get("shape", []))
    ]
    dtype_mismatch = [
        (k, local_tensors[k].get("dtype"), remote_tensors[k].get("dtype"))
        for k in common
        if local_tensors[k].get("dtype") != remote_tensors[k].get("dtype")
    ]
    print(f"common: {len(common)}, only_local: {len(only_local)}, only_remote: {len(only_remote)}")
    print(f"shape mismatches: {len(shape_mismatch)}")
    for item in shape_mismatch[:10]:
        print("  ", item)
    print(f"dtype mismatches: {len(dtype_mismatch)}")
    for item in dtype_mismatch[:10]:
        print("  ", item)
    if only_local:
        print("sample only_local:", sorted(only_local)[:8])
    if only_remote:
        print("sample only_remote:", sorted(only_remote)[:8])


if __name__ == "__main__":
    main()
