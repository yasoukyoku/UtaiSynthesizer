# -*- coding: utf-8 -*-
"""Runtime-pack builder (S42) — produces the distributable embedded-Python packs.

    python build_pack.py --variant cpu [--version 1] [--workdir build]

Output (in <workdir>/dist/):
    runtime-<variant>-v<N>.tar.zst[.partNN]   the pack archive (split at 1.9 GiB
                                              for GitHub-release hosting)
    runtime-<variant>-v<N>.manifest.json      per-part sha256 — what the app's
                                              downloader verifies

Pack layout inside the archive (see src-tauri/src/pyenv/mod.rs):
    pack.json                 metadata (id/variant/python/torch/disk_bytes)
    python/                   python-build-standalone (msvc-shared, install_only)
                              with the unified dependency set flat in Lib/site-packages

Design invariants (s42_training_env_design.md §2.2):
  - NO venv — flat site-packages inside the standalone tree; everything is invoked
    as `python.exe -m ...`, never through Scripts/*.exe launchers (which bake
    absolute paths and break on relocation — they are PRUNED from the pack).
  - Interpreter pinned to an exact python-build-standalone release asset, verified
    against its published .sha256.
  - Dependencies pinned by the per-variant lockfile in locks/ (generated from the
    VALIDATED staging env freeze — regenerate only after re-running the gates).

Builder host requirements: any Python 3.10+ with `zstandard` installed, network
access, and a warm pip cache if any dependency is sdist-only (pyworld: this dev
machine's pip cache holds the locally built cp310 wheel; a future CI needs a
wheelhouse — see the design doc's risk list).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── pinned interpreter (python-build-standalone, Astral) ────────────────────
PBS_TAG = "20260623"
PBS_PY = "3.10.20"
PBS_ASSET = f"cpython-{PBS_PY}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz"
PBS_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PBS_TAG}/{PBS_ASSET.replace('+', '%2B')}"
)

VARIANTS = {
    "cpu": {
        "lock": "runtime-cpu.lock.txt",
        "label": "CPU 运行时（模型转换基座 + CPU 训练）",
    },
    "nv-cu130": {
        "lock": "runtime-nv-cu130.lock.txt",
        "label": "NVIDIA 运行时（cu130；RTX 20-50 训练 + 模型转换）",
    },
    "amd": {
        "lock": "runtime-amd.lock.txt",
        # AMD/TheRock ROCm; cp310 (SAME PBS interpreter as cpu/nv — no per-variant
        # python needed, the earlier ABI worry was moot). Experimental tier.
        "label": "AMD 运行时（TheRock ROCm；RDNA3/4 训练 + 模型转换，实验性）",
    },
    # Phase C: "xpu" — one lockfile, same recipe (cp310, download.pytorch.org/whl/xpu).
}

PART_BYTES = 1_900_000_000  # < GitHub release 2 GiB per-file cap


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path):
    if dest.exists():
        log(f"cached: {dest.name}")
        return
    log(f"download {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # builder-side only; the APP uses the resumable Rust downloader
    tmp.rename(dest)


def fetch_interpreter(dl_dir: Path) -> Path:
    dl_dir.mkdir(parents=True, exist_ok=True)
    archive = dl_dir / PBS_ASSET
    download(PBS_URL, archive)
    # PBS publishes one release-level SHA256SUMS (no per-asset sidecars) — pull it
    # and match our asset's line.
    sums_file = dl_dir / f"SHA256SUMS-{PBS_TAG}"
    download(
        f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_TAG}/SHA256SUMS",
        sums_file,
    )
    want = None
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == PBS_ASSET:
            want = parts[0].strip().lower()
            break
    if not want:
        raise SystemExit(f"{PBS_ASSET} not found in SHA256SUMS")
    got = sha256_file(archive)
    if got != want:
        archive.unlink()
        raise SystemExit(f"PBS archive sha256 mismatch: {got} != {want}")
    log("interpreter sha256 OK")
    return archive


def _clean_env():
    """Builder-host isolation: an inherited PYTHONHOME/PYTHONPATH/user-site would
    leak the host's packages into the pack (or into the sanity probe, masking a
    missing dependency). Mirrors util::python_command's runtime posture."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=_clean_env(), **kw)


def build(variant: str, version: int, workdir: Path):
    spec = VARIANTS[variant]
    pack_id = f"runtime-{variant}-v{version}"
    stage = workdir / "stage" / pack_id
    dist = workdir / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # 1. interpreter
    archive = fetch_interpreter(workdir / "dl")
    log("extracting interpreter ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(stage)  # yields stage/python/
    py = stage / "python" / "python.exe"
    assert py.exists(), "unexpected PBS layout"

    # 2. dependencies (flat into the standalone tree's own site-packages)
    lock = HERE / "locks" / spec["lock"]
    run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet", "--no-warn-script-location"])
    run([py, "-m", "pip", "install", "-r", lock, "--no-warn-script-location"])

    # 3. sanity probe INSIDE the pack interpreter (the full numeric gate is the
    #    app-side envtest; this catches a broken install before we spend minutes
    #    compressing garbage)
    probe = (
        "import torch, torchaudio, numpy, librosa, soundfile, sklearn, onnx, "
        "onnxruntime, onnxconverter_common, faiss, parselmouth, pyworld, yaml, "
        "lightning, torchmetrics, matplotlib, scipy, click; "
        "import json,sys; print(json.dumps({'torch': torch.__version__, "
        "'numpy': numpy.__version__, 'py': sys.version.split()[0]}))"
    )
    out = subprocess.run(
        [str(py), "-c", probe], capture_output=True, text=True, encoding="utf-8",
        env=_clean_env(), check=True,
    )
    info = json.loads(out.stdout.strip().splitlines()[-1])
    log(f"probe: {info}")

    # 4. prune: bytecode caches (regenerate on use) + Scripts launchers (absolute
    #    paths baked in — the app NEVER calls them; leaving them would invite
    #    exactly the relocation breakage the flat layout avoids)
    pruned = 0
    for pycache in (stage / "python").rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
        pruned += 1
    scripts = stage / "python" / "Scripts"
    if scripts.exists():
        for exe in scripts.glob("*.exe"):
            exe.unlink()
    log(f"pruned {pruned} __pycache__ dirs + Scripts launchers")

    # 5. pack.json
    disk_bytes = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    meta = {
        "schema": 1,
        "id": pack_id,
        "variant": variant,
        # Same-variant coexistence picks the HIGHEST version (pyenv::converter_python)
        "version": version,
        "label": spec["label"],
        "python": info["py"],
        "torch": info["torch"],
        "disk_bytes": disk_bytes,
        "built": time.strftime("%Y-%m-%d"),
        "pbs": f"{PBS_PY}+{PBS_TAG}",
    }
    (stage / "pack.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"disk_bytes = {disk_bytes/1e9:.2f} GB")

    # 6. tar → zstd(19, multithread) → split → sha256 manifest
    import zstandard

    tar_zst = dist / f"{pack_id}.tar.zst"
    log("compressing (zstd -19, this takes a few minutes) ...")
    cctx = zstandard.ZstdCompressor(level=19, threads=-1)
    with open(tar_zst, "wb") as raw:
        with cctx.stream_writer(raw) as zw:
            with tarfile.open(fileobj=zw, mode="w|") as tf:
                # deterministic order; archive root holds pack.json + python/
                for p in sorted(stage.rglob("*")):
                    tf.add(p, arcname=str(p.relative_to(stage)), recursive=False)
    size = tar_zst.stat().st_size
    log(f"archive = {size/1e6:.1f} MB")

    parts = []
    if size > PART_BYTES:
        log("splitting ...")
        with open(tar_zst, "rb") as f:
            n = 1
            while True:
                chunk = f.read(PART_BYTES)
                if not chunk:
                    break
                part = dist / f"{pack_id}.tar.zst.part{n:02d}"
                part.write_bytes(chunk)
                parts.append(part)
                n += 1
        tar_zst.unlink()
    else:
        parts = [tar_zst]

    manifest = {
        "schema": 1,
        "id": pack_id,
        "variant": variant,
        "label": spec["label"],
        "disk_bytes": disk_bytes,
        "parts": [
            {"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)}
            for p in parts
        ],
    }
    man_path = dist / f"{pack_id}.manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"manifest -> {man_path}")
    for p in parts:
        log(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")
    log("DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--workdir", default=str(HERE / "build"))
    args = ap.parse_args()
    build(args.variant, args.version, Path(args.workdir))


if __name__ == "__main__":
    main()
