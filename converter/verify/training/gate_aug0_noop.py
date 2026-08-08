# -*- coding: utf-8 -*-
"""S41 gate_aug0_noop — with aug_copies=0 the WHOLE preprocessing product tree
must be byte-identical to the pre-S41 code (design B5; red-team V1/V3/V4/V5/V6).

Cold-run protocol (anti-self-certification, V1):
  1. git worktree of BASELINE (HEAD by default — pre-S41 code) is the reference
     implementation; both sides run through gate_aug0_driver (pipeline.run
     orchestration layer, CPU-pinned)
  2. wipe workspace -> run BASELINE cold -> rename tree aside as the snapshot
  3. run OURS cold at the SAME workspace path (filelists/config embed absolute
     paths, V4)
  4. compare trees file-by-file with per-suffix comparators (V6):
     bytes for wav/npy/txt/json/fingerprint; .pt = bytes first then exact
     tensor fallback (torch zip archive-name axis, S39); .wav = bytes first
     then exact sample fallback (libsndfile stamps a PEAK-chunk TIMESTAMP into
     float32 wavs — vocoder slices differ by 1 header byte across runs of the
     SAME code; measured 2026-07-07); train.log excluded

    ..\\..\\..\\training\\.venv\\Scripts\\python.exe ^
        ..\\converter\\verify\\training\\gate_aug0_noop.py [--backend sovits]
        [--baseline-rev HEAD]

Fixture dataset: TESTING/utai-v2-testing/gate_aug/dataset (built once from
training/assets/audition_10s.wav + a kazane excerpt; deleting it only changes
the dataset fingerprint, not the gate's validity)."""
import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VENV_PY = os.path.join(APP, "training", ".venv", "Scripts", "python.exe")
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_aug0_driver.py")
GATE_ROOT = r"D:\MyDev\TESTING\utai-v2-testing\gate_aug"
FIXTURE = os.path.join(GATE_ROOT, "dataset")
EXCLUDE_FILES = {"train.log"}


def ensure_fixture():
    if os.path.isdir(FIXTURE) and any(n.endswith(".wav") for n in os.listdir(FIXTURE)):
        return
    os.makedirs(FIXTURE, exist_ok=True)
    import librosa
    import soundfile as sf

    shutil.copy2(
        os.path.join(APP, "training", "assets", "audition_10s.wav"),
        os.path.join(FIXTURE, "a_teto.wav"),
    )
    x, sr = librosa.load(
        r"D:\MyDev\TESTING\Kazano_Sayo\dataset\20260704-004726.mp3",
        sr=None, mono=True, offset=80.0, duration=15.0,
    )
    sf.write(os.path.join(FIXTURE, "b_kazane.wav"), x, int(sr))
    print("fixture dataset created: %s" % FIXTURE)


def build_cfg(backend, workspace):
    aux = os.path.join(APP, "data", "models", "auxiliary")
    tr = os.path.join(APP, "data", "models", "training")
    cfg = {
        "backend": backend,
        "workspace": workspace,
        "dataset_dir": FIXTURE,
        "model_slug": "gateaug",
        "model_name": "gateaug",
        "seed": 1234,
        "stop_file": os.path.join(workspace, "stop.flag"),
        "gpu": "-1",
        "assets": {
            "ffmpeg": "ffmpeg",
            "contentvec_onnx": os.path.join(aux, "contentvec_768l12.onnx"),
            "rmvpe_pt": os.path.join(tr, "sovits", "rmvpe.pt"),
            "configs_dir": os.path.join(APP, "training", "assets", "configs", "sovits"),
        },
        # never reached (driver stops at train_prep) but present for parsers
        "pretrain_g": "unused",
        "pretrain_d": "unused",
    }
    if backend == "sovits":
        cfg.update({
            "version": "4.1", "total_epoch": 1, "batch_size": 2,
            "fp16": False, "vol_embedding": False, "loudnorm": False,
            "kmeans": False, "save_every_steps": 800, "keep_ckpts": 3,
            "all_in_mem": False, "aug_copies": 0,
        })
    elif backend == "rvc":
        cfg["assets"]["configs_dir"] = os.path.join(
            APP, "training", "assets", "configs", "rvc"
        )
        cfg["assets"]["rmvpe_pt"] = os.path.join(APP, "data", "models", "auxiliary", "rmvpe.pt")
        cfg["assets"]["mute_dir"] = os.path.join(APP, "training", "assets", "mute")
        cfg.update({
            "version": "v2", "sample_rate": "48k", "total_epoch": 1,
            "batch_size": 2, "fp16": False, "aug_copies": 0,
        })
    elif backend == "sovits_diff":
        cfg["assets"]["nsf_hifigan_model"] = os.path.join(
            tr, "sovits", "nsf_hifigan", "model"
        )
        cfg["assets"]["diffusion_pretrain"] = ""  # from-scratch path; never
        # loaded before train_prep anyway beyond the seeding no-op
        cfg.update({
            "version": "4.1", "total_steps": 20, "batch_size": 2,
            "save_every_steps": 10, "interval_force_save": 10, "k_step_max": 0,
            "fp16": False, "cache_all_data": True, "vol_embedding": False,
            "loudnorm": False, "aug_copies": 0,
        })
    elif backend == "vocoder":
        # run() isfile-checks the base model up front (never loaded — the
        # driver stops at train_prep), so point at the real asset
        cfg["assets"]["vocoder_pretrain"] = os.path.join(
            tr, "vocoder", "nsf_hifigan_44.1k_hop512_128bin_2024.02.ckpt"
        )
        cfg.update({
            "total_steps": 10, "save_every_steps": 5, "batch_size": 2,
            "keep_ckpts": 2, "crop_mel_frames": 32, "freeze_mpd": False,
            "aug_copies": 0,
        })
    else:
        raise SystemExit("backend %s not wired yet" % backend)
    return cfg


def run_side(label, code_root, backend, cfg_path):
    print("-- running %s side (code root %s)" % (label, code_root))
    r = subprocess.run(
        [VENV_PY, DRIVER, "--code-root", code_root, "--backend", backend,
         "--config", cfg_path],
        cwd=os.path.join(APP, "training"),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0 or "STOPPED_AT_TRAIN_PREP" not in (r.stdout or ""):
        print(r.stdout)
        print(r.stderr[-4000:] if r.stderr else "")
        raise SystemExit("%s side failed (rc=%d)" % (label, r.returncode))


def tensors_equal(pa, pb):
    import torch

    a = torch.load(pa, map_location="cpu", weights_only=False)
    b = torch.load(pb, map_location="cpu", weights_only=False)

    def eq(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
        if isinstance(x, dict):
            return isinstance(y, dict) and x.keys() == y.keys() and all(
                eq(x[k], y[k]) for k in x
            )
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(eq(i, j) for i, j in zip(x, y))
        return x == y

    return eq(a, b)


def samples_equal(pa, pb):
    import numpy as np
    import soundfile as sf

    da, ra = sf.read(pa, dtype="float32", always_2d=True)
    db, rb = sf.read(pb, dtype="float32", always_2d=True)
    return ra == rb and da.shape == db.shape and np.array_equal(da, db)


def pools_in(root):
    """Pool ids under `<slot>/pools`, sorted. Dot entries are migration staging, never a pool."""
    pools = os.path.join(root, "pools")
    if not os.path.isdir(pools):
        return []
    return sorted(n for n in os.listdir(pools) if not n.startswith("."))


def text_equal_modulo_pool(pa, pb, prefix_a, prefix_b):
    """Text equality after deleting the pool path segment from both sides.

    Both separators are handled because filelists are written with forward slashes while the
    workspace paths they are built from use the platform separator.
    """
    def norm(path, prefix):
        s = open(path, encoding="utf-8", errors="replace").read()
        for p in filter(None, {prefix, prefix.replace(os.sep, "/")}):
            s = s.replace(p, "")
        return s

    return norm(pa, prefix_a) == norm(pb, prefix_b)


def pool_prefix(root):
    """The ONE declared relocation of §F2⒝: `<slot>/pools/<pool_id>/`, or "" if this side has none.

    Returned so the comparison can state exactly what it normalised. Refuses to guess when a slot
    holds more than one pool — the whole point of the layout is that several can coexist, and a
    normaliser that silently picked one would be comparing an arbitrary half of the tree.
    """
    ids = pools_in(root)
    if not ids:
        return ""
    if len(ids) != 1:
        raise SystemExit(
            "gate cannot normalise a slot holding %d pools (%s) — a cold single-identity run must "
            "produce exactly one" % (len(ids), ids)
        )
    return os.path.join("pools", ids[0]) + os.sep


def pool_of(root):
    """The preprocessing pool directory of a slot, or the slot itself when it has none.

    Shared with `gate_aug_pipeline` so both gates ask ONE question about where the products are.
    """
    p = pool_prefix(root)
    return os.path.join(root, p.rstrip(os.sep)) if p else root


def compare_trees(base, ours):
    # ⚠ §F2⒝: the preprocessing products moved from the slot root into `pools/<id>/`, and the
    # absolute paths embedded in filelists moved with them. That is ONE declared relocation, so it
    # is normalised away here rather than being reported 30 times as "extra in ours" — but the
    # relocation itself is then NOT what this gate proves. It is pinned separately, by
    # `tpool::POOL_ENTRIES` + `gate_pool_table.py` + the migration's multiset comparison.
    # What survives here is this gate's actual claim: the product BYTES are unchanged.
    prefix_a, prefix_b = pool_prefix(base), pool_prefix(ours)
    if prefix_a or prefix_b:
        print("  [note] normalising the pool relocation: base=%r ours=%r"
              % (prefix_a or "<none>", prefix_b or "<none>"))

    def walk(root, strip):
        out = {}
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f in EXCLUDE_FILES:
                    continue
                p = os.path.join(dirpath, f)
                rel = os.path.relpath(p, root)
                if strip and rel.startswith(strip):
                    rel = rel[len(strip):]
                if rel in out:
                    raise SystemExit(
                        "pool normalisation collided on %r — the same relative path exists both "
                        "inside and outside the pool, so this comparison cannot be trusted" % rel
                    )
                out[rel] = p
        return out

    a, b = walk(base, prefix_a), walk(ours, prefix_b)
    bad = []
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    for rel in only_a:
        bad.append("missing in ours: %s" % rel)
    for rel in only_b:
        bad.append("extra in ours: %s" % rel)
    same_ct = 0
    for rel in sorted(set(a) & set(b)):
        if filecmp.cmp(a[rel], b[rel], shallow=False):
            same_ct += 1
            continue
        if rel.endswith(".pt") and tensors_equal(a[rel], b[rel]):
            same_ct += 1
            print("  [note] %s: bytes differ, tensors exact-equal (archive-name axis)" % rel)
            continue
        if rel.endswith(".wav") and samples_equal(a[rel], b[rel]):
            same_ct += 1
            print("  [note] %s: bytes differ, samples exact-equal (PEAK-chunk timestamp axis)" % rel)
            continue
        # filelists hold ABSOLUTE paths into the products, so the same relocation shows up inside
        # them. Normalised on the same terms, and only for text — never for a product.
        if rel.endswith((".txt", ".json")) and text_equal_modulo_pool(
            a[rel], b[rel], prefix_a, prefix_b
        ):
            same_ct += 1
            print("  [note] %s: identical after normalising the pool path segment" % rel)
            continue
        bad.append("content differs: %s" % rel)
    print("compared %d files: %d identical, %d problems" % (len(a | b.keys() if False else set(a) | set(b)), same_ct, len(bad)))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sovits")
    ap.add_argument("--baseline-rev", default="HEAD")
    args = ap.parse_args()

    ensure_fixture()
    ws = os.path.join(GATE_ROOT, "ws_noop_%s" % args.backend)
    snap = ws + "_baseline"
    cfg_path = os.path.join(GATE_ROOT, "cfg_noop_%s.json" % args.backend)
    os.makedirs(GATE_ROOT, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(build_cfg(args.backend, ws), f, ensure_ascii=False, indent=1)

    wt = tempfile.mkdtemp(prefix="s41_baseline_wt_")
    try:
        subprocess.run(
            ["git", "-C", APP, "worktree", "add", "--detach", wt, args.baseline_rev],
            check=True, capture_output=True,
        )
        for d in (ws, snap):
            if os.path.isdir(d):
                shutil.rmtree(d)
        run_side("baseline(%s)" % args.baseline_rev, os.path.join(wt, "training"),
                 args.backend, cfg_path)
        os.rename(ws, snap)
        run_side("ours", os.path.join(APP, "training"), args.backend, cfg_path)
        bad = compare_trees(snap, ws)
        if bad:
            for line in bad:
                print("  [FAIL] %s" % line)
            print("RESULT: FAIL (%d diffs)" % len(bad))
            sys.exit(1)
        print("RESULT: PASS — copies=0 tree is byte-identical to %s" % args.baseline_rev)
    finally:
        subprocess.run(
            ["git", "-C", APP, "worktree", "remove", "--force", wt],
            capture_output=True,
        )


if __name__ == "__main__":
    main()
