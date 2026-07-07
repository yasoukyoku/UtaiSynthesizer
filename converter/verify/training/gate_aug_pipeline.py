# -*- coding: utf-8 -*-
"""S41 gate_aug_pipeline — end-to-end pipeline invariants of the PSOLA
augmentation across knob changes (design B5; red-team F1/F3/F10/R3/R5/V7/V18/
V19/V21). Drives the REAL pipeline.run orchestration via gate_aug0_driver
subprocesses (CPU-pinned), all runs of one backend at the SAME workspace path
(filelists embed absolute paths).

Per backend (sovits / rvc / vocoder):
  step0  fresh copies=0  -> baseline snapshot (tree copy)
  step2  fresh copies=2  -> val identical to baseline (bytes); aug in train
         only; retrieval/index assets identical to baseline (originals-only);
         meta count == kept aug count
  rerun2 copies=2 again  -> skip-if-exists honored (sovits/vocoder: aug wav
         mtime unchanged; rvc: regenerated but BITWISE identical); original
         slices' feature caches untouched (mtime)
  step3  copies=3        -> incremental (aug1/2 preserved, aug3 added)
  step1  copies=1        -> stale aug2/aug3 fully removed (wav + companions +
         npz + meta)
  stepZ  copies=0        -> tree equals the copies=0 baseline (comparators
         from gate_aug0_noop: bytes / tensor / sample fallbacks)
Plus a dirty-material rejection consistency run (sovits, human OpenUtau
source): >=1 aug rejected, and every surviving aug is fully materialized
while no rejected residue exists anywhere (filelists, products, index).

    ..\\..\\..\\training\\.venv\\Scripts\\python.exe ^
        ..\\converter\\verify\\training\\gate_aug_pipeline.py [--backend all]
"""
import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_aug0_noop as noop  # build_cfg / comparators / VENV_PY / fixtures

APP = noop.APP
GATE_ROOT = noop.GATE_ROOT
FAILURES = []


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILURES.append(name)


def run_pipeline(backend, ws, copies, dataset_dir=None):
    cfg = noop.build_cfg(backend, ws)
    cfg["aug_copies"] = int(copies)
    if dataset_dir:
        cfg["dataset_dir"] = dataset_dir
    cfg_path = os.path.join(GATE_ROOT, "cfg_pipe_%s.json" % backend)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    r = subprocess.run(
        [noop.VENV_PY, noop.DRIVER, "--code-root",
         os.path.join(APP, "training"), "--backend", backend,
         "--config", cfg_path],
        cwd=os.path.join(APP, "training"),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0 or "STOPPED_AT_TRAIN_PREP" not in (r.stdout or ""):
        print(r.stdout)
        print((r.stderr or "")[-4000:])
        raise SystemExit("pipeline run failed (backend=%s copies=%s)" % (backend, copies))


def snap_tree(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def wipe(*paths):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)


def list_rel(root, pred=None):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            if pred is None or pred(rel):
                out.append(rel)
    return sorted(out)


def is_aug_rel(rel):
    import re

    first = os.path.basename(rel).split(".")[0]
    return re.search(r"_aug\d+$", first) is not None


def aug_idx(rel):
    import re

    m = re.search(r"_aug(\d+)$", os.path.basename(rel).split(".")[0])
    return int(m.group(1)) if m else None


def mtimes(root, rels):
    return {r: os.stat(os.path.join(root, r)).st_mtime_ns for r in rels}


def tree_equal_to(base, ours, label):
    bad = noop.compare_trees(base, ours)
    check(label, not bad, "" if not bad else "(%d diffs: %s...)" % (len(bad), bad[:3]))


# ─── per-backend descriptors ─────────────────────────────────────────────────

def backend_paths(backend, ws):
    if backend == "sovits":
        return {
            "slice_dir": os.path.join(ws, "dataset_44k", "gateaug"),
            "val": os.path.join(ws, "filelists", "val.txt"),
            "train": os.path.join(ws, "filelists", "train.txt"),
            "index": os.path.join(ws, "cluster", "0.index_vectors.npy"),
        }
    if backend == "rvc":
        return {
            "slice_dir": os.path.join(ws, "0_gt_wavs"),
            "val": None,
            "train": os.path.join(ws, "filelist.txt"),
            "index": os.path.join(ws, "total_fea.npy"),
        }
    if backend == "vocoder":
        return {
            "slice_dir": os.path.join(ws, "slices"),
            "val": os.path.join(ws, "filelists", "valid"),
            "train": os.path.join(ws, "filelists", "train"),
            "index": None,
        }
    raise SystemExit(backend)


def exercise(backend):
    print("== %s" % backend)
    ws = os.path.join(GATE_ROOT, "ws_pipe_%s" % backend)
    base_snap = ws + "_c0"
    P = backend_paths(backend, ws)

    # step0: fresh copies=0 baseline
    wipe(ws, base_snap)
    run_pipeline(backend, ws, 0)
    snap_tree(ws, base_snap)
    check("%s: baseline has no aug artifacts" % backend,
          not [r for r in list_rel(base_snap) if is_aug_rel(r)]
          and not os.path.isdir(os.path.join(base_snap, "aug_meta")))

    # step2: fresh copies=2
    wipe(ws)
    run_pipeline(backend, ws, 2)
    aug_wavs = list_rel(P["slice_dir"], lambda r: is_aug_rel(r) and r.endswith(".wav"))
    check("%s: aug slices generated" % backend, len(aug_wavs) > 0, "(%d)" % len(aug_wavs))
    if P["val"]:
        val_now = open(P["val"], "rb").read()
        val_base = open(os.path.join(base_snap, os.path.relpath(P["val"], ws)), "rb").read()
        check("%s: val bytes identical to copies=0" % backend, val_now == val_base)
        val_lines = open(P["val"], encoding="utf-8").read().splitlines()
        check("%s: no aug in val" % backend,
              not [l for l in val_lines if is_aug_rel(l)])
    train_lines = open(P["train"], encoding="utf-8").read().splitlines()
    train_aug = [l for l in train_lines if is_aug_rel(l.split("|")[0])]
    check("%s: aug present in train" % backend, len(train_aug) > 0,
          "(%d lines)" % len(train_aug))
    if P["index"]:
        idx_base = os.path.join(base_snap, os.path.relpath(P["index"], ws))
        check("%s: retrieval asset identical to copies=0 (originals-only)" % backend,
              filecmp.cmp(P["index"], idx_base, shallow=False))
    metas = [n for n in os.listdir(os.path.join(ws, "aug_meta"))
             if n.endswith(".json") and not n.startswith("_")]
    check("%s: meta count == surviving aug count" % backend,
          len(metas) == len(aug_wavs), "(%d/%d)" % (len(metas), len(aug_wavs)))

    # rerun2: cache semantics — feature caches live in the slice dir (sovits)
    # or sibling dirs (rvc 2a/2b/3_feature*, vocoder npz)
    slice_dir = P["slice_dir"]
    cache_roots = {"sovits": [slice_dir],
                   "rvc": [os.path.join(ws, d) for d in
                           ("2a_f0", "2b-f0nsf", "3_feature768")],
                   "vocoder": [os.path.join(ws, "npz")]}[backend]
    before = {r: open(os.path.join(slice_dir, r), "rb").read() for r in aug_wavs}
    mt_before = mtimes(slice_dir, aug_wavs)

    def cache_mtimes():
        out = {}
        for root in cache_roots:
            if os.path.isdir(root):
                for r in list_rel(root, lambda x: not x.endswith(".wav")):
                    out[os.path.join(root, r)] = os.stat(os.path.join(root, r)).st_mtime_ns
        return out

    mt_orig_before = cache_mtimes()
    run_pipeline(backend, ws, 2)
    after = {r: open(os.path.join(slice_dir, r), "rb").read() for r in aug_wavs}
    check("%s: rerun aug wav content bitwise stable" % backend,
          all(before[r] == after[r] for r in aug_wavs))
    mt_after = mtimes(slice_dir, aug_wavs)
    if backend == "rvc":
        pass  # slice dirs are rebuilt every run by design; bitwise checked above
    else:
        check("%s: rerun aug wav mtime unchanged (skip-if-exists hit)" % backend,
              mt_before == mt_after)
    if mt_orig_before:
        check("%s: feature caches untouched on rerun (skip-if-exists)" % backend,
              mt_orig_before == cache_mtimes())

    # step3 incremental
    run_pipeline(backend, ws, 3)
    aug3 = list_rel(slice_dir, lambda r: is_aug_rel(r) and r.endswith(".wav"))
    check("%s: copies=3 adds aug3" % backend,
          any(aug_idx(r) == 3 for r in aug3) and set(aug_wavs) <= set(aug3))
    check("%s: copies=3 preserves aug1/2 bytes" % backend,
          all(open(os.path.join(slice_dir, r), "rb").read() == before[r] for r in aug_wavs))

    # step1 downgrade
    run_pipeline(backend, ws, 1)
    residue = [r for r in list_rel(ws) if is_aug_rel(r) and (aug_idx(r) or 0) >= 2]
    check("%s: copies=1 removes ALL idx>=2 products" % backend, not residue,
          "" if not residue else str(residue[:5]))
    train_lines = open(P["train"], encoding="utf-8").read().splitlines()
    check("%s: filelist has no idx>=2 aug" % backend,
          not [l for l in train_lines if is_aug_rel(l.split("|")[0]) and (aug_idx(l.split("|")[0]) or 0) >= 2])

    # stepZ: back to zero == fresh baseline
    run_pipeline(backend, ws, 0)
    tree_equal_to(base_snap, ws, "%s: copies 2->..->0 tree equals fresh copies=0" % backend)


def dirty_rejection():
    print("== sovits dirty-material rejection consistency")
    # dirty + clean mix: the human (OpenUtau) file slices into ONE big piece —
    # alone it trips the >=3-originals filelist floor; mixing with clean
    # material is also the realistic user scenario (a few bad takes in a set)
    dirty_ds = os.path.join(GATE_ROOT, "dataset_dirty")
    if not os.path.isdir(dirty_ds):
        os.makedirs(dirty_ds)
        shutil.copy2(
            r"D:\MyDev\TESTING\utai-v2-testing\aug_engine_ab\human_00_orig.wav",
            os.path.join(dirty_ds, "dirty.wav"),
        )
        shutil.copy2(
            os.path.join(noop.FIXTURE, "b_kazane.wav"),
            os.path.join(dirty_ds, "clean.wav"),
        )
    ws = os.path.join(GATE_ROOT, "ws_pipe_dirty")
    wipe(ws)
    run_pipeline("sovits", ws, 1, dataset_dir=dirty_ds)
    spk = os.path.join(ws, "dataset_44k", "gateaug")
    kept = [n for n in os.listdir(spk) if n.endswith(".wav") and is_aug_rel(n)]
    metas = [n for n in os.listdir(os.path.join(ws, "aug_meta"))
             if n.endswith(".json") and not n.startswith("_")]
    train = open(os.path.join(ws, "filelists", "train.txt"), encoding="utf-8").read().splitlines()
    train_aug = [l for l in train if is_aug_rel(l)]
    srcs = [n for n in os.listdir(spk) if n.endswith(".wav") and not is_aug_rel(n)]
    check("dirty: >=1 aug rejected", len(kept) < len(srcs), "(%d kept / %d sources)" % (len(kept), len(srcs)))
    # vol.npy only exists under vol_embedding/diff — assert the always-present set
    check("dirty: every train aug exists with full products",
          all(any(os.path.basename(l) == k for k in kept) for l in train_aug)
          and all(
              os.path.exists(os.path.join(spk, os.path.splitext(k)[0] + suffix))
              for k in kept
              for suffix in (".wav.soft.pt", ".wav.f0.npy", ".spec.pt")
          ) if train_aug or kept else True)
    check("dirty: meta count matches kept", len(metas) == len(kept))
    # no rejected residue: every aug-named file in spk dir belongs to a kept stem
    residue = [
        n for n in os.listdir(spk)
        if is_aug_rel(n) and os.path.basename(n).split(".")[0] not in
        {os.path.splitext(k)[0] for k in kept}
    ]
    check("dirty: zero rejected residue", not residue, str(residue[:5]))


def diff_inherit():
    """R2/A5: the diff run must carry the inherited copies through BOTH the
    incremental path (shared workspace, caches hot) and the cache-wipe path
    (dataset changed -> dataset_44k rebuilt from scratch by the diff run)."""
    print("== sovits_diff inheritance")
    ws = os.path.join(GATE_ROOT, "ws_pipe_diff")
    wipe(ws)
    run_pipeline("sovits", ws, 2)
    spk = os.path.join(ws, "dataset_44k", "gateaug")
    aug_wavs = list_rel(spk, lambda r: is_aug_rel(r) and r.endswith(".wav"))
    mt = mtimes(spk, aug_wavs)

    # incremental: same dataset, diff run with the inherited copies
    run_pipeline("sovits_diff", ws, 2)
    check("diff: aug wavs untouched on incremental run", mt == mtimes(spk, aug_wavs))
    check("diff: aug slices got diff products",
          all(os.path.exists(os.path.join(spk, r.replace(".wav", ".wav.aug_mel.npy")))
              for r in aug_wavs))
    val = open(os.path.join(ws, "filelists", "val.txt"), encoding="utf-8").read().splitlines()
    check("diff: val still aug-free", not [l for l in val if is_aug_rel(l)])

    # cache-wipe: different dataset -> fingerprint change -> full rebuild
    dirty_ds = os.path.join(GATE_ROOT, "dataset_dirty")
    run_pipeline("sovits_diff", ws, 2, dataset_dir=dirty_ds)
    aug2 = list_rel(spk, lambda r: is_aug_rel(r) and r.endswith(".wav"))
    check("diff: cache-wipe rebuild regenerates aug slices", len(aug2) > 0,
          "(%d)" % len(aug2))
    check("diff: rebuilt aug have diff products",
          all(os.path.exists(os.path.join(spk, r.replace(".wav", ".wav.aug_mel.npy")))
              for r in aug2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all")
    args = ap.parse_args()
    backends = ["sovits", "rvc", "vocoder"] if args.backend == "all" else [args.backend]
    noop.ensure_fixture()
    for b in backends:
        exercise(b)
    if args.backend in ("all", "sovits"):
        dirty_rejection()
        diff_inherit()
    if FAILURES:
        print("RESULT: FAIL (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
