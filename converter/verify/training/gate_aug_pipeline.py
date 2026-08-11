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
CHECKS = []

#: Backends this gate can drive END TO END. ⛔ Narrower than `noop.BACKENDS`:
#: `sovits_diff` has no `backend_paths` arm — it is only ever run *through*
#: `diff_inherit()`, never exercised on its own. Before S136 `--backend sovits_diff`
#: died at `raise SystemExit(backend)`, whose entire message was the word "sovits_diff".
DRIVABLE = ("sovits", "rvc", "vocoder")

#: Floor on how many checks a full `--backend all` run must have printed.
#: ⛔ Three of this gate's checks are behind `if` gates (`if P["val"]`, `if P["index"]`,
#: `if mt_orig_before`). When a path stops resolving, those do not go red — they
#: **disappear**, and the total silently shrinks while everything left prints PASS.
#: A floor, not a pin: today `--backend all` prints 52 (README:610) and it may grow.
MIN_CHECKS_ALL = 48


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    CHECKS.append(name)
    if not ok:
        FAILURES.append(name)


def run_pipeline(backend, ws, copies, dataset_dir=None, identity_version=None):
    cfg = noop.build_cfg(backend, ws, identity_version=identity_version)
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
        # ⛔ exit 3, not 1: the pipeline never produced a tree, so nothing downstream of
        # here was measured. Same code as a real byte difference is what S129 forbids.
        raise noop.GateUnrunnable(
            "pipeline run failed (backend=%s copies=%s) — no products were produced, so the "
            "invariants below were never tested" % (backend, copies))


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
    """⚠ §F2⒝ — every path here is now explicitly on ONE side of the pool/run divide.

    `pool` holds the products a preprocessing identity owns (slices, features, aug_meta); `ws` is
    the slot, where the run's own artifacts stay (filelists, retrieval asset, config). Deriving
    them all from `ws` was correct only while the two were the same directory, and having this
    gate keep its own second opinion about which is which is exactly how the two would drift.
    """
    pool = noop.pool_of(ws)
    if backend == "sovits":
        return {
            "slice_dir": os.path.join(pool, "dataset_44k", "gateaug"),
            "meta": os.path.join(pool, "aug_meta"),
            "cache_roots": [os.path.join(pool, "dataset_44k", "gateaug")],
            "val": os.path.join(ws, "filelists", "val.txt"),
            "train": os.path.join(ws, "filelists", "train.txt"),
            "index": os.path.join(ws, "cluster", "0.index_vectors.npy"),
        }
    if backend == "rvc":
        return {
            "slice_dir": os.path.join(pool, "0_gt_wavs"),
            "meta": os.path.join(pool, "aug_meta"),
            "cache_roots": [os.path.join(pool, d) for d in
                            ("2a_f0", "2b-f0nsf", "3_feature768")],
            "val": None,
            "train": os.path.join(ws, "filelist.txt"),
            "index": os.path.join(ws, "total_fea.npy"),
        }
    if backend == "vocoder":
        return {
            "slice_dir": os.path.join(pool, "slices"),
            "meta": os.path.join(pool, "aug_meta"),
            "cache_roots": [os.path.join(pool, "npz")],
            "val": os.path.join(ws, "filelists", "valid"),
            "train": os.path.join(ws, "filelists", "train"),
            "index": None,
        }
    raise noop.GateUnrunnable(
        "backend %r has no backend_paths arm — this gate cannot say where its products live, "
        "so it cannot test anything about them. Drivable here: %s "
        "(sovits_diff is only exercised through diff_inherit)" % (backend, ", ".join(DRIVABLE)))


def exercise(backend):
    print("== %s" % backend)
    ws = os.path.join(GATE_ROOT, "ws_pipe_%s" % backend)
    base_snap = ws + "_c0"

    # step0: fresh copies=0 baseline
    wipe(ws, base_snap)
    run_pipeline(backend, ws, 0)
    snap_tree(ws, base_snap)
    check("%s: baseline has no aug artifacts" % backend,
          not [r for r in list_rel(base_snap) if is_aug_rel(r)]
          # aug_meta lives inside the pool now, so "is there one" is asked of the whole tree
          # rather than of one hard-coded depth — otherwise this check would pass by looking in
          # a place where the directory can no longer appear.
          and not [r for r in list_rel(base_snap) if r.replace("\\", "/").split("/")[-2:-1] == ["aug_meta"]])

    # step2: fresh copies=2
    wipe(ws)
    run_pipeline(backend, ws, 2)
    # ⚠ AFTER the run: the pool directory does not exist until the pipeline resolves it.
    P = backend_paths(backend, ws)
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
    metas = [n for n in os.listdir(P["meta"])
             if n.endswith(".json") and not n.startswith("_")]
    check("%s: meta count == surviving aug count" % backend,
          len(metas) == len(aug_wavs), "(%d/%d)" % (len(metas), len(aug_wavs)))

    # rerun2: cache semantics — feature caches live in the slice dir (sovits)
    # or sibling dirs (rvc 2a/2b/3_feature*, vocoder npz)
    slice_dir = P["slice_dir"]
    cache_roots = P["cache_roots"]
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
    P = backend_paths("sovits", ws)
    spk = P["slice_dir"]
    kept = [n for n in os.listdir(spk) if n.endswith(".wav") and is_aug_rel(n)]
    metas = [n for n in os.listdir(P["meta"])
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
    spk = backend_paths("sovits", ws)["slice_dir"]
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

    # ★★ §F2⒝ — this used to be "cache-wipe: different dataset -> fingerprint change -> full
    # rebuild", and it pinned the behaviour this batch exists to remove: a parameter or dataset
    # change `shutil.rmtree`'d hours of preprocessing. The replacement pins the NEW guarantee, in
    # the same place, driven by the same production pipeline:
    #     a different identity gets a SIBLING pool, and the first one is still there, byte for byte.
    # (The half of the old check that is still meaningful — "the new material really does get
    # built, with diff products" — is kept below and simply asks it of the new pool.)
    before_pools = noop.pools_in(ws)
    old_pool_shape = {
        r: open(os.path.join(spk, r), "rb").read()
        for r in list_rel(spk, lambda r: r.endswith(".wav"))
    }
    dirty_ds = os.path.join(GATE_ROOT, "dataset_dirty")
    run_pipeline("sovits_diff", ws, 2, dataset_dir=dirty_ds)
    after_pools = noop.pools_in(ws)
    check("diff: a different dataset mints a SIBLING pool (it does not wipe the old one)",
          len(after_pools) == len(before_pools) + 1,
          "(%s -> %s)" % (before_pools, after_pools))
    check("★ diff: every wav of the previous pool survived, byte for byte",
          all(os.path.isfile(os.path.join(spk, r))
              and open(os.path.join(spk, r), "rb").read() == v
              for r, v in old_pool_shape.items()),
          "(%d wavs)" % len(old_pool_shape))
    new_id = [p for p in after_pools if p not in before_pools][0]
    new_spk = os.path.join(ws, "pools", new_id, "dataset_44k", "gateaug")
    aug2 = list_rel(new_spk, lambda r: is_aug_rel(r) and r.endswith(".wav"))
    check("diff: the new pool really was built (aug slices present)", len(aug2) > 0,
          "(%d)" % len(aug2))
    check("diff: the new pool's aug have diff products",
          all(os.path.exists(os.path.join(new_spk, r.replace(".wav", ".wav.aug_mel.npy")))
              for r in aug2))


def main():
    ap = argparse.ArgumentParser()
    # ⛔ `choices=` mirrors gate_aug0_driver.py, which has always had it. A mistyped
    # backend used to reach `backend_paths` and die with a one-word message.
    ap.add_argument("--backend", default="all", choices=("all",) + DRIVABLE)
    args = ap.parse_args()
    backends = list(DRIVABLE) if args.backend == "all" else [args.backend]
    noop.ensure_fixture()
    for b in backends:
        exercise(b)
    if args.backend in ("all", "sovits"):
        dirty_rejection()
        diff_inherit()
    if FAILURES:
        print("RESULT: FAIL (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    # ⛔ Only meaningful for the full run; a single-backend run legitimately prints fewer.
    if args.backend == "all" and len(CHECKS) < MIN_CHECKS_ALL:
        print("GATE-UNRUNNABLE: only %d checks ran (floor %d) — checks behind `if` gates "
              "(val / index / feature-cache mtimes) DISAPPEAR rather than go red when a path "
              "stops resolving, so a short green run is not a green run." % (len(CHECKS),
                                                                             MIN_CHECKS_ALL))
        return noop.EXIT_UNRUNNABLE
    print("RESULT: ALL PASS (%d checks)" % len(CHECKS))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except noop.GateUnrunnable as e:
        print("GATE-UNRUNNABLE: %s" % e)
        sys.exit(noop.EXIT_UNRUNNABLE)
