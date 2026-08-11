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

#: Floor on the checks ONE chain's identity-v2 ladder must print. Counted, not guessed:
#: 4 (step0) + 24 (three rungs, including the cumulative 「earlier pools intact」 checks)
#: + 3 (back to zero) + 3-4 (positive observations) = 34-35.
MIN_CHECKS_V2 = 30


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    CHECKS.append(name)
    if not ok:
        FAILURES.append(name)


#: The identity-v2 arm writes EVERYTHING under this subdirectory of GATE_ROOT.
#:
#: ⛔ Not a `_v2` suffix on the file names, and that is not a style choice: for
#: backend="sovits" the obvious name `cfg_pipe_sovits_v2.json` **is already taken** — it is
#: `legs_s129.py`'s hand-written template for its fourth chain (`workspace` literally reads
#: "SET-BY-THE-LEG"). `run_pipeline` overwrites its cfg path unconditionally, so the v2 arm
#: would have eaten that fixture, and the damage would have been SILENT: the ladder's last
#: step is copies=0, `|aug=` is only appended above 0, so the leg would go on stamping
#: `aug_copies: 0` into its manifest and stay green while no longer covering ④d's token.
#:
#: A whole separate directory also makes 「did the v2 arm touch anything legs owns」 a
#: one-line question instead of a per-name audit.
IDV2_DIR = "idv2"


def arena(identity_version):
    """Where this arm's workspaces and configs live. v1 == GATE_ROOT, byte-for-byte as before."""
    if identity_version is None or int(identity_version) < 2:
        return GATE_ROOT
    d = os.path.join(GATE_ROOT, IDV2_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def run_pipeline(backend, ws, copies, dataset_dir=None, identity_version=None):
    cfg = noop.build_cfg(backend, ws, identity_version=identity_version)
    cfg["aug_copies"] = int(copies)
    if dataset_dir:
        cfg["dataset_dir"] = dataset_dir
    # ⛔ The cfg lands NEXT TO the workspace it configures, never at `arena(identity_version)`.
    # Those two can disagree, and when they do the damage is silent and lands on someone else:
    # a caller running this ladder with the knob OFF but the v2 workspaces ON writes
    # `cfg_pipe_rvc.json` back into GATE_ROOT — which is `legs_s129.py`'s template, and legs
    # reads `aug_copies` out of it to stamp `run_manifest.json`.
    # ⚠ Measured, not imagined: S136's own negative control did exactly this and moved
    # cfg_pipe_rvc.json's `aug_copies` from 0 to 2 before the drift checker caught it.
    # Deriving the directory from `ws` makes the two structurally incapable of disagreeing.
    cfg_path = os.path.join(os.path.dirname(ws), "cfg_pipe_%s.json" % backend)
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

    ⛔ Keeps `noop.pool_of`'s 「exactly one pool」 refusal, and that refusal is a REAL criterion
    for the v1 arm: one identity, one cold run, one pool. The identity-v2 arm legitimately holds
    several sibling pools at once (that is the whole of ④d), so it calls `backend_paths_in` with
    the pool it means — it does not get to relax this one.
    """
    return backend_paths_in(backend, ws, noop.pool_of(ws))


def backend_paths_in(backend, ws, pool):
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


# ─── identity-v2 arm ────────────────────────────────────────────────────────────────────
#
# ⛔ This is NOT 「the same ladder with a knob」. Under ④d `aug_copies` is part of the pool
# identity, so each copies value gets its own SIBLING pool — which means two of the v1
# ladder's checks stop asking a question that exists:
#   · 「copies=3 preserves aug1/2 bytes」 was about INCREMENTAL growth inside one pool.
#     Under v2 copies=3 is a different pool, so asserting the old pool is untouched is
#     VACUOUSLY true — and it is exactly what ④d guarantees anyway, so it is re-asked here
#     as an explicit 「every earlier pool survives byte-for-byte」.
#   · 「copies=1 removes ALL idx>=2 products」 was about `augment_slices`' stale pruning.
#     Under v2 nothing is pruned (the low count gets a fresh pool), and the v1 check would
#     go RED for a correct implementation because it scans the WHOLE workspace and finds
#     the other pools' legitimate aug2/aug3. That pruning branch is still live in
#     production for un-migrated slots (a brand-new slot is born v1, `tpool.rs`), so the
#     **v1 arm keeps it** — it is the only coverage of a shape that still ships.
#
# What this arm pins is what `pool.py:204-208` says `|aug=` was added FOR: two runs with
# different counts no longer destroy each other's products.


def _pool_api():
    """Production's own identity helpers. ⛔ Never re-implement the formula here — the five
    chains build `fp_text` five different ways, and a sixth copy in a gate would drift.
    Precedent: `gate_pool_table.py` imports `pool_id_for` the same way."""
    sys.path.insert(0, os.path.join(APP, "training"))
    from utai_train.pool import FINGERPRINT, POOLS_DIR, identity_suffix, pool_id_for
    return identity_suffix, pool_id_for, POOLS_DIR, FINGERPRINT


def _sr_hz(cfg):
    """The value production feeds `identity_suffix` — rvc only, and it is the Hz int, not the
    "48k" UI string (`pool.py:200-202` explains why routing it through a display string would
    be a second encoding of one fact)."""
    if cfg.get("backend") != "rvc":
        return None
    sys.path.insert(0, os.path.join(APP, "training"))
    from utai_train.rvc.pipeline import SR_MAP
    return SR_MAP[cfg["sample_rate"]]


def _pool_dir(ws, pid):
    _s, _p, pools_dir, _f = _pool_api()
    return os.path.join(ws, pools_dir, pid)


def pool_fp(ws, pid):
    _s, _p, _d, fname = _pool_api()
    with open(os.path.join(_pool_dir(ws, pid), fname), encoding="utf-8") as f:
        return f.read().strip()


def snapshot_pool(ws, pid):
    """{rel: sha256} of everything in one pool. Content, not mtime: a rebuild that produced
    identical bytes is not a violation of anything ④d promises."""
    import hashlib
    root = _pool_dir(ws, pid)
    out = {}
    for rel in list_rel(root):
        h = hashlib.sha256()
        with open(os.path.join(root, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[rel] = h.hexdigest()
    return out


def assert_pool_intact(ws, pid, before, label):
    now = snapshot_pool(ws, pid)
    lost = sorted(set(before) - set(now))
    changed = sorted(r for r in set(before) & set(now) if before[r] != now[r])
    # ⛔ A floor, because `all(... for r in {})` is the vacuous-green shape this whole arm
    # exists to avoid: an empty `before` would make every one of these checks free.
    if not before:
        check(label, False, "(the snapshot was EMPTY — this check would have been free)")
        return
    check(label, not lost and not changed,
          "" if not (lost or changed) else "lost=%s changed=%s" % (lost[:4], changed[:4]))


def observe_v2(backend, ws, pool_ids, fp_of):
    """⭐ Positive proof that the v2 code path was actually taken — not just that nothing broke.

    ⛔ Per chain, because they are not equally observable, and pretending otherwise is how a
    structurally-empty criterion gets counted as coverage:
      · rvc         `|sr=` is UNCONDITIONAL ⇒ observable at every count, including 0.
      · sovits fam  the single-speaker slice dir is renamed to `pool.SOLE_SPEAKER_DIR`.
      · vocoder     at copies=0 there is NOTHING: no `|sr=`, no `|aug=` (omitted at 0), no
                    `dataset_44k` at all. Its v1 and v2 strings are byte-identical there.
                    The only knob that makes it observable is copies>0 — which this ladder
                    does drive, so vocoder DOES get real v2 coverage here (unlike the
                    copies=0-only `gate_aug0_noop`).
    """
    if backend == "rvc":
        check("%s v2: fingerprints carry the unconditional |sr= token" % backend,
              all("|sr=" in fp_of[n] for n in pool_ids), str(fp_of))
    elif backend in ("sovits", "sovits_diff", "sovits_v2"):
        sys.path.insert(0, os.path.join(APP, "training"))
        from utai_train.pool import SOLE_SPEAKER_DIR
        for n, pid in pool_ids.items():
            d44 = os.path.join(_pool_dir(ws, pid), "dataset_44k")
            subs = sorted(os.listdir(d44)) if os.path.isdir(d44) else []
            check("%s v2: copies=%d slices live under the constant %r, not the run's name"
                  % (backend, n, SOLE_SPEAKER_DIR), subs == [SOLE_SPEAKER_DIR], str(subs))
    # every chain: above 0 the shared suffix must actually be there
    for n in sorted(pool_ids):
        if n > 0:
            check("%s v2: the copies=%d fingerprint ends with |aug=%d" % (backend, n, n),
                  fp_of[n].endswith("|aug=%d" % n), fp_of[n])
    if backend == "vocoder":
        print("  [NOTE] %s copies=0 carries NO v1/v2 observable (no |sr=, |aug= omitted at 0, "
              "no dataset_44k) — that格 is a structurally empty criterion and is not counted "
              "as v2 coverage." % backend)


def exercise_v2(backend):
    print("== %s [identity v2]" % backend)
    identity_suffix, pool_id_for, _pools_dir, _fname = _pool_api()
    ws = os.path.join(arena(2), "ws_pipe_%s" % backend)
    wipe(ws)
    cfg = noop.build_cfg(backend, ws, identity_version=2)
    sr = _sr_hz(cfg)

    def suffix(n):
        return identity_suffix(cfg, n, sample_rate=sr)

    # ── step0: the un-augmented identity ────────────────────────────────────────────────
    run_pipeline(backend, ws, 0, identity_version=2)
    ids = noop.pools_in(ws)
    check("%s v2: copies=0 minted exactly one pool" % backend, len(ids) == 1, str(ids))
    if len(ids) != 1:
        return
    p0 = ids[0]
    fp0 = pool_fp(ws, p0)
    s0 = suffix(0)
    check("%s v2: the copies=0 fingerprint ends with the shared suffix %r" % (backend, s0),
          fp0.endswith(s0), fp0)
    # ⭐ BASE is DERIVED from the disk, never re-computed: the five chains build the part
    # before the shared suffix five different ways, and a sixth copy here would be the
    # 「两侧同意一起漂」 shape `gate_pool_table.py` warns about.
    base = fp0[:len(fp0) - len(s0)] if s0 else fp0
    check("%s v2: the pool's directory name IS pool_id_for(its fingerprint)" % backend,
          p0 == pool_id_for(fp0), "%s vs %s" % (p0, pool_id_for(fp0)))

    seen = {0: p0}
    fps = {0: fp0}
    snaps = {0: snapshot_pool(ws, p0)}
    check("%s v2: the copies=0 pool is not empty" % backend, len(snaps[0]) > 0,
          "(%d files)" % len(snaps[0]))

    # ── the ladder: every count gets its OWN pool, and the earlier ones must survive ─────
    for n in (2, 3, 1):
        run_pipeline(backend, ws, n, identity_version=2)
        want_fp = base + suffix(n)
        want_id = pool_id_for(want_fp)
        live = noop.pools_in(ws)
        check("%s v2: copies=%d resolved to the pool named by BASE+%r" % (backend, n, suffix(n)),
              want_id in live, "want %s, live %s" % (want_id, live))
        if want_id not in live:
            return
        check("%s v2: …and that pool's fingerprint is EXACTLY that string" % backend,
              pool_fp(ws, want_id) == want_fp, pool_fp(ws, want_id))
        check("%s v2: …and it is a NEW sibling, not one of the earlier pools" % backend,
              want_id not in seen.values(), "%s in %s" % (want_id, seen))
        # ⭐⭐ THE ④d guarantee: `pool.py:206-208` — 「two runs sharing a pool with different
        # counts destroy each other's products」. This is the check that says they no longer do.
        for m, pid in sorted(seen.items()):
            assert_pool_intact(ws, pid, snaps[m],
                               "%s v2: copies=%d left the copies=%d pool byte-intact"
                               % (backend, n, m))
        seen[n] = want_id
        fps[n] = want_fp
        snaps[n] = snapshot_pool(ws, want_id)

        P = backend_paths_in(backend, ws, _pool_dir(ws, want_id))
        augs = list_rel(P["slice_dir"], lambda r: is_aug_rel(r) and r.endswith(".wav"))
        check("%s v2: copies=%d pool holds aug slices" % (backend, n), len(augs) > 0,
              "(%d)" % len(augs))
        idxs = sorted({aug_idx(r) for r in augs})
        check("%s v2: copies=%d pool holds exactly aug1..aug%d" % (backend, n, n),
              idxs == list(range(1, n + 1)), str(idxs))
        metas = [x for x in os.listdir(P["meta"])
                 if x.endswith(".json") and not x.startswith("_")]
        check("%s v2: copies=%d meta count == aug count" % (backend, n),
              len(metas) == len(augs), "(%d/%d)" % (len(metas), len(augs)))

    # ── back to zero: the ORIGINAL pool is re-selected, byte for byte ────────────────────
    run_pipeline(backend, ws, 0, identity_version=2)
    check("%s v2: copies back to 0 re-selects the SAME pool it started with" % backend,
          p0 in noop.pools_in(ws), "%s not in %s" % (p0, noop.pools_in(ws)))
    assert_pool_intact(ws, p0, snaps[0],
                       "%s v2: …and that pool is byte-identical to before the whole ladder"
                       % backend)
    check("%s v2: the ladder left exactly the 4 pools it declared" % backend,
          sorted(noop.pools_in(ws)) == sorted(seen.values()),
          "live=%s declared=%s" % (sorted(noop.pools_in(ws)), sorted(seen.values())))

    observe_v2(backend, ws, seen, fps)


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


#: Which chains the identity-v2 ladder can drive today.
#: ⛔ The sovits family is NOT here yet and that is deliberate, not an oversight: under v2 its
#: single-speaker slices move to `pool.SOLE_SPEAKER_DIR`, and `backend_paths_in` still hard-codes
#: `dataset_44k/gateaug` in two places (plus `diff_inherit`'s third). Adding the name without
#: that handling would not go red — `list_rel` returns `[]` for a missing directory, so the
#: counts would quietly be 0 and several `all(...)` checks would pass vacuously.
#: rvc and vocoder need no change at all: rvc has no speaker directory, vocoder has no
#: `dataset_44k`. (Confirmed against `flist.py:88-89`, which is sovits-family only.)
DRIVABLE_V2 = ("rvc", "vocoder")

#: What `legs_s129.py` reads out of GATE_ROOT. The v2 arm must not touch ANY of it.
LEGS_OWNED = ("ws_pipe_rvc", "ws_pipe_sovits", "ws_pipe_vocoder",
              "cfg_pipe_rvc.json", "cfg_pipe_sovits.json", "cfg_pipe_sovits_diff.json",
              "cfg_pipe_sovits_v2.json", "cfg_pipe_vocoder.json")


def legs_fixture_digest():
    """{rel: sha256} of everything legs_s129 consumes from GATE_ROOT. Missing entries are
    recorded as None so 「it vanished」 and 「it changed」 are different answers."""
    import hashlib
    out = {}
    for name in LEGS_OWNED:
        p = os.path.join(GATE_ROOT, name)
        if os.path.isfile(p):
            files = [(name, p)]
        elif os.path.isdir(p):
            files = [(os.path.join(name, r), os.path.join(p, r)) for r in list_rel(p)]
        else:
            out[name] = None
            continue
        for rel, fp in files:
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[rel] = h.hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    # ⛔ `choices=` mirrors gate_aug0_driver.py, which has always had it. A mistyped
    # backend used to reach `backend_paths` and die with a one-word message.
    ap.add_argument("--backend", default="all", choices=("all",) + DRIVABLE)
    ap.add_argument("--arm", default="v1", choices=("v1", "v2", "both"),
                    help="v1 = the pre-④d identity formula (the existing baseline, unchanged); "
                         "v2 = ④d's, where aug_copies is part of the pool identity")
    args = ap.parse_args()
    backends = list(DRIVABLE) if args.backend == "all" else [args.backend]
    noop.ensure_fixture()

    if args.arm in ("v2", "both"):
        # ⛔ The acceptance criterion S134 wrote for M15 —「跑完 v2 档之后 legs 的夹具没变」—
        # lives INSIDE the gate, not in my head. Its own version of this check was
        # 「pools/ 里仍然只有 v1 那个 id」, which is satisfied EXACTLY WHEN the contamination
        # happens: at aug=0 the sovits family's v2 fingerprint is byte-identical to v1, so a
        # v2 run against a v1 workspace resolves to the SAME pool and grows a second slice
        # tree INSIDE it. Hence: per-file digests, and 「extra file」 counts as damage.
        before = legs_fixture_digest()
        for b in [x for x in backends if x in DRIVABLE_V2]:
            n_before = len(CHECKS)
            exercise_v2(b)
            # ⛔ The ladder `return`s early at three points (one pool expected but N found;
            # the wanted pool absent). Those returns are right — carrying on would compare
            # against a pool nobody selected — but they leave a SHORT green run, and a short
            # green run reads exactly like a complete one. A floor, not a pin.
            ran = len(CHECKS) - n_before
            check("%s v2: the ladder ran to the end (%d checks, floor %d)"
                  % (b, ran, MIN_CHECKS_V2), ran >= MIN_CHECKS_V2)
        skipped = [x for x in backends if x not in DRIVABLE_V2]
        if skipped:
            print("  [NOTE] identity-v2 arm SKIPPED for %s — not yet wired (see DRIVABLE_V2). "
                  "⛔ Their v1 green says nothing about ④d." % ", ".join(skipped))
        check("v2 arm touched nothing legs_s129 owns", legs_fixture_digest() == before,
              "" if legs_fixture_digest() == before else "⛔ STOP — legs' 存量池 fixtures moved")

    if args.arm in ("v1", "both"):
        for b in backends:
            exercise(b)
        if args.backend in ("all", "sovits"):
            dirty_rejection()
            diff_inherit()
    if FAILURES:
        print("RESULT: FAIL (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    # ⛔ Only meaningful for the full v1 run; a single-backend or v2-only run legitimately
    # prints fewer, and the v2 ladder's own count is pinned by its own floor below.
    if args.arm in ("v1", "both") and args.backend == "all" and len(CHECKS) < MIN_CHECKS_ALL:
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
