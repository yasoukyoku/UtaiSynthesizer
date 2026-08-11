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
import tempfile

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
#: A floor, not a pin: today the v1 arm at `--backend all` prints 53 and it may grow.
#: ⚠ Counted over the v1 arm ALONE since S137 — `CHECKS` is shared with the v2 arm, which runs
#: first, so `len(CHECKS)` here used to be fed by the other arm entirely.
MIN_CHECKS_ALL = 48

#: Floor on the checks ONE chain's identity-v2 ladder must print. Counted, not guessed:
#: 4 (step0) + 3 rungs × (3 identity + 1..3 「earlier pools intact」 + 4 product-layer)
#: + 3 (back to zero) + 3..7 (positive observations, most for the sovits family) = 37..44.
#: ⛔ It is a floor on 「did the ladder run to the end」, NOT on coverage: all nine product-layer
#: checks can be RED and `ran` is unchanged. Per-chain expected totals live in the README.
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

def speaker_dir_for(backend, ws, identity_version):
    """The sovits family's SOLE-speaker slice sub-directory, PREDICTED from the version knob.

    ⛔ Predicted, never observed. Deriving it from 「whatever directory happens to sit under
    `dataset_44k`」 would make every product-layer check below SELF-CERTIFYING: production writes
    a name, the gate looks wherever that is, and a wrong name becomes indistinguishable from a
    right one. The single place the disk is consulted about this is `observe_v2`, and that is a
    CHECK, not a lookup.

    Production's decision is one line, `sovits/flist.py:88-89`:
        if len(out) == 1 and identity_version(cfg) >= 2: out[0]["slug"] = SOLE_SPEAKER_DIR
    · v1 keeps the run's own slug (`flist.py:85` ← `cfg["model_slug"]`). Taken from `build_cfg`
      rather than spelled "gateaug" here, so the gate cannot drift from its own config.
    · v2 is the constant, imported from production for the same reason `_pool_api` imports the
      identity helpers: a second copy of a cross-language constant is how the two agree to drift.

    ⚠ SOLE speaker only. A multi-speaker list keeps every slug, because those DO fold into the
    fingerprint (`sovits/pipeline.py:70-77`). This gate's fixture is one flat directory of wavs
    and `build_cfg` writes no `speakers` key, so `resolve_speakers` always returns exactly one —
    if that ever stops being true, the paths below stop meaning anything and `observe_v2`'s
    `subs == [<this>]` is what goes red.
    """
    if identity_version is not None and int(identity_version) >= 2:
        sys.path.insert(0, os.path.join(APP, "training"))
        from utai_train.pool import SOLE_SPEAKER_DIR
        return SOLE_SPEAKER_DIR
    return noop.build_cfg(backend, ws)["model_slug"]


def backend_paths(backend, ws, identity_version=None):
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
    return backend_paths_in(backend, ws, noop.pool_of(ws),
                            speaker_dir_for(backend, ws, identity_version))


def backend_paths_in(backend, ws, pool, speaker):
    """⛔ `speaker` is a REQUIRED positional and stays that way. A default would be a default for
    ONE identity version, and the version that got the default would be silently right while the
    other was silently wrong — `list_rel` answers `[]` for a directory that is not there, so the
    wrong answer arrives as a count of zero rather than as an error. Every call site must say
    which formula it is asking about. Non-sovits backends ignore it (rvc has no speaker
    directory, vocoder has no `dataset_44k` at all — `flist.py:88-89` is sovits-family only)."""
    if backend == "sovits":
        return {
            "slice_dir": os.path.join(pool, "dataset_44k", speaker),
            "meta": os.path.join(pool, "aug_meta"),
            # ⚠ NOT `speaker`-free: the companion feature caches (`.soft.pt` / `.f0.npy` /
            # `.spec.pt`) live next to the slices they describe, so they move with the rename.
            # `aug_meta` above does NOT — a sole speaker's meta is flat at the pool root
            # (`sovits/pipeline.py:81-90`), which is why it keeps its own hard-coded name.
            "cache_roots": [os.path.join(pool, "dataset_44k", speaker)],
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
    """Every file this pool held is still there, byte for byte — AND nothing was added to it.

    ⛔⛔ `gained` is not symmetry for its own sake, and leaving it out was this check's blind
    spot for the whole of S136. ④d has two halves and they fail in opposite directions:
      · `|aug=<n>` in the identity  → a run with a different count DELETES / OVERWRITES another
        run's products. Caught by `lost` + `changed`.
      · `SOLE_SPEAKER_DIR`          → a second run of the same slot GROWS A SECOND COMPLETE
        SLICE TREE inside one pool (`pool.py:100-107`, and `:676-679` below says exactly this).
        That is 100% ADDITION: nothing is lost, nothing changes bytes.
    So on the sovits family — the only family that has a speaker directory at all, and the one
    being wired into this arm now — the six checks labelled 「THE ④d guarantee」 were
    structurally incapable of seeing the accident they name. `legs_fixture_digest` below already
    uses the stricter ruler (`:679`: 「extra file」 counts as damage); it was guarding someone
    else's fixtures while this gate's own central claim used the loose one.
    ⚠ S136's only negative control for this family DELETED a file, so `lost` is the only arm
    that has ever executed. `--selftest` now fires all three (S129: an error branch that has
    never run is an empty criterion).
    """
    now = snapshot_pool(ws, pid)
    lost = sorted(set(before) - set(now))
    gained = sorted(set(now) - set(before))
    changed = sorted(r for r in set(before) & set(now) if before[r] != now[r])
    # ⛔ A floor, because `all(... for r in {})` is the vacuous-green shape this whole arm
    # exists to avoid: an empty `before` would make every one of these checks free.
    if not before:
        check(label, False, "(the snapshot was EMPTY — this check would have been free)")
        return
    check(label, not lost and not gained and not changed,
          "(%d files)" % len(before) if not (lost or gained or changed)
          else "lost=%s gained=%s changed=%s" % (lost[:4], gained[:4], changed[:4]))


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
    # ⚠ `sovits_diff` / `sovits_v2` are listed because the property is a family property, but
    # neither is reachable from the CLI today (they are not in `DRIVABLE`, so argparse rejects
    # them and the v2 loop never sees them). `sovits` is the only one of the three that has ever
    # executed this branch — first time in S137. Kept rather than trimmed so that wiring either
    # chain later does not need this to be re-derived; the NOTE in `main` is what keeps their
    # zero coverage visible in the meantime.
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
    # ⛔ REFUSE BEFORE the rmtree, not after it. `wipe(ws)` is the first thing this arm does, and
    # the ONLY thing separating it from `legs_s129.py`'s 存量池 fixtures is the `2` handed to
    # `arena()`. S136's blood lesson was precisely that decoupling (the knob went one way, the
    # arm's directory went the other, and the damage was silent). The guard that caught it —
    # `legs_fixture_digest` at `main`:680/:695 — is a DETECTOR, not a preventer: by the time it
    # reds the tree is already gone, and `backup_pre_m15` has no restore script.
    # ⚠ The stakes rose with sovits: `ws_pipe_sovits` is the only 存量池 fixture holding
    # `.spec.pt`/`.soft.pt`, and sovits' v2 copies=0 pool id is BYTE-IDENTICAL to the v1 one
    # (sole-speaker slug does not enter the fingerprint) — so a leak would not even mint a
    # new directory, it would grow a second slice tree inside the existing fixture.
    if os.path.normcase(os.path.dirname(ws)) == os.path.normcase(GATE_ROOT):
        raise noop.GateUnrunnable(
            "the identity-v2 arm resolved its workspace to %s — that is GATE_ROOT itself, where "
            "legs_s129.py's 存量池 fixtures live, and this arm's next statement is rmtree. "
            "Refusing. The v2 arm belongs under arena(2) = %s" % (ws, arena(2)))
    wipe(ws)
    cfg = noop.build_cfg(backend, ws, identity_version=2)
    sr = _sr_hz(cfg)
    # The slice sub-directory this arm EXPECTS production to use. Predicted from the knob, then
    # checked against the disk per rung (below) and per pool (`observe_v2`) — never read off it.
    speaker = speaker_dir_for(backend, ws, 2)

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

        P = backend_paths_in(backend, ws, _pool_dir(ws, want_id), speaker)
        # ⛔ Ask 「is this gate looking in the right place」 BEFORE asking anything about the
        # products, and print what is ACTUALLY on disk next to what was expected. Without it, a
        # slice directory the gate cannot find yields three checks that all read
        # 「production stopped producing aug slices」 — an accusation pointed at the wrong
        # component. That is the same half of S129's iron rule as an unattributable exit code:
        # 「闸找错了地方」 and 「被测的东西不对」 must not print the same sentence.
        # ⚠ A FAIL, not a `GateUnrunnable`: a name this gate predicted wrongly is a defect in
        # this gate, and it has to stay visible NEXT TO the evidence rather than abort the ladder.
        parent = os.path.dirname(P["slice_dir"])
        check("%s v2: copies=%d slice dir is where this gate predicted (%s)"
              % (backend, n, os.path.basename(P["slice_dir"])),
              os.path.isdir(P["slice_dir"]),
              "expected %s — %s holds %s" % (
                  P["slice_dir"], parent,
                  sorted(os.listdir(parent)) if os.path.isdir(parent) else "<no such directory>"))
        augs = list_rel(P["slice_dir"], lambda r: is_aug_rel(r) and r.endswith(".wav"))
        check("%s v2: copies=%d pool holds aug slices" % (backend, n), len(augs) > 0,
              "(%d)" % len(augs))
        idxs = sorted({aug_idx(r) for r in augs})
        check("%s v2: copies=%d pool holds exactly aug1..aug%d" % (backend, n, n),
              idxs == list(range(1, n + 1)), str(idxs))
        # ⛔ `aug_meta` deliberately does NOT branch on the speaker: a sole speaker's meta is flat
        # at the pool root (`sovits/pipeline.py:81-90`), so the rename does not move it. Guarded
        # rather than a bare `os.listdir` because a missing directory here used to be an uncaught
        # FileNotFoundError → exit 1 → the SAME code as a real byte difference (S129).
        have_meta = os.path.isdir(P["meta"])
        metas = [x for x in os.listdir(P["meta"])
                 if x.endswith(".json") and not x.startswith("_")] if have_meta else []
        check("%s v2: copies=%d meta count == aug count" % (backend, n),
              have_meta and len(metas) == len(augs),
              "(%s/%d)" % (len(metas) if have_meta else "no aug_meta dir", len(augs)))

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
    # `P["train"]` computes this exact path (:200). Two opinions about one fact in one function
    # is how they drift apart, and the bare `open` was also the third uncaught-FileNotFoundError
    # (⇒ exit 1 ⇒ the same code as a real byte difference) on this path.
    train = open(P["train"], encoding="utf-8").read().splitlines()
    train_aug = [l for l in train if is_aug_rel(l)]
    srcs = [n for n in os.listdir(spk) if n.endswith(".wav") and not is_aug_rel(n)]
    check("dirty: >=1 aug rejected", len(kept) < len(srcs), "(%d kept / %d sources)" % (len(kept), len(srcs)))
    # ⛔ The floor that makes the next three checks mean anything. 「every aug got culled」 is a
    # legitimate OUTCOME of the f0 gate, but it is not a legitimate READING here: with kept==[]
    # and train_aug==[], `>=1 aug rejected` passes (0 < 4), the materialisation check below is
    # `… if train_aug or kept else True` ⇒ vacuously True, `meta count matches kept` is 0 == 0,
    # and 「zero residue」 is trivially satisfied — four PASS lines describing nothing. This
    # fixture deliberately MIXES dirty and clean material so survivors are guaranteed; if none
    # survive, the fixture or the gate changed, and that is what needs saying.
    check("dirty: …and >=1 aug SURVIVED (without a survivor the next three checks are empty)",
          len(kept) > 0, "(%d kept)" % len(kept))
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
    # ⛔ Two of this gate's own second opinions about production's layout used to live in this one
    # line: the literal `"pools"` (production's `pool.POOLS_DIR`, already imported correctly by
    # `_pool_dir`) and the literal speaker slug. Both now come from the same places the rest of
    # the file uses. ⚠ `identity_version=None` on purpose — `diff_inherit` runs ONLY in the v1
    # arm (`main`), so resolving it to the v2 constant would be a guaranteed self-inflicted red.
    new_spk = os.path.join(_pool_dir(ws, new_id), "dataset_44k",
                           speaker_dir_for("sovits", ws, None))
    aug2 = list_rel(new_spk, lambda r: is_aug_rel(r) and r.endswith(".wav"))
    check("diff: the new pool really was built (aug slices present)", len(aug2) > 0,
          "(%d)" % len(aug2))
    check("diff: the new pool's aug have diff products",
          all(os.path.exists(os.path.join(new_spk, r.replace(".wav", ".wav.aug_mel.npy")))
              for r in aug2))


#: Which chains the identity-v2 ladder can drive today. ⛔ Must stay a subset of `DRIVABLE`:
#: `main` builds its universe from `DRIVABLE` (:769) and only FILTERS by this tuple, so a name
#: that is here and not there is an invisible no-op — it neither runs nor gets reported as
#: skipped. `--selftest` asserts the subset relation.
#:
#: ⛔⛔ `sovits_diff` and `sovits_v2` are still absent, and NOT because they are hard: adding
#: them means adding them to `DRIVABLE`, and `DRIVABLE` also feeds the **v1** arm (:797-802),
#: whose `run_pipeline` writes its config to `os.path.dirname(ws)` = GATE_ROOT (:110). For
#: b="sovits_v2" that path is `GATE_ROOT/cfg_pipe_sovits_v2.json` — legs_s129's HAND-WRITTEN,
#: unreproducible template for its fifth chain (`"workspace": "SET-BY-THE-LEG"`). The v1 arm has
#: no fixture digest around it, so that destruction is structurally invisible. Their absence is
#: reported every run by the NOTE in `main`, which diffs against `noop.BACKENDS` rather than
#: against this call's `--backend`, precisely so it cannot fall silent once the tuples match.
#:
#: ⚠ The comment that used to live here said adding a name 「would not go red — `list_rel`
#: returns [] … several `all(...)` checks would pass vacuously」. That describes `exercise()`
#: (the v1 arm), NOT this one: `exercise_v2` contains no `all(...)` over an aug collection. What
#: actually happened when sovits was wired in was **9 loud FAILs** (three per rung: no aug
#: slices / wrong index set / meta count) while `observe_v2` printed four PASSes saying the
#: slices were exactly where production put them. Red pointed at the wrong component, with its
#: own refutation printed underneath — worse to read than a vacuous green, not better. That is
#: what the per-rung 「slice dir is where this gate predicted」 check exists to name.
DRIVABLE_V2 = ("sovits", "rvc", "vocoder")

#: What `legs_s129.py` reads out of GATE_ROOT. The v2 arm must not touch ANY of it.
#: ⛔ `dataset` is on this list and it is not a workspace: `legs_s129.py:250` points EVERY leg's
#: `dataset_dir` at it, so it is the source of every pool id on this disk. Change one byte of it
#: and every leg's 「no new pool was minted」 goes red with no way to trace why. It was missing
#: from this list for as long as the list existed. (`dataset_dirty` is NOT a legs input — it is
#: only ever consumed by this gate's own v1 arm — so it deliberately stays off.)
LEGS_OWNED = ("ws_pipe_rvc", "ws_pipe_sovits", "ws_pipe_vocoder", "dataset",
              "cfg_pipe_rvc.json", "cfg_pipe_sovits.json", "cfg_pipe_sovits_diff.json",
              "cfg_pipe_sovits_v2.json", "cfg_pipe_vocoder.json")

#: Floor on how many files the legs digest must have hashed. ⛔ Without it the guard below is
#: `{} == {}`: an emptied / renamed / mistyped `LEGS_OWNED` produces two empty dicts and a green
#: 「touched nothing」 forever. `assert_pool_intact` (:300) and `noop.compare_trees` (:399) each
#: grew this same floor after the same lesson; the one guarding someone else's fixtures never
#: did. Today the three workspace trees plus the fixture dataset hash 86 files.
MIN_LEGS_FILES = 60


def legs_fixture_digest():
    """{rel: sha256} of everything legs_s129 consumes from GATE_ROOT. Missing entries are
    recorded as None so 「it vanished」 and 「it changed」 are different answers.

    ⚠ Content only. Two mtime-based criteria live in the v1 arm (:508 rerun mtime, :511 feature
    caches), so 「touched nothing」 here means 「changed no bytes」, not 「never opened」."""
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


def require_usable_legs_digest(before):
    """⛔ The floor on the guard itself. An emptied / renamed / mistyped `LEGS_OWNED` makes the
    check at the end of the v2 arm `{} == {}` — permanently, silently green, while the thing it
    claims to protect is unprotected. Separated from `main` so `--selftest` can really fire it
    (S129: an error branch that has never executed is an empty criterion)."""
    absent = sum(1 for v in before.values() if v is None)
    if len(before) < MIN_LEGS_FILES or all(v is None for v in before.values()):
        raise noop.GateUnrunnable(
            "the legs-fixture digest hashed only %d file(s) (floor %d, %d declared name(s) "
            "absent) — the guard that proves this arm did not touch legs_s129's 存量池 fixtures "
            "would be comparing two empty dicts and passing. LEGS_OWNED=%s"
            % (len(before), MIN_LEGS_FILES, absent, list(LEGS_OWNED)))


def require_nonempty_v2_arm(drivable, backend_arg, backends):
    """⛔ 「this call drove nothing」 must not be indistinguishable from 「everything passed」."""
    if not drivable:
        raise noop.GateUnrunnable(
            "the identity-v2 arm drove ZERO chains: --backend %s selected %s, and none of them "
            "is in DRIVABLE_V2=%s. No ④d criterion was evaluated, so this run says nothing about "
            "④d — it is not a pass." % (backend_arg, backends, list(DRIVABLE_V2)))


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
    # Initialised here, not inside the v1 block: the floor at the bottom is guarded by the same
    # `args.arm` test, so today it short-circuits — but if that guard is ever loosened, 0 makes
    # it fail CLOSED (「the v1 arm printed nothing」 ⇒ unrunnable) instead of NameError.
    v1_ran = 0

    if args.arm in ("v2", "both"):
        # ⛔ The acceptance criterion S134 wrote for M15 —「跑完 v2 档之后 legs 的夹具没变」—
        # lives INSIDE the gate, not in my head. Its own version of this check was
        # 「pools/ 里仍然只有 v1 那个 id」, which is satisfied EXACTLY WHEN the contamination
        # happens: at aug=0 the sovits family's v2 fingerprint is byte-identical to v1, so a
        # v2 run against a v1 workspace resolves to the SAME pool and grows a second slice
        # tree INSIDE it. Hence: per-file digests, and 「extra file」 counts as damage.
        before = legs_fixture_digest()
        require_usable_legs_digest(before)
        drivable = [x for x in backends if x in DRIVABLE_V2]
        try:
            for b in drivable:
                n_before = len(CHECKS)
                exercise_v2(b)
                # ⛔ The ladder `return`s early at two points (one pool expected but N found;
                # the wanted pool absent). Those returns are right — carrying on would compare
                # against a pool nobody selected — but they leave a SHORT green run, and a short
                # green run reads exactly like a complete one. A floor, not a pin.
                ran = len(CHECKS) - n_before
                check("%s v2: the ladder ran to the end (%d checks, floor %d)"
                      % (b, ran, MIN_CHECKS_V2), ran >= MIN_CHECKS_V2)
        finally:
            # ⛔ `finally`, because until S137 it was not: `exercise_v2`'s very first statement is
            # an rmtree and `run_pipeline` raises `GateUnrunnable` whenever a chain fails to reach
            # train_prep. So 「the pipeline crashed」 and 「this arm just corrupted legs' fixtures」
            # collapsed into one silence — the guard below simply never executed. sovits is the
            # first chain here that can realistically crash (four cold ContentVec + RMVPE runs).
            after = legs_fixture_digest()
            check("v2 arm touched nothing legs_s129 owns", after == before,
                  "(%d files)" % len(before) if after == before
                  else "⛔ STOP — legs' 存量池 fixtures moved: %s"
                       % sorted(set(k for k in set(after) | set(before)
                                    if after.get(k) != before.get(k)))[:6])
        # ⛔ 「this call drove nothing」 must not be indistinguishable from 「everything passed」.
        # Before S137, `--arm v2 --backend sovits` printed `RESULT: ALL PASS (1 checks)` and
        # exited 0 — the ladder never ran, `MIN_CHECKS_V2` lives INSIDE the loop so it was never
        # evaluated, and `MIN_CHECKS_ALL` is gated to the v1 arm. That is the same defect S136
        # bought back one file over in `smoke_aug.py:118-124` (`--only <unknown>` ⇒ zero criteria,
        # `RESULT: ALL PASS`), at the outermost layer of the gate that was supposed to have
        # learned it. Exit 3: nothing was measured, so this reading is not attributable.
        require_nonempty_v2_arm(drivable, args.backend, backends)
        # ⛔ Diffed against `noop.BACKENDS` (all five chains), NOT against this call's `backends`.
        # With the latter the line falls silent the moment `DRIVABLE_V2 == DRIVABLE` — which is
        # exactly what S137 made true — and it is the only sentence in the whole transcript that
        # says 「this chain has NO ④d coverage」. sovits_diff and sovits_v2 must keep being named
        # until they are genuinely wired, not until the two tuples happen to match.
        missing = [x for x in noop.BACKENDS if x not in DRIVABLE_V2]
        print("  [NOTE] identity-v2 coverage: %s driven this run; ⛔ STILL ZERO for %s "
              "(not wired — their v1 green says nothing about ④d)."
              % (", ".join(drivable) or "<none>", ", ".join(missing) or "<none>"))

    if args.arm in ("v1", "both"):
        # ⛔ Snapshot, because `CHECKS` is shared by both arms and the v2 arm runs FIRST. The
        # floor below used to read `len(CHECKS)`, so under `--arm both` the v2 ladder's ~100
        # checks fed it single-handedly and the v1 arm could have printed ZERO and still passed —
        # the precise failure this floor was created to catch, one level up. Measured, not
        # feared: v2-only already prints far more than MIN_CHECKS_ALL.
        v1_start = len(CHECKS)
        for b in backends:
            exercise(b)
        if args.backend in ("all", "sovits"):
            dirty_rejection()
            diff_inherit()
        v1_ran = len(CHECKS) - v1_start
    if FAILURES:
        print("RESULT: FAIL (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    # ⛔ Only meaningful for the full v1 run; a single-backend or v2-only run legitimately
    # prints fewer, and the v2 ladder's own count is pinned by its own floor above.
    if args.arm in ("v1", "both") and args.backend == "all" and v1_ran < MIN_CHECKS_ALL:
        print("GATE-UNRUNNABLE: only %d v1-arm checks ran (floor %d) — checks behind `if` gates "
              "(val / index / feature-cache mtimes) DISAPPEAR rather than go red when a path "
              "stops resolving, so a short green run is not a green run." % (v1_ran,
                                                                             MIN_CHECKS_ALL))
        return noop.EXIT_UNRUNNABLE
    print("RESULT: ALL PASS (%d checks)" % len(CHECKS))
    return 0


def selftest():
    """Fire every refusal branch this gate owns, once, for real.

    ⛔ S129's iron rule: 「一条从没被执行过的错误分支就是一条空判据」. Every guard below was
    added because the shape it catches had already happened somewhere in this repository, and
    each one is cheap to write and impossible to notice when it silently stops working. Runs in
    milliseconds, touches nothing outside a temp directory, and needs no fixtures.
    """
    fails = []

    def ok(name, cond, detail=""):
        print("  [%s] %s %s" % ("OK" if cond else "FAIL", name, detail))
        if not cond:
            fails.append(name)

    def capture(fn):
        """Run something that calls `check()` and return (names, failed_names) without letting
        it contaminate this process's real tallies."""
        c0, f0 = len(CHECKS), len(FAILURES)
        fn()
        names, failed = CHECKS[c0:], FAILURES[f0:]
        del CHECKS[c0:]
        del FAILURES[f0:]
        return names, failed

    # ⑴ DRIVABLE_V2 ⊆ DRIVABLE. A name in only the former neither runs nor gets reported as
    #    skipped — `main` builds its universe from DRIVABLE and merely FILTERS by DRIVABLE_V2.
    ok("DRIVABLE_V2 is a subset of DRIVABLE",
       set(DRIVABLE_V2) <= set(DRIVABLE), "%s vs %s" % (list(DRIVABLE_V2), list(DRIVABLE)))
    ok("every DRIVABLE_V2 name is reachable from --backend",
       all(b in ("all",) + DRIVABLE for b in DRIVABLE_V2))
    ok("DRIVABLE_V2 names all build a config",
       all(isinstance(noop.build_cfg(b, os.path.join(tempfile.gettempdir(), "s137_st"),
                                     identity_version=2), dict) for b in DRIVABLE_V2))

    # ⑵ the speaker-directory prediction really is version-dependent, and v1 comes from build_cfg
    sys.path.insert(0, os.path.join(APP, "training"))
    from utai_train.pool import SOLE_SPEAKER_DIR
    probe_ws = os.path.join(tempfile.gettempdir(), "s137_st")
    v1_name = speaker_dir_for("sovits", probe_ws, None)
    ok("speaker_dir_for(v1) == build_cfg's model_slug",
       v1_name == noop.build_cfg("sovits", probe_ws)["model_slug"], v1_name)
    ok("speaker_dir_for(v2) == production's SOLE_SPEAKER_DIR",
       speaker_dir_for("sovits", probe_ws, 2) == SOLE_SPEAKER_DIR, SOLE_SPEAKER_DIR)
    # ⚠ explicit 1 is NOT the same state as absent for `pool.identity_version`, but it IS the
    #   same slug — production writes the key unconditionally, so a field slot reads 1, not None.
    ok("speaker_dir_for(explicit 1) == the v1 name",
       speaker_dir_for("sovits", probe_ws, 1) == v1_name)
    ok("the two names differ (otherwise every check below is version-blind)",
       v1_name != SOLE_SPEAKER_DIR)
    ok("the sovits slice_dir really carries the speaker it was handed",
       backend_paths_in("sovits", probe_ws, "P", "zzz")["slice_dir"].endswith("zzz")
       and backend_paths_in("sovits", probe_ws, "P", "zzz")["cache_roots"][0].endswith("zzz"))
    ok("…and aug_meta does NOT (a sole speaker's meta is flat at the pool root)",
       not backend_paths_in("sovits", probe_ws, "P", "zzz")["meta"].endswith("zzz"))
    try:
        backend_paths_in("sovits", probe_ws, "P")  # noqa — the point is that this is an error
        ok("backend_paths_in refuses to guess a speaker", False, "(it accepted three args)")
    except TypeError:
        ok("backend_paths_in refuses to guess a speaker", True)

    # ⑶ assert_pool_intact fires on ALL THREE arms. ⛔ `gained` is why this exists: S136's only
    #    negative control for this family deleted a file, so until S137 `gained` and `changed`
    #    had never executed — and `gained` is the ONLY shape the sovits family's contamination
    #    (a second slice tree inside one pool) can take.
    d = tempfile.mkdtemp(prefix="s137_intact_")
    try:
        pool_root = os.path.join(d, "pools", "pTEST")
        os.makedirs(pool_root)
        for i in range(3):
            with open(os.path.join(pool_root, "f%d.bin" % i), "wb") as fh:
                fh.write(b"x" * (i + 1))
        snap = snapshot_pool(d, "pTEST")
        ok("snapshot_pool saw the files", len(snap) == 3, str(sorted(snap)))
        _n, failed = capture(lambda: assert_pool_intact(d, "pTEST", snap, "clean"))
        ok("intact: an untouched pool passes", not failed)
        with open(os.path.join(pool_root, "INTRUDER.bin"), "wb") as fh:
            fh.write(b"second slice tree")
        _n, failed = capture(lambda: assert_pool_intact(d, "pTEST", snap, "gained"))
        ok("intact: an ADDED file is damage ⭐ (the sovits contamination shape)", len(failed) == 1)
        os.remove(os.path.join(pool_root, "INTRUDER.bin"))
        with open(os.path.join(pool_root, "f1.bin"), "wb") as fh:
            fh.write(b"different")
        _n, failed = capture(lambda: assert_pool_intact(d, "pTEST", snap, "changed"))
        ok("intact: a CHANGED file is damage", len(failed) == 1)
        os.remove(os.path.join(pool_root, "f1.bin"))
        _n, failed = capture(lambda: assert_pool_intact(d, "pTEST", snap, "lost"))
        ok("intact: a LOST file is damage", len(failed) == 1)
        _n, failed = capture(lambda: assert_pool_intact(d, "pTEST", {}, "empty"))
        ok("intact: an EMPTY snapshot is refused, not a free pass", len(failed) == 1)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ⑷ the two outermost refusals — both of which used to be a silent `RESULT: ALL PASS`
    try:
        require_nonempty_v2_arm([], "sovits_v2", ["sovits_v2"])
        ok("a v2 arm that drove zero chains is refused", False, "(it returned)")
    except noop.GateUnrunnable as e:
        ok("a v2 arm that drove zero chains is refused", "ZERO chains" in str(e))
    try:
        require_usable_legs_digest({"a": "h1", "b": None})
        ok("an under-sized legs digest is refused", False, "(it returned)")
    except noop.GateUnrunnable as e:
        ok("an under-sized legs digest is refused", "floor" in str(e))
    try:
        require_usable_legs_digest(dict((str(i), None) for i in range(MIN_LEGS_FILES + 5)))
        ok("an all-absent legs digest is refused even above the floor", False, "(it returned)")
    except noop.GateUnrunnable as e:
        ok("an all-absent legs digest is refused even above the floor", "absent" in str(e))
    ok("…and a real digest of the declared fixtures passes that floor",
       _digest_floor_ok())

    # ⑸ the v2 arm refuses to rmtree anything living directly in GATE_ROOT
    real_arena = globals()["arena"]
    globals()["arena"] = lambda v: GATE_ROOT
    try:
        exercise_v2("rvc")
        ok("v2 arm refuses a workspace inside GATE_ROOT", False, "(it proceeded to wipe)")
    except noop.GateUnrunnable as e:
        ok("v2 arm refuses a workspace inside GATE_ROOT", "Refusing" in str(e), str(e)[:70])
    finally:
        globals()["arena"] = real_arena

    print("SELFTEST: %s" % ("ALL OK" if not fails else "FAILED (%s)" % ", ".join(fails)))
    return noop.EXIT_PASS if not fails else noop.EXIT_SELFTEST


def _digest_floor_ok():
    """The positive side of ⑷: today's real fixtures must clear the floor, or the floor is a
    trap rather than a guard."""
    try:
        require_usable_legs_digest(legs_fixture_digest())
        return True
    except noop.GateUnrunnable as e:
        print("       (real digest rejected: %s)" % e)
        return False


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    try:
        sys.exit(main())
    except noop.GateUnrunnable as e:
        print("GATE-UNRUNNABLE: %s" % e)
        sys.exit(noop.EXIT_UNRUNNABLE)
