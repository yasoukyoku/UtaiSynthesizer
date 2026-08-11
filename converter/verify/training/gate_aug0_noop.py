# -*- coding: utf-8 -*-
"""S41 gate_aug0_noop — with aug_copies=0 the WHOLE preprocessing product tree
must be byte-identical to the BASELINE code (design B5; red-team V1/V3/V4/V5/V6).

⛔⛔ S136: `--baseline-rev` defaults to `HEAD`, and that default was correct for
EXACTLY ONE DAY. This script and `utai_train/augment.py` were born in the same
commit (`af8ad8b`, 2026-07-07), so on the day it was written HEAD was still the
parent (`c82ca55`, 2026-07-06) = genuinely pre-augmentation code, and the aug
changes lived uncommitted in the working tree. The moment `af8ad8b` landed, the
same default turned into **the working tree compared against itself**: with a
clean tree both arms run byte-identical code and `RESULT: PASS — copies=0 tree
is byte-identical to HEAD` becomes true and empty.

⇒ The default is kept (it is the right answer while you are holding uncommitted
changes — that is what makes this a mutation detector), but the script now SAYS
which of the two things it measured, because one RESULT line meant both:
  · baseline-rev resolves to a commit != HEAD          -> a real cross-code claim
  · baseline-rev resolves to HEAD and the tree is clean -> A/A determinism only
The genuinely pre-augmentation revision is `c82ca55`; pass it explicitly to make
the S41 claim. ⚠ `sovits_v2` did not exist then (born `f599e76`, 2026-07-17, and
it reused `augment` from birth) — that chain has NO pre-aug baseline at all.

Cold-run protocol (anti-self-certification, V1):
  1. git worktree of BASELINE is the reference implementation; both sides run
     through gate_aug0_driver (pipeline.run orchestration layer, CPU-pinned)
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
import glob
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

#: Exit codes — the SAME four tiers `gate0_guard.py` established in S135, deliberately
#: not a second scheme. S129's first iron rule: 「闸自己跑不起来」and 「被测的东西不对」
#: must not report as the same red, or the second occurrence gets shrugged off.
#: ⛔ Before S136 this gate had only 0/1: a failed driver (`run_side`), a slot the
#: normaliser refuses (`pool_prefix`), and a genuine byte difference all exited 1.
EXIT_PASS = 0
EXIT_RED = 1           # the thing under test is wrong
EXIT_UNRUNNABLE = 3    # the gate could not run / this reading is not attributable
EXIT_SELFTEST = 4      # this script's own self-check failed

#: Floor on how many files a comparison must have looked at. ⛔ Two empty trees make
#: `set()==set()` ⇒ zero iterations ⇒ zero diffs ⇒ `RESULT: PASS`. That is the exact
#: shape S68 judged [major] and S135 had to fix across gate0; it was never carried
#: over here. **A floor, not a pin** — counts are allowed to grow (S125 added
#: `pool.json`, so today every chain has one more file than the README records).
MIN_COMPARED_FILES = 10

#: The backends `build_cfg` can actually build a config for. Used for `--backend choices=`
#: here and in `gate_aug_pipeline`. ⛔ This is a SECOND place that lists them, so
#: `--selftest` asserts every name here really gets a config out of `build_cfg` — a
#: choices list that has drifted from the dispatch is worse than no choices list.
BACKENDS = ("rvc", "sovits", "sovits_diff", "vocoder")


class GateUnrunnable(RuntimeError):
    """This reading is not attributable. ⛔ Must never be read as 『通过』."""


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
        # §F2⒝ batch 2 — the RUN directory. `workspace` is the SLOT (python resolves
        # `<slot>/pools/<id>/` against it); this is where the run's own products go.
        #
        # ★ Equal to the workspace here, and that is the PRODUCTION shape rather than a shortcut:
        # a slot with no `runs/` container IS its own single run (`trun::resolve_run_dir`), which is
        # every slot that has not been through the layout-3 migration. So this gate keeps comparing
        # byte-for-byte trees across a baseline worktree that predates the key — both halves write
        # to the same directory — while still exercising the split that batch 2 introduced.
        "run_dir": workspace,
        # Whether the SLOT holds a main model at all. These fixtures are built fresh per backend,
        # so the honest answer is False — and the shallow-diffusion chain needs it stated rather
        # than inferred from a missing `config.json`, which per-run can no longer be told apart
        # from「pointed at the wrong run」. The `sovits` → `sovits_diff` pair in `smoke_aug` reuses
        # one workspace on purpose and re-derives this before the second leg.
        "run_has_main_model": bool(glob.glob(os.path.join(workspace, "G_*.pth"))),
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
        raise GateUnrunnable("backend %s not wired yet" % backend)
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
        # ⛔ NOT a red: one side of the comparison never produced a tree, so there is
        # nothing to compare. Reporting this as 1 (like a byte difference) is what
        # S129's iron rule forbids.
        raise GateUnrunnable("%s side failed (rc=%d) — no tree was produced, so this "
                             "run says nothing about the copies=0 no-op property"
                             % (label, r.returncode))


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
        # ⛔ The gate cannot normalise this shape — that is a statement about the GATE,
        # not about the products. Exit 3, never 1.
        raise GateUnrunnable(
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


def compare_trees(base, ours, min_files=MIN_COMPARED_FILES):
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
                    raise GateUnrunnable(
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
    seen = set(a) | set(b)
    print("compared %d files: %d identical, %d problems" % (len(seen), same_ct, len(bad)))
    # ⛔ The floor goes here and not in main(): both trees empty ⇒ zero diffs ⇒ `bad`
    # is empty ⇒ every caller reads that as PASS. `gate_aug_pipeline.tree_equal_to`
    # would turn it into a green `[PASS]` line with no hint that it looked at nothing.
    if len(seen) < min_files:
        raise GateUnrunnable(
            "only %d file(s) to compare (floor %d) — a comparison this small cannot carry the "
            "claim. base=%s ours=%s. ⚠ Two empty trees compare equal, so this is a refusal, "
            "not a PASS." % (len(seen), min_files, base, ours)
        )
    return bad


def assert_no_reparse_points(root):
    """⛔ Refuse to hand a tree to `git worktree remove --force` if anything in it is a link.

    This repository has one catastrophic incident on record and this is its exact operation:
    `git worktree remove --force` walked THROUGH a hand-made junction and emptied `data/`,
    `runtime/` and `bin/` in the real checkout. Nothing in this script creates a link today, so
    this costs one directory walk and finds nothing — which is the point. The moment someone
    junctions a gitignored asset (models, runtimes) into the baseline worktree so the training
    code can find it, that walk is the difference between a loud refusal and the same accident.

    Symlinks AND junctions: `os.path.islink` misses a Windows junction, so the reparse-point bit
    is read directly from the stat result.
    """
    FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames) + filenames:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.islink(p) or (
                getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT
            ):
                found.append(p)
    if found:
        raise GateUnrunnable(
            "REFUSING to `git worktree remove --force` %s: it contains %d reparse point(s), and "
            "that removal walks through them into whatever they point at:\n  %s"
            % (root, len(found), "\n  ".join(found[:10]))
        )
    return len(found)


def describe_baseline_axis(baseline_rev):
    """WHICH of the two things this run is about to measure. Printed, never inferred.

    ⛔ One `RESULT:` line used to mean both 「今天的码与 pre-aug 的码产物相同」 and
    「同一份码跑两遍结果相同」, and which one you got depended on a default plus the
    state of the working tree. That is the shape S135 spent a whole session buying
    back on gate0 (「删目录 = 正确地红,清空目录 = 假 PASS」), one level up: the
    reading looks perfect in both cases.
    """
    def git(*a):
        r = subprocess.run(["git", "-C", APP] + list(a), capture_output=True, text=True)
        return (r.stdout or "").strip() if r.returncode == 0 else None

    base_sha = git("rev-parse", baseline_rev)
    head_sha = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    if base_sha is None:
        raise GateUnrunnable("cannot resolve --baseline-rev %r" % baseline_rev)
    if base_sha != head_sha:
        return ("CROSS-CODE", "baseline %s != HEAD %s — this run compares two different "
                              "revisions of training/" % (base_sha[:12], head_sha[:12]))
    if dirty:
        return ("MUTATION", "baseline == HEAD but the working tree is dirty (%d changed path(s)) "
                            "— this run compares HEAD against your uncommitted changes"
                            % len(dirty.splitlines()))
    return ("A/A", "baseline == HEAD and the working tree is clean — ⛔ both arms run BYTE-"
                   "IDENTICAL code, so this run measures rerun determinism ONLY and says "
                   "NOTHING about the copies=0 no-op property. Pass --baseline-rev c82ca55 "
                   "(the pre-augmentation parent of af8ad8b) to make that claim.")


def main():
    ap = argparse.ArgumentParser()
    # ⛔ `choices=` mirrors gate_aug0_driver.py, which has always had it. Without it a
    # mistyped backend name fell through to build_cfg's `raise ... not wired yet` — the
    # same sentence and the same exit code as 「这条链真没接线」.
    ap.add_argument("--backend", default="sovits", choices=sorted(BACKENDS))
    ap.add_argument("--baseline-rev", default="HEAD")
    args = ap.parse_args()

    axis, why = describe_baseline_axis(args.baseline_rev)
    print("[axis] %s — %s" % (axis, why))

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
        # ⛔ S124 added `assert_no_reparse_points` for exactly the `--force` removal in the
        # `finally` below and NEVER CALLED IT (34 lines of function body, zero call sites,
        # from 4e2a9d9 all the way to S136). A guard that has never executed is an empty
        # criterion — S129's iron rule, in the one place where the failure mode on record
        # is S96 emptying `data/`, `runtime/` and `bin/`.
        assert_no_reparse_points(wt)
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
            return EXIT_RED
        print("RESULT: PASS [%s] — copies=0 tree is byte-identical to %s"
              % (axis, args.baseline_rev))
        return EXIT_PASS
    finally:
        # Re-checked immediately before the destructive call: the baseline run above is
        # what could have created a link in there.
        #
        # ⚠ If this refuses, the worktree is deliberately LEFT BEHIND (registered in
        # `.git/worktrees`) — removing it is the exact operation being refused. Say so,
        # because a leftover registration makes the NEXT run fail in a stranger way.
        try:
            assert_no_reparse_points(wt)
        except GateUnrunnable:
            print("⛔ leaving the baseline worktree at %s in place ON PURPOSE.\n"
                  "   Inspect what the reparse point(s) point at, delete them BY HAND, then\n"
                  "   `git -C %s worktree remove --force %s` (or `worktree prune`)." % (wt, APP, wt))
            raise
        subprocess.run(
            ["git", "-C", APP, "worktree", "remove", "--force", wt],
            capture_output=True,
        )


def selftest():
    """Actually trigger every refusal branch once. S129: 一条从没被执行过的错误分支就是一条空判据。

    ⛔ Every check here must be able to FAIL. A self-test that cannot go red (e.g. because
    the OS refused to create the junction it needed) reports that as a self-test failure,
    never as a pass — that is the whole point of the exercise.
    """
    fails = []

    def check(name, ok, detail=""):
        print("  [%s] %s %s" % ("OK" if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    # ⑴ choices list vs the dispatch it claims to mirror — both directions.
    for b in BACKENDS:
        try:
            cfg = build_cfg(b, os.path.join(tempfile.gettempdir(), "s136_selftest_ws"))
            check("build_cfg builds %r" % b, isinstance(cfg, dict) and cfg.get("backend") == b)
        except GateUnrunnable as e:
            check("build_cfg builds %r" % b, False, str(e))
    try:
        build_cfg("__not_a_backend__", os.path.join(tempfile.gettempdir(), "s136_selftest_ws"))
        check("build_cfg refuses an unknown backend", False, "(it returned a config)")
    except GateUnrunnable:
        check("build_cfg refuses an unknown backend", True)

    # ⑵ the empty-set floor — the trap this gate shipped with.
    d = tempfile.mkdtemp(prefix="s136_floor_")
    try:
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        os.makedirs(a)
        os.makedirs(b)
        try:
            compare_trees(a, b)
            check("two empty trees are refused, not PASSed", False,
                  "(compare_trees returned no diffs ⇒ callers read it as PASS)")
        except GateUnrunnable as e:
            check("two empty trees are refused, not PASSed", "floor" in str(e))
        # and the floor must not fire when there IS enough to look at
        for i in range(MIN_COMPARED_FILES):
            for side in (a, b):
                with open(os.path.join(side, "f%02d.bin" % i), "wb") as fh:
                    fh.write(b"x" * (i + 1))
        try:
            bad = compare_trees(a, b)
            check("a big enough comparison is not refused", not bad)
        except GateUnrunnable as e:
            check("a big enough comparison is not refused", False, str(e))
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ⑶ the reparse-point guard — the one S124 wrote and nobody ever called.
    d = tempfile.mkdtemp(prefix="s136_reparse_")
    try:
        target = os.path.join(d, "target")
        inside = os.path.join(d, "tree")
        os.makedirs(target)
        os.makedirs(inside)
        link = os.path.join(inside, "junction")
        # Directory junctions do not need administrator rights on Windows.
        rc = subprocess.run(["cmd", "/c", "mklink", "/J", link, target],
                            capture_output=True, text=True).returncode
        if rc != 0 or not os.path.exists(link):
            check("reparse guard fires on a junction", False,
                  "(could not create a junction to test with — this check is EMPTY, "
                  "not passing)")
        else:
            try:
                assert_no_reparse_points(inside)
                check("reparse guard fires on a junction", False, "(it walked straight past)")
            except GateUnrunnable as e:
                check("reparse guard fires on a junction", "REFUSING" in str(e))
            # negative side: a tree with no links must NOT be refused
            try:
                assert_no_reparse_points(target)
                check("reparse guard passes a clean tree", True)
            except GateUnrunnable as e:
                check("reparse guard passes a clean tree", False, str(e))
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ⑷ the baseline axis really distinguishes the two claims.
    axis_head, _ = describe_baseline_axis("HEAD")
    check("axis(HEAD) is A/A or MUTATION, never CROSS-CODE", axis_head in ("A/A", "MUTATION"),
          "(got %s)" % axis_head)
    axis_old, _ = describe_baseline_axis("HEAD~1")
    check("axis(HEAD~1) is CROSS-CODE", axis_old == "CROSS-CODE", "(got %s)" % axis_old)
    try:
        describe_baseline_axis("__no_such_rev__")
        check("axis refuses an unresolvable rev", False)
    except GateUnrunnable:
        check("axis refuses an unresolvable rev", True)

    print("SELFTEST: %s" % ("ALL OK" if not fails else "FAILED (%s)" % ", ".join(fails)))
    return EXIT_PASS if not fails else EXIT_SELFTEST


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    try:
        sys.exit(main())
    except GateUnrunnable as e:
        # ⛔ 3, never 1: the gate could not produce an attributable reading. Printing it
        # as a red is what makes the second occurrence get shrugged off.
        print("GATE-UNRUNNABLE: %s" % e)
        sys.exit(EXIT_UNRUNNABLE)
