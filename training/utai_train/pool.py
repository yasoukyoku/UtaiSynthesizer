"""Choosing the preprocessing POOL for a run — the single home for that decision.

Every training chain has always had a cache identity: a text built from the dataset fingerprint
plus whatever preprocessing parameters change what the products would be (see
`utai_train.cache`). Until now that identity lived in ONE directory per slot, so a run whose
identity differed from the stored one had exactly one way forward: ``shutil.rmtree`` the caches
and rebuild them. Flip ``loudnorm``, lose the slices; flip it back, pay for them again.

The identity now names a DIRECTORY instead of gating a deletion::

    <slot>/pools/<pool_id>/
        dataset.fingerprint   <- the identity text; the same file we have always written
        …the preprocessing products…

**Nothing about WHICH artifacts share an identity changed.** The fp_text formulas in the five
pipelines are untouched, so the pool boundary is byte-for-byte today's cache-invalidation
boundary. The only change is that a non-matching pool is now a SIBLING instead of a deletion.

## Why the slot root is itself a pool

A slot that has not been through the layout migration keeps its products at the slot root next to
a `dataset.fingerprint` — which is to say it *is* a pool, it just lives one level up. [`open_pool`]
therefore accepts it as one when the identity matches. Two things follow, both deliberate:

* every intermediate state of this change is production-correct — python can read pools before
  Rust starts creating them, and neither half has to wait for the other;
* a slot that somehow escapes migration (a data root swapped in from a backup, a project restored
  by hand) keeps its hours of preprocessing instead of silently rebuilding them.

⚠ This is NOT a "fall back to the old path when the new one is missing" arm — that shape is how a
forgotten wiring change becomes a silent regression. It fires only on the POSITIVE fact that the
slot root holds an identity file whose text equals this run's, and it says so in the log.

## The identity is the CONTENT, never the name

`pool_id_for` derives the directory name from the identity text so that Rust (which mints the
first pool during the layout migration, out of the `dataset.fingerprint` the flat slot already
had) and python (which mints every later one) agree on a name without coordinating. Selection
matches on the FILE CONTENT, so a pool whose name came from an older derivation keeps being
reused instead of silently abandoned — the name is a convenience, the file is the truth.

## Why the RUN directory is validated here too

§F2⒝ batch 3 split one directory into two: `workspace` is the SLOT (what `open_pool` resolves
against) and `run_dir` is where this run's own products go. Telling them apart is this module's
subject, and getting it backwards has exactly one symptom — a brand-new empty pool inside the run
and every preprocessing stage re-run, under a single `logger.info`. So [`checked_run_dir`] lives
beside [`open_pool`] rather than in each pipeline: one rule, five callers, and the fail-closed
posture is stated once.

⚠ The id charset is ``[p0-9a-f]`` on purpose. A pool path segment containing ``.wav`` would be
rewritten by ``filename.replace(".wav", ".spec.pt")`` (sovits/extract.py and both data_utils),
and ``wav`` / ``spec`` substrings break the RVC name surgery (rvc/extract_f0.py,
rvc/extract_feature.py). Hex cannot contain any of them, and the leading ``p`` keeps the name from
ever being read as a number. `src-tauri/src/training/tpool.rs` states the same three rules and
asserts them; `gate_pool_table.py` drives the two derivations against each other.
"""
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

#: Container for every pool of one slot. Must equal `tpool::POOLS_DIR`.
POOLS_DIR = "pools"

#: The identity file inside a pool. Must equal `tpool::FINGERPRINT`.
FINGERPRINT = "dataset.fingerprint"

#: Container for every RUN of one slot. Must equal `trun::RUNS_DIR`.
RUNS_DIR = "runs"


def checked_run_dir(cfg, slot_dir):
    """THE run directory for this run, fail-closed.

    ⛔ Never ``cfg.get("run_dir", cfg["workspace"])``. Every pipeline opens with
    ``os.makedirs(run_dir, exist_ok=True)`` and ``get_logger`` makes the directory too, so a
    missing, empty or misspelled value does not fail — it CREATES a directory and trains a whole
    run into it, out of sight of `trun::run_dirs` (which stops scanning the slot root the moment
    the container exists). That is the same shape as handing `open_pool` a run directory: silent,
    expensive, and announced by nothing.

    So the positive fact `trun::resolve_run_dir` establishes on the Rust side is re-asserted here:
    **the run is either the slot itself (layout ≤ 2, where the slot root IS the run) or a direct
    child of ``<slot>/runs/``.** Anything else means the caller built this config by hand — a gate
    driver that was not updated, or an old `run.json` from before the key existed — and training
    into whatever it names would be worse than refusing.
    """
    run_dir = cfg.get("run_dir")
    if not run_dir:
        raise RuntimeError(
            "RUN_DIR_MISSING: the run config has no 'run_dir'. Rust writes it beside 'workspace' "
            "(the slot); a hand-built config must set it too — to the slot itself for an "
            "unmigrated slot, or to <slot>/%s/<run_id>." % RUNS_DIR
        )
    norm = lambda p: os.path.normcase(os.path.abspath(p))
    if norm(run_dir) != norm(slot_dir) and norm(os.path.dirname(run_dir)) != norm(
        os.path.join(slot_dir, RUNS_DIR)
    ):
        raise RuntimeError(
            "RUN_DIR_NOT_IN_SLOT: %r is neither the slot %r nor a run under its %r container"
            % (run_dir, slot_dir, RUNS_DIR)
        )
    return run_dir


def pool_id_for(fp_text):
    """``p`` + the first 12 hex digits of sha256(identity). Must equal `tpool::pool_id_for`."""
    return "p" + hashlib.sha256(fp_text.encode("utf-8")).hexdigest()[:12]


def pools_root(slot_dir):
    return os.path.join(slot_dir, POOLS_DIR)


def read_fingerprint(pool_dir):
    """The identity text stored in a directory, or ``None`` when there is none."""
    try:
        with open(os.path.join(pool_dir, FINGERPRINT), encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def write_fingerprint(pool_dir, fp_text):
    """Atomically, because this file is the pool's identity.

    The old `cache.invalidate_extract_caches` wrote it with a bare ``open(..., "w")``. A kill
    mid-write left a truncated fingerprint, which was survivable while the file only gated a
    deletion. Now it NAMES a directory holding hours of preprocessing, and a truncated identity
    would fail to match its own pool and rebuild everything into a second one.
    """
    final = os.path.join(pool_dir, FINGERPRINT)
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(fp_text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)


def list_pools(slot_dir):
    """``[(pool_id, fp_text_or_None)]`` for every pool under ``<slot>/pools``, sorted by id.

    Dot-prefixed entries are the layout migration's staging directories (`tpool::STAGING_PREFIX`)
    and are never a pool: a half-filled one would otherwise be selectable, and a run that matched
    it would train on half a cache.
    """
    out = []
    root = pools_root(slot_dir)
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        if name.startswith("."):
            continue
        d = os.path.join(root, name)
        if os.path.isdir(d):
            out.append((name, read_fingerprint(d)))
    return out


def open_pool(slot_dir, fp_text):
    """THE entry point: the directory this run's preprocessing products belong in.

    Resolution order, and why it is this order:

    1. **an existing pool whose stored identity equals ours** — content match, so a pool minted by
       an older name derivation keeps being reused;
    2. **the slot root, when it holds a matching identity** — an unmigrated slot (see the module
       docstring). Logged, never silent;
    3. **a new directory under ``<slot>/pools/``** — this is where "flip a parameter" used to be a
       deletion. Nothing is removed; the other pools stay on disk, visible and reclaimable.

    Creating a pool is not the same event as "the products are missing": every stage already
    decides what to do by skip-if-exists, so a fresh pool simply has nothing to skip.
    """
    if not fp_text:
        # Every chain builds a non-empty identity. An empty one would collapse every run in this
        # slot into a single anonymous pool, which is worse than any rebuild.
        raise RuntimeError("POOL_IDENTITY_EMPTY: refusing to resolve a pool with no identity")

    existing = list_pools(slot_dir)
    for name, fp in existing:
        if fp is not None and fp == fp_text:
            return os.path.join(pools_root(slot_dir), name)

    if read_fingerprint(slot_dir) == fp_text:
        logger.info(
            "using the preprocessing products at the slot root (this slot has not been through "
            "the pool layout migration yet)"
        )
        return slot_dir

    root = pools_root(slot_dir)
    os.makedirs(root, exist_ok=True)
    base = pool_id_for(fp_text)
    taken = {name for name, _ in existing}
    name = base
    suffix = 1
    # A taken name means either a 48-bit collision or a pool whose fingerprint went missing.
    # Step aside rather than write our products into a directory that is not ours.
    while name in taken:
        suffix += 1
        if suffix > 64:
            raise RuntimeError("POOL_ID_EXHAUSTED: %s" % base)
        name = "%s_%d" % (base, suffix)

    pool_dir = os.path.join(root, name)
    os.makedirs(pool_dir, exist_ok=True)
    write_fingerprint(pool_dir, fp_text)
    logger.info(
        "new preprocessing pool %s (%d other pool(s) in this slot are kept)", name, len(existing)
    )
    return pool_dir


def assert_identity(pool_dir, fp_text):
    """Fail-closed check that we are about to preprocess into the pool we resolved.

    This is what is left of ``cache.invalidate_extract_caches``. That function compared the stored
    fingerprint against the run's and, on a mismatch, deleted the named cache directories — the
    deletion this whole change exists to remove. The pool is now CHOSEN by identity, so a mismatch
    here cannot be a dataset change: it is a plumbing bug (a caller that resolved one pool and
    preprocessed into another), and destroying hours of work on the strength of a bug would be the
    worst available response.
    """
    stored = read_fingerprint(pool_dir)
    if stored is None:
        write_fingerprint(pool_dir, fp_text)
        return
    if stored != fp_text:
        raise RuntimeError(
            "POOL_IDENTITY_MISMATCH: %s holds %r but this run computed %r"
            % (pool_dir, stored, fp_text)
        )
