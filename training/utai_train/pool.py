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

Batch 1 changed nothing about WHICH artifacts share an identity: the fp_text formulas in the five
pipelines were untouched, so the pool boundary was byte-for-byte the old cache-invalidation
boundary, and the only difference was that a non-matching pool became a SIBLING instead of a
deletion.

§F2⒝ ④d is where the boundary itself moves, because two knobs that DO decide what the products
are were never in it (rvc's sample rate; every chain's augmentation count). Both halves of that
move — the formula here and the text already written on disk — have to change at the same
instant, so the formula is VERSIONED and the version rides in on `run.json`: see
[`identity_version`] for the carrier and [`identity_suffix`] for the tokens.

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
import json
import logging
import os

from . import ckpt_guard

logger = logging.getLogger(__name__)

#: Container for every pool of one slot. Must equal `tpool::POOLS_DIR`.
POOLS_DIR = "pools"

#: The identity file inside a pool. Must equal `tpool::FINGERPRINT`.
FINGERPRINT = "dataset.fingerprint"

#: Container for every RUN of one slot. Must equal `trun::RUNS_DIR`.
RUNS_DIR = "runs"

#: What a RUN records about the pool it preprocessed into. Must equal `trun::POOL_REF`.
#:
#: ★§F2⒝ ④d — nothing on disk has ever recorded the run↔pool edge, and §F2⒝ ④e ("重训 = a new
#: run", plus managing and DELETING old runs) cannot reclaim a pool without it: a slot will hold
#: several runs sharing several pools, and "which bytes may go" has no answer.
#:
#: ⛔ It is a file of its own rather than a key in `run.json`, and that is not a style choice:
#: Rust REWRITES `run.json` wholesale on every start, so a key python appended would vanish on the
#: next one — silently, and precisely for the runs that have been trained more than once.
#: ⛔ And it is written HERE rather than by Rust, because Rust cannot compute it: the pool is named
#: by the dataset fingerprint, and computing that means reading every imported file.
POOL_REF = "pool.json"

#: The pool-identity FORMULA version this python understands. Must equal
#: `tpool::POOL_IDENTITY_VERSION`. See [`identity_version`] for what the number gates.
POOL_IDENTITY_VERSION = 2

#: The `dataset_44k` subdirectory a SINGLE-speaker sovits-family run slices into, from identity
#: version 2 on. Must equal `tpool::SOLE_SPEAKER_DIR`.
#:
#: Before version 2 this was the run's `model_slug`, which made a POOL product carry a RUN's name:
#: two runs of the same slot with different names grew two complete slice trees inside one pool,
#: each paying for its own features, and `extract_all` walks EVERY subdirectory of `dataset_44k`
#: so the abandoned one was re-scanned forever. §F2⒝ ④e (重训 = a NEW run) turns that from a
#: historical accident into the normal case, which is why it is fixed here rather than there.
#:
#: ⛔ Three rules the name obeys, each with a real failure behind it:
#: ⑴ no ``.wav`` substring — `filename.replace(".wav", ".spec.pt")` rewrites PATH segments, so a
#:    directory carrying it would send every spec cache to a path that can never be hit again;
#: ⑵ not ``nul`` — `os.makedirs` succeeds on Windows and creates nothing;
#: ⑶ not derived from user text — Windows silently eats trailing spaces/dots, so the name on disk
#:    and the string in `config.spk` would differ.
#: ⚠ It also cannot collide with a real speaker slug: `slugify` (`training::slugify`) always
#: appends ``_`` + 8 hex, and this name has no underscore.
#: ⚠ MULTI-speaker keeps each speaker's own slug — those slugs are folded into the fingerprint
#: (`sovits/pipeline.py`), so renaming them would re-identify every multi-speaker pool on disk.
SOLE_SPEAKER_DIR = "spk0"


def identity_version(cfg):
    """WHICH pool-identity formula this run must use — 1 (pre-④d) or 2.

    ## Why the formula is versioned at all

    ④d adds two tokens (`|sr=` for rvc, `|aug=` for every chain) to the text that NAMES the
    directory hours of preprocessing live in. Both halves of that change have to happen at the
    same instant or the user pays for it: a python that computes the new text against a disk still
    stamped with the old one finds no match and rebuilds everything into a sibling pool, and the
    reverse order is the same bill. The two halves are in different languages and different
    processes, so "at the same instant" needs a carrier.

    ⛔ It cannot be `slot.json`: python never reads it (`open_pool` has no layout concept, and
    nothing under `utai_train/` opens that file). It is `run.json`, which is already the channel
    Rust uses to hand python EFFECTIVE values it alone can compute (`loudnorm`, `aug_copies`,
    `slot_has_main_model`).

    ⇒ The number is a pure function of the slot's layout on the Rust side: version 2 is written
    ONLY for a slot whose 3→4 migration has committed, i.e. only when every pool in that slot has
    already been re-stamped with the new text. A slot the migration skipped, failed on, or never
    reached (another instance was alive at boot; a data root restored from a backup) keeps
    answering 1, and its pools keep matching. That is the whole point: a refusal on the Rust side
    is now visible to python instead of being a log line python cannot see.

    ## Why absent means 1 and why the upper bound is fail-closed

    Absent = an old `run.json`, or a hand-built gate config. Both describe a disk built by the old
    formula, so 1 is the answer that matches what is actually there — the same "fire on a positive
    fact" posture `open_pool`'s slot-root arm takes.

    A version ABOVE what this python knows is the opposite case: some newer Rust has stamped the
    disk in a shape this code cannot reproduce, and computing an identity we do not understand
    would either rebuild everything or, worse, claim someone else's products. Refuse loudly.
    """
    raw = cfg.get("pool_identity_version")
    # ⚠ `is None`, not truthiness: an explicit `0` is a value someone WROTE, and collapsing it into
    # "the key is absent" would hide a writer that is producing nonsense. Rust only ever writes 1
    # or 2, so this is about keeping the two states distinguishable rather than about today.
    v = 1 if raw is None else int(raw)
    if v < 1:
        raise RuntimeError(
            "POOL_IDENTITY_VERSION_INVALID: run.json says pool identity v%d; the lowest formula "
            "that has ever existed is v1" % v
        )
    if v > POOL_IDENTITY_VERSION:
        raise RuntimeError(
            "POOL_IDENTITY_VERSION_UNKNOWN: run.json asks for pool identity v%d but this "
            "training package only knows v%d — refusing to guess at the identity of the "
            "preprocessing products on disk" % (v, POOL_IDENTITY_VERSION)
        )
    return v


def identity_suffix(cfg, aug_copies, sample_rate=None):
    """The trailing identity tokens EVERY chain appends to its own fp_text, and appends LAST.

    ## Why one helper, and why the call is the last thing each formula does

    The five chains do not share a formula: sovits / sovits_diff / sovits_v2 go through
    `extract_cache_fp_text`, sovits_v2 then appends a conditional `|f0=` of its own, and rvc and
    vocoder build their text inline without touching a single shared symbol. So "add a token to
    the identity" is a four-site edit where 「改 3 漏 2」 costs the user hours and says nothing.

    ⛔ The tokens are NOT added inside `extract_cache_fp_text`, and that is not a style choice.
    sovits_v2 appends `|f0=` AFTER that helper returns, so a token added inside it would come out
    as ``…|loudnorm=0|aug=2|f0=dio`` for v2 and ``…|loudnorm=0|aug=2`` for the others — two
    different concatenation ORDERS for one logical set of tokens, and the Rust migration would
    have to know which chain it is looking at to reproduce either. Appending last means the order
    is the same everywhere: the chain's own text, then this suffix, in this suffix's own order.
    Byte-for-byte agreement with `tpool::identity_suffix` is what lets Rust re-stamp an existing
    pool instead of the user re-preprocessing it.

    ## What each token is for

    `|sr=` (rvc only, unconditional): `1_16k_wavs` is resampled FROM the target rate, and every f0
    / feature product is cached by slice NAME — so a slot that changed sample rate used to reuse
    the features computed for the other rate, silently. rvc is the only chain that has this knob;
    the other four hard-code 44100.
    ⚠ The value is the rate in Hz, not the `"40k"` UI string: the migration's authority for an
    EXISTING pool is the header of a wav already sitting in `0_gt_wavs`, which is an integer.
    Routing that through a display string would be a second encoding of one fact.

    `|aug=` (every chain, only when > 0): `augment_slices` prunes by `idx > copies` and takes the
    slice's companions with it, so two runs sharing a pool with different counts destroy each
    other's products. Omitting the token at 0 is what keeps every existing un-augmented pool
    matching after this change — the overwhelming majority.
    ⚠ rvc would have survived sharing (`_wipe_slice_dirs` rebuilds its slice dirs every run and
    PSOLA is deterministic, so its aug wavs come back identical). It is included anyway: one rule
    for five chains is the property that makes 「改 3 漏 2」 impossible, and it is what makes the
    lock table's `costly("augCopies")` label — 「改它会重新指纹化」 — true for the first time.
    """
    if identity_version(cfg) < 2:
        return ""
    out = ""
    if sample_rate is not None:
        out += "|sr=%d" % int(sample_rate)
    if int(aug_copies) > 0:
        out += "|aug=%d" % int(aug_copies)
    return out


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
    # ★§F2⒝ ④e 笔 1 — and while we are the one place every chain resolves its run, ask the other
    # question about that directory: if this start says it MINTED it, nobody may have trained
    # there. See `ckpt_guard.refuse_to_resume_into_a_fresh_run` for why here and not in `runner`.
    # ⇒ no-op until Rust starts writing `run_is_fresh: true`, which is ④e's next 笔.
    ckpt_guard.refuse_to_resume_into_a_fresh_run(cfg, run_dir)
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


def _record_pool_choice(run_dir, slot_dir, pool_dir):
    """Write `<run>/pool.json` — WHICH pool this run just resolved. Best-effort.

    Best-effort on purpose: this is an ANNOTATION for §F2⒝ ④e's reclamation, never an authority.
    Failing to write it must not fail a training run that is otherwise fine, and any reader has to
    cope with its absence anyway (every run that predates the key).

    `pool_id` is the directory NAME, not `pool_id_for(fp_text)`: a pool minted under an older
    derivation, or one that stepped aside from a name collision, keeps working — and its name is
    the handle that addresses it. ``None`` = the slot root IS the pool (an unmigrated slot), which
    is a real answer rather than a failure.
    """
    if not run_dir:
        return
    try:
        norm = lambda p: os.path.normcase(os.path.abspath(p))
        under_pools = norm(os.path.dirname(pool_dir)) == norm(pools_root(slot_dir))
        blob = {"pool_id": os.path.basename(pool_dir) if under_pools else None}
        final = os.path.join(run_dir, POOL_REF)
        tmp = final + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp, final)
    except Exception as e:  # noqa: BLE001 — an annotation must never break a run
        logger.warning("could not record which pool this run used: %s", e)


def open_pool(slot_dir, fp_text, run_dir=None):
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

    ★§F2⒝ ④d — pass `run_dir` and the answer is RECORDED (`POOL_REF`). One recording point for
    five chains, and it sits here because this function is the only place that knows the answer.
    """
    pool_dir = _resolve_pool(slot_dir, fp_text)
    _record_pool_choice(run_dir, slot_dir, pool_dir)
    return pool_dir


def _resolve_pool(slot_dir, fp_text):
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
