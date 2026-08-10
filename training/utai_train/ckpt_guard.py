"""Resume-checkpoint guard — THE single source for all three GAN trainers (S116 §F5-③ⓒ).

WHY THIS EXISTS. `load_checkpoint` is vendored verbatim from upstream RVC / so-vits-svc, and all
three copies contain the same tolerance::

    for k, v in state_dict.items():          # the keys the MODEL needs
        try:
            new_state_dict[k] = saved_state_dict[k]
            if shape differs: raise KeyError
        except:
            logger.info("%s is not in the checkpoint", k)   # comment says: "pretrain 缺失的"
            new_state_dict[k] = v                            # ← the model's OWN RANDOM INIT

Two facts turn that from harmless into data-destroying HERE:

1. **In this repo `load_checkpoint` is called from exactly one place per trainer: the RESUME
   block.** (`rvc/train.py:203,207` · `sovits/train.py:189,191` · `sovits_v2/train.py:148,150` —
   grepped, S116.) The pretrained-model path the comment is defending does NOT go through it; it
   calls `net.load_state_dict(torch.load(pretrainG)["model"])` directly. So every silent
   random-fill this code performs is a corrupt resume being papered over, never a pretrain.

2. **The optimizer is restored anyway, unconditionally, right afterwards.** Measured on the real
   functions (S116 fixture, `TESTING\s116_g16\s116_resume_probe.py`): drop one key from a
   checkpoint, or change one shape, and you get that parameter back at its FRESH RANDOM INIT while
   all four parameter states still carry `step=24` Adam moments from the run you are resuming.
   Random weights driven by stale momentum is a textbook delayed blow-up — and the only trace is
   an `INFO: <key> is not in the checkpoint` line, which this app forwards at DEBUG level, i.e.
   the user sees nothing at all.

⚠ WHAT THIS IS *NOT*. The S116 fixture also measured the HEALTHY resume path and found it
bit-identical to never having stopped: model weights, every Adam moment, `step`, `initial_lr` and
the learning rate all match, and the LR it produces reproduces the exact value in community issue
#2's log (`1e-4 * 0.999875**3 == 9.996250468730469e-05`). So this guard is NOT a claim about what
caused that report — that report's checkpoint loaded cleanly. It closes a different, silent way to
reach the same outcome.

⛔ THE INTERACTION THAT MAKES THE SHAPE OF THIS FILE NON-OBVIOUS. Upstream wraps the resume in a
BARE `except:` whose fallback is "load the pretrained model and restart from epoch 1". So simply
raising from `load_checkpoint` would be SWALLOWED, and a corrupt checkpoint would silently throw
away the user's completed epochs and begin overwriting from step 0 — trading one silent failure
for a worse one. That is why the refusal is its own exception type that each `train.py` re-raises
BEFORE the bare handler, and why `resume_was_intended` exists: the bare handler must still mean
"there is nothing to resume from" and nothing else.
"""

import glob
import os

# Stable CODE → utai_train.runner → backendError.ts → all three locales. Same chain as
# TRAINING_NUMERICS_DIVERGED (S114 §F5-③ⓐ).
CODE = "TRAINING_RESUME_CHECKPOINT_INCOMPLETE"
FAILED_CODE = "TRAINING_RESUME_CHECKPOINT_UNREADABLE"


class ResumeRefused(Exception):
    """Raised when a resume checkpoint exists but cannot be used as one.

    ⛔ Every `train.py` MUST re-raise this before its bare `except:` — see the module docstring.
    """


def _fmt(pairs, limit=6):
    head = ", ".join(pairs[:limit])
    return head + (f" (+{len(pairs) - limit} more)" if len(pairs) > limit else "")


def check_resume_state_dict(checkpoint_path, wanted, saved):
    """Refuse a resume whose checkpoint cannot supply every parameter the model needs.

    `wanted` = the freshly-built model's `state_dict()`, `saved` = the checkpoint's `["model"]`.
    Raises `ResumeRefused`; returns None when the checkpoint covers the model exactly.

    ⚠ Deliberately checks ONLY presence + shape, i.e. exactly the two conditions the vendored
    loop silently papers over. It does not look at dtype or values: a legitimately different
    dtype loads fine, and "the weights look wrong" is not something a loader can judge.
    """
    missing = [k for k in wanted if k not in saved]
    mismatched = [
        f"{k}: need {tuple(wanted[k].shape)}, got {tuple(saved[k].shape)}"
        for k in wanted
        if k in saved and tuple(saved[k].shape) != tuple(wanted[k].shape)
    ]
    if not missing and not mismatched:
        return
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing [{_fmt(missing)}]")
    if mismatched:
        parts.append(f"{len(mismatched)} shape-mismatched [{_fmt(mismatched)}]")
    # Detail stays raw/technical (backendError.ts convention): the localized remedy lives in
    # backend.TRAINING_RESUME_CHECKPOINT_INCOMPLETE, and repeating advice here would print it
    # twice — once localized, once not.
    raise ResumeRefused(
        f"{CODE}: {os.path.basename(checkpoint_path)} — {'; '.join(parts)}"
    )


def _suffix(path):
    """Trailing integer of a G_<n>.pth / D_<n>.pth name; None when it is not a number."""
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def resume_was_intended(model_dir, seeded_base_is_step_zero=False):
    """True when this directory HAS resume checkpoints, i.e. a load failure is a real failure.

    The fallback branch in each trainer must keep meaning "there was nothing to resume from".
    When there IS something and it could not be read, that is `TRAINING_RESUME_CHECKPOINT_UNREADABLE`
    — a truncated / unreadable file must not silently restart from the pretrained model at step 0,
    discarding whatever the user had already trained.

    ⛔ `seeded_base_is_step_zero` is NOT a style knob — getting it wrong breaks every FRESH run.
    The two so-vits trainers copy their BASE model into the workspace as `G_0.pth` / `D_0.pth`
    (upstream's logs/44k drop-in; see those files' headers), so a brand-new training directory
    already contains `G_*`/`D_*`. Treating that as a resume would run the strict coverage check
    against a base model — and a base legitimately disagrees with the model about `emb_g.weight`
    (its speaker count is not the user's), which is precisely the case the tolerant loop exists
    for. RVC is the opposite: its base is a separate `hps.pretrainG` path and its resume file is
    the constant name `G_2333333.pth`, so any `G_*` there really is a resume.
    """
    def real(pattern):
        found = glob.glob(os.path.join(model_dir, pattern))
        if seeded_base_is_step_zero:
            found = [p for p in found if _suffix(p) != 0]
        return bool(found)

    return real("G_*.pth") and real("D_*.pth")


#: What a trainer's startup must do with what it finds on disk. THREE outcomes, not two — see
#: `plan_load`.
LOAD_RESUME = "resume"
LOAD_SEEDED_BASE = "seeded_base"
LOAD_PRETRAIN = "pretrain"


def plan_load(model_dir, seeded_base_is_step_zero=False):
    """The single decision every trainer's startup makes. THREE outcomes, and they live here
    together because splitting them across call sites lost one of them for a whole release cycle.

    * `LOAD_RESUME` — a real resume point is here: load it STRICTLY (`resume=True`).
    * `LOAD_SEEDED_BASE` — only the step-0 base pair is here, i.e. this is a FRESH so-vits run.
      Load it TOLERANTLY (`resume=False`): a base legitimately disagrees with the model about
      `emb_g.weight` — its speaker count is not the user's — and absorbing exactly that is what
      the vendored loop's random-fill is for, and why `seeded_base_is_step_zero` exists at all.
      Unreachable without that flag, which is what says "this trainer's base lives under those
      names".
    * `LOAD_PRETRAIN` — nothing resumable here. RVC loads `hps.pretrainG/D` from its own paths;
      a so-vits workspace whose base was never seeded starts from random init.

    ⛔ THE REGRESSION THIS PREVENTS (S117, and it is why the middle case is not optional). The
    two so-vits trainers have no separate pretrain branch: upstream's entire pretrain mechanism
    is "copy the base models in as `G_0.pth`/`D_0.pth` and let `latest_checkpoint_path` pick them
    up" (`sovits/pipeline.py:_seed_base_checkpoints`, whose docstring says exactly that). S116
    put that single load site behind `resume_was_intended(..., seeded_base_is_step_zero=True)` —
    a predicate that filters out `*_0.pth` **by design** — so from `c44dec6` until this fix a
    fresh so-vits or so-vits-v2 run fell to the `else` branch, set `epoch_str = 1`, and trained
    the whole model from RANDOM INIT with a 180-425 MB base sitting unread on disk. Silent: no
    exception, no warning, just a far worse model after the same hours.
    """
    if resume_was_intended(model_dir, seeded_base_is_step_zero=seeded_base_is_step_zero):
        return LOAD_RESUME
    if seeded_base_is_step_zero and resume_was_intended(model_dir):
        return LOAD_SEEDED_BASE
    return LOAD_PRETRAIN


def refuse_unreadable(model_dir, exc):
    return ResumeRefused(
        f"{FAILED_CODE}: {os.path.basename(os.path.normpath(model_dir))} — "
        f"{type(exc).__name__}: {exc}"
    )


# ─────────────────── §F2⒝ ④e 笔 1:「这个 run 是新铸的」 ───────────────────

#: The `run.json` key Rust uses to say so. Must equal `tpool`'s sibling — see
#: `trun::FRESH_RUN_KEY`, whose doc carries the whole rationale.
FRESH_RUN_KEY = "run_is_fresh"

#: A start that declared itself a fresh mint landed on a directory somebody already trained in.
FRESH_CODE = "FRESH_RUN_HAS_PRODUCTS"
#: The carrier itself is malformed — a writer is producing something Rust never emits.
FRESH_FLAG_CODE = "FRESH_RUN_FLAG_INVALID"


def declares_fresh_run(cfg):
    """Does this config say「the run directory is supposed to be untouched」?

    ⛔ ABSENT ⇒ False, and that is the truthful answer rather than a default: every `run.json`
    written before ④e describes a start under the old regime, where 「fresh」 was expressed
    IMPLICITLY by the workspace having just been deleted. Same posture as
    `pool.identity_version`'s absent⇒1.

    ⛔ Anything that is not a JSON boolean is REFUSED, not coerced. `bool(raw)` would read the
    string ``"false"`` as True (refusing every resume) and ``0``/``""`` as False (switching the
    guard off forever). Both are silent, and a carrier that cannot notice its own writer is
    exactly the failure `trun::FRESH_RUN_KEY` exists to prevent.
    """
    raw = cfg.get(FRESH_RUN_KEY)
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise RuntimeError(
            "%s: run.json's %r must be a JSON boolean, got %r (%s). Rust only ever writes true "
            "or false; a value of another type means something else wrote this file."
            % (FRESH_FLAG_CODE, FRESH_RUN_KEY, raw, type(raw).__name__)
        )
    return raw


def _main_resume_point(run_dir):
    """A GAN checkpoint from a previous run of this directory, or None.

    ⚠ Step 0 is excluded on purpose and it is not the same exclusion `resume_was_intended` makes
    for so-vits: there it means「a base is not a resume point」, here it means「a fresh start that
    died after seeding its base left these behind, and refusing because of them would block the
    retry」. Same files, different question, same answer.

    ⚠ Either half counts — unlike `resume_was_intended`, which needs the PAIR. A lone `G_800.pth`
    is not resumable, but it is still a previous run's work that this start would overwrite, and
    this function's job is 「would anything be destroyed」, not 「could it be continued」.
    """
    for pattern in ("G_*.pth", "D_*.pth"):
        for p in sorted(glob.glob(os.path.join(run_dir, pattern))):
            if _suffix(p) not in (0, None):
                return os.path.basename(p)
    return None


def _vocoder_resume_point(run_dir):
    for p in sorted(glob.glob(os.path.join(run_dir, "model_ckpt_steps_*.ckpt"))):
        return os.path.basename(p)
    if os.path.isfile(os.path.join(run_dir, "resume_best", "state.json")):
        return "resume_best"
    return None


def _diffusion_resume_point(run_dir):
    # ⚠ Every path here is rooted at `run_dir` rather than at a `diff_dir` local, and that is not
    # style: `gate_pool_table.py` keeps a census of every `os.path.join` BASE NAME in this tree and
    # turns red on a new one, because a base it cannot classify as pool-derived or run-derived
    # silently drops those sites out of its pool/run checks. It caught this function's first form.
    for p in sorted(glob.glob(os.path.join(run_dir, "diffusion", "model_*.pt"))):
        if os.path.isfile(p):
            return "diffusion/" + os.path.basename(p)
    for snap in ("resume_latest", "resume_best"):
        if os.path.isfile(os.path.join(run_dir, "diffusion", snap, "state.json")):
            return "diffusion/" + snap
    return None


def refuse_to_resume_into_a_fresh_run(cfg, run_dir):
    """⛔ THE ④e guard: a start that says it minted a new run must not find somebody's work there.

    ## Why this exists at all

    「重训」 is becoming 「mint a run beside the old ones」. If the mint ever hands back a directory
    that is already in use — an id that is a pure function of the family, a migration that did not
    finish, a hand-built config — python cannot tell: `fresh` never crosses the boundary, an empty
    `resume_from` is normalised to `"latest"` before it gets here, and `plan_load` judges purely by
    which files exist. So the trainer would resume the OTHER run and overwrite it while the UI
    reports a new run.

    ⚠ Honest scope: the five pipelines DO raise 「没有执行任何训练步」 when the collided run has
    already passed its target, so that case is loud today — after hours of preprocessing are
    already paid for. The silent case is colliding with a run that is still SHORT of its target,
    which simply continues it. This guard fires before either.

    ## Why it lives on the `checked_run_dir` path

    That is the one place all five chains pass through with `cfg` in hand, at the top of each
    `pipeline.run` before anything is created. Putting it in `runner.main` would cover the same
    five chains and be invisible to `gate_aug0_driver.py`, which imports each pipeline and calls
    `run()` directly — i.e. the behaviour legs could never drive it.

    ## What counts as「somebody's work」

    The run-level arms of `training::slot_holds_work`, mirrored: the GAN pair, the vocoder's
    numbered ckpts + resumable snapshot, and the diffusion sub-tree. Checking only the GAN pair
    would leave the vocoder and diffusion chains structurally uncovered while reading as complete.
    ⚠ Deliberately BROADER than the Rust probes for the snapshots (a `state.json` is enough; Rust
    also verifies the payload list). A half-written snapshot is still a previous run's work, and
    for a refusal the broad side is the safe one — a freshly minted directory has none of these.
    """
    if not declares_fresh_run(cfg):
        return
    found = (
        _main_resume_point(run_dir)
        or _diffusion_resume_point(run_dir)
        or _vocoder_resume_point(run_dir)
    )
    if found is None:
        return
    raise RuntimeError(
        "%s: this start declared a freshly minted run, but %r already holds %r — continuing "
        "would train on top of another run's checkpoints and overwrite them. Refusing before "
        "any preprocessing." % (FRESH_CODE, os.path.basename(os.path.normpath(run_dir)), found)
    )
