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
