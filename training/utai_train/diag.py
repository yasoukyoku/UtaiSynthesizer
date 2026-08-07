"""Diagnostic mode, python side — the sidecar's half of S115's general diagnostics channel.

WHY (project_v2_resume_divergence_open §4). Community issue #2 could not be answered because the
evidence was not in the log: `log_interval=200`, so the collapse ("all five losses finite at step
12200, all five nan at 12400") was already total by the next sample. 200 steps of blind spot is
the difference between "a single point explosion" and "a slow drift", and those want opposite
fixes. The user's instruction after that report was explicit: 「至少不要说这次查不出来、复发了
再报上来之后还查不出来」.

So diagnostic mode now also turns on, inside the trainer:

1. **per-step logging** (`log_interval` → 1). The blind spot disappears.
2. **the GradScaler's scale, and whether the step was SKIPPED.** This is the one thing no log has
   ever carried, and it decides a whole branch of the investigation by itself: under fp16 a
   skipped step trains nothing while emitting a perfectly normal, finite loss line — measured
   S117, and the reason `numerics.DivergenceGuard` cannot see it (it observes losses, and the
   overflow is in the gradients).
3. **the gradient norms.** `clip_grad_value_(…, None)` computes them on every step for every
   parameter tensor and then throws them away on 199 steps out of 200 — they are the only
   per-step signal that would reveal a gradient that is huge but finite, which measurably poisons
   AdamW's second moment (and above ~1.8e20 freezes a parameter forever) without producing a
   single nan anywhere.

⛔ OFF BY DEFAULT AND FREE WHEN OFF. `enabled()` reads the environment once at import; the call
sites are one `if` on a module-level bool. Nothing here changes any number the training math
produces — it only decides what gets written down.
"""

import os

#: Set by `training::diagnostics::apply` (Rust) alongside the per-runtime torch knobs, whenever
#: the user has diagnostic mode on in Settings. ⚠ Its NAME is a cross-language contract; the Rust
#: gate parses it out of this file.
ENV = "UTAI_DIAGNOSTICS"

_ON = os.environ.get(ENV) == "1"


def enabled():
    return _ON


def log_interval(configured):
    """The interval to actually use. Diagnostic mode collapses the 200-step blind spot to zero."""
    return 1 if _ON else configured


def scaler_note(scaler):
    """`"scale=… skipped=…"` for this step, or "" when the mode is off / there is no scaler.

    `skipped` is derived from torch's own per-optimizer `found_inf` flags rather than from a
    before/after weight comparison: comparing weights would need a clone of every parameter on
    every step, and the flags are what `GradScaler.step` itself consulted.
    """
    if not _ON or scaler is None or not scaler.is_enabled():
        return ""
    skipped = False
    try:
        for st in scaler._per_optimizer_states.values():
            found = st.get("found_inf_per_device") or {}
            if any(bool(v.item()) for v in found.values()):
                skipped = True
                break
    except Exception:
        # A private-API change must degrade to "we do not know", never crash a training run.
        return " scale=%s skipped=?" % scaler.get_scale()
    return " scale=%s skipped=%s" % (scaler.get_scale(), int(skipped))
