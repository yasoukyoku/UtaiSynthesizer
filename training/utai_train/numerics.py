"""Numerical-divergence guard for the GAN training loops (S114, §F5-3).

Community bug (GitHub issue #2, RTX 2080Ti, RVC *continuation* training, 0.11.0):
about 600 steps after a resume every loss went nan at once --

    step 12200  loss_disc=3.972, loss_gen=3.167, loss_fm=8.741, loss_mel=15.991, loss_kl=1.899
    step 12400  loss_disc=nan,   loss_gen=nan,   loss_fm=nan,   loss_mel=nan,    loss_kl=nan

-- the loop kept running for 13+ hours poisoning every later weight, nothing in
the log said ERROR, and the reporter's previously valid ``<name>_best.pth`` came
back OVERWRITTEN with nan weights.

That is TWO defects, and they are independent -- fixing either one alone still
leaves the other:

a) **Nothing ever noticed.** There was no divergence check anywhere in the loop,
   so a run that was already dead kept burning wall-clock and disk.

b) **``save_best``'s guard read the METRIC, not the WEIGHTS.** Both loops update
   ``ema_mel`` only on finite steps ("a single non-finite step must not poison
   the EMA forever"), so once the losses go nan the EMA *freezes* at its last
   pre-collapse value. A frozen value that is still a new minimum makes the
   epoch-boundary test ``ema_mel < best_metric`` fire **exactly once**, writing
   the already-poisoned ``net_g.state_dict()`` under a pre-collapse score. One
   write is enough to destroy the file, and the score recorded next to it looks
   healthy afterwards.
   (b) is the data-destroying half: (a) only costs time.

Both ``rvc/train.py`` and ``sovits/train.py`` carried the same open-coded best
tracking, so the guard lives here once instead of twice -- two copies of a guard
are two chances to fix only one of them.

Wiring: ``DivergenceGuard.observe`` raises ``RuntimeError("<CODE>: <detail>")``;
``runner.py`` turns any exception into a protocol ``error`` line, Rust surfaces
it, and ``src/lib/backendError.ts`` maps the stable English CODE to the localized
text (same contract as ``device.require_wanted_accelerator``'s
``TRAINING_GPU_UNAVAILABLE``). No user-visible prose is hard-coded here.
"""

import math

import torch

#: Stable CODE the frontend maps to i18n. Never localize it here.
CODE_DIVERGED = "TRAINING_NUMERICS_DIVERGED"

#: How many CONSECUTIVE steps with a non-finite reported loss it takes before the
#: run is declared dead.
#:
#: It must not be 1. Under fp16 the forward pass can overflow on a single awkward
#: batch, and ``GradScaler`` exists precisely to absorb that: it skips the
#: optimizer step and lowers the scale, so the weights are untouched and the next
#: batch is normally fine. Halting on the first nan would kill runs that recover
#: by design.
#:
#: 20 is far inside the observed failure: issue #2 logs at ``log_interval=200``
#: and the collapse was already total at the next log line, i.e. it persisted for
#: at least 200 consecutive steps. A run whose losses are non-finite 20 steps in a
#: row has not "hit a bad batch" -- either the weights are already nan (checked
#: below) or the data is producing nan, and both are terminal.
DEFAULT_PATIENCE = 20


def first_nonfinite_tensor(named_tensors):
    """Name of the first floating tensor holding a nan/inf, else ``None``.

    ``named_tensors`` is any iterable of ``(name, tensor)`` -- typically
    ``state_dict().items()`` or ``chain(named_parameters(), named_buffers())``.

    ⚠ The two skips are NOT the same strength, and an earlier draft of this
    docstring got it wrong (S114, caught by a mutation probe that refused to go
    red). Measured on torch 2.5.1: ``torch.isfinite`` **accepts** int64 / bool /
    uint8 and returns all-True, so dropping ``is_floating_point()`` changes no
    answer today -- it is there to avoid materializing a bool mask over large
    integer buffers and to keep the predicate meaningful if a future dtype
    stops being accepted. ``torch.is_tensor`` IS load-bearing: this helper is
    public and takes any ``(name, value)`` iterable, and a non-tensor value would
    raise instead of being skipped.
    """
    for name, t in named_tensors:
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        if not bool(torch.isfinite(t).all()):
            return name
    return None


def first_nonfinite_module(modules):
    """First ``"<tag>.<param>"`` holding a nan/inf across ``modules``, else ``None``.

    ``modules`` is an iterable of ``(tag, module)``, e.g. ``(("G", net_g), ("D", net_d))``.
    Both parameters AND buffers are scanned -- a poisoned running statistic is
    just as unrecoverable as a poisoned weight, and it would not show up in
    ``named_parameters()``.
    """
    for tag, module in modules:
        entries = list(module.named_parameters()) + list(module.named_buffers())
        bad = first_nonfinite_tensor(entries)
        if bad is not None:
            return "%s.%s" % (tag, bad)
    return None


class DivergenceGuard:
    """Per-step watchdog: consecutive non-finite losses -> loud, terminal error.

    Cheap by construction. The per-step half only looks at floats the loop has
    already computed for its own reporting (no extra device sync); the expensive
    half -- scanning every weight -- runs at most once, on the step that raises.
    """

    def __init__(self, modules, patience=DEFAULT_PATIENCE, logger=None):
        self._modules = tuple(modules)
        self._patience = int(patience)
        self._logger = logger
        self._consecutive = 0
        self._first_step = None

    @property
    def consecutive(self):
        return self._consecutive

    def observe(self, step, values):
        """Record one step. ``values`` maps a loss name to its float value.

        Raises ``RuntimeError(CODE_DIVERGED + ": ...")`` once the run is dead.
        """
        bad = sorted(k for k, v in values.items() if not math.isfinite(float(v)))
        if not bad:
            self._consecutive = 0
            self._first_step = None
            return

        if self._consecutive == 0:
            self._first_step = step
            if self._logger is not None:
                self._logger.warning(
                    "non-finite loss at step %s (%s) - watching for %s consecutive steps",
                    step,
                    ",".join(bad),
                    self._patience,
                )
        self._consecutive += 1
        if self._consecutive < self._patience:
            return

        # Only now is it worth paying for a full weight scan. Which side is dead
        # changes what the user should do, so say which one it is.
        poisoned = first_nonfinite_module(self._modules)
        weights = "weights=%s is non-finite" % poisoned if poisoned else "weights=all finite"
        raise RuntimeError(
            "%s: %d consecutive non-finite steps (first at step %s, latest %s; fields %s); %s"
            % (
                CODE_DIVERGED,
                self._consecutive,
                self._first_step,
                step,
                ",".join(bad),
                weights,
            )
        )


def best_save_is_safe(state_dict, logger=None):
    """Last line of defence for ``save_best`` -- see defect (b) in the module doc.

    Returns ``True`` when every floating entry of ``state_dict`` is finite. On
    ``False`` the caller MUST leave both the existing best file and its recorded
    metric untouched: the old file still describes the old metric, so advancing
    the metric without writing the file would make the sidecar lie in the other
    direction.

    This is deliberately NOT expressed as "the guard above already halted, so a
    poisoned state cannot reach here". The halting guard needs
    ``DEFAULT_PATIENCE`` consecutive bad steps, while a single poisoned step
    landing exactly on an epoch boundary is enough to reach ``save_best`` -- and
    the frozen-EMA mechanism means the metric it is compared against is a *stale
    healthy* number, so the comparison cannot catch it either. The two guards
    protect different failure shapes on purpose.
    """
    bad = first_nonfinite_tensor(state_dict.items())
    if bad is None:
        return True
    if logger is not None:
        logger.error(
            "REFUSING to overwrite the best checkpoint: model weights contain nan/inf (first: %s). "
            "The previous best file and its recorded metric are left untouched.",
            bad,
        )
    return False
