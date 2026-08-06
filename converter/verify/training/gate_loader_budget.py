"""Gate: the S114 §F5-1 DataLoader commit budget.

    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate_loader_budget.py

Self-contained: no dataset, no GPU, seconds. UTAI_GATE_REPO points it at a copy
of the tree (the mutation driver uses that so probes never touch the repo).

WHAT IT PROVES
--------------
(A) plan_loader is REDUCE-ONLY and EXACT. The whole justification for adapting
    instead of lowering the defaults is "a roomy machine keeps today's numbers
    byte-for-byte", so that is asserted directly, plus a brute-force sweep that
    no input anywhere raises either knob.
(B) the ladder is the documented one (prefetch first, then workers, then 0).
(C) ★ THE RNG DISCIPLINE. sovits/sovits_v2 datasets draw from the GLOBAL random
    streams inside __getitem__, so reading one extra batch to measure it would
    silently move every later training step. The probe snapshots and restores
    python/numpy/torch, and C3 is the control that proves this check CAN fail:
    the same probe with restoration disabled leaves all three states changed.
    Without C3, C2 would just be "two identical states compared equal".
(D) measure_batch_bytes counts nested tensors and only tensors.
(E) every loader that got the guard actually calls it, and passes
    prefetch_factor/persistent_workers ONLY when workers > 0 — (F) shows torch
    itself rejects the other combination, so this is not a style preference.

MUTATION PROBES (driver: scratchpad mutate_s114_loader.py; never edits the repo)
-------------------------------------------------------------------------------
    L1  plan_loader: return the request unchanged always     -> A4/A5/B red
    L2  plan_loader: drop the "fits >= want" early return    -> A1 red
    L3  plan_loader: reduce workers before prefetch          -> B1 red
    L4  probe_batch_bytes: skip the random.setstate restore  -> C2 red
    L5  probe_batch_bytes: skip the torch restore            -> C2 red
    L6  plan_loader: allow raising prefetch when it fits     -> A2 red
    L7  rvc/train.py: pass prefetch_factor unconditionally   -> E2 red
"""

import io
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("UTAI_GATE_REPO") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from utai_train import loader_budget as lb  # noqa: E402


class Captured(logging.Handler):
    """The decision log is part of the deliverable, so it gets captured and
    checked (G) rather than muted outright — and muted everywhere else, because
    a 600-combination sweep that prints 600 warnings hides its own verdict."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []
        self.on = False

    def emit(self, rec):
        if self.on:
            self.records.append(rec)


CAP = Captured()
_LBLOG = logging.getLogger("utai_train.loader_budget")
_LBLOG.handlers[:] = [CAP]
_LBLOG.propagate = False
_LBLOG.setLevel(logging.DEBUG)

FAILURES = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILURES.append(name)


def read(rel):
    with io.open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


MiB = 1048576
GiB = 1024 * MiB

# ── A. reduce-only, and exact when it fits ───────────────────────────────────

print("A. reduce-only")

# a 64 GB-commit box with 8 MiB batches: today's numbers must survive untouched
roomy = lb.plan_loader(4, 8, 8 * MiB, 64 * GiB)
roomy_sovits = lb.plan_loader(5, 2, 8 * MiB, 64 * GiB)
check("A1 roomy machine keeps rvc's 4/8 exactly", roomy == (4, 8), str(roomy))
check("A1b roomy machine keeps sovits' 5/2 exactly", roomy_sovits == (5, 2), str(roomy_sovits))

raised = []
for w in (1, 2, 4, 5, 8, 16):
    for p in (2, 4, 8):
        for batch in (MiB // 4, MiB, 16 * MiB, 256 * MiB, 4 * GiB):
            for avail in (256 * MiB, 2 * GiB, 8 * GiB, 20 * GiB, 128 * GiB):
                gw, gp = lb.plan_loader(w, p, batch, avail)
                if gw > w or gp > p:
                    raised.append((w, p, batch, avail, gw, gp))
check("A2 no input anywhere RAISES either knob (600 combos)", not raised, str(raised[:2]))

for args in [(4, 8, None, 20 * GiB), (4, 8, 8 * MiB, None), (0, 8, 8 * MiB, 20 * GiB)]:
    got = lb.plan_loader(*args)
    check(
        "A3 unmeasurable input -> unchanged %s" % (args[:2] + (args[2] is None, args[3] is None),),
        got == (args[0], args[1]),
        str(got),
    )

tight = lb.plan_loader(4, 8, 512 * MiB, 8 * GiB)  # 4*8*512MiB = 16 GiB wanted, 2 GiB budget
check("A4 a tight budget really does reduce", tight != (4, 8) and tight[0] * tight[1] <= 4 * 8, str(tight))
check("A5 the two arms disagree", roomy != tight, "%s vs %s" % (roomy, tight))

starved = lb.plan_loader(4, 8, 4 * GiB, 2 * GiB)  # one batch alone busts the budget
check("A6 nothing fits -> 0 workers (in-process, no file mappings)", starved[0] == 0, str(starved))


# ── B. the documented ladder ─────────────────────────────────────────────────

print("B. ladder order")

# want 4*8=32 in flight; budget allows 16 -> prefetch alone can absorb it (4*4=16)
step1 = lb.plan_loader(4, 8, MiB, int(16 * MiB / lb.BUDGET_FRACTION))
check("B1 prefetch drops FIRST, workers untouched", step1[0] == 4 and step1[1] < 8, str(step1))
# budget allows 4 -> prefetch is already at the floor, workers must give
step2 = lb.plan_loader(4, 8, MiB, int(4 * MiB / lb.BUDGET_FRACTION))
check("B2 then workers drop, prefetch pinned at the floor",
      step2[0] < 4 and step2[1] == lb.MIN_PREFETCH, str(step2))
check("B3 the floor is torch's own default", lb.MIN_PREFETCH == 2, str(lb.MIN_PREFETCH))

# ⚠ v1 of this check asserted "a 20 GiB box with 256 MiB batches reduces" and it
# measured FAIL — 5*2*256 MiB = 2560 MiB fits inside 25% of 20 GiB. The inputs
# were MY invention (the reporting machine's real batch size is unknown and is
# NOT claimed anywhere in this gate), so the fabricated assertion was deleted
# rather than the number tuned until it agreed. What IS a property of the code is
# where the boundary sits, so that is what gets pinned:
budget = int(20 * GiB * lb.BUDGET_FRACTION)
edge = budget // 10  # 5 workers x prefetch 2 = 10 in flight
check("B4 exactly at the boundary it still fits", lb.plan_loader(5, 2, edge, 20 * GiB) == (5, 2),
      "%.1f MiB batches" % (edge / MiB))
check("B4b one byte over the boundary it reduces", lb.plan_loader(5, 2, edge + 1, 20 * GiB) != (5, 2),
      str(lb.plan_loader(5, 2, edge + 1, 20 * GiB)))


# ── C. ★ the RNG discipline ──────────────────────────────────────────────────

print("C. global RNG is not disturbed by the probe")


class RngHungryDataset(torch.utils.data.Dataset):
    """Stands in for sovits/sovits_v2: __getitem__ draws from all three streams."""

    def __len__(self):
        return 64

    def __getitem__(self, i):
        random.random()
        np.random.rand()
        return torch.rand(3, 128)


def collate(xs):
    return {"x": torch.stack(xs), "meta": [1, 2, 3]}


def rng_snapshot():
    return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().clone())


def rng_equal(a, b):
    return a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


random.seed(11)
np.random.seed(11)
torch.manual_seed(11)
before = rng_snapshot()
size = lb.probe_batch_bytes(RngHungryDataset(), collate, batch_size=4)
after = rng_snapshot()
check("C1 the probe measures a real batch", size == 4 * 3 * 128 * 4, "%s bytes" % size)
check("C2 python+numpy+torch RNG all byte-identical afterwards", rng_equal(before, after))

# ★ control: without the restore, the state DOES move. Without this, C2 could be
# comparing two states that were never going to differ (S98: a control that
# cannot move proves nothing).
random.seed(11)
np.random.seed(11)
torch.manual_seed(11)
ctl_before = rng_snapshot()
_ = torch.utils.data.DataLoader(RngHungryDataset(), batch_size=4, num_workers=0, collate_fn=collate)
_ = next(iter(_))
ctl_after = rng_snapshot()
check("C3 CONTROL: an unrestored read really does move all three",
      not rng_equal(ctl_before, ctl_after), "so C2 is capable of failing")

# a dataset that explodes must not take the run with it
class Boom(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        raise ValueError("boom")


check("C4 a failing probe returns None instead of raising",
      lb.probe_batch_bytes(Boom(), collate, batch_size=2) is None)
check("C5 ...and an unmeasurable probe means 'change nothing'",
      lb.plan_loader(4, 8, None, 20 * GiB) == (4, 8))


# ── D. measure_batch_bytes ───────────────────────────────────────────────────

print("D. measurement")

nested = {"a": torch.zeros(10, dtype=torch.float32), "b": [torch.zeros(5, dtype=torch.float64), "str", 7]}
check("D1 counts nested tensors, ignores non-tensors",
      lb.measure_batch_bytes(nested) == 10 * 4 + 5 * 8,
      "%d bytes" % lb.measure_batch_bytes(nested))
check("D2 dtype-aware (fp16 half of fp32)",
      lb.measure_batch_bytes(torch.zeros(100, dtype=torch.float16)) * 2
      == lb.measure_batch_bytes(torch.zeros(100, dtype=torch.float32)))


# ── E. call sites ────────────────────────────────────────────────────────────

print("E. call sites")

SITES = [
    ("rvc", "training/utai_train/rvc/train.py"),
    ("sovits", "training/utai_train/sovits/train.py"),
    ("sovits_v2", "training/utai_train/sovits_v2/data_utils.py"),
]
for name, rel in SITES:
    src = read(rel)
    check("E1 %-9s calls loader_budget.plan_loader(" % name, "loader_budget.plan_loader(" in src)
    check("E1b %-9s measures a real batch" % name, "loader_budget.probe_batch_bytes(" in src)
    # torch REJECTS prefetch_factor/persistent_workers when workers == 0 (see F),
    # so the guard's own 0-worker rung would crash without the conditional.
    check(
        "E2 %-9s passes prefetch/persistent conditionally" % name,
        ('if num_workers > 0' in src) or ('if _lb_workers > 0' in src),
    )


# ── F. why E2 is not a style preference ──────────────────────────────────────

print("F. torch's own contract")

ds = RngHungryDataset()
rejected = 0
for kw in ({"prefetch_factor": 2}, {"persistent_workers": True}):
    try:
        torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate, **kw)
    except ValueError:
        rejected += 1
check("F1 torch rejects prefetch_factor AND persistent_workers at 0 workers",
      rejected == 2, "%d/2 rejected on torch %s" % (rejected, torch.__version__))


# ── G. the decision is never silent ──────────────────────────────────────────

print("G. the decision log")

CAP.on = True
CAP.records.clear()
lb.plan_loader(4, 8, 8 * MiB, 64 * GiB)
fit_recs = list(CAP.records)
CAP.records.clear()
lb.plan_loader(4, 8, 512 * MiB, 8 * GiB)
cut_recs = list(CAP.records)
CAP.on = False

# ⚠ v1 of G1 only counted records, and probe L2 (delete the early return, so a
# fitting machine falls through and logs the REDUCING warning while changing
# nothing) measured GREEN against it. The level is the load-bearing half: a run
# that changed nothing must not scream 1455 at the user, or the warning that
# matters gets trained away.
check("G1 a KEPT decision logs exactly one INFO (not a warning)",
      len(fit_recs) == 1 and fit_recs[0].levelno < logging.WARNING,
      "%d record(s) @ %s" % (len(fit_recs), fit_recs[0].levelname if fit_recs else "-"))
check("G2 a REDUCED decision logs one WARNING", len(cut_recs) == 1 and cut_recs[0].levelno >= logging.WARNING,
      "%d record(s) @ %s" % (len(cut_recs), cut_recs[0].levelname if cut_recs else "-"))
cut_text = cut_recs[0].getMessage() if cut_recs else ""
for token in ("1455", "pagefile", "available commit"):
    check("G3 the warning names %-16s" % token, token in cut_text)
# S110: a machine that "correctly did nothing" and a machine where the guard never
# ran must not look the same in the log.
check("G4 kept and reduced lines are distinguishable",
      fit_recs and cut_recs and fit_recs[0].getMessage() != cut_text)


print("")
if FAILURES:
    print("FAIL: %d/%d checks failed -> %s" % (len(FAILURES), CHECKS, ", ".join(FAILURES)))
    sys.exit(1)
print("PASS: %d/%d checks" % (CHECKS, CHECKS))
