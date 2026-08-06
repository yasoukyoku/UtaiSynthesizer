"""Gate: the S114 §F5-3 numerical-divergence guard (community GitHub issue #2).

    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate_numerics_guard.py

Self-contained: no dataset, no GPU, seconds. Every check below drives PRODUCTION
code (utai_train.numerics + the real train.py text), never a re-implementation.

WHAT IT PROVES, and why each part is here
-----------------------------------------
(1) THE DEFECT IS REAL, and it is specifically the *stale* metric.
    The epoch-boundary predicate is unchanged by the fix:

        ema_mel is not None and math.isfinite(ema_mel)
            and (best_metric is None or ema_mel < best_metric)

    Check A1 pins that exact text in all three trainers, so the replay below is
    about the predicate production actually evaluates, not one I typed out.
    Check A2 replays the issue-#2 shape through it: losses descend, the EMA
    follows, everything collapses to nan, the EMA FREEZES at its last healthy
    value -- and at the next epoch boundary the predicate is still True. That is
    the single write that destroyed the reporter's best.pth.
    ★ A2 carries its own DISCRIMINATING CONTROL: the same replay with a metric
    RECOMPUTED from the current weights (what vocoder/harness.py and
    diffusion/solver.py do) returns False. Without that control A2 would only
    show "a predicate returned True", not "the staleness is the cause".

(2) THE FIX BLOCKS IT (B), and the halting guard behaves (C/D).

(3) THE WIRING EXISTS (E/F) -- the CODE reaches i18n in all three languages, and
    every trainer actually CALLS both guards. A guard nobody calls is the S109
    §G14 shape: it passes review and protects nothing.

(4) THE SAFE SIBLINGS STAY SAFE (G). vocoder/harness.py and diffusion/solver.py
    are NOT vulnerable, but not by luck: their best metric is recomputed from the
    current model every time, so a poisoned model yields a nan metric and their
    own isfinite/comparison rejects it. If someone "unifies" them onto an EMA
    later, that property silently disappears -- G goes red first.

MUTATION PROBES (run by hand; each must turn exactly one check red)
------------------------------------------------------------------
    M1  numerics.best_save_is_safe: "return True" at the top      -> B2/B3/B4/D2 red
    M2  numerics.DEFAULT_PATIENCE = 1                             -> C1/C2 red
    M3  DivergenceGuard.observe: drop the "self._consecutive = 0" reset
                                                                  -> C1/C3 red
    M4b numerics.first_nonfinite_tensor: drop the torch.is_tensor skip
                                                                  -> D1b red
    M5  delete "divergence.observe(" from any one trainer         -> F1 red
    M6  delete the TRAINING_NUMERICS_DIVERGED line from en.json   -> E2 red
    M7  change the epoch predicate text in any one trainer        -> A1 red
    M8  advance best_metric without checking save_best's result   -> F4 red
    M9  give the vocoder an EMA-based best                        -> G1/G4 red

⛔ ONE PROBE MEASURED GREEN AND IS RECORDED AS SUCH, not deleted:
    M4  drop the is_floating_point() skip -> NOTHING goes red. torch 2.5.1's
        isfinite accepts int64/bool/uint8 and returns True, so that skip changes
        no answer today. D1 is therefore a behaviour PIN with no mutation behind
        it, and it says so in its own name. (Deleting the inconvenient probe
        instead would leave a gate that looks fully covered and isn't.)
Driver + full log: scratchpad mutate_s114_numerics.py — it never edits the repo,
it copies the read set into a sandbox and points UTAI_GATE_REPO at it, and it
refuses to report anything unless the unmutated sandbox is green first.
"""

import io
import json
import math
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
# UTAI_GATE_REPO points the whole gate at a COPY of the tree. It exists for the
# mutation probes above: they must be able to break a file without ever touching
# the repository (this project has twice lost real files to a script that edited
# in place). Unset = the real repo.
REPO = os.environ.get("UTAI_GATE_REPO") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

import torch  # noqa: E402

from utai_train import numerics  # noqa: E402

FAILURES = []
CHECKS = 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    mark = "PASS" if ok else "FAIL"
    print("  [%s] %-46s %s" % (mark, name, detail))
    if not ok:
        FAILURES.append(name)


def read(rel):
    with io.open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


TRAINERS = [
    "training/utai_train/rvc/train.py",
    "training/utai_train/sovits/train.py",
    "training/utai_train/sovits_v2/train.py",
]

# The epoch-boundary predicate, verbatim from production (whitespace-normalized).
# The fix did NOT touch it -- it only changed what happens once it is True.
EPOCH_PREDICATE = (
    "ema_mel is not None and math.isfinite(ema_mel) "
    "and (best_metric is None or ema_mel < best_metric)"
)


def norm_ws(s):
    return re.sub(r"\s+", " ", s)


# ── A. the defect is real, and staleness is the cause ────────────────────────

print("A. the defect")

a1_ok = True
a1_detail = []
for rel in TRAINERS:
    hits = norm_ws(read(rel)).count(EPOCH_PREDICATE)
    a1_detail.append("%s=%d" % (rel.split("/")[-2], hits))
    # each trainer evaluates it twice: the graceful-stop path and the epoch end
    if hits != 2:
        a1_ok = False
check("A1 epoch predicate pinned in all 3 trainers", a1_ok, " ".join(a1_detail))

# Real production constant -- imported, not re-typed.
from utai_train.rvc.train import BEST_EMA_ALPHA  # noqa: E402


def replay(stale_metric):
    """Issue-#2 shape. Returns (fired, ema_at_boundary, best_at_boundary).

    stale_metric=True  -> the trainers' EMA: only updated on finite steps.
    stale_metric=False -> the vocoder/diffusion shape: the metric IS the current
                          step's value, so a poisoned model yields nan.
    """
    ema = None
    best = None
    fired = False
    EPOCH = 200
    COLLAPSE_AT = 1050
    for step in range(1, 1201):
        # a healthy run whose mel is still slowly improving, then total collapse
        raw = float("nan") if step >= COLLAPSE_AT else 20.0 - 4.0 * (step / 1200.0)
        if stale_metric:
            if math.isfinite(raw):
                ema = raw if ema is None else BEST_EMA_ALPHA * raw + (1 - BEST_EMA_ALPHA) * ema
        else:
            ema = raw  # recomputed from the CURRENT model every time
        if step % EPOCH == 0:
            # ↓ production's predicate, character-for-character
            if ema is not None and math.isfinite(ema) and (best is None or ema < best):
                best = ema
                if step >= COLLAPSE_AT:
                    fired = True  # a save with POISONED weights
    return fired, ema, best


fired_stale, ema_stale, _ = replay(stale_metric=True)
fired_live, ema_live, _ = replay(stale_metric=False)
check(
    "A2 stale EMA fires save_best AFTER the collapse",
    fired_stale is True,
    "frozen ema=%.4f at the post-collapse boundary" % ema_stale,
)
check(
    "A2c control: a live metric does NOT fire",
    fired_live is False and math.isnan(ema_live),
    "live metric = nan -> isfinite() rejects it",
)
check(
    "A2d the two arms really disagree",
    fired_stale != fired_live,
    "stale=%s live=%s" % (fired_stale, fired_live),
)


# ── B. the fix blocks it ─────────────────────────────────────────────────────

print("B. best_save_is_safe")


def tiny_net(poison=None):
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    if poison is not None:
        with torch.no_grad():
            list(m.parameters())[2][0, 0] = poison
    return m


clean_ok = numerics.best_save_is_safe(tiny_net().state_dict())
nan_ok = numerics.best_save_is_safe(tiny_net(float("nan")).state_dict())
inf_ok = numerics.best_save_is_safe(tiny_net(float("inf")).state_dict())
check("B1 clean weights -> save allowed", clean_ok is True)
check("B2 nan weights -> save refused", nan_ok is False)
check("B3 inf weights -> save refused", inf_ok is False)
check("B4 the two arms really disagree", clean_ok != nan_ok, "True vs False")


# ── C. the halting guard ─────────────────────────────────────────────────────

print("C. DivergenceGuard")

NAN = {"mel": float("nan"), "gen": float("nan")}
OK = {"mel": 1.0, "gen": 2.0}


def guard(net=None):
    return numerics.DivergenceGuard(
        (("G", net if net is not None else tiny_net()),), logger=None
    )


g = guard()
c1_err = None
try:
    g.observe(1, OK)
    g.observe(2, NAN)
    g.observe(3, OK)
except RuntimeError as e:
    # a guard that halts here would kill every legitimate fp16 run; report it as
    # a FAILED CHECK, not as a gate crash that hides everything below
    c1_err = str(e)
check(
    "C1 a SINGLE nan step does not halt (fp16 GradScaler case)",
    c1_err is None and g.consecutive == 0,
    c1_err[:70] if c1_err else "counter reset by the finite step",
)

g = guard()
raised = None
c2_err = None
try:
    for i in range(numerics.DEFAULT_PATIENCE - 1):
        g.observe(i, NAN)
except RuntimeError as e:
    c2_err = str(e)
check(
    "C2 patience-1 consecutive nan steps still do not halt",
    c2_err is None and g.consecutive == numerics.DEFAULT_PATIENCE - 1,
    c2_err[:70] if c2_err else "consecutive=%d patience=%d" % (g.consecutive, numerics.DEFAULT_PATIENCE),
)
try:
    g.observe(numerics.DEFAULT_PATIENCE, NAN)
except RuntimeError as e:
    raised = str(e)
check(
    "C2b the patience-th consecutive nan step halts",
    raised is not None and raised.startswith(numerics.CODE_DIVERGED + ":"),
    (raised or "NO RAISE")[:96],
)

# consecutive, not cumulative: patience-1 bad, one good, patience-1 bad again
g = guard()
for i in range(numerics.DEFAULT_PATIENCE - 1):
    g.observe(i, NAN)
g.observe(999, OK)
still_alive = True
try:
    for i in range(numerics.DEFAULT_PATIENCE - 1):
        g.observe(1000 + i, NAN)
except RuntimeError:
    still_alive = False
check(
    "C3 the counter is CONSECUTIVE, not cumulative",
    still_alive,
    "2x(patience-1) with a good step between them survives",
)

# the message must say WHICH side is dead -- that changes what the user should do
msg_poisoned = None
g = guard(tiny_net(float("nan")))
try:
    for i in range(numerics.DEFAULT_PATIENCE):
        g.observe(i, NAN)
except RuntimeError as e:
    msg_poisoned = str(e)
msg_clean = None
g = guard(tiny_net())
try:
    for i in range(numerics.DEFAULT_PATIENCE):
        g.observe(i, NAN)
except RuntimeError as e:
    msg_clean = str(e)
check(
    "C4 message names the poisoned weight",
    msg_poisoned is not None and "is non-finite" in msg_poisoned,
    (msg_poisoned or "")[-58:],
)
check(
    "C4b and says so when the weights are still clean",
    msg_clean is not None and "weights=all finite" in msg_clean,
    "the two messages differ: %s" % (msg_poisoned != msg_clean),
)


# ── D. non-float entries must not be misread ─────────────────────────────────

print("D. non-float state entries")


class WithIntBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.register_buffer("num_batches_tracked", torch.tensor(7, dtype=torch.long))
        self.register_buffer("flag", torch.tensor(True))


try:
    d_ok = numerics.best_save_is_safe(WithIntBuffer().state_dict()) is True
    d_detail = "int64 + bool buffers allowed"
except Exception as e:
    d_ok = False
    d_detail = "raised %s: %s" % (type(e).__name__, e)
# ⚠ HONEST LABEL: this is a behaviour PIN, not a test of the is_floating_point()
# skip. Probe M4 (delete that skip) measured GREEN — torch 2.5.1's isfinite
# accepts int64/bool/uint8 and returns True, so the skip changes no answer today.
# Recorded rather than quietly dropped: an assertion nobody can turn red is worth
# exactly as much as its documented reason.
check("D1 integer/bool buffers neither crash nor false-positive [no mutation]", d_ok, d_detail)

# ...whereas THIS skip is load-bearing, and M4b turns it red.
try:
    d1b = numerics.first_nonfinite_tensor([("meta", "not-a-tensor"), ("w", torch.zeros(2))]) is None
    d1b_detail = "non-tensor entry skipped"
except Exception as e:
    d1b = False
    d1b_detail = "raised %s: %s" % (type(e).__name__, e)
check("D1b a non-tensor entry is skipped, not raised", d1b, d1b_detail)

# and a poisoned FLOAT buffer must still be caught (buffers are not just params)
class WithFloatBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.register_buffer("running_mean", torch.tensor([float("nan"), 0.0]))


check(
    "D2 a poisoned float BUFFER is caught too",
    numerics.best_save_is_safe(WithFloatBuffer().state_dict()) is False,
    "not only named_parameters()",
)
check(
    "D3 module scan covers buffers as well",
    numerics.first_nonfinite_module((("G", WithFloatBuffer()),)) == "G.running_mean",
    numerics.first_nonfinite_module((("G", WithFloatBuffer()),)) or "None",
)


# ── E. cross-language wiring ─────────────────────────────────────────────────

print("E. i18n wiring")

CODE = numerics.CODE_DIVERGED
be = read("src/lib/backendError.ts")
check(
    "E1 backendError.ts maps the CODE",
    ('%s: { key: "backend.%s" }' % (CODE, CODE)) in be,
    CODE,
)
for lang in ("zh", "en", "ja"):
    data = json.loads(read("src/i18n/%s.json" % lang))
    txt = data.get("backend", {}).get(CODE)
    check(
        "E2 %s.json has backend.%s" % (lang, CODE),
        isinstance(txt, str) and len(txt) > 30,
        "%d chars" % (len(txt) if isinstance(txt, str) else 0),
    )


# ── F. every trainer actually calls both guards ──────────────────────────────

print("F. call sites")

for rel in TRAINERS:
    src = read(rel)
    name = rel.split("/")[-2]
    check(
        "F1 %-9s calls divergence.observe(" % name,
        src.count("divergence.observe(") == 1,
        "%d call(s)" % src.count("divergence.observe("),
    )
    check(
        "F2 %-9s save_best consults numerics" % name,
        "numerics.best_save_is_safe(" in src,
    )
    # the old single-argument form must be gone, or a call site was missed
    check(
        "F3 %-9s no stale save_best(epoch) call" % name,
        not re.search(r"save_best\(epoch\)\s*$", src, re.M),
    )
    # and the caller must only advance the metric on a successful save
    check(
        "F4 %-9s advances best only when saved" % name,
        src.count("if save_best(epoch, ema_mel, global_step):") == 2,
        "%d guarded call(s)" % src.count("if save_best(epoch, ema_mel, global_step):"),
    )


# ── G. the structurally-safe siblings must STAY that way ─────────────────────

print("G. safe-by-construction siblings")

voc = read("training/utai_train/vocoder/harness.py")
check(
    "G1 vocoder best still uses the LIVE validation loss",
    "if math.isfinite(v) and (self.best_val is None or v < self.best_val):" in voc,
    "recomputed per validation -> a poisoned model yields nan",
)
sol = read("training/utai_train/sovits/diffusion/solver.py")
check(
    "G2 diffusion still raises on a nan loss",
    "if torch.isnan(loss):" in sol,
    "upstream halt kept",
)
check(
    "G3 diffusion best still compares the LIVE test_loss",
    "if best_state is not None and (best_metric is None or test_loss < best_metric):" in sol,
)
check(
    "G4 neither sibling grew an EMA-based best",
    "BEST_EMA_ALPHA" not in voc and "BEST_EMA_ALPHA" not in sol,
    "an EMA here would reintroduce the stale-metric hole",
)


# ── verdict ──────────────────────────────────────────────────────────────────

print("")
if FAILURES:
    print("FAIL: %d/%d checks failed -> %s" % (len(FAILURES), CHECKS, ", ".join(FAILURES)))
    sys.exit(1)
print("PASS: %d/%d checks" % (CHECKS, CHECKS))
