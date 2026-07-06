"""gate1_diff step 4 — step-for-step loss-trajectory comparison via the
tensorboard event files both sides wrote (full float32 precision; stdout logs
only carry 3 decimals).

Series: train/loss (every step, interval_log=1) + validation/loss (steps
8/16/24 — the first boundary contains the lazy NSF-HiFiGAN Generator
construction's RNG block, so aligning PAST it proves the RNG-consumption
model is complete).

PASS: train/loss same step sets, max |rel diff| <= 1e-6 (same torch, same
fp32 CPU math — S39 measured EXACTLY 0.0 over all 24 steps).
validation/loss: compared on the step INTERSECTION (>= 2 boundaries required)
— the original never closes its SummaryWriter, so the LAST validation scalar
can sit in the unflushed buffer when the process exits (S39: orig TB held
[8,16] while its stdout printed all three; flush_secs=120 default). Ours must
be a superset of orig's steps, and any orig-TB-missing point is cross-checked
against the 3-decimal value in the orig stdout log.

Run (our venv):
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate1_diff_compare.py
"""
import os

from tensorboard.backend.event_processing import event_accumulator

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
REL_LINE = 1e-6


def load_scalars(expdir):
    logdir = os.path.join(expdir, "logs")
    ea = event_accumulator.EventAccumulator(
        logdir, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    out = {}
    for tag in ("train/loss", "validation/loss"):
        if tag in ea.Tags()["scalars"]:
            out[tag] = {e.step: e.value for e in ea.Scalars(tag)}
    return out


def stdout_val_losses():
    """The '--- <validation> ---\\nloss: X.XXX.' pairs from the orig stdout log,
    in run order (3-decimal precision — the fallback oracle for TB-buffer-lost
    points)."""
    vals = []
    log = os.path.join(TESTING, "gate1_diff_orig_stdout.log")
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if "<validation>" in line and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt.startswith("loss:"):
                vals.append(float(nxt[len("loss:"):].strip().rstrip(".")))
    return vals


def main():
    orig = load_scalars(os.path.join(TESTING, "gate1_diff_orig"))
    ours = load_scalars(os.path.join(TESTING, "gate1_diff_ours"))
    ok = True

    a, b = orig.get("train/loss", {}), ours.get("train/loss", {})
    if set(a) != set(b):
        print(f"train/loss: STEP SET MISMATCH orig={sorted(a)} ours={sorted(b)}")
        ok = False
    else:
        worst = max(
            (abs(a[s] - b[s]) / max(abs(a[s]), 1e-12) for s in a), default=0.0
        )
        print(f"train/loss: {len(a)} steps aligned, max_rel {worst:.3e}")
        ok = ok and worst <= REL_LINE

    va, vb = orig.get("validation/loss", {}), ours.get("validation/loss", {})
    inter = sorted(set(va) & set(vb))
    print(f"validation/loss: orig steps {sorted(va)}, ours steps {sorted(vb)}")
    if not set(va) <= set(vb) or len(inter) < 2:
        print("validation/loss: ours must cover orig's steps with >=2 boundaries")
        ok = False
    else:
        worst = max(
            (abs(va[s] - vb[s]) / max(abs(va[s]), 1e-12) for s in inter), default=0.0
        )
        print(f"validation/loss: {len(inter)} intersecting steps, max_rel {worst:.3e}")
        ok = ok and worst <= REL_LINE
        missing = sorted(set(vb) - set(va))
        if missing:
            stdout_vals = stdout_val_losses()
            if len(stdout_vals) != len(vb):
                print(f"validation/loss: stdout printed {len(stdout_vals)} vals, "
                      f"ours has {len(vb)} — cannot cross-check")
                ok = False
            else:
                by_step = dict(zip(sorted(vb), stdout_vals))
                for s in missing:
                    d = abs(by_step[s] - vb[s])
                    print(f"validation/loss step {s} (orig TB buffer-lost): "
                          f"stdout {by_step[s]:.3f} vs ours {vb[s]:.6f} (|d|={d:.2e})")
                    ok = ok and d <= 5e-4  # 3-decimal print precision

    print("GATE1 DIFF:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
