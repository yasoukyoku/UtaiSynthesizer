# -*- coding: utf-8 -*-
"""Export/compare the NAME-level test baseline for the whole workspace.

WHY NAMES AND NOT A COUNT (S99 blood lesson): a count is structurally blind to a silently
deregistered test. S99 inserted a new `#[test]` between an existing attribute and its function,
which deregistered `s94_en_onset_vote_gate` — and because one test went out as one came in, the
total was UNCHANGED. Only a name-level diff catches that.

WHY THIS FILE EXISTS AT ALL (the debt it pays off): S100 wrote this as a scratch script with the
output path hard-coded, and `os.remove(OUT)` on start. Running it unmodified DESTROYS the very
baseline it is meant to be compared against, silently. S101 and S102 each had to copy it and edit
the constant — three rounds paying the same tax. Here the paths are ARGUMENTS, nothing is deleted
unless you name it, and --check never writes.

Covers all 19 test binaries. ⚠ The first version of this baseline (S100) covered only 15: it missed
`utai_dsp` (30 tests) and `utai_stretch` (8), i.e. 38 tests that were never in the baseline at all.

Run (from anywhere):
  py -3.10 scripts/testnames.py --out baseline.txt          # write a baseline
  py -3.10 scripts/testnames.py --check baseline.txt        # diff HEAD against it (writes nothing)
  py -3.10 scripts/testnames.py --check old.txt --out new.txt

Exit code is 1 when --check finds any name added or removed, so it can gate a release script.
⚠ It lists what is COMPILED in target/debug/deps — run `cargo test --workspace --no-run` (or a full
`cargo test`) first, or you will be diffing stale binaries (recurring pitfall #4).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src-tauri", "target", "debug", "deps")
STEMS = [
    "utai_lib", "audition_render", "autotune_parity", "download_http",
    "e5_breathy_probe", "game_batch", "game_parity", "onnx_inference",
    "pyenv_extract_gates", "pyenv_pack", "separation_pipeline",
    "tempo_oracle", "vocoder_import", "voice_mem_profile", "voice_pipeline",
    "dictionary_distribution",
    "utai", "utai_dsp", "utai_stretch",
]
LINE = re.compile(r"^(.+): test$")


def collect(verbose=True):
    rows, total, missing = [], 0, []
    for stem in STEMS:
        cands = glob.glob(os.path.join(DEPS, stem + "-*.exe"))
        if not cands:
            missing.append(stem)
            continue
        exe = max(cands, key=os.path.getmtime)
        p = subprocess.run([exe, "--list"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        n = 0
        for ln in p.stdout.splitlines():
            m = LINE.match(ln.strip())
            if m:
                rows.append(f"{stem}\t{m.group(1)}")
                n += 1
        total += n
        if verbose:
            print(f"{stem:24s} {n:4d}  ({os.path.basename(exe)})")
    rows.sort()
    return rows, total, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the baseline here (never a default — S100's hard-coded path is the bug this fixes)")
    ap.add_argument("--check", help="diff the current build against this baseline file")
    a = ap.parse_args()
    if not a.out and not a.check:
        ap.error("give --out, --check, or both")

    rows, total, missing = collect()
    print(f"\nTOTAL test names = {total}")
    if missing:
        # LOUD: a stem whose binary is absent silently shrinks the baseline, which reads as
        # "tests disappeared" on the next --check. Say so instead.
        print(f"⚠ NOT BUILT (excluded from this listing): {', '.join(missing)} — run `cargo test --workspace --no-run` first")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"wrote {a.out}")

    if a.check:
        with open(a.check, encoding="utf-8") as f:
            old = {l for l in f.read().splitlines() if l.strip()}
        new = set(rows)
        gone, added = sorted(old - new), sorted(new - old)
        print(f"\n=== vs {a.check} ({len(old)} names) ===")
        print(f"  DISAPPEARED : {len(gone)}")
        for x in gone:
            print(f"     - {x}")
        print(f"  NEW         : {len(added)}")
        for x in added:
            print(f"     + {x}")
        if gone or added:
            print("\n⚠ A DISAPPEARED name is the dangerous direction — a test can be deregistered by "
                  "inserting a new #[test] between an existing attribute and its fn (S99).")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
