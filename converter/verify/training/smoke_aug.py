# -*- coding: utf-8 -*-
"""S41 smoke matrix — REAL short training runs through the runner (protocol
JSONL on stdout), aug_copies=1, GPU allowed (this is the real-world axis; the
bitwise gates run CPU-pinned separately).

Per run asserts: clean protocol (every stdout line parses as JSON), the stage
sequence contains augment + aug_check, a done message with reason=completed,
no error messages, >=1 step message. The dirty-mix sovits run additionally
asserts the gate rejected >=1 copy (the 剔除 message).

    ..\\..\\..\\training\\.venv\\Scripts\\python.exe ^
        ..\\converter\\verify\\training\\smoke_aug.py [--only sovits]

Runs: sovits (dirty-mix dataset, aug rejection live) -> sovits_diff (same
workspace, inherited copies) -> rvc -> vocoder. JSONL transcripts land next to
the workspaces (TESTING/utai-v2-testing/gate_aug/smoke_*.jsonl).
"""
import argparse
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_aug0_noop as noop  # build_cfg / paths / fixture

APP = noop.APP
GATE_ROOT = noop.GATE_ROOT
FAILURES = []


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILURES.append(name)


def train_cfg(backend, ws, copies, dataset_dir):
    tr = os.path.join(APP, "data", "models", "training")
    cfg = noop.build_cfg(backend, ws)
    cfg["aug_copies"] = copies
    cfg["dataset_dir"] = dataset_dir
    # real training needs the real base models (the noop driver never reached
    # them; the runner does)
    if backend == "sovits":
        cfg["pretrain_g"] = os.path.join(tr, "sovits", "vec768", "G_0.pth")
        cfg["pretrain_d"] = os.path.join(tr, "sovits", "vec768", "D_0.pth")
        cfg.update({"total_epoch": 2, "batch_size": 2, "save_every_steps": 800,
                    "keep_ckpts": 2})
    elif backend == "rvc":
        cfg["pretrain_g"] = os.path.join(tr, "rvc", "pretrained_v2", "f0G48k.pth")
        cfg["pretrain_d"] = os.path.join(tr, "rvc", "pretrained_v2", "f0D48k.pth")
        cfg.update({"total_epoch": 2, "batch_size": 2, "save_every_epoch": 1})
    elif backend == "sovits_diff":
        cfg["assets"]["diffusion_pretrain"] = os.path.join(
            tr, "sovits", "diffusion", "vec768", "model_0.pt"
        )
        cfg.update({"total_steps": 6, "batch_size": 2, "save_every_steps": 3,
                    "interval_force_save": 6, "k_step_max": 0})
    elif backend == "vocoder":
        cfg.update({"total_steps": 4, "save_every_steps": 2, "batch_size": 2,
                    "keep_ckpts": 2, "crop_mel_frames": 32, "freeze_mpd": False})
    return cfg


def run_smoke(backend, ws, copies, dataset_dir, wipe=True, tag=None):
    tag = tag or backend
    print("== smoke %s (copies=%d)" % (tag, copies))
    if wipe and os.path.isdir(ws):
        import shutil

        shutil.rmtree(ws)
    cfg = train_cfg(backend, ws, copies, dataset_dir)
    cfg_path = os.path.join(GATE_ROOT, "smoke_%s.json" % tag)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    r = subprocess.run(
        [noop.VENV_PY, "-m", "utai_train.runner", "--config", cfg_path],
        cwd=os.path.join(APP, "training"),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800,
    )
    jsonl_path = os.path.join(GATE_ROOT, "smoke_%s.jsonl" % tag)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(r.stdout or "")

    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    msgs, bad_lines = [], []
    for ln in lines:
        try:
            msgs.append(json.loads(ln))
        except Exception:
            bad_lines.append(ln)
    check("%s: protocol clean (all stdout lines are JSON)" % tag, not bad_lines,
          str(bad_lines[:2]))
    stages = [m.get("stage") for m in msgs if m.get("type") == "stage"]
    check("%s: augment stage emitted" % tag, "augment" in stages)
    check("%s: aug_check stage emitted" % tag, "aug_check" in stages)
    dones = [m for m in msgs if m.get("type") == "done"]
    errors = [m for m in msgs if m.get("type") == "error"]
    steps = [m for m in msgs if m.get("type") == "step"]
    check("%s: completed" % tag,
          len(dones) == 1 and dones[0].get("reason") == "completed" and not errors,
          "rc=%d errors=%s" % (r.returncode, [e.get("message") for e in errors][:1]))
    check("%s: trained steps" % tag, len(steps) >= 1, "(%d step msgs)" % len(steps))
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    noop.ensure_fixture()
    dirty_ds = os.path.join(GATE_ROOT, "dataset_dirty")
    clean_ds = noop.FIXTURE
    ws_sov = os.path.join(GATE_ROOT, "smoke_ws_sovits")

    if args.only in ("all", "sovits"):
        msgs = run_smoke("sovits", ws_sov, 1, dirty_ds)
        gate_msgs = [m.get("message", "") for m in msgs
                     if m.get("type") == "stage" and m.get("stage") == "aug_check"]
        check("sovits: gate rejected >=1 (dirty mix)",
              any("剔除" in (m or "") and "剔除 0" not in (m or "") for m in gate_msgs),
              str(gate_msgs[-1:]))
    if args.only in ("all", "sovits_diff"):
        # same workspace — the inherited-copies path (runner-level; the Rust
        # manifest layer is exercised in the live app test)
        run_smoke("sovits_diff", ws_sov, 1, dirty_ds, wipe=False)
    if args.only in ("all", "rvc"):
        run_smoke("rvc", os.path.join(GATE_ROOT, "smoke_ws_rvc"), 1, clean_ds)
    if args.only in ("all", "vocoder"):
        run_smoke("vocoder", os.path.join(GATE_ROOT, "smoke_ws_vocoder"), 1, clean_ds)

    if FAILURES:
        print("RESULT: FAIL (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
