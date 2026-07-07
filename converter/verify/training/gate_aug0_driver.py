# -*- coding: utf-8 -*-
"""S41 noop-gate driver: run ONE backend's pipeline.run() in-process (the
ORCHESTRATION layer — stage-function direct calls would not cover the S41
stage reordering, red-team V3) up to the train_prep stage, then stop.

    python gate_aug0_driver.py --code-root <repo>/training --backend sovits \
        --config cfg.json

CPU-pinned (CUDA_VISIBLE_DEVICES=-1 BEFORE torch import — bitwise noop
comparison is a CPU-axis gate, red-team V5; note the Windows "empty env var =
deleted" trap: -1 sentinel, never ""). Exit 0 = reached train_prep; 3 = the
pipeline returned without ever emitting train_prep (unexpected)."""
import argparse
import importlib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

PIPELINES = {
    "sovits": "utai_train.sovits.pipeline",
    "sovits_diff": "utai_train.sovits.diff_pipeline",
    "rvc": "utai_train.rvc.pipeline",
    "vocoder": "utai_train.vocoder.pipeline",
}


class StopAtTrainPrep(Exception):
    pass


class GateReporter:
    def stage(self, stage, done=None, total=None, message=None, force=False):
        if stage == "train_prep":
            raise StopAtTrainPrep()

    def step(self, *a, **k):
        pass

    def ckpt(self, *a, **k):
        pass

    def done(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


class NoStop:
    def check(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True, help="<repo>/training to import from")
    ap.add_argument("--backend", required=True, choices=sorted(PIPELINES))
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, os.path.abspath(args.code_root))

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    mod = importlib.import_module(PIPELINES[args.backend])
    try:
        mod.run(cfg, GateReporter(), NoStop())
    except StopAtTrainPrep:
        print("STOPPED_AT_TRAIN_PREP")
        return
    print("pipeline returned without train_prep", file=sys.stderr)
    sys.exit(3)


if __name__ == "__main__":
    main()
