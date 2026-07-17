"""Training sidecar entry point.

    python -m utai_train.runner --config <run.json>

Spawned by the Rust TrainingManager with cwd = <app>/training (so the package is
importable), UTF-8 env forced by util::python_command. stdout = JSONL protocol
(protocol.py), stderr = python logging/tracebacks (Rust keeps a ring buffer and
mirrors to the app log). CUDA_VISIBLE_DEVICES is set here from cfg["gpu"] BEFORE
torch is imported anywhere.
"""
import argparse
import json
import os
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to the run config json")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # device visibility MUST be set before torch is imported anywhere — the shim's
    # setup_visibility is pure os.environ (no torch), backend-aware (cuda/xpu), and
    # byte-identical to the old inline CUDA_VISIBLE_DEVICES for the cuda default.
    from .device import setup_visibility

    setup_visibility(cfg)

    from .protocol import Reporter
    from .stopfile import StopFlag, StopRequested

    reporter = Reporter()
    stop = StopFlag(cfg["stop_file"])
    rc = 0
    try:
        # S67 loud-degradation guard: the user asked for a GPU — a masked visibility
        # env / broken driver must FAIL the run with a mappable CODE, never silently
        # train on CPU. Also states the effective device once so the log always
        # answers "what did this run actually train on".
        from .device import require_wanted_accelerator, resolve_backend

        require_wanted_accelerator(cfg)
        print(
            "[device] backend=%s gpu=%r -> effective=%s"
            % (cfg.get("device_backend", "cuda"), cfg.get("gpu"), resolve_backend(cfg)),
            file=sys.stderr,
            flush=True,
        )
        backend = cfg.get("backend")
        if backend == "rvc":
            from .rvc import pipeline

            pipeline.run(cfg, reporter, stop)
        elif backend == "sovits":
            from .sovits import pipeline

            pipeline.run(cfg, reporter, stop)
        elif backend == "sovits_v2":
            from .sovits_v2 import pipeline

            pipeline.run(cfg, reporter, stop)
        elif backend == "sovits_diff":
            from .sovits import diff_pipeline

            diff_pipeline.run(cfg, reporter, stop)
        elif backend == "vocoder":
            from .vocoder import pipeline

            pipeline.run(cfg, reporter, stop)
        else:
            raise RuntimeError("未知训练后端: %s" % backend)
    except StopRequested:
        # stop observed during a preprocessing stage — nothing was trained
        reporter.done("stopped", {"phase": "preprocess"})
    except Exception as e:
        traceback.print_exc()
        reporter.error("%s: %s" % (type(e).__name__, e))
        rc = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    # DataLoader workers (persistent, spawn) are prone to hanging the interpreter
    # on Windows at exit — upstream solved this with os._exit(2333333); we exit
    # hard too, but only AFTER the protocol "done"/"error" line is flushed.
    os._exit(rc)


if __name__ == "__main__":
    main()
