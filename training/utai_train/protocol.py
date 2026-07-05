"""stdout JSONL protocol between the training sidecar and the Rust TrainingManager.

stdout is owned EXCLUSIVELY by this module — nothing else in the sidecar may print
to stdout (python logging goes to stderr + the run-dir file log). One JSON object
per line, ensure_ascii=False, flushed immediately.

Message shapes (protocol v2):
  {"type":"stage","stage":str,"done":int?,"total":int?,"progress":float?,"message":str?}
  {"type":"step","step":int,"total_steps":int,"epoch":int,"total_epochs":int,
   "lr":float,"losses":{name:float,...},"eta_secs":int?}
  {"type":"ckpt","kind":"periodic|best|final|stop","path":str,"step":int,"epoch":int,
   "metric":float?}
  {"type":"done","reason":"completed|stopped","summary":{...}}
  {"type":"error","message":str}
"""
import json
import math
import sys
import time
from collections import deque


def _clean(obj):
    """json.dumps would emit bare NaN/Infinity (invalid JSON — the Rust side drops
    the line, poisoning step/ckpt/done). Replace non-finite floats with null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


class Reporter:
    def __init__(self, throttle_secs=0.4):
        # the pipe encoding follows the console codepage (e.g. cp932/cp936) when
        # PYTHONUTF8 isn't set — the protocol is UTF-8 by definition, force it
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass
        self.throttle_secs = throttle_secs
        self._last_emit = {}  # throttle key -> wall time
        self._rate_window = deque(maxlen=50)  # (wall_time, step)

    def _emit(self, obj):
        sys.stdout.write(json.dumps(_clean(obj), ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _throttled(self, key, force):
        now = time.monotonic()
        if not force and now - self._last_emit.get(key, 0.0) < self.throttle_secs:
            return True
        self._last_emit[key] = now
        return False

    def stage(self, stage, done=None, total=None, message=None):
        # per-item calls inside a stage are throttled; the final item always emits
        force = done is not None and total is not None and done >= total
        if self._throttled("stage:" + stage, force):
            return
        obj = {"type": "stage", "stage": stage}
        if done is not None:
            obj["done"] = done
        if total is not None:
            obj["total"] = total
            if total > 0 and done is not None:
                obj["progress"] = round(done / total, 4)
        if message is not None:
            obj["message"] = message
        self._emit(obj)

    def step(self, step, total_steps, epoch, total_epochs, lr, losses, force=False):
        self._rate_window.append((time.monotonic(), step))
        if self._throttled("step", force):
            return
        obj = {
            "type": "step",
            "step": step,
            "total_steps": total_steps,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "lr": lr,
            "losses": {k: round(float(v), 6) for k, v in losses.items()},
        }
        if len(self._rate_window) >= 2:
            (t0, s0), (t1, s1) = self._rate_window[0], self._rate_window[-1]
            if t1 > t0 and s1 > s0:
                rate = (s1 - s0) / (t1 - t0)
                obj["eta_secs"] = int(max(0, total_steps - step) / rate)
        self._emit(obj)

    def ckpt(self, kind, path, step, epoch, metric=None):
        obj = {
            "type": "ckpt",
            "kind": kind,
            "path": str(path),
            "step": step,
            "epoch": epoch,
        }
        if metric is not None:
            obj["metric"] = round(float(metric), 6)
        self._emit(obj)

    def done(self, reason, summary=None):
        self._emit({"type": "done", "reason": reason, "summary": summary or {}})

    def error(self, message):
        self._emit({"type": "error", "message": str(message)})
