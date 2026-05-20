"""Training progress reporting protocol.

Communicates with the Rust sidecar manager via JSON lines on stdout.
Protocol:
  {"type": "state", "value": "preparing|preprocessing|training|generating_index|completed|stopped"}
  {"type": "epoch", "value": <int>}
  {"type": "loss", "value": <float>}
  {"type": "eta", "value": <int_seconds>}
"""

import json
import sys
import time


class ProgressReporter:
    def __init__(self):
        self._start_time = time.time()
        self._epoch_times: list[float] = []

    def report_state(self, state: str):
        self._emit("state", state)

    def report_epoch(self, epoch: int, total: int, loss: float | None = None):
        now = time.time()
        self._epoch_times.append(now)
        self._emit("epoch", epoch)

        if loss is not None:
            self._emit("loss", round(loss, 6))

        if len(self._epoch_times) >= 2 and epoch < total:
            avg_time = (self._epoch_times[-1] - self._epoch_times[0]) / (len(self._epoch_times) - 1)
            eta = int(avg_time * (total - epoch))
            self._emit("eta", eta)

    def report_completed(self):
        self._emit("state", "completed")

    def report_stopped(self):
        self._emit("state", "stopped")

    def _emit(self, msg_type: str, value):
        line = json.dumps({"type": msg_type, "value": value})
        print(line, flush=True)
