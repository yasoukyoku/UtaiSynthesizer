"""Graceful stop signal handling.

The Rust sidecar manager writes a .stop_{pid} file to signal the training
process to stop. This module checks for that signal and returns True when
the process should begin its graceful shutdown sequence (save checkpoint,
generate index, then exit).
"""

import os
from pathlib import Path


_STOP_DIR = Path("training")


def should_stop() -> bool:
    pid = os.getpid()
    stop_file = _STOP_DIR / f".stop_{pid}"
    if stop_file.exists():
        stop_file.unlink(missing_ok=True)
        return True
    return False


def cleanup():
    """Remove any lingering stop files for this process."""
    pid = os.getpid()
    stop_file = _STOP_DIR / f".stop_{pid}"
    stop_file.unlink(missing_ok=True)
