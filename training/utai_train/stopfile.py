"""Graceful-stop flag. The Rust side creates the file (absolute path from the run
config); the sidecar polls it at safe boundaries (per training step / per file in
preprocessing). The file is NOT deleted by the sidecar — the Rust side owns its
lifecycle (cleans it up when the run ends), so a re-check after a save still sees it.
"""
import os


class StopRequested(Exception):
    """Raised at a safe boundary when the stop flag is observed during a
    preprocessing stage (the training loop handles stop inline instead)."""


class StopFlag:
    def __init__(self, path):
        self.path = path

    def requested(self):
        return os.path.exists(self.path)

    def check(self):
        if self.requested():
            raise StopRequested()
