# load_audio ported from RVC 20240604 infer/lib/audio.py, minus the ffmpeg-python
# wrapper: same ffmpeg CLI invocation (decode -> mono f32le at target sr), but with
# an explicit ffmpeg executable path (the app bundles its own ffmpeg.exe) and
# list-argv (no shell, unicode-safe).
import subprocess
import sys

import numpy as np

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def load_audio(file, sr, ffmpeg="ffmpeg"):
    cmd = [
        ffmpeg,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(file),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-v",
        "error",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        creationflags=CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to load audio %s: %s"
            % (file, proc.stderr.decode("utf-8", errors="replace").strip())
        )
    return np.frombuffer(proc.stdout, np.float32).flatten()
