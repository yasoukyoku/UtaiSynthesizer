# Vendored from RVC 20240604 infer/modules/train/preprocess.py.
# The per-file math is UNCHANGED (slicer params, 48Hz 5th-order Butterworth lfilter,
# 3.7s/0.3s window split, peak>2.5 reject, 0.9/0.75 mixed peak normalization, dual
# gt-sr + 16k output). Deviations: sequential instead of multiprocessing (identical
# outputs — files were independent), explicit paths/ffmpeg, JSONL progress, stop flag
# between files, errors go to stderr logging instead of a cwd-relative log file.
import logging
import os
import traceback

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

from ..audio import load_audio
from .slicer2 import Slicer

logger = logging.getLogger(__name__)


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7, ffmpeg="ffmpeg"):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.ffmpeg = ffmpeg
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs")
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            logger.info("%s-%s-%s-filtered", idx0, idx1, tmp_max)
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            os.path.join(self.gt_wavs_dir, "%s_%s.wav" % (idx0, idx1)),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(
            os.path.join(self.wavs16k_dir, "%s_%s.wav" % (idx0, idx1)),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        audio = load_audio(path, self.sr, self.ffmpeg)
        # zero phased digital filter cause pre-ringing noise...
        audio = signal.lfilter(self.bh, self.ah, audio)

        idx1 = 0
        for audio in self.slicer.slice(audio):
            i = 0
            while 1:
                start = int(self.sr * (self.per - self.overlap) * i)
                i += 1
                if len(audio[start:]) > self.tail * self.sr:
                    tmp_audio = audio[start : start + int(self.per * self.sr)]
                    self.norm_write(tmp_audio, idx0, idx1)
                    idx1 += 1
                else:
                    tmp_audio = audio[start:]
                    idx1 += 1
                    break
            self.norm_write(tmp_audio, idx0, idx1)


def preprocess_trainset(inp_root, sr, exp_dir, per, ffmpeg, reporter, stop):
    # Rebuild the slice dirs from scratch: slicing is deterministic, and stale
    # slices from a previous (different) dataset would otherwise pollute the
    # filelist (the 4-dir intersection then excludes stale f0/feature entries).
    for sub in ("0_gt_wavs", "1_16k_wavs"):
        d = os.path.join(exp_dir, sub)
        if os.path.isdir(d):
            for name in os.listdir(d):
                os.remove(os.path.join(d, name))

    pp = PreProcess(sr, exp_dir, per, ffmpeg)
    infos = [
        (os.path.join(inp_root, name), idx)
        for idx, name in enumerate(sorted(os.listdir(inp_root)))
    ]
    failed = 0
    for n, (path, idx0) in enumerate(infos):
        stop.check()
        reporter.stage("slice", done=n, total=len(infos), message=os.path.basename(path))
        try:
            pp.pipeline(path, idx0)
        except Exception:
            failed += 1
            logger.error("preprocess failed for %s\n%s", path, traceback.format_exc())
    reporter.stage("slice", done=len(infos), total=len(infos))
    if infos and failed == len(infos):
        raise RuntimeError("所有音频文件预处理均失败（详见日志）")
