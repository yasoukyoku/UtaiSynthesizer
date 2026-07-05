# Vendored from RVC 20240604 infer/modules/train/extract/extract_f0_rmvpe.py.
# coarse_f0 quantization and output layout (2a_f0 coarse int / 2b-f0nsf Hz float,
# np.save onto the .wav name so files end in .wav.npy) are UNCHANGED. Deviations:
# sequential single-device loop instead of per-GPU process split (same outputs),
# explicit rmvpe.pt path, JSONL progress, stop flag between files.
# NB the original webui passes is_half as the STRING "True"/"False", which is always
# truthy — upstream RMVPE f0 extraction therefore effectively always ran half on
# NVIDIA. We default is_half=True on CUDA to match that actual behavior.
import logging
import os
import traceback

import numpy as np

from ..audio import load_audio

logger = logging.getLogger(__name__)


class FeatureInput(object):
    def __init__(self, rmvpe_pt, device, is_half, ffmpeg, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.rmvpe_pt = rmvpe_pt
        self.device = device
        self.is_half = is_half
        self.ffmpeg = ffmpeg

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path):
        x = load_audio(path, self.fs, self.ffmpeg)
        if not hasattr(self, "model_rmvpe"):
            from .rmvpe import RMVPE

            logger.info("Loading rmvpe model %s", self.rmvpe_pt)
            self.model_rmvpe = RMVPE(
                self.rmvpe_pt, is_half=self.is_half, device=self.device
            )
        return self.model_rmvpe.infer_from_audio(x, thred=0.03)

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse


def extract_f0(exp_dir, rmvpe_pt, device, is_half, ffmpeg, reporter, stop):
    inp_root = os.path.join(exp_dir, "1_16k_wavs")
    opt_root1 = os.path.join(exp_dir, "2a_f0")
    opt_root2 = os.path.join(exp_dir, "2b-f0nsf")
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    fi = FeatureInput(rmvpe_pt, device, is_half, ffmpeg)
    paths = []
    for name in sorted(os.listdir(inp_root)):
        if "spec" in name:
            continue
        paths.append(
            (
                os.path.join(inp_root, name),
                os.path.join(opt_root1, name),
                os.path.join(opt_root2, name),
            )
        )

    failed = 0
    for n, (inp_path, opt_path1, opt_path2) in enumerate(paths):
        stop.check()
        reporter.stage("f0", done=n, total=len(paths), message=os.path.basename(inp_path))
        try:
            if os.path.exists(opt_path1 + ".npy") and os.path.exists(
                opt_path2 + ".npy"
            ):
                continue
            featur_pit = fi.compute_f0(inp_path)
            np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
            coarse_pit = fi.coarse_f0(featur_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)  # ori
        except Exception:
            failed += 1
            logger.error("f0 failed for %s\n%s", inp_path, traceback.format_exc())
    reporter.stage("f0", done=len(paths), total=len(paths))
    if paths and failed == len(paths):
        raise RuntimeError("所有切片的 f0 提取均失败（详见日志）")
