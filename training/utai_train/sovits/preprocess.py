# SoVITS dataset preparation: slicing + the resample.py chain.
#
# Upstream so-vits-svc 4.1 ships NO slicer — its README requires the user to
# pre-slice to 5-15s with the openvpi audio-slicer and then runs resample.py on
# the slices. We fold both into one stage:
#   decode (librosa.load sr=None, same loader as upstream resample.py; ffmpeg
#   fallback for formats libsndfile can't read) -> Slicer (the rvc/slicer2.py
#   vendored openvpi slicer-v2, DEFAULT parameters = the openvpi defaults the
#   so-vits README points at) -> per slice, verbatim resample.py math:
#   trim(top_db=40) -> peak>1 ? 0.98*wav/peak : wav -> librosa.resample(->44100)
#   -> optional loudnorm (wav /= max|wav|; upstream default ON, ours default OFF
#   — upstream's own README flags it as lossy) -> int16 write.
# Deviations vs upstream resample.py: sequential instead of ProcessPoolExecutor
# (identical per-file math), guards for empty/silent slices (upstream would
# divide by zero on an all-zero slice), JSONL progress + stop flag.
import logging
import os
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

from ..audio import load_audio
from ..augment import is_aug_name
from ..rvc.slicer2 import Slicer  # single source — the vendored openvpi slicer

logger = logging.getLogger(__name__)

TARGET_SR = 44100


def _decode(path, ffmpeg):
    """librosa.load(sr=None) like upstream; ffmpeg fallback for exotic formats."""
    try:
        wav, sr = librosa.load(path, sr=None, mono=True)
        return wav.astype(np.float32), int(sr)
    except Exception:
        logger.warning("librosa/libsndfile could not decode %s, using ffmpeg", path)
        return load_audio(path, TARGET_SR, ffmpeg), TARGET_SR


def _resample_chain(slice_wav, sr, loudnorm, trim_top_db=40):
    """resample.py process() per-file math, verbatim order. trim_top_db: 40 =
    the 4.x resample.py value (default, byte-identical to pre-v2 callers);
    the 4.0-v2 branch's resample.py uses 20 (its pipeline passes it)."""
    wav, _ = librosa.effects.trim(slice_wav, top_db=trim_top_db)
    if wav.size == 0:
        return None
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = 0.98 * wav / peak
    wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    if loudnorm:
        m = np.max(np.abs(wav))
        if m > 0:  # guard: upstream divides unconditionally (NaN on silence)
            wav = wav / m
    if wav.size == 0:
        return None
    return (wav * np.iinfo(np.int16).max).astype(np.int16)


def slice_and_resample(dataset_dir, out_spk_dir, loudnorm, ffmpeg, reporter, stop, trim_top_db=40):
    """dataset_dir (imported originals) -> out_spk_dir/<idx0>_<idx1>.wav @44100 i16.
    Existing *.wav in out_spk_dir are rebuilt every run (slicing is deterministic;
    stale slices from a previous dataset would pollute the filelists) — companion
    extraction caches (*.soft.pt/...) are kept and invalidated separately by the
    dataset fingerprint."""
    os.makedirs(out_spk_dir, exist_ok=True)
    for name in os.listdir(out_spk_dir):
        # S41: _aug slices are a cross-run skip-if-exists cache owned by the
        # augment stage (which prunes stale ones itself); base slices are
        # rebuilt every run as before
        if name.endswith(".wav") and not is_aug_name(name):
            os.remove(os.path.join(out_spk_dir, name))

    names = sorted(os.listdir(dataset_dir))
    failed = 0
    written = 0
    for n, name in enumerate(names):
        stop.check()
        reporter.stage("slice", done=n, total=len(names), message=name)
        path = os.path.join(dataset_dir, name)
        try:
            wav, sr = _decode(path, ffmpeg)
            slicer = Slicer(sr=sr)  # openpvi defaults: -40dB/5000ms/300ms/20ms/5000ms
            for idx1, chunk in enumerate(slicer.slice(wav)):
                out = _resample_chain(chunk, sr, loudnorm, trim_top_db)
                if out is None:
                    continue
                wavfile.write(
                    os.path.join(out_spk_dir, "%03d_%03d.wav" % (n, idx1)),
                    TARGET_SR,
                    out,
                )
                written += 1
        except Exception:
            failed += 1
            logger.error("slice failed for %s\n%s", path, traceback.format_exc())
    reporter.stage("slice", done=len(names), total=len(names))
    if names and failed == len(names):
        raise RuntimeError("所有音频文件切片/重采样均失败（详见日志）")
    if written == 0:
        raise RuntimeError("切片后没有任何有效样本（素材可能全为静音或过短）")
    return written
