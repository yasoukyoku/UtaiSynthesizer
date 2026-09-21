# SoVITS 4.0-v2 feature extraction — the v2 analog of sovits/extract.py,
# ported from so-vits-svc 4.0-v2 preprocess_hubert_f0.py process_one() plus the
# aam mel that upstream generated LAZILY inside data_utils.SingDataset.
# Per-file products (all skip-if-exists):
#   <wav>.soft.pt    ContentVec vec256l9 features, torch tensor [1, 256, T]
#                    (same name/layout as upstream)
#   <wav>.f0.npy     RAW f0 array (Hz, 0 = unvoiced) — v2's dataset does its own
#                    interpolate_f0 at load time, so unlike the 4.x product this
#                    is NOT an (f0, uv) object pair
#   <wav>.aam80.npy  the aam mel [T, 80] float32 (modules/audio.melspectrogram
#                    recipe) — upstream computed this lazily as `<wav>.mel.npy`
#                    inside the dataset; we pre-compute it as an explicit stage
#                    under a v2-owned name (the diffusion pipeline's nsf 128-mel
#                    already claims `.mel.npy`), byte-identical output
# Deviations (deliberate):
#   - ContentVec via the project's ONNX extractor instead of fairseq
#     (aux contentvec_256l9.onnx — gate-verified against real fairseq), CPU EP
#   - f0 defaults to RMVPE (house standard, S68 user decision; upstream trains
#     on dio) with the raw array reconstructed as f0*voiced so the on-disk
#     format matches upstream's; f0_method="dio" runs the vendored
#     utils.compute_f0_dio verbatim — the preprocessing gate uses it to prove
#     the port against upstream byte-for-byte
#   - RMVPE predictor constructed ONCE per run; sequential over sorted files
#     (same rationale as sovits/extract.py)
import logging
import os
import traceback

import librosa
import numpy as np
import torch

from ..augment import is_aug_name
from ..sovits.f0.RMVPEF0Predictor import RMVPEF0Predictor
from .modules import audio
from . import utils

logger = logging.getLogger(__name__)

# ContentVec conv frontend needs at least 400 samples @16k (S35 aux contract)
MIN_SAMPLES_16K = 400


def extract_all(
    dataset_44k_dir,
    hps,               # HParams from the workspace config.json
    contentvec_onnx,
    rmvpe_pt,
    device,            # "cuda" | "xpu" | "cpu" (for the f0 predictor)
    reporter,
    stop,
    f0_method="rmvpe",  # "rmvpe" (product default) | "dio" (upstream/gate)
):
    import onnxruntime as ort

    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(contentvec_onnx, so, providers=["CPUExecutionProvider"])

    f0_predictor = None
    if f0_method == "rmvpe":
        f0_predictor = RMVPEF0Predictor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            dtype=torch.float32,
            device=device,
            threshold=0.05,
            model_path=rmvpe_pt,
        )
    elif f0_method != "dio":
        raise RuntimeError("未知 f0 提取器: %s" % f0_method)

    filenames = []
    for spk in sorted(os.listdir(dataset_44k_dir)):
        spk_dir = os.path.join(dataset_44k_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for name in sorted(os.listdir(spk_dir)):
            if name.endswith(".wav"):
                filenames.append(os.path.join(spk_dir, name))

    # fail-fast on ANY base slice; _aug failures degrade to "reject this copy"
    # (same policy + rationale as sovits/extract.py)
    failed_aug = []
    for n, filename in enumerate(filenames):
        stop.check()
        reporter.stage(
            "extract", done=n, total=len(filenames), message=os.path.basename(filename)
        )
        try:
            _process_one(filename, sess, f0_predictor, sampling_rate, hop_length, hps, f0_method)
        except Exception:
            logger.error("extract failed for %s\n%s", filename, traceback.format_exc())
            if is_aug_name(filename):
                failed_aug.append(filename)
                continue
            raise RuntimeError(
                "切片 %s 特征提取失败（详见日志）" % os.path.basename(filename)
            )
    reporter.stage("extract", done=len(filenames), total=len(filenames))
    return failed_aug


def _atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_np_save(arr, path):
    tmp = path + ".tmp.npy"  # np.save appends .npy to extension-less paths
    np.save(tmp, arr)
    os.replace(tmp, path)


def _process_one(filename, sess, f0_predictor, sampling_rate, hop_length, hps, f0_method):
    wav, sr = librosa.load(filename, sr=sampling_rate)

    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        if len(wav16k) < MIN_SAMPLES_16K:
            raise RuntimeError("切片过短（<400 采样点 @16k），无法提取特征")
        feats = sess.run(
            ["features"], {"waveform": wav16k.astype(np.float32)[None, :]}
        )[0][0]  # [T, dim]
        if np.isnan(feats).sum() > 0:
            raise RuntimeError("ContentVec 特征包含 NaN")
        # upstream layout: [1, dim, T] cpu tensor
        c = torch.from_numpy(np.ascontiguousarray(feats.T))[None, :, :].float()
        _atomic_torch_save(c, soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        if f0_method == "dio":
            # upstream preprocess_hubert_f0.py verbatim: raw dio+stonemask
            f0 = utils.compute_f0_dio(
                wav, sampling_rate=sampling_rate, hop_length=hop_length
            )
        else:
            # RMVPE gives (interpolated f0, voiced mask); the v2 dataset expects
            # the RAW array (0 in unvoiced frames) and re-interpolates at load
            # time — masking restores exactly the pre-interpolation values
            f0_interp, uv = f0_predictor.compute_f0_uv(wav)
            f0 = np.where(np.asarray(uv) > 0.5, np.asarray(f0_interp), 0.0)
        _atomic_np_save(np.asarray(f0), f0_path)

    mel_path = filename + ".aam80.npy"
    if not os.path.exists(mel_path):
        # data_utils.SingDataset lazy-gen verbatim: same wav decode (librosa
        # int16->f32), same recipe, same dtype/transpose
        mel = audio.melspectrogram(wav, hps.data).astype(np.float32).T
        _atomic_np_save(mel, mel_path)
