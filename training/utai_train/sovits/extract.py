# Ported from so-vits-svc 4.1-Stable preprocess_hubert_f0.py process_one().
# Per-file products (all skip-if-exists, same names/layout as upstream):
#   <wav>.soft.pt   ContentVec features, torch tensor [1, dim, T]
#   <wav>.f0.npy    np object array (f0, uv) from RMVPEF0Predictor (thr 0.05)
#   <stem>.spec.pt  linear spectrogram_torch 2048/512/2048 center=False, [1025, T]
#   <wav>.vol.npy   Volume_Extractor(hop) frame RMS — only when vol_embedding
# Deviations (deliberate):
#   - ContentVec via the project's ONNX extractors instead of fairseq
#     (aux/contentvec_768l12.onnx | contentvec_256l9.onnx — gate-verified against
#     real fairseq: max 7.7e-4 / cos 1-1e-9; kills the fairseq dependency and
#     guarantees training feature space == inference feature space), CPU EP like
#     the RVC trainer
#   - the RMVPE predictor is constructed ONCE per run (upstream re-loads the
#     180MB checkpoint per file inside process_one); math identical
#   - sequential over sorted files instead of shuffle + ProcessPoolExecutor
#     (per-file products are independent; upstream's spawn workers each loading
#     an encoder copy is a Windows minefield)
# --use_diff products (diff_mode=True, the shallow-diffusion trainer), same
# names/layout as upstream preprocess_hubert_f0.py:84-103:
#   <wav>.mel.npy      nsf_hifigan-recipe mel [T,128] (Vocoder.extract — the
#                      diffusion mel, NOT the 80-mel loss recipe / 2048 spec)
#   <wav>.aug_mel.npy  np object (aug_mel, keyshift): loudness shift
#                      uniform(-1, min(1, log10(1/max_amp))) + keyshift
#                      uniform(-5,5) re-extraction
#   <wav>.aug_vol.npy  Volume_Extractor of the SAME loudness-shifted audio
#   (and .vol.npy gates on `diff_mode or vol_embedding` — upstream :77)
# diff deviations (deliberate):
#   - (aug_mel, aug_vol) are ONE augmentation unit: if EITHER file is missing
#     BOTH are recomputed from a single fresh draw and rewritten (upstream
#     draws every run and checks each file independently — a kill between the
#     two writes would permanently pair a mel with a foreign volume, silently
#     poisoning the aug branch of every later training run)
#   - draws come from a per-run seeded random.Random (upstream: unseeded
#     global random); compute-if-missing instead of upstream's
#     recompute-but-never-overwrite (their recompute is pure waste)
import logging
import os
import traceback

import librosa
import numpy as np
import torch

from ..augment import is_aug_name
from .f0.RMVPEF0Predictor import RMVPEF0Predictor
from .modules.mel_processing import spectrogram_torch
from .utils import Volume_Extractor

logger = logging.getLogger(__name__)

# ContentVec conv frontend needs at least 400 samples @16k (S35 aux contract)
MIN_SAMPLES_16K = 400


def extract_all(
    dataset_44k_dir,
    hps,               # HParams from the workspace config.json
    contentvec_onnx,
    rmvpe_pt,
    device,            # "cuda" | "cpu" (for the f0 predictor / mel extractor)
    reporter,
    stop,
    diff_mode=False,   # also produce the --use_diff products (see header)
    nsf_hifigan_model=None,  # path to the nsf_hifigan torch ckpt (diff_mode only)
    aug_seed=1234,     # seed for the augmentation draws (diff_mode only)
):
    import onnxruntime as ort

    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length
    vol_embedding = bool(hps.model.vol_embedding)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(contentvec_onnx, so, providers=["CPUExecutionProvider"])

    f0_predictor = RMVPEF0Predictor(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        dtype=torch.float32,
        device=device,
        threshold=0.05,
        model_path=rmvpe_pt,
    )
    volume_extractor = Volume_Extractor(hop_length)

    mel_extractor = None
    aug_rng = None
    if diff_mode:
        if not nsf_hifigan_model:
            raise RuntimeError("扩散预处理缺少 NSF-HiFiGAN 声码器资产路径")
        import random

        from .diffusion.vocoder import Vocoder

        # constructed ONCE like upstream's main; extraction device follows the
        # run device (upstream passes the same device it extracts features on)
        mel_extractor = Vocoder("nsf-hifigan", nsf_hifigan_model, device=device)
        aug_rng = random.Random(aug_seed)

    filenames = []
    for spk in sorted(os.listdir(dataset_44k_dir)):
        spk_dir = os.path.join(dataset_44k_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for name in sorted(os.listdir(spk_dir)):
            if name.endswith(".wav"):
                filenames.append(os.path.join(spk_dir, name))

    # fail-fast on ANY base slice: the filelists reference them — a tolerated
    # failure would surface 800 steps later as a raw FileNotFoundError inside
    # a DataLoader worker (upstream preprocess aborts loudly too).
    # S41 deviation: _aug slices are OUR OWN generated products — a failure
    # there degrades to "reject this aug copy" (returned to the caller, which
    # removes it before the filelists are built) instead of killing the run.
    failed_aug = []
    for n, filename in enumerate(filenames):
        stop.check()
        reporter.stage(
            "extract", done=n, total=len(filenames), message=os.path.basename(filename)
        )
        try:
            _process_one(
                filename,
                sess,
                f0_predictor,
                volume_extractor,
                sampling_rate,
                hps,
                vol_embedding,
                diff_mode,
                mel_extractor,
                aug_rng,
                device,
            )
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
    # products live under skip-if-exists caches: a kill mid-write must not leave
    # a truncated file that every later run treats as a valid cache hit
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_np_save(arr, path):
    tmp = path + ".tmp.npy"  # np.save appends .npy to extension-less paths
    np.save(tmp, arr)
    os.replace(tmp, path)


def _process_one(filename, sess, f0_predictor, volume_extractor, sampling_rate, hps, vol_embedding,
                 diff_mode=False, mel_extractor=None, aug_rng=None, device="cpu"):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)

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
        f0, uv = f0_predictor.compute_f0_uv(wav)
        _atomic_np_save(np.asanyarray((f0, uv), dtype=object), f0_path)

    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(sr, hps.data.sampling_rate)
            )
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        _atomic_torch_save(spec, spec_path)

    # upstream preprocess_hubert_f0.py:77 — .vol.npy gates on `diff or vol_embedding`
    if diff_mode or vol_embedding:
        volume_path = filename + ".vol.npy"
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            _atomic_np_save(volume.to("cpu").numpy(), volume_path)

    if diff_mode:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path):
            mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().to("cpu").numpy()
            _atomic_np_save(mel, mel_path)

        # (aug_mel, aug_vol) = ONE augmentation unit sharing one draw — if
        # either is missing, recompute BOTH (see file header)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        if not (os.path.exists(aug_mel_path) and os.path.exists(aug_vol_path)):
            # upstream :92-98 verbatim math, draws in the same order
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = aug_rng.uniform(-1, max_shift)
            keyshift = aug_rng.uniform(-5, 5)
            aug_mel_t = mel_extractor.extract(
                audio_norm.to(device) * (10 ** log10_vol_shift), sampling_rate, keyshift=keyshift
            )
            aug_mel = aug_mel_t.squeeze().to("cpu").numpy()
            aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
            _atomic_np_save(np.asanyarray((aug_mel, keyshift), dtype=object), aug_mel_path)
            _atomic_np_save(aug_vol.to("cpu").numpy(), aug_vol_path)
