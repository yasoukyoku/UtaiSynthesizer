"""SoVITS (so-vits-svc 4.1-Stable) training pipeline orchestration — the stages
upstream ran as separate scripts (slice by hand -> resample.py ->
preprocess_flist_config.py -> preprocess_hubert_f0.py -> train_index.py /
cluster/train_cluster.py -> train.py), driven from one run config. Stage order
deviations vs upstream:
  - retrieval/kmeans asset is built right after feature extraction instead of
    after training (early stop still leaves a usable index; RVC policy)
  - S41: stage order is slice -> augment -> config -> extract -> aug_check ->
    filelist -> index. The PSOLA augmentation writes extra slices; the f0
    quality gate (aug_check) consumes the .f0.npy products of BOTH source and
    aug slices (zero extra compute) and deletes rejected aug slices + all
    companions, so the filelists MUST be written after the gate — config.json
    is split out (extract needs hps from it) and written before extract.

Run config (JSON, written by the Rust TrainingManager) — required keys:
  backend "sovits", workspace, dataset_dir, model_slug, version "4.1|4.0",
  total_epoch, batch_size, stop_file, pretrain_g, pretrain_d,
  assets{ffmpeg, rmvpe_pt, contentvec_onnx, configs_dir}
optional: model_name (display name for the release config), seed(1234),
  fp16(false), vol_embedding(false), loudnorm(false), kmeans(false),
  save_every_steps(800), keep_ckpts(3), all_in_mem(false),
  aug_copies(0, S41 PSOLA augmentation copies per slice, 0-3),
  gpu (handled by runner via CUDA_VISIBLE_DEVICES)

The version picks the ContentVec space: 4.1 -> vec768l12, 4.0 -> vec256l9
(4.0 = the same 4.1-Stable code with the vec256l9 encoder and default switches —
verified weight-isomorphic to old 4.0 checkpoints). version / vol_embedding /
sample rate are per-workspace immutables, guarded by the Rust run manifest;
aug_copies is manifest-recorded and INHERITED by diffusion runs (shared
dataset_44k — a diff run regenerating the tree must re-augment identically).
"""
import glob
import logging
import os
import shutil

import numpy as np

from .. import device as device_shim
from ..augment import augment_slices, list_aug_entries, read_wav, run_f0_gate
from ..cache import dataset_fingerprint, invalidate_extract_caches
from ..rvc.train_utils import get_logger  # shared harness helper (single source)
from . import utils
from .cluster import build_kmeans, build_retrieval
from .extract import extract_all
from .flist import build_config, build_filelists
from .preprocess import slice_and_resample
from .train import train

logger = logging.getLogger(__name__)

VERSION_ENCODER = {"4.1": "vec768l12", "4.0": "vec256l9"}


def extract_cache_fp_text(dataset_dir, encoder, loudnorm):
    """THE cache-identity string for the dataset_44k tree — the sovits main
    pipeline and the diffusion pipeline share the workspace, so both MUST
    build this string identically or a diff run would silently wipe the main
    run's feature caches (and vice versa). Single source, do not inline."""
    return "%s|enc=%s|loudnorm=%d" % (
        dataset_fingerprint(dataset_dir), encoder, int(loudnorm)
    )


def run(cfg, reporter, stop):
    # backend = effective device (cuda|xpu|cpu), single source (shim). Byte-identical
    # to torch.cuda.is_available() on cpu/cuda; resolves to "xpu" on an Intel box.
    backend = device_shim.resolve_backend(cfg)

    exp_dir = cfg["workspace"]
    os.makedirs(exp_dir, exist_ok=True)
    get_logger(exp_dir)  # file log train.log (utf-8) in the run dir

    assets = cfg["assets"]
    version = cfg["version"]
    if version not in VERSION_ENCODER:
        raise RuntimeError("非法 SoVITS 版本: %s（可选 4.1/4.0）" % version)
    encoder = VERSION_ENCODER[version]
    slug = cfg["model_slug"]
    seed = int(cfg.get("seed", 1234))
    fp16 = bool(cfg.get("fp16", False)) and backend == "cuda"
    vol_embedding = bool(cfg.get("vol_embedding", False))
    loudnorm = bool(cfg.get("loudnorm", False))
    ffmpeg = assets["ffmpeg"]

    # the slice/extract products live together under dataset_44k — invalidate the
    # whole tree when the dataset OR any parameter that changes slice output /
    # feature space changes (loudnorm rewrites every wav; encoder switches the
    # .soft.pt dimension — version is manifest-immutable, belt and suspenders)
    fp_text = extract_cache_fp_text(cfg["dataset_dir"], encoder, loudnorm)
    invalidate_extract_caches(exp_dir, fp_text, ("dataset_44k",))

    dataset_44k = os.path.join(exp_dir, "dataset_44k")
    spk_dir = os.path.join(dataset_44k, slug)
    slice_and_resample(cfg["dataset_dir"], spk_dir, loudnorm, ffmpeg, reporter, stop)

    stop.check()
    aug_copies = int(cfg.get("aug_copies", 0))
    meta_dir = os.path.join(exp_dir, "aug_meta")
    augment_slices(
        spk_dir,
        aug_copies,
        seed,
        meta_dir,
        read_wav,
        _write_slice_int16,
        lambda stem: _remove_aug_products(spk_dir, meta_dir, stem),
        reporter,
        stop,
    )

    stop.check()
    build_config(
        exp_dir,
        slug,
        encoder,
        vol_embedding,
        fp16,
        int(cfg["total_epoch"]),
        int(cfg["batch_size"]),
        int(cfg.get("save_every_steps", 800)),
        int(cfg.get("keep_ckpts", 3)),
        bool(cfg.get("all_in_mem", False)),
        seed,
        assets["configs_dir"],
    )

    stop.check()
    hps = utils.get_hparams_from_file(os.path.join(exp_dir, "config.json"))
    failed_aug = extract_all(
        dataset_44k,
        hps,
        assets["contentvec_onnx"],
        assets["rmvpe_pt"],
        backend,  # "cuda"|"xpu"|"cpu" device for f0 predictor + mel extractor (sovits rmvpe is fp32 on every backend)
        reporter,
        stop,
    )
    # aug slices whose extraction failed have PARTIAL companion sets — remove
    # them before the gate/filelists (the gate would only catch missing f0)
    for filename in failed_aug or ():
        stem = os.path.basename(filename).split(".")[0]
        logger.warning("removing aug slice with failed extraction: %s", stem)
        _remove_aug_products(spk_dir, meta_dir, stem)

    stop.check()
    run_f0_gate(
        list_aug_entries(spk_dir, meta_dir),
        lambda stem: _load_gate_f0(spk_dir, stem),
        lambda stem: _remove_aug_products(spk_dir, meta_dir, stem),
        reporter,
        stop,
        report_path=os.path.join(exp_dir, "aug_gate_report.json"),
    )

    stop.check()
    build_filelists(exp_dir, slug, dataset_44k, seed, reporter)

    stop.check()
    if bool(cfg.get("kmeans", False)):
        display = cfg.get("model_name") or slug
        index_path, index_rows = build_kmeans(exp_dir, spk_dir, display, reporter, stop)
    else:
        index_path, index_rows = build_retrieval(exp_dir, spk_dir, seed, reporter, stop)

    stop.check()
    reporter.stage("train_prep", message="加载模型与数据，训练即将开始")
    _seed_base_checkpoints(exp_dir, cfg)
    summary = train(cfg, exp_dir, reporter, stop)

    if summary["final_weight"] is None and not summary["stopped"]:
        raise RuntimeError(
            "没有执行任何训练步：目标 epoch (%s) 不大于已训练进度，请增大总 epoch 后再续训"
            % cfg["total_epoch"]
        )

    summary["index"] = index_path
    summary["index_rows"] = index_rows
    reporter.done("stopped" if summary.pop("stopped") else "completed", summary)


def _write_slice_int16(tmp_path, samples, sr):
    """Aug slice writer — MUST match the base-slice disk format (int16 PCM;
    stdlib `wave` in flist.py rejects IEEE-float wavs, red-team F7)."""
    from scipy.io import wavfile

    pcm = (np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(tmp_path, sr, pcm)


def _remove_aug_products(spk_dir, meta_dir, aug_stem):
    """Delete an aug slice and EVERY companion product (first dot-segment match:
    .wav / .wav.soft.pt / .spec.pt / .wav.f0.npy / .wav.vol.npy / diff's
    .wav.mel.npy + aug pair) plus its meta json."""
    for name in os.listdir(spk_dir):
        if name.split(".")[0] == aug_stem:
            try:
                os.remove(os.path.join(spk_dir, name))
            except OSError:
                pass
    try:
        os.remove(os.path.join(meta_dir, aug_stem + ".json"))
    except OSError:
        pass


def _load_gate_f0(spk_dir, stem):
    """(f0_hz, voiced_mask) from the extraction product for the aug gate.
    .f0.npy = np object (f0, uv); f0 is INTERPOLATED through unvoiced spans,
    uv is float with 1.0 = voiced — the mask is mandatory (red-team F6)."""
    path = os.path.join(spk_dir, stem + ".wav.f0.npy")
    try:
        f0, uv = np.load(path, allow_pickle=True)
        f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
        uv = np.asarray(uv, dtype=np.float64).reshape(-1)
        n = min(len(f0), len(uv))
        return f0[:n], uv[:n] > 0.5
    except Exception:
        logger.warning("gate: unreadable f0 product for %s", stem)
        return None


def _seed_base_checkpoints(exp_dir, cfg):
    """Upstream's pretrain mechanism is literally 'put G_0.pth/D_0.pth into the
    log dir' — latest_checkpoint_path picks them up, global_step parses to 0,
    clean_checkpoints never deletes *_0.pth. Reproduce exactly: copy each base
    model in when its family (G_*/D_*) is absent. G and D are checked
    INDEPENDENTLY — a kill between the two copies must self-heal on the next
    run, not silently train a pretrained G against a random D."""
    for key, pattern, dst_name in (
        ("pretrain_g", "G_*.pth", "G_0.pth"),
        ("pretrain_d", "D_*.pth", "D_0.pth"),
    ):
        if glob.glob(os.path.join(exp_dir, pattern)):
            continue
        src = cfg.get(key, "") or ""
        if not src:
            raise RuntimeError("缺少底模路径: %s" % key)
        logger.info("seeding base checkpoint %s -> %s", src, dst_name)
        dst = os.path.join(exp_dir, dst_name)
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
