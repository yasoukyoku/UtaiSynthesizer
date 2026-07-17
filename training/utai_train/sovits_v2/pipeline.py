"""SoVITS 4.0-v2 (VISinger2) training pipeline orchestration — the v2 sibling
of sovits/pipeline.py, driving upstream's separate scripts (slice by hand ->
resample.py -> preprocess_flist_config.py -> preprocess_hubert_f0.py ->
cluster/train_cluster.py -> train.py) from one run config. Stage order and
policies mirror the 4.x pipeline exactly (slice -> augment -> config ->
extract -> aug_check -> filelist -> index -> train_prep -> train); shared
plumbing (slicer, PSOLA augment + f0 gate, cache fingerprint, cluster/retrieval
builders, filelist split, base-checkpoint seeding) is imported from the sovits
package — single source, do not copy.

Run config (JSON, written by the Rust TrainingManager) — required keys:
  backend "sovits_v2", workspace, dataset_dir, model_slug, version "4.0-v2",
  total_epoch, batch_size, stop_file, pretrain_g, pretrain_d,
  assets{ffmpeg, rmvpe_pt, contentvec_onnx, configs_dir}
optional: model_name, seed(1234), loudnorm(false), kmeans(false),
  save_every_steps(800), keep_ckpts(3), aug_copies(0), speakers[] (multi),
  gpu/device_backend (runner-level), skip_optimizer (gate-only, train.py hdr)

v2 differences vs the 4.x pipeline (all registered):
  - encoder is vec256l9 ONLY (v2 = the 4.0 ecosystem with a VISinger2 decoder)
  - slice trim uses the v2 branch's resample.py top_db=20 (4.x: 40)
  - extraction products: .soft.pt + RAW .f0.npy + .aam80.npy (see extract.py);
    f0 = RMVPE (house standard, S68 user decision; upstream trains on dio)
  - no vol_embedding / all_in_mem / fp16 (v2 has none of these; fp16_run is
    forced off in the config builder)
  - workspaces are family "sovits_v2" (Rust manifest) — NOT shared with 4.x
    (the diffusion pool lives in sovits-family workspaces; v2 diffusion attach
    is a possible later project)
"""
import logging
import os

from .. import device as device_shim
from ..augment import augment_slices, list_aug_entries, read_wav, run_f0_gate
from ..cache import invalidate_extract_caches
from ..rvc.train_utils import get_logger  # shared harness helper (single source)
from ..sovits.cluster import build_kmeans, build_retrieval
from ..sovits.flist import build_filelists, resolve_speakers
from ..sovits.pipeline import (
    _remove_aug_products,
    _seed_base_checkpoints,
    _speaker_meta_dir,
    _write_slice_int16,
    extract_cache_fp_text,
)
from ..sovits.preprocess import slice_and_resample
from . import utils
from .extract import extract_all
from .flist import build_config
from .train import train

import numpy as np

logger = logging.getLogger(__name__)

VERSION = "4.0-v2"
ENCODER = "vec256l9"
TRIM_TOP_DB = 20  # v2 branch resample.py value (4.x uses 40)


def run(cfg, reporter, stop):
    backend = device_shim.resolve_backend(cfg)

    exp_dir = cfg["workspace"]
    os.makedirs(exp_dir, exist_ok=True)
    get_logger(exp_dir)  # file log train.log (utf-8) in the run dir

    assets = cfg["assets"]
    version = cfg["version"]
    if version != VERSION:
        raise RuntimeError("非法 SoVITS 4.0-v2 版本: %s（仅支持 %s）" % (version, VERSION))
    seed = int(cfg.get("seed", 1234))
    loudnorm = bool(cfg.get("loudnorm", False))
    ffmpeg = assets["ffmpeg"]
    f0_method = cfg.get("f0_method", "rmvpe")  # "dio" = gate-only override

    # same cache-identity mechanism as the 4.x/diff pipelines (single source);
    # the non-default f0 method (gate workspaces) folds into the string so a
    # dio-extracted tree can never be mistaken for the rmvpe product tree
    speakers = resolve_speakers(cfg)
    is_multi = len(speakers) > 1
    fp_text = extract_cache_fp_text(speakers, ENCODER, loudnorm)
    if f0_method != "rmvpe":
        fp_text += "|f0=%s" % f0_method
    invalidate_extract_caches(exp_dir, fp_text, ("dataset_44k",))

    dataset_44k = os.path.join(exp_dir, "dataset_44k")
    aug_copies = int(cfg.get("aug_copies", 0))

    for sp in speakers:
        spk_dir = os.path.join(dataset_44k, sp["slug"])
        meta_dir = _speaker_meta_dir(exp_dir, is_multi, sp["slug"])
        slice_and_resample(
            sp["dataset_dir"], spk_dir, loudnorm, ffmpeg, reporter, stop,
            trim_top_db=TRIM_TOP_DB,
        )
        stop.check()
        augment_slices(
            spk_dir,
            aug_copies,
            seed,
            meta_dir,
            read_wav,
            _write_slice_int16,
            lambda stem, d=spk_dir, m=meta_dir: _remove_aug_products(d, m, stem),
            reporter,
            stop,
        )
        stop.check()

    build_config(
        exp_dir,
        speakers[0]["slug"],
        int(cfg["total_epoch"]),
        int(cfg["batch_size"]),
        int(cfg.get("save_every_steps", 800)),
        int(cfg.get("keep_ckpts", 3)),
        seed,
        assets["configs_dir"],
        speakers=speakers,
        num_workers=cfg.get("num_workers"),
    )

    stop.check()
    hps = utils.get_hparams_from_file(os.path.join(exp_dir, "config.json"))
    failed_aug = extract_all(
        dataset_44k,
        hps,
        assets["contentvec_onnx"],
        assets["rmvpe_pt"],
        backend,
        reporter,
        stop,
        f0_method=f0_method,
    )
    for filename in failed_aug or ():
        stem = os.path.basename(filename).split(".")[0]
        fslug = os.path.basename(os.path.dirname(filename))
        logger.warning("removing aug slice with failed extraction: %s/%s", fslug, stem)
        _remove_aug_products(
            os.path.join(dataset_44k, fslug),
            _speaker_meta_dir(exp_dir, is_multi, fslug),
            stem,
        )

    stop.check()
    for sp in speakers:
        spk_dir = os.path.join(dataset_44k, sp["slug"])
        meta_dir = _speaker_meta_dir(exp_dir, is_multi, sp["slug"])
        report = os.path.join(
            exp_dir,
            "aug_gate_report_%s.json" % sp["slug"] if is_multi else "aug_gate_report.json",
        )
        run_f0_gate(
            list_aug_entries(spk_dir, meta_dir),
            lambda stem, d=spk_dir: _load_gate_f0_v2(d, stem),
            lambda stem, d=spk_dir, m=meta_dir: _remove_aug_products(d, m, stem),
            reporter,
            stop,
            report_path=report,
        )

    stop.check()
    # min_dur 0.35 (4.x floor: 0.3): the v2 SingDataset drops mel<30-frame
    # (~0.337s) items at LOAD time — a slice in [0.300, 0.337) would pass the
    # 0.3 floor but yield None in every epoch, and an all-None batch (e.g. the
    # 2-slice val loader) crashes the collate. Registered deviation vs the
    # upstream 0.3s floor (upstream has the same latent crash window).
    build_filelists(
        exp_dir, speakers[0]["slug"], dataset_44k, seed, reporter,
        speakers=speakers, min_dur=0.35,
    )

    stop.check()
    if bool(cfg.get("kmeans", False)):
        named = [(sp["name"], os.path.join(dataset_44k, sp["slug"])) for sp in speakers]
        index_path, index_rows = build_kmeans(exp_dir, named, reporter, stop)
    else:
        index_path = None
        index_rows = 0
        for i, sp in enumerate(speakers):
            index_path, rows = build_retrieval(
                exp_dir,
                os.path.join(dataset_44k, sp["slug"]),
                seed,
                reporter,
                stop,
                spk_id=i,
            )
            index_rows += rows

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


def _load_gate_f0_v2(spk_dir, stem):
    """(f0_hz, voiced_mask) for the aug gate. The v2 product is a RAW f0 array
    (0 = unvoiced, no uv companion) — the mask derives directly from f0 > 0;
    the gate only reads both-voiced frames, so interpolation is irrelevant."""
    path = os.path.join(spk_dir, stem + ".wav.f0.npy")
    try:
        f0 = np.load(path, allow_pickle=True)
        f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
        return f0, f0 > 0
    except Exception:
        logger.warning("gate: unreadable f0 product for %s", stem)
        return None
