"""SoVITS (so-vits-svc 4.1-Stable) training pipeline orchestration — the stages
upstream ran as separate scripts (slice by hand -> resample.py ->
preprocess_flist_config.py -> preprocess_hubert_f0.py -> train_index.py /
cluster/train_cluster.py -> train.py), driven from one run config. Stage order
deviation vs upstream: the retrieval/kmeans asset is built right after feature
extraction instead of after training — it only depends on the extracted
features, so it exists even if the user stops mid-training (same policy as the
RVC trainer).

Run config (JSON, written by the Rust TrainingManager) — required keys:
  backend "sovits", workspace, dataset_dir, model_slug, version "4.1|4.0",
  total_epoch, batch_size, stop_file, pretrain_g, pretrain_d,
  assets{ffmpeg, rmvpe_pt, contentvec_onnx, configs_dir}
optional: model_name (display name for the release config), seed(1234),
  fp16(false), vol_embedding(false), loudnorm(false), kmeans(false),
  save_every_steps(800), keep_ckpts(3), all_in_mem(false),
  gpu (handled by runner via CUDA_VISIBLE_DEVICES)

The version picks the ContentVec space: 4.1 -> vec768l12, 4.0 -> vec256l9
(4.0 = the same 4.1-Stable code with the vec256l9 encoder and default switches —
verified weight-isomorphic to old 4.0 checkpoints). version / vol_embedding /
sample rate are per-workspace immutables, guarded by the Rust run manifest.
"""
import glob
import logging
import os
import shutil

from ..cache import dataset_fingerprint, invalidate_extract_caches
from ..rvc.train_utils import get_logger  # shared harness helper (single source)
from . import utils
from .cluster import build_kmeans, build_retrieval
from .extract import extract_all
from .flist import build_flist_and_config
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
    import torch

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
    fp16 = bool(cfg.get("fp16", False)) and torch.cuda.is_available()
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
    build_flist_and_config(
        exp_dir,
        slug,
        dataset_44k,
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
        reporter,
    )

    stop.check()
    hps = utils.get_hparams_from_file(os.path.join(exp_dir, "config.json"))
    extract_all(
        dataset_44k,
        hps,
        assets["contentvec_onnx"],
        assets["rmvpe_pt"],
        "cuda" if torch.cuda.is_available() else "cpu",
        reporter,
        stop,
    )

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
