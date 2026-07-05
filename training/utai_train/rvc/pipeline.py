"""RVC training pipeline orchestration: the six stages the original webui ran as
separate buttons (preprocess -> f0 -> feature -> index -> filelist/config ->
train), driven from one run config. Stage order deviation vs upstream: the
retrieval matrix ("index") is built right after feature extraction instead of
after training — it only depends on the extracted features, so it exists even if
the user stops mid-training (the "index gen on stop" requirement, solved by
construction).

Run config (JSON, written by the Rust TrainingManager) — required keys:
  backend "rvc", workspace, dataset_dir, model_slug, sample_rate "32k|40k|48k",
  version "v1|v2", total_epoch, batch_size, stop_file,
  assets{ffmpeg, rmvpe_pt, contentvec_onnx, configs_dir, mute_dir}
optional: spk_id(0), seed(1234), per(3.7), fp16(true), save_every_epoch(5),
  save_every_weights(true), keep_only_latest(true), cache_gpu(false),
  pretrain_g(""), pretrain_d(""), gpu (handled by runner via CUDA_VISIBLE_DEVICES)
"""
import hashlib
import logging
import os
import shutil

from . import train_utils as utils
from .extract_f0 import extract_f0
from .extract_feature import extract_features
from .filelist import build_filelist_and_config
from .index_npy import build_index
from .preprocess import preprocess_trainset
from .train import train

logger = logging.getLogger(__name__)

SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def _dataset_fingerprint(dataset_dir):
    """Content identity of the imported dataset (name + size + head/tail sample).
    The f0/feature caches are keyed by SLICE FILE NAME — after a dataset change the
    re-sliced wavs reuse the same names with different content, so stale cache
    entries would silently mismatch. Fingerprint change → wipe those caches."""
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(os.listdir(dataset_dir)):
        p = os.path.join(dataset_dir, name)
        st = os.stat(p)
        h.update(name.encode("utf-8"))
        h.update(str(st.st_size).encode())
        with open(p, "rb") as f:
            h.update(f.read(65536))
            if st.st_size > 131072:
                f.seek(-65536, 2)
                h.update(f.read(65536))
    return h.hexdigest()


def _invalidate_extract_caches(exp_dir, fingerprint):
    fp_file = os.path.join(exp_dir, "dataset.fingerprint")
    old = None
    if os.path.exists(fp_file):
        with open(fp_file, encoding="utf-8") as f:
            old = f.read().strip()
    if old != fingerprint:
        if old is not None:
            logger.info("dataset changed — clearing stale f0/feature caches")
        for sub in ("2a_f0", "2b-f0nsf", "3_feature256", "3_feature768"):
            d = os.path.join(exp_dir, sub)
            if os.path.isdir(d):
                shutil.rmtree(d)
    with open(fp_file, "w", encoding="utf-8") as f:
        f.write(fingerprint)


def run(cfg, reporter, stop):
    import torch

    exp_dir = cfg["workspace"]
    os.makedirs(exp_dir, exist_ok=True)
    utils.get_logger(exp_dir)  # file log train.log (utf-8) in the run dir

    assets = cfg["assets"]
    sr_str = cfg["sample_rate"]
    if sr_str not in SR_MAP:
        raise RuntimeError("非法采样率: %s（可选 32k/40k/48k）" % sr_str)
    version = cfg["version"]
    seed = int(cfg.get("seed", 1234))
    fp16 = bool(cfg.get("fp16", True)) and torch.cuda.is_available()
    ffmpeg = assets["ffmpeg"]

    _invalidate_extract_caches(exp_dir, _dataset_fingerprint(cfg["dataset_dir"]))

    preprocess_trainset(
        cfg["dataset_dir"],
        SR_MAP[sr_str],
        exp_dir,
        float(cfg.get("per", 3.7)),
        ffmpeg,
        reporter,
        stop,
    )

    use_cuda = torch.cuda.is_available()
    extract_f0(
        exp_dir,
        assets["rmvpe_pt"],
        "cuda" if use_cuda else "cpu",
        use_cuda,  # matches upstream's de-facto always-half-on-NVIDIA behavior
        ffmpeg,
        reporter,
        stop,
    )

    extract_features(exp_dir, version, assets["contentvec_onnx"], reporter, stop)

    stop.check()
    index_path, index_rows = build_index(exp_dir, version, seed, reporter)

    stop.check()
    build_filelist_and_config(
        exp_dir,
        sr_str,
        version,
        int(cfg.get("spk_id", 0)),
        assets["configs_dir"],
        assets["mute_dir"],
        seed,
        fp16,
        reporter,
    )

    stop.check()
    reporter.stage("train_prep", message="加载模型与数据，训练即将开始")
    summary = train(cfg, exp_dir, reporter, stop)

    if summary["final_weight"] is None and not summary["stopped"]:
        raise RuntimeError(
            "没有执行任何训练步：目标 epoch (%s) 不大于已训练进度，请增大总 epoch 后再续训"
            % cfg["total_epoch"]
        )

    summary["index"] = index_path
    summary["index_rows"] = index_rows
    reporter.done("stopped" if summary.pop("stopped") else "completed", summary)
