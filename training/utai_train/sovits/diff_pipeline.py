"""Shallow-diffusion (train_diff) pipeline — backend "sovits_diff".

Upstream flow (so-vits-svc 4.1-Stable): the SAME dataset/preprocessing as the
main model plus the --use_diff products, then `python train_diff.py -c
configs/diffusion.yaml`. We run it in the SAME workspace as the same-named
sovits main training on purpose: slice/soft/f0/spec/vol caches are shared
(fingerprint-guarded), a diff run on a prepared workspace only computes
mel/aug_mel/aug_vol incrementally.

Run config (JSON, written by the Rust TrainingManager) — required keys:
  backend "sovits_diff", workspace, dataset_dir, model_slug, version "4.1|4.0",
  total_steps, batch_size, save_every_steps (=interval_val),
  interval_force_save (Rust-normalized to a multiple of save_every_steps),
  k_step_max (0 = full diffusion), stop_file,
  assets{ffmpeg, rmvpe_pt, contentvec_onnx, configs_dir, nsf_hifigan_model,
         diffusion_pretrain?("" when absent -> train from scratch)}
optional: model_name (display name -> the yaml spk map / exported sidecar),
  seed(1234), fp16(false = amp fp32), cache_all_data(true),
  vol_embedding / loudnorm (inherited from the workspace manifest by the Rust
  side — a diff run must never flip them or it would wipe / desync the main
  model's caches), gpu (handled by runner via CUDA_VISIBLE_DEVICES)

Deviations vs upstream (deliberate):
  - completion is total_steps-based (yaml epochs stays the upstream 100000
    sentinel; diffusion epochs are tiny units — batch 48 x 2s on a normal
    dataset is single-digit batches per epoch, upstream itself thinks in steps)
  - an existing main config.json is NOT rewritten (only filelists are rebuilt,
    deterministically identical for the same slices+seed) — the main model's
    train section must not be clobbered with diffusion values
  - short-sample pre-check: train AND val filelists each need >= 1 sample
    longer than data.duration+0.1s, else AudioDataset.__getitem__'s skip
    recursion (data_loaders.py:212) would crash at the FIRST validation after
    2000 trained steps (val) or at the first batch (train)
  - base-model seeding normalizes ckpt['global_step'] to 0 (a community
    checkpoint with a nonzero step would instantly satisfy the total_steps
    completion check and skew the lr-decay resume math)
  - torch/random are seeded (upstream is unseeded; the loss-trajectory gate
    and reproducibility need it)
  - amp fp16 is forced off on CPU (torch CPU autocast has no fp16 path; same
    policy as the sovits trainer's CPU fallback)
"""
import json
import logging
import os
import random
import re
import shutil

import yaml

from ..cache import invalidate_extract_caches
from ..rvc.train_utils import get_logger  # shared harness helper (single source)
from .extract import extract_all
from .flist import ENCODER_DIMS, _wav_duration, build_filelists, build_flist_and_config
from .pipeline import VERSION_ENCODER, extract_cache_fp_text
from .preprocess import slice_and_resample
from . import utils as sovits_utils

logger = logging.getLogger(__name__)

# placeholder train-section values for a diff-first workspace's config.json
# (the sovits main pipeline rewrites them from its own request when/if a main
# run happens; extract only reads data.*/model.vol_embedding from this file)
_PLACEHOLDER_MAIN = dict(
    fp16=False, total_epoch=10000, batch_size=6, save_every_steps=800,
    keep_ckpts=3, all_in_mem=False,
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
    dim = ENCODER_DIMS[encoder]
    slug = cfg["model_slug"]
    seed = int(cfg.get("seed", 1234))
    loudnorm = bool(cfg.get("loudnorm", False))
    vol_embedding = bool(cfg.get("vol_embedding", False))
    ffmpeg = assets["ffmpeg"]

    # same cache identity as the sovits main pipeline (single source — a
    # format drift here would wipe the shared caches on every backend switch)
    fp_text = extract_cache_fp_text(cfg["dataset_dir"], encoder, loudnorm)
    invalidate_extract_caches(exp_dir, fp_text, ("dataset_44k",))

    dataset_44k = os.path.join(exp_dir, "dataset_44k")
    spk_dir = os.path.join(dataset_44k, slug)
    slice_and_resample(cfg["dataset_dir"], spk_dir, loudnorm, ffmpeg, reporter, stop)

    stop.check()
    config_path = os.path.join(exp_dir, "config.json")
    hps_probe = None
    if os.path.exists(config_path):
        try:
            hps_probe = sovits_utils.get_hparams_from_file(config_path)
        except Exception:
            # a corrupt config (pre-atomic-write era crash) must not brick
            # every diff run — fall through to a template rewrite; the main
            # pipeline rewrites it from its own request anyway
            logger.warning("workspace config.json unreadable — regenerating")
            hps_probe = None
    if hps_probe is not None:
        # main config present: rebuild filelists only (deterministic — same
        # slices + same seed give the same split the main run trained on)
        existing_encoder = hps_probe.model.speech_encoder
        if existing_encoder != encoder:
            raise RuntimeError(
                "工作区主模型的语音编码器 (%s) 与所选版本 %s (%s) 不一致——"
                "扩散模型必须与主模型同版本" % (existing_encoder, version, encoder)
            )
        build_filelists(exp_dir, slug, dataset_44k, seed, reporter)
    else:
        build_flist_and_config(
            exp_dir, slug, dataset_44k, encoder, vol_embedding,
            _PLACEHOLDER_MAIN["fp16"], _PLACEHOLDER_MAIN["total_epoch"],
            _PLACEHOLDER_MAIN["batch_size"], _PLACEHOLDER_MAIN["save_every_steps"],
            _PLACEHOLDER_MAIN["keep_ckpts"], _PLACEHOLDER_MAIN["all_in_mem"],
            seed, assets["configs_dir"], reporter,
        )

    stop.check()
    hps = sovits_utils.get_hparams_from_file(config_path)
    extract_all(
        dataset_44k,
        hps,
        assets["contentvec_onnx"],
        assets["rmvpe_pt"],
        "cuda" if torch.cuda.is_available() else "cpu",
        reporter,
        stop,
        diff_mode=True,
        nsf_hifigan_model=assets["nsf_hifigan_model"],
        aug_seed=seed,
    )

    stop.check()
    reporter.stage("diff_prep", message="准备扩散配置与底模")
    expdir = os.path.join(exp_dir, "diffusion")
    os.makedirs(expdir, exist_ok=True)
    duration = _write_diffusion_yaml(cfg, exp_dir, expdir, encoder, dim)
    _ensure_long_samples(exp_dir, duration)
    _seed_base_model(expdir, assets.get("diffusion_pretrain") or "", reporter)

    stop.check()
    reporter.stage("train_prep", message="加载扩散模型与数据，训练即将开始")
    summary = _train_diff(cfg, exp_dir, reporter, stop)

    if summary["steps_this_run"] == 0 and not summary["stopped"]:
        raise RuntimeError(
            "没有执行任何训练步：目标总步数 (%s) 不大于已训练进度 (%s)，"
            "请增大总步数后再续训" % (cfg["total_steps"], summary["steps"])
        )

    summary["expdir"] = expdir
    summary["config_yaml"] = os.path.join(expdir, "config.yaml").replace("\\", "/")
    # the frontend attach flow filters installed SoVITS models by ContentVec dim
    summary["encoder_dim"] = dim
    stopped = summary.pop("stopped")
    reporter.done("stopped" if stopped else "completed", summary)


def _p(path):
    return str(path).replace("\\", "/")


def _write_diffusion_yaml(cfg, exp_dir, expdir, encoder, dim):
    """configs_template/diffusion_template.yaml -> workspace/diffusion.yaml.
    Template values stay verbatim except the documented fills; Saver re-dumps
    the whole config as expdir/config.yaml at train start = the exact pair
    export_diffusion.py consumes. spk carries the DISPLAY name: with n_spk=1
    the data pipeline never consults the spk map (data_loaders.py:149), it
    only reaches the exported sidecar's speakers list. Returns data.duration
    (the short-sample pre-check needs it)."""
    import torch

    with open(
        os.path.join(cfg["assets"]["configs_dir"], "diffusion_template.yaml"),
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    display = cfg.get("model_name") or cfg["model_slug"]
    flist_dir = os.path.join(exp_dir, "filelists")
    use_cuda = torch.cuda.is_available()
    fp16 = bool(cfg.get("fp16", False)) and use_cuda  # CPU autocast has no fp16

    config["data"]["encoder"] = encoder
    config["data"]["encoder_out_channels"] = dim
    config["data"]["training_files"] = _p(os.path.join(flist_dir, "train.txt"))
    config["data"]["validation_files"] = _p(os.path.join(flist_dir, "val.txt"))
    config["model"]["n_spk"] = 1
    config["model"]["k_step_max"] = int(cfg.get("k_step_max", 0))
    config["spk"] = {display: 0}
    config["device"] = "cuda" if use_cuda else "cpu"
    config["vocoder"]["ckpt"] = _p(cfg["assets"]["nsf_hifigan_model"])
    config["env"]["expdir"] = _p(expdir)
    config["env"]["gpu_id"] = 0  # device selection is CUDA_VISIBLE_DEVICES (runner)
    config["train"]["batch_size"] = int(cfg["batch_size"])
    config["train"]["num_workers"] = 0  # cache_all_data + Windows spawn: workers
    #   would each pickle-copy the whole cached dataset for zero gain
    config["train"]["amp_dtype"] = "fp16" if fp16 else "fp32"
    config["train"]["cache_all_data"] = bool(cfg.get("cache_all_data", True))
    config["train"]["interval_val"] = int(cfg["save_every_steps"])
    config["train"]["interval_force_save"] = int(cfg["interval_force_save"])
    # train.epochs stays the upstream 100000 sentinel — completion is
    # total_steps-based (solver deviation); lr/decay/gamma/save_opt verbatim

    out = os.path.join(exp_dir, "diffusion.yaml")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    os.replace(tmp, out)
    return float(config["data"]["duration"])


def _read_flist(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_flist(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def _ensure_long_samples(exp_dir, duration):
    """data_loaders.__getitem__ skips samples shorter than duration+0.1s by
    recursing to the next index — a filelist with no long sample recurses
    forever (train: first batch; val: the FIRST validation, i.e. only after
    save_every_steps trained steps burn). The seeded split takes val = first
    2 shuffled slices, so a dataset with plenty of long material can still
    deal two short ones into val — a REPRODUCIBLE dead end (same seed, same
    split; review F8). Fix instead of refuse: swap the first long train slice
    against the shortest val slice. Diff-only and safe — filelists are
    regenerated by EVERY run's flist stage, so a later main run rebuilds its
    own split from the same seed untouched."""
    need = duration + 0.1
    flist_dir = os.path.join(exp_dir, "filelists")
    train_list = os.path.join(flist_dir, "train.txt")
    val_list = os.path.join(flist_dir, "val.txt")
    train = _read_flist(train_list)
    val = _read_flist(val_list)
    train_long = [p for p in train if _wav_duration(p) >= need]
    val_has_long = any(_wav_duration(p) >= need for p in val)

    if not train_long or (not val_has_long and len(train_long) < 2):
        raise RuntimeError(
            "时长 ≥ %.1f 秒的切片不足（扩散训练按 %.0f 秒窗口采样，训练集与验证集"
            "各需至少 1 个）。请提供更长的连续干声素材" % (need, duration)
        )
    if not val_has_long:
        promote = train_long[0]
        demote = min(val, key=_wav_duration)
        train[train.index(promote)] = demote
        val[val.index(demote)] = promote
        _write_flist(train_list, train)
        _write_flist(val_list, val)
        logger.info(
            "val had no >=%.1fs slice — swapped %s (val<-train) against %s",
            need, os.path.basename(promote), os.path.basename(demote),
        )


def _seed_base_model(expdir, pretrain_path, reporter):
    """Upstream's base-model mechanism is 'drop model_0.pt into the expdir'
    (load_model scans for the max numbered step). Seed it when no numbered
    checkpoint exists. global_step is normalized to 0 (deviation, see module
    header). An empty pretrain path = train from scratch (the 4.0/vec256
    ecosystem has no public diffusion base model) — loudly logged."""
    import torch

    for name in os.listdir(expdir):
        if re.fullmatch(r"model_(\d+)\.pt", name):
            return  # resume state present — never reseed
    if not pretrain_path:
        logger.warning("no diffusion base model — training from scratch")
        # force past the Reporter throttle — this notice follows the stage's
        # opening message within the throttle window and must not be swallowed
        reporter.stage("diff_prep", message="无扩散底模，将从零训练", force=True)
        return
    dst = os.path.join(expdir, "model_0.pt")
    tmp = dst + ".tmp"
    ckpt = torch.load(pretrain_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("扩散底模格式不符（缺少 'model' 键）: %s" % pretrain_path)
    if int(ckpt.get("global_step") or 0) == 0 and "optimizer" not in ckpt:
        shutil.copyfile(pretrain_path, tmp)
    else:
        logger.info(
            "normalizing base model global_step %s -> 0", ckpt.get("global_step")
        )
        torch.save({"global_step": 0, "model": ckpt["model"]}, tmp)
    os.replace(tmp, dst)
    logger.info("seeded diffusion base model %s -> model_0.pt", pretrain_path)


def _train_diff(cfg, exp_dir, reporter, stop):
    """Port of upstream train_diff.py __main__ (@ 730930d) — construction,
    resume-lr math (incl. the max(...,0) clamp: the base model's
    global_step=0 makes (0-2)//decay_step == -1, without the clamp every
    fresh run would START at 2x lr) and device moves are verbatim; the seeds
    at the top and the solver harness hooks are the registered deviations."""
    import torch
    from torch.optim import lr_scheduler

    from .diffusion import solver
    from .diffusion.data_loaders import get_data_loaders
    from .diffusion.logger import utils as du
    from .diffusion.unit2mel import Unit2Mel
    from .diffusion.vocoder import Vocoder

    seed = int(cfg.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)

    args = du.load_config(os.path.join(exp_dir, "diffusion.yaml"))
    logger.info(" > exp: %s", args.env.expdir)

    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)

    # load model
    model = Unit2Mel(
                args.data.encoder_out_channels,
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max
                )

    logger.info(' > Now model timesteps is %s, and k_step_max is %s',
                model.timesteps, model.k_step_max)

    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = du.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * (args.train.gamma ** max(((initial_global_step-2)//args.train.decay_step),0) )
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma,last_epoch=initial_global_step-2)

    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)

    # datas — the cache-loading loop doubles as the train_prep progress bar
    # and polls the stop flag (a stop during a minutes-long cache load must
    # not wait for the training loop to notice)
    def progress(done, total):
        stop.check()
        reporter.stage("train_prep", done=done, total=total, message="缓存训练数据")

    loader_train, loader_valid = get_data_loaders(args, whole_audio=False, progress=progress)

    # run
    return solver.train(
        args, initial_global_step, model, optimizer, scheduler, vocoder,
        loader_train, loader_valid,
        reporter=reporter, stop=stop,
        total_steps=int(cfg["total_steps"]), best_state=True,
    )
