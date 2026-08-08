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
import collections
import json
import logging
import os
import random
import re
import shutil

import yaml

from .. import resume_state
from ..augment import (
    augment_slices,
    is_aug_name,
    list_aug_entries,
    read_wav,
    run_f0_gate,
)
from .. import device as device_shim
from ..cache import invalidate_extract_caches
from ..rvc.train_utils import get_logger  # shared harness helper (single source)
from .extract import extract_all
from .flist import ENCODER_DIMS, _wav_duration, build_config, build_filelists, resolve_speakers
from .pipeline import (
    VERSION_ENCODER,
    _load_gate_f0,
    _remove_aug_products,
    _write_slice_int16,
    extract_cache_fp_text,
)
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
    dim = ENCODER_DIMS[encoder]
    slug = cfg["model_slug"]
    seed = int(cfg.get("seed", 1234))
    loudnorm = bool(cfg.get("loudnorm", False))
    vol_embedding = bool(cfg.get("vol_embedding", False))
    ffmpeg = assets["ffmpeg"]

    # ①c: shallow diffusion does not yet support multi-speaker models — the
    # diffusion yaml carries ONE spk map that doubles as loader keys (need dir
    # slugs) AND the exported sidecar names (want display names); the main
    # model's dual-config split isn't wired here. Refuse BEFORE any cache/slice
    # op so a multi-speaker workspace's shared caches are never touched.
    if len(cfg.get("speakers") or []) > 1:
        raise RuntimeError("浅扩散暂不支持多说话人模型（请先训练单说话人扩散）")
    _existing_cfg = os.path.join(exp_dir, "config.json")
    if os.path.exists(_existing_cfg):
        try:
            _probe_n = int(getattr(
                sovits_utils.get_hparams_from_file(_existing_cfg).model, "n_speakers", 1
            ))
        except Exception:
            _probe_n = 1
        if _probe_n > 1:
            raise RuntimeError("浅扩散暂不支持多说话人模型（该工作区主模型为多说话人）")

    # same cache identity as the sovits main pipeline (single source — a
    # format drift here would wipe the shared caches on every backend switch).
    # resolve_speakers is single-speaker here (multi is refused above) so this
    # is byte-identical to the pre-①c fingerprint.
    fp_text = extract_cache_fp_text(resolve_speakers(cfg), encoder, loudnorm)
    invalidate_extract_caches(exp_dir, fp_text, ("dataset_44k",))

    dataset_44k = os.path.join(exp_dir, "dataset_44k")
    spk_dir = os.path.join(dataset_44k, slug)
    slice_and_resample(cfg["dataset_dir"], spk_dir, loudnorm, ffmpeg, reporter, stop)

    # S41: the diff run runs the SAME augment stage with the manifest-inherited
    # copies (Rust writes the effective value into run.json) — a cache-wipe
    # path (dataset change) rebuilds dataset_44k here, and without this the
    # workspace would claim "augmented" (manifest) while holding zero aug
    # slices (red-team R2/A5)
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
        existing_encoder = hps_probe.model.speech_encoder
        if existing_encoder != encoder:
            raise RuntimeError(
                "工作区主模型的语音编码器 (%s) 与所选版本 %s (%s) 不一致——"
                "扩散模型必须与主模型同版本" % (existing_encoder, version, encoder)
            )
    else:
        build_config(
            exp_dir, slug, encoder, vol_embedding,
            _PLACEHOLDER_MAIN["fp16"], _PLACEHOLDER_MAIN["total_epoch"],
            _PLACEHOLDER_MAIN["batch_size"], _PLACEHOLDER_MAIN["save_every_steps"],
            _PLACEHOLDER_MAIN["keep_ckpts"], _PLACEHOLDER_MAIN["all_in_mem"],
            seed, assets["configs_dir"],
        )

    stop.check()
    hps = sovits_utils.get_hparams_from_file(config_path)
    failed_aug = extract_all(
        dataset_44k,
        hps,
        assets["contentvec_onnx"],
        assets["rmvpe_pt"],
        backend,  # "cuda"|"xpu"|"cpu" for the f0 predictor + diff mel extractor (Vocoder)
        reporter,
        stop,
        diff_mode=True,
        nsf_hifigan_model=assets["nsf_hifigan_model"],
        aug_seed=seed,
    )
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

    # filelists AFTER the gate (they must not reference rejected aug slices);
    # deterministic — a main-model run rebuilds the identical split from the
    # same seed (val = originals only, S41 split protocol in flist.py)
    stop.check()
    build_filelists(exp_dir, slug, dataset_44k, seed, reporter)

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
    with open(
        os.path.join(cfg["assets"]["configs_dir"], "diffusion_template.yaml"),
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    display = cfg.get("model_name") or cfg["model_slug"]
    flist_dir = os.path.join(exp_dir, "filelists")
    backend = device_shim.resolve_backend(cfg)
    fp16 = bool(cfg.get("fp16", False)) and backend == "cuda"  # xpu/cpu = fp32; CPU autocast has no fp16

    config["data"]["encoder"] = encoder
    config["data"]["encoder_out_channels"] = dim
    config["data"]["training_files"] = _p(os.path.join(flist_dir, "train.txt"))
    config["data"]["validation_files"] = _p(os.path.join(flist_dir, "val.txt"))
    config["model"]["n_spk"] = 1
    config["model"]["k_step_max"] = int(cfg.get("k_step_max", 0))
    config["spk"] = {display: 0}
    config["device"] = backend  # "cuda"|"xpu"|"cpu" -> solver args.device (shim-driven amp; xpu=fp32)
    config["vocoder"]["ckpt"] = _p(cfg["assets"]["nsf_hifigan_model"])
    # ★S119 §F9 — WHICH vocoder this diffusion model is being trained against, recorded so the
    # answer survives the trip to somebody else's machine.
    #
    # A shallow-diffusion model predicts MEL; the vocoder turns that mel into audio, and a
    # FINE-TUNED vocoder is a different function from the stock one. Until now nothing anywhere
    # wrote down which one a given diffusion model was fitted to — not the checkpoint, not the
    # exported sidecar — so a model shared without its vocoder arrives sounding wrong with no
    # way for the receiving app, or the user, to even suspect why (the 东雪莲 case, S117b).
    #
    # ⚠ Today this can only ever say「the stock one」: `training/mod.rs` hard-wires
    # `assets.nsf_hifigan_model` to the shipped `training/sovits/nsf_hifigan/model` and the UI
    # offers no choice. Recording it anyway is the point — it makes the field's ABSENCE mean
    #「this did not come from us」, which is exactly the signal the import side needs.
    config["vocoder"]["utai_identity"] = _vocoder_identity(cfg["assets"]["nsf_hifigan_model"])
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


#: The stock NSF-HiFiGAN this app trains shallow diffusion against — `commands/assets.rs`'s
#: `training/sovits/nsf_hifigan/model`. Recorded as a CONSTANT rather than re-hashed at import
#: time so that「是不是出厂那一个」is answerable without the file being present.
STOCK_VOCODER_SHA256 = "2c576b63b7ed952161b70fad34e0562ace502ce689195520d8a2a6c051de29d6"


def _vocoder_identity(path):
    """`{"sha256", "bytes", "is_stock"}` for the vocoder a diffusion run is fitted to.

    ★ The hash is the identity, not the file NAME: everybody's fine-tuned vocoder is called
    `model` or `model.ckpt`, so a name would answer nothing. ~56 MB, measured well under a second
    — and it runs once per run, at config time.

    ⚠ Failing to read it must not take the training run down: this is provenance for a hint, and
    a hint is never worth losing a training run over. An unreadable file records `sha256: None`,
    which the import side reads exactly like a missing field — 「we do not know」.
    """
    import hashlib

    try:
        h = hashlib.sha256()
        n = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                n += len(chunk)
        digest = h.hexdigest()
        return {"sha256": digest, "bytes": n, "is_stock": digest == STOCK_VOCODER_SHA256}
    except OSError as e:
        logger.warning("could not fingerprint the vocoder at %s (%s) — the exported model will "
                       "not be able to say which vocoder it was trained against", path, e)
        return {"sha256": None, "bytes": None, "is_stock": None}


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
        # S41 (red-team V20): the promoted slice must not be an aug copy — val
        # is originals-only by protocol. An aug slice can only be long when its
        # source is equally long (same duration), so a non-aug candidate
        # normally exists; the fallback is a belt with a loud log.
        train_long_orig = [p for p in train_long if not is_aug_name(p)]
        if train_long_orig:
            promote = train_long_orig[0]
        else:
            promote = train_long[0]
            logger.warning(
                "no non-aug long slice available — promoting AUG slice %s into "
                "val (should be impossible: aug duration == source duration)",
                os.path.basename(promote),
            )
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
    # ★S118 §F8⒜ — a snapshot alone is resume state too. Without this a workspace whose numbered
    # grid was cleaned away (or hand-deleted for disk space) would get a base model dropped back
    # in and log 「seeded diffusion base model」 during what is actually a resume from step 4000.
    for sub in (resume_state.LATEST_DIR, resume_state.BEST_DIR):
        if resume_state.read_snapshot(expdir, sub) is not None:
            return
    if not pretrain_path:
        logger.warning("no diffusion base model — training from scratch")
        # force past the Reporter throttle — this notice follows the stage's
        # opening message within the throttle window and must not be swallowed
        reporter.stage("diff_prep", message="无扩散底模，将从零训练", force=True)
        return
    dst = os.path.join(expdir, "model_0.pt")
    tmp = dst + ".tmp"
    ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
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


#: What `load_start_state` decided. A named tuple rather than a bare tuple on purpose: four
#: same-typed fields in a row is the shape S85 banned after `(i64,i64,i64)` silently swapped.
#: `poisoned_skipped` = how many candidates were rejected for holding nan/inf weights (§F8⒡).
DiffStart = collections.namedtuple(
    "DiffStart", "step blob source had_optimizer path poisoned_skipped")

#: `source` values, in the order the chooser prefers them.
SRC_BEST = "best"                # resume_best/  — the user asked for the best point
SRC_SNAPSHOT = "latest_snapshot"  # resume_latest/ — our rolling COMPLETE resume point
SRC_NUMBERED = "numbered"        # model_<n>.pt   — upstream's grid; may carry no optimizer
SRC_FRESH = "fresh"              # nothing to restore


def load_start_state(expdir, model, optimizer, device="cpu", prefer=None):
    """WHICH diffusion checkpoint this run starts from, and what state comes with it.

    ★ MODULE-LEVEL on purpose. Inside `_train_diff` no test could drive it without a vocoder, a
    dataset and a GPU — and "the startup load is unreachable from a test" is exactly how S117's
    regression (every fresh so-vits run training from random init) shipped and survived two
    sessions. Returns a :class:`DiffStart`.

    Three sources, and the order matters:

    1. ``prefer="best"`` → ``resume_best/`` when it holds a COMPLETE snapshot. Falling back is
       LOUD: silently continuing from somewhere else than the button said is the same class of
       lie this feature exists to remove.
    2. ``resume_latest/`` when its step is >= the numbered grid's. The tie goes to the snapshot
       BY DESIGN — at a graceful stop both describe the same step, and only the snapshot has the
       scale/RNG/dataset identity.
    3. the numbered grid, i.e. exactly what upstream would have picked.

    ⚠ Two things this function does that upstream's scan does not, both measured
    (`TESTING/s118_f8a/probe_nested_pt_arithmetic.py`):
    * it checks that the rebuilt numbered path EXISTS. Upstream synthesises
      ``model_<maxstep>.pt`` from a number rather than keeping the file it found, so a hijacked
      or non-canonical name (``model_0100.pt`` -> ``model_100.pt``) produces a path that is not
      there and dies inside ``torch.load``. With a snapshot present that is now recoverable.
    * it REPORTS ``had_optimizer``. An optimizer-less resume is not a rounding error: measured,
      the AdamW state comes back ``{}``, the step counter resets to 1, so bias correction makes
      the first post-resume update exactly ±lr for EVERY parameter at once (4.7x-25.5x the warm
      optimizer's step in the same gradient state), and the second moment needs ~1700-2000 steps
      to come back within 10% of where it was.

    ⛔★S118 §F8⒡ — and it REFUSES a poisoned archive. `torch.isnan(inf)` is False (measured), so
    an inf loss walks through the loop's upstream nan gate; in fp32 — the DEFAULT path, with no
    GradScaler to skip the step — one such step leaves every weight nan, and only the step after
    it trips the gate, by which time a save boundary may already have written the poisoned state.
    So the candidates are tried IN ORDER and each one's weights are scanned after loading: a
    poisoned one is skipped LOUDLY and the next is tried; if none is healthy the run refuses
    rather than training from nan. (Not silent, and not "quietly something else": picking the
    highest number was always OUR heuristic, and correcting it when it yields garbage is what the
    user asked for — `poisoned_skipped` makes the frontend say it happened.)
    """
    prefer = (prefer or resume_state.PREFER_LATEST).strip() or resume_state.PREFER_LATEST
    from .. import numerics
    from .diffusion.logger import utils as du

    def snap_candidate(sub, source):
        blob = resume_state.read_snapshot(expdir, sub)
        if blob is None:
            return None
        path = os.path.join(resume_state.snapshot_dir(expdir, sub), resume_state.BEST_MODEL)
        return (path, blob, source, int(blob.get("global_step", -1)))

    candidates = []
    if prefer == resume_state.PREFER_BEST:
        best = snap_candidate(resume_state.BEST_DIR, SRC_BEST)
        if best is not None:
            candidates.append(best)
        else:
            logger.warning(
                "asked to continue from the BEST diffusion snapshot, but %s holds no complete one "
                "— continuing from the latest instead",
                resume_state.snapshot_dir(expdir, resume_state.BEST_DIR),
            )

    numbered, numbered_step = du.latest_numbered_path(expdir)
    if numbered is not None and not os.path.isfile(numbered):
        logger.warning(
            "the checkpoint scan points at %s, which does not exist — treating the numbered grid "
            "as empty (a non-canonical or nested .pt name can do this)", numbered,
        )
        numbered, numbered_step = None, -1

    rest = []
    snap = snap_candidate(resume_state.LATEST_DIR, SRC_SNAPSHOT)
    if snap is not None:
        rest.append(snap)
    if numbered is not None:
        rest.append((numbered, None, SRC_NUMBERED, int(numbered_step)))
    # Highest step first; a TIE goes to the snapshot BY DESIGN — at a graceful stop both describe
    # the same step and only the snapshot carries the scale/RNG/dataset identity.
    rest.sort(key=lambda c: (c[3], c[2] != SRC_NUMBERED), reverse=True)
    candidates.extend(c for c in rest if c not in candidates)

    if not candidates:
        logger.info("no diffusion checkpoint to restore — starting from step 0")
        return DiffStart(0, None, SRC_FRESH, False, "", 0)

    poisoned = 0
    for path, blob, source, _cand_step in candidates:
        step, had_optimizer = du.load_checkpoint_file(path, model, optimizer, device)
        bad = numerics.first_nonfinite_tensor(model.state_dict().items())
        if bad is None:
            logger.info("diffusion resume: %s (step %s) via %s",
                        os.path.basename(path), step, source)
            return DiffStart(int(step), blob, source, had_optimizer, path, poisoned)
        poisoned += 1
        logger.error(
            "%s: %s (step %s) holds NON-FINITE weights (first: %s) — refusing to continue from it; "
            "a previous run went numerically dead before writing it",
            resume_state.CODE_ARCHIVE_POISONED, os.path.basename(path), step, bad,
        )
        # ⛔ The poisoned MOMENTS must not survive into the next attempt: the fallback may be a
        # checkpoint with no optimizer at all, and `load_state_dict` would then simply not run.
        optimizer.state.clear()

    raise RuntimeError(
        "%s: every diffusion archive in %s holds nan/inf weights (%d tried)"
        % (resume_state.CODE_ARCHIVE_POISONED, expdir, poisoned)
    )


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
    # ★S118 §F8⒜ — WHICH archive, decided in ONE module-level place (see load_start_state).
    # Upstream's `du.load_model` only knows "the highest number wins", which cannot express
    # 「从最佳存档继续」 and cannot see that the highest number may be optimizer-less.
    start = load_start_state(
        args.env.expdir, model, optimizer, device=args.device,
        prefer=cfg.get("resume_from"),
    )
    initial_global_step = start.step
    if start.source == SRC_NUMBERED and start.step > 0 and not start.had_optimizer:
        # ⚠ A WARNING, not a refusal: the moments are gone either way, and refusing would leave
        # the user with no way to continue at all. It is worth the user's attention because the
        # first steps after this really are mis-scaled — see resume_state.CODE_OPTIMIZER_NOT_RESTORED.
        logger.warning(
            "%s: %s carries no optimizer state (upstream's save_opt=false), so the AdamW moments "
            "restart from zero", resume_state.CODE_OPTIMIZER_NOT_RESTORED, os.path.basename(start.path),
        )
        reporter.warn(resume_state.CODE_OPTIMIZER_NOT_RESTORED)
    if start.poisoned_skipped:
        # ★S118 §F8⒡ — we DID continue, but not from where the highest number said. Saying it is
        # the whole point: the user's newest archive is dead, and the run they are watching is
        # standing on an older one.
        logger.error(
            "%s: skipped %d archive(s) whose weights were nan/inf; continuing from %s (step %s)",
            resume_state.CODE_ARCHIVE_POISONED, start.poisoned_skipped,
            os.path.basename(start.path), start.step,
        )
        reporter.warn(resume_state.CODE_ARCHIVE_POISONED)
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
        # ★S118 §F8⒜ — `resumed` carries the scale/RNG/dataset identity of whatever archive we
        # just loaded; `ws_dir` is the WORKSPACE (the expdir's parent), which is where
        # `dataset.fingerprint` lives — the expdir itself never has one.
        resumed=start.blob, ws_dir=exp_dir,
        # ⛔ None on a REWIND (resuming from the BEST snapshot starts BELOW files an earlier run
        # wrote, and the sweep must not walk over them); otherwise the step we resumed at, whose
        # numbered file our own first save legitimately supersedes. See _sweep_old_checkpoints.
        superseded_step=(start.step if start.source in (SRC_NUMBERED, SRC_SNAPSHOT) else None),
    )
