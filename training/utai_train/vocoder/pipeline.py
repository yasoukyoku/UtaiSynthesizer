"""NSF-HiFiGAN vocoder fine-tune pipeline — backend "vocoder" (S40).

Upstream: openvpi/SingingVocoders @4d0889c (process.py → train.py → export_
ckpt.py), vendored verbatim under utai_train/vocoder/. This file replaces the
CLI entry points (click / ProcessPoolExecutor orchestration) with the utai
stage/protocol harness; the Trainer construction transcribes train.py:31-104.

Run config keys (run.json, written by the Rust TrainingManager):
  backend "vocoder", workspace, dataset_dir, model_slug, model_name,
  total_steps (REAL steps — lightning global = 2x, see harness.py),
  save_every_steps (REAL = val_check_interval batches), batch_size,
  keep_ckpts, crop_mel_frames, freeze_mpd, seed, stop_file,
  assets{ffmpeg, vocoder_pretrain}

Deviations vs upstream (deliberate; gate/README-covered; design doc appendix A):
  1. slicing stage: upstream requires a pre-sliced 44.1kHz dataset (README:118
     微调请使用 44100Hz 采样率音频); we vendor the same openvpi slicer2 its
     README tells users to run by hand, then resample every slice to 44100
     (librosa/soxr) and write float32 wav. Rationale beyond convenience —
     红队 A1: upstream wav2spec computes mel on the RESAMPLED audio but f0 on
     the ORIGINAL-sr array interpreted as 44100 (process.py:34-49 +
     wav2F0.py:61-73), so any >44.1k source trains with f0 flattened by
     sr/44100 and drifting per frame. Normalizing to 44100 here makes that
     branch dead code and keeps f0/mel/audio coherent.
  2. source sr < 44100 fails the RUN loudly listing files (upstream skips
     silently = dataset shrinks without a trace). ffmpeg-fallback decodes
     (exotic containers librosa can't read) are decoded AT 44100 and噪 logged
     — their true source rate is unknowable without ffprobe (edge of an edge).
  3. any wav2spec failure (corrupt slice / f0 None / parselmouth assert) fails
     the RUN loudly after the scan — never a silent skip (红队 A10).
  4. npz written atomically (*.tmp.npz + replace) + dataset-fingerprint cache
     invalidation over slices/ + npz/ (skip-if-exists caches: S37 lesson);
     slices/ carries a .complete marker so a crash mid-slice rebuilds.
  5. train/val split is seeded (upstream random.sample on an unordered set)
     and val_num adapts to small datasets: min(10, max(1, n//10)) — upstream's
     fixed 10 crashes below 10 files. Slices shorter than crop_mel_frames are
     dropped LOUDLY (log + stage message): the upstream collater deletes them
     per batch (silent shrink) and crashes outright on an all-short batch
     (np.stack([]) — 红队 A19).
  6. config dict assembled in code (base.yaml→base_hifi.yaml flattened; the
     ft_hifigan finetune knobs re-derived for the CLASSIC arch — there is no
     upstream classic-finetune yaml). Decision table for keys where base and
     ft disagree (红队 A23): lr 1e-5 (ft; base trains from scratch at 1e-4),
     clip_grad_norm 1 (ft), num_sanity_val_steps 2 (ft), crop_mel_frames 32
     default (ft 16G preset), val_num adaptive≤10 (ft: 10), pc_aug/mini_nsf
     OFF (PC-NSF-only; the shipped base model is the classic 2024.02 ckpt),
     key_aug OFF (base/ft default; README warns it may hurt quality).
     volume_aug true 固定项 (红队 A22): its collater clamp — applied to EVERY
     record, outside the probability branch — is what floors training mels at
     ln(1e-5), the same floor inference/mel.rs feeds the vocoder; never expose
     it as a switch. The dict is dumped verbatim to workspace/config.yaml
     (train.py:42-43 semantics).
  7. no print_config, lightning progress bar disabled (stdout is protocol-
     owned; tqdm/ANSI would spam the stderr ring) — display-only, zero RNG.
  8. stop flag → trainer.should_stop; a resume-grade checkpoint is saved
     AFTER fit on EVERY path that trained (stop AND completed — DsModel-
     Checkpoint only saves on val boundaries, an off-grid tail would silently
     retrain on resume; 红队 A8). Manually saved checkpoints are outside
     DsModelCheckpoint's top-k bookkeeping, so the workspace is pruned to
     keep_ckpts afterwards (红队 A24).
  9. weights/ snapshots ({'generator':sd}+config.json, export_ckpt.py
     semantics) on every periodic/best/stop/final — the import-candidate pool
     (S38: the protocol must reference convert-ready artifacts).

  10. slices are chopped to ≤15s pieces (SLICE_MAX_SEC): continuous singing
      rarely has -40dB/300ms gaps, so slicer2 alone can yield a couple of
      60s+ mega-chunks (smoke-proven on a 126s full-song vocal — 2 slices) —
      too few for a train/val split, and validation runs a FULL-LENGTH
      generator forward per file (unbounded VRAM). 15s = the community's
      5-15s dataset-slice practice; the 0.37s training crop is indifferent
      to chop boundaries.

RNG discipline (gate1 axis, 红队 A12): between pl.seed_everything and
trainer.fit there is NO python/numpy/torch RNG consumption in this file or
harness.py — model init_weights (inside fit's setup, before the finetune
overwrite) is part of the stream and aligns because both gate sides run the
identical sequence. The train dataloader has NO shuffle (upstream Sequential-
Sampler; order = filelist line order) — never "fix" that.
"""
import copy
import json
import logging
import os
import pathlib
import random
import re

import numpy as np
import yaml

from ..augment import (
    augment_slices,
    is_aug_name,
    list_aug_entries,
    read_wav,
    run_f0_gate,
)
from .. import device as device_shim
from ..cache import dataset_fingerprint, invalidate_extract_caches
from ..rvc.train_utils import get_logger  # shared harness helper (single source)
from ..rvc.slicer2 import Slicer  # single source — the vendored openvpi slicer
from ..sovits.preprocess import _decode  # single source decoder (librosa+ffmpeg)

logger = logging.getLogger(__name__)

TARGET_SR = 44100
SLICE_MAX_SEC = 15.0  # deviation 10 — chop mega-chunks (see module header)
# 审查修复: pieces below this are skipped (deviation 10 补充) — anything under
# (win-hop)//2 = 768 samples CRASHES wav2spec's reflect pad outright, and
# anything under crop_mel_frames dies in the filelist guard anyway; 0.5s is a
# comfortable floor for material that contributes nothing to training
MIN_PIECE_SAMPLES = TARGET_SR // 2


def run(cfg, reporter, stop):
    exp_dir = cfg["workspace"]
    os.makedirs(exp_dir, exist_ok=True)
    # root logger → stderr + train.log BEFORE any vendored/lightning import:
    # base_task_gan's module-level logging.basicConfig(stream=sys.stdout) must
    # stay a no-op (root already has handlers) or it would poison the protocol
    get_logger(exp_dir)

    assets = cfg["assets"]
    # backend = effective device (cuda|xpu|cpu). Drives the aug-gate rmvpe device;
    # the Lightning trainer handles xpu separately in _train (Lightning has no XPU).
    backend = device_shim.resolve_backend(cfg)
    pretrain = assets.get("vocoder_pretrain") or ""
    if not os.path.isfile(pretrain):
        raise RuntimeError("找不到声码器微调底模: %s" % pretrain)

    # ---- cache identity (slices + npz are keyed on slice names — S37 lesson;
    # bump the version tag whenever slice/process semantics change) ----
    fp_text = "%s|vocoder-v3" % dataset_fingerprint(cfg["dataset_dir"])
    invalidate_extract_caches(exp_dir, fp_text, ("slices", "npz"))

    slices_dir = os.path.join(exp_dir, "slices")
    npz_dir = os.path.join(exp_dir, "npz")
    flist_dir = os.path.join(exp_dir, "filelists")

    slice_dataset(cfg["dataset_dir"], slices_dir, assets["ffmpeg"], reporter, stop)

    stop.check()
    _augment_vocoder(exp_dir, slices_dir, npz_dir,
                     int(cfg.get("aug_copies", 0)), int(cfg.get("seed", 1234)),
                     reporter, stop)

    stop.check()
    config = build_train_config(cfg, pretrain, flist_dir)
    process_slices(slices_dir, npz_dir, config, reporter, stop)

    stop.check()
    # S41 quality gate — rmvpe-blooded BY DESIGN: parselmouth (this chain's own
    # f0 lineage) is measurably blind to the PSOLA glitch tail (its continuity
    # prior smooths over frames rmvpe reads as 300+ cents off — gate_aug_semantic
    # part 4 keeps that blind spot on record), so the gate must not use the
    # npz f0 products
    _vocoder_aug_gate(exp_dir, slices_dir, npz_dir,
                      assets.get("rmvpe_pt", ""), backend, reporter, stop)

    stop.check()
    build_filelists(npz_dir, flist_dir, int(cfg.get("seed", 1234)),
                    int(config["crop_mel_frames"]), reporter)

    stop.check()
    reporter.stage("train_prep", message="加载底模与数据，训练即将开始")
    summary = _train(cfg, exp_dir, config, reporter, stop)

    stopped = summary.pop("stopped")
    reporter.done("stopped" if stopped else "completed", summary)


# ─── stage: slice ────────────────────────────────────────────────────────────

def _probe_sr(path):
    """Header-only sample-rate probe (no decode)."""
    import librosa

    try:
        return int(librosa.get_samplerate(path))
    except Exception:
        return None


def slice_dataset(dataset_dir, slices_dir, ffmpeg, reporter, stop):
    """dataset originals -> slices/<idx0>_<idx1>.wav @44100 float32.
    Cache-valid (fingerprint survived + .complete marker) → skipped entirely."""
    import librosa
    import soundfile as sf

    marker = os.path.join(slices_dir, ".complete")
    names = sorted(os.listdir(dataset_dir))
    if os.path.isfile(marker):
        reporter.stage("slice", done=len(names), total=len(names),
                       message="切片缓存有效，跳过")
        return
    os.makedirs(slices_dir, exist_ok=True)
    for name in os.listdir(slices_dir):  # crash-mid-slice leftovers
        p = os.path.join(slices_dir, name)
        if os.path.isfile(p):
            os.remove(p)

    # sr guard BEFORE any slicing — collect ALL violations for one loud error
    # (deviation 2; upsampling low-rate material is pointless for a vocoder)
    violations = []
    for name in names:
        sr = _probe_sr(os.path.join(dataset_dir, name))
        if sr is not None and sr < TARGET_SR:
            violations.append("  %s (%dHz)" % (name, sr))
    if violations:
        raise RuntimeError(
            "以下素材采样率低于 44100Hz（声码器微调要求 ≥44.1kHz 干声）：\n"
            + "\n".join(violations)
        )

    written = 0
    skipped_tiny = 0
    for n, name in enumerate(names):
        stop.check()
        reporter.stage("slice", done=n, total=len(names), message=name)
        wav, sr = _decode(os.path.join(dataset_dir, name), ffmpeg)
        if sr < TARGET_SR:  # probe couldn't read the header — belt over braces
            raise RuntimeError("素材采样率低于 44100Hz: %s (%dHz)" % (name, sr))
        slicer = Slicer(sr=sr)  # openvpi defaults — the slicer upstream README prescribes
        idx1 = 0
        max_len = int(SLICE_MAX_SEC * TARGET_SR)
        for chunk in slicer.slice(wav):
            if sr != TARGET_SR:
                # normalize to 44100 HERE (deviation 1 / 红队 A1): keeps
                # wav2spec's f0 and mel on the same clock
                chunk = librosa.resample(chunk, orig_sr=sr, target_sr=TARGET_SR)
            # deviation 10: chop mega-chunks to ≤15s pieces; sub-0.5s pieces
            # (chop tails / stray click-length source files) are SKIPPED — a
            # ≤768-sample piece would crash wav2spec's reflect pad and, per
            # deviation 3, take the whole run down with it (审查修复)
            for off in range(0, max(1, chunk.size), max_len):
                piece = chunk[off:off + max_len]
                if piece.size < MIN_PIECE_SAMPLES:
                    if piece.size > 0:
                        skipped_tiny += 1
                    continue
                sf.write(
                    os.path.join(slices_dir, "%03d_%03d.wav" % (n, idx1)),
                    piece, TARGET_SR, subtype="FLOAT",
                )
                idx1 += 1
                written += 1
    if skipped_tiny:
        logger.warning("skipped %d sub-%.1fs slice fragments (training-irrelevant)",
                       skipped_tiny, MIN_PIECE_SAMPLES / TARGET_SR)
    reporter.stage("slice", done=len(names), total=len(names))
    if written == 0:
        raise RuntimeError("切片后没有任何有效样本（素材可能全为静音或过短）")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")


# ─── stage: process (wav2spec → npz) ─────────────────────────────────────────

def process_slices(slices_dir, npz_dir, config, reporter, stop):
    from .process_sv import wav2spec  # vendored process.py (heavy imports)

    os.makedirs(npz_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(slices_dir) if n.endswith(".wav"))
    failures = []
    aug_dropped = 0
    for i, name in enumerate(names):
        stop.check()
        reporter.stage("process", done=i, total=len(names), message=name)
        out = os.path.join(npz_dir, os.path.splitext(name)[0] + ".npz")
        if os.path.exists(out):  # atomic writes make skip-if-exists safe
            continue
        tmp = out[:-4] + ".tmp.npz"  # np.savez appends .npz unless present
        ok, result = wav2spec(
            config, pathlib.Path(os.path.join(slices_dir, name)), pathlib.Path(tmp)
        )
        if not ok:
            if is_aug_name(name):
                # S41: aug slices are OUR OWN products — degrade to "drop this
                # copy" instead of taking the whole run down (deviation 3 is a
                # user-material contract; red-team A4)
                logger.warning("dropping aug slice with failed wav2spec: %s (%s)",
                               name, result)
                _remove_vocoder_aug_products(
                    slices_dir, npz_dir, os.path.splitext(name)[0]
                )
                aug_dropped += 1
            else:
                failures.append("  %s: %s" % (name, result))
            try:
                os.remove(tmp)
            except OSError:
                pass
            continue
        os.replace(tmp, out)
    reporter.stage("process", done=len(names), total=len(names))
    if aug_dropped:
        reporter.stage("process", message="剔除 %d 个特征提取失败的增强片" % aug_dropped,
                       force=True)
    if failures:  # deviation 3: never a silent skip (user material only)
        raise RuntimeError("以下切片特征提取失败：\n" + "\n".join(failures))


# ─── stage: filelist ─────────────────────────────────────────────────────────

def build_filelists(npz_dir, flist_dir, seed, crop_frames, reporter):
    reporter.stage("filelist", message="划分训练/验证集")
    names = sorted(
        os.path.join(npz_dir, n) for n in os.listdir(npz_dir) if n.endswith(".npz")
    )
    usable, short = [], []
    for p in names:
        with np.load(p) as z:
            frames = int(z["mel"].shape[0])  # npz mel = [T, num_mels]
        (usable if frames >= crop_frames else short).append(p)
    if short:  # deviation 5: loud, and the collater's crash face goes unreachable
        logger.error(
            "剔除 %d 个过短切片（mel < %d 帧，无法参与 %d 帧随机裁剪）：%s",
            len(short), crop_frames, crop_frames,
            ", ".join(os.path.basename(p) for p in short),
        )
        reporter.stage("filelist", message="剔除 %d 个过短切片" % len(short), force=True)
    # S41 split protocol: val is drawn from ORIGINAL slices only, with the
    # exact pre-aug rng semantics — copies=0 stays byte-identical, and the val
    # set is identical across aug settings (val loss stays comparable);
    # surviving aug slices go to the train side only. The 4-slice floor is
    # judged on originals (aug copies must not rescue a too-small dataset).
    originals = [p for p in usable if not is_aug_name(p)]
    augs = [p for p in usable if is_aug_name(p)]
    if len(originals) < 4:
        raise RuntimeError(
            "有效切片过少（%d 个，至少需要 4 个）：请提供更多/更长的连续干声素材"
            % len(originals)
        )
    val_num = min(10, max(1, len(originals) // 10))
    rng = random.Random(seed)  # deviation 5: upstream samples an unordered set, unseeded
    val = set(rng.sample(originals, val_num))
    train = [p for p in originals if p not in val] + augs

    os.makedirs(flist_dir, exist_ok=True)
    for fname, rows in (("train", sorted(train)), ("valid", sorted(val))):
        with open(os.path.join(flist_dir, fname), "w", encoding="utf8") as f:
            for p in rows:  # upstream writes sorted posix lines (process.py:94-97)
                print(pathlib.Path(p).as_posix(), file=f)
    logger.info("filelists: %d train / %d val", len(train), len(val))


# ─── S41 PSOLA augmentation (design doc B3, vocoder row) ─────────────────────

def _vocoder_meta_dir(exp_dir):
    return os.path.join(exp_dir, "aug_meta")


def _remove_vocoder_aug_products(slices_dir, npz_dir, stem):
    """wav + npz + meta as ONE unit — the trainer reads npz only (audio is
    embedded), so a deleted wav with a surviving npz would keep training on
    'removed' data invisibly (red-team F4/V18, the quietest pollution channel)."""
    for p in (
        os.path.join(slices_dir, stem + ".wav"),
        os.path.join(npz_dir, stem + ".npz"),
        os.path.join(os.path.dirname(slices_dir), "aug_meta", stem + ".json"),
    ):
        try:
            os.remove(p)
        except OSError:
            pass


def _augment_vocoder(exp_dir, slices_dir, npz_dir, copies, seed, reporter, stop):
    import soundfile as sf

    meta_dir = _vocoder_meta_dir(exp_dir)

    def write_float(tmp_path, samples, sr):
        # explicit format: the .tmp suffix defeats soundfile's inference; the
        # slice dir contract is float32 WAV (FLOAT subtype), matching
        # slice_dataset's own writes
        sf.write(tmp_path, samples, sr, format="WAV", subtype="FLOAT")

    augment_slices(
        slices_dir,
        copies,
        seed,
        meta_dir,
        read_wav,
        write_float,
        lambda stem: _remove_vocoder_aug_products(slices_dir, npz_dir, stem),
        reporter,
        stop,
    )

    # orphan npz sweep: a crash between wav removal and npz removal leaves an
    # npz the stale sweep can no longer see (it enumerates wavs)
    if os.path.isdir(npz_dir):
        for name in os.listdir(npz_dir):
            if not name.endswith(".npz") or not is_aug_name(name):
                continue
            stem = os.path.splitext(name)[0]
            if not os.path.exists(os.path.join(slices_dir, stem + ".wav")):
                try:
                    os.remove(os.path.join(npz_dir, name))
                except OSError:
                    pass


def _vocoder_aug_gate(exp_dir, slices_dir, npz_dir, rmvpe_pt, backend, reporter, stop):
    """rmvpe-blooded f0 gate over the aug AUDIO pairs (never the npz f0 — its
    parselmouth lineage is blind to the PSOLA glitch tail, see run())."""
    entries = list_aug_entries(slices_dir, _vocoder_meta_dir(exp_dir))
    if not entries:
        run_f0_gate(entries, lambda stem: None,
                    lambda stem: None, reporter, stop)
        return
    if not rmvpe_pt or not os.path.isfile(rmvpe_pt):
        raise RuntimeError(
            "数据增强质检需要 RMVPE 资产（assets.rmvpe_pt），未找到: %s" % rmvpe_pt
        )
    import torch

    from ..sovits.f0.RMVPEF0Predictor import RMVPEF0Predictor

    predictor = RMVPEF0Predictor(
        hop_length=512,
        sampling_rate=TARGET_SR,
        dtype=torch.float32,
        device=backend,  # "cuda"|"xpu"|"cpu"; sovits rmvpe is fp32 on every backend (no is_half)
        threshold=0.05,
        model_path=rmvpe_pt,
    )

    def load_f0(stem):
        try:
            wav, _sr = read_wav(os.path.join(slices_dir, stem + ".wav"))
            f0, uv = predictor.compute_f0_uv(wav)
            f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
            uv = np.asarray(uv, dtype=np.float64).reshape(-1)
            n = min(len(f0), len(uv))
            return f0[:n], uv[:n] > 0.5
        except Exception:
            logger.warning("gate: f0 failed for %s", stem)
            return None

    run_f0_gate(
        entries,
        load_f0,
        lambda stem: _remove_vocoder_aug_products(slices_dir, npz_dir, stem),
        reporter,
        stop,
        report_path=os.path.join(exp_dir, "aug_gate_report.json"),
    )


# ─── config assembly (deviation 6 — see module header decision table) ────────

def build_train_config(cfg, pretrain, flist_dir):
    freeze_mpd = bool(cfg.get("freeze_mpd", False))
    total_real = int(cfg["total_steps"])
    save_real = max(1, int(cfg["save_every_steps"]))
    return {
        # mel/preprocessing — the LOCKED OpenVPI standard format (base_hifi.yaml:49-57;
        # README: 微调请勿修改 mel 参数)
        "audio_sample_rate": 44100,
        "audio_num_mel_bins": 128,
        "hop_size": 512,
        "fft_size": 2048,
        "win_size": 2048,
        "fmin": 40,
        "fmax": 16000,
        "fmax_for_loss": None,
        "crop_mel_frames": int(cfg.get("crop_mel_frames", 32)),
        # f0 (base_hifi:9-11)
        "pe": "parselmouth",
        "f0_min": 65,
        "f0_max": 1100,
        # data index — our workspace layout (upstream base_hifi:36-38)
        "DataIndexPath": flist_dir,
        "train_set_name": "train",
        "valid_set_name": "valid",
        # augmentation: classic finetune keeps key_aug/pc_aug OFF (decision
        # table); volume_aug is a FIXED-ON item — its collater clamp is the
        # ln(1e-5) mel floor that matches the inference mel (红队 A22)
        "key_aug": False,
        "key_aug_prob": 0.5,
        "aug_min": 0.9,
        "aug_max": 1.4,
        "pc_aug": False,
        "pc_aug_rate": 0.5,
        "pc_aug_key": 5,
        "volume_aug": True,
        "volume_aug_prob": 0.5,
        # losses (base_hifi:23-28, 45-46)
        "use_stftloss": False,
        "lab_aux_melloss": 45,
        "lab_aux_stftloss": 2.5,
        "loss_fft_sizes": [2048, 2048, 4096, 1024, 512, 256, 128, 1024, 2048, 512],
        "loss_hop_sizes": [512, 240, 480, 100, 50, 25, 12, 120, 240, 50],
        "loss_win_lengths": [2048, 1200, 2400, 480, 240, 120, 60, 600, 1200, 240],
        "mel_vmin": -6.0,
        "mel_vmax": 1.5,
        # model — classic NSF-HiFiGAN (base_hifi:67-77; mini_nsf stays False,
        # the exporter has no source_conv branch and the base ckpt is classic)
        "model_args": {
            "mini_nsf": False,
            "upsample_rates": [8, 8, 2, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "discriminator_periods": [3, 5, 7, 11, 17, 23, 37],
            "resblock": "1",
        },
        "task_cls": "utai_train.vocoder.training.nsf_HiFigan_task.nsf_HiFigan",
        # finetune lr 1e-5 double-AdamW (ft_hifigan:92-104; the lr_scheduler_args
        # block is DEAD upstream config — configure_optimizers never builds a
        # scheduler (base_task_gan.py:465-469), lr is constant — kept for the dump)
        "discriminate_optimizer_args": {
            "optimizer_cls": "torch.optim.AdamW",
            "lr": 0.00001, "beta1": 0.8, "beta2": 0.99, "weight_decay": 0,
        },
        "generater_optimizer_args": {
            "optimizer_cls": "torch.optim.AdamW",
            "lr": 0.00001, "beta1": 0.8, "beta2": 0.99, "weight_decay": 0,
        },
        "lr_scheduler_args": {
            "scheduler_cls": "lr_scheduler.scheduler.WarmupLR",
            "warmup_steps": 5000, "min_lr": 0.00001,
        },
        "clip_grad_norm": 1,  # ft_hifigan:111 (base: null — the finetune config clips)
        "ds_workers": 4,
        "dataloader_prefetch_factor": 2,
        "batch_size": int(cfg["batch_size"]),
        "num_valid_plots": 100,
        "log_interval": 100,  # GLOBAL-step units (2x real; global is always even here)
        "num_sanity_val_steps": 2,  # ft_hifigan:123
        # REAL steps = training batches: train.py sets check_val_every_n_epoch=None,
        # making an int val_check_interval a CROSS-epoch cumulative batch count
        "val_check_interval": save_real,
        "num_ckpt_keep": int(cfg.get("keep_ckpts", 5)),
        "max_updates": 2 * total_real,  # lightning global units (D+G each count)
        "permanent_ckpt_start": 200000,
        "permanent_ckpt_interval": 40000,
        # lightning plumbing (base_hifi:126-136)
        "pl_trainer_accelerator": "auto",
        "pl_trainer_devices": "auto",
        "pl_trainer_precision": "32-true",  # README: bf16 damages quality; fixed
        "pl_trainer_num_nodes": 1,
        "pl_trainer_strategy": {
            "name": "auto",
            "process_group_backend": "nccl",
            "find_unused_parameters": True,
        },
        "nccl_p2p": True,
        "seed": int(cfg.get("seed", 1234)),
        # finetune (ft_hifigan:150-156; freezing gate 红队 A3 — frozen_params
        # alone is a silent no-op, freezing_enabled is the switch)
        "finetune_enabled": True,
        "finetune_ckpt_path": str(pretrain),
        "finetune_ignored_params": [],
        "finetune_strict_shapes": True,
        "freezing_enabled": freeze_mpd,
        "frozen_params": ["discriminator.mpd"] if freeze_mpd else [],
    }


# ─── stage: train ────────────────────────────────────────────────────────────

def _prune_workspace_ckpts(exp_dir, keep):
    """Manually saved tail checkpoints live outside DsModelCheckpoint's top-k
    bookkeeping and would accumulate across stop/resume cycles (红队 A24) —
    prune the workspace to the newest `keep` by step number."""
    ckpts = []
    for name in os.listdir(exp_dir):
        m = re.fullmatch(r"model_ckpt_steps_(\d+)\.ckpt", name)
        if m:
            ckpts.append((int(m.group(1)), name))
    ckpts.sort(reverse=True)
    for _, name in ckpts[max(1, int(keep)):]:
        try:
            os.remove(os.path.join(exp_dir, name))
            logger.info("pruned workspace checkpoint %s", name)
        except OSError:
            pass


def _train(cfg, exp_dir, config, reporter, stop):
    """Transcription of upstream train.py:31-104 (same statement order — the
    RNG stream from seed_everything through fit must match the original for
    gate1) + the utai protocol callback. No RNG consumption is allowed between
    seed_everything and fit (红队 A12)."""
    import torch  # noqa: F401  (CUDA_VISIBLE_DEVICES already set by runner)

    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # train.py:107

    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger

    from .harness import UtaiNsfTask, UtaiProtocolCallback, save_weights_snapshot
    from .utils.training_utils import DsModelCheckpoint, get_latest_checkpoint_path

    # Lightning 2.6.5 ships NO XPU accelerator (accelerators/ = cpu/cuda/mps/xla only),
    # so accelerator="auto" would SILENTLY pick CPU on an Intel box — design §4.5 forbids
    # a silent fallback. The other four training objects are plain torch loops on the
    # device shim and DO use the Intel GPU; only this Lightning trainer cannot. So when the
    # resolved backend is xpu, force CPU EXPLICITLY and warn LOUDLY (log + protocol). Set
    # before the config.yaml dump so the record reflects the real accelerator. cuda/cpu keep
    # the config's "auto" (byte-noop — auto resolves to the same accelerator there, and
    # gate1/noop run on CPU where backend != "xpu", so this branch never fires under a gate).
    if device_shim.resolve_backend(cfg) == "xpu":
        config["pl_trainer_accelerator"] = "cpu"
        _xpu_msg = (
            "声码器微调：Intel(XPU) 显卡暂不支持 GPU 训练（Lightning 无 XPU 后端），"
            "本对象将在 CPU 上训练（可能很慢）——其余训练对象正常使用 Intel GPU"
        )
        logger.warning(_xpu_msg)
        reporter.stage("train_prep", message=_xpu_msg, force=True)

    work_dir = str(exp_dir)
    # config.yaml dump BEFORE the work_dir key lands in the dict (train.py:42-44)
    with open(os.path.join(work_dir, "config.yaml"), "w", encoding="utf8") as f:
        yaml.safe_dump(config, f)
    config.update({"work_dir": work_dir})
    # pristine copy for the weights config.json — build_model mutates
    # config['model_args'] in place (harness.export_config_json reads this)
    pristine = copy.deepcopy(config)

    pl.seed_everything(config["seed"], workers=True)  # train.py:50 — BEFORE the task
    task = UtaiNsfTask(config=config)

    protocol_cb = UtaiProtocolCallback(
        reporter, stop, cfg["total_steps"], work_dir, pristine
    )
    trainer = pl.Trainer(
        accelerator=config["pl_trainer_accelerator"],
        devices=config["pl_trainer_devices"],
        num_nodes=config["pl_trainer_num_nodes"],
        # 登记偏离(红队 A21 实弹):上游 get_strategy 深走 lightning ≤2.5 的私有
        # accelerator_connector API(_register_external_accelerators_and_strategies
        # 等),lightning 2.6 已移除,直接 AttributeError。单设备场景下它的返回
        # 与 Trainer 自身的 "auto" 解析完全同义(SingleDeviceStrategy,策略选择
        # 不消耗 RNG);多卡训练本就不在支持面(RVC DDP 同款 backlog)。
        # gate1 原版侧 harness 以同款 shim(get_strategy -> "auto")对齐。
        strategy="auto",
        precision=config["pl_trainer_precision"],
        # NB: lightning force-reorders Checkpoint-class callbacks to the END of
        # the list — our protocol callback always runs first on shared hooks;
        # snapshots therefore come from live state_dict, never from ckpt files
        callbacks=[
            DsModelCheckpoint(
                dirpath=work_dir,
                filename="model_ckpt_steps_{step}",
                auto_insert_metric_name=False,
                monitor="step",
                mode="max",
                save_last=False,
                save_top_k=config["num_ckpt_keep"],
                permanent_ckpt_start=config["permanent_ckpt_start"],
                permanent_ckpt_interval=config["permanent_ckpt_interval"],
                verbose=True,
            ),
            protocol_cb,
        ],
        logger=TensorBoardLogger(
            save_dir=work_dir, name="lightning_logs", version="lastest"
        ),
        val_check_interval=config["val_check_interval"],
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        max_steps=config["max_updates"],
        use_distributed_sampler=True,
        num_sanity_val_steps=config["num_sanity_val_steps"],
        enable_progress_bar=False,  # deviation 7 (stdout/stderr hygiene)
    )
    ckpt_path = get_latest_checkpoint_path(work_dir)
    if ckpt_path:
        logger.info("resuming from %s", ckpt_path)
    trainer.fit(task, ckpt_path=ckpt_path)

    # ---- post-fit accounting (deviation 8) ----
    stopped = protocol_cb.stop_requested
    final_global = trainer.global_step
    initial_global = (
        protocol_cb.initial_global
        if protocol_cb.initial_global is not None
        else final_global
    )
    steps_run = (final_global - initial_global) // 2
    real_final = final_global // 2
    total_real = int(cfg["total_steps"])

    if steps_run == 0 and not stopped:
        raise RuntimeError(
            "没有执行任何训练步：目标总步数 (%s) 不大于已训练进度 (%s)，"
            "请增大总步数后再续训" % (total_real, real_final)
        )

    if steps_run > 0:
        # 尾档公平上擂台(登记偏离,用户 S40 走查提出):自然完训停在存档网格之间
        # 时 lightning 不再跑 val(优雅停会跑——冒烟与真训两场实测),final 档就
        # 从未与 best 比较过。补一次显式验证:on_validation_end 照常走 periodic/
        # best 逻辑(DsModelCheckpoint 在非 fit 状态自动跳过自己的存档),
        # verbose=False 是硬要求(lightning 的结果表用裸 print 打 stdout=协议)。
        # 依赖 UtaiNsfTask.setup 的幂等化——上游 setup 会无条件重建模型,
        # trainer.validate 二次进入 setup 时会把训练好的权重换成随机初始化。
        if protocol_cb.last_val_global != final_global:
            trainer.validate(task, verbose=False)
        # resume-grade tail checkpoint: DsModelCheckpoint only saves on val
        # boundaries — an off-grid tail would silently retrain on resume
        trainer.save_checkpoint(
            os.path.join(work_dir, "model_ckpt_steps_%d.ckpt" % final_global)
        )
        _prune_workspace_ckpts(work_dir, config["num_ckpt_keep"])
        snap = save_weights_snapshot(
            protocol_cb.weights_dir, "vocoder_%d.ckpt" % real_final, task, pristine
        )
        tail_metric = (
            protocol_cb.last_val_value
            if protocol_cb.last_val_global == final_global
            else None
        )
        reporter.ckpt(
            "stop" if stopped else "final", snap, real_final, trainer.current_epoch,
            metric=tail_metric,
        )
        losses = getattr(task, "_utai_losses", {})
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else 0.0
        reporter.step(real_final, total_real, trainer.current_epoch, 0, lr, losses,
                      force=True)

    return {
        "stopped": stopped,
        "steps": real_final,
        "steps_this_run": steps_run,
        "best_val": protocol_cb.best_val,
        "weights_dir": protocol_cb.weights_dir.replace("\\", "/"),
        "format": "44100Hz / hop 512 / 128 mel",
    }
