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
  backend "sovits", dataset_dir, model_slug, version "4.1|4.0",
  workspace = the family SLOT (`open_pool` resolves `<slot>/pools/<id>/` against it),
  run_dir = THIS run's directory (weights, config, filelists, logs, resume sidecars),
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
import hashlib
import logging
import os
import shutil

import numpy as np

from .. import device as device_shim
from ..augment import augment_slices, list_aug_entries, read_wav, run_f0_gate
from ..cache import dataset_fingerprint
from ..pool import assert_identity, checked_run_dir, identity_suffix, open_pool
from ..rvc.train_utils import get_logger  # shared harness helper (single source)
from . import utils
from .cluster import build_kmeans, build_retrieval
from .extract import extract_all
from .flist import build_config, build_filelists, resolve_speakers
from .preprocess import slice_and_resample
from .train import train

logger = logging.getLogger(__name__)

VERSION_ENCODER = {"4.1": "vec768l12", "4.0": "vec256l9"}


def extract_cache_fp_text(speakers, encoder, loudnorm):
    """THE cache-identity string for the dataset_44k tree — the sovits main
    pipeline and the diffusion pipeline share the workspace, so both MUST
    build this string identically or a diff run would silently wipe the main
    run's feature caches (and vice versa). Single source, do not inline.

    ①c: `speakers` = resolve_speakers list. A single speaker fingerprints its
    one dataset_dir EXACTLY as the pre-①c code did (byte-identical string, so
    existing workspaces don't re-preprocess); multiple speakers fold each
    (slug, dataset_dir fingerprint) in list order so adding / removing /
    reordering a speaker invalidates the shared tree."""
    if len(speakers) == 1:
        fp = dataset_fingerprint(speakers[0]["dataset_dir"])
    else:
        h = hashlib.blake2b(digest_size=16)
        for sp in speakers:
            h.update(sp["slug"].encode("utf-8"))
            h.update(dataset_fingerprint(sp["dataset_dir"]).encode())
        fp = h.hexdigest()
    return "%s|enc=%s|loudnorm=%d" % (fp, encoder, int(loudnorm))


def _speaker_meta_dir(pool_dir, is_multi, slug):
    """aug_meta location for a speaker. Single-speaker keeps the flat
    <pool>/aug_meta (byte-identical to pre-①c); multi-speaker namespaces per
    slug because slice stems (000_000, ...) recur across speakers and the
    meta json is keyed by stem — a shared dir would collide.

    §F2⒝: aug_meta is a POOL product — it records which slices are augmentations, of what, with
    which seed, and it is only meaningful next to the slices it describes."""
    base = os.path.join(pool_dir, "aug_meta")
    return os.path.join(base, slug) if is_multi else base


def run(cfg, reporter, stop):
    # backend = effective device (cuda|xpu|cpu), single source (shim). Byte-identical
    # to torch.cuda.is_available() on cpu/cuda; resolves to "xpu" on an Intel box.
    backend = device_shim.resolve_backend(cfg)

    # §F2⒝ batch 2 — the SLOT and this RUN are two directories now. `workspace` still names the
    # slot because that is what `open_pool` resolves `<slot>/pools/<id>/` against; everything a RUN
    # owns (weights, config.json, filelists/, the resume sidecars, train.log) hangs off `run_dir`.
    slot_dir = cfg["workspace"]
    run_dir = checked_run_dir(cfg, slot_dir)
    os.makedirs(run_dir, exist_ok=True)
    get_logger(run_dir)  # file log train.log (utf-8) in the run dir

    assets = cfg["assets"]
    version = cfg["version"]
    if version not in VERSION_ENCODER:
        raise RuntimeError("非法 SoVITS 版本: %s（可选 4.1/4.0）" % version)
    encoder = VERSION_ENCODER[version]
    seed = int(cfg.get("seed", 1234))
    fp16 = bool(cfg.get("fp16", False)) and backend == "cuda"
    vol_embedding = bool(cfg.get("vol_embedding", False))
    loudnorm = bool(cfg.get("loudnorm", False))
    ffmpeg = assets["ffmpeg"]

    # the slice/extract products live together under dataset_44k — invalidate the
    # whole tree when the dataset OR any parameter that changes slice output /
    # feature space changes (loudnorm rewrites every wav; encoder switches the
    # .soft.pt dimension — version is manifest-immutable, belt and suspenders).
    # ①c: resolve_speakers gives N (name, slug, dataset_dir) in id order; a run
    # with no "speakers" key is a 1-element list = pre-①c single-speaker path.
    speakers = resolve_speakers(cfg)
    is_multi = len(speakers) > 1
    aug_copies = int(cfg.get("aug_copies", 0))
    # ★§F2⒝ ④d — this chain's own text, then the SHARED suffix, and the suffix is appended LAST
    # by all five chains so the concatenation order is one rule instead of five. Byte-for-byte
    # agreement with `tpool::identity_suffix` is what lets the layout 3→4 migration re-stamp an
    # existing pool rather than making the user re-preprocess it. See `pool.identity_suffix`.
    fp_text = extract_cache_fp_text(speakers, encoder, loudnorm) + identity_suffix(cfg, aug_copies)
    # §F2⒝ — the identity now NAMES the directory instead of gating a `shutil.rmtree` of it.
    # ⛔ `sovits_diff` MUST resolve to the same pool: it builds the same fp_text from the same
    # helper and shares this slot, which is exactly why that string has one home.
    # ⛔ The SLOT, never `run_dir`: this resolves `<slot>/pools/*` and treats a matching slot root
    # as a pool. Handed a run it would find neither, mint an empty pool INSIDE the run and re-run
    # hours of preprocessing — with one info line as the only sign.
    pool_dir = open_pool(slot_dir, fp_text)
    assert_identity(pool_dir, fp_text)

    dataset_44k = os.path.join(pool_dir, "dataset_44k")

    # slice + augment EACH speaker into its own dataset_44k/<slug> subdir (the
    # data loader derives the speaker id from the parent-dir name). aug_meta is
    # namespaced per speaker for multi; single-speaker keeps the flat path.
    for sp in speakers:
        spk_dir = os.path.join(dataset_44k, sp["slug"])
        meta_dir = _speaker_meta_dir(pool_dir, is_multi, sp["slug"])
        slice_and_resample(sp["dataset_dir"], spk_dir, loudnorm, ffmpeg, reporter, stop)
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
        run_dir,
        speakers[0]["slug"],
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
        speakers=speakers,
    )

    stop.check()
    hps = utils.get_hparams_from_file(os.path.join(run_dir, "config.json"))
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
    # them before the gate/filelists (mapped back to their owning speaker dir)
    for filename in failed_aug or ():
        stem = os.path.basename(filename).split(".")[0]
        fslug = os.path.basename(os.path.dirname(filename))
        logger.warning("removing aug slice with failed extraction: %s/%s", fslug, stem)
        _remove_aug_products(
            os.path.join(dataset_44k, fslug),
            _speaker_meta_dir(pool_dir, is_multi, fslug),
            stem,
        )

    stop.check()
    # f0 quality gate per speaker (report is per-speaker for multi; single keeps
    # the flat aug_gate_report.json)
    for sp in speakers:
        spk_dir = os.path.join(dataset_44k, sp["slug"])
        meta_dir = _speaker_meta_dir(pool_dir, is_multi, sp["slug"])
        report = os.path.join(
            run_dir,
            "aug_gate_report_%s.json" % sp["slug"] if is_multi else "aug_gate_report.json",
        )
        run_f0_gate(
            list_aug_entries(spk_dir, meta_dir),
            lambda stem, d=spk_dir: _load_gate_f0(d, stem),
            lambda stem, d=spk_dir, m=meta_dir: _remove_aug_products(d, m, stem),
            reporter,
            stop,
            report_path=report,
        )

    stop.check()
    build_filelists(run_dir, speakers[0]["slug"], dataset_44k, seed, reporter, speakers=speakers)

    stop.check()
    if bool(cfg.get("kmeans", False)):
        named = [(sp["name"], os.path.join(dataset_44k, sp["slug"])) for sp in speakers]
        index_path, index_rows = build_kmeans(run_dir, named, reporter, stop)
    else:
        # retrieval matrix per speaker -> <id>.index_vectors.npy (id = list order)
        index_path = None
        index_rows = 0
        for i, sp in enumerate(speakers):
            index_path, rows = build_retrieval(
                run_dir,
                os.path.join(dataset_44k, sp["slug"]),
                seed,
                reporter,
                stop,
                spk_id=i,
            )
            index_rows += rows

    stop.check()
    reporter.stage("train_prep", message="加载模型与数据，训练即将开始")
    _seed_base_checkpoints(run_dir, cfg)
    summary = train(cfg, run_dir, pool_dir, reporter, stop)

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


def _seed_base_checkpoints(run_dir, cfg):
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
        if glob.glob(os.path.join(run_dir, pattern)):
            continue
        src = cfg.get(key, "") or ""
        if not src:
            raise RuntimeError("缺少底模路径: %s" % key)
        logger.info("seeding base checkpoint %s -> %s", src, dst_name)
        dst = os.path.join(run_dir, dst_name)
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
