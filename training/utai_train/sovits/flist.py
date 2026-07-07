# Ported from so-vits-svc 4.1-Stable preprocess_flist_config.py.
# Semantics preserved: skip <0.3s wavs (wave module), val = first 2 of the
# shuffled list / train = rest, both re-shuffled; config template + per-encoder
# dim rules (vec768l12 -> ssl=filter=gin=768; vec256l9 -> ssl=gin=256, filter
# untouched); --vol_aug couples train.vol_aug with model.vol_embedding.
# Deviations (deliberate):
#   - seeded shuffles (upstream is unseeded — irreproducible splits)
#   - filelists + config are written UTF-8 with absolute forward-slash paths
#     (upstream wrote with the locale codec and read back as UTF-8 — mojibake on
#     CJK Windows; and its relative ./dataset/44k paths assume cwd = repo root)
#   - config.json is REwritten every run so mutable train params (epochs/batch/
#     fp16_run/intervals) follow the current request; immutable params (version/
#     encoder/vol_embedding/sample rate) are guarded by the Rust run manifest
#   - diffusion.yaml is NOT generated here (the shallow-diffusion trainer is a
#     separate backend and brings its own preprocessing products)
#   - hard error when a speaker has fewer than 3 slices (upstream would write an
#     empty train list and crash mid-training)
#   - S41 augmentation protocol: PSOLA _aug slices NEVER enter val; the val
#     split is computed on the original (non-aug) pool with the exact RNG
#     stream of the pre-aug code (copies=0 stays byte-identical; val.txt is
#     identical across aug settings so val loss stays comparable), surviving
#     aug slices are then appended to the train side and shuffled with a
#     SECOND rng (touching the primary stream would shift the val order)
import json
import logging
import os
import random
import wave

from ..augment import is_aug_name

logger = logging.getLogger(__name__)

ENCODER_DIMS = {"vec768l12": 768, "vec256l9": 256}


def _wav_duration(path):
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _p(path):
    return path.replace("\\", "/")


def build_filelists(exp_dir, spk, dataset_44k_dir, seed, reporter):
    """Slice collection + seeded train/val split + filelist write — the shared
    half of build_flist_and_config (the diffusion pipeline rebuilds filelists
    every run but must NOT rewrite an existing main config.json, so this half
    stands alone). Returns (train_list, val_list, n_train, n_val)."""
    # stage name "filelist" matches the RVC trainer's — the UI label is shared
    reporter.stage("filelist", message="生成训练清单与配置")

    spk_dir = os.path.join(dataset_44k_dir, spk)
    wavs = []
    augs = []
    for file_name in sorted(os.listdir(spk_dir)):
        if not file_name.endswith("wav"):
            continue
        if file_name.startswith("."):
            continue
        file_path = _p(os.path.join(spk_dir, file_name))
        if _wav_duration(file_path) < 0.3:
            logger.info("Skip too short audio: %s", file_path)
            continue
        (augs if is_aug_name(file_name) else wavs).append(file_path)

    # the 3-slice floor is judged on ORIGINALS — aug copies must not rescue a
    # dataset that is too small to split honestly
    if len(wavs) < 3:
        raise RuntimeError(
            "切片后可用样本只有 %d 个（至少需要 3 个：2 个验证 + 1 个训练）。"
            "请提供更长的干声素材" % len(wavs)
        )

    rng = random.Random(seed)
    rng.shuffle(wavs)
    train = wavs[2:]
    val = wavs[:2]
    rng.shuffle(train)
    rng.shuffle(val)
    if augs:
        # append-then-shuffle with an independent rng: the primary stream above
        # is byte-compatible with the pre-aug code, so copies=0 output and the
        # val split/order under ANY copies stay identical to baseline
        train = train + augs
        random.Random("%s|aug-train" % seed).shuffle(train)

    flist_dir = os.path.join(exp_dir, "filelists")
    os.makedirs(flist_dir, exist_ok=True)
    train_list = os.path.join(flist_dir, "train.txt")
    val_list = os.path.join(flist_dir, "val.txt")
    with open(train_list, "w", encoding="utf-8") as f:
        for fname in train:
            f.write(fname + "\n")
    with open(val_list, "w", encoding="utf-8") as f:
        for fname in val:
            f.write(fname + "\n")
    logger.info("filelists written: %d train / %d val", len(train), len(val))
    return train_list, val_list, len(train), len(val)


def build_config(
    exp_dir,
    spk,               # speaker key in the training config = workspace slug (ASCII)
    encoder,           # "vec768l12" | "vec256l9"
    vol_embedding,
    fp16,
    total_epoch,
    batch_size,
    save_every_steps,
    keep_ckpts,
    all_in_mem,
    seed,
    configs_dir,
):
    """config.json only — the filelist PATHS it references are static
    (<exp_dir>/filelists/{train,val}.txt), so the config can be written before
    the filelists exist. S41 split: extract_all needs hps from config.json,
    while the filelists must be (re)built AFTER the aug quality gate — the
    pre-S41 build_flist_and_config coupling made that ordering impossible."""
    if encoder not in ENCODER_DIMS:
        raise RuntimeError("未知语音编码器: %s" % encoder)

    flist_dir = os.path.join(exp_dir, "filelists")
    train_list = os.path.join(flist_dir, "train.txt")
    val_list = os.path.join(flist_dir, "val.txt")

    with open(
        os.path.join(configs_dir, "config_template.json"), encoding="utf-8"
    ) as f:
        config = json.load(f)

    config["spk"] = {spk: 0}
    config["model"]["n_speakers"] = 1
    config["model"]["speech_encoder"] = encoder
    if encoder == "vec768l12":
        config["model"]["ssl_dim"] = config["model"]["filter_channels"] = config[
            "model"
        ]["gin_channels"] = 768
    elif encoder == "vec256l9":
        config["model"]["ssl_dim"] = config["model"]["gin_channels"] = 256

    if vol_embedding:
        config["train"]["vol_aug"] = config["model"]["vol_embedding"] = True

    config["train"]["seed"] = int(seed)
    config["train"]["epochs"] = int(total_epoch)
    config["train"]["batch_size"] = int(batch_size)
    config["train"]["eval_interval"] = int(save_every_steps)
    config["train"]["keep_ckpts"] = int(keep_ckpts)
    config["train"]["fp16_run"] = bool(fp16)
    config["train"]["all_in_mem"] = bool(all_in_mem)
    config["data"]["training_files"] = _p(train_list)
    config["data"]["validation_files"] = _p(val_list)

    # atomic: the diffusion pipeline trusts an EXISTING config.json (it must
    # not clobber the main model's train section) — a kill mid-write must not
    # strand a truncated file for it to trip over
    config_path = os.path.join(exp_dir, "config.json")
    tmp = config_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, config_path)


def build_flist_and_config(
    exp_dir,
    spk,
    dataset_44k_dir,
    encoder,
    vol_embedding,
    fp16,
    total_epoch,
    batch_size,
    save_every_steps,
    keep_ckpts,
    all_in_mem,
    seed,
    configs_dir,
    reporter,
):
    """Pre-S41 combined entry, kept as a thin wrapper so existing verify/gate
    scripts (gate0_sovits_run_ours etc.) keep their exact old call face and
    semantics. The production pipeline now calls build_config and
    build_filelists separately (gate between them)."""
    _, _, n_train, n_val = build_filelists(
        exp_dir, spk, dataset_44k_dir, seed, reporter
    )
    build_config(
        exp_dir,
        spk,
        encoder,
        vol_embedding,
        fp16,
        total_epoch,
        batch_size,
        save_every_steps,
        keep_ckpts,
        all_in_mem,
        seed,
        configs_dir,
    )
    return n_train, n_val
