# SoVITS 4.0-v2 config.json builder — the v2 analog of sovits/flist.py's
# build_config. The filelist split/write itself is version-agnostic and REUSED
# from sovits/flist.py (resolve_speakers / build_filelists — plain wav paths,
# forward slashes; v2's SingDataset derives the speaker from the parent dir via
# split('/')[-2], same contract as the 4.x loader).
# v2 template differences vs 4.x that this builder owns:
#   - filelist keys are data.training_filelist/validation_filelist (4.x:
#     training_files/validation_files)
#   - no encoder dim rules (v2 is vec256l9-only: data.c_dim 256 fixed in the
#     template), no vol_embedding/vol_aug, no all_in_mem
#   - train.fp16_run is FORCED False: upstream v2 train.py has no amp at all
#     (the config key exists upstream but is never read); the UI hides the
#     switch and the Rust layer normalizes the request
#   - train.num_workers is a registered knob (default 4 = upstream's hardcoded
#     DataLoader workers; the determinism gate pins 0)
import json
import logging
import os

from ..sovits.flist import _p

logger = logging.getLogger(__name__)


def build_config(
    exp_dir,
    spk,               # speaker key in the training config = workspace slug (ASCII)
    total_epoch,
    batch_size,
    save_every_steps,
    keep_ckpts,
    seed,
    configs_dir,
    speakers=None,     # resolve_speakers list -> multi-speaker spk map / n_speakers
    num_workers=None,  # None -> keep the template default (4)
):
    flist_dir = os.path.join(exp_dir, "filelists")
    train_list = os.path.join(flist_dir, "train.txt")
    val_list = os.path.join(flist_dir, "val.txt")

    with open(
        os.path.join(configs_dir, "config_template.json"), encoding="utf-8"
    ) as f:
        config = json.load(f)

    # config.spk is keyed by the ASCII dir SLUG (the data loader resolves a
    # slice's speaker from its parent directory name), id = list order; the
    # release config (train.py) swaps these for display names for the sidecar.
    # n_speakers stays the TEMPLATE value (200) — upstream v2's flist never
    # touches it, so emb_spk keeps the base model's full 200-row table and our
    # speakers (rows 0..N-1) START FROM the base's trained speaker embeddings
    # instead of random init (the official v2 fine-tune behavior; the 4.x
    # backends set n_speakers=N because their upstream does — different branch,
    # different mechanism). The Rust layer caps co-training at 200 speakers.
    if speakers is None:
        config["spk"] = {spk: 0}
    else:
        config["spk"] = {sp["slug"]: i for i, sp in enumerate(speakers)}

    config["train"]["seed"] = int(seed)
    config["train"]["epochs"] = int(total_epoch)
    config["train"]["batch_size"] = int(batch_size)
    config["train"]["eval_interval"] = int(save_every_steps)
    config["train"]["keep_ckpts"] = int(keep_ckpts)
    config["train"]["fp16_run"] = False  # v2 is pure fp32 (see header)
    if num_workers is not None:
        config["train"]["num_workers"] = int(num_workers)
    config["data"]["training_filelist"] = _p(train_list)
    config["data"]["validation_filelist"] = _p(val_list)

    config_path = os.path.join(exp_dir, "config.json")
    tmp = config_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, config_path)
