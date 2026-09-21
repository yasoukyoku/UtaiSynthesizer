# Ported from RVC 20240604 infer-web.py click_train() lines 482-570 (filelist +
# config generation). Semantics preserved: names = intersection of the 4 artifact
# dirs, 5-field lines "gt|feature|f0coarse|f0nsf|spk_id", 2 appended mute rows,
# shuffled; config template v1/<sr>.json unless (v2 and sr!=40k) -> v2/<sr>.json;
# existing config.json is kept (resume consistency). Deviations: mute assets are
# COPIED into the workspace first (the trainer writes .spec.pt caches next to gt
# wavs — the install dir may be read-only), forward-slash paths instead of the
# escaped-backslash hack, UTF-8 filelist, seeded shuffle.
import json
import logging
import os
import random
import shutil
from .. import stage_codes
from .. import prep_codes

logger = logging.getLogger(__name__)


def _copy_mute_assets(mute_assets_dir, pool_dir, sr_str, fea_dim):
    # into the POOL, not the slot root: the copy is keyed by sample rate and feature dim (both in
    # the preprocessing identity) and the trainer writes a `.spec.pt` next to the gt wav, so it is
    # a preprocessing product in every sense. Its absolute path is baked into filelist.txt.
    ws_mute = os.path.join(pool_dir, "mute")
    pairs = [
        ("0_gt_wavs/mute%s.wav" % sr_str, "0_gt_wavs/mute%s.wav" % sr_str),
        ("3_feature%s/mute.npy" % fea_dim, "3_feature%s/mute.npy" % fea_dim),
        ("2a_f0/mute.wav.npy", "2a_f0/mute.wav.npy"),
        ("2b-f0nsf/mute.wav.npy", "2b-f0nsf/mute.wav.npy"),
    ]
    for src_rel, dst_rel in pairs:
        src = os.path.join(mute_assets_dir, src_rel)
        dst = os.path.join(ws_mute, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
    return ws_mute


def _p(path):
    return path.replace("\\", "/")


def build_filelist_and_config(
    exp_dir,
    pool_dir,
    sr_str,
    version,
    spk_id,
    configs_dir,
    mute_assets_dir,
    seed,
    fp16_run,
    reporter,
    multi_speaker=False,
):
    # ①c: for a multi-speaker run every slice stem is "<spk_id>_<idx0>_<idx1>[_aug<n>]" (the
    # preprocess prefix), so the per-line spk_id is recovered from the stem's first "_"-token
    # instead of the single scalar `spk_id` (which is used only for the mute rows then). The
    # single-speaker path (multi_speaker=False) stamps every line with `spk_id` = byte-identical.
    # ⚠ TWO directories on purpose: every artifact READ here is a pool product, while both
    # artifacts WRITTEN (filelist.txt, config.json) stay at the slot root — the filelist is
    # rewritten in full by every run and holds absolute paths INTO the pool, so pooling it would
    # only duplicate it, and config.json is per-run by design (fp16_run follows the run's toggle).
    reporter.stage("filelist", message=stage_codes.FILELIST)
    gt_wavs_dir = os.path.join(pool_dir, "0_gt_wavs")
    fea_dim = 256 if version == "v1" else 768
    feature_dir = os.path.join(pool_dir, "3_feature%s" % fea_dim)
    f0_dir = os.path.join(pool_dir, "2a_f0")
    f0nsf_dir = os.path.join(pool_dir, "2b-f0nsf")

    # S172: the four sets are bound before the intersection so the refusal can say HOW they
    # disagree. "0 slices have all four products" is not actionable; "gt=370 feature=1 f0=370"
    # names the stage that failed, and it is the only clue we get from a log.
    gt_names = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    fea_names = set(name.split(".")[0] for name in os.listdir(feature_dir))
    f0_names = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_names = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
    names = gt_names & fea_names & f0_names & f0nsf_names
    if not names:
        raise RuntimeError(
            "%s: no slice has all 4 products: gt=%d feature%d=%d f0=%d f0nsf=%d"
            % (
                prep_codes.PREP_PRODUCTS_MISMATCH_CODE,
                len(gt_names),
                fea_dim,
                len(fea_names),
                len(f0_names),
                len(f0nsf_names),
            )
        )

    # sorted() before the seeded shuffle: set iteration order is per-process random
    # (str hash randomization), which would defeat the seed
    opt = []
    for name in sorted(names):
        line_spk = int(name.split("_")[0]) if multi_speaker else spk_id
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
            % (
                _p(gt_wavs_dir),
                name,
                _p(feature_dir),
                name,
                _p(f0_dir),
                name,
                _p(f0nsf_dir),
                name,
                line_spk,
            )
        )

    ws_mute = _copy_mute_assets(mute_assets_dir, pool_dir, sr_str, fea_dim)
    for _ in range(2):
        opt.append(
            "%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|%s/2a_f0/mute.wav.npy|%s/2b-f0nsf/mute.wav.npy|%s"
            % (_p(ws_mute), sr_str, _p(ws_mute), fea_dim, _p(ws_mute), _p(ws_mute), spk_id)
        )

    random.Random(seed).shuffle(opt)
    with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(opt))
    logger.info("filelist written: %s entries", len(opt))

    # config template selection: v2 has no 40k template upstream, reuses v1/40k
    if version == "v1" or sr_str == "40k":
        template = os.path.join(configs_dir, "v1", "%s.json" % sr_str)
    else:
        template = os.path.join(configs_dir, "v2", "%s.json" % sr_str)
    config_save_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_save_path):
        with open(template, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["train"]["fp16_run"] = bool(fp16_run)
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")
    else:
        # resume keeps the original config EXCEPT fp16_run: the precision toggle
        # must take effect on resume (GradScaler/is_half read this value) —
        # otherwise turning fp16 off after a NaN blowup would be silently ignored
        with open(config_save_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("train", {}).get("fp16_run") != bool(fp16_run):
            config["train"]["fp16_run"] = bool(fp16_run)
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=True)
                f.write("\n")
    return len(opt)
