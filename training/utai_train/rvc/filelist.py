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

logger = logging.getLogger(__name__)


def _copy_mute_assets(mute_assets_dir, exp_dir, sr_str, fea_dim):
    ws_mute = os.path.join(exp_dir, "mute")
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
    reporter.stage("filelist", message="生成训练清单与配置")
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    fea_dim = 256 if version == "v1" else 768
    feature_dir = os.path.join(exp_dir, "3_feature%s" % fea_dim)
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")

    names = (
        set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
        & set([name.split(".")[0] for name in os.listdir(feature_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
    )
    if not names:
        raise RuntimeError("预处理产物为空：没有任何切片同时具备音频/特征/f0")

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

    ws_mute = _copy_mute_assets(mute_assets_dir, exp_dir, sr_str, fea_dim)
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
