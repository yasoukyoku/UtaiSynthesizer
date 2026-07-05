# Vendored from RVC 20240604 infer/lib/train/process_ckpt.py (savee only).
# Deviations: explicit output path (original hardcodes cwd-relative assets/weights/),
# no i18n, atomic write, raises on failure instead of returning a traceback string.
# The saved dict layout (weight/config/info/sr/f0/version) is byte-compatible with
# the original small-model format that converter/convert.py --type rvc consumes.
import os
from collections import OrderedDict

import torch


def savee(ckpt, sr, if_f0, out_path, epoch, version, hps):
    opt = OrderedDict()
    opt["weight"] = {}
    for key in ckpt.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = ckpt[key].half()
    opt["config"] = [
        hps.data.filter_length // 2 + 1,
        32,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        hps.model.spk_embed_dim,
        hps.model.gin_channels,
        hps.data.sampling_rate,
    ]
    opt["info"] = "%sepoch" % epoch
    opt["sr"] = sr
    opt["f0"] = if_f0
    opt["version"] = version
    tmp_path = out_path + ".tmp"
    torch.save(opt, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path
