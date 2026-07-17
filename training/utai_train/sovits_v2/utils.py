# Vendored from so-vits-svc 4.0-v2 utils.py (@ cf5a8fb), trimmed to the
# training closure. Function bodies are verbatim unless noted.
# Removed (unused by the v2 training chain, and their imports pulled fairseq/
# librosa/argparse/matplotlib into every training import): interpolate_f0
# (SingDataset carries its own copy), compute_f0_parselmouth, f0_to_coarse
# (v2 feeds CONTINUOUS LF0, no coarse embedding), get_hubert_model/
# get_hubert_content/get_content (ContentVec runs via ONNX in extract.py),
# get_hparams (argparse), get_hparams_from_dir, check_git_hash, get_logger,
# plot_alignment_to_numpy, load_wav_to_torch, load_filepaths_and_text.
# Deviations (deliberate):
#   - upstream's module-level logging.basicConfig(stream=sys.stdout, DEBUG)
#     DROPPED and every bare print() routed to logging: stdout belongs
#     exclusively to the JSONL protocol (protocol.py), and the basicConfig
#     would flood DEBUG from numba/matplotlib (same preemption the S68
#     reference harnesses need)
#   - torch.load with explicit weights_only=False (torch>=2.6 default flip;
#     S43 convention)
#   - normalize_f0: upstream exit(0) on NaN -> RuntimeError (a silent exit(0)
#     would end the sidecar without a protocol verdict; same policy as the
#     4.1 tree)
#   - save_checkpoint writes via temp file + os.replace; clean_checkpoints
#     filters to *.pth so a stale G_*.pth.tmp left by a crash is never counted
#     as (or deleted in place of) a real checkpoint (S38 policy)
#   - plot_*: np.fromstring/tostring_rgb -> buffer_rgba (removed in mpl >= 3.8;
#     display-only tensorboard path)
#   - load_wav: librosa.core.load/resample (removed namespace / 0.8 positional
#     args) -> librosa.load / keyword resample; the resample+pad branch is
#     never taken in training (raw_sr == target_sr == 44100)
#   - get_hparams_from_file opens with encoding="utf-8" (upstream used the
#     locale codec — mojibake on CJK Windows)
import glob
import json
import logging
import os
import re

import numpy as np
import torch
import librosa

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)


def normalize_f0(f0, x_mask, uv, random_scale=True):
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        raise RuntimeError("normalize_f0 produced NaN")  # upstream: exit(0)
    return f0_norm * x_mask


def resize_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res


def compute_f0_dio(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    import pyworld
    if p_len is None:
        p_len = wav_numpy.shape[0] // hop_length
    f0, t = pyworld.dio(
        wav_numpy.astype(np.double),
        fs=sampling_rate,
        f0_ceil=800,
        frame_period=1000 * hop_length / sampling_rate,
    )
    f0 = pyworld.stonemask(wav_numpy.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return resize_f0(f0, p_len)


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception:
            # shape-tolerant merge (upstream verbatim semantics): missing /
            # mismatched keys keep the freshly initialized value — this is THE
            # base-model mechanism (emb_spk 200 rows vs our n_speakers rows)
            logger.info("%s is not in the checkpoint", k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '%s' (iteration %s)", checkpoint_path, iteration)
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration %s to %s", iteration, checkpoint_path)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    tmp_path = checkpoint_path + ".tmp"
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted([f for f in ckpts_files if f.startswith(_x) and f.endswith(".pth") and not f.endswith("_0.pth")], key=sort_key)

    to_del = [os.path.join(path_to_models, fn) for fn in
              (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
    for fn in to_del:
        os.remove(fn)
        logger.info(".. Free up space by deleting ckpt %s", fn)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.debug(x)
    return x


def plot_data_to_numpy(x, y):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close()
    return data


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def repeat_expand_2d(content, target_len):
    # content : [h, t]  (v2 "left" variant, verbatim)

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


def load_wav(wav_path, raw_sr, target_sr=16000, win_size=800, hop_size=200):
    audio = librosa.load(wav_path, sr=raw_sr)[0]
    if raw_sr != target_sr:
        audio = librosa.resample(audio,
                                 orig_sr=raw_sr,
                                 target_sr=target_sr,
                                 res_type='kaiser_best')
        target_length = (audio.size // hop_size +
                         win_size // hop_size) * hop_size
        pad_len = (target_length - audio.size) // 2
        if audio.size % 2 == 0:
            audio = np.pad(audio, (pad_len, pad_len), mode='reflect')
        else:
            audio = np.pad(audio, (pad_len, pad_len + 1), mode='reflect')
    return audio


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
