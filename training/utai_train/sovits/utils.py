# Vendored from so-vits-svc 4.1-Stable utils.py (@ 730930d), trimmed to the
# training closure. Function bodies are verbatim unless noted.
# Removed (unused here, and their imports pulled faiss/librosa/sklearn/argparse
# into every training import): get_content, get_f0_predictor, get_speech_encoder,
# mix_model, change_rms, train_index, get_hparams, get_hparams_from_dir,
# check_git_hash, get_logger, plot_alignment_to_numpy, InferHParams.
# Deviations (deliberate):
#   - module-level logging.basicConfig(stream=sys.stdout) DROPPED and the bare
#     print() calls in load_checkpoint / latest_checkpoint_path routed to logging:
#     stdout belongs exclusively to the JSONL protocol (protocol.py)
#   - normalize_f0: upstream exit(0) on NaN -> RuntimeError (a silent exit(0)
#     would end the sidecar without a protocol verdict)
#   - save_checkpoint writes via temp file + os.replace (a kill mid-save must not
#     corrupt the resume checkpoint; same policy as the RVC trainer);
#     clean_checkpoints additionally filters to *.pth so a stale G_*.pth.tmp left
#     by a crash is never counted as (or deleted in place of) a real checkpoint
#   - plot_*: np.fromstring/tostring_rgb -> buffer_rgba (removed in mpl >= 3.8;
#     display-only tensorboard path)
import glob
import json
import logging
import os
import re

import numpy as np
import torch

from .. import ckpt_guard
from scipy.io.wavfile import read
from torch.nn import functional as F

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

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
def plot_data_to_numpy(x, y):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close()
    return data


def f0_to_coarse(f0):
  f0_mel = 1127 * (1 + f0 / 700).log()
  a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
  b = f0_mel_min * a - 1.
  f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
  # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
  f0_coarse = torch.round(f0_mel).long()
  f0_coarse = f0_coarse * (f0_coarse > 0)
  f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
  f0_coarse = f0_coarse * (f0_coarse < f0_bin)
  f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
  return f0_coarse


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False, resume=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # ★S116 §F5-③ⓒ: on a RESUME, refuse instead of quietly substituting fresh random weights
    # for keys the checkpoint cannot supply — and note the optimizer above has ALREADY been
    # restored by then, so the result is random weights driven by stale Adam moments.
    # (Measured on the real functions; see utai_train/ckpt_guard.py.)
    if resume:
        ckpt_guard.check_resume_state_dict(checkpoint_path, state_dict, saved_state_dict)
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
              logger.info("%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning." % k)
              new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
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
  def del_info(fn):
      return logger.info(f".. Free up space by deleting ckpt {fn}")
  def del_routine(x):
      return [os.remove(x), del_info(x)]
  [del_routine(fn) for fn in to_del]

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


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10,2))
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


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams_from_file(config_path, infer_mode = False):
  with open(config_path, "r", encoding="utf-8") as f:
    data = f.read()
  config = json.loads(data)
  hparams = HParams(**config)
  return hparams


def repeat_expand_2d(content, target_len, mode = 'left'):
    # content : [h, t]
    return repeat_expand_2d_left(content, target_len) if mode == 'left' else repeat_expand_2d_other(content, target_len, mode)



def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode = 'nearest'):
    # content : [h, t]
    content = content[None,:,:]
    target = F.interpolate(content,size=target_len,mode=mode)[0]
    return target


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

  def get(self,index):
    return self.__dict__.get(index)


class Volume_Extractor:
    def __init__(self, hop_size = 512):
        self.hop_size = hop_size

    def extract(self, audio): # audio: 2d tensor array
        if not isinstance(audio,torch.Tensor):
           audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio ** 2
        audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = torch.nn.functional.unfold(audio2[:,None,None,:],(1,self.hop_size),stride=self.hop_size)[:,:,:n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume
