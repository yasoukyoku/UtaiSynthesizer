"""SoVITS 4.0-v2 关卡1（原版侧）：在 training venv（torch 2.5.1，与我方侧同一
torch —— 隔离代码轴，4.1 关卡1 惯例）里 runpy 驱动**未改动的**官方 v2 train.py。
CUDA 被藏起 → main() 自然落入上游**原生 cpurun** 路径（skip_optimizer=True、
epoch/step 强制从 1/0 起；无 DDP/dist/mp.spawn —— 我方 train 的 skip_optimizer
gate 旗镜像的就是这套语义）。损失从 tensorboard events 读（stdout 只有对象 repr）。

Harness 补丁（零数值影响，全部登记）：
  - root logger 抢占（basicConfig WARNING 在上游 import 前 —— 上游 utils.py
    import 时 basicConfig(DEBUG, stdout) 会放 numba/matplotlib 洪水；first-call-wins）
  - torch.stft 垫片：上游 mel_processing 传 return_complex=False（torch 2.x 已
    移除）→ 改 True + view_as_real（与我方 vendored 适配同式，位恒等）
  - torch.istft 垫片：上游 Generator_Noise 喂实数 [...,2] spec（torch 2.x 已
    移除）→ view_as_complex 同缓冲视图（S68 参照 harness 同款，位恒等）
  - DataLoader 补丁：num_workers 强制 0 + 剥 persistent_workers（RNG 流对齐，
    双侧一致；上游硬编码 4/1）
  - data_utils.load_wav 替换为同数学 shim（上游 librosa.core.load 的 core 命名
    空间在新 librosa 不保证存在；raw==target 路径 = 纯 librosa.load，逐位同）

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_v2_run_orig.py
"""
import logging
import os
import sys

# ---- CPU + quiet logging BEFORE any upstream import ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.basicConfig(level=logging.WARNING)

SOVITS_V2 = r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
GATE_CFG = os.path.join(TESTING, "gate1_sovits_v2_config.json")

sys.path.insert(0, SOVITS_V2)
os.chdir(SOVITS_V2)  # get_hparams uses ./logs/<model>; filelists are absolute

import torch  # noqa: E402

# ---- torch>=2 shims (bit-identical equivalences, see header) ----
_orig_stft = torch.stft


def _stft_shim(input, *args, **kwargs):
    if kwargs.get("return_complex") is False:
        kwargs["return_complex"] = True
        return torch.view_as_real(_orig_stft(input, *args, **kwargs))
    return _orig_stft(input, *args, **kwargs)


torch.stft = _stft_shim

_orig_istft = torch.istft


def _istft_shim(input, *args, **kwargs):
    if not torch.is_complex(input):
        input = torch.view_as_complex(input.contiguous())
    return _orig_istft(input, *args, **kwargs)


torch.istft = _istft_shim

# ---- DataLoader worker pin (RNG stream alignment, both sides 0) ----
import torch.utils.data as _tud  # noqa: E402

_OrigLoader = _tud.DataLoader


def _loader_shim(*args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs.pop("persistent_workers", None)
    return _OrigLoader(*args, **kwargs)


_tud.DataLoader = _loader_shim

# data_utils does `from torch.utils.data import DataLoader` at ITS import time —
# patching the attribute BEFORE importing data_utils covers it
import data_utils as v2_data_utils  # noqa: E402

assert v2_data_utils.DataLoader is _loader_shim

# load_wav shim (identical math for the raw==target path the gate takes)
import librosa  # noqa: E402
import numpy as np  # noqa: E402


def _load_wav_shim(wav_path, raw_sr, target_sr=16000, win_size=800, hop_size=200):
    assert raw_sr == target_sr, "gate only exercises the same-rate path"
    return librosa.load(wav_path, sr=raw_sr)[0]


v2_data_utils.load_wav = _load_wav_shim


def main():
    import runpy

    sys.argv = ["train.py", "-c", GATE_CFG, "-m", "gate1_sovits_v2"]
    print("== running upstream v2 train.py (native cpurun) ==")
    runpy.run_path(os.path.join(SOVITS_V2, "train.py"), run_name="__main__")
    print("GATE1 SOVITS_V2 ORIG SIDE DONE")


if __name__ == "__main__":
    main()
