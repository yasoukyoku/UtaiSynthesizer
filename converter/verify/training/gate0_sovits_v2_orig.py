"""SoVITS 4.0-v2 关卡0（原版侧，ground truth）：在「原版时代环境」= RVC 整合包
runtime（python3.9 + torch 2.0.0 + fairseq 0.12.2 + librosa 0.9.1 —— 与 4.0-v2
requirements 钉的 librosa==0.8.1/fairseq 同代最近可用环境，同 4.1 关卡0 惯例）里，
runpy 原样执行官方 4.0-v2 快照（@cf5a8fb）的预处理脚本 + ContentVec/aam-mel
oracle。全程 CPU fp32（CUDA_VISIBLE_DEVICES=-1 在 torch import 前设置）。

运行（cwd 任意）：
    D:\\MyDev\\RVC\\RVC20240604Nvidia\\runtime\\python.exe ^
        converter\\verify\\training\\gate0_sovits_v2_orig.py

Harness 补丁（零数值影响，全部登记）：
  - multiprocessing.Pool -> 串行内联（v2 resample.py 用 Pool.imap_unordered，
    runpy 的 __main__ 无法被 spawn unpickle；逐文件数学不变）
  - multiprocessing.Process -> 串行内联（v2 preprocess_hubert_f0.py 用
    Process(target=process_batch)，同因；num_processes 本就是 1）
  - configs/config.json 运行前快照、运行后恢复（v2 flist 脚本硬编码写
    configs/config.json；v2 快照里该文件是 0 字节空档，恢复为空）
  - hubert/checkpoint_best_legacy_500.pt 若缺失则从 4.1 repo pretrain/ 拷入
    （上游本来就要求用户自放；两个 gate 用同一份权重文件）
  - oracle：per-44k-wav 固定 16k 重采样 + 上游 utils.get_hubert_content 特征
    （把提取器轴与重采样轴剥离，4.1 关卡0 同款）+ 上游 modules/audio.py 的
    aam mel（librosa 0.9.1 代码轴 ground truth——上游在 data_utils 里懒生成，
    此处直接驱动同一函数）
"""
import os
import shutil
import sys

# ---- CPU fp32: before ANY torch import ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# cp932 console + upstream's Chinese warning prints (its filename pattern check
# flags every absolute Windows path) = UnicodeEncodeError — force utf-8 (S40)
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

SOVITS_V2 = r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc"
SOVITS_41 = r"D:\MyDev\so-vits-svc\so-vits-svc"
UTAI = r"D:\MyDev\Utai_v2-dev"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SLICES_ROOT = os.path.join(TESTING, "sovits_slices")  # contains speaker dir "gate"
ORIG = os.path.join(TESTING, "sovits_v2_orig")
D44K = os.path.join(ORIG, "dataset44k")
ORACLE = os.path.join(ORIG, "oracle")

sys.path.insert(0, SOVITS_V2)


def install_inline_multiprocessing():
    import multiprocessing as mp

    class _InlineProcess:
        def __init__(self, target=None, args=(), kwargs=None, **_kw):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    class _InlinePool:
        def __init__(self, processes=None, *a, **kw):
            pass

        def imap_unordered(self, fn, iterable):
            for item in iterable:
                yield fn(item)

        def close(self):
            pass

        def join(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    mp.Process = _InlineProcess
    mp.Pool = _InlinePool
    # v2 preprocess_hubert_f0 calls set_start_method('spawn', force=True) — harmless


def run_script(name, argv):
    import runpy

    sys.argv = [name] + argv
    print("== running %s %s" % (name, " ".join(argv)))
    runpy.run_path(os.path.join(SOVITS_V2, name), run_name="__main__")


def main():
    os.chdir(SOVITS_V2)
    install_inline_multiprocessing()

    # ContentVec fairseq ckpt: v2's utils.get_hubert_model loads the cwd-relative
    # hubert/checkpoint_best_legacy_500.pt — the same file the 4.1 repo carries
    hub_dst = os.path.join(SOVITS_V2, "hubert", "checkpoint_best_legacy_500.pt")
    if not os.path.exists(hub_dst):
        shutil.copyfile(
            os.path.join(SOVITS_41, "pretrain", "checkpoint_best_legacy_500.pt"),
            hub_dst,
        )
        print("copied checkpoint_best_legacy_500.pt into hubert/")

    # snapshot the repo config (flist writes it at a hardcoded path; the v2
    # snapshot ships it as an EMPTY file — restore keeps the tree pristine)
    cfg_path = os.path.join(SOVITS_V2, "configs", "config.json")
    cfg_snap = open(cfg_path, "rb").read() if os.path.exists(cfg_path) else None

    try:
        if os.path.isdir(D44K):
            shutil.rmtree(D44K)
        os.makedirs(D44K, exist_ok=True)
        flists = os.path.join(ORIG, "filelists")
        os.makedirs(flists, exist_ok=True)

        # ① resample.py — v2 chain (trim top_db=20 + unconditional peak
        #    normalize + int16), upstream defaults
        run_script("resample.py", ["--in_dir", SLICES_ROOT, "--out_dir2", D44K])

        # ② flist + config (v2 CLI has no encoder/vol flags; also writes test.txt)
        run_script(
            "preprocess_flist_config.py",
            [
                "--source_dir", D44K,
                "--train_list", os.path.join(flists, "train.txt"),
                "--val_list", os.path.join(flists, "val.txt"),
                "--test_list", os.path.join(flists, "test.txt"),
            ],
        )
        shutil.copyfile(cfg_path, os.path.join(ORIG, "config.json"))

        # ③ hubert (fairseq vec256l9) + f0 (dio) — CPU, sequential
        run_script("preprocess_hubert_f0.py", ["--in_dir", D44K])

        # ④ oracle: fixed 16k input + fairseq features + aam mel, per 44k wav
        import librosa
        import numpy as np
        import torch

        import utils as v2_utils
        from modules import audio as v2_audio

        os.makedirs(ORACLE, exist_ok=True)
        hps = v2_utils.get_hparams_from_file(os.path.join(ORIG, "config.json"))
        hmodel = v2_utils.get_hubert_model()
        spk_dir = os.path.join(D44K, "gate")
        wavs = sorted(n for n in os.listdir(spk_dir) if n.endswith(".wav"))
        for n in wavs:
            wav, _sr = librosa.load(os.path.join(spk_dir, n), sr=44100)
            wav16k = librosa.resample(wav, orig_sr=44100, target_sr=16000)
            np.save(os.path.join(ORACLE, n + ".wav16k.npy"), wav16k.astype(np.float32))
            with torch.no_grad():
                c = v2_utils.get_hubert_content(
                    hmodel, wav_16k_tensor=torch.from_numpy(wav16k)
                )  # [1, 256, T]
            np.save(os.path.join(ORACLE, n + ".venc256.npy"), c.cpu().numpy())
            # the aam mel exactly as SingDataset lazily generates it:
            # load_wav(raw==target -> pure librosa.load) -> melspectrogram -> f32.T
            wav_lw = v2_utils.load_wav(
                os.path.join(spk_dir, n),
                raw_sr=hps.data.sampling_rate,
                target_sr=hps.data.sampling_rate,
                win_size=hps.data.win_size,
                hop_size=hps.data.hop_length,
            )
            mel = v2_audio.melspectrogram(wav_lw, hps.data).astype(np.float32).T
            np.save(os.path.join(ORACLE, n + ".mel80.npy"), mel)
        print("oracle saved for %d wavs" % len(wavs))
    finally:
        if cfg_snap is None:
            if os.path.exists(cfg_path):
                os.remove(cfg_path)
        else:
            with open(cfg_path, "wb") as f:
                f.write(cfg_snap)
        print("repo config restored")

    print("GATE0 SOVITS_V2 ORIG SIDE DONE")


if __name__ == "__main__":
    main()
