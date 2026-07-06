# -*- coding: utf-8 -*-
"""gate1_vocoder_prepare — 声码器训练轨迹对拍的共享输入（S40）。

产出（D:/MyDev/TESTING/gate1_vocoder/）：
  npz/            共享特征（gate0 已证逐位 = 原版 process 产物）
  filelists/      train/valid（双侧共用同一份清单 = 同数据同顺序）
  gate_config.yaml   原版侧 train.py --config 输入（我方 build_train_config 的
                     gate 小型化版本 dump——双侧同值同源）
  ours/ orig/     两侧工作区（run_orig / run_ours 分别写入）

gate 小型化（双侧同值，README 登记）：
  batch 2 / crop 16 / ds_workers 0（原版侧由 run_orig 的 monkeypatch shim 对齐
  vendored A2 补丁）/ log_interval 1（红队 A11：默认 100 下 24 global 步只有
  step0 一个点 = 空交集假 PASS）/ val_check_interval 5（≥2 个 val 边界，S39 铁律）/
  num_sanity_val_steps 2 / max_updates 24（global = 12 实际步）/ seed 1234 /
  finetune = 正式底模 / fp32 CPU（库内共识：训练侧 bitwise 对拍只能 CPU）。
"""
import copy
import pathlib
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

APP = pathlib.Path(r"D:\MyDev\Utai_v2-dev")
GATE = pathlib.Path(r"D:\MyDev\TESTING\gate1_vocoder")
SLICES = pathlib.Path(r"D:\MyDev\TESTING\smoke_vocoder\ws\slices")
PRETRAIN = (APP / "data/models/training/vocoder/nsf_hifigan_44.1k_hop512_128bin_2024.02.ckpt")

sys.path.insert(0, str(APP / "training"))
from utai_train.vocoder import pipeline as vpipe  # noqa: E402
from utai_train.vocoder import process_sv  # noqa: E402


class _Rep:
    def stage(self, *a, **k):
        pass


class _Stop:
    def check(self):
        pass

    def requested(self):
        return False


def cpu_pretrain():
    """CUDA-archived base ckpt -> a CPU-storage copy: the ORIG side's verbatim
    load_pre_train_model (bare torch.load, no map_location) crashes on the CPU
    gate otherwise. Tensor values identical; our vendored side carries the
    registered map_location deviation and would accept either file."""
    import torch

    dst = GATE / "pretrain_cpu.ckpt"
    if not dst.exists():
        ck = torch.load(str(PRETRAIN), map_location="cpu")
        torch.save(ck, str(dst))
    return dst


def gate_config():
    cfg = vpipe.build_train_config(
        {"total_steps": 15, "save_every_steps": 5, "batch_size": 2,
         "keep_ckpts": 5, "crop_mel_frames": 16, "seed": 1234},
        str(cpu_pretrain()), str(GATE / "filelists"),
    )
    cfg["ds_workers"] = 0          # RNG all in-process (A2 conditional path)
    cfg["log_interval"] = 1        # A11: every batch logs training/*
    # val_check_interval already = save_every_steps = 5 (real batches)
    return cfg


def main():
    GATE.mkdir(parents=True, exist_ok=True)
    for sub in ("ours", "orig"):
        d = GATE / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    cfg = gate_config()

    npz_dir = GATE / "npz"
    npz_dir.mkdir(exist_ok=True)
    vpipe.process_slices(str(SLICES), str(npz_dir), cfg, _Rep(), _Stop())
    vpipe.build_filelists(str(npz_dir), str(GATE / "filelists"), 1234,
                          int(cfg["crop_mel_frames"]), _Rep())

    dump = copy.deepcopy(cfg)
    # the orig side's train.py importlib-loads task_cls — point it at the
    # ORIGINAL class (our side constructs its UtaiNsfTask subclass directly
    # and never reads this key; the subclass only adds loss capture + a
    # logging print_arch, zero math)
    dump["task_cls"] = "training.nsf_HiFigan_task.nsf_HiFigan"
    with open(GATE / "gate_config.yaml", "w", encoding="utf8") as f:
        yaml.safe_dump(dump, f)
    print("prepared:", GATE)
    n_train = len((GATE / "filelists" / "train").read_text(encoding="utf8").splitlines())
    n_val = len((GATE / "filelists" / "valid").read_text(encoding="utf8").splitlines())
    print(f"filelists: {n_train} train / {n_val} val; config dumped")


if __name__ == "__main__":
    main()
