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
  num_sanity_val_steps 2 / max_updates 30（global = 15 实际步）/ seed 1234 /
  ⛔ S134 更正：这一行此前写「max_updates 24（global = 12 实际步）」，两个数都是陈的。
  真值由 vocoder/pipeline.py:577 `"max_updates": 2 * total_real` 与本文件驱动的
  total_steps=15 现算 = 30 global / 15 实际步；盘上实证 = SingingVocoders/experiments/
  gate1_voc 的 model_ckpt_steps_{10,20,30}.ckpt；gate1_vocoder_compare.py 的
  EXPECT_TRAIN_POINTS=15 / EXPECT_VAL_MIN=4 也是按真值写的。照旧文字核点数会把对的判成错的。/
  finetune = 正式底模 / fp32 CPU（库内共识：训练侧 bitwise 对拍只能 CPU）。
"""
import copy
import pathlib
import shutil
import sys
import time

import os                                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402
# ⚠ guard 在 import 时把 stdout **与 stderr** 都钉成 UTF-8(原来这里只钉了 stdout)

import yaml

APP = pathlib.Path(r"D:\MyDev\Utai_v2-dev")
GATE = pathlib.Path(r"D:\MyDev\TESTING\gate1_vocoder")
SLICES = pathlib.Path(r"D:\MyDev\TESTING\smoke_vocoder\ws\slices")
PRETRAIN = (APP / "data/models/training/vocoder/nsf_hifigan_44.1k_hop512_128bin_2024.02.ckpt")

sys.path.insert(0, str(APP / "training"))
from utai_train.vocoder import pipeline as vpipe  # noqa: E402
from utai_train.vocoder import process_sv  # noqa: E402


class _Rep:
    """⛔ S140:形参**逐字照抄** `training/utai_train/protocol.py` 的 Reporter
    (:60/:78/:98/:110/:125/:128)。此前仓内四个桩类有**三种**写法(`*a, **k` / 窄签名缺
    `force` / 少五个方法),而 `gate_driver_arity` **结构上看不见类方法**(它只解析模块级
    `def`,:97-99 与 :184-186)⇒ 桩与生产 Reporter 的签名漂移是一条**全仓无人看守**的面,
    而它的 `VERDICT ALL-BIND` 那条绿对这一面是空的。
    ⚠ 生产里 `stage`/`step` 各有 8 处与 4 处 `force=True` 的调用点 ⇒ 窄签名不只在 stage 上会炸。
    """

    def stage(self, stage, done=None, total=None, message=None, force=False):
        pass

    def step(self, step, total_steps, epoch, total_epochs, lr, losses, force=False):
        pass

    def ckpt(self, kind, path, step, epoch, metric=None):
        pass

    def warn(self, code):
        pass

    def done(self, reason, summary=None):
        pass

    def error(self, message):
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
    # ⛔⛔ S139:前置断言 + 输入身份,全部跑在第一句 rmtree 之前。
    #    ⚠ 这条链的 orig 侧 expdir **不在这里**(在上游树 `SingingVocoders\experiments\gate1_voc`,
    #      由 `gate1_vocoder_run_orig.py:75-77` 自清)—— 那是**设计**不是缺陷,别去「修 prepare」:
    #      正因为 run_orig 自清,声码器是五条链里**唯一免疫「忘了跑 prepare ⇒ 参照侧续训」**的一条。
    if not SLICES.is_dir() or not any(SLICES.iterdir()):
        raise G1.GateUnrunnable(
            "声码器的输入切片不在 / 是空的:%s\n"
            "       ⛔ gate0/gate1 里**没有任何脚本能重建它**(它是声码器冒烟的产物)。" % SLICES)
    if not PRETRAIN.is_file():
        raise G1.GateUnrunnable("声码器底模不在:%s" % PRETRAIN)
    # ⛔ S140:地板从与真值无关的 5 改成实测真值 10(9 片 .wav + 1 个 `.complete` 标记)。
    ident = G1.src_identity("gate1/vocoder 的输入(冒烟切片,不可再生)",
                            str(SLICES), [""], min_files=10)

    GATE.mkdir(parents=True, exist_ok=True)
    for sub in ("ours", "orig"):
        d = GATE / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        G1.write_input_identity(str(d), ident)

    cfg = gate_config()

    npz_dir = GATE / "npz"
    npz_dir.mkdir(exist_ok=True)
    # §F2⒝: the pool is the parent of both product directories in this fixture — the aug
    # cleanup path needs it to reach `<pool>/aug_meta`.
    vpipe.process_slices(str(SLICES.parent), str(SLICES), str(npz_dir), cfg, _Rep(), _Stop())

    # ⛔⛔ S140:**`--rebuild-fixtures` 重建不了这条链真正吃的东西。**
    #    两侧训练消费的是 `npz/*.npz`,而 `vocoder/pipeline.py:296` 是 skip-if-exists,
    #    且 npz 目录**不在上面那段 rmtree 的名单里**(只删 ours/orig)⇒ 盘上那批 npz
    #    在「重建夹具」之后仍然是原来那批。实测(2026-08-12):9 份 npz 全部 mtime
    #    **2026-07-06 20:09**,而它们自称的来源切片是 **20:56** —— 特征比输入早 47 分钟。
    #    ⇒ 上面那行 `[INPUT]` 描述的是 SLICES,而被消费的是 npz,**两者不是一回事**。
    #    这里补两件:⑴ 一条**真判据**(件数必须配对,少一个 npz 就是漏算);
    #               ⑵ 一行把上面这个事实打进转录 —— ⛔「打印是汇报不是判据」,所以两件都要。
    wavs = sorted(p.name for p in SLICES.iterdir() if p.suffix == ".wav")
    npzs = sorted(p.stem for p in npz_dir.iterdir() if p.suffix == ".npz")
    if sorted(os.path.splitext(w)[0] for w in wavs) != npzs:
        raise G1.GateUnrunnable(
            "gate1/vocoder:切片与特征对不上 —— %d 片 .wav vs %d 份 .npz\n"
            "       只在切片侧:%s;只在特征侧:%s\n"
            "       ⇒ `process_slices` 是 skip-if-exists,一份漏算的特征在这里不会自愈。"
            % (len(wavs), len(npzs),
               sorted(set(os.path.splitext(w)[0] for w in wavs) - set(npzs))[:5],
               sorted(set(npzs) - set(os.path.splitext(w)[0] for w in wavs))[:5]))
    try:
        _newest_npz = max(p.stat().st_mtime for p in npz_dir.iterdir() if p.suffix == ".npz")
        _newest_wav = max(p.stat().st_mtime for p in SLICES.iterdir() if p.suffix == ".wav")
        G1._say("[NOTE] 声码器的特征是 skip-if-exists 的:npz 最新 %s / 切片最新 %s ⇒ "
                "`--rebuild-fixtures` **不重建特征**,这一跑吃的是盘上已有的那 %d 份。"
                % (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_newest_npz)),
                   time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_newest_wav)), len(npzs)))
    except ValueError:
        pass

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
    G1.run("gate1_vocoder_prepare", main)
