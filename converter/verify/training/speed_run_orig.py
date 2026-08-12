# -*- coding: utf-8 -*-
"""S138 · 上游侧(RVC)的**速度**臂 —— GPU + fp16,生产配置。

⛔ 与 `gate1_run_orig.py` 的关系:形状照抄,但**轴不同,不许混**。
   gate1 那条是 **CPU fp32**(`-g -1` 是有意为之的 CPU 垫片,为了逐 step 数值确定性);
   这一条是 **GPU fp16**(用户会经历的路径)⇒ `-g 0`,并且 fp16 由 config.json 给。

⛔ 三条从 `gate1_run_orig.py` 原样继承的机理(它们的理由与轴无关):
   ⑴ 绕过 `main()` 直调 `train.run(0, 1, hps, logger)` —— `main()` 用 `mp.Process`,
      Windows 是 spawn ⇒ 子进程重新 import `__main__`(= 本脚本)⇒ 无限递归;
   ⑵ 自己设 `MASTER_ADDR`/`MASTER_PORT`(本来由 `main()` 设);
   ⑶ `USE_LIBUV=0` —— torch>=2.4 在 Windows 上的 TCPStore 需要。
   ⚠ 但 GPU 分支与 CPU 分支有一处**实质不同**:`run()` 在 CUDA 可见时会把 net_g/net_d
      包进 **DDP(gloo, world_size=1)**(上游 `train.py:202-206`)。**那正是要量的东西**
      (我方无 DDP)⇒ 绝不许绕开它。

⛔ 上游「正常完训」是 `os._exit(2333333)`(`train.py:635`),**不是 rc=0**。
   ⇒ 判成功不能用 rc==0;判据放在**日志里有没有够数的 `====> Epoch:` 行**。
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"
PRETRAIN = os.path.join(RVC, "assets", "pretrained_v2")

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="s138speed")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=4)
args, _ = ap.parse_known_args()

EXP_DIR = os.path.join(RVC, "logs", args.exp)

os.environ["USE_LIBUV"] = "0"
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "8007")   # 与 gate1 的 8003 错开

# ⛔ preflight:诊断模式的两个变量会让这次读数测的是别的东西
_bad = [k for k in ("UTAI_DIAGNOSTICS", "CUDA_LAUNCH_BLOCKING") if os.environ.get(k)]
if _bad:
    sys.exit("SPEED ORIG: 环境里有 %s ⇒ 【闸的前提不满足】,不是【慢了】" % _bad)

if not os.path.isdir(EXP_DIR):
    sys.exit("SPEED ORIG: %s 不存在 —— 先跑 speed_arena_setup.py(闸没准备好)" % EXP_DIR)
for n in ("f0G48k.pth", "f0D48k.pth"):
    if not os.path.isfile(os.path.join(PRETRAIN, n)):
        sys.exit("SPEED ORIG: 缺底模 %s —— 闸的前提不满足" % n)

# ⛔ 顺序承重:上游 train.py 在 **import 时**就 get_hparams() 读 argv 并设 CUDA_VISIBLE_DEVICES
sys.argv = [
    os.path.join(RVC, "infer", "modules", "train", "train.py"),
    "-e", args.exp,
    "-sr", "48k",
    "-f0", "1",
    "-bs", str(args.bs),
    "-g", "0",            # ⛔ 与 gate1 的 `-g -1` 相反:这一条要 GPU
    "-te", str(args.epochs),
    "-se", "1",
    "-pg", os.path.join(PRETRAIN, "f0G48k.pth"),
    "-pd", os.path.join(PRETRAIN, "f0D48k.pth"),
    "-l", "1",
    "-c", "0",
    "-sw", "0",
    "-v", "v2",
]

os.chdir(RVC)
sys.path.insert(0, RVC)

import torch  # noqa: E402

import infer.modules.train.train as train  # noqa: E402  (原版文件,零改动)
from infer.lib.train import utils  # noqa: E402

# ⛔ 与 gate1 那条**方向相反**的断言:那条要求 CUDA 不可见,这条要求它可见。
if not torch.cuda.is_available():
    sys.exit("SPEED ORIG: CUDA 不可见(CUDA_VISIBLE_DEVICES=%r)—— 这条臂要的是用户会经历的 GPU 路径"
             % os.environ.get("CUDA_VISIBLE_DEVICES"))


def main():
    hps = train.hps
    assert os.path.abspath(hps.model_dir) == os.path.abspath(EXP_DIR), \
        "model_dir=%s 与预期 %s 不一致" % (hps.model_dir, EXP_DIR)
    logger = utils.get_logger(hps.model_dir)
    logger.info("SPEED ORIG: GPU fp16_run=%s device=%s CUDA_VISIBLE_DEVICES=%r"
                % (hps.train.fp16_run, torch.cuda.get_device_name(0),
                   os.environ.get("CUDA_VISIBLE_DEVICES")))
    train.run(0, 1, hps, logger)
    print("SPEED ORIG SIDE DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
