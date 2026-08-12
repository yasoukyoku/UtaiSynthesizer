# -*- coding: utf-8 -*-
"""S138 · 我方侧(RVC)的**速度**臂 —— GPU + fp16,生产配置。与 `speed_run_orig.py` 对称。

⛔ 与 `gate1_run_ours.py` 的关系:**不许复用它**。三个理由,每一条都会污染读数:
   ⑴ 它 `:12` 在模块导入时 `CUDA_VISIBLE_DEVICES="-1"`(gate1 要 CPU 确定性)——
      而这一条要的是用户会经历的 GPU 路径;
   ⑵ 它 `:33` 的 CFG 是 `fp16: False`;
   ⑶ ⛔ 它 `:42` 用 `Reporter(throttle_secs=0.0)`「每 step 全量 emit(关卡用)」
      ⇒ 每步多一次 `json.dumps + write + flush 到管道`,而生产是 0.4
      ⇒ 会量出一个**用户永远经历不到、且是我们自己注入的**减速(S134 F8)。
   ⑷ 它 `:19` 的 `EXP` 常量指向 `gate1_ours` —— S134 那份唯一的 gate1 我方侧夹具。

★ 逐 step 墙钟靠**注入一个 Reporter 子类**(`rvc/train.py:549` 每步调它,而 reporter 是实参)
  ⇒ **零生产代码改动**,且保持生产 throttle。
★ 但**两侧共同的读数**是每个 epoch 的 `====> Epoch:` 时间戳之差
  —— 两侧 formatter 逐字节相同,而 `log_interval=200`(生产)下上游拿不到逐 step。
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
# ⛔ arena 落在 TESTING 而不是仓里:它是几百 MB 的训练产物,
#    而且 C: 的写性能今天本身就是一个会漂移的变量(S138 实测)。
ARENA = r"D:\MyDev\TESTING\s138_f7\arena"
REPO = r"D:\MyDev\Utai_v2-dev"
RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=os.path.join(ARENA, "ours_run"))
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--inject-ms", type=float, default=0.0)
ap.add_argument("--out", default=os.path.join(ARENA, "ours_out.json"))
args = ap.parse_args()

bad = [k for k in ("UTAI_DIAGNOSTICS", "CUDA_LAUNCH_BLOCKING") if os.environ.get(k)]
if bad:
    sys.exit("SPEED OURS: 环境里有 %s ⇒ 【闸的前提不满足】,不是【慢了】" % bad)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

RUN = args.run_dir
if not os.path.isdir(RUN):
    sys.exit("SPEED OURS: %s 不存在 —— 先跑 speed_arena_setup.py(闸没准备好)" % RUN)

# ⛔ arena 必须自包含:拷来的 filelist 会指回源工作区,而 data_utils 会把新算的
#    .spec.pt 写【回源目录】,且这个写入不报错不变红(S138 血训)。
with open(os.path.join(RUN, "filelist.txt"), encoding="utf-8") as f:
    outside = [p for line in f for p in line.strip().split("|")
               if p[1:3] in (":/", ":\\")
               and not os.path.normcase(os.path.abspath(p.replace("/", os.sep)))
               .startswith(os.path.normcase(os.path.abspath(RUN)))]
if outside:
    sys.exit("SPEED OURS: filelist 有 %d 个路径落在 arena 之外 ⇒ 拒绝开跑(会写回别人的目录)\n  %s"
             % (len(outside), outside[:3]))

sys.path.insert(0, os.path.join(REPO, "training"))
from utai_train.protocol import Reporter            # noqa: E402
from utai_train.stopfile import StopFlag            # noqa: E402
from utai_train.rvc import train_utils              # noqa: E402
from utai_train.rvc import train as train_mod       # noqa: E402


def gpu_state():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return [x.strip() for x in r.stdout.strip().split(",")]
    except Exception as exc:  # noqa: BLE001
        return ["err", str(exc), ""]


_loader_seen = {}
_Orig = train_mod.DataLoader


def _recording_loader(*a, **kw):
    """⛔ 只记录、不动任何旋钮。`gate1_sovits_v2_run_orig.py:68-71` 那个 shim 强制
    `num_workers=0` 并丢掉 `persistent_workers` —— 而那正是这把尺子要测的优化本身。"""
    loader = _Orig(*a, **kw)
    if not _loader_seen:
        _loader_seen.update({
            "num_workers": getattr(loader, "num_workers", None),
            "prefetch_factor": getattr(loader, "prefetch_factor", None),
            "persistent_workers": getattr(loader, "persistent_workers", None),
            "pin_memory": getattr(loader, "pin_memory", None),
        })
    return loader


train_mod.DataLoader = _recording_loader


class TimingReporter(Reporter):
    def __init__(self):
        super().__init__(throttle_secs=0.4)     # ⛔ 生产默认
        self.marks = []
        self.mid = []
        self.sample_at = {max(2, args.epochs * 14 // 3), max(3, args.epochs * 14 * 2 // 3)}

    def step(self, step, total_steps, epoch, total_epochs, lr, losses, force=False):
        if args.inject_ms:
            time.sleep(args.inject_ms / 1000.0)
        self.marks.append((time.perf_counter(), int(step), int(epoch)))
        if int(step) in self.sample_at:
            self.mid.append((int(step), gpu_state()))
        return super().step(step, total_steps, epoch, total_epochs, lr, losses, force=force)


CFG = {
    "workspace": RUN,
    "model_slug": "s138speed",
    "sample_rate": "48k",
    "version": "v2",
    "total_epoch": args.epochs,
    "batch_size": args.bs,
    "save_every_epoch": 1,          # 与上游 `-se 1` 对称
    "save_every_weights": False,    # 与上游 `-sw 0` 对称
    "keep_only_latest": True,
    "cache_gpu": False,             # 与上游 `-c 0` 对称
    "fp16": True,                   # ⛔ 生产配置(gate1 是 False)
    "seed": 1234,
    "pretrain_g": os.path.join(RVC, "assets", "pretrained_v2", "f0G48k.pth"),
    "pretrain_d": os.path.join(RVC, "assets", "pretrained_v2", "f0D48k.pth"),
}


def main():
    train_utils.get_logger(RUN)
    rep = TimingReporter()
    stop = StopFlag(os.path.join(RUN, "stop.flag.never"))
    before = gpu_state()
    t0 = time.perf_counter()
    err = None
    try:
        train_mod.train(CFG, RUN, RUN, rep, stop)
    except BaseException as exc:  # noqa: BLE001
        err = "%s: %s" % (type(exc).__name__, exc)
    wall = time.perf_counter() - t0
    marks = rep.marks
    deltas = []
    for i in range(1, len(marks)):
        t_a, s_a, e_a = marks[i - 1]
        t_b, s_b, e_b = marks[i]
        deltas.append({"dt": t_b - t_a, "step": s_b, "epoch": e_b,
                       "epoch_boundary": e_b != e_a,
                       "tail": i == len(marks) - 1,
                       "sampled": s_b in rep.sample_at})
    out = {"err": err, "wall_total_s": wall, "n_marks": len(marks), "deltas": deltas,
           "loader": _loader_seen, "gpu_before": before, "gpu_mid": rep.mid,
           "gpu_after": gpu_state(), "inject_ms": args.inject_ms, "epochs": args.epochs,
           "fp16": CFG["fp16"], "run_dir": RUN}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("SPEED OURS SIDE DONE err=%s" % err, file=sys.stderr)
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
