"""RVC 关卡1（原版侧，ground truth）—— S134 新增。

在**我们的解释器**里运行 `D:\\MyDev\\RVC\\RVC20240604Nvidia` 的**未改动**
`infer/modules/train/train.py`，CPU fp32 确定性。与 sovits / sovits_v2 / diff /
vocoder 四条的 `*_run_orig.py` 同形。

    <python> converter/verify/training/gate1_run_orig.py

为什么它到 S134 才存在：README 里 RVC 那条原版侧一直是一行**手拼的命令行**
（`USE_LIBUV=0 <python> infer\\modules\\train\\train.py -e gate1 ...`），
而那一行有两个照抄跑不通的地方 —— ① `USE_LIBUV=0 <cmd>` 是 bash 前缀语法，
PowerShell/cmd 下不是合法命令；② 参数里全是 `<f0G48k>` 这种占位符。
于是 RVC 成了五条链里唯一一条每次都要人手拼参数的，而人手拼参数正是
「跑出来的红说不清是闸还是被测对象」的高发区（S129 铁律）。

⛔⛔ **`-g -1` 是【有意为之的 CPU 垫片】，不许「修」它。** 机理（实测自上游源码）：
`train.py:15` 无条件执行 `os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")`
⇒ `"-1"` 变成 `",1"`，是一个非法设备列表 ⇒ CUDA 一张卡都看不见。
`train.py:16` 那个 `n_gpus = len(hps.gpus.split("-"))` 是**死变量**（`main()` 在 :96
用 `torch.cuda.device_count()` 整个覆盖它，:100-103 再兜底成 1），所以它算出 2 没有任何后果。
把 `-g -1` 换成别的值 ⇒ 原版侧跑到 GPU 上 ⇒ 逐 step 对拍的确定性前提当场消失，
而那种红会**伪装成数值不一致**。README:55 的「双方 fp32 CPU（确定性）」与 S37 记的
「30/30 step 对齐」都建立在这一行之上。

harness 补丁（只动执行环境，零数值影响，逐条登记 —— 与另外四条同款纪律）：
  - 绕过 `main()`，直接调 `train.run(0, 1, hps, logger)`。理由不是省事：`main()` 用
    `mp.Process` 派生子进程，而 Windows 是 spawn ⇒ 子进程会重新 import `__main__`，
    而 `__main__` 是这个脚本 ⇒ 无限递归/unpickle 失败（S39 在 diff 那条链上付过这笔学费）。
    `run()` 里每一处 `.cuda()` 都被 `torch.cuda.is_available()` 守着，CPU 分支连 DDP
    都不套（:199-203），所以 world_size=1 直调与上游 CPU 路径逐行同构。
  - `MASTER_ADDR`/`MASTER_PORT`：本来由 `main()` 设（:104-105，端口是 randint），
    绕过它就得自己设；固定端口反而比随机的好复现。
  - `USE_LIBUV=0`：torch>=2.4 在 Windows 上的 TCPStore 需要（与 sovits 那条同款）。
  - `sys.argv`：把 README 那行的占位符固化成真路径，底模用**与我方侧同一份**
    （`gate1_run_ours.py` 的 CFG 指的也是 RVC assets 里那两个文件）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402

RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"
ORIG_EXP = os.path.join(RVC, "logs", "gate1")
PRETRAIN = os.path.join(RVC, "assets", "pretrained_v2")


def _unrunnable(msg):
    """⛔ S139:这三条前置此前是 `sys.exit(<字符串>)` —— CPython 对字符串参数退 **1**,
    而 1 在 orig 段会被跑器读成 `ORIG-FAILED`(参照物没跑起来)。
    可这个脚本自己说的是「这是【闸没准备好】,不是被测对象的问题」——
    **一条红,两种归因**,正是 S129 铁律要拆开的东西。⇒ 退 3(不可归因),
    跑器把它翻成 `VERDICT UNRUNNABLE` / exit 6。"""
    G1._say("GATE1 RVC ORIG: 闸跑不起来 / 读数不可归因(exit %d)" % G1.EXIT_UNRUNNABLE)
    G1._say("  " + msg)
    G1._say("⛔ 这【不是】一次判定。别读成通过,也别读成被测代码红了。")
    sys.exit(G1.EXIT_UNRUNNABLE)

# gloo env:// 所需（本来在 main() 里，我们绕过了它）
os.environ["USE_LIBUV"] = "0"
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "8003")

# ⛔ S139:判据从「目录在不在」换成「目录里有没有上一轮留下的存档」。
#    `os.path.isdir` 在目录**从上一轮活下来**时一声不吭 —— 而上游 train.py:209-218 会
#    `global_step=(epoch_str-1)*len(train_loader)` **续训**,于是「忘了跑 prepare」这件事
#    在原版侧连一句话都不会说,而两侧步集缩水之后 compare 的旧地板(`<10`)也看不见。
#    ⇒ 现在:有 `G_*.pth` 就说明它不是一份干净的 expdir,响亮地判不可归因。
if not os.path.isdir(ORIG_EXP):
    _unrunnable("%s 不存在 —— 先用 run_gate1_chain.py rvc --rebuild-fixtures 铺夹具。" % ORIG_EXP)
_stale = sorted(n for n in os.listdir(ORIG_EXP)
                if n.startswith(("G_", "D_")) and n.endswith(".pth"))
if _stale:
    _unrunnable(
        "%s 里已经有上一轮的存档:%s\n"
        "       ⇒ 上游 train.py:209-218 会从它**续训**(global_step 从上一轮接着走),\n"
        "         而那会让两侧步集静默缩水 —— 「忘了跑 prepare」在原版侧连一句话都不会说。\n"
        "       ⇒ 要么重铺夹具(--rebuild-fixtures),要么先把这些存档挪走。"
        % (ORIG_EXP, _stale[:6]))
for n in ("f0G48k.pth", "f0D48k.pth"):
    if not os.path.isfile(os.path.join(PRETRAIN, n)):
        _unrunnable("缺底模 %s —— 闸的前提不满足" % os.path.join(PRETRAIN, n))

# ⛔ 顺序是承重的:上游 train.py 在 **import 时** 就 get_hparams() 读 argv 并设
#    CUDA_VISIBLE_DEVICES,所以 argv 必须在 import 之前铺好。
sys.argv = [
    os.path.join(RVC, "infer", "modules", "train", "train.py"),
    "-e", "gate1",
    "-sr", "48k",
    "-f0", "1",
    "-bs", "4",
    "-g", "-1",          # ⛔ 见头注:这是 CPU 垫片,不是笔误
    "-te", "2",
    "-se", "1",
    "-pg", os.path.join(PRETRAIN, "f0G48k.pth"),
    "-pd", os.path.join(PRETRAIN, "f0D48k.pth"),
    "-l", "1",
    "-c", "0",
    "-sw", "0",
    "-v", "v2",
]

os.chdir(RVC)  # get_hparams 用 ./logs/<exp>;train.py 顶部还把 cwd 塞进 sys.path
sys.path.insert(0, RVC)

import torch  # noqa: E402

import infer.modules.train.train as train  # noqa: E402  (原版文件,零改动)
from infer.lib.train import utils  # noqa: E402

if torch.cuda.is_available():
    _unrunnable(
        "CUDA 仍然可见(CUDA_VISIBLE_DEVICES=%r)—— `-g -1` 的 CPU 垫片没生效。\n"
        "       两侧必须都在 fp32 CPU 上跑,否则逐 step 对拍的确定性前提不成立,\n"
        "       而那种红会**伪装成数值不一致**(S134 核验者原话)。"
        % os.environ.get("CUDA_VISIBLE_DEVICES"))


def main():
    hps = train.hps
    assert os.path.abspath(hps.model_dir) == os.path.abspath(ORIG_EXP), (
        "model_dir=%s 与预期的 %s 不一致" % (hps.model_dir, ORIG_EXP)
    )
    logger = utils.get_logger(hps.model_dir)
    logger.info("GATE1 RVC ORIG: CPU fp32, CUDA_VISIBLE_DEVICES=%r"
                % os.environ.get("CUDA_VISIBLE_DEVICES"))
    train.run(0, 1, hps, logger)
    print("GATE1 RVC ORIG SIDE DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
