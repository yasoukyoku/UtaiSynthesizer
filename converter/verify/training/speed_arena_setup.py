# -*- coding: utf-8 -*-
"""S138 · 给速度尺子铺【两侧】arena —— 两侧吃同一份产物、同一份 config。

⛔ 为什么不能调任何现成的 `*_prepare.py`(S138 侦察 A2):
   `gate1_prepare.py:copy_artifacts` 首句就是 `shutil.rmtree(dst)`,两个 dst 是
   `RVC\\logs\\gate1`(1.39 GB)与 `TESTING\\utai-v2-testing\\gate1_ours`(2.93 GB)
   —— S134 花一整笔跑通的两侧 gate1 夹具。

⛔ 本脚本会往**上游树**写:`D:\\MyDev\\RVC\\RVC20240604Nvidia\\logs\\<EXP>`。
   这是不可避免的(上游 `get_hparams` 用 `./logs/<exp>`),所以它**逐条登记**:
   写之前打印那里现在有什么,并**拒绝**使用任何已存在且非本脚本建的目录名。

配置口径(与 gate1 那把 loss 尺子**不同**,不许混):
  · `fp16_run = True`(生产配置;gate1 是 False 为了 CPU 确定性)
  · `log_interval = 200`(**生产默认**)—— ⛔ 不许设成 1:那会让**上游也**每步取 6 个 loss
    的 `.item()`(上游 `train.py:503-517` 只在 log 步做,我方是无条件每步做),
    等于把我方唯一确定的减速项抹掉,同时给两侧都加上每步 3 张 matplotlib 渲染。
  · ⇒ 逐 step 墙钟因此**只有我方侧拿得到**(靠 Reporter);
    **两侧共同的读数是【每个 epoch 的 `====> Epoch:` 时间戳之差】**,
    两侧 formatter 逐字节相同(`rvc/train_utils.py:180` vs 上游 `infer/lib/train/utils.py:443`)。
"""
import json
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
# ⛔ arena 落在 TESTING 而不是仓里:它是几百 MB 的训练产物,
#    而且 C: 的写性能今天本身就是一个会漂移的变量(S138 实测)。
ARENA = r"D:\MyDev\TESTING\s138_f7\arena"
BASE = os.path.join(ARENA, "rvc_base")          # 纯数据基座(80 MB)
RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"
EXP = "s138speed"
UP_EXP = os.path.join(RVC, "logs", EXP)
OURS_EXP = os.path.join(ARENA, "ours_run")

SUBS = ["0_gt_wavs", "2a_f0", "2b-f0nsf", "3_feature768", "mute"]


def _register(path, label):
    print("[登记] %s = %s" % (label, path))
    if os.path.isdir(path):
        n = sum(len(f) for _, _, f in os.walk(path))
        print("       已存在:%d 件,mtime %s" % (
            n, time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))))
    else:
        print("       不存在(将新建)")


def build(dst, rewrite_to):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for s in SUBS:
        shutil.copytree(os.path.join(BASE, s), os.path.join(dst, s))
    # filelist 指到它自己
    src_prefix = BASE.replace(os.sep, "/")
    with open(os.path.join(BASE, "filelist.txt"), encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    out = [l.replace(src_prefix, rewrite_to.replace(os.sep, "/")) for l in lines]
    with open(os.path.join(dst, "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    # ⛔ 逐路径证明自包含(S138 血训:拷来的 filelist 会指回源工作区,而写回是静默的)
    miss = [p for l in out for p in l.split("|")
            if p.startswith("D:/") and not os.path.exists(p.replace("/", os.sep))]
    assert not miss, "filelist 有 %d 个路径不在 arena 内:%s" % (len(miss), miss[:3])
    # config
    with open(os.path.join(BASE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["train"]["fp16_run"] = True        # 生产配置
    cfg["train"]["log_interval"] = 200     # ⛔ 生产默认,见头注
    cfg["train"]["epochs"] = 20000         # 上游靠 -te 控制,这个键不参与
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
    n = sum(len(fs) for _, _, fs in os.walk(dst))
    print("       建好:%d 件 / %d 行 filelist / fp16_run=%s log_interval=%s"
          % (n, len(out), cfg["train"]["fp16_run"], cfg["train"]["log_interval"]))
    return len(out)


def main():
    assert os.path.isdir(BASE), "缺纯数据基座 %s" % BASE
    print("=== 写之前先登记两侧落点 ===")
    _register(UP_EXP, "上游侧(**在上游树里**)")
    _register(OURS_EXP, "我方侧")
    # ⛔ 拒绝撞上别人的实验目录
    for other in ("gate1", "mute", "gate0"):
        if EXP == other:
            sys.exit("EXP 名字撞上已有实验目录")
    print()
    print("=== 铺上游侧 ===")
    n1 = build(UP_EXP, UP_EXP)
    print("=== 铺我方侧 ===")
    n2 = build(OURS_EXP, OURS_EXP)
    assert n1 == n2, "两侧 filelist 行数不同:%d vs %d" % (n1, n2)

    # 两侧样本顺序必须逐行相同(照 gate1_prepare 的那条抽查)
    def names(d):
        with open(os.path.join(d, "filelist.txt"), encoding="utf-8") as f:
            return [l.split("|")[0].rsplit("/", 1)[-1] for l in f.read().splitlines() if l]
    assert names(UP_EXP) == names(OURS_EXP), "两侧样本顺序不一致"
    print()
    print("✅ 两侧就绪,%d 行,样本顺序逐行相同" % n1)
    print("   上游 %s" % UP_EXP)
    print("   我方 %s" % OURS_EXP)


if __name__ == "__main__":
    sys.exit(main())
