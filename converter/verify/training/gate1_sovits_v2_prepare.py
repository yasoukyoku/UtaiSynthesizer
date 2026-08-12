"""SoVITS 4.0-v2 关卡1 prepare：双侧工作区就位（training venv）。

前置：gate0_sovits_v2_run_ours.py 已跑完（复用其 dataset_44k 产物 + filelists +
config —— batch 组成由 filelist 行序 + seed 决定）。
  - 原版侧 = <v2 repo>\\logs\\gate1_sovits_v2（train.py 硬编码 ./logs/<name>）
  - 我方侧 = TESTING\\gate1_sovits_v2_ours
两侧各拷入同一对官方底模 G_0/D_0（~1GB/侧）。
gate 配置：epochs=2 / batch=4 / log_interval=1 / eval_interval=1000（唯一命中的
边界 = 上游必然触发的 step-0 evaluate，RNG 流对拍点）/ num_workers=0（双侧）/
fp16_run=False（v2 恒 fp32）。
★ 上游 data_utils 懒生成 `.mel.npy` —— 我方产物名是 `.aam80.npy`（防 diff 池
同名冲突的登记偏差）→ 此处把每个 .aam80.npy 复制为同目录 .mel.npy，让上游直接
命中缓存（两侧消费**逐字节相同**的 mel 文件，mel 生成轴已由关卡0 C4 单独定审）。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_v2_prepare.py
"""
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SOVITS_V2 = r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc"
GATE0_EXP = os.path.join(TESTING, "sovits_v2_ours")
ORIG_EXP = os.path.join(SOVITS_V2, "logs", "gate1_sovits_v2")
OURS_EXP = os.path.join(TESTING, "gate1_sovits_v2_ours")
GATE_CFG = os.path.join(TESTING, "gate1_sovits_v2_config.json")
BASE_DIR = os.path.join(REPO, "data", "models", "training", "sovits_v2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402
# ⚠ `gate1_guard` 在 import 时把 stdout **与 stderr** 都钉成 UTF-8;原来这里只钉了 stdout,
#   而这个文件的断言(中文)走的是 stderr —— 本机 locale 是 cp932。


def main():
    with open(os.path.join(GATE0_EXP, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["train"]["fp16_run"] is False
    assert cfg["train"]["epochs"] == 2 and cfg["train"]["batch_size"] == 4
    assert cfg["train"]["num_workers"] == 0, "gate 需要双侧 num_workers=0"
    cfg["train"]["log_interval"] = 1
    cfg["train"]["eval_interval"] = 1000

    # filelist sanity: absolute paths must exist (batch composition = line order + seed)
    # ⛔ S140:裸 `assert` 走 `G1.run` 的 `except Exception` 兜底 ⇒ 退 3 但措辞是
    #    「闸自己炸了 —— 下面是转录」,而它说的其实是「输入不合法 / gate0 没跑过」。
    #    一条红两种归因,正是 S129 铁律要拆开的。同批里 rvc 与 diff 早就换成 GateUnrunnable 了。
    for key in ("training_filelist", "validation_filelist"):
        with open(cfg["data"][key], encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p and not os.path.exists(p):
                    raise G1.GateUnrunnable(
                        "gate1/sovits_v2:filelist(%s)里的路径不在盘上:%s\n"
                        "       ⇒ gate0(v2)的产物缺件 ⇒ 这一轮不构成一次对拍。" % (key, p))

    # ⛔⛔ S140:底模断言与输入身份**必须排在下面那段 33 次 copyfile 之前**。
    #    S139 这一笔的主题就是「全部前置断言前移到第一句破坏之前」,而这个文件**没做干净**:
    #    原顺序是 写 GATE_CFG → 33 次 copyfile 写进 **gate0 的池** → 算身份 → 才查底模
    #    ⇒ 底模缺席时判 UNRUNNABLE,而 **gate0 的池已经被改过了**(纯新增,而 S137 刚买回
    #      「`assert_pool_intact` 从不算新增的文件」⇒ gate0 侧的完整性判据对它结构上是瞎的)。
    for n in ("G_0.pth", "D_0.pth"):
        if not os.path.isfile(os.path.join(BASE_DIR, n)):
            raise G1.GateUnrunnable("底模缺 %s:%s" % (n, BASE_DIR))

    spk_dir = os.path.join(GATE0_EXP, "dataset_44k", "gate")
    # ⛔ S140 输入身份 —— **算在我们自己往这棵树里写 .mel.npy 之前**,而且用后缀白名单
    #    把 `.mel.npy` 排除在外。两条理由:
    #    ⑴ 算在写之后 ⇒ 身份描述的是「一棵被本次 prepare 改过的树」,它答不了
    #       「这是哪一棵 gate0 树」这个它自称要答的问题;
    #    ⑵ `.mel.npy` 是 prepare 自己造的**副本**,不是 gate0 的产物 ⇒ 把它算进去会让
    #       同一棵 gate0 树在「第一次跑」与「第二次跑」得到两个不同的 sha(132 vs 165)。
    #    真值(2026-08-12 实测):33 .wav + 33 .aam80.npy + 33 .f0.npy + 33 .soft.pt = 132。
    ID_SUFFIXES = (".wav", ".aam80.npy", ".f0.npy", ".soft.pt")
    ident = G1.src_identity("gate1/sovits_v2 的输入(gate0 v2 的 dataset_44k/gate,不含我们造的 .mel.npy)",
                            spk_dir, [""], min_files=132, suffixes=list(ID_SUFFIXES))

    with open(GATE_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # upstream lazy-mel cache: duplicate .aam80.npy -> .mel.npy (see header)
    #
    # ⛔ S134 (§F7 first pass) — this used to be guarded by `if not os.path.exists(dst)`, and that
    # guard could turn this file's own promise ("both sides consume a BYTE-IDENTICAL mel", header)
    # into a lie on the second run: our side eats .aam80.npy, upstream eats .mel.npy, and a
    # surviving stale .mel.npy from an earlier gate0 means the two sides are comparing different
    # mel. It never said a word either — the second run just printed "duplicated 0".
    # The copy is cheap (33 files) and this is a prepare step, so: copy UNCONDITIONALLY and report
    # fresh vs overwritten separately, so "0 new / 33 refreshed" reads as the normal steady state
    # and "0 / 0" reads as "gate0 has not run" instead of hiding under the same number.
    fresh = refreshed = 0
    for n in os.listdir(spk_dir):
        if n.endswith(".aam80.npy"):
            dst = os.path.join(spk_dir, n.replace(".aam80.npy", ".mel.npy"))
            if os.path.exists(dst):
                refreshed += 1
            else:
                fresh += 1
            shutil.copyfile(os.path.join(spk_dir, n), dst)
    if fresh + refreshed == 0:
        raise G1.GateUnrunnable(
            "gate1/sovits_v2:%s 下一个 .aam80.npy 都没有 ⇒ gate0(v2)什么也没产出,\n"
            "       而上游的 lazy-mel 缓存就会是更早一轮留下的任何东西。" % spk_dir)
    # ⚠ S140 预期读数:今天该目录 `.mel.npy` = **0 个**(08-11 17:53 gate0 v2 重跑把上一轮的清了)
    #    ⇒ 这一跑应当打 **33 new, 0 refreshed**。打出 refreshed>0 说明有一轮没清干净的陈 mel,
    #    当场停下来查(S134 那次打的是 0 new / 33 refreshed,那是它自己上一跑留下的)。
    G1._say("aam80 -> mel.npy for the upstream lazy cache: %d new, %d refreshed" % (fresh, refreshed))

    for exp in (ORIG_EXP, OURS_EXP):
        if os.path.isdir(exp):
            shutil.rmtree(exp)
        os.makedirs(exp, exist_ok=True)
        for n in ("G_0.pth", "D_0.pth"):
            shutil.copyfile(os.path.join(BASE_DIR, n), os.path.join(exp, n))
        G1.write_input_identity(exp, ident)
    # ours reads config from its exp dir (train() contract)
    shutil.copyfile(GATE_CFG, os.path.join(OURS_EXP, "config.json"))

    print("GATE1 SOVITS_V2 PREPARE DONE")
    print("  orig exp:", ORIG_EXP)
    print("  ours exp:", OURS_EXP)


if __name__ == "__main__":
    G1.run("gate1_sovits_v2_prepare", main)
