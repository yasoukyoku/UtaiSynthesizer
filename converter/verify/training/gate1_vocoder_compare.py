# -*- coding: utf-8 -*-
"""声码器 关卡1 对拍 —— 双侧 TB 标量逐步对拍(S40 建立)。

    training/.venv/Scripts/python.exe converter/verify/training/run_gate1_chain.py vocoder

⛔ **别再直接手敲这个脚本**:它现在要求跑器钉的 `GATE1_T0`(新鲜度)。

判定(真值,S139 从盘上逐个量过,来源 = `vocoder/pipeline.py:577` 的
`max_updates = 2 * total_real` 与驱动的 `total_steps: 15`):
  · **11 个 tag**:9 个 `training/*` + 2 个 `validation/*`,**名字逐个登记在 REQUIRED_TAGS**
  · `training/*` 每分量 **15 点**(步轴 2,4,…,30)
  · `validation/*` 每分量 **4 点**(global 0/10/20/30)
  · 两侧 (key, step) 值 max_rel ≤ 1e-7(TB f32 序列化噪声轴;期望逐位 0.0 —— 同库同版同 RNG 流)

⚠ 这段文字此前写的是「每分量 12 点 / validation ≥3」——**两个数都是陈的**,而
   `gate1_vocoder_prepare.py:16-20` 与 README:466-472 早就为这一组数字专门更正过一次,
   并写下了代价原文:「按这一段的旧文字去核对点数,会拿 24/3 去量一个 30/4 的东西,
   **然后把对的判成错的**。」—— 那次更正到了 README 与 prepare,**没到这个文件自己的头注**,
   而文件头才是红了以后第一个被读的东西。S139 补上。

⛔⛔⛔ **S139 修掉的那条,是这五条链里唯一一条真正的空集假 PASS:**
   原来 `:52-54` 只判「两侧 tag 集合相等」,而两个空集也是相等的 ⇒
   `for tag in sorted(tags_a & tags_b)` **循环零次** ⇒ 循环体内的
   `EXPECT_TRAIN_POINTS` / `EXPECT_VAL_MIN` / 步轴 / max_rel **一条都不求值**
   ⇒ `ok` 停在初值 True ⇒ 打印 `PASS tag sets identical (0 tags)` → `=== gate1_vocoder: PASS ===`
   **退 0**(S139 实测)。
   ⭐ 这正是 S135 在 gate0 钉死的那条:**删掉目录 = 正确地红,清空目录 = 假 PASS,
      两种清法后果相反**;也是 S136(`smoke_aug --only 未知串`)与 S137
      (`--arm v2 --backend sovits` 打 `ALL PASS (1 checks)`)刚买回两次的同一个形状:
      **地板写在循环体内 ⇒ 零轮时没有地板**。本仓连续第三场。
   ⛔ **而且不需要「空」**:两侧各写满 15 点、数值差 1000 倍,只要 tag 前缀不叫
      `training/`/`validation/`(`:52-53` 那个过滤器)就**同样退 0** —— 而两侧吃**同一份**
      `gate_config.yaml`,tag 改名天生对称 ⇒ 这道闸对「**同源同改**」结构上是瞎的,
      而同源同改正是 vendored 移植最常见的形状。
   ⇒ 所以判据不能是「两侧个数一样」,必须是**登记的名单**(REQUIRED_TAGS)。

⚠ 另一条同批修的:此前 `EventAccumulator(str(logdir))` 是五条里**唯一没钉 size_guidance** 的
   ⇒ 走默认(scalars 上限 10000,超了 reservoir **随机抽样**)。今天 15 点用不到,但它是一颗
   「把 gate 放大一点就两侧各自随机丢点」的地雷,而 tag 集合判据对这种丢失是瞎的。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402

CHAIN = "vocoder"
GATE = "GATE1 VOCODER"
GATE_ROOT = r"D:\MyDev\TESTING\gate1_vocoder"
# orig 跑在上游仓库树里(它的 DsModelCheckpoint 断言 cwd 是 work_dir 的祖先,见 run_orig 头注)
ORIG_LOGS = os.path.join(r"D:\MyDev\SingingVocoders\experiments\gate1_voc",
                         "lightning_logs", "lastest")
OURS_LOGS = os.path.join(GATE_ROOT, "ours", "gate1_voc", "lightning_logs", "lastest")
TOL = 1e-7

# ⛔ **登记的名单,不是个数**。S139 从两侧真夹具逐个量出来的 11 个。
#    ⚠ 目录名 `lastest`(拼错的)由我方 `vocoder/pipeline.py:913` 与上游
#      `SingingVocoders/train.py:92` **各自硬编码一遍** ⇒ 一条必须同时改才不红的隐式耦合。
REQUIRED_TAGS = (
    "training/DmpdlossF", "training/DmpdlossT", "training/DmsdlossF", "training/DmsdlossT",
    "training/Gmpd_feature_loss", "training/Gmpdloss",
    "training/Gmsd_feature_loss", "training/Gmsdloss", "training/aux_mel_loss",
    "validation/stft_loss", "validation/total_loss",
)


def main():
    allow_uncovered = "--allow-uncovered" in sys.argv
    t0 = G1.read_t0(GATE)
    orig_frozen = "orig" in G1.skipped_stages()
    G1.header(GATE, CHAIN, [("orig logs", ORIG_LOGS), ("ours logs", OURS_LOGS)])

    exp = G1.EXPECT[CHAIN]
    if len(REQUIRED_TAGS) != exp["tags"]:
        raise G1.GateUnrunnable(
            "闸自己的两份登记对不上:REQUIRED_TAGS 有 %d 个,而 EXPECT[vocoder][tags]=%d"
            % (len(REQUIRED_TAGS), exp["tags"]))

    # tb_scalars 里的 tag 齐全判据吃的就是这份名单 ⇒ 少一个、改一个名字都当场 exit 3
    a = G1.tb_scalars("orig/TB", ORIG_LOGS, REQUIRED_TAGS, t0,
                      frozen_why=("--skip-orig:参照侧本轮**故意没有重跑**,按冻结参照记账"
                                  if orig_frozen else None))
    b = G1.tb_scalars("ours/TB", OURS_LOGS, REQUIRED_TAGS, t0)
    if orig_frozen:
        G1.note_uncovered("参照侧未重跑(--skip-orig)",
                          "这一轮只证明了我方侧与**上一次**跑出来的参照一致")

    failures = []
    G1._say("[COVERAGE] 两侧都带齐了登记的 %d 个 tag(9 training + 2 validation)"
            % len(REQUIRED_TAGS))

    for tag in REQUIRED_TAGS:
        want = exp["train_points"] if tag.startswith("training/") else exp["val_points"]
        pa, pb = a[tag], b[tag]
        if len(pa) != want or len(pb) != want:
            raise G1.GateUnrunnable(
                "%s: 点数 orig=%d / ours=%d,登记的真值是 %d\n"
                "       ⇒ 有一侧没跑完 / interval 配置变了 ⇒ 这一轮不构成一次对拍。"
                % (tag, len(pa), len(pb), want))
        if sorted(pa) != sorted(pb):
            raise G1.GateUnrunnable(
                "%s: 两侧步轴不同 orig=%s ours=%s" % (tag, sorted(pa)[:8], sorted(pb)[:8]))
        worst = 0.0
        for s in sorted(pa):
            va, vb = pa[s], pb[s]
            # 审查修复(S40 起):python max() 会静默丢掉 NaN 操作数 ——
            # 任何一侧的非有限值必须 FAIL,绝不许消失进一个绿色的 max_rel
            if not (math.isfinite(va) and math.isfinite(vb)):
                failures.append("%s@%d 非有限(%r vs %r)" % (tag, s, va, vb))
                continue
            worst = max(worst, abs(va - vb) / max(abs(va), abs(vb), 1e-12))
        ok = worst <= TOL
        G1._say("[%s] %-28s %2d 点, max_rel %.3e <= %.0e"
                % ("PASS" if ok else "FAIL", tag, len(pa), worst, TOL))
        if not ok:
            failures.append(tag)

    G1.finish(GATE, failures, allow_uncovered=allow_uncovered)


if __name__ == "__main__":
    G1.run(GATE, main)
