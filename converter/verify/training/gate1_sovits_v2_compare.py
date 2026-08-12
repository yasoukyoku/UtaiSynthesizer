# -*- coding: utf-8 -*-
"""SoVITS 4.0-v2 关卡1 对拍:原版 tensorboard events vs 我方 JSONL 步流。

    training/.venv/Scripts/python.exe converter/verify/training/run_gate1_chain.py sovits_v2

九分量(v2 的 TB tag → 我方 losses 键),无 clamp(上游 TB 写原始值)。
结构性移植错误 = O(0.1~1) 的 rel;期望 ~1e-6 级(同 torch/CPU/fp32/seed/RNG 流)。
⚠ 这条链是五条里**唯一**用 `training\\.venv`(torch 2.5.1)的 —— 它的历史基线是那个解释器
   产的,所以它的文件头此前**没有**陈货问题。实际版本仍由 header 打进转录。

⛔⛔ **S139 修掉的那条,是这五条链里最贵的一条**:
   原来筛步用的是 `losses.get("g_total") is not None` ——
   **值为 None 的那一整步根本不进 `ours`**,于是:
     · 下面那条「缺分量 ⇒ FAIL」的分支**永远到不了**(实测);
     · 而 `protocol.py` 的 `_clean` 正是把**非有限值(nan/inf)写成 None** 的
       ⇒ **发散的那几步整个从判据里消失**;
     · 剩下的步照样对得上 ⇒ 打印 `aligned: 12` 并 **[PASS] 退 0**。
   实测两条:g_total 单独 null@step9 ⇒ `aligned: 13` [PASS];
             九个分量**全** null@steps 9,10(真发散形状)⇒ `aligned: 12` [PASS]。
   ⚠ 而 **NaN 正是这台闸存在的理由**(§F5「续训崩溃」那条未结案卷宗、`gate_numerics_guard`
      的全部立项理由都是它)。
   ⇒ 现在:筛步一律用「**键在不在**」(`gate1_guard.jsonl_steps`),
      值为 None ⇒ `require_no_none` **立刻红并点名 step/分量**;
      步数不足则由 `require_exact_steps` 判 exit 3(闸没跑成),不是 exit 1。
   ⛔ 旧的「对齐步数 < 10 ⇒ FAIL」也一起换掉:那个 10 与这条链的真值(**14**)无关,
      丢 4 步仍然打 [PASS]。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402

CHAIN = "sovits_v2"
GATE = "GATE1 SOVITS_V2"
SOVITS_V2 = r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc"
ORIG_TB_DIR = os.path.join(SOVITS_V2, "logs", "gate1_sovits_v2")
OURS_JSONL = r"D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_v2_ours_steps.jsonl"
# ⛔ S140:见 gate1_compare.py 同名常量的注释(内联字面量让阴性对照够不着它)
OURS_EXP = r"D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_v2_ours"

PAIRS = [
    ("loss/total", "g_total"),
    ("loss/mel", "mel"),
    ("loss/adv", "adv"),
    ("loss/fm", "fm"),
    ("loss/mel_ddsp", "mel_ddsp"),
    ("loss/spec_ddsp", "spec_ddsp"),
    ("loss/mel_am", "mel_am"),
    ("loss/kl_div", "kl"),
    ("loss/lf0", "lf0"),
]
MAX_REL = 1e-3


def main():
    allow_uncovered = "--allow-uncovered" in sys.argv
    t0 = G1.read_t0(GATE)
    orig_frozen = "orig" in G1.skipped_stages()
    G1.header(GATE, CHAIN, [("orig TB", ORIG_TB_DIR), ("ours JSONL", OURS_JSONL)])
    G1.say_input_identity([("orig", ORIG_TB_DIR), ("ours", OURS_EXP)])

    orig = G1.tb_scalars(
        "orig/TB", ORIG_TB_DIR, [t for t, _k in PAIRS], t0,
        frozen_why=("--skip-orig:参照侧本轮**故意没有重跑**,按冻结参照记账"
                    if orig_frozen else None))
    if orig_frozen:
        G1.note_uncovered("参照侧未重跑(--skip-orig)",
                          "这一轮只证明了我方侧与**上一次**跑出来的参照一致")

    ours, _nones = G1.jsonl_steps("ours/JSONL", OURS_JSONL, "g_total", t0)
    exp = G1.EXPECT[CHAIN]["steps"]
    steps = G1.require_exact_steps("ours/JSONL", CHAIN, ours, exp,
                                   other=orig["loss/total"], other_label="orig/TB")
    # ⛔ 发现 None 就**立刻收尾**,不许往下走进算术(否则 `a - b` 拿 float 减 None,
    #    抛 TypeError 被归到「闸自己炸了」exit 3,而真相是**发散了** exit 1)。
    failures = G1.require_no_none("ours/JSONL", ours, [k for _t, k in PAIRS])
    if failures:
        G1.finish(GATE, failures, allow_uncovered=allow_uncovered)

    # ⛔ 分量缺席(键根本不在)与分量为 None(发散)是**两件事**,要分开报:
    #    前者是「我方侧没发出这个字段」= 闸没跑成;后者是真的红。
    missing = [(s, k) for s in steps for _t, k in PAIRS if k not in ours[s]]
    if missing:
        raise G1.GateUnrunnable(
            "ours/JSONL: %d 处**缺分量**(键根本不在,不是 None):%s\n"
            "       ⇒ 我方侧没有发出该发的字段 ⇒ 这一轮不构成一次对拍。"
            % (len(missing), missing[:8]))

    # ⛔ S140:登记的分量数变成判据(此前 EXPECT[sovits_v2][components]=9 零读者)。
    #    ⚠ 顺带钉住一个口径:我方 jsonl 每步发 **10** 个键,而这里比 **9** 个 ——
    #      不比的那个是 `d_total`。「14 步 × 9 分量」这句话里**不含判别器总损失**。
    G1.require_components(CHAIN, len(PAIRS))
    for tag, _k in PAIRS:
        G1.require_same_step_set("orig/TB", orig[tag], steps, tag)

    # ⛔⛔ S140:原来这里是 `worst=(0.0,"",-1)` + `if rel > worst[0]` 的**滚动比较**,
    #    而 NaN 的一切比较都是 False ⇒ **无论 NaN 落在哪一步都被丢掉**。
    #    实测:九个分量在所有步全是 NaN 时,它打 `max_rel=0.000e+00 ( @ step -1)` 并 **PASS**
    #    —— 而参照(TB)那一侧**没有任何有限性判据**(`require_no_none` 只吃我方 JSONL),
    #    所以这条路今天就是活的。⇒ 全部改走 `gate1_guard.compare_pairs`(否定式 + 点名)。
    items = [(s, tag, orig[tag][s], ours[s][key]) for s in steps for tag, key in PAIRS]
    r = G1.compare_pairs("ours/JSONL", items, MAX_REL, floor=1e-6,
                         min_cmp=len(steps) * len(PAIRS))
    ok = not r["failures"]
    failures.extend(r["failures"])
    G1._say("[%s] %s: %d 步 × %d 分量 = %d 对真比过, max_rel=%.3e (%s @ step %s), 线=%.0e"
            % ("PASS" if ok else "FAIL", GATE, len(steps), len(PAIRS), r["n_cmp"],
               r["worst"], r["worst_tag"], r["worst_step"], MAX_REL))

    G1.finish(GATE, failures, allow_uncovered=allow_uncovered)


if __name__ == "__main__":
    G1.run(GATE, main)
