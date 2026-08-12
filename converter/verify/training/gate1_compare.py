# -*- coding: utf-8 -*-
"""关卡1 对拍(RVC):逐 step loss 轨迹 —— 原版 train.py vs 我们 vendored train()。

    training/.venv/Scripts/python.exe converter/verify/training/run_gate1_chain.py rvc

⛔ **别再直接手敲这个脚本**:它现在要求跑器钉的 `GATE1_T0`(新鲜度),没有它会
   响亮地判 exit 3(不可归因)而不是给你一个没有意义的绿。理由见 `gate1_guard` 头注 ⑴。

两侧同数据/同序(1234)/同底模/fp32 CPU(确定性)。原版侧取 tensorboard events(全精度;
stdout 只有 3 位小数),我方侧取协议 JSONL。
⚠ **torch 轴**:文件头此前写着「双方同 torch(2.5.1)」——**那是陈货**。经 `run_gate1_chain.py`
   跑时这条链用的是 `envs\\s42_staging_nv_cu130`(**torch 2.11.0+cu130**,也正好是出货 runtime
   pack 的版本),两侧同一个。S134 §3 就记过这件事,四个月没改到文件头里。
   ⇒ 现在实际版本由 `gate1_guard.header` 打进转录,别再靠这段散文。

结构性移植错误(损失权重/数据顺序/模型接线)会造成 O(0.1~1) 的相对差;
通过线设在 max 相对差 ≤1e-3(实测期望 ~1e-6 级)。

⛔ **夹取(clamp)是这条链特有的一条致盲**,S139 实测,写清楚别当没有:
   原版把 mel>75 / kl>9 夹到上限**之后**才写 TB,所以比较时必须对我方值施加同一夹取。
   后果:凡是**原版侧已顶到上限**的 step,判据退化成 `min(ours, 9.0) vs 9.0`
   ⇒ 我方值只要 ≥9,相对差**恒等于 0**,无论真值是 9.001 还是 1e9(实测:改成 1e9 仍 ALL PASS)。
   而今天盘上的真夹具里 **kl 有 2/30 步顶满,其中包括 step 0** —— 而 step 0 恰恰是
   初始化/底模装载/第一次前向这类错误表现得最赤裸的一步。
   ⛔ **不许「去掉 clamp」**:两个神谕(TB 与 orig stdout)存的都是**夹过的值**,
      去掉会把它变成一条必红的假判据。⇒ 唯一诚实的做法是**逐分量记账**(见下)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402

CHAIN = "rvc"
GATE = "GATE1 (RVC)"
ORIG_TB_DIR = r"D:\MyDev\RVC\RVC20240604Nvidia\logs\gate1"
OURS_JSONL = r"D:\MyDev\TESTING\utai-v2-testing\gate1_ours_steps.jsonl"
# ⛔ S140:身份目录必须是**模块常量**,不是内联字面量 —— 内联的那一份让
#    `gate1_negctl` 的合成夹具够不着它,于是阴性对照跑的时候读的是**真夹具**的身份,
#    而工装与真夹具形状不同这件事本身就是 S140 侦察点名的一条(「健康那条对照臂
#    验的是一个盘上不存在的形状」)。
OURS_EXP = r"D:\MyDev\TESTING\utai-v2-testing\gate1_ours"

PAIRS = [  # (TB tag, ours key, clamp)
    ("loss/g/total", "g_total", None),
    ("loss/d/total", "d_total", None),
    ("loss/g/fm", "fm", None),
    ("loss/g/mel", "mel", 75.0),
    ("loss/g/kl", "kl", 9.0),
]
MAX_REL = 1e-3


def main():
    allow_uncovered = "--allow-uncovered" in sys.argv
    t0 = G1.read_t0(GATE)
    orig_frozen = "orig" in G1.skipped_stages()
    G1.header(GATE, CHAIN, [("orig TB", ORIG_TB_DIR), ("ours JSONL", OURS_JSONL)])
    # ⛔ 「这一轮的输入是哪一次 gate0 产的」—— 由 prepare 记下,这里读出来并**两侧对拍**。
    #    缺席时响亮说明(见 gate1_guard.say_input_identity)。
    G1.say_input_identity([("orig", ORIG_TB_DIR), ("ours", OURS_EXP)])

    orig = G1.tb_scalars(
        "orig/TB", ORIG_TB_DIR, [t for t, _k, _c in PAIRS], t0,
        frozen_why=("--skip-orig:参照侧本轮**故意没有重跑**,按冻结参照记账"
                    if orig_frozen else None))
    if orig_frozen:
        G1.note_uncovered("参照侧未重跑(--skip-orig)",
                          "这一轮只证明了我方侧与**上一次**跑出来的参照一致")

    ours, _nones = G1.jsonl_steps("ours/JSONL", OURS_JSONL, "g_total", t0)

    exp = G1.EXPECT[CHAIN]["steps"]
    steps = G1.require_exact_steps("ours/JSONL", CHAIN, ours, exp,
                                   other=orig["loss/g/total"], other_label="orig/TB")

    # ⛔ None = protocol 的 _clean 把非有限值写成的 ⇒ **发散**,而发散正是这台闸要抓的东西。
    #    ⚠ 发现 None 就**立刻收尾**,不许往下走进算术 —— 否则 `min(b, clamp)` 会拿
    #      float 和 None 比大小,抛 TypeError,被 run() 归到「闸自己炸了」(exit 3),
    #      而真相是**被测的东西发散了**(exit 1)。这条是 gate1_negctl 抓住的。
    failures = G1.require_no_none("ours/JSONL", ours, [k for _t, k, _c in PAIRS])
    if failures:
        G1.finish(GATE, failures, allow_uncovered=allow_uncovered)

    # ⛔ S140:让 `EXPECT[rvc]["components"]` 从一个**零读者的登记数**变成判据 ——
    #    否则从 PAIRS 里删掉一个分量,每行照打 [PASS]、总判照打 ALL PASS,转录上没有任何数字会变。
    G1.require_components(CHAIN, len(PAIRS))

    for tag, key, clamp in PAIRS:
        # ⛔ S140:此前只有 loss/g/total 那一个 tag 的步集被判过(:72-73 的 other=),
        #    另外四个一个都没判 —— 少点是 KeyError 被归成「闸自己炸了」,**多点完全静默**。
        G1.require_same_step_set("orig/TB", orig[tag], steps, tag)
        items, clamped = [], []
        for s in steps:
            a = orig[tag][s]
            b = ours[s][key]
            if clamp is not None and a >= clamp * (1 - 1e-9):
                clamped.append(s)          # 原版顶满 ⇒ 这一步在数值上不可证伪
                continue
            if clamp is not None:
                b = min(b, clamp)
            items.append((s, tag, a, b))
        if clamp is not None:
            # ⛔ 登记式记账,不是 note_uncovered:见 gate1_guard.check_clamped 的头注
            G1.check_clamped("%s(%s)" % (tag, key), clamped, len(steps),
                             G1.EXPECT[CHAIN]["clamped"].get(tag, 0))
        # ⛔ S140:五条 compare 的比较实现收成 `gate1_guard.compare_pairs` 一处。
        #    这条链原本走 numpy(`np.array(rels).max()` 传播 NaN)所以**侥幸**安全,
        #    但它不点名是哪一步非有限,而孪生的 diff / sovits_v2 用纯 python 的
        #    `max()` / `if r > worst` ⇒ **静默丢 NaN**。一个实现,四种写法,收掉。
        r = G1.compare_pairs("ours/JSONL", items, MAX_REL, floor=1e-6,
                             min_cmp=len(steps) - len(clamped))
        ok = not r["failures"]
        G1._say("[%s] %14s vs %7s: max_rel=%.3e @step %s, mean_rel=%.3e  (%d/%d 步可比%s)"
                % ("PASS" if ok else "FAIL", tag, key, r["worst"], r["worst_step"], r["mean"],
                   r["n_cmp"], len(steps),
                   ",%d 步被夹取致盲" % len(clamped) if clamped else ""))
        failures.extend(r["failures"])

    G1.finish(GATE, failures, allow_uncovered=allow_uncovered)


if __name__ == "__main__":
    G1.run(GATE, main)
