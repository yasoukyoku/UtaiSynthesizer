# -*- coding: utf-8 -*-
"""浅扩散 关卡1 对拍 —— 两侧 tensorboard events 的逐步 loss 轨迹(全精度;stdout 只有 3 位小数)。

    training/.venv/Scripts/python.exe converter/verify/training/run_gate1_chain.py diff

⛔ **别再直接手敲这个脚本**:它现在要求跑器钉的 `GATE1_T0`(新鲜度)。

系列:`train/loss`(每步,interval_log=1)+ `validation/loss`(step 8/16/24 ——
第一个边界里含 NSF-HiFiGAN Generator 的惰性构造那一段 RNG,能对齐**过**它就证明
RNG 消耗模型是完整的)。

⛔⛔ **S139 修掉的:`train/loss` 这一半此前是【空集恒真】的,而它是这条链的主判据。**
   `load_scalars` 的 `if tag in ea.Tags()['scalars']` 让 `train/loss` 可以**单独缺席**;
   缺席时 `set(a) != set(b)` 对两个空集为**假** ⇒ 落 else ⇒ `max(..., default=0.0)`
   给出 0.0 ⇒ 打印 `train/loss: 0 steps aligned, max_rel 0.000e+00` 且 `ok` 保持 True。
   ⚠ 交接把它记成「两侧同空 ⇒ PASS」,**那半句是错的**:两侧全空时下面
   `validation` 的 `len(inter) < 2` 会把它救回 FAIL(实测)。
   **真正的洞更窄也更难看见**:`validation` 两侧齐全而 `train/loss` 两侧缺席
   ⇒ 实测 `GATE1 DIFF: PASS` **退 0**,而转录与健康跑只差一个数字(0 vs 24 steps aligned)。
   造出它最普通的原因就是 `interval_log` 配错 —— 而 `gate1_vocoder_prepare.py:14`
   亲口把这个场景登记成「红队 A11:默认 100 下 24 global 步只有 step0 一个点 = 空交集假 PASS」。
   **同一个已知威胁,vocoder 那条防了,diff 这条压根没防。**
   ⇒ 现在:`train/loss` 两侧都必须**恰好 24 步**(= `gate1_diff_prepare.py:7-11` 的
   `ceil(31/4)*3` 硬算出来的真值),不足判 exit 3。

⛔ **兜底神谕 `gate1_diff_orig_stdout.log` 是一份【七月】的文件,而两侧 TB 是八月的。**
   全仓**没有任何写入者**(唯一命中就是下面读它那一行);照 README 的命令跑也产不出它
   (原版驱动的 loguru 桩写 stderr,跑器也不重定向它)。而按本文件原来的设计,
   它坐在**正常路径**上(orig 的最后一个 validation 标量常卡在未 flush 的缓冲里,S39 实测)。
   今天它是休眠的(两侧 val 都是 [8,16,24] ⇒ missing 为空),而且陈值恰好还对得上 ——
   **「陈货恰好还对得上,所以没人发现」正是 gate0_guard 头注记着实名受害者的那个形状**。
   ⇒ 现在:真要用它的时候,先判它的新鲜度;是陈的就 **exit 3**,不许拿七月的 3 位小数
   去核八月的读数,更不许让那种红被跑器贴上「★ 这一种才是被测的东西不对」。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402

CHAIN = "diff"
GATE = "GATE1 DIFF"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
ORIG_DIR = os.path.join(TESTING, "gate1_diff_orig", "logs")
OURS_DIR = os.path.join(TESTING, "gate1_diff_ours", "logs")
ORIG_STDOUT = os.path.join(TESTING, "gate1_diff_orig_stdout.log")
REL_LINE = 1e-6


def stdout_val_losses():
    """orig stdout 里 `--- <validation> ---\\nloss: X.XXX.` 的成对值(3 位小数)——
    TB 缓冲丢点时的兜底神谕。⛔ 见头注:用它之前必须先判它是不是本轮的。"""
    vals = []
    with open(ORIG_STDOUT, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if "<validation>" in line and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt.startswith("loss:"):
                vals.append(float(nxt[len("loss:"):].strip().rstrip(".")))
    return vals


def main():
    allow_uncovered = "--allow-uncovered" in sys.argv
    t0 = G1.read_t0(GATE)
    orig_frozen = "orig" in G1.skipped_stages()
    G1.header(GATE, CHAIN, [("orig logs", ORIG_DIR), ("ours logs", OURS_DIR)])

    tags = ["train/loss", "validation/loss"]
    orig = G1.tb_scalars(
        "orig/TB", ORIG_DIR, tags, t0,
        frozen_why=("--skip-orig:参照侧本轮**故意没有重跑**,按冻结参照记账"
                    if orig_frozen else None))
    ours = G1.tb_scalars("ours/TB", OURS_DIR, tags, t0)
    if orig_frozen:
        G1.note_uncovered("参照侧未重跑(--skip-orig)",
                          "这一轮只证明了我方侧与**上一次**跑出来的参照一致")

    failures = []
    exp = G1.EXPECT[CHAIN]

    # ── train/loss:这条链的主判据。⛔ 两侧都要**恰好** exp["steps"] 步,且步集相同。
    a, b = orig["train/loss"], ours["train/loss"]
    G1.require_exact_steps("orig/train_loss", CHAIN, a, exp["steps"])
    steps = G1.require_exact_steps("ours/train_loss", CHAIN, b, exp["steps"],
                                   other=a, other_label="orig/train_loss")
    worst = max(abs(a[s] - b[s]) / max(abs(a[s]), 1e-12) for s in steps)
    G1._say("[%s] train/loss: %d 步对齐, max_rel %.3e (线 %.0e)"
            % ("PASS" if worst <= REL_LINE else "FAIL", len(steps), worst, REL_LINE))
    if worst > REL_LINE:
        failures.append("train/loss")

    # ── validation/loss:我方侧要**恰好** exp["val_boundaries"] 个边界;
    #    orig 允许少最后一个(它从不关 SummaryWriter,最后一个标量会卡在未 flush 的缓冲里,
    #    S39 实测:orig TB 只有 [8,16] 而它的 stdout 打了三个)。
    va, vb = orig["validation/loss"], ours["validation/loss"]
    G1._say("           validation/loss: orig 步 %s, ours 步 %s" % (sorted(va), sorted(vb)))
    G1.require_exact_steps("ours/validation", CHAIN, vb, exp["val_boundaries"])
    if not set(va) <= set(vb):
        raise G1.GateUnrunnable(
            "orig 的 validation 步 %s 不是 ours %s 的子集 ⇒ 两侧对不齐,不构成对拍"
            % (sorted(va), sorted(vb)))
    inter = sorted(set(va) & set(vb))
    if len(inter) < 2:
        raise G1.GateUnrunnable(
            "validation 只有 %d 个共同边界(要 ≥2 才能证明对齐【过】了第一个 RNG 边界)" % len(inter))
    vworst = max(abs(va[s] - vb[s]) / max(abs(va[s]), 1e-12) for s in inter)
    G1._say("[%s] validation/loss: %d 个共同边界, max_rel %.3e"
            % ("PASS" if vworst <= REL_LINE else "FAIL", len(inter), vworst))
    if vworst > REL_LINE:
        failures.append("validation/loss")

    # ── 兜底神谕:只有真的缺点时才动它,而动它之前先判它是不是本轮的
    missing = sorted(set(vb) - set(va))
    if missing:
        if not os.path.isfile(ORIG_STDOUT):
            raise G1.GateUnrunnable(
                "orig 的 TB 缺了 validation 步 %s,而兜底神谕不在:%s\n"
                "       ⛔ 全仓没有任何脚本写这个文件(原版驱动的 loguru 桩写 stderr)⇒ 它只能是手工留下的。"
                % (missing, ORIG_STDOUT))
        if os.path.getmtime(ORIG_STDOUT) < t0:
            raise G1.GateUnrunnable(
                "orig 的 TB 缺了 validation 步 %s,而兜底神谕**不是本轮的**"
                "(%s,t0 之前)⇒ 用一份旧的 3 位小数去核本轮读数,得出的红说不清是谁的问题。\n"
                "       ⛔ 别把它读成「被测的东西不对」。"
                % (missing, __import__("time").strftime(
                    "%Y-%m-%dT%H:%M:%S", __import__("time").localtime(
                        os.path.getmtime(ORIG_STDOUT)))))
        vals = stdout_val_losses()
        if len(vals) != len(vb):
            raise G1.GateUnrunnable(
                "兜底神谕打了 %d 个 validation 值,而 ours 有 %d 个 ⇒ 对不上,没法交叉核"
                % (len(vals), len(vb)))
        by_step = dict(zip(sorted(vb), vals))
        for s in missing:
            d = abs(by_step[s] - vb[s])
            ok = d <= 5e-4                       # 3 位小数的打印精度
            G1._say("[%s] validation step %d(orig TB 缓冲丢点):stdout %.3f vs ours %.6f (|d|=%.2e)"
                    % ("PASS" if ok else "FAIL", s, by_step[s], vb[s], d))
            if not ok:
                failures.append("validation@%d(兜底神谕)" % s)

    G1.finish(GATE, failures, allow_uncovered=allow_uncovered)


if __name__ == "__main__":
    G1.run(GATE, main)
