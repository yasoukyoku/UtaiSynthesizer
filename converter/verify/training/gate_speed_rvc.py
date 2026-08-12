# -*- coding: utf-8 -*-
"""§F7 笔 D · RVC 链的速度闸 —— 我方 vs 上游,**同会话内交替配对**。

回答的是用户 2026-08-07 那句:「按我们现在这一堆优化,**我们只可能比原包快不可能比原包慢**」。

================================ 为什么长成这样 ================================
S138 在这台机器上实测出三条**会凭空造出「速度差」**的状态,全部与被测代码无关:

 ⑴ **冷卡**:GPU 长时间空闲后的第一次训练只有 **780 MHz**(不是 2010),step 慢 **1.85 倍**;
    温度 51°C、功耗 88 W ⇒ **不是节流,是没升频**。
    ⛔ **开跑前的 preflight 结构上看不见它**(跑之前卡本来就是 idle 档)
    ⇒ 必须做成**运行中不变量**:时钟低于地板 ⇒ 该次**判不可归因并剔除**,不是判红。
 ⑵ **会话内系统漂移**:同一批 A/A 里后半比前半快 **3.19%**,而被测效应实测也是 **≈3%**
    ⇒ **「先跑 A 三遍再跑 B 三遍」会凭空产出一个正确量级、正确方向、完全合法的结论。**
    ⇒ 必须 **A/B/A/B 交替**,统计量是**对内比值**。
 ⑶ **写盘的持久档位迁移**:纯计算 spread 1.1-1.45%,一叠写盘就变成稳态 8-10%、瞬态 45-56%,
    而且是**档位迁移不是随机抖动**。而每个 epoch 边界两侧都要写数百 MB。

以及一条统计量上的:交接原本给的 `tol = max(3×(max−min)/median, 3%)` 用的是**极差**,
同一批数据它给 5.53% 而真实相对标准差只有 2.12% ⇒ **把噪声高估 2.6 倍**,
3σ 分辨力从 **0.80%** 掉到 **16.6%** —— 而靶子是 3%。
⇒ **一把那样的尺子会对着它要防的东西打绿,而每一步都合规。**
⇒ 本闸用**配对比值的均值 ± SEM**,并配一条硬天花板。

================================ 读数取自哪里 ================================
**两侧共同的读数 = 每个 epoch 的 `====> Epoch:` 日志行时间戳之差。**
两侧 formatter **逐字节相同**(我方 `rvc/train_utils.py:180` vs 上游 `infer/lib/train/utils.py:443`),
⇒ 它是**被测代码自己写下、这个闸改不动**的外部记录。
⛔ 不用 `EpochRecorder` 自报的那个括号里的值:两侧都在 batch 循环**之前**构造它
   ⇒ 它**不含 epoch 边界**,而边界正是我方已知更贵的那一段。
⛔ **不许把 `log_interval` 设成 1** 去换逐 step 读数:那会让**上游也**每步取 6 个 loss 的
   `.item()`(上游只在 log 步做,我方无条件每步做)⇒ 抹掉我方唯一确定的减速项;
   而且给两侧每步都加 3 张 matplotlib 渲染(实测 **0.22 s/步 = 40%**)。

出口码(与 `gate0_guard.py` 同一套):
  0  PASS   1  真红(我方更慢且超过门限)   3  不可归因   4  自检失败

================================ 实测过的三条出口 ================================
* **PASS**:真实代码,对内比值 0.9559 / 0.9613 / 0.9527 ⇒ 我方快 **4.34%**,3σ 门限 0.75%。
* **RED**:每步注射 60 ms(`--inject-ms 60`)⇒ 1.0707 / 1.0715 / 1.0713 ⇒ 我方慢 **7.12%**。
* **UNRUNNABLE**:三种都真发生过 ——
  ⑴ 上游侧只有 `gpu_after` 那一版,三对全被时钟不变量剔掉(**闸对,探针错**);
  ⑵ 不变量写成 `min(clocks)` 那一版,把我方 epoch 边界的**正常掉档**当成冷卡
     (⛔ 而那**只会剔掉我方那条臂** —— 偏倚不是噪声);
  ⑶ ⭐ 一次真跑里上游侧整跑慢 30%(10.6-11.7 s vs 8.3),**GPU 时钟全程 2025**
     ⇒ 不是显卡,是**写盘瞬态**(两侧每跑各写 ≈1.2 GB 存档,而这台机器写盘会进持久慢档)。
     那一跑的比值是 0.7281,若无天花板,闸会报「我方快 12.39%」并**打绿**。

⛔ **诚实边界(实测,别当成没有)**:在 `-se 1`(每 epoch 存档)下,这把尺子是
**写盘噪声受限**的 —— 约三对里可能撞上一次污染。**天花板会正确地拒绝**,但那意味着
需要重跑。⇒ 下一步二选一:**加对数**(SEM ∝ 1/√N),或做一个**低写盘变体**
(两侧同时把存档间隔调大)—— ⚠ 后者会同时抹掉「我方 epoch 边界更贵」这条真信号,
所以它只能是**另一条**读数,不是这条的替代。
"""
import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import math

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
# ⛔ arena 落在 TESTING 而不是仓里:它是几百 MB 的训练产物,
#    而且 C: 的写性能今天本身就是一个会漂移的变量(S138 实测)。
ARENA = r"D:\MyDev\TESTING\s138_f7\arena"
RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"
EXP = "s138speed"
UP_EXP = os.path.join(RVC, "logs", EXP)
OURS_EXP = os.path.join(ARENA, "ours_run")

EXIT_PASS, EXIT_RED, EXIT_UNRUNNABLE, EXIT_SELFTEST = 0, 1, 3, 4

CLOCK_FLOOR_MHZ = 1500        # 实测:冷卡 780,热卡 1935-2025
WARM_SKIP_S = 20.0            # 起进程 + 建模 + 载底模那段不算(那时卡本来就没升频)
SAMPLE_EVERY_S = 5.0
CEILING = 0.08                # ⛔ 3×SEM 超过它 ⇒ 本轮没有分辨力,判不可归因
DROP_EPOCHS = 1               # 第一个 epoch 含暖机 + 底模加载,两侧都丢
MIN_EPOCH_SAMPLES = 2
MIN_PAIRS = 3

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\t")
EPOCH_LINE = re.compile(r"====> Epoch:\s*(\d+)")
CKPT_PAT = ("G_", "D_")


class Unrunnable(RuntimeError):
    """这一轮的读数不可归因。⛔ 绝不许被读成『通过』,也不许被读成『慢了』。"""


# --------------------------------------------------------------- 读数解析
def epoch_marks(log_path):
    """从 train.log 解出 [(时刻, epoch)] —— 两侧同一个解析器。"""
    out = []
    if not os.path.isfile(log_path):
        return out
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m, e = TS.match(line), EPOCH_LINE.search(line)
            if m and e:
                out.append((datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f"),
                            int(e.group(1))))
    return out


def epoch_times(log_path, side):
    marks = epoch_marks(log_path)
    if len(marks) < DROP_EPOCHS + 1 + MIN_EPOCH_SAMPLES:
        raise Unrunnable(
            "%s 侧只解出 %d 条 `====> Epoch:` 行(至少要 %d)⇒ 这一跑没跑起来或日志没落地。\n"
            "       日志:%s" % (side, len(marks), DROP_EPOCHS + 1 + MIN_EPOCH_SAMPLES, log_path))
    # 相邻两条之差 = 完整一个 epoch(含边界:存档 + 我方的全张量 isfinite 扫描)
    dts = [(marks[i][0] - marks[i - 1][0]).total_seconds() for i in range(1, len(marks))]
    return dts[DROP_EPOCHS:]


# --------------------------------------------------------------- 跑一侧
def reset_side(d):
    """把这一侧退回「没练过」的状态。⛔ 不重置的话第二次跑会【续训】,
    而续训与从底模起是两件事,读数不可比,且这个差别不会有任何东西报错。"""
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if os.path.isfile(p) and (n.startswith(CKPT_PAT) or n in ("train.log",
                                                                  "resume_state.json",
                                                                  "best_state.json")):
            os.remove(p)
        elif os.path.isdir(p) and n in ("resume_best", "weights", "eval", "logs"):
            shutil.rmtree(p)
        elif os.path.isfile(p) and n.startswith("events.out.tfevents"):
            os.remove(p)


def gpu_now():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return [x.strip() for x in r.stdout.strip().split(",")]
    except Exception as exc:  # noqa: BLE001
        return ["err", str(exc), ""]


def run_side(side, epochs, inject_ms, transcript):
    py = sys.executable
    reset_side(UP_EXP if side == "orig" else OURS_EXP)
    if side == "orig":
        cmd = [py, os.path.join(HERE, "speed_run_orig.py"), "--exp", EXP, "--epochs", str(epochs)]
        log = os.path.join(UP_EXP, "train.log")
    else:
        cmd = [py, os.path.join(HERE, "speed_run_ours.py"), "--run-dir", OURS_EXP,
               "--epochs", str(epochs), "--inject-ms", str(inject_ms),
               "--out", os.path.join(ARENA, "ours_out.json")]
        log = os.path.join(OURS_EXP, "train.log")
    before = gpu_now()
    # ⛔⛔ 运行中采样必须由**父进程**做,而且**两侧同法**。
    #    第一版只有我方侧有运行中采样(它在被测进程内自己采),上游侧只剩 `gpu_after`
    #    —— 而那是**跑完之后**采的,卡已经掉回 idle(实测 615-750 MHz)⇒ 三对全被
    #    时钟不变量剔掉,闸判「不可归因」。**闸的行为是对的,不对称的是我的探针。**
    #    ⇒ 改成父进程边跑边采,两侧用同一把尺,而且它顺带覆盖了「我方侧」也不必自证。
    mid = []
    with open(transcript, "a", encoding="utf-8", errors="backslashreplace") as tf:
        tf.write("\n===== %s  cmd=%s\n" % (side, cmd))
        proc = subprocess.Popen(cmd, stdout=tf, stderr=subprocess.STDOUT)
        import time as _t
        _t.sleep(WARM_SKIP_S)          # 跳过起进程/建模那段(那时卡本来就没升频)
        while proc.poll() is None:
            mid.append(gpu_now())
            _t.sleep(SAMPLE_EVERY_S)
        r = proc
    after = gpu_now()
    # ⛔ 上游「正常完训」是 os._exit(2333333) ⇒ rc = 2333333 & 0xFF = 149,**不是 0**。
    #    ⇒ 判成功不能看 rc;判据放在「日志里有没有够数的 epoch 行」。
    return {"rc": r.returncode, "log": log, "gpu_before": before,
            "gpu_mid": mid, "gpu_after": after}


def clock_ok(info, _unused=None):
    """运行中时钟不变量 —— **两侧同法**,取自父进程边跑边采的样本。

    ⛔ 绝不许回落到 `gpu_after`:那是**跑完之后**采的,卡已经掉回 idle
       (实测 615-750 MHz)⇒ 会把每一次好读数都判成不可归因。
       第一版就是这么写的,而闸当场把三对全剔了 —— 闸对,探针错。"""
    vals = []
    for row in info.get("gpu_mid") or []:
        try:
            vals.append(int(row[0]))
        except Exception:  # noqa: BLE001
            pass
    if len(vals) < 3:
        return None          # 采样太少 ⇒ 不判,交给上层记账
    # ⛔⛔ 用**中位数**不是 min。第一版用 min,而实测:我方侧运行中会**正常掉档**
    #    (1035 / 795 / 375 MHz)—— 那不是冷卡,是 epoch 边界那几秒 GPU 在等
    #    `resume_best` 的 G/D 存档与全张量 isfinite 扫描(**CPU/磁盘活,我方独有**)。
    #    而真正的冷卡事故是**全程 780**。⇒ min 把「正常掉档」当成「冷卡」,
    #    并且**只会系统性地剔掉我方那条臂** —— 那是偏倚不是噪声。
    #    ⇒ 判据 = 中位时钟;掉档比例作为读数上下文一起报(它本身就是一条信号)。
    med = statistics.median(vals)
    low = sum(1 for v in vals if v < CLOCK_FLOOR_MHZ)
    return med >= CLOCK_FLOOR_MHZ, {"median": med, "n": len(vals), "below_floor": low,
                                    "min": min(vals), "max": max(vals)}


# --------------------------------------------------------------- 主流程
def measure(args):
    transcript = os.path.join(ARENA, "gate_speed_transcript.txt")
    open(transcript, "w", encoding="utf-8").close()
    print("[axis] 我方 vs 上游 · **代码轴**(两侧同一个解释器 %s)· GPU + fp16 · 生产 log_interval"
          % os.path.basename(os.path.dirname(os.path.dirname(sys.executable))))
    print("[axis] ⛔ 它**不**回答『我们的安装包 vs 用户手上的整合包』(环境轴)——"
          "那一轴 ContentVec 走 CPU ORT,我方大输,落在②号尺子。")
    print("[setup] 每对 = 先我方后上游,交替 %d 对,每次 %d epoch(丢前 %d)"
          % (args.pairs, args.epochs, DROP_EPOCHS))
    print()

    pairs, notes = [], []
    for i in range(args.pairs):
        row = {}
        for side in ("ours", "orig"):
            info = run_side(side, args.epochs, args.inject_ms if side == "ours" else 0.0,
                            transcript)
            try:
                ts = epoch_times(info["log"], side)
            except Unrunnable as exc:
                raise Unrunnable("第 %d 对的 %s 侧:%s" % (i, side, exc))
            mid = None
            if side == "ours":
                try:
                    with open(os.path.join(ARENA, "ours_out.json"), encoding="utf-8") as f:
                        mid = json.load(f).get("gpu_mid")
                except Exception:  # noqa: BLE001
                    pass
            ok = clock_ok(info, mid)
            row[side] = {"times": ts, "median": statistics.median(ts), "rc": info["rc"],
                         "clock_ok": None if ok is None else ok[0],
                         "clocks": None if ok is None else ok[1]}
            print("  对 %d · %-4s  epoch 时间 %s  中位 %.3f s  rc=%s  时钟 %s"
                  % (i, side, ["%.2f" % t for t in ts], row[side]["median"],
                     info["rc"], row[side]["clocks"]))
        if row["ours"]["clock_ok"] is False or row["orig"]["clock_ok"] is False:
            notes.append("对 %d 因运行中 GPU 时钟低于地板 %d MHz 被剔除(不可归因,不是慢了)"
                         % (i, CLOCK_FLOOR_MHZ))
            continue
        pairs.append(row["ours"]["median"] / row["orig"]["median"])
    for n in notes:
        print("  ⛔ " + n)
    return pairs, notes


def judge(pairs, notes, args):
    print()
    if len(pairs) < MIN_PAIRS:
        raise Unrunnable("有效对数 %d < %d(剔除 %d 对)⇒ 本轮不构成一次判定"
                         % (len(pairs), MIN_PAIRS, len(notes)))
    mean = statistics.mean(pairs)
    sd = statistics.stdev(pairs)
    sem = sd / math.sqrt(len(pairs))
    tol = 3 * sem
    print("★ 对内比值(我方/上游): %s" % ["%.4f" % p for p in pairs])
    print("★ 均值 %.4f  ⇒ 我方比上游 **%s %.2f%%**" % (mean, "慢" if mean > 1 else "快",
                                                     abs(mean - 1) * 100))
    print("★ 标准差 %.2f%% · SEM %.2f%% · **3σ 门限 = %.2f%%**" % (sd * 100, sem * 100, tol * 100))
    if tol > CEILING:
        raise Unrunnable(
            "3×SEM = %.2f%% 超过天花板 %.2f%% ⇒ **这台机器上本轮这把尺子没有分辨力**,\n"
            "       不构成一次判定(⛔ 不许把它读成绿:`gate0_guard.py:173-175` —— 打印是汇报不是判据)"
            % (tol * 100, CEILING * 100))
    slower = (mean - 1) > tol
    print()
    if slower:
        print("RESULT: RED —— 我方比上游慢 %.2f%%,超过 3σ 门限 %.2f%%" % ((mean - 1) * 100, tol * 100))
        return EXIT_RED
    print("RESULT: ALL PASS —— 用户那句「只可能比原包快不可能比原包慢」在本轮**成立**")
    print("        (我方 %s %.2f%%,门限 %.2f%%;判据是单边的)"
          % ("慢" if mean > 1 else "快", abs(mean - 1) * 100, tol * 100))
    return EXIT_PASS


def selftest():
    """⛔ 一条从没被执行过的错误分支就是一条空判据 —— 这里逐条真触发。"""
    fails = []

    def expect(label, fn, want):
        try:
            fn()
            got = "没抛"
        except Unrunnable as e:
            got = "Unrunnable"
            if want == "Unrunnable":
                print("  ok   %-44s -> %s" % (label, str(e).splitlines()[0][:70]))
                return
        except Exception as e:  # noqa: BLE001
            got = type(e).__name__
        if got != want:
            fails.append("%s: 期望 %s,实得 %s" % (label, want, got))

    tmp = os.path.join(ARENA, "_selftest")
    os.makedirs(tmp, exist_ok=True)
    # ⑴ 日志缺失 ⇒ 不可归因
    expect("日志不存在", lambda: epoch_times(os.path.join(tmp, "nope.log"), "x"), "Unrunnable")
    # ⑵ 日志在但 epoch 行不够 ⇒ 不可归因(不是打绿!)
    p = os.path.join(tmp, "short.log")
    with open(p, "w", encoding="utf-8") as f:
        f.write("2026-08-12 10:00:00,000\tx\tINFO\t====> Epoch: 1 [..] | (0:00:01)\n")
    expect("epoch 行不够", lambda: epoch_times(p, "x"), "Unrunnable")
    # ⑶ 有效对数不足 ⇒ 不可归因
    expect("有效对数不足", lambda: judge([1.0, 1.0], ["x"], None), "Unrunnable")
    # ⑷ 噪声超天花板 ⇒ 不可归因(而不是打绿)
    expect("噪声超天花板", lambda: judge([1.0, 1.4, 0.6], [], None), "Unrunnable")
    # ⑸ 正常判绿 / 判红
    ns = argparse.Namespace()
    got = judge([0.97, 0.971, 0.969], [], ns)
    if got != EXIT_PASS:
        fails.append("三条一致的『我方更快』应当 PASS,实得 %s" % got)
    else:
        print("  ok   %-44s -> PASS" % "我方更快 ⇒ 绿")
    got = judge([1.05, 1.051, 1.049], [], ns)
    if got != EXIT_RED:
        fails.append("三条一致的『我方更慢』应当 RED,实得 %s" % got)
    else:
        print("  ok   %-44s -> RED" % "我方更慢 ⇒ 红")
    # ⑹ 解析器真的解得出两侧的现存日志(不是只对我造的假日志有效)
    for lp, lab in ((os.path.join(UP_EXP, "train.log"), "上游 arena"),
                    (os.path.join(OURS_EXP, "train.log"), "我方 arena")):
        if os.path.isfile(lp):
            n = len(epoch_marks(lp))
            print("  ok   %-44s -> 解出 %d 条 epoch 行" % ("解析器对 " + lab, n))
    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if fails:
        for f in fails:
            print("  FAIL %s" % f)
        print("gate_speed_rvc 自检: FAILED(%d)" % len(fails))
        return EXIT_SELFTEST
    print("gate_speed_rvc 自检: ALL OK")
    return EXIT_PASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--inject-ms", type=float, default=0.0,
                    help="阴性对照:给我方侧每步注射已知延迟(毫秒)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    bad = [k for k in ("UTAI_DIAGNOSTICS", "CUDA_LAUNCH_BLOCKING") if os.environ.get(k)]
    if bad:
        print("⛔ preflight:环境里有 %s ⇒ 【闸的前提不满足】" % bad)
        return EXIT_UNRUNNABLE
    for d in (UP_EXP, OURS_EXP):
        if not os.path.isdir(d):
            print("⛔ preflight:%s 不存在 —— 先跑 speed_arena_setup.py" % d)
            return EXIT_UNRUNNABLE
    if args.inject_ms:
        print("⛔ 本轮是**阴性对照**:给我方侧每步注射 %.1f ms(名义值;实测会超射,见转录)\n"
              % args.inject_ms)
    try:
        pairs, notes = measure(args)
        return judge(pairs, notes, args)
    except Unrunnable as exc:
        print()
        print("RESULT: GATE-UNRUNNABLE ⇒ %s" % exc)
        print("        ⛔ 这**不是**「通过」,也**不是**「慢了」。")
        return EXIT_UNRUNNABLE


if __name__ == "__main__":
    sys.exit(main())
