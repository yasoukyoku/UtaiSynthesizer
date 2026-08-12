# -*- coding: utf-8 -*-
"""gate0 链跑器(S135,§F7 笔 2)—— 一条链一条命令,而且【绿有意义、红能归因】。

为什么要有它:
  ⑴ **gate0 的 compare 现在要求 `GATE0_T0`**(本轮起始 epoch 秒)。没有它就没有
     新鲜度判据 ⇒ 分不出读到的产物是今天算的还是七月的。这个跑器负责在**第一个
     产出步骤之前**把 t0 钉下来,并传给每一段。
  ⑵ 每条链是 清货 → prepare → 原版侧 → 我方侧 → C1 → compare 五六段,任何一段挂掉
     在终端上都长得像"对拍失败"(S129 铁律)。这里把每段分开报,出口码分七档。
  ⑶ **清货必须是"删目录"而不是"清空目录"**:compare 的守卫只看 isdir,
     清空会让 `compare_wav_dir/compare_feat` 打印 `max|Δ|=0.000e+00` 的假 PASS,
     删掉才会正确地红。清单硬编码在下面,而且过**前缀白名单** —— 这是 S96 炸仓
     (`git worktree remove --force` 穿过 junction 清空了三个目录)之后的硬规矩:
     ⛔ 绝不用通配、绝不用 rm -rf 的任何变体、绝不碰白名单以外的任何路径。

出口码:
    0 = PASS
    1 = COMPARE 判负        <- ★ 只有这一种是「被测的东西不对」
    2 = prepare 失败
    3 = 原版侧失败          <- 参照物没跑起来,不是我们的代码的问题
    4 = 我方侧失败          <- 被测代码抛了
    5 = 用法 / 夹具缺件 / 清货被拒
    6 = 某一段判「不可归因」 <- gate0_guard 的 exit 3(读数不构成一次判定)

解释器轴(⛔ 与 gate1 **不同**,别照搬):gate0 的两侧解释器**天生就该不一样** ——
原版侧要"原版时代环境"(RVC 整合包 runtime:py3.9 / torch 2.0 / fairseq 0.12.2 /
librosa 0.9.1,本机唯一能真跑上游 fairseq 预处理的环境),我方侧是 training/.venv
(torch 2.5.1)。gate1 那条"两侧同一个解释器"的口径是为了隔离**代码轴**,
gate0 这边恰恰要保留环境轴并用 C 层逐轴剥离。**不许把它"修"成 gate1 的形状。**
⚠ 但注意轴的落点:`gate0_sovits_compare.py` 会在 compare 进程里**现算**我方侧的
C3/C4/C5 ⇒ "compare 用哪个 python"同样能换轴;而 RVC 的 `gate0_compare.py` 全文
不 import torch ⇒ 那条链完全不吃 torch 轴。

⛔ 跑之前:关掉 dev build;确认 backup 已做(gate0 的好几段会 rmtree 唯一副本 ——
   `gate0_sovits_orig.py:116` 会 rmtree sovits_orig/dataset44k,而 rmtree 发生在
   任何再生动作**之前**;`gate0_diff_prepare.py:36` 会 rmtree diff 双侧)。
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GT = os.path.join(REPO, "converter", "verify", "training")
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"

VENV = os.path.join(REPO, "training", ".venv", "Scripts", "python.exe")
RVCPY = r"D:\MyDev\RVC\RVC20240604Nvidia\runtime\python.exe"

# —— 清货白名单:任何要删的路径必须落在这些前缀之下,否则当场拒绝。
CLEAR_ALLOW_PREFIXES = (
    os.path.join(TESTING, "rvc_ours") + os.sep,
    os.path.join(TESTING, "rvc_B2_ours") + os.sep,
    os.path.join(TESTING, "sovits_ours", "dataset_44k") + os.sep,
    os.path.join(TESTING, "sovits_v2_ours", "dataset_44k") + os.sep,
)

# —— 每条链要手清什么。⚠ 这份清单比队列 §F7 记的那份**短得多**,因为逐个查过
#    "谁会重建它":凡是脚本自己 rmtree 或无条件覆写的,手清没有收益反而多一分风险。
#    留在这里的都是**没有任何人会清的 skip-if-exists 产物**。
#    生产者会重建目录本身(RVC preprocess.py:50-52 / extract_f0.py:67-68 /
#    extract_feature.py:31 / filelist.py:32;sovits preprocess.py:70 全是
#    makedirs(exist_ok=True))⇒ 删目录是安全的。
CLEAR = {
    "rvc": [
        os.path.join(TESTING, "rvc_ours", "0_gt_wavs"),
        os.path.join(TESTING, "rvc_ours", "1_16k_wavs"),
        os.path.join(TESTING, "rvc_ours", "2a_f0"),        # extract_f0.py:88 skip-if-exists
        os.path.join(TESTING, "rvc_ours", "2b-f0nsf"),     # 同上(两个 .npy 同时存在才跳)
        os.path.join(TESTING, "rvc_ours", "3_feature768"), # extract_feature.py:47 skip-if-exists
        os.path.join(TESTING, "rvc_ours", "mute"),         # filelist.py:33 skip-if-exists
        os.path.join(TESTING, "rvc_ours", "config.json"),  # filelist.py:117 skip-if-exists
        os.path.join(TESTING, "rvc_ours", "filelist.txt"),
        os.path.join(TESTING, "rvc_ours", "total_fea.npy"),
    ],
    # sovits 我方侧:slice_and_resample 只删 .wav(preprocess.py:75),
    # .soft.pt/.f0.npy/.spec.pt/.vol.npy 四类是 skip-if-exists(extract.py:169/183/188/207)
    # ⇒ 整个 gate 目录删掉最干净;preprocess.py:70 会重建。
    "sovits": [os.path.join(TESTING, "sovits_ours", "dataset_44k", "gate")],
    # v2 同族:.soft.pt/.f0.npy/.aam80.npy 三类 skip-if-exists(extract.py:123/137/152)
    # ⚠ 顺带清掉 gate1_sovits_v2_prepare 拷进来的 33 个 .mel.npy(08-11 09:01)——
    #    只清 .aam80.npy 不清 .mel.npy 会留下一批指向已消失数据的孤儿,
    #    而 gate1 v2 prepare 会把它报成 "0 new / 33 refreshed" = 正常态,看不出来。
    "sovits_v2": [os.path.join(TESTING, "sovits_v2_ours", "dataset_44k", "gate")],
    # 这两条不用手清:gate0_diff_prepare.py:35-36 自己 rmtree 双侧;
    # gate0_vocoder.py:106-109 自己 rmtree slices48k,npz 无条件覆写。
    # ⛔ 尤其 **不许**清 D:\MyDev\TESTING\smoke_vocoder\ws\slices ——
    #    那是声码器冒烟的产物,gate0/gate1 里没有任何脚本能重建它。
    "diff": [],
    "vocoder": [],
}

# name, python, script, kind(用来定出口码), optional
CHAINS = {
    "rvc": [
        # ⛔ 原版侧 rvc_orig **故意不跑**:它是「不变输入 × 不变上游代码」的函数,
        #    而且 3_feature768 是 CUDA/TF32 产物、不逐位可复现 ⇒ 重跑反而换掉参照物。
        ("3_ours", VENV, "gate0_run_ours.py", "ours"),
        ("3b_rebuild_b2", VENV, "gate0_rebuild_b2_ours.py", "ours"),
        ("4_compare", VENV, "gate0_compare.py", "compare"),
    ],
    "sovits": [
        ("1_prepare", VENV, "gate0_sovits_prepare.py", "prepare"),
        ("2_orig", RVCPY, "gate0_sovits_orig.py", "orig"),
        ("3_ours", VENV, "gate0_sovits_run_ours.py", "ours"),
        ("3c_c1", RVCPY, "gate0_sovits_c_resample.py", "compare"),
        ("4_compare", VENV, "gate0_sovits_compare.py", "compare"),
    ],
    "sovits_v2": [
        # ⛔ 没有自己的 prepare:它吃的 sovits_slices\gate 是 **4.1 那条链的
        #    gate0_sovits_prepare.py** 产的(:28-30 无条件清空重建)。
        #    ⇒ v2 必须排在 sovits 之后跑,而且中间不许再跑一次 4.1 的 prepare。
        ("2_orig", RVCPY, "gate0_sovits_v2_orig.py", "orig"),
        ("3_ours", VENV, "gate0_sovits_v2_run_ours.py", "ours"),
        ("3c_c1", RVCPY, "gate0_sovits_v2_c_resample.py", "compare"),
        ("4_compare", VENV, "gate0_sovits_v2_compare.py", "compare"),
    ],
    "diff": [
        # ⛔ prepare 的源是 sovits_ours\dataset_44k\gate ⇒ 必须排在 sovits 的
        #    3_ours 之后;而且 prepare:36 的 rmtree 在 :43 的 assert **之前**,
        #    源少一个伴生文件就会留下"一侧空、一侧陈"。
        ("1_prepare", VENV, "gate0_diff_prepare.py", "prepare"),
        ("2_orig", RVCPY, "gate0_diff_orig.py", "orig"),
        ("3_ours", VENV, "gate0_diff_run_ours.py", "ours"),
        ("4_compare", VENV, "gate0_diff_compare.py", "compare"),
    ],
    "vocoder": [
        ("4_compare", VENV, "gate0_vocoder.py", "compare"),
    ],
}

ORDER = ["rvc", "sovits", "sovits_v2", "diff", "vocoder"]
KIND_EXIT = {"prepare": 2, "orig": 3, "ours": 4, "compare": 1}
UNRUNNABLE = 3          # gate0_guard.EXIT_UNRUNNABLE


def _refuse(msg):
    print("REFUSING: %s" % msg)
    sys.exit(5)


# ⛔ S139(§F7 笔 F)—— S135 立的那条,而当时写着「**今天没有任何闸在检查这一点**」:
#    `f1f0347` 之后 `gate_dataset` 是一个**「不许有任何 `.part`」的受保护目录**。
#    机理:我方的 `utai_train.cache.dataset_entries()` **跳过** `.part`(那一笔的全部内容),
#    而**上游的枚举永远不跳** ⇒ 目录里出现一个 `.part`,两侧当场吃的就不是同一批文件,
#    而 gate0 会把它报成一条**数值差** —— 一条由夹具造成的红,被归因成我们的代码。
#    ⚠ 同族还有「非 .wav 的杂物」:上游 preprocess 对非音频文件的行为与我方不一定一致。
GATE_DATASET = os.path.join(TESTING, "gate_dataset")


def assert_dataset_clean(root=None):
    """开跑前断言:数据集目录里只有 .wav,且**一个 `.part` 都没有**。

    返回 (件数, 总字节)。⛔ 空目录同样拒绝 —— 空集不是通过。
    """
    root = root or GATE_DATASET
    if not os.path.isdir(root):
        _refuse("数据集目录不在:%s" % root)
    names = sorted(os.listdir(root))
    parts = [n for n in names if n.endswith(".part")]
    if parts:
        _refuse("⛔ `%s` 里有 %d 个 `.part`:%s\n"
                "          我方 `cache.dataset_entries()` 会**跳过**它们,而上游的枚举**永远不跳**\n"
                "          ⇒ 两侧当场吃的不是同一批文件,而 gate0 会把它报成一条【数值差】。\n"
                "          (这条判据是 S135 记下、S139 补上的:在此之前没有任何闸在看它)"
                % (root, len(parts), parts[:5]))
    others = [n for n in names
              if not n.lower().endswith(".wav") and os.path.isfile(os.path.join(root, n))]
    if others:
        _refuse("`%s` 里有非 .wav 的文件:%s —— 两侧对它们的枚举行为不保证一致" % (root, others[:5]))
    wavs = [n for n in names if n.lower().endswith(".wav")]
    if not wavs:
        _refuse("`%s` 里一个 .wav 都没有 —— 空集不是通过" % root)
    total = sum(os.path.getsize(os.path.join(root, n)) for n in wavs)
    empty = [n for n in wavs if os.path.getsize(os.path.join(root, n)) == 0]
    if empty:
        _refuse("`%s` 里有 0 字节的 wav:%s(崩在写一半 / 占位)" % (root, empty))
    print("[DATASET] %s:%d 个 wav / %.1f MB,无 .part、无杂物、无 0 字节"
          % (root, len(wavs), total / 1e6))
    return len(wavs), total


def do_clear(chain, dry):
    paths = CLEAR[chain]
    if not paths:
        print("  (%s 不需要手清:它那几段自己会 rmtree / 无条件覆写)" % chain)
        return
    for p in paths:
        real = os.path.abspath(p)
        if not any(real.startswith(pref) for pref in CLEAR_ALLOW_PREFIXES):
            _refuse("清货路径落在白名单之外:%s" % real)
        if ".." in p:
            _refuse("清货路径含 '..':%s" % p)
    for p in paths:
        if os.path.isdir(p):
            n = sum(len(fs) for _r, _d, fs in os.walk(p))
            print("  %-9s %s  (目录, %d 件)" % ("WOULD-RM" if dry else "RM", p, n))
            if not dry:
                shutil.rmtree(p)
        elif os.path.isfile(p):
            print("  %-9s %s  (文件)" % ("WOULD-RM" if dry else "RM", p))
            if not dry:
                os.remove(p)
        else:
            print("  %-9s %s  (本来就不在)" % ("ABSENT", p))


def step(chain, name, py, script, out_dir, t0):
    log = os.path.join(out_dir, "%s.log" % name)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "GATE0_T0": "%.3f" % t0}
    env.pop("UTAI_DIAGNOSTICS", None)
    started = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("$ %s -u %s   (GATE0_T0=%.3f)\n\n" % (py, script, t0))
        fh.flush()
        rc = subprocess.run([py, "-u", os.path.join(GT, script)], cwd=REPO,
                            stdout=fh, stderr=subprocess.STDOUT, text=True,
                            env=env).returncode
    dt = time.time() - started
    print("  %-10s %-22s rc=%-3d %6.1fs  %s" % (chain, name, rc, dt, log))
    return rc, log


def run_chain(chain, runs_root, t0, do_clear_first, dry, only_clear=False):
    out = os.path.join(runs_root, chain)
    os.makedirs(out, exist_ok=True)
    print("\n===== chain %s =====" % chain)
    if do_clear_first:
        do_clear(chain, dry)
    if only_clear:
        return 0
    if dry:
        for name, py, script, _kind in CHAINS[chain]:
            print("  WOULD-RUN  %-22s %s  %s" % (name, os.path.basename(py), script))
        return 0
    for name, py, script, kind in CHAINS[chain]:
        if not os.path.isfile(py):
            print("MISSING interpreter for %s/%s: %s" % (chain, name, py))
            return 5
        rc, log = step(chain, name, py, script, out, t0)
        if rc == UNRUNNABLE:
            tail = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-8:]
            print("VERDICT UNRUNNABLE (读数不可归因,这不是一次判定) — %s" % log)
            for ln in tail:
                print("   " + ln)
            return 6
        if rc != 0:
            tail = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-10:]
            label = {"prepare": "PREPARE-FAILED (闸自己没准备好)",
                     "orig": "ORIG-FAILED (参照物没跑起来 ≠ 我们的代码不对)",
                     "ours": "OURS-FAILED (被测代码抛了)",
                     "compare": "COMPARE-FAILED (★ 这一种才是「被测的东西不对」)"}[kind]
            print("VERDICT %s rc=%d — %s" % (label, rc, log))
            for ln in tail:
                print("   " + ln)
            return KIND_EXIT[kind]
    print("VERDICT PASS (%s)" % chain)
    return 0


def main():
    ap = argparse.ArgumentParser()
    # nargs="?" —— 让 `--selftest` 不必再给一条链(S139:自检不跑任何链)
    ap.add_argument("chain", nargs="?", choices=sorted(CHAINS) + ["all"])
    ap.add_argument("--clear", action="store_true",
                    help="跑之前先删掉那几个没人会清的 skip-if-exists 产物目录(白名单内)")
    ap.add_argument("--dry-run", action="store_true", help="只打印会删什么、会跑什么")
    ap.add_argument("--clear-only", action="store_true",
                    help="只清货不跑 —— 用来做那个免费的阴性对照:清完先单跑一次 compare,"
                         "它必须以 exit 3 点名『0 件』,而不是打印 max|Δ|=0.000e+00 的假 PASS")
    ap.add_argument("--runs", default=r"D:\MyDev\TESTING\s135_f7\runs")
    ap.add_argument("--t0", type=float, default=None,
                    help="显式指定 t0(epoch 秒)。一般不用 —— 默认会复用本次 gate0 会话的 t0")
    ap.add_argument("--new-session", action="store_true",
                    help="重新起一个 gate0 会话(丢弃 <runs>/T0.txt 里记的 t0)")
    ap.add_argument("--selftest", action="store_true",
                    help="阴性对照:把开跑前断言与清货白名单的每一条拒绝分支真触发一次")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.chain:
        _refuse("要跑哪条链?%s(或 all);只想跑自检用 --selftest" % ", ".join(sorted(CHAINS)))

    # ⛔ S139:开跑前断言,先于清货与任何一段 —— 见 assert_dataset_clean 的头注。
    assert_dataset_clean()

    os.makedirs(args.runs, exist_ok=True)

    # ⛔ 新鲜度必须是【整场 gate0 会话】级的,不是【每条链】级的:
    #    `sovits_slices\gate` 由 **sovits(4.1) 那条链的 prepare** 产出,而
    #    sovits_v2 的 C1、v2 的原版侧、4.1 的原版侧 **四个读者** 都吃它。
    #    一条链一个 t0 ⇒ 后跑的链会把兄弟链刚产的共享输入判成"陈货"。
    #    (S135 实测踩到过一次:v2 的 C1 报 "33/33 件不是本轮产物",而那批切片
    #     是三分钟前 4.1 的 prepare 刚产的。)
    #    ⇒ t0 落盘到 <runs>/T0.txt,同一场会话里所有链复用它。
    # ⛔⛔ S135 二审自己抓出来的:t0 落盘复用**必须有时效上界**,否则这把尺子有一条
    #    默认打开的失效通道 —— `--runs` 默认落在固定路径,第二次调用起,凡是落在
    #    [旧 t0, now] 窗口里的陈货一律被判 FRESH 并绿。
    #    ⇒ 花一整笔买来的「不许拿陈货比陈货」,分辨力就取决于调用者记不记得加 --new-session。
    #    现在:超过 SESSION_MAX_AGE 一律**当新会话**,并**响亮说明**为什么。
    SESSION_MAX_AGE = 6 * 3600.0
    t0_file = os.path.join(args.runs, "T0.txt")
    reuse = None
    if args.t0 is None and not args.new_session and os.path.isfile(t0_file):
        cand = float(open(t0_file, encoding="utf-8").read().strip())
        age = time.time() - cand
        if age <= SESSION_MAX_AGE:
            reuse = cand
        else:
            print("⛔ %s 里的 t0 已经 %.1f 小时前了(上界 %.0f 小时)—— **不复用**,"
                  "否则这期间产生的任何陈货都会被判成本轮产物。改起新会话。"
                  % (t0_file, age / 3600.0, SESSION_MAX_AGE / 3600.0))
    if args.t0 is not None:
        t0 = args.t0
        print("GATE0_T0 = %.3f (显式指定 —— ⚠ 你自己负责它早于本轮全部产物)" % t0)
    elif reuse is not None:
        t0 = reuse
        print("GATE0_T0 = %.3f (复用 %s,%.1f 分钟前记下的)"
              % (t0, t0_file, (time.time() - t0) / 60.0))
    else:
        # 让 t0 严格早于任何产物 —— 文件系统时间戳的分辨率与时钟漂移都可能让
        # "同一秒产出的文件" mtime 略早于 t0,那会报成一条假的"陈货"。
        t0 = time.time() - 2.0
        # ⛔ S139:`--dry-run` **不许写盘**。此前它照样 stamp 一个新 t0 并覆盖 T0.txt ——
        #    一次「只说不做」的调用改掉了整场会话的新鲜度基准,而那正是这份文件里
        #    花一整笔(S135 二审)买来的东西。S139 跑 `rvc --dry-run` 时当场踩到并修。
        if args.dry_run:
            print("GATE0_T0 = %.3f (新会话;⚠ --dry-run ⇒ **没有**写进 %s)" % (t0, t0_file))
        else:
            with open(t0_file, "w", encoding="utf-8") as f:
                f.write("%.3f\n" % t0)
            print("GATE0_T0 = %.3f (新会话,已记入 %s)" % (t0, t0_file))
    print("           = %s" % time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)))

    chains = ORDER if args.chain == "all" else [args.chain]
    if args.chain == "all":
        print("顺序是硬的:sovits 的 prepare 产 sovits_slices(v2 与两条 C1 都吃它),"
              "diff 的 prepare 源是 sovits_ours/dataset_44k/gate")
    worst = 0
    results = []
    for c in chains:
        rc = run_chain(c, args.runs, t0, args.clear, args.dry_run, args.clear_only)
        results.append((c, rc))
        if rc != 0 and worst == 0:
            worst = rc
    print("\n===== 汇总 (t0=%.3f) =====" % t0)
    for c, rc in results:
        print("  %-10s rc=%d %s" % (c, rc, "PASS" if rc == 0 else "^^^"))
    sys.exit(worst)


def _selftest():
    """⛔ S139:这个跑器此前**没有自检**,而它的拒绝分支(清货白名单、`..`、t0 时效上界)
    从没被执行过 —— S129:一条从没被执行过的错误分支就是一条空判据。
    这里逐条真触发,并**顺带证明干净的输入不会被误拒**(对照臂)。
    """
    import tempfile
    fails = []

    def expect_refuse(name, fn, needle=None):
        try:
            fn()
        except SystemExit as e:
            if e.code != 5:
                fails.append("%s 应该 exit 5,实际 %r" % (name, e.code))
            else:
                print("  ok   %-42s -> REFUSING exit 5" % name)
            return
        except Exception as e:                       # noqa: BLE001
            fails.append("%s 应该 REFUSING,却抛了 %r" % (name, e))
            return
        fails.append("%s 应该 REFUSING,却过了" % name)

    def expect_ok(name, fn):
        try:
            fn()
            print("  ok   %-42s -> 正常通过" % name)
        except Exception as e:                       # noqa: BLE001
            fails.append("%s 不该拒,却拒了 %r" % (name, e))

    tmp = tempfile.mkdtemp(prefix="run_gate0_selftest_")
    try:
        d = os.path.join(tmp, "gate_dataset")
        os.makedirs(d)
        expect_refuse("数据集目录空", lambda: assert_dataset_clean(d))
        for n in ("a.wav", "b.wav"):
            with open(os.path.join(d, n), "wb") as f:
                f.write(b"RIFFxxxx")
        expect_ok("数据集干净(2 个 wav)", lambda: assert_dataset_clean(d))
        # ⭐ 本笔的主角:一个 .part 就会让两侧吃不同的文件集合
        with open(os.path.join(d, "c.wav.part"), "wb") as f:
            f.write(b"half")
        expect_refuse("数据集里有 .part", lambda: assert_dataset_clean(d))
        os.remove(os.path.join(d, "c.wav.part"))
        with open(os.path.join(d, "notes.txt"), "w") as f:
            f.write("x")
        expect_refuse("数据集里有非 .wav 杂物", lambda: assert_dataset_clean(d))
        os.remove(os.path.join(d, "notes.txt"))
        open(os.path.join(d, "zero.wav"), "wb").close()
        expect_refuse("数据集里有 0 字节 wav", lambda: assert_dataset_clean(d))
        os.remove(os.path.join(d, "zero.wav"))
        expect_refuse("数据集目录不在", lambda: assert_dataset_clean(os.path.join(tmp, "nope")))

        # 清货白名单:落在白名单之外 / 含 '..' —— 两条都必须当场拒
        saved = dict(CLEAR)
        try:
            CLEAR["rvc"] = [os.path.join(tmp, "somewhere_else")]
            expect_refuse("清货路径落在白名单之外", lambda: do_clear("rvc", dry=True))
            # ⚠ 第一版这条用的是 `rvc_ours\..\evil` —— 而 `abspath` 先把 `..` 规范化掉了,
            #   于是它被**白名单**那一条拦下,`".." in p` 那条分支根本没跑到。
            #   ⇒ 要真触发它,路径必须**规范化之后仍落在白名单内**但字面上带 `..`。
            #   (这就是「红了、断言也对,但它在回答另一个问题」的又一次小复现。)
            CLEAR["rvc"] = [os.path.join(TESTING, "rvc_ours", "sub", "..", "2a_f0")]
            expect_refuse("清货路径含 '..'(规范化后仍在白名单内)",
                          lambda: do_clear("rvc", dry=True))
            CLEAR["rvc"] = [os.path.join(TESTING, "rvc_ours", "2a_f0")]
            expect_ok("清货路径在白名单内(--dry-run)", lambda: do_clear("rvc", dry=True))
        finally:
            CLEAR.clear()
            CLEAR.update(saved)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if fails:
        for f in fails:
            print("  FAIL %s" % f)
        print("run_gate0_chain 自检: FAILED(%d)" % len(fails))
        return 4
    print("run_gate0_chain 自检: ALL OK")
    return 0


if __name__ == "__main__":
    main()
