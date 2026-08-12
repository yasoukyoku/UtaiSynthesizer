# -*- coding: utf-8 -*-
"""gate1 链跑器 —— 一条链一条命令,而且【红了能归因】。

S134 建于仓库之外(`TESTING\\s134_f7\\`,无 git、只有一份);**S139 收进仓库并加固**。
搬运不是原样拷贝:侦察 + 逐面对抗核验查明它有五条今天就咬人的缺陷,逐条写在下面。

为什么要有它(S129 铁律):每条链是 prepare → 原版侧 → 我方侧 → compare 四段,
任何一段挂掉在终端上都长得像「对拍失败」。这个跑器把四段分开报,并给出不同的出口码
(⛔ 与 `run_gate0_chain.py` **同一张表**,别让两层的同一个数字含义不同):

    0 = PASS
    1 = COMPARE 判负        <- ★ 只有这一种是「被测的东西不对」
    2 = prepare 失败        <- 闸自己没准备好
    3 = 原版侧失败          <- 参照物没跑起来,不是我们的代码的问题
    4 = 我方侧失败          <- 被测代码抛了
    5 = 用法 / 夹具缺件 / 被拒绝
    6 = 某一段判「不可归因」 <- gate0_guard.EXIT_UNRUNNABLE(读数不构成一次判定)
    7 = 历史对拍判负        <- ★ gate1 独有:今天的数字与 07-07/07-17、08-11 不一样了
                              (与 1 是**两个不同的问题**:1 说移植不对,7 说它变了)

每段的完整 stdout+stderr 落盘到 `<runs>/<chain>/<时间戳>/`,一个字都不丢。

────────────────────────────────────────────────────────────────────────────
S139 修掉的五条(全部实测,机理写在各自落点)
────────────────────────────────────────────────────────────────────────────
⑴ ⛔⛔⛔ **它会静默截掉自己的转录**。原来 `open(log, "w")` 在子进程启动**之前**就截断,
   而 `RUNS` 硬编码指着 `TESTING\\s134_f7\\runs` —— S134 五条链**唯一的在盘转录**。
   ⇒ 交接给这一笔的验收判据是「从仓内那份真跑一段」,而照原样跑下去第一件事就是把
   它自己要证明的那份证据抹掉。更毒的是**一次中途失败的重跑会留下半新半旧的日志目录**,
   而目录里没有任何东西能把两轮分开。
   ⭐ 同一个文件里早写着这条血训的近亲(`step()` 的 tmp 落位保护:「直接 `>` 会在脚本失败前
   就把历史产物截成 0 字节」)—— **它保护了别人的历史产物,没保护自己的转录**。
   ⇒ 现在:每次调用一个**时间戳子目录**;`step()` **拒绝**覆盖已存在的非空日志;
   每份日志第一行写来源(照 `run_gate0_chain.py:177` 的形状)。
⑵ ⛔⛔ **默认就跑 prepare**,而五个 `*_prepare.py` 首句都是 `shutil.rmtree` 双侧 expdir。
   rvc 那条 = `RVC\\logs\\gate1`(1.3 GB)+ `gate1_ours`(2.8 GB);声码器那条会删掉
   `SingingVocoders\\experiments\\gate1_voc`(3.5 GB = S134 声码器我方侧**唯一**证据)。
   ⚠ README 的 gate1 五节**第一行就叫你跑 prepare**,而同一份 README 746 行之后写着
   「⛔ 绝不许调任何 `*_prepare.py`」。⇒ 现在:**默认不跑**,要跑得同时给
   `--rebuild-fixtures` **和** 环境变量 `GATE1_ALLOW_REBUILD=1`,而且先把要删的体积逐条打出来。
⑶ ⛔ **`--skip-prepare` 单独用是允许的、退 0、在任何输出上留不下影子** ——
   而那正是 `pending_cleanups` 点名的「刻意制造『产物不是本轮的』那个状态」。
   ⇒ 现在:这一跑用了哪些跳过,写进 **VERDICT 行**与日志头,不许只活在调用者的记忆里。
⑷ ⛔ **`orig is None` 是一条死分支**:五条链没有一条的 orig 是 None,而它会**跳过整段
   并仍然打 PASS 退 0** —— 一条绕过 `--skip-orig` 联锁、还不打 REFUSING 的旁路。
   ⇒ 现在:没有参照物就是 `NO-REFERENCE` 退 5,绝不许退 0。
⑸ ⛔ **零新鲜度**。五条 compare 里没有一处检查产物是不是本轮算的;实测拿一份
   2026-07-07 的 jsonl 喂给八月的参照,它打出与 S134 转录**逐字符相同**的 ALL PASS。
   ⇒ 现在:跑器钉 `GATE1_T0`(会话级 + `T0.txt` + 6 小时上界,整块照 `run_gate0_chain.py:249-265`
   —— ⛔ 那条上界是 S135 二审自己买回来的,只搬「传 t0」会把已修的洞重新引进来)。
   ⚠ **另起 `GATE1_T0` 而不是复用 `GATE0_T0`**:同一个 shell 里先跑 gate0 再跑 gate1 时,
   复用会让 gate1 读到几小时前的 t0 而**静默变宽**。

✅ **S140:历史对拍已经进仓并接成每条链的第 5 段**(`gate1_history_compare.py`,退 7)。
   它问的是**另一个问题**:「自上一次跑 gate1 以来的那些 commit,数值上动过什么吗」。
   ⚠ 它的前身 `TESTING\s134_f7\compare_vs_history.py` 今天仍然在盘上,**别再跑它** ——
   S140 实测它整条链上没有一处能红:新鲜度守卫是一条**单调恒真**的谓词
   (比的是「live 是不是比一份 copy2 冻在七月的备份新」),缺件走**静默绿**,
   地板是与真值无关的 `步数×5`(真实分量数 6/7/10/2 ⇒ rvc 删任一个键仍恰好过线、
   sovits_v2 可以删五个),NaN 对 `if d > worst[0]` 恒不可见。
   ⇒ **同一天、同一份盘的实测对照**:老脚本一条链都没跑就打四行 `BITWISE-SAME` 退 **0**;
     新闸退 **3** 并点名「live/diff:1/1 件不是本轮产物」。

解释器轴(S134 决定,理由写在 EVIDENCE):每条链用【当年产出它自己那份历史基线的那个解释器】,
两侧同一个 ⇒ 代码轴仍然隔离,同时白拿一条「与历史 jsonl 对拍」的回归线。
  · rvc / sovits / diff / vocoder -> envs\\s42_staging_nv_cu130 (torch 2.11.0+cu130,
    也正好是出货 runtime pack 的版本)
  · sovits_v2                     -> training\\.venv (torch 2.5.1)
⛔ 与 `run_gate0_chain.py:25-29` 的口径**正好相反,别互相「修」**:gate0 两侧解释器天生不同
   (要原版时代环境),gate1 故意同一个(隔离代码轴)。
⚠ 四个 `gate1_*_compare.py` 的文件头与 README 写的「双方同 torch(2.5.1)」是**陈货**
   —— 经本跑器跑的那四条实际是 **2.11.0**(S134 §3 记过,四个月没改)。

⛔ 跑之前请确认:没有 dev build 在跑;工作树不要在这期间被改动
   (legs 之外,gate 也会 import utai_train —— 改一半的树会让它测一个从不存在的中间态)。

用法
    training/.venv/Scripts/python.exe converter/verify/training/run_gate1_chain.py rvc
    ... --selftest            # 阴性对照:每一档出口码用桩脚本真触发一次
    ... --dry-run             # 只说会跑什么、会删什么
    ... --rebuild-fixtures    # ⛔ 会 rmtree 双侧 expdir,还要 GATE1_ALLOW_REBUILD=1
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate0_guard as G        # noqa: E402  —— 只借 _say / EXIT_UNRUNNABLE 与它的编码修复
import gate1_guard as G1       # noqa: E402  —— 环境变量名是**它**的,别在这里抄第二份

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GT = HERE
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"

# ⛔ 转录根**不带会话号**:仓内先例 `run_gate0_chain.py:233` 把 `TESTING\s135_f7\runs`
#    冻进了默认值,而那正是 S135 五条链唯一的转录 —— 一个指向【不可再生历史证据】的
#    路径当默认值,是一个正在长大的形状(仓内今天已有 5 处)。这里不再加第 6 处。
RUNS_DEFAULT = r"D:\MyDev\TESTING\gate1_runs"

STAGING = os.path.join(TESTING, "envs", "s42_staging_nv_cu130", "Scripts", "python.exe")
VENV = os.path.join(REPO, "training", ".venv", "Scripts", "python.exe")

# ⛔ 两个环境变量名的**唯一真源是 `gate1_guard`**(消费者在那边)。这里只转引,
#    别抄第二份 —— 抄了之后哪天改一边就是一条静默失效的新鲜度判据。
T0_ENV = G1.T0_ENV
SKIPPED_ENV = G1.SKIPPED_ENV
SESSION_MAX_AGE = 6 * 3600.0
ALLOW_REBUILD_ENV = "GATE1_ALLOW_REBUILD"

EXIT_COMPARE, EXIT_PREPARE, EXIT_ORIG, EXIT_OURS = 1, 2, 3, 4
EXIT_USAGE, EXIT_UNRUNNABLE = 5, 6
# ⛔ S140:**7 = 历史对拍判负**,gate1 独有(gate0 那张表没有 7,不会撞)。
#    为什么不复用 1:两条判据问的是**两个不同的问题** —— 1 说「今天我们的循环 != 上游」,
#    7 说「今天的数字与七月/08-11 不一样了」。复用同一个码,汇总表上「移植不对」与
#    「历史漂了」就变成同一个数字,而那正是 S129 铁律要拆开的东西。
EXIT_HISTORY = 7
HISTORY_DRIVER = "gate1_history_compare.py"
KIND_EXIT = {"prepare": EXIT_PREPARE, "orig": EXIT_ORIG,
             "ours": EXIT_OURS, "compare": EXIT_COMPARE, "history": EXIT_HISTORY}

# ⛔⛔ S140:`all` 的退出码此前取**第一条非零**(而变量名叫 `worst`)。
#    配上 `ORDER` 把 rvc 排在第一位,一条 exit 6(读数不可归因,语义是「重跑一次就好」)
#    会把后面四条链里**任何一条真红(exit 1)整个盖住** —— 而这套码的全部价值就在于
#    `1` 被单列为「★ 只有这一种是被测的东西不对」。⇒ 改成按**严重度**取。
#    排序理由:真红最重;其次是被测代码抛了;再是闸自己没准备好/参照物没跑起来;
#    用法错误与不可归因最轻(它们都不是关于被测对象的判决)。
SEVERITY = [EXIT_COMPARE, EXIT_HISTORY, EXIT_OURS, EXIT_PREPARE, EXIT_ORIG,
            EXIT_USAGE, EXIT_UNRUNNABLE]


def _worst_rc(results):
    """(码, 是哪条链)。⛔ 不是「第一条非零」。"""
    bad = [(SEVERITY.index(rc) if rc in SEVERITY else 99, rc, c)
           for c, rc in results if rc != 0]
    if not bad:
        return 0, None
    bad.sort()
    return bad[0][1], bad[0][2]

CHAINS = {
    "diff": dict(py=STAGING, prepare="gate1_diff_prepare.py", orig="gate1_diff_run_orig.py",
                 ours="gate1_diff_run_ours.py",
                 ours_stdout=os.path.join(TESTING, "gate1_diff_ours_steps.jsonl"),
                 compare="gate1_diff_compare.py", history=HISTORY_DRIVER,
                 wipes=[os.path.join(TESTING, "gate1_diff_orig"),
                        os.path.join(TESTING, "gate1_diff_ours")]),
    "sovits": dict(py=STAGING, prepare="gate1_sovits_prepare.py", orig="gate1_sovits_run_orig.py",
                   ours="gate1_sovits_run_ours.py",
                   ours_stdout=os.path.join(TESTING, "gate1_sovits_ours_steps.jsonl"),
                   compare="gate1_sovits_compare.py", history=HISTORY_DRIVER,
                   wipes=[r"D:\MyDev\so-vits-svc\so-vits-svc\logs\gate1_sovits",
                          os.path.join(TESTING, "gate1_sovits_ours")]),
    "sovits_v2": dict(py=VENV, prepare="gate1_sovits_v2_prepare.py",
                      orig="gate1_sovits_v2_run_orig.py", ours="gate1_sovits_v2_run_ours.py",
                      ours_stdout=os.path.join(TESTING, "gate1_sovits_v2_ours_steps.jsonl"),
                      compare="gate1_sovits_v2_compare.py", history=HISTORY_DRIVER,
                      wipes=[r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc\logs\gate1_sovits_v2",
                             os.path.join(TESTING, "gate1_sovits_v2_ours")]),
    # ⛔ 声码器:`ours_stdout=None` —— 它没有 JSONL 步流(两侧都从 TB 取)⇒ tmp 落位保护
    #    对它不适用,而它恰好也是唯一没有历史对拍的一条(`compare_vs_history.CASES` 只有四条)。
    # ⛔⛔ S140:这条链的删除账**此前是错的,而且错在最贵的方向**。
    #    ⑴ `SingingVocoders\experiments\gate1_voc`(**3.66 GB**,S134 声码器**原版侧**
    #       唯一在盘证据 —— 头注此前写成「我方侧」,照那句话去备份会拷错一棵树)
    #       **不是 prepare 删的**,是 `gate1_vocoder_run_orig.py:75-77` 在 import 期
    #       **无条件** rmtree ⇒ 它在**不给 `--rebuild-fixtures`、不给 `GATE1_ALLOW_REBUILD`、
    #       一行警告都不打**的默认跑里就会没,而 S139 造的那道联锁只在 `if rebuild:` 里
    #       ⇒ **那道锁在这条链上是装饰**。⇒ 单开 `orig_wipes`,由 2_orig 段自己的联锁管。
    #    ⑵ prepare 只删 `gate1_vocoder\{ours,orig}`(5.11 GB),整棵 5.54 GB 里
    #       npz(24.9 MB)/ `pretrain_cpu.ckpt`(405.6 MB)/ 两份 07-06 的 *_run.log
    #       **一个都不删** ⇒ 打出来的体积必须等于真会没的体积。
    "vocoder": dict(py=STAGING, prepare="gate1_vocoder_prepare.py",
                    orig="gate1_vocoder_run_orig.py", ours="gate1_vocoder_run_ours.py",
                    ours_stdout=None, compare="gate1_vocoder_compare.py", history=HISTORY_DRIVER,
                    wipes=[r"D:\MyDev\TESTING\gate1_vocoder\ours",
                           r"D:\MyDev\TESTING\gate1_vocoder\orig"],
                    orig_wipes=[r"D:\MyDev\SingingVocoders\experiments\gate1_voc"]),
    # ⛔ orig_ok:上游 RVC 正常完训是 `os._exit(2333333)`(train.py:635),不是失败。
    #    S134 实测:第一版跑器把这次【成功】报成了 ORIG-FAILED —— 幸好它把真 rc 打了出来。
    #    ⚠ 这个数在两个地方读出来**不一样**:python 的 `returncode` 拿到 2333333,
    #      而 shell(`$?` / `$LASTEXITCODE`)拿到 `2333333 & 0xFF = 149`。
    #      本跑器是 python ⇒ 认 2333333;README:821 写的 149 是给写 shell 包装的人的。
    "rvc": dict(py=STAGING, prepare="gate1_prepare.py", orig="gate1_run_orig.py",
                orig_ok=(0, 2333333), ours="gate1_run_ours.py",
                ours_stdout=os.path.join(TESTING, "gate1_ours_steps.jsonl"),
                compare="gate1_compare.py", history=HISTORY_DRIVER,
                wipes=[r"D:\MyDev\RVC\RVC20240604Nvidia\logs\gate1",
                       os.path.join(TESTING, "gate1_ours")]),
}
ORDER = ["rvc", "sovits", "sovits_v2", "diff", "vocoder"]


def _refuse(msg):
    G._say("REFUSING: %s" % msg)
    sys.exit(EXIT_USAGE)


def _measure(path):
    """(件数, 字节)。⛔ 拿真实读数说话 —— 「会删掉一些东西」不是警告,「会删掉 3.5 GB
    / 6 个文件,那是声码器我方侧唯一证据」才是。"""
    if not os.path.isdir(path):
        return (0, 0) if not os.path.isfile(path) else (1, os.path.getsize(path))
    n = total = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return n, total


def report_wipes(chain, cfg, key="wipes", who="prepare"):
    """⛔ 逐条报「会没的东西」。措辞按**谁删的**分开 —— S140 之前这里对五条链统一写
    「这条链的 prepare 会 rmtree(首句就删,删在任何断言之前)」,而对声码器那条**两句都是假的**
    (删的是 run_orig;而 S139 已经把断言前移到 rmtree 之前)。一句错的账目会决定备份什么。
    """
    G._say("  ⛔ 这一跑的 **%s** 会 rmtree 下面这些(不可再生的历史证据,先拷走再来):" % who)
    grand = 0
    for p in cfg.get(key, []):
        n, b = _measure(p)
        grand += b
        G._say("       %-58s %5d 件 %9.1f MB%s"
               % (p, n, b / 1e6, "   ⚠ 不在盘上" if n == 0 else ""))
    G._say("     合计 %.2f GB" % (grand / 1e9))
    # ⛔ S140:**只报「删」不报「覆盖」的账目,对不可再生产物的保护只有半个。**
    #    四条链的 `ours_stdout` 不在任何 `wipes` 里 ⇒ `--dry-run` 一个字都不提,
    #    而 `step()` 在 rc==0 时 `os.replace(tmp, stdout_to)` 直接落位覆盖。
    #    盘上那四份 08-11 的 jsonl 是 S134 我方侧读数的**协议层**载体。
    out = cfg.get("ours_stdout")
    if out and key == "wipes":
        n, b = _measure(out)
        G._say("     ⚠ 另外还会**覆盖**(不是删):%s(%d 件 %.1f KB)" % (out, n, b / 1e3))
    return grand


def step(chain, name, py, script, out_dir, t0, flags, extra_args=()):
    """跑一段。返回 (rc, 秒数, 日志路径)。

    stdout_to 非空时 stdout 走【临时文件】再落位 —— 直接 `>` 会在脚本失败前就把历史产物
    截成 0 字节(S134 实测踩过一次)。
    ⛔ S139:日志本身也**拒绝覆盖** —— 见头注 ⑴。
    """
    stdout_to = CHAINS.get(chain, {}).get("ours_stdout") if name == "3_ours" else None
    log = os.path.join(out_dir, "%s.log" % name)
    if os.path.isfile(log) and os.path.getsize(log) > 0:
        _refuse("转录已存在且非空,拒绝覆盖:%s\n"
                "          (S139:一份能被下一次运行静默改写、而且能半新半旧的转录不满足"
                "『失败时必须留下能查的转录』)" % log)
    tmp = log + ".stdout.tmp"
    # ⛔ `GATE1_SKIPPED` 把「这一跑故意跳过了哪几段」透传给 compare —— 被跳过那一侧的产物
    #    **本来就不是本轮的**,它该走 declare_frozen + note_uncovered(结论是
    #    PASS-WITH-GAPS 或 exit 3),而不是被新鲜度判据当成一条陈货红。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", T0_ENV: "%.3f" % t0,
           SKIPPED_ENV: ",".join(flags)}
    env.pop("UTAI_DIAGNOSTICS", None)  # 诊断模式会往 .log 里多打行,让日志面不可比
    cmd = [py, "-u", os.path.join(GT, script)] + list(extra_args)
    t_start = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        # 照 run_gate0_chain.py:177 —— 转录必须自陈它是谁在什么条件下产的
        fh.write("$ %s\n  cwd=%s\n  %s=%.3f (%s)\n  %s=%s\n\n"
                 % (" ".join(cmd), REPO, T0_ENV, t0,
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                    SKIPPED_ENV, ",".join(flags) or "(none)"))
        fh.flush()
        if stdout_to:
            with open(tmp, "w", encoding="utf-8", errors="replace") as so:
                rc = subprocess.run(cmd, cwd=REPO, stdout=so, stderr=fh,
                                    text=True, env=env).returncode
        else:
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                text=True, env=env).returncode
    dt = time.time() - t_start
    if stdout_to:
        if rc == 0:
            os.replace(tmp, stdout_to)
        else:
            G._say("     (stdout 留在 %s —— **没有**落位,因为 rc=%d)" % (tmp, rc))
    G._say("  %-10s %-24s rc=%-8d %6.1fs  %s" % (chain, name, rc, dt, log))
    return rc, dt, log


def _tail(log, n=10):
    try:
        body = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()
    except OSError:
        return []
    return body[-n:]


def run_chain(chain, out, t0, rebuild, skip_orig, dry, flags, allow_uncovered=False):
    c = CHAINS[chain]
    G._say("\n===== chain %s  python=%s =====" % (chain, c["py"]))

    if not os.path.isfile(c["py"]):
        G._say("MISSING interpreter: %s" % c["py"])
        return EXIT_USAGE
    for key in ("prepare", "orig", "ours", "compare", "history"):
        # ⛔ S139:原来 `orig is None` 会**跳过整段并仍然打 PASS**。五条链没有一条是 None,
        #    所以那是一条死分支 —— 而任何人临时把某条置 None(「原版侧太慢先跳过」)
        #    拿到的就是 VERDICT PASS + exit 0,而那一跑根本没有参照物。
        if not c.get(key):
            G._say("VERDICT NO-REFERENCE — 这条链没有 %s 驱动 ⇒ 这一跑不构成一次对拍" % key)
            return EXIT_USAGE
        if not os.path.isfile(os.path.join(GT, c[key])):
            G._say("VERDICT MISSING-DRIVER — %s 不在:%s" % (key, c[key]))
            return EXIT_USAGE

    if rebuild:
        report_wipes(chain, c)
        if dry:
            G._say("  (--dry-run:不删)")
        elif os.environ.get(ALLOW_REBUILD_ENV) != "1":
            _refuse("--rebuild-fixtures 还需要 %s=1。\n"
                    "          上面那些是**不可再生的历史证据**(S134 五条链在盘的唯一副本),"
                    "先拷走再来。" % ALLOW_REBUILD_ENV)

    # ⛔⛔ S140:`orig_wipes` —— 由**原版侧驱动自己**在 import 期删掉的东西。
    #    声码器那条会无条件 rmtree 3.66 GB(S134 原版侧唯一在盘证据),而它**不受
    #    `--rebuild-fixtures` 管辖** ⇒ 此前 `run_gate1_chain.py vocoder`(不加任何参数)
    #    就能把它抹掉,终端上一个字的警告都没有,因为 `report_wipes` 只在 `if rebuild:` 里被调。
    #    ⇒ 只要这一跑的 plan 里有 2_orig,就用**同一道联锁**管它。
    if c.get("orig_wipes") and not skip_orig:
        report_wipes(chain, c, key="orig_wipes", who="原版侧驱动(import 期自清,⛔ 不受 --rebuild-fixtures 管)")
        if dry:
            G._say("  (--dry-run:不删)")
        elif os.environ.get(ALLOW_REBUILD_ENV) != "1":
            _refuse("这条链的**原版侧驱动**会在 import 期自清上面那个目录,而它不受"
                    "`--rebuild-fixtures` 管辖。\n"
                    "          要跑 2_orig 就得同样给 %s=1(或用 --skip-orig 跳过原版侧)。"
                    % ALLOW_REBUILD_ENV)

    if dry:
        for name, key in (("1_prepare", "prepare"), ("2_orig", "orig"),
                          ("3_ours", "ours"), ("4_compare", "compare"),
                          ("5_history", "history")):
            if name == "1_prepare" and not rebuild:
                G._say("  WOULD-SKIP 1_prepare               (默认不重建夹具)")
                continue
            if name == "2_orig" and skip_orig:
                G._say("  WOULD-SKIP 2_orig                  (--skip-orig)")
                continue
            G._say("  WOULD-RUN  %-22s %s" % (name, c[key]))
        return 0

    os.makedirs(out, exist_ok=True)
    plan = [("3_ours", "ours"), ("4_compare", "compare"), ("5_history", "history")]
    if not skip_orig:
        plan.insert(0, ("2_orig", "orig"))
    if rebuild:
        plan.insert(0, ("1_prepare", "prepare"))

    for name, kind in plan:
        # ⛔ `--allow-uncovered` 只透传给 compare —— 它是「我知道这一轮有缺口,仍然接受」的
        #    显式表态,而 gate0_guard.finish 的默认是**有缺口就 exit 3**。
        extra = ()
        if kind == "history":
            extra = ("--chain", chain)
        if allow_uncovered and kind in ("compare", "history"):
            extra = extra + ("--allow-uncovered",)
        rc, _dt, log = step(chain, name, c["py"], c[kind], out, t0, flags, extra)
        if kind == "orig" and rc in c.get("orig_ok", (0,)):
            continue
        # ⛔ S139:UNRUNNABLE 对**每一段**都成立,不只 compare —— 照 `run_gate0_chain.py:204-209`。
        #    起因是 `gate1_run_orig.py` 那三处 `sys.exit(<字符串>)`:CPython 对字符串参数退 **1**,
        #    而 1 在 orig 段会被这里读成 `ORIG-FAILED`(参照物没跑起来)—— 可它自己说的是
        #    「这是【闸没准备好】,不是被测对象的问题」。**一条红,两种归因**,正是 S129 铁律要拆开的。
        if rc == G.EXIT_UNRUNNABLE:
            G._say("VERDICT UNRUNNABLE (读数不可归因,这不是一次判定) — %s" % log)
            for ln in _tail(log, 8):
                G._say("   " + ln)
            return EXIT_UNRUNNABLE
        if rc != 0:
            label = {"prepare": "PREPARE-FAILED (闸自己没准备好)",
                     "orig": "ORIG-FAILED (参照物没跑起来 ≠ 我们的代码不对)",
                     "ours": "OURS-FAILED (被测代码抛了)",
                     "compare": "COMPARE-FAILED (★ 这一种才是「被测的东西不对」)",
                     "history": "HISTORY-DRIFT (数值自 07-07/07-17 与 08-11 起变了 "
                                "≠ 今天的移植不对 —— 两条判据问的是两个问题)"}[kind]
            G._say("VERDICT %s rc=%d — %s" % (label, rc, log))
            for ln in _tail(log):
                G._say("   " + ln)
            return KIND_EXIT[kind]
        if kind in ("compare", "history"):
            G._say("---- %s ----" % kind)
            for ln in _tail(log, 12):
                G._say("   " + ln)

    # ⛔ S139:这一跑跳过了什么,必须写在**结论里**,不许只活在调用者的记忆里。
    G._say("VERDICT PASS (%s)%s" % (chain, ("  [SKIPPED: %s]" % ", ".join(flags)) if flags else ""))
    return 0


def _resolve_t0(runs_root, explicit, new_session, dry=False):
    """会话级 t0。⛔ 整块照 `run_gate0_chain.py:249-265` —— 尤其那条 **6 小时上界**:
    它是 S135 二审自己买回来的,只搬「传 t0」会把一条已经修过的洞重新引进来
    (`--runs` 落在固定路径 ⇒ 第二次调用起,凡是落在 [旧 t0, now] 窗口里的陈货一律判 FRESH)。
    """
    os.makedirs(runs_root, exist_ok=True)
    t0_file = os.path.join(runs_root, "T0.txt")
    if explicit is not None:
        G._say("%s = %.3f (显式指定 —— ⚠ 你自己负责它早于本轮全部产物)" % (T0_ENV, explicit))
        return explicit
    if not new_session and os.path.isfile(t0_file):
        cand = float(open(t0_file, encoding="utf-8").read().strip())
        age = time.time() - cand
        if age <= SESSION_MAX_AGE:
            G._say("%s = %.3f (复用 %s,%.1f 分钟前记下的)" % (T0_ENV, cand, t0_file, age / 60.0))
            return cand
        G._say("⛔ %s 里的 t0 已经 %.1f 小时前了(上界 %.0f 小时)—— **不复用**,"
               "否则这期间产生的任何陈货都会被判成本轮产物。改起新会话。"
               % (t0_file, age / 3600.0, SESSION_MAX_AGE / 3600.0))
    # 让 t0 严格早于任何产物 —— 文件系统时间戳分辨率与时钟漂移会让「同一秒产出的文件」
    # mtime 略早于 t0,那会报成一条假的「陈货」。
    t0 = time.time() - 2.0
    # ⛔⛔ S140:**`--dry-run` 不许写盘。** 一次「只说会做什么」的调用不许改掉整场会话的
    #    新鲜度基准 —— 而这里原来是无条件写的:`main()` 在 `if dry` 的早退**之前**就调它。
    #    ⚠ 这条 S139 **在 gate0 跑器上当场买到并修掉了**(`run_gate0_chain.py:329-333`),
    #      而它同一场自己新建的这个 gate1 跑器**没有回移** —— 「修在孪生脚本」第 N 次。
    if dry:
        G._say("%s = %.3f (⚠ --dry-run ⇒ **没有**写进 %s)" % (T0_ENV, t0, t0_file))
        return t0
    with open(t0_file, "w", encoding="utf-8") as f:
        f.write("%.3f\n" % t0)
    G._say("%s = %.3f (新会话,已记入 %s)" % (T0_ENV, t0, t0_file))
    return t0


class _Parser(argparse.ArgumentParser):
    """⛔ S139:argparse 默认用法错误退 **2**,而 2 在这套出口码里是「prepare 失败」。
    一个打错命令的人会拿到一条「闸自己没准备好」的归因。⇒ 改成 5。"""

    def error(self, message):
        G._say("usage error: %s" % message)
        sys.exit(EXIT_USAGE)


def main(argv=None):
    ap = _Parser(prog="run_gate1_chain")
    ap.add_argument("chain", nargs="?", choices=sorted(CHAINS) + ["all"])
    ap.add_argument("--rebuild-fixtures", action="store_true",
                    help="跑 prepare(⛔ 它首句就 rmtree 双侧 expdir,还要 %s=1)" % ALLOW_REBUILD_ENV)
    ap.add_argument("--skip-orig", action="store_true",
                    help="原版侧已经跑过且产物还在时用(⛔ 与 --rebuild-fixtures 互斥)")
    ap.add_argument("--runs", default=RUNS_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--new-session", action="store_true")
    ap.add_argument("--allow-uncovered", action="store_true",
                    help="接受这一轮的零覆盖缺口(透传给 compare)—— ⛔ 有缺口的默认是 exit 3")
    ap.add_argument("--selftest", action="store_true",
                    help="阴性对照:用桩脚本把每一档出口码真触发一次")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.chain:
        _refuse("要跑哪条链?%s(或 all)" % ", ".join(sorted(CHAINS)))
    if args.skip_orig and args.rebuild_fixtures:
        _refuse("--skip-orig 与 --rebuild-fixtures 互斥 —— prepare 会把原版侧产物一起删掉,"
                "跳过 orig 就再也补不回来了")

    flags = []
    if not args.rebuild_fixtures:
        flags.append("prepare")
    if args.skip_orig:
        flags.append("orig")

    t0 = _resolve_t0(args.runs, args.t0, args.new_session, dry=args.dry_run)
    G._say("           = %s" % time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)))
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    chains = ORDER if args.chain == "all" else [args.chain]

    # ⛔⛔ S140:**全局预检**。此前 `report_wipes` 与破坏是**交错**的 —— `all` 跑到
    #    声码器那 9.20 GB 的行出现在屏幕上时,前四条链的 15.07 GB 已经删掉了;
    #    放行(`GATE1_ALLOW_REBUILD`)同样是逐链的。⇒ 头注 ⑵ 承诺的「先把要删的体积
    #    逐条打出来」此前**只对第一条链成立**。这里在任何 run_chain 之前先把全账打完。
    if (args.rebuild_fixtures or not args.skip_orig) and len(chains) > 1:
        G._say("\n===== 预检:这一跑总共会毁掉多少(⛔ 在任何一条链动手之前)=====")
        total = 0
        for c in chains:
            cfg = CHAINS[c]
            for key, who in (("wipes", "prepare"), ("orig_wipes", "原版侧驱动自清")):
                if key == "wipes" and not args.rebuild_fixtures:
                    continue
                if key == "orig_wipes" and (args.skip_orig or not cfg.get("orig_wipes")):
                    continue
                for p in cfg.get(key, []):
                    n, b = _measure(p)
                    total += b
                    G._say("  %-10s %-10s %-56s %5d 件 %9.1f MB" % (c, who, p, n, b / 1e6))
        G._say("  ---- 全局合计 %d B = %.2f GB(= %.3f GiB)----" % (total, total / 1e9, total / 2 ** 30))

    worst, results = 0, []
    for c in chains:
        out = os.path.join(args.runs, c, stamp)
        rc = run_chain(c, out, t0, args.rebuild_fixtures, args.skip_orig, args.dry_run,
                       flags, args.allow_uncovered)
        results.append((c, rc))
    # ⛔ S140:按**严重度**取,不是「第一条非零」—— 见 SEVERITY 的注释。
    worst, why = _worst_rc(results)
    G._say("\n===== 汇总 (t0=%.3f, 转录 %s) =====" % (t0, os.path.join(args.runs, "<chain>", stamp)))
    for c, rc in results:
        G._say("  %-10s rc=%d %s" % (c, rc, "PASS" if rc == 0 else "^^^"))
    if flags:
        G._say("  ⚠ 这一跑跳过了:%s ⇒ 相关产物**不保证是本轮的**" % ", ".join(flags))
    G._say("  EXIT=%d%s" % (worst, "" if why is None else "  因为 %s(按严重度取,不是第一条非零)" % why))
    return worst


# --------------------------------------------------------------------------- 自检
_STUB = ("import sys\n"
         "open(sys.argv[0] + '.ran', 'w').close()\n"
         "print('stub %s speaking: 桩脚本说的中文')\n"
         "sys.exit(%d)\n")


def _selftest():
    """⛔ 每一档出口码用**桩脚本**真触发一次。

    S129 铁律的同族:一条从没被执行过的错误分支就是一条空判据 —— 而这个跑器的六档
    出口码里,除了 0 与 3(S134 撞过一次 `os._exit(2333333)`)之外**从没被执行过**。
    """
    import tempfile
    fails = []
    saved_chains = dict(CHAINS)
    saved_gt = globals()["GT"]

    def scenario(name, codes, want, rebuild=None, skip_orig=False, allow_rebuild=True,
                 **chain_kw):
        # ⚠ 工装自己踩过一次:`kw.pop` 写在 `dict(**kw)` 之后 ⇒ 每个场景都走进了重建分支,
        #   八个场景一起报 exit 5。**是自检抓住的,不是我看出来的** —— 而这正是
        #   S128 血训「每种闸都要有基线自检:工装报的 RED 有三分之一是假的」的正面例子。
        if rebuild is None:
            rebuild = "prepare" in codes
        td = tempfile.mkdtemp(prefix="gate1_runner_selftest_")
        old_env = os.environ.get(ALLOW_REBUILD_ENV)
        try:
            if allow_rebuild:
                os.environ[ALLOW_REBUILD_ENV] = "1"
            else:
                os.environ.pop(ALLOW_REBUILD_ENV, None)
            globals()["GT"] = td
            names = {}
            for kind, code in codes.items():
                fn = "stub_%s.py" % kind
                with open(os.path.join(td, fn), "w", encoding="utf-8") as f:
                    f.write(_STUB % (kind, code))
                names[kind] = fn
            CHAINS.clear()
            # ⛔ S140:**第 5 段也要给桩**。不给的话 :296 的驱动存在性检查会先返回
            #    EXIT_USAGE ⇒ 每一个场景都在测「缺驱动」而不是它自己那一档,
            #    而其中几条恰好也期望 5 ⇒ **红了、码也对,但理由变了**。
            CHAINS["st"] = dict(py=sys.executable, prepare=names.get("prepare"),
                                orig=names.get("orig"), ours=names.get("ours"),
                                compare=names.get("compare"),
                                history=names.get("history"), ours_stdout=None,
                                wipes=[], **chain_kw)
            got = run_chain("st", os.path.join(td, "runs"), time.time() - 2.0,
                            rebuild, skip_orig, False, [])
            if got == want:
                G._say("  ok   %-34s -> exit %d" % (name, got))
            else:
                fails.append("%s 应该 exit %d,实际 %d" % (name, want, got))
        except SystemExit as e:
            got = e.code if isinstance(e.code, int) else 0
            if got == want:
                G._say("  ok   %-34s -> exit %d (SystemExit)" % (name, got))
            else:
                fails.append("%s 应该 exit %d,实际 SystemExit(%r)" % (name, want, e.code))
        finally:
            CHAINS.clear()
            CHAINS.update(saved_chains)
            globals()["GT"] = saved_gt
            if old_env is not None:
                os.environ[ALLOW_REBUILD_ENV] = old_env
            else:
                os.environ.pop(ALLOW_REBUILD_ENV, None)
            shutil.rmtree(td, ignore_errors=True)

    ok = dict(prepare=0, orig=0, ours=0, compare=0, history=0)
    scenario("四段全过", ok, 0)
    scenario("prepare 失败", dict(ok, prepare=7), EXIT_PREPARE)
    scenario("原版侧失败", dict(ok, orig=9), EXIT_ORIG)
    scenario("我方侧抛了", dict(ok, ours=9), EXIT_OURS)
    scenario("compare 判负", dict(ok, compare=1), EXIT_COMPARE)
    # ⛔ S140:第 5 段 —— 历史漂了退 **7**,不许和「移植不对」共用 1
    scenario("历史对拍判负", dict(ok, history=1), EXIT_HISTORY)
    # ⛔ 「不可归因」对**每一段**都成立,不只 compare —— 起因是 `gate1_run_orig.py` 那三条
    #    前置(它们自称「闸没准备好」,而退 1 会被读成「参照物没跑起来」)。逐段各触发一次。
    for kind in ("prepare", "orig", "ours", "compare", "history"):
        scenario("%s 判不可归因(退 3)" % kind, dict(ok, **{kind: G.EXIT_UNRUNNABLE}),
                 EXIT_UNRUNNABLE)
    # ⛔ 上游 RVC 正常完训:配了 orig_ok 要放行,没配要判 ORIG-FAILED —— 两条一起才是判据
    scenario("上游 2333333 + orig_ok", dict(ok, orig=2333333), 0, orig_ok=(0, 2333333))
    scenario("上游 2333333 无 orig_ok", dict(ok, orig=2333333), EXIT_ORIG)
    # ⛔ 死分支:没有参照物绝不许退 0
    scenario("没有 orig 驱动", dict(prepare=0, ours=0, compare=0, history=0), EXIT_USAGE)
    scenario("没有 history 驱动", dict(prepare=0, orig=0, ours=0, compare=0), EXIT_USAGE)

    # 转录不许被覆盖
    td = tempfile.mkdtemp(prefix="gate1_runner_selftest2_")
    try:
        globals()["GT"] = td
        with open(os.path.join(td, "stub_ours.py"), "w", encoding="utf-8") as f:
            f.write(_STUB % ("ours", 0))
        out = os.path.join(td, "runs")
        os.makedirs(out)
        with open(os.path.join(out, "3_ours.log"), "w", encoding="utf-8") as f:
            f.write("S134-HISTORICAL-TRANSCRIPT-DO-NOT-LOSE\n")
        CHAINS.clear()
        CHAINS["st"] = dict(py=sys.executable, prepare="stub_ours.py", orig="stub_ours.py",
                            ours="stub_ours.py", compare="stub_ours.py",
                            history="stub_ours.py", ours_stdout=None, wipes=[])
        try:
            run_chain("st", out, time.time() - 2.0, False, False, False, [])
            fails.append("转录覆盖:应该被拒绝,却跑过去了")
        except SystemExit as e:
            body = open(os.path.join(out, "3_ours.log"), encoding="utf-8").read()
            if e.code == EXIT_USAGE and "DO-NOT-LOSE" in body:
                G._say("  ok   %-34s -> exit 5 且历史转录原样还在" % "拒绝覆盖已有转录")
            else:
                fails.append("转录覆盖:exit=%r,历史内容还在=%s"
                             % (e.code, "DO-NOT-LOSE" in body))
    finally:
        CHAINS.clear()
        CHAINS.update(saved_chains)
        globals()["GT"] = saved_gt
        shutil.rmtree(td, ignore_errors=True)

    # 用法错误不许撞上 prepare 的 2
    for argv, want in ((["nosuchchain"], EXIT_USAGE), ([], EXIT_USAGE),
                       (["rvc", "--skip-orig", "--rebuild-fixtures"], EXIT_USAGE)):
        try:
            main(argv)
            fails.append("用法错误 %r 应该 exit %d,却没退出" % (argv, want))
        except SystemExit as e:
            got = e.code if isinstance(e.code, int) else 0
            if got == want:
                G._say("  ok   %-34s -> exit %d" % ("用法错误 %r" % (argv,), got))
            else:
                fails.append("用法错误 %r 应该 exit %d,实际 %r" % (argv, want, e.code))

    # ⛔ --rebuild-fixtures 没有环境变量时必须被拒;而有环境变量时 prepare 必须**真的跑到**
    scenario("重建夹具但没有放行环境变量", ok, EXIT_USAGE, rebuild=True, allow_rebuild=False)
    scenario("默认不跑 prepare(它是删东西那一段)", dict(ok, prepare=7), 0, rebuild=False)
    # ⛔ S140:`orig_wipes` —— 原版侧驱动 import 期自清的目录,**不受 --rebuild-fixtures 管**。
    #    两条一起才是判据:没放行必须被拒(而且**默认跑就要拒**,不是只有 rebuild 时),
    #    放行了必须跑得过去。
    scenario("原版侧会自清但没放行(默认跑)", ok, EXIT_USAGE, rebuild=False,
             allow_rebuild=False, orig_wipes=[os.path.join(HERE, "__nonexistent_orig_wipe__")])
    scenario("原版侧会自清 + 放行", ok, 0, rebuild=False, allow_rebuild=True,
             orig_wipes=[os.path.join(HERE, "__nonexistent_orig_wipe__")])
    scenario("--skip-orig ⇒ 原版侧自清那条联锁不该拦", ok, 0, rebuild=False,
             skip_orig=True, allow_rebuild=False,
             orig_wipes=[os.path.join(HERE, "__nonexistent_orig_wipe__")])

    # ⛔⛔ S140:`all` 的退出码按**严重度**取,不是「第一条非零」。
    #    此前 rvc 排第一 + 一条 exit 6 会把后面任何一条真红(exit 1)整个盖住。
    for name, res, want in (
            ("真红排在不可归因之后", [("a", EXIT_UNRUNNABLE), ("b", EXIT_COMPARE)], EXIT_COMPARE),
            ("真红 vs 历史漂了", [("a", EXIT_HISTORY), ("b", EXIT_COMPARE)], EXIT_COMPARE),
            ("历史漂了 vs 不可归因", [("a", EXIT_UNRUNNABLE), ("b", EXIT_HISTORY)], EXIT_HISTORY),
            ("真红排在最前", [("a", EXIT_COMPARE), ("b", EXIT_UNRUNNABLE)], EXIT_COMPARE),
            ("我方抛了 vs 用法错误", [("a", EXIT_USAGE), ("b", EXIT_OURS)], EXIT_OURS),
            ("全过", [("a", 0), ("b", 0)], 0)):
        got, _why = _worst_rc(res)
        if got == want:
            G._say("  ok   %-34s -> exit %d" % ("汇总严重度:" + name, got))
        else:
            fails.append("汇总严重度 %s 应该 %d,实际 %d" % (name, want, got))

    # ⛔⛔ S140:`--dry-run` **不许写盘** —— 一次「只说不做」的调用不许改掉整场会话的
    #    新鲜度基准。S139 在 gate0 跑器上买到并修了这条,而它同一场建的这个跑器没有回移。
    td = tempfile.mkdtemp(prefix="gate1_runner_selftest3_")
    try:
        t0_file = os.path.join(td, "T0.txt")
        _resolve_t0(td, None, True, dry=True)
        if os.path.exists(t0_file):
            fails.append("--dry-run 写了 T0.txt(它不许改变下一次判定的基准)")
        else:
            G._say("  ok   %-34s -> 没有写 T0.txt" % "--dry-run 不落盘")
        _resolve_t0(td, None, True, dry=False)
        if os.path.isfile(t0_file):
            G._say("  ok   %-34s -> 写了 T0.txt" % "真跑落盘(对照)")
        else:
            fails.append("真跑没有写 T0.txt")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    G._say()
    if fails:
        for f in fails:
            G._say("  FAIL %s" % f)
        G._say("run_gate1_chain 自检: FAILED(%d)" % len(fails))
        return G.EXIT_SELFTEST
    G._say("run_gate1_chain 自检: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
