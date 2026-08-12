"""gate0 判据护栏 —— 让这套关卡的【绿有意义、红能归因】。

S135(§F7 笔 2)新建。起因是侦察 + 逐面对抗核验查明:**gate0 今天跑出来的绿说明不了
今天的 HEAD**,而且这不是"缓存没清"这种操作问题,是判据本身的三个结构缺口:

  ⑴ **空集假 PASS**。`gate0_compare.py` 的 `compare_wav_dir` / `compare_feat` 在两侧
     目录都为空时:文件集合 set()==set() 通过、循环 0 次 ⇒ worst 停在初值 0.0、
     cmin 停在 1.0 ⇒ 打印 `max|Δ|=0.000e+00, min_cos=1.000000000` 并 PASS。
     `gate0_sovits_compare.py` 的 A/44k、A/soft、A/spec、A/vol、C4、C5 同病;
     两个 `*_c_resample.py` 在切片目录为空时打印 `[PASS] ... 0 文件` 并 exit 0。
     ⛔ 直接后果:**删掉目录 = 正确地红,清空目录 = 假 PASS**,两种清法后果相反。
     ⚠ 这条缺陷 S68(`f599e76`, 2026-07-17)的对抗审查已判为 [major] 并在孪生脚本
     `gate0_sovits_v2_compare.py:72-75` 修好,**从没回移到 RVC 与 4.1**;同一笔 commit
     碰过 `gate0_sovits_compare.py`,只改了 `aux→auxiliary` 一个词。开了 25 天。

  ⑵ **没有任何非同源守卫**。全套 gate0 的 compare 里零个 mtime / 时间戳 / run-id 检查
     ⇒ 闸完全不知道自己读的是今天的产物还是七月的。而两侧的提取阶段全是 skip-if-exists,
     所以"跑一遍 gate0"的默认结果就是拿陈货比陈货。
     ⛔ 这条已经有**实名受害者**:`4e525ef`(S43, 2026-07-07)的 commit message 写着
     「全套关卡 2.11 轴复跑 gate0×4 ... 全 PASS 逐位吻合 S37-40」,而 RVC gate0 里唯一
     吃 torch 的 stage 是 f0,那一次 f0 一个都没重算(盘上 mtime + `train.log` 里没有
     `Loading rmvpe model` 那一行)。sovits 4.1 的 ①② 自 2026-07-07 起也没再跑过。

  ⑶ **「跑不起来」与「被测的东西不对」共用同一个非零退出码**(S129 铁律第一条)。
     产物缺失走裸 assert / FileNotFoundError / ZeroDivisionError,给的是 traceback,
     而真红给的是 `sys.exit(1)` —— 第二次出现一定被耸肩带过。

⛔ 本模块**不是**把"陈货"一律判红:gate0 的原版侧【本来就该是冻结参照】——
   它是「不变输入 × 不变上游代码」的函数,而且 `rvc_orig/3_feature768` 是 CUDA/TF32
   产物、不保证逐位可复现,重算反而会换掉参照物。所以这里区分两种角色:
     · `require_fresh(...)`  我方侧产物 —— 必须是本轮算出来的
     · `declare_frozen(...)` 冻结参照   —— 必须存在且非空,并把它的日期打进转录自证
     · `note_uncovered(...)` 两侧都冻结 ⇒ 这条判据本轮零覆盖,必须响亮记账

退出码(四档,与 S129 铁律对齐):
    0  ALL PASS(且没有零覆盖的判据,除非显式 --allow-uncovered)
    1  被测的东西不对(真红)
    3  闸自己跑不起来 / 这一轮的读数不可归因(缺产物、缺参照、没有 run token、样本不足、
       判据零覆盖而没有显式放行)
    4  本模块自检失败

⛔⛔⛔ S139 追加的第四条(而它比上面三条更靠底层):**这台闸自己在本机说不出话。**
   本模块**全部**归因文字都是中文,而本机 `locale.getpreferredencoding(False)` 是 **cp932**、
   `sys.flags.utf8_mode` 是 **0** ⇒ 只要 stdout 不是交互式控制台(重定向到文件 / 进管道 /
   被别的工具抓 —— 而 README 教的正是手敲命令 + `>`),`print` 就 **UnicodeEncodeError**,
   进程以 **rc=1** 收场,而 **1 在这套出口码里的含义是【被测的东西不对】**。
   S139 实测(`--selftest` 里有常驻的阴性对照):
     · `GateUnrunnable` 本该 exit 3 → 实测 **exit 1**
     · `note_uncovered` ⇒ 零覆盖 本该 exit 3 → 实测 **exit 1**
     · 真红 exit 1(码碰巧对了,但 FAIL 那一行**一个字都没打出来**,连 `run()` 的兜底
       `except Exception` 自己也炸在同一个地方)
     · **`ALL PASS` 是纯 ASCII,所以它活着**
   ⇒ **形状是最坏的那一种:这台闸在非 UTF-8 控制台上只说得出「通过」。**
   ⚠ 跑器(`run_gate0_chain.py:173`)给子进程塞了 `PYTHONIOENCODING=utf-8`,**掩盖了它** ——
   所以这条缺陷在「照跑器跑」时看不见,只在照 README 手敲时发作。
   ⛔ **不许拿「跑器会设 PYTHONIOENCODING」当解决方案**:那是在跑器里补一个本该在判据里的洞。
   ⇒ 修法两层(与 S138 给 `memory_hook_guard.py` 的解药同形):
     ① `_ensure_utf8()` 在 import 时把 stdout/stderr 钉成 UTF-8;
     ② **每一行输出都走 `_say()`**,它有三级退路,最后一级是纯 ASCII 转义 ——
        因为「stdout 已经被别人换成一个敌对的流」是 ① 够不着的。
   ⭐ **通用规矩(S138 立)**:凡是「出问题时应该安静」的机制,**它的『安静』和它『坏掉了』
      长得一模一样** ⇒ 必须有一条阴性对照证明它在**敌对条件下**仍然开得了口。

自检:`python gate0_guard.py --selftest`
"""
import hashlib
import io
import math
import os
import subprocess
import sys
import time

EXIT_PASS = 0
EXIT_RED = 1
EXIT_UNRUNNABLE = 3
EXIT_SELFTEST = 4

T0_ENV = "GATE0_T0"

_uncovered = []


def _ensure_utf8():
    """把 stdout/stderr 钉成 UTF-8。⛔ 见头注第四条:不做这件事,这台闸只说得出 ALL PASS。

    在 import 时调用(而不是留给每个调用者)—— 八个 gate0 脚本里已经有一个
    (`gate0_diff_compare.py`)自己没 reconfigure,而「谁记得加那一行」正是
    这类缺陷唯一的防线时,它就一定会漏。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        if enc == "utf8":
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:                       # noqa: BLE001
            # 流不是 TextIOWrapper(被换成 StringIO / 被别的工具包过)⇒ 交给 _say 的退路
            pass


_ensure_utf8()


def _say(msg=""):
    """输出一行 —— **三级退路**,任何一级成功就返回,绝不让「说话」这件事本身把闸弄红。

    ⛔ 为什么不能只靠 `_ensure_utf8()`:调用者完全可能在 import 之后把 `sys.stdout`
       换成一个编码敌对的流(pytest 捕获 / 别的 harness / 显式 reconfigure 回去),
       而那一刻正是这台闸最需要说出「exit 3」的时候。
    退路顺序:① 直接 print ② 绕过文本层写 stdout.buffer 的 UTF-8 字节
              ③ 纯 ASCII 转义(难看,但**说得出来**,而且退出码是对的)
    """
    try:
        print(msg)
        return
    except Exception:                           # noqa: BLE001
        pass
    try:
        buf = getattr(sys.stdout, "buffer", None)
        if buf is not None:
            buf.write(msg.encode("utf-8", "backslashreplace") + b"\n")
            buf.flush()
            return
    except Exception:                           # noqa: BLE001
        pass
    try:
        print(msg.encode("unicode_escape").decode("ascii", "replace"))
    except Exception:                           # noqa: BLE001
        pass                                    # 三级都塌了 ⇒ 宁可无声也不许改变退出码


class GateUnrunnable(RuntimeError):
    """这一轮的读数不可归因。⛔ 绝不许被读成『通过』。"""


def _fmt(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def read_t0(gate_name, env=None):
    """本轮起始时刻(epoch 秒),由跑器通过环境变量传进来(gate0 用 `GATE0_T0`)。

    ⛔ 没有它就没有新鲜度判据 ⇒ 分不出产物是今天算的还是七月的
       ⇒ 这一轮不构成一次判定。响亮地判 UNRUNNABLE,不给默认值。

    ⚠ S139:`env` 可以换名 —— gate1 那一层用 `GATE1_T0`(`gate1_guard.T0_ENV`)。
      **必须分层**:同一个 shell 里先跑 gate0 会话再跑 gate1,复用同一个变量名会让
      gate1 读到几小时前的 t0,而这种失效静默、且只会让判据**变宽**。
    """
    env = env or T0_ENV
    raw = os.environ.get(env)
    if not raw:
        raise GateUnrunnable(
            "没有 %s ⇒ 无法判断读到的产物是不是本轮算的 ⇒ 这一轮的读数不可归因。\n"
            "       用对应的 run_gate*_chain.py 跑,或手动 set %s=<epoch 秒>。\n"
            "       (%s)" % (env, env, gate_name)
        )
    try:
        val = float(raw)
    except ValueError:
        raise GateUnrunnable("%s=%r 不是一个 epoch 秒" % (env, raw))
    # ⛔ S139:`float()` 过得去 ≠ 是一个合法的 t0。三种退化值各有各的伤,而**最阴的是 0**:
    #    `t0=0` ⇒ 1970-01-01 ⇒ 盘上任何陈货都 `mtime >= t0` ⇒ 新鲜度判据整体失效,
    #    而它打出来的是一行**格式完全合格的绿**(`[FRESH] ... t0=1970-01-01T08:00:00`),
    #    与真绿只差一个日期。负数 / nan / inf 则在 `time.localtime` 里抛**裸异常**,
    #    被 `run()` 归到「闸自己炸了」—— 码对了但措辞错了:那是输入不合法,不是闸炸了。
    #    ⚠ 尤其 nan:`m < nan` 恒为 False ⇒ **stale 判据先被整个跳过**,才在打印时炸。
    #    ⇒ 一律在入口判死,归 UNRUNNABLE。`run_gate0_chain.py` 的 `--t0` 一个命令就能走到这里。
    if not math.isfinite(val) or val <= 0:
        raise GateUnrunnable(
            "%s=%r 不是一个合法的 t0(要求有限且 > 0)。\n"
            "       ⛔ t0=0 会让【任何】陈货都通过新鲜度判据,而且打出来的是一行格式合格的绿;\n"
            "         负数 / nan / inf 会让 stale 判据静默失效再炸在打印上。" % (env, raw)
        )
    return val


def collect(root, subs, suffixes=None):
    """列出 root/sub 下匹配后缀的文件 [(path, mtime)]。subs 可以是 [""] 表示 root 自己。"""
    out = []
    for sub in subs:
        d = os.path.join(root, sub) if sub else root
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if not os.path.isfile(p):
                continue
            if suffixes and not any(n.endswith(s) for s in suffixes):
                continue
            out.append((p, os.path.getmtime(p)))
    return out


def dirhash(root, subs, suffixes=None):
    """目录内容的可复算指纹(按相对路径排序,喂 路径 + 字节)。"""
    h = hashlib.sha256()
    for p, _m in collect(root, subs, suffixes):
        h.update(os.path.relpath(p, root).replace("\\", "/").encode("utf-8"))
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
    return h.hexdigest()


def require_fresh(label, root, subs, t0, minimum, suffixes=None,
                  classes=None, per_class=None):
    """我方侧产物:必须存在、够数、而且**全部是本轮算出来的**。

    ⛔ 别拿「stage 的 done 计数走满」当判据 —— 三处 reporter.stage 全在 skip 的
       continue 之前(`extract_f0.py:86` vs `:91`),全跳过时 done 照样从 0 走到满,
       `:99` 还无条件再报一次满。唯一可用的是产物 mtime。
    """
    items = collect(root, subs, suffixes)
    if len(items) < minimum:
        raise GateUnrunnable(
            "%s: 只有 %d 件(下限 %d)—— 这不是一次比较,是一次空转。root=%s subs=%s"
            % (label, len(items), minimum, root, subs)
        )
    # ⛔ S135 二审自己抓出来的洞(M12):合计件数下限**挡不住整整一类产物消失**。
    #    实测算术:sovits 那条写的是 MIN_SLICES*4 = 120,而目录里是 33 片 × 5 类 = 165
    #    ⇒ 少掉一整类(33 件)后剩 132 >= 120,照样判 FRESH。
    #    ⇒ 一个目录里有多类产物时,必须**逐类**报下限。
    if classes:
        pc = per_class if per_class is not None else minimum
        counts = {}
        for c in classes:
            counts[c] = sum(1 for p, _m in items if os.path.basename(p).endswith(c))
        short = {c: n for c, n in counts.items() if n < pc}
        if short:
            raise GateUnrunnable(
                "%s: 逐类下限没走满(每类应 >= %d)—— 缺的类:%s;全表:%s\n"
                "       ⇒ 合计件数够不代表每一类都在:整整一类产物消失时合计仍可能达标。"
                % (label, pc, short, counts)
            )
    # ⛔ 同族(S134 的 .part 血训):0 字节的新鲜文件照样是新鲜的,但它不是产物。
    empty = [p for p, _m in items if os.path.getsize(p) == 0]
    if empty:
        raise GateUnrunnable(
            "%s: %d 件是 0 字节(崩在写一半 / 占位)—— 新鲜但不是产物:%s"
            % (label, len(empty), [os.path.basename(p) for p in empty[:5]])
        )
    stale = [(p, m) for p, m in items if m < t0]
    if stale:
        stale.sort(key=lambda x: x[1])
        head = "; ".join("%s(%s)" % (os.path.basename(p), _fmt(m)) for p, m in stale[:3])
        raise GateUnrunnable(
            "%s: %d/%d 件不是本轮产物(t0=%s)。最旧的几件:%s\n"
            "       ⇒ 提取阶段被 skip-if-exists 跳过了,清干净再跑。"
            % (label, len(stale), len(items), _fmt(t0), head)
        )
    newest = max(m for _p, m in items)
    oldest = min(m for _p, m in items)
    _say("[FRESH] %s: %d 件,全部 mtime >= t0=%s(实测 %s ~ %s)"
         % (label, len(items), _fmt(t0), _fmt(oldest), _fmt(newest)))
    return items


def declare_frozen(label, root, subs, minimum, why, suffixes=None, expect_sha=None):
    """冻结参照:**故意**不重算的一侧。必须存在、够数、**而且还是同一份东西**。

    理由要写进 why —— 转录里看得见「这一侧是冻结的、为什么冻结、它有多旧」,
    下一个人才不会把「参照没变」误读成「本轮验过了」。

    ⛔ S135 二审抓出来的洞(M11):这个函数原本**只判件数下限,然后把 mtime 打印出来** ——
       而**打印是汇报不是判据**。参照侧被重跑、被从别的备份还原、被指到另一个目录,
       三种情况一律照常返回,唯一差别是那一行的日期变了。
       ⇒ 加 `expect_sha`:给了就**必须逐字节对上**。
       (同一批代码里本来就有正确形状:`gate0_rebuild_b2_ours.py` 的 dirhash 同源守卫。)
    """
    items = collect(root, subs, suffixes)
    if len(items) < minimum:
        raise GateUnrunnable(
            "%s: 冻结参照只有 %d 件(下限 %d)—— 参照物没了,这一轮判不了。root=%s"
            % (label, len(items), minimum, root)
        )
    # ⛔ S139:`require_fresh` 有 0 字节判据(`.part` 血训),而它的孪生**没有** ——
    #    而这一半管的正是「没人能重建」的那些参照(`gate0_vocoder.py:77-80` 亲口写着那 9 片
    #    gate0/gate1 里没有任何脚本能重建)。一份被截成 0 字节的冻结参照仍然满足件数下限,
    #    而 `expect_sha` 是 **opt-in** ⇒ 没传 sha 的调用点(今天就有一处,就是上面那一处)
    #    对「参照被掏空」结构上是瞎的。
    empty = [p for p, _m in items if os.path.getsize(p) == 0]
    if empty:
        raise GateUnrunnable(
            "%s: 冻结参照里有 %d 件是 0 字节 —— 它还在、还够数,但它已经不是参照了:%s"
            % (label, len(empty), [os.path.basename(p) for p in empty[:5]])
        )
    lo = _fmt(min(m for _p, m in items))
    hi = _fmt(max(m for _p, m in items))
    got = dirhash(root, subs, suffixes)
    if expect_sha is not None and got != expect_sha:
        raise GateUnrunnable(
            "%s: 冻结参照的内容**变了**。期望 %s,实测 %s。\n"
            "       ⇒ 它被重跑 / 从别的备份还原 / 指到了另一个目录。这一轮的对拍失去参照身份,\n"
            "         **不是**被测代码红了。要接受新参照必须显式改掉登记的 sha。"
            % (label, expect_sha[:16], got[:16])
        )
    _say("[FROZEN-REF] %s: %d 件,mtime %s ~ %s,sha %s —— %s"
         % (label, len(items), lo, hi, got[:16], why))
    if expect_sha is None:
        _say("             ⚠ 这一处没有登记 expect_sha ⇒ 它只判了「够数、非空」,"
             "**没判「还是同一份东西」**(换掉参照只会换掉上面那一行的日期)。")
    return items


def note_uncovered(label, why):
    """两侧都是冻结产物 ⇒ 这条判据本轮对今天的代码零覆盖。响亮记账。"""
    _say("[NO-COVERAGE] %s —— %s" % (label, why))
    _uncovered.append(label)


def require_min(label, n, minimum, detail=""):
    """防空集守卫。S68(f599e76)已判 [major],补丁形状见 gate0_sovits_v2_compare.py:72-75。"""
    if n < minimum:
        raise GateUnrunnable(
            "%s: 只有 %d 件(下限 %d)—— 空集/缩水会让后面每一条判据 PASS 而什么都没比。%s"
            % (label, n, minimum, detail)
        )


def finish(gate_name, failures, allow_uncovered=False):
    """统一收尾。⛔ 绿必须自陈它的覆盖面,不许把 ALL PASS 说成它并不具备的东西。"""
    _say()
    if failures:
        _say("%s: FAIL — %s" % (gate_name, ", ".join(failures)))
        _reset_uncovered()
        sys.exit(EXIT_RED)
    if _uncovered:
        listed = ", ".join(_uncovered)
        n = len(_uncovered)
        _reset_uncovered()
        if not allow_uncovered:
            _say("%s: 不可归因 — %d 条判据本轮零覆盖:%s" % (gate_name, n, listed))
            _say("       ⇒ 没有真红,但也不构成一次通过。要接受这些缺口必须显式 --allow-uncovered。")
            sys.exit(EXIT_UNRUNNABLE)
        _say("%s: PASS-WITH-GAPS(⛔ %d 条判据本轮零覆盖:%s)" % (gate_name, n, listed))
        sys.exit(EXIT_PASS)
    _say("%s: ALL PASS" % gate_name)
    sys.exit(EXIT_PASS)


def _reset_uncovered():
    """收尾之后清账。⛔ 不清的话,同一进程里连着做两次判定时第二次会继承第一次的缺口
    —— 而 `_selftest` 正是这样用它的(此前只有自检自己记得清)。"""
    del _uncovered[:]


def run(gate_name, main_fn):
    """把 main 包起来,让『闸自己跑不起来』与『被测的东西不对』**分开报**(S129 铁律)。"""
    try:
        main_fn()
    except SystemExit:
        raise
    except GateUnrunnable as e:
        _say()
        _say("=" * 72)
        _say("%s: 闸跑不起来 / 读数不可归因(exit %d)" % (gate_name, EXIT_UNRUNNABLE))
        _say("  %s" % e)
        _say("⛔ 这【不是】一次判定。别把它读成通过,也别读成被测代码红了。")
        _say("=" * 72)
        sys.exit(EXIT_UNRUNNABLE)
    except Exception:                      # noqa: BLE001
        import traceback
        _say()
        _say("=" * 72)
        _say("%s: 闸自己炸了(exit %d)—— 下面是转录,这不是一次判定" % (gate_name, EXIT_UNRUNNABLE))
        # ⛔ S139:`traceback.print_exc()` 默认写 **stderr**,而 `_ensure_utf8` 之外的敌对流
        #    在这里同样会炸 —— 而这是**兜底路径**,它炸掉就等于整台闸失声。走 _say。
        _say(traceback.format_exc().rstrip())
        _say("=" * 72)
        sys.exit(EXIT_UNRUNNABLE)


# --------------------------------------------------------------------------- 自检
def _selftest():
    """每种闸都要有基线自检(S128 血训:工装报的 RED 有三分之一是假的)。

    这里逐条**真的触发**每一条错误分支 —— S129 铁律的同族:一条从没被执行过的
    错误分支就是一条空判据。
    """
    import shutil
    import tempfile

    fails = []

    def expect_unrunnable(name, fn):
        try:
            fn()
        except GateUnrunnable:
            _say("  ok   %s -> GateUnrunnable" % name)
            return
        except Exception as e:                      # noqa: BLE001
            fails.append("%s 抛的是 %r,不是 GateUnrunnable" % (name, e))
            return
        fails.append("%s 应该抛 GateUnrunnable,却过了" % name)

    def expect_ok(name, fn):
        try:
            fn()
            _say("  ok   %s -> 正常通过" % name)
        except Exception as e:                      # noqa: BLE001
            fails.append("%s 不该抛,却抛了 %r" % (name, e))

    def expect_exit(name, code, fn):
        """⛔ S139:`finish()` 与 `run()` 的每一条出口此前**在自检里一行都没执行过** ——
        而它们正是这台闸对外的全部结论。一条从没被执行过的出口就是一条空判据(S129)。"""
        try:
            fn()
        except SystemExit as e:
            got = e.code if isinstance(e.code, int) else 0
            if got == code:
                _say("  ok   %s -> exit %d" % (name, got))
            else:
                fails.append("%s 应该 exit %d,实际 exit %r" % (name, code, got))
            return
        except Exception as e:                      # noqa: BLE001
            fails.append("%s 应该 exit %d,却抛了 %r" % (name, code, e))
            return
        fails.append("%s 应该 exit %d,却什么都没发生" % (name, code))

    tmp = tempfile.mkdtemp(prefix="gate0_guard_selftest_")
    try:
        d = os.path.join(tmp, "sub")
        os.makedirs(d)
        now = time.time()

        # 1) 没有 t0
        old = os.environ.pop(T0_ENV, None)
        expect_unrunnable("read_t0(缺 GATE0_T0)", lambda: read_t0("selftest"))
        os.environ[T0_ENV] = "not-a-number"
        expect_unrunnable("read_t0(不是数字)", lambda: read_t0("selftest"))
        # ⛔ S139:`float()` 过得去的四种退化值 —— 其中 **0 是唯一会产出一行合格的绿**的那个
        for bad in ("0", "-1", "nan", "inf"):
            os.environ[T0_ENV] = bad
            expect_unrunnable("read_t0(退化值 %s)" % bad, lambda: read_t0("selftest"))
        os.environ[T0_ENV] = "%.3f" % now
        expect_ok("read_t0(正常)", lambda: read_t0("selftest"))

        # 2) 空目录 —— 这是本模块存在的理由:空集必须是 UNRUNNABLE 而不是 PASS
        expect_unrunnable("require_fresh(空目录)",
                          lambda: require_fresh("st/空", tmp, ["sub"], now, minimum=1))
        expect_unrunnable("declare_frozen(空目录)",
                          lambda: declare_frozen("st/空", tmp, ["sub"], minimum=1, why="自检"))
        expect_unrunnable("require_min(0 < 1)", lambda: require_min("st/min", 0, 1))

        # 3) 陈货 —— 造一个 mtime 在 t0 之前的文件
        stale_p = os.path.join(d, "stale.npy")
        with open(stale_p, "wb") as f:
            f.write(b"x")
        os.utime(stale_p, (now - 86400, now - 86400))
        expect_unrunnable("require_fresh(陈货)",
                          lambda: require_fresh("st/陈", tmp, ["sub"], now, minimum=1))
        # 冻结参照对同一批陈货是【正常】的 —— 两者必须给出不同结论,否则这个区分是空的
        expect_ok("declare_frozen(同一批陈货)",
                  lambda: declare_frozen("st/冻", tmp, ["sub"], minimum=1, why="自检"))

        # 4) 新货
        fresh_p = os.path.join(d, "fresh.npy")
        with open(fresh_p, "wb") as f:
            f.write(b"y")
        os.utime(fresh_p, (now + 5, now + 5))
        os.remove(stale_p)
        expect_ok("require_fresh(新货)",
                  lambda: require_fresh("st/新", tmp, ["sub"], now, minimum=1))
        # 样本不足要与陈货分开报
        expect_unrunnable("require_fresh(新货但不够数)",
                          lambda: require_fresh("st/少", tmp, ["sub"], now, minimum=2))

        # 5) 后缀过滤真的在过滤(否则 minimum 判据会被无关文件顶满)
        with open(os.path.join(d, "note.txt"), "w") as f:
            f.write("z")
        expect_unrunnable(
            "require_fresh(后缀过滤后不够数)",
            lambda: require_fresh("st/滤", tmp, ["sub"], now, minimum=2, suffixes=[".npy"]))

        # 6) M12:合计够但**整整一类**消失 —— 必须被逐类下限抓住
        for i in range(3):
            for ext in (".a.npy", ".b.npy"):
                q = os.path.join(d, "x%d%s" % (i, ext))
                with open(q, "wb") as f:
                    f.write(b"z")
                os.utime(q, (now + 5, now + 5))
        expect_ok("require_fresh(逐类都在)",
                  lambda: require_fresh("st/类", tmp, ["sub"], now, minimum=6,
                                        suffixes=[".npy"], classes=[".a.npy", ".b.npy"],
                                        per_class=3))
        for i in range(3):                       # 把 .b 整类删掉,再补 3 个 .a 顶上合计
            os.remove(os.path.join(d, "x%d.b.npy" % i))
            q = os.path.join(d, "y%d.a.npy" % i)
            with open(q, "wb") as f:
                f.write(b"z")
            os.utime(q, (now + 5, now + 5))
        expect_unrunnable(
            "require_fresh(合计够但少了一整类)",
            lambda: require_fresh("st/类缺", tmp, ["sub"], now, minimum=6,
                                  suffixes=[".npy"], classes=[".a.npy", ".b.npy"],
                                  per_class=3))

        # 7) 0 字节的新鲜文件不是产物(.part 血训同族)
        z = os.path.join(d, "zero.a.npy")
        open(z, "wb").close()
        os.utime(z, (now + 5, now + 5))
        expect_unrunnable("require_fresh(0 字节)",
                          lambda: require_fresh("st/零", tmp, ["sub"], now, minimum=1,
                                                suffixes=[".npy"]))
        os.remove(z)

        # 8) M11:冻结参照被换掉必须被抓住,而不是只换一行日期
        good = dirhash(tmp, ["sub"], [".npy"])
        expect_ok("declare_frozen(sha 对得上)",
                  lambda: declare_frozen("st/冻sha", tmp, ["sub"], 1, "自检",
                                         suffixes=[".npy"], expect_sha=good))
        swap = os.path.join(d, "y0.a.npy")
        with open(swap, "wb") as f:
            f.write(b"CHANGED")
        expect_unrunnable(
            "declare_frozen(参照内容被换掉)",
            lambda: declare_frozen("st/冻换", tmp, ["sub"], 1, "自检",
                                   suffixes=[".npy"], expect_sha=good))

        # 8b) ⛔ S139:冻结参照被**掏空**(0 字节)—— 它还在、还够数,expect_sha 又是 opt-in
        zf = os.path.join(d, "hollow.a.npy")
        open(zf, "wb").close()
        expect_unrunnable(
            "declare_frozen(参照被掏空成 0 字节)",
            lambda: declare_frozen("st/冻空", tmp, ["sub"], 1, "自检", suffixes=[".npy"]))
        os.remove(zf)

        if old is not None:
            os.environ[T0_ENV] = old
        else:
            os.environ.pop(T0_ENV, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 9) note_uncovered 记账
    _reset_uncovered()
    note_uncovered("st/未覆盖", "自检")
    if len(_uncovered) != 1:
        fails.append("note_uncovered 没记账")
    _reset_uncovered()

    # 10) ⛔⛔ S139:`finish()` 的三条出口 + `run()` 的两条兜底 —— 此前**一条都没被执行过**,
    #     而这五条就是这台闸对外的全部结论。逐条真触发。
    expect_exit("finish(ALL PASS)", EXIT_PASS, lambda: finish("st", []))
    expect_exit("finish(真红)", EXIT_RED, lambda: finish("st", ["某条判据"]))

    def _gaps(allow):
        _reset_uncovered()
        note_uncovered("st/缺口", "自检")
        finish("st", [], allow_uncovered=allow)

    expect_exit("finish(有缺口,未放行)", EXIT_UNRUNNABLE, lambda: _gaps(False))
    expect_exit("finish(有缺口,--allow-uncovered)", EXIT_PASS, lambda: _gaps(True))
    # ⛔ 上一条走的是 EXIT_PASS ⇒ 如果 finish 不清账,下一条会继承那个缺口而静默变成 3
    expect_exit("finish(上一条之后账已清)", EXIT_PASS, lambda: finish("st", []))

    def _boom():
        raise GateUnrunnable("自检:产物不在")

    def _crash():
        raise RuntimeError("自检:闸自己炸了")

    expect_exit("run(GateUnrunnable)", EXIT_UNRUNNABLE, lambda: run("st", _boom))
    expect_exit("run(任意异常)", EXIT_UNRUNNABLE, lambda: run("st", _crash))
    expect_exit("run(main 自己 sys.exit 透传)", EXIT_RED,
                lambda: run("st", lambda: sys.exit(EXIT_RED)))
    _reset_uncovered()

    # 11) ⛔⛔⛔ S139 的阴性对照 —— **本模块存在的理由级别的一条**。
    #     这台闸的每一条失败出口都是中文,而本机 locale 是 cp932。
    #     「它安静了」和「它坏掉了」长得一模一样(S138 血训)⇒ 必须证明它在**敌对流**上
    #     仍然说得出话,而且**退出码不变**。
    #     ⚠ 两层都要:① 进程内把 sys.stdout 换成一个真的 cp932 流(_ensure_utf8 够不着的场景)
    #                ② 真开一个子进程、真用 cp932 locale(README 教的手敲 + `>` 那条路)
    for enc in ("cp932", "ascii", "latin-1"):
        raw = io.BytesIO()
        hostile = io.TextIOWrapper(raw, encoding=enc, errors="strict", newline="")
        saved = sys.stdout
        sys.stdout = hostile
        try:
            code = None
            try:
                run("st", _boom)
            except SystemExit as e:
                code = e.code
            hostile.flush()
        except Exception as e:                      # noqa: BLE001
            sys.stdout = saved
            fails.append("敌对流(%s):说话这件事本身把闸弄炸了 %r" % (enc, e))
            continue
        finally:
            sys.stdout = saved
        got = raw.getvalue()
        if code != EXIT_UNRUNNABLE:
            fails.append("敌对流(%s):exit 应为 %d,实际 %r —— 归因被编码问题改写了"
                         % (enc, EXIT_UNRUNNABLE, code))
        elif not got.strip():
            fails.append("敌对流(%s):退出码对了,但**一个字都没说出来**" % enc)
        else:
            _say("  ok   敌对流(%-8s) -> exit 3 且 %d 字节说得出来" % (enc, len(got)))

    # 11b) 子进程 × 真 cp932 —— 这一条对的是「照 README 手敲 + 重定向」那条真实路径
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    for case, want in (("unrunnable", EXIT_UNRUNNABLE), ("gaps", EXIT_UNRUNNABLE),
                       ("red", EXIT_RED), ("pass", EXIT_PASS),
                       ("caller_print", EXIT_PASS)):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-u", os.path.abspath(__file__),
             "--emit-case", case],
            capture_output=True, env=child_env, timeout=120)
        body = (proc.stdout + proc.stderr)
        if proc.returncode != want:
            fails.append("子进程 cp932 / %s:exit 应为 %d,实际 %d\n        转录尾:%s"
                         % (case, want, proc.returncode,
                            body.decode("utf-8", "replace")[-300:]))
        elif not body.strip():
            fails.append("子进程 cp932 / %s:退出码对了,但一个字都没说出来" % case)
        else:
            _say("  ok   子进程 cp932 / %-11s -> exit %d 且说得出来" % (case, proc.returncode))

    _say()
    if fails:
        for f in fails:
            _say("  FAIL %s" % f)
        _say("gate0_guard 自检: FAILED(%d)" % len(fails))
        return EXIT_SELFTEST
    _say("gate0_guard 自检: ALL OK")
    return EXIT_PASS


def _emit_case(case):
    """自检 11b 用的靶子:在**子进程**里真的走一次某条出口。

    ⛔ 它必须是一个真进程 —— 「进程内 catch SystemExit」证明不了退出码真的传出去了,
       而 S139 买回来的那条缺陷恰恰发生在**解释器把缓冲区 flush 出去的那一刻**。
    """
    if case == "unrunnable":
        def m():
            raise GateUnrunnable("子进程自检:产物不在,这一轮的读数不可归因")
        run("emit", m)
    elif case == "gaps":
        note_uncovered("子进程自检/缺口", "两侧都是冻结产物")
        finish("emit", [])
    elif case == "red":
        finish("emit", ["子进程自检:某条判据红了"])
    elif case == "pass":
        finish("emit", [])
    elif case == "caller_print":
        # ⛔ S139 变异 M3 暴露的缺口:`_say` 的退路只护得住**本模块自己**的输出,
        #    而 import 这台闸的八个 gate0 脚本**自己**打的是裸 `print` 的中文
        #    (`gate0_diff_compare.py` 是其中唯一没有自己 reconfigure 的那个)。
        #    ⇒ `_ensure_utf8()` 那一半的判据就是这一条:import 之后,调用方的裸 print 必须活。
        print("调用方的裸 print:两侧目录都空 ⇒ 读数不可归因")
        sys.exit(EXIT_PASS)
    else:
        _say("未知 case:%r" % case)
        sys.exit(EXIT_SELFTEST)


if __name__ == "__main__":
    if "--emit-case" in sys.argv:
        _emit_case(sys.argv[sys.argv.index("--emit-case") + 1])
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    _say(__doc__)
    sys.exit(EXIT_PASS)
