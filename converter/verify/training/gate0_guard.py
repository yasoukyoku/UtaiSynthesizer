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

自检:`python gate0_guard.py --selftest`
"""
import os
import sys
import time

EXIT_PASS = 0
EXIT_RED = 1
EXIT_UNRUNNABLE = 3
EXIT_SELFTEST = 4

T0_ENV = "GATE0_T0"

_uncovered = []


class GateUnrunnable(RuntimeError):
    """这一轮的读数不可归因。⛔ 绝不许被读成『通过』。"""


def _fmt(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def read_t0(gate_name):
    """本轮起始时刻(epoch 秒),由跑器通过环境变量 GATE0_T0 传进来。

    ⛔ 没有它就没有新鲜度判据 ⇒ 分不出产物是今天算的还是七月的
       ⇒ 这一轮不构成一次判定。响亮地判 UNRUNNABLE,不给默认值。
    """
    raw = os.environ.get(T0_ENV)
    if not raw:
        raise GateUnrunnable(
            "没有 %s ⇒ 无法判断读到的产物是不是本轮算的 ⇒ 这一轮的读数不可归因。\n"
            "       用 run_gate0_chain.py 跑,或手动 set %s=<epoch 秒>。\n"
            "       (%s)" % (T0_ENV, T0_ENV, gate_name)
        )
    try:
        return float(raw)
    except ValueError:
        raise GateUnrunnable("%s=%r 不是一个 epoch 秒" % (T0_ENV, raw))


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


def require_fresh(label, root, subs, t0, minimum, suffixes=None):
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
    print("[FRESH] %s: %d 件,全部 mtime >= t0=%s(实测 %s ~ %s)"
          % (label, len(items), _fmt(t0), _fmt(oldest), _fmt(newest)))
    return items


def declare_frozen(label, root, subs, minimum, why, suffixes=None):
    """冻结参照:**故意**不重算的一侧。必须存在且够数,并把日期打进转录自证。

    这样做的理由要写进 why —— 转录里看得见「这一侧是冻结的、为什么冻结、它有多旧」,
    下一个人才不会把「参照没变」误读成「本轮验过了」。
    """
    items = collect(root, subs, suffixes)
    if len(items) < minimum:
        raise GateUnrunnable(
            "%s: 冻结参照只有 %d 件(下限 %d)—— 参照物没了,这一轮判不了。root=%s"
            % (label, len(items), minimum, root)
        )
    lo = _fmt(min(m for _p, m in items))
    hi = _fmt(max(m for _p, m in items))
    print("[FROZEN-REF] %s: %d 件,mtime %s ~ %s —— %s" % (label, len(items), lo, hi, why))
    return items


def note_uncovered(label, why):
    """两侧都是冻结产物 ⇒ 这条判据本轮对今天的代码零覆盖。响亮记账。"""
    print("[NO-COVERAGE] %s —— %s" % (label, why))
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
    print()
    if failures:
        print("%s: FAIL — %s" % (gate_name, ", ".join(failures)))
        sys.exit(EXIT_RED)
    if _uncovered:
        listed = ", ".join(_uncovered)
        if not allow_uncovered:
            print("%s: 不可归因 — %d 条判据本轮零覆盖:%s" % (gate_name, len(_uncovered), listed))
            print("       ⇒ 没有真红,但也不构成一次通过。要接受这些缺口必须显式 --allow-uncovered。")
            sys.exit(EXIT_UNRUNNABLE)
        print("%s: PASS-WITH-GAPS(⛔ %d 条判据本轮零覆盖:%s)" % (gate_name, len(_uncovered), listed))
        sys.exit(EXIT_PASS)
    print("%s: ALL PASS" % gate_name)
    sys.exit(EXIT_PASS)


def run(gate_name, main_fn):
    """把 main 包起来,让『闸自己跑不起来』与『被测的东西不对』**分开报**(S129 铁律)。"""
    try:
        main_fn()
    except SystemExit:
        raise
    except GateUnrunnable as e:
        print()
        print("=" * 72)
        print("%s: 闸跑不起来 / 读数不可归因(exit %d)" % (gate_name, EXIT_UNRUNNABLE))
        print("  %s" % e)
        print("⛔ 这【不是】一次判定。别把它读成通过,也别读成被测代码红了。")
        print("=" * 72)
        sys.exit(EXIT_UNRUNNABLE)
    except Exception:                      # noqa: BLE001
        import traceback
        print()
        print("=" * 72)
        print("%s: 闸自己炸了(exit %d)—— 下面是转录,这不是一次判定" % (gate_name, EXIT_UNRUNNABLE))
        traceback.print_exc()
        print("=" * 72)
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
            print("  ok   %s -> GateUnrunnable" % name)
            return
        except Exception as e:                      # noqa: BLE001
            fails.append("%s 抛的是 %r,不是 GateUnrunnable" % (name, e))
            return
        fails.append("%s 应该抛 GateUnrunnable,却过了" % name)

    def expect_ok(name, fn):
        try:
            fn()
            print("  ok   %s -> 正常通过" % name)
        except Exception as e:                      # noqa: BLE001
            fails.append("%s 不该抛,却抛了 %r" % (name, e))

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

        if old is not None:
            os.environ[T0_ENV] = old
        else:
            os.environ.pop(T0_ENV, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 6) note_uncovered / finish 的分支
    del _uncovered[:]
    note_uncovered("st/未覆盖", "自检")
    if len(_uncovered) != 1:
        fails.append("note_uncovered 没记账")
    del _uncovered[:]

    print()
    if fails:
        for f in fails:
            print("  FAIL %s" % f)
        print("gate0_guard 自检: FAILED(%d)" % len(fails))
        return EXIT_SELFTEST
    print("gate0_guard 自检: ALL OK")
    return EXIT_PASS


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
    sys.exit(EXIT_PASS)
