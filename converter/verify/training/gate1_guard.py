# -*- coding: utf-8 -*-
"""gate1 判据护栏 —— 让 gate1 那一层的【绿有意义、红能归因】。

S139(§F7 笔 E)新建。它是 `gate0_guard` 的**同层兄弟**:那台机器 S135 已经为 gate0 造好,
而 **gate1 这一层四个月里一条都没拿到** —— `gate0_guard.py:24-27` 的头注逐字写着这条罪状
(「跑不起来与被测的东西不对共用同一个非零退出码」),而它只在 gate0 修了。
八个 `gate0_*` 脚本全部 import `gate0_guard`,五个 `gate1_*_compare.py` **一个都没有**。

⛔ 为什么另起一个模块而不是直接用 `gate0_guard`:
   ① t0 的环境变量必须**分层**(见 `T0_ENV` 那一段);
   ② gate1 的产物形状是 **TB event 文件 + 协议 JSONL**,不是 gate0 那种目录里一堆 .npy
      ⇒ 「够数 / 新鲜 / 同源」这三问在这一层有不同的落点;
   ③ 这一层有三条 gate0 没有的事故形状(多 events 文件、步集不等、None 分量被筛掉)。
   ⛔ 但**共用的部分一律转调 `gate0_guard`,不许抄第二份**(_say / GateUnrunnable /
      require_fresh / declare_frozen / note_uncovered / finish / run 全部转调)。

────────────────────────────────────────────────────────────────────────────
S139 实测、这个模块要挡住的六种事故(全部在真代码上复现过)
────────────────────────────────────────────────────────────────────────────
⑴ ⛔⛔⛔ **零新鲜度**。把一份 **2026-07-07** 的 `gate1_ours_steps.jsonl` 喂给八月的参照侧,
   `gate1_compare.py` 打出 `GATE1: ALL PASS (30 steps compared)` —— 与 S134 那次的转录
   **逐字符相同**。gate1 的全部价值是「我方侧**今天**跑出来的轨迹与原版一致」,而五条
   compare 在结构上**分不出「跑了」和「根本没跑、读的是一个月前的文件」**。
   ⚠ 更阴的一层:正因为移植是逐位确定的,陈货与新货的数字**完全一样** ——
   同一件事既是「移植正确」的证明,也是「闸看不见自己有没有跑」的证明。
⑵ ⛔⛔ **归因**。五条 compare 在缺产物时全部走**未捕获异常**(`EventAccumulator.Scalars`
   对不存在的 tag 抛 KeyError;`open(jsonl)` 抛 FileNotFoundError),而 Python 未捕获异常的
   退出码**正好也是 1** = 真红的码。⇒ S129 铁律第一条在这一层结构上成立。
⑶ ⛔⛔ **空集/半集假 PASS**。`gate1_vocoder_compare` 两侧 logdir 存在但空 ⇒
   `PASS tag sets identical (0 tags)` → `=== gate1_vocoder: PASS ===` 退 0
   (⚠ 而且**不需要空**:两侧各写满 15 点、数值差 1000 倍,只要 tag 前缀不叫
   `training/`/`validation/` 就同样退 0 —— 而两侧吃同一份 yaml,改名天生对称);
   `gate1_diff_compare` 在「train/loss 两侧都空、validation 三点齐全」时打 `PASS` 退 0。
⑷ ⛔⛔ **地板量的不是这条链的真值**。rvc/sovits/sovits_v2 的下限都是与真值无关的**常数 10**,
   而真值是 **30/16/14** ⇒ 我方只跑出三分之一,照打 `ALL PASS (10 steps compared)` 退 0。
   ⇒ 判据必须是**步集相等 + 等于登记的真值**,不是「交集 ≥ 某个数」。
   ⚠ 而且「我方没跑完」是**闸没跑成**,不是被测的东西不对 ⇒ 退 3 不是退 1。
⑸ ⛔⛔ **None 分量被筛掉**。`gate1_sovits_v2_compare.py:50` 用
   `losses.get("g_total") is not None` 筛步 ⇒ **发散(非有限 ⇒ protocol 写 None)的那几步
   整个不进 ours**,`aligned: 12` 打 [PASS] —— 而 **NaN 正是 gate1 存在的理由**。
   ⇒ 筛步一律用「键在不在」,值为 None **立刻红并点名 step/分量**。
⑹ ⛔ **多个 events 文件**。十个 gate1 跑器里只有一个自清输出目录,而唯一清 TB 目录的是
   被硬禁的 `*_prepare.py` ⇒ 在禁令之下,「再跑一次原版侧」的唯一路径就是往同一个 logdir
   **再加一个 events 文件**,而 `EventAccumulator` 按文件名序**拼接、不去重**,
   compare 的 `{e.step: e.value}` 让**后写的赢** —— 打印的步数一点异常都没有。
   实测:陈货在前/新货在后 ⇒ 全绿;新货在前/陈货在后 ⇒ 全红。**同一份数据,两种结论。**

⛔ 边界(写清楚,别把这台机器当成它不是的东西):
   * 它**不**回答数值对不对 —— 那是各条 compare 自己的阈值。
   * 它**不**证明参照侧的上游代码没被改过(那是另一条线,见 `_ref_identity`)。
   * `--skip-orig` 那一跑里,参照侧是**冻结**的 ⇒ 走 `declare_frozen` + `note_uncovered`,
     结论是 PASS-WITH-GAPS 或 exit 3,**不许是干净的绿**。

自检:`python gate1_guard.py --selftest`
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate0_guard as G                                        # noqa: E402

# —— 共用的一律转调,不抄第二份
GateUnrunnable = G.GateUnrunnable
EXIT_PASS, EXIT_RED = G.EXIT_PASS, G.EXIT_RED
EXIT_UNRUNNABLE, EXIT_SELFTEST = G.EXIT_UNRUNNABLE, G.EXIT_SELFTEST
_say = G._say
declare_frozen = G.declare_frozen
note_uncovered = G.note_uncovered
finish = G.finish
run = G.run
dirhash = G.dirhash

# ⛔ **另起 `GATE1_T0`,不复用 `GATE0_T0`**:同一个 shell 里先跑 gate0 会话(它把 t0 落盘
#    并 export)再跑 gate1,复用同一个变量名会让 gate1 读到**几小时前**的 t0
#    ⇒ 这期间产生的任何陈货都被判 FRESH。而这种失效**静默、且只会让判据变宽**。
T0_ENV = "GATE1_T0"
# 跑器把「这一跑故意跳过了哪几段」透传进来 ⇒ 被跳过那一侧的产物**本来就不是本轮的**,
# 它该走 declare_frozen + note_uncovered,而不是被判成一条陈货红。
SKIPPED_ENV = "GATE1_SKIPPED"


def read_t0(gate_name):
    return G.read_t0(gate_name, env=T0_ENV)


def skipped_stages():
    raw = os.environ.get(SKIPPED_ENV, "")
    return [s for s in (x.strip() for x in raw.split(",")) if s]


# ────────────────────────────────────────────────────────────────── 登记的真值
#
# ⛔ 地板必须是**这条链的真值**,不是一个与真值无关的常数(见头注 ⑷)。
#    每一条都写出它是怎么来的 —— 「README 上写着」不算依据(S134:README 的数字不算依据,
#    照旧文字核点数会把对的判成错的)。
#    ⚠ 改夹具就必须改这里,而且要在 commit 里说为什么 —— 这正是它存在的意义。
EXPECT = {
    # gate1_run_ours.py:27 total_epoch=2 × loader_len=15(resume_state.json 实测)= 30
    # `clamped` = **今天这份夹具上已知、已量、被接受的致盲面**(见 note_clamped 的头注):
    #   原版侧 kl 在 step 0 与 2 顶满 9.0 ⇒ 那两步在数值上不可证伪;mel 从没到过 75。
    # ⛔ 它是**登记值**不是容忍值:实测数与它不等 ⇒ 判 UNRUNNABLE(致盲面变了 = 新消息),
    #   相等 ⇒ 每一跑打一行 [NOTE] 把它说出来。这条纪律与 declare_frozen 的 expect_sha 同形。
    "rvc": dict(steps=30, components=5, clamped={"loss/g/kl": 2, "loss/g/mel": 0}),
    # gate1_sovits_run_ours.py 的 total_epoch × loader_len;S134/S37-40 实测 16
    "sovits": dict(steps=16, components=6),
    # 同上;S134 与 S68批4 两次实测 14
    "sovits_v2": dict(steps=14, components=9),
    # gate1_diff_prepare.py:7-11 的 ceil(31/4)*3 = 8*3 = 24;validation 边界 8/16/24
    "diff": dict(steps=24, val_boundaries=3),
    # vocoder/pipeline.py:577 max_updates = 2*total_real,驱动 total_steps=15
    # ⇒ training 每分量 15 点、validation 每分量 4 点 @global [0,10,20,30];共 11 个 tag
    "vocoder": dict(train_points=15, val_points=4, tags=11),
}


# ────────────────────────────────────────────────────────────────── TB 侧
def _events_files(logdir):
    if not os.path.isdir(logdir):
        return None
    return sorted(f for f in os.listdir(logdir) if "tfevents" in f)


def tb_scalars(label, logdir, tags, t0, frozen_why=None):
    """读一侧的 TB 标量,并在读之前把三件事判掉。

    ⛔ 顺序是承重的:目录 → **events 文件恰好一个** → 新鲜度/冻结 → tag 齐全 → 取值。
       任何一步不成立都是 `GateUnrunnable`(exit 3),**不是**判负。
    """
    files = _events_files(logdir)
    if files is None:
        raise GateUnrunnable("%s: logdir 不在:%s\n"
                             "       ⇒ 参照物/产物缺席不是「被测的东西不对」。" % (label, logdir))
    if not files:
        raise GateUnrunnable(
            "%s: logdir 在,但里面**一个 events 文件都没有**:%s\n"
            "       ⛔ 这正是 S135 在 gate0 钉过的那条:删掉目录 = 正确地红,"
            "清空目录 = 假 PASS,两种清法后果相反。" % (label, logdir))
    if len(files) > 1:
        raise GateUnrunnable(
            "%s: logdir 里有 **%d 个** events 文件,应当只有 1 个:%s\n"
            "       ⛔ EventAccumulator 按文件名序【拼接、不去重】,而 compare 用 {step: value}\n"
            "         ⇒ **后写的赢**,而打印出来的步数一点异常都没有(实测:陈货在前全绿,\n"
            "           新货在前全红 —— 同一份数据两种结论)。\n"
            "       ⇒ 跑原版侧之前要先删掉旧的 events(⛔ 别用跑 prepare 来解决,它会 rmtree 掉\n"
            "         不可再生的历史证据)。文件:%s"
            % (label, len(files), logdir, files))

    # ⛔ 过滤器用**那一个文件的确切名字**,不是 `suffixes=["tfevents"]`:
    #    TB 的文件名是 `events.out.tfevents.<ts>.<host>.<pid>.<n>` —— **`tfevents` 在中间**,
    #    而 `gate0_guard.collect` 的 suffixes 是 `endswith` ⇒ 那样写永远匹配 0 件,
    #    于是新鲜度判据**每次都以「只有 0 件」的理由**抛 GateUnrunnable。
    #    ⚠ S139 我自己就这么写过一版,而 `gate1_guard --selftest` 的「陈货」那一条**因此而绿**
    #      —— 红对了,但它在回答另一个问题(S134 §5.1 的形状:「红了、组也对、断言也对,
    #      但它在回答另一个问题」)。是 `gate1_negctl` 抓住的,不是我看出来的。
    only = files[0]
    if frozen_why:
        declare_frozen(label, logdir, [""], minimum=1, why=frozen_why, suffixes=[only])
    else:
        G.require_fresh(label, logdir, [""], t0, minimum=1, suffixes=[only])

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # ⛔ size_guidance 必须钉死:默认值(scalars 上限 10000)会在放大 gate 时
    #    **两侧各自随机抽样**丢点,而 tag 集合判据对这种丢失是瞎的。
    #    五条里 `gate1_vocoder_compare.py:41` 是唯一没钉的那个。
    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()
    have = set(acc.Tags()["scalars"])
    want = set(tags)
    missing = sorted(want - have)
    if missing:
        raise GateUnrunnable(
            "%s: 少了 %d 个标量 tag:%s\n       在场的是:%s\n"
            "       ⇒ 这一侧没写出该写的东西(配置改了 / 跑到一半死了),**不是**数值不一致。"
            % (label, len(missing), missing, sorted(have)))
    return {t: {e.step: e.value for e in acc.Scalars(t)} for t in tags}


def tb_all_scalars(label, logdir, t0, prefixes, frozen_why=None):
    """声码器那条用:tag 名不预先知道,按前缀取。⛔ 但**前缀本身要当判据**(见头注 ⑶)。"""
    scal = tb_scalars(label, logdir, [], t0, frozen_why=frozen_why)  # 先过前四道
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()
    out = {t: [(e.step, e.value) for e in acc.Scalars(t)]
           for t in acc.Tags()["scalars"] if t.startswith(tuple(prefixes))}
    del scal
    return out


# ────────────────────────────────────────────────────────────────── JSONL 侧
def jsonl_steps(label, path, need_key, t0):
    """读我方侧的协议 JSONL 步流。

    ⛔ 两条与原来不同的地方,各自对着一条实测事故:
      ① **新鲜度**:这个文件是 gate1 唯一的我方侧读数载体,而它可以是一个月前的
         (跑器在 ours 段失败时**故意不落位**,于是上一轮的内容原地保留)。
      ② **筛步一律用「键在不在」,不是「值非 None」** —— 后者会让发散的那几步
         整个从判据里消失,而那正是这台闸要抓的东西。
    """
    import json
    if not os.path.isfile(path):
        raise GateUnrunnable("%s: 我方侧步流不在:%s\n"
                             "       ⇒ 我方侧没跑起来 ≠ 我们的数学不对。" % (label, path))
    G.require_fresh(label, os.path.dirname(path), [""], t0, minimum=1,
                    suffixes=[os.path.basename(path)])
    steps, nones = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "step":
                continue
            losses = obj.get("losses") or {}
            if need_key not in losses:
                continue
            if losses.get(need_key) is None:
                nones.append(obj.get("step"))
            steps[obj["step"]] = losses
    return steps, nones


def require_no_none(label, steps, keys):
    """任何被比较的分量出现 None ⇒ **立刻红并点名**。

    ⛔ `protocol.py` 的 `_clean` 把非有限值(nan/inf)写成 None ⇒ None 就是**发散**。
       原来三条链的做法各不相同,而最坏的那条(sovits_v2)把它整步筛掉、打 [PASS]。
    """
    bad = []
    for s in sorted(steps):
        for k in keys:
            if k in steps[s] and steps[s][k] is None:
                bad.append((s, k))
    if bad:
        head = ", ".join("step %s / %s" % (s, k) for s, k in bad[:8])
        return ["%s: %d 个分量是 None(= 非有限值,protocol 的 _clean 写成 None)⇒ **发散**:%s"
                % (label, len(bad), head)]
    return []


# ────────────────────────────────────────────────────────────────── 覆盖判据
def require_exact_steps(label, chain, got_steps, expect_n, other=None, other_label=""):
    """⛔ 步集**相等且等于登记的真值** —— 不是「交集 ≥ 10」。

    三种今天都会静默通过的事故,这一条一次挡住:
      ① 我方跑到一半死了(Ctrl-C / OOM / 早退)⇒ 交集缩到 12,照打 ALL PASS;
      ② 参照侧跑到一半死了 ⇒ 同上,而上游「正常完训」是 os._exit(2333333),
         rc 也说不出话;
      ③ 两侧都少,但少的地方一样 ⇒ 交集判据完全看不见。
    ⚠ 「跑到一半」是**闸没跑成**,不是被测的东西不对 ⇒ GateUnrunnable(exit 3)。
    """
    got = set(got_steps)
    if len(got) != expect_n:
        raise GateUnrunnable(
            "%s: %d 步,登记的真值是 %d 步(链 %s)。\n"
            "       ⇒ 有一侧没跑完 / 配置变了。这不是「数值不一致」,是这一轮**没有构成一次对拍**。\n"
            "       ⚠ 如果是**夹具本身**变了,去改 gate1_guard.EXPECT 并在 commit 里说明为什么 ——\n"
            "         那个数就是为这一刻存在的。实际步号:%s"
            % (label, len(got), expect_n, chain, sorted(got)[:40]))
    if other is not None:
        o = set(other)
        if got != o:
            only_a, only_b = sorted(got - o)[:10], sorted(o - got)[:10]
            raise GateUnrunnable(
                "%s 与 %s 的**步集不同**(各 %d/%d 步):只在前者 %s;只在后者 %s\n"
                "       ⇒ 两侧对齐不上,交集再大也不构成对拍。"
                % (label, other_label, len(got), len(o), only_a, only_b))
    _say("[COVERAGE] %s: %d 步,与登记的真值(%d)相等%s"
         % (label, len(got), expect_n,
            ",且与 %s 步集逐个相同" % other_label if other is not None else ""))
    return sorted(got)


def check_clamped(label, clamped, total, expect_n):
    """夹取记账的**登记式**版本 —— 见 EXPECT["rvc"]["clamped"] 的注释。

    ⛔ 为什么不能一律 `note_uncovered`(⇒ exit 3):这份夹具上 kl 恒有 2/30 步顶满,
       那样的话 rvc 这条链**每一跑都要 --allow-uncovered**,而一个每次都要加的开关
       三天之内就会变成肌肉记忆,等于没有。
    ⇒ 正确形状是**登记 + 对拍**:数对得上 ⇒ 每跑打一行 `[NOTE]` 把致盲面说出来;
       数变了 ⇒ 那是新消息,判 UNRUNNABLE 并要求人来更新登记值。
    """
    n = len(clamped)
    if n != expect_n:
        raise GateUnrunnable(
            "%s: 被夹取致盲的步数是 %d,而登记值是 %d(总 %d 步)。\n"
            "       ⇒ 致盲面变了 —— 要么夹具变了,要么被测代码在这一路的量级变了。\n"
            "         这不是「数值不一致」,是**这一路能证伪多少**这件事本身变了。\n"
            "       ⇒ 核实之后去改 gate1_guard.EXPECT[...]['clamped'],并在 commit 里说明。\n"
            "         实际被夹的步号:%s" % (label, n, expect_n, total, sorted(clamped)[:10]))
    if n:
        _say("[NOTE] %s: %d/%d 步被夹取致盲(登记值 %d)—— 原版侧在这些 step 已顶到上限 ⇒ "
             "判据退化成 min(ours,上限) vs 上限,我方值只要 ≥ 上限就恒等 ⇒ "
             "**这几步在数值上不可证伪**。步号:%s" % (label, n, total, expect_n, sorted(clamped)))
    if n >= total:
        raise GateUnrunnable(
            "%s: %d/%d 步**全部**被夹取致盲 ⇒ 这一路本轮零覆盖,不构成一次判定"
            % (label, n, total))


def note_clamped(label, clamped, total, limit_frac=0.5):
    """夹取记账。⛔ RVC 那条对我方值施加与上游相同的 clamp(mel>75 / kl>9)——
    凡是**原版侧已顶到上限**的 step,判据退化成 `min(ours, 9.0) vs 9.0`
    ⇒ 只要我方值 ≥9,相对差**恒等于 0**,无论真值是 9.001 还是 1e9(实测:改成 1e9 仍 ALL PASS)。
    而今天盘上的真夹具里 **kl 有 2/30 步顶满,含 step 0** —— 而 step 0 恰恰是
    初始化/底模装载/第一次前向这类结构性移植错误表现得最赤裸的一步。
    ⛔ 不许「去掉 clamp」:两个神谕(TB 与 stdout)存的都是**夹过的值**,去掉会变成一条必红的假判据。
    ⇒ 唯一诚实的做法是**记账**:说出这一路实际比过几步。
    """
    if not clamped:
        return
    note_uncovered("%s 的 %d/%d 步被夹取致盲" % (label, len(clamped), total),
                   "原版侧在这些 step 已顶到上限 ⇒ 判据退化成 min(ours,上限) vs 上限,"
                   "我方值只要 ≥ 上限就恒等 ⇒ 这几步**在数值上不可证伪**。步号:%s"
                   % (sorted(clamped)[:10],))
    if len(clamped) > total * limit_frac:
        raise GateUnrunnable(
            "%s: %d/%d 步被夹取致盲,超过 %.0f%% ⇒ 这一路已经不构成一次判定"
            % (label, len(clamped), total, limit_frac * 100))


def header(gate_name, chain, sides):
    """读数头 —— ⛔ 绿必须自陈它这一轮到底量了什么(gate0_guard.finish 的同一条纪律)。"""
    _say("=" * 72)
    _say("%s(链 %s)· 解释器 %s" % (gate_name, chain, sys.executable))
    _say("  torch 轴:%s" % _torch_version())
    for k, v in sides:
        _say("  %-10s %s" % (k, v))
    sk = skipped_stages()
    if sk:
        _say("  ⚠ 这一跑跳过了:%s ⇒ 相关那一侧**不是本轮产物**,按冻结参照记账" % ", ".join(sk))
    _say("=" * 72)


def _torch_version():
    """⚠ 四个 compare 的文件头与 README 都写着「双方同 torch(2.5.1)」,而跑器把其中四条
    路由到 **2.11.0+cu130**(S134 §3 就记过,四个月没改)。这条链的全部价值建立在
    「两侧同 torch」上,而**今天没有任何判据在看它** ⇒ 至少先把它打进转录。"""
    try:
        import torch
        return torch.__version__
    except Exception:                                # noqa: BLE001
        return "(未安装 / 这条 compare 不吃 torch)"


# --------------------------------------------------------------------------- 自检
def _selftest():
    import shutil
    import tempfile
    fails = []

    def expect_unrunnable(name, fn, because=None):
        """⛔ `because` 不是装饰:S139 实测,这个自检的「陈货」那一条曾经**因为另一个原因**
        (过滤器写错 ⇒ 匹配到 0 件)而绿 —— 红了、类型也对,但它在回答另一个问题。
        ⇒ 断言必须钉到**理由**上。"""
        try:
            fn()
        except GateUnrunnable as e:
            if because and because not in str(e):
                fails.append("%s 红了,但**理由不对**:期望提到 %r,实际说的是:%s"
                             % (name, because, str(e).splitlines()[0][:120]))
                return
            _say("  ok   %s -> GateUnrunnable%s" % (name, "(%s)" % because if because else ""))
            return
        except Exception as e:                       # noqa: BLE001
            fails.append("%s 抛的是 %r,不是 GateUnrunnable" % (name, e))
            return
        fails.append("%s 应该抛 GateUnrunnable,却过了" % name)

    def expect_ok(name, fn):
        try:
            fn()
            _say("  ok   %s -> 正常通过" % name)
        except Exception as e:                       # noqa: BLE001
            fails.append("%s 不该抛,却抛了 %r" % (name, e))

    import time
    tmp = tempfile.mkdtemp(prefix="gate1_guard_selftest_")
    old_t0 = os.environ.pop(T0_ENV, None)
    try:
        now = time.time()
        # 1) t0 分层:GATE0_T0 不许被 gate1 读到
        os.environ["GATE0_T0"] = "%.3f" % now
        expect_unrunnable("read_t0(只有 GATE0_T0,没有 GATE1_T0)", lambda: read_t0("st"))
        os.environ[T0_ENV] = "%.3f" % now
        expect_ok("read_t0(有 GATE1_T0)", lambda: read_t0("st"))

        # 2) events 文件数:0 / 2 / 目录不在 —— 三种都必须 UNRUNNABLE 且措辞不同
        # ⚠ 文件名照真实形状造:`events.out.tfevents.<ts>.<host>.<pid>.<n>`
        #   —— **`tfevents` 在中间**,这正是过滤器写错时会被掩盖掉的那一点。
        d = os.path.join(tmp, "logs")
        expect_unrunnable("tb_scalars(目录不在)", lambda: tb_scalars("st", d, ["a"], now),
                          because="logdir 不在")
        os.makedirs(d)
        expect_unrunnable("tb_scalars(目录在但空)", lambda: tb_scalars("st", d, ["a"], now),
                          because="一个 events 文件都没有")
        f1 = "events.out.tfevents.1786400000.HOST.111.0"
        f2 = "events.out.tfevents.1786400001.HOST.222.0"
        for n in (f1, f2):
            with open(os.path.join(d, n), "wb") as f:
                f.write(b"x")
            os.utime(os.path.join(d, n), (now + 5, now + 5))
        expect_unrunnable("tb_scalars(两个 events 文件)",
                          lambda: tb_scalars("st", d, ["a"], now), because="应当只有 1 个")
        os.remove(os.path.join(d, f2))
        # 只剩一个,但内容不是真 events ⇒ tag 缺失(仍然是 UNRUNNABLE 不是判负)
        expect_unrunnable("tb_scalars(tag 缺失)", lambda: tb_scalars("st", d, ["a"], now),
                          because="少了")
        # 陈货 —— ⛔ 这一条必须以「不是本轮产物」为理由红,不许以「只有 0 件」为理由红
        os.utime(os.path.join(d, f1), (now - 86400, now - 86400))
        expect_unrunnable("tb_scalars(陈货)", lambda: tb_scalars("st", d, ["a"], now),
                          because="不是本轮产物")

        # 3) 步集
        expect_ok("require_exact_steps(相等)",
                  lambda: require_exact_steps("st", "rvc", range(30), 30,
                                              other=range(30), other_label="orig"))
        expect_unrunnable("require_exact_steps(只跑了 10 步,而真值 30)",
                          lambda: require_exact_steps("st", "rvc", range(10), 30))
        expect_unrunnable("require_exact_steps(两侧步集不同)",
                          lambda: require_exact_steps("st", "rvc", range(30), 30,
                                                      other=range(1, 31), other_label="orig"))

        # 4) None 分量
        if require_no_none("st", {1: {"g": 1.0}, 2: {"g": None}}, ["g"]):
            _say("  ok   require_no_none(有 None) -> 点名")
        else:
            fails.append("require_no_none 没抓住 None")
        if require_no_none("st", {1: {"g": 1.0}}, ["g"]):
            fails.append("require_no_none 对干净数据误报")
        else:
            _say("  ok   require_no_none(干净) -> 不报")

        # 5) 夹取记账
        G._reset_uncovered()
        note_clamped("st/kl", [0, 2], 30)
        if len(G._uncovered) != 1:
            fails.append("note_clamped 没记账")
        G._reset_uncovered()
        expect_unrunnable("note_clamped(夹取过半 ⇒ 不构成判定)",
                          lambda: note_clamped("st/kl", list(range(20)), 30))
        G._reset_uncovered()

        # 6) jsonl:不在 / 陈货 / None 不许被筛掉
        jp = os.path.join(tmp, "steps.jsonl")
        expect_unrunnable("jsonl_steps(文件不在)",
                          lambda: jsonl_steps("st", jp, "g_total", now))
        with open(jp, "w", encoding="utf-8") as f:
            f.write('{"type":"step","step":0,"losses":{"g_total":1.0}}\n')
            f.write('{"type":"step","step":1,"losses":{"g_total":null}}\n')
        os.utime(jp, (now + 5, now + 5))
        got, nones = jsonl_steps("st", jp, "g_total", now)
        if sorted(got) == [0, 1] and nones == [1]:
            _say("  ok   jsonl_steps(None 那一步**留在集合里**并被点名)")
        else:
            fails.append("jsonl_steps 把 None 那步筛掉了:steps=%s nones=%s"
                         % (sorted(got), nones))
        os.utime(jp, (now - 86400, now - 86400))
        expect_unrunnable("jsonl_steps(陈货)",
                          lambda: jsonl_steps("st", jp, "g_total", now))
    finally:
        os.environ.pop("GATE0_T0", None)
        if old_t0 is not None:
            os.environ[T0_ENV] = old_t0
        else:
            os.environ.pop(T0_ENV, None)
        shutil.rmtree(tmp, ignore_errors=True)

    _say()
    if fails:
        for f in fails:
            _say("  FAIL %s" % f)
        _say("gate1_guard 自检: FAILED(%d)" % len(fails))
        return EXIT_SELFTEST
    _say("gate1_guard 自检: ALL OK")
    return EXIT_PASS


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    _say(__doc__)
    sys.exit(EXIT_PASS)
