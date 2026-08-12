# -*- coding: utf-8 -*-
"""gate1 五条 compare 的**阴性对照** —— 用合成夹具把每一条判据真触发一次。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_negctl.py

S139(§F7 笔 E)新建。立项理由是 S129 铁律的同族:**一条从没被执行过的错误分支就是
一条空判据**;而 S128 的血训更狠一档:**工装报的 RED 有三分之一是假的** ⇒ 每种闸都要有
基线自检。五条 gate1 compare 在 S139 之前 **0/5 有自检或阴性对照**,而同目录里另外 7 个闸有。

⛔ 它**只写系统临时目录**,一个字节都不碰:仓库 / 上游树 / `TESTING\\utai-v2-testing` /
   `TESTING\\gate1_vocoder` / `SingingVocoders`。每个场景在**子进程**里跑(退出码必须是
   真的传出来的,不是 try/except 模拟的)。

────────────────────────────────────────────────────────────────────────────
每一行都对着一条 S139 实测的、**修之前会打绿或归因错**的事故
────────────────────────────────────────────────────────────────────────────
  健康           exit 0   对照臂 —— 证明这些新判据没有把真绿弄没
  缺 t0          exit 3   此前没有这个概念(零新鲜度)
  陈货           exit 3   ⛔ 此前:2026-07-07 的 jsonl 打出与 S134 转录逐字符相同的 ALL PASS
  logdir 不在    exit 3   ⛔ 此前:未捕获 DirectoryDeletedError ⇒ rc=1 = 真红的码
  logdir 空      exit 3   ⛔ 此前:vocoder 打 `PASS tag sets identical (0 tags)` **退 0**
  两个 events    exit 3   ⛔ 此前:拼接 + 后写者赢,而打印的步数一点异常都没有
  步数不足       exit 3   ⛔ 此前:地板是常数 10 而真值 30/16/14 ⇒ `ALL PASS (10 steps)` 退 0
  两侧同改 tag   exit 3   ⛔ 此前:vocoder 两侧各 15 点、数值差 1000 倍,**同样退 0**
  train 缺席     exit 3   ⛔ 此前:diff 的 validation 齐全时打 `GATE1 DIFF: PASS` 退 0
  None 分量      exit 1   ⛔ 此前:v2 把整步筛掉 ⇒ `aligned: 12` [PASS] —— 而 NaN 正是它的立项理由
  数值真的差     exit 1   阳性对照 —— 证明它还会红
  夹取致盲       exit 3   ⛔ 此前:我方 kl 改成 1e9 仍打 `ALL PASS (30 steps compared)`
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate1_guard as G1                                        # noqa: E402

_say = G1._say

RUNNER = textwrap.dedent(r'''
    import importlib.util, os, sys
    gt, mod_name, over_json = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, gt)
    import gate1_guard as G1
    spec = importlib.util.spec_from_file_location("t", os.path.join(gt, mod_name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k, v in __import__("json").loads(over_json).items():
        setattr(m, k, v)
    G1.run(getattr(m, "GATE", mod_name), m.main)
''').strip()


# ─────────────────────────────────────────────────────────── 合成夹具
def write_tb(logdir, series, jitter=0.0):
    """series = {tag: {step: value}};写成一个真的 events 文件。"""
    os.makedirs(logdir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir=logdir)
    for tag, pts in series.items():
        for s, v in sorted(pts.items()):
            w.add_scalar(tag, v + jitter, s)
    w.close()


def write_jsonl(path, steps, keys, override=None):
    override = override or {}
    with open(path, "w", encoding="utf-8") as f:
        for s in steps:
            losses = {k: 1.0 + 0.01 * s + 0.001 * i for i, k in enumerate(keys)}
            for (os_, ok_), ov in override.items():
                if os_ == s:
                    losses[ok_] = ov
            f.write(json.dumps({"type": "step", "step": s, "losses": losses}) + "\n")


SHA_A = "a" * 64
SHA_B = "b" * 64


def seed_ident(dirs, sha=SHA_A):
    """往这些目录里写 `gate1_input.identity.json`。

    ⛔ S140:此前合成夹具**一个身份文件都不写**,而真 prepare 五条链每一条都写两侧
       ⇒ 「健康」那条对照臂验的是一个**盘上不存在的形状**(S140 侦察点名的那一条)。
       补上之后,身份缺席/不符才有资格成为**另外两条**独立的阴性对照。
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        G1.write_input_identity(d, {
            "root": "(negctl 合成夹具)", "subs": [], "files": 208, "bytes": 1,
            "dirhash_sha256": sha,
            "src_newest_mtime": "2026-01-01T00:00:00",
            "recorded_at": "2026-01-01T00:00:00"})


def apply_ident(over, orig_dir, ours_dir, ident):
    """ident: same / differ / absent_orig / absent_both"""
    if ident == "absent_both":
        return
    if ident == "absent_orig":
        seed_ident([ours_dir])
        return
    seed_ident([orig_dir])
    seed_ident([ours_dir], SHA_B if ident == "differ" else SHA_A)


def touch(path, when):
    for root, _d, files in os.walk(path) if os.path.isdir(path) else [(None, None, None)]:
        if root is None:
            break
        for fn in files:
            os.utime(os.path.join(root, fn), (when, when))
    if os.path.isfile(path):
        os.utime(path, (when, when))


# ─────────────────────────────────────────────────────────── 链的合成规格
def rvc_fixture(td, steps=None, kl_override=None, jitter=0.0, clamp_steps=(0, 2),
                ident="same", pairs=None):
    steps = steps if steps is not None else list(range(30))
    tb = os.path.join(td, "orig")
    keys = ["g_total", "d_total", "fm", "mel", "kl"]
    tags = ["loss/g/total", "loss/d/total", "loss/g/fm", "loss/g/mel", "loss/g/kl"]
    # ⛔ 让 kl 在 step 0 与 2 顶满 9.0 —— 这是**今天盘上真夹具的形状**(2/30 步顶满,含 step 0)
    series = {}
    for i, t in enumerate(tags):
        series[t] = {s: 1.0 + 0.01 * s + 0.001 * i for s in steps}
    for s in clamp_steps:
        if s in steps:
            series["loss/g/kl"][s] = 9.0
    write_tb(tb, series, jitter=0.0)
    jp = os.path.join(td, "ours.jsonl")
    write_jsonl(jp, steps, keys, override=kl_override)
    if jitter:
        # 我方侧整体偏一个量 ⇒ 阳性对照(必须红)
        rows = [json.loads(l) for l in open(jp, encoding="utf-8") if l.strip()]
        with open(jp, "w", encoding="utf-8") as f:
            for r in rows:
                r["losses"] = {k: (v + jitter if v is not None else None)
                               for k, v in r["losses"].items()}
                f.write(json.dumps(r) + "\n")
    ours_exp = os.path.join(td, "ours_exp")
    over = dict(ORIG_TB_DIR=tb, OURS_JSONL=jp, OURS_EXP=ours_exp)
    if pairs is not None:
        over["PAIRS"] = pairs
    apply_ident(over, tb, ours_exp, ident)
    return over, [tb, jp]


def sovits_fixture(td, steps=None, jitter=0.0):
    steps = steps if steps is not None else list(range(16))
    tb = os.path.join(td, "orig")
    keys = ["g_total", "d_total", "fm", "mel", "kl", "lf0"]
    tags = ["loss/g/total", "loss/d/total", "loss/g/fm", "loss/g/mel",
            "loss/g/kl", "loss/g/lf0"]
    write_tb(tb, {t: {s: 1.0 + 0.01 * s + 0.001 * i for s in steps}
                  for i, t in enumerate(tags)})
    jp = os.path.join(td, "ours.jsonl")
    write_jsonl(jp, steps, keys)
    if jitter:
        rows = [json.loads(l) for l in open(jp, encoding="utf-8") if l.strip()]
        with open(jp, "w", encoding="utf-8") as f:
            for r in rows:
                r["losses"] = {k: v + jitter for k, v in r["losses"].items()}
                f.write(json.dumps(r) + "\n")
    ours_exp = os.path.join(td, "ours_exp")
    over = dict(ORIG_TB_DIR=tb, OURS_JSONL=jp, OURS_EXP=ours_exp)
    apply_ident(over, tb, ours_exp, "same")
    return over, [tb, jp]


V2_TAGS = ["loss/total", "loss/mel", "loss/adv", "loss/fm", "loss/mel_ddsp",
           "loss/spec_ddsp", "loss/mel_am", "loss/kl_div", "loss/lf0"]
V2_KEYS = ["g_total", "mel", "adv", "fm", "mel_ddsp", "spec_ddsp", "mel_am", "kl", "lf0"]


def v2_fixture(td, steps=None, none_at=None, jitter=0.0, orig_nan_at=None):
    steps = steps if steps is not None else list(range(14))
    tb = os.path.join(td, "orig")
    series = {t: {s: 1.0 + 0.01 * s + 0.001 * i for s in steps}
              for i, t in enumerate(V2_TAGS)}
    # ⛔ S140:往**参照（TB）那一侧**注射非有限值 —— `require_no_none` 只吃我方 JSONL，
    #    而旧的滚动比较 `if rel > worst[0]` 对 NaN 恒为 False ⇒ 无论在哪一步都被丢掉。
    for _s in (orig_nan_at or []):
        if _s in series[V2_TAGS[0]]:
            series[V2_TAGS[0]][_s] = float("nan")
    write_tb(tb, series)
    jp = os.path.join(td, "ours.jsonl")
    ov = {}
    for s in (none_at or []):
        for k in V2_KEYS:
            ov[(s, k)] = None
    write_jsonl(jp, steps, V2_KEYS, override=ov)
    if jitter:
        rows = [json.loads(l) for l in open(jp, encoding="utf-8") if l.strip()]
        with open(jp, "w", encoding="utf-8") as f:
            for r in rows:
                r["losses"] = {k: (v + jitter if v is not None else None)
                               for k, v in r["losses"].items()}
                f.write(json.dumps(r) + "\n")
    ours_exp = os.path.join(td, "ours_exp")
    over = dict(ORIG_TB_DIR=tb, OURS_JSONL=jp, OURS_EXP=ours_exp)
    apply_ident(over, tb, ours_exp, "same")
    return over, [tb, jp]


def diff_fixture(td, train_steps=None, val_steps=(8, 16, 24), jitter=0.0, orig_nan_at=None):
    train_steps = list(range(1, 25)) if train_steps is None else train_steps
    o, u = os.path.join(td, "orig", "logs"), os.path.join(td, "ours", "logs")
    for d, j in ((o, 0.0), (u, jitter)):
        series = {}
        if train_steps:
            series["train/loss"] = {s: 1.0 + 0.01 * s + j for s in train_steps}
        series["validation/loss"] = {s: 0.5 + 0.001 * s + j for s in val_steps}
        if d is o:
            for _s in (orig_nan_at or []):
                if "train/loss" in series and _s in series["train/loss"]:
                    series["train/loss"][_s] = float("nan")
        write_tb(d, series)
    over = dict(ORIG_DIR=o, OURS_DIR=u)
    apply_ident(over, os.path.dirname(o), os.path.dirname(u), "same")
    return over, [o, u]


VOC_TRAIN = ["training/DmpdlossF", "training/DmpdlossT", "training/DmsdlossF",
             "training/DmsdlossT", "training/Gmpd_feature_loss", "training/Gmpdloss",
             "training/Gmsd_feature_loss", "training/Gmsdloss", "training/aux_mel_loss"]
VOC_VAL = ["validation/stft_loss", "validation/total_loss"]


def voc_fixture(td, n_train=15, rename=False, jitter=0.0, empty=False,
                nan_tag=None, extra_tag_ours=False, tally_dead=False):
    o, u = os.path.join(td, "orig"), os.path.join(td, "ours")
    tsteps = [2 * (i + 1) for i in range(n_train)]
    vsteps = [0, 10, 20, 30]
    for d, j in ((o, 0.0), (u, jitter)):
        if empty:
            os.makedirs(d, exist_ok=True)
            continue
        series = {}
        for i, t in enumerate(VOC_TRAIN):
            name = t.replace("training/", "train/") if rename else t
            series[name] = {s: 1.0 + 0.01 * s + 0.001 * i + j for s in tsteps}
        for i, t in enumerate(VOC_VAL):
            name = t.replace("validation/", "val/") if rename else t
            series[name] = {s: 0.5 + 0.001 * s + j for s in vsteps}
        if nan_tag and d is o:
            series[nan_tag] = {s: float("nan") for s in series[nan_tag]}
        if extra_tag_ours and d is u:
            series["training/BRAND_NEW"] = {s: 1.0 for s in tsteps}
        write_tb(d, series)
    orig_exp = os.path.join(td, "orig_exp")
    ours_exp = os.path.join(td, "ours_exp")
    # ⛔ S140:reporter 记账也要造 —— 真跑一定会写它,合成夹具不写就等于让
    #    「健康」那条对照臂又去验一个盘上不存在的形状(这一场刚为身份文件补过同一件事)。
    os.makedirs(ours_exp, exist_ok=True)
    with open(os.path.join(ours_exp, "reporter_tally.json"), "w", encoding="utf-8") as f:
        # ⛔ 与 `gate1_guard.EXPECT["vocoder"]["tally"]` 的登记值**逐项一致** ——
        #    S140 那一跑量出来的真值。合成夹具与真夹具不同形,「健康」那条对照臂就在
        #    验一个盘上不存在的形状(这一场已经为身份文件补过同一件事)。
        tally = dict(G1.EXPECT["vocoder"]["tally"])
        tally["n_done"] = 0
        if tally_dead:
            tally["n_step"] = 0
        json.dump(tally, f)
    over = dict(ORIG_LOGS=o, OURS_LOGS=u, ORIG_EXP=orig_exp, OURS_EXP=ours_exp, GATE_ROOT=td)
    apply_ident(over, orig_exp, u, "same")
    return over, [o, u]


# ─────────────────────────────────────────────────────────── 场景表
def build_cases():
    """(名字, 模块, 造夹具的函数, 期望退出码, 期望在转录里出现的关键词)"""
    return [
        # ── 对照臂:健康数据必须绿 ────────────────────────────────
        ("rvc/健康", "gate1_compare.py", lambda td: rvc_fixture(td), 0, "ALL PASS"),
        ("sovits/健康", "gate1_sovits_compare.py", lambda td: sovits_fixture(td), 0, "ALL PASS"),
        ("v2/健康", "gate1_sovits_v2_compare.py", lambda td: v2_fixture(td), 0, "ALL PASS"),
        ("diff/健康", "gate1_diff_compare.py", lambda td: diff_fixture(td), 0, "ALL PASS"),
        ("voc/健康", "gate1_vocoder_compare.py", lambda td: voc_fixture(td), 0, "ALL PASS"),

        # ── ⛔ 空集/半集假 PASS(修之前这四条全是 exit 0)────────────
        ("voc/两侧 logdir 空", "gate1_vocoder_compare.py",
         lambda td: voc_fixture(td, empty=True), 3, "一个 events 文件都没有"),
        ("voc/两侧同时改 tag 名", "gate1_vocoder_compare.py",
         lambda td: voc_fixture(td, rename=True), 3, "少了"),
        ("diff/train 缺席而 validation 齐全", "gate1_diff_compare.py",
         lambda td: diff_fixture(td, train_steps=[]), 3, "少了"),
        ("rvc/只跑了 10 步(真值 30)", "gate1_compare.py",
         lambda td: rvc_fixture(td, steps=list(range(10))), 3, "登记的真值"),
        ("sovits/只跑了 10 步(真值 16)", "gate1_sovits_compare.py",
         lambda td: sovits_fixture(td, steps=list(range(10))), 3, "登记的真值"),
        ("v2/只跑了 10 步(真值 14)", "gate1_sovits_v2_compare.py",
         lambda td: v2_fixture(td, steps=list(range(10))), 3, "登记的真值"),
        ("voc/训练点数只有 10(真值 15)", "gate1_vocoder_compare.py",
         lambda td: voc_fixture(td, n_train=10), 3, "登记的真值"),

        # ── ⛔ None = 发散,此前整步被筛掉 ──────────────────────────
        ("v2/九分量全 None@两步(真发散形状)", "gate1_sovits_v2_compare.py",
         lambda td: v2_fixture(td, none_at=[9, 10]), 1, "发散"),
        ("rvc/kl 单步 None", "gate1_compare.py",
         lambda td: rvc_fixture(td, kl_override={(5, "kl"): None}), 1, "发散"),

        # ── ⛔ 夹取致盲:两条,回答的是**两个不同的问题** ────────────
        #    ⒜ 致盲面**变了** ⇒ 那是新消息,必须 exit 3 要求人来核实并更新登记值。
        ("rvc/夹取致盲面从 2 变成 5 步", "gate1_compare.py",
         lambda td: rvc_fixture(td, clamp_steps=(0, 2, 4, 6, 8)), 3, "致盲面变了"),
        #    ⒝ 致盲面**没变**,而我方在被致盲的那两步给出 1e9 ⇒ 这台闸**确实看不见**,
        #       ⛔ 但它必须**在转录里自己说出来**。这一条买的不是「它能抓住」,
        #          而是「它不会假装自己抓得住」—— 诚实边界也是一条判据。
        ("rvc/致盲区内我方=1e9(闸自陈看不见)", "gate1_compare.py",
         lambda td: rvc_fixture(td, kl_override={(0, "kl"): 1e9, (2, "kl"): 1e9}),
         0, "在数值上不可证伪"),

        # ── ⛔⛔ S140:**参照(TB)那一侧的非有限值**。此前五条里只有 vocoder 判它;
        #    diff 用内置 `max(生成器)`、sovits_v2 用滚动 `if rel > worst[0]`,
        #    而 NaN 的一切比较都是 False ⇒ **静默丢掉**(v2 连首位都丢,九分量全 NaN 时
        #    打 `max_rel=0.000e+00 @ step -1` 并 PASS)。而 NaN 正是这台闸的立项理由。
        #    ⚠ **首步与中间步各一条** —— 内置 max 的行为**取决于 NaN 落在哪一位**,
        #      只测一处会得到一条「红对了但理由不是我以为的那个」的绿。
        ("v2/参照 TB 首步 NaN", "gate1_sovits_v2_compare.py",
         lambda td: v2_fixture(td, orig_nan_at=[0]), 1, "非有限"),
        ("v2/参照 TB 中间步 NaN", "gate1_sovits_v2_compare.py",
         lambda td: v2_fixture(td, orig_nan_at=[7]), 1, "非有限"),
        ("diff/参照 TB 首步 NaN", "gate1_diff_compare.py",
         lambda td: diff_fixture(td, orig_nan_at=[1]), 1, "非有限"),
        ("diff/参照 TB 中间步 NaN", "gate1_diff_compare.py",
         lambda td: diff_fixture(td, orig_nan_at=[12]), 1, "非有限"),
        ("voc/某 tag 全非有限", "gate1_vocoder_compare.py",
         lambda td: voc_fixture(td, nan_tag="training/Gmpdloss"), 1, "非有限"),

        # ── ⛔ S140:单侧多一个 tag。S139 重写时把「两臂 tag 集合相等」删掉了而没记账
        #    ⇒ 「我方多 log 了一个量 / 上游新增了一个」今天是**零信号**。
        ("voc/我方单侧多一个 tag", "gate1_vocoder_compare.py",
         lambda td: voc_fixture(td, extra_tag_ours=True), 3, "tag 集合"),

        # ── ⛔ S140:reporter 通道。此前 `_Rep` 三个方法全 `pass` ⇒ 这一面零判据。
        #    「记账里 n_step=0」正是那个原始状态,它必须红 —— 否则改回去没人看得见。
        ("voc/reporter 记账说它收到 0 条", "gate1_vocoder_compare.py",
         lambda td: voc_fixture(td, tally_dead=True), 3, "reporter 记账与登记值不同"),

        # ── ⛔ S140:登记的分量数变判据。此前 EXPECT[*]["components"] 是**零读者**,
        #    从 PAIRS 里删掉一个分量 ⇒ 每行照打 [PASS]、总判照打 ALL PASS,转录零变化。
        ("rvc/PAIRS 少一个分量(登记 5)", "gate1_compare.py",
         lambda td: rvc_fixture(td, pairs=[["loss/g/total", "g_total", None],
                                           ["loss/d/total", "d_total", None],
                                           ["loss/g/fm", "fm", None],
                                           ["loss/g/mel", "mel", 75.0]]),
         3, "两份登记对不上"),

        # ── ⛔ S140:输入身份。**缺席**此前只打一行字(汇报不是判据);
        #    **两侧不同**那条在生产链上结构不可能红(五个 prepare 把同一个 ident 写两侧),
        #    所以它只能在这里被真触发一次 —— 两条一起才说明这台机器还有牙。
        ("rvc/输入身份缺席(orig 侧)", "gate1_compare.py",
         lambda td: rvc_fixture(td, ident="absent_orig"), 3, "输入身份缺席"),
        ("rvc/输入身份缺席 +放行", "gate1_compare.py",
         lambda td: rvc_fixture(td, ident="absent_orig"), 0, "PASS-WITH-GAPS",
         "normal", ("--allow-uncovered",)),
        ("rvc/两侧输入身份不同", "gate1_compare.py",
         lambda td: rvc_fixture(td, ident="differ"), 3, "两侧吃的不是同一棵树"),

        # ── 阳性对照:数值真的差必须红(exit 1),而不是 3 ────────────
        ("rvc/数值差 5%", "gate1_compare.py", lambda td: rvc_fixture(td, jitter=0.05),
         1, "FAIL"),
        ("v2/数值差 5%", "gate1_sovits_v2_compare.py", lambda td: v2_fixture(td, jitter=0.05),
         1, "FAIL"),
        ("diff/数值差 5%", "gate1_diff_compare.py", lambda td: diff_fixture(td, jitter=0.05),
         1, "FAIL"),
        ("voc/数值差 5%", "gate1_vocoder_compare.py", lambda td: voc_fixture(td, jitter=0.05),
         1, "FAIL"),
    ]


def run_case(name, mod, make, want, needle, mode="normal", extra_argv=()):
    td = tempfile.mkdtemp(prefix="gate1_negctl_")
    try:
        over, paths = make(td)
        now = time.time()
        t0 = now - 5.0
        if mode == "stale":
            for p in paths:
                touch(p, t0 - 86400)
        elif mode.startswith("skip_orig"):
            # ⛔ `--skip-orig` 那一跑里参照侧**本来就不是本轮的** ⇒ 它该走 declare_frozen
            #    + note_uncovered(结论是 exit 3 或 PASS-WITH-GAPS),**不许是干净的绿**,
            #    也不许被新鲜度当成一条陈货红。paths[0] 恒为参照侧(五条链的夹具都这么排)。
            touch(paths[0], t0 - 86400)
            for p in paths[1:]:
                touch(p, now)
        else:
            for p in paths:
                touch(p, now)
        if mode == "missing":
            for p in paths:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)
        if mode == "two_events":
            for p in paths:
                if os.path.isdir(p):
                    src = [f for f in os.listdir(p) if "tfevents" in f]
                    if src:
                        shutil.copy2(os.path.join(p, src[0]),
                                     os.path.join(p, src[0] + ".dup"))
                        os.rename(os.path.join(p, src[0] + ".dup"),
                                  os.path.join(p, "events.out.tfevents.9.dup.0"))
        rp = os.path.join(td, "_runner.py")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(RUNNER)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env.pop(G1.SKIPPED_ENV, None)
        if mode.startswith("skip_orig"):
            env[G1.SKIPPED_ENV] = "orig"
        if mode != "no_t0":
            env[G1.T0_ENV] = "%.3f" % t0
        else:
            env.pop(G1.T0_ENV, None)
        p = subprocess.run([sys.executable, "-u", rp, HERE, mod, json.dumps(over)]
                           + list(extra_argv),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=600)
        body = (p.stdout or "") + (p.stderr or "")
        ok_code = p.returncode == want
        ok_needle = needle in body
        if ok_code and ok_needle:
            _say("  ok   %-46s exit %d  (点名:%s)" % (name, p.returncode, needle))
            return None
        why = []
        if not ok_code:
            why.append("exit 应为 %d,实际 %d" % (want, p.returncode))
        if not ok_needle:
            why.append("转录里没有 %r" % needle)
        tail = [l for l in body.strip().splitlines() if l.strip()][-4:]
        return "%s:%s\n        %s" % (name, ";".join(why), "\n        ".join(tail))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    fails = []
    _say("=" * 78)
    _say("gate1 五条 compare 的阴性对照(合成夹具,全部只写系统临时目录)")
    _say("=" * 78)
    for case in build_cases():
        r = run_case(*case)
        if r:
            fails.append(r)

    # ── 跨链的三条:缺 t0 / 陈货 / 目录不在 / 两个 events —— 每条链都要过一遍
    _say("")
    per_chain = [
        ("rvc", "gate1_compare.py", rvc_fixture),
        ("sovits", "gate1_sovits_compare.py", sovits_fixture),
        ("v2", "gate1_sovits_v2_compare.py", v2_fixture),
        ("diff", "gate1_diff_compare.py", diff_fixture),
        ("voc", "gate1_vocoder_compare.py", voc_fixture),
    ]
    for label, mod, fx in per_chain:
        for mode, needle in (("no_t0", "不可归因"), ("stale", "不是本轮产物"),
                             ("missing", "不在"), ("two_events", "events 文件")):
            r = run_case("%s/%s" % (label, mode), mod, lambda td, f=fx: f(td), 3, needle,
                         mode=mode)
            if r:
                fails.append(r)

    # ── ⛔ `--skip-orig` 那条路:S139 **自己新造的分支**,而第一版的 40 条对照
    #    把 GATE1_SKIPPED 清掉了 ⇒ 它一次没被执行过 = S129 说的空判据。补上。
    #    两条一起才是判据:不放行 ⇒ exit 3(**不许是干净的绿**);显式放行 ⇒ PASS-WITH-GAPS 退 0。
    _say("")
    for label, mod, fx in per_chain:
        for extra, want, needle in (((), 3, "零覆盖"),
                                    (("--allow-uncovered",), 0, "PASS-WITH-GAPS")):
            nm = "%s/--skip-orig%s" % (label, " +放行" if extra else "")
            r = run_case(nm, mod, lambda td, f=fx: f(td), want, needle,
                         mode="skip_orig", extra_argv=extra)
            if r:
                fails.append(r)

    _say("")
    if fails:
        for f in fails:
            _say("  FAIL %s" % f)
        _say("gate1_negctl: FAILED(%d)" % len(fails))
        return 1
    _say("gate1_negctl: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
