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


def touch(path, when):
    for root, _d, files in os.walk(path) if os.path.isdir(path) else [(None, None, None)]:
        if root is None:
            break
        for fn in files:
            os.utime(os.path.join(root, fn), (when, when))
    if os.path.isfile(path):
        os.utime(path, (when, when))


# ─────────────────────────────────────────────────────────── 链的合成规格
def rvc_fixture(td, steps=None, kl_override=None, jitter=0.0, clamp_steps=(0, 2)):
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
    return dict(ORIG_TB_DIR=tb, OURS_JSONL=jp), [tb, jp]


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
    return dict(ORIG_TB_DIR=tb, OURS_JSONL=jp), [tb, jp]


V2_TAGS = ["loss/total", "loss/mel", "loss/adv", "loss/fm", "loss/mel_ddsp",
           "loss/spec_ddsp", "loss/mel_am", "loss/kl_div", "loss/lf0"]
V2_KEYS = ["g_total", "mel", "adv", "fm", "mel_ddsp", "spec_ddsp", "mel_am", "kl", "lf0"]


def v2_fixture(td, steps=None, none_at=None, jitter=0.0):
    steps = steps if steps is not None else list(range(14))
    tb = os.path.join(td, "orig")
    write_tb(tb, {t: {s: 1.0 + 0.01 * s + 0.001 * i for s in steps}
                  for i, t in enumerate(V2_TAGS)})
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
    return dict(ORIG_TB_DIR=tb, OURS_JSONL=jp), [tb, jp]


def diff_fixture(td, train_steps=None, val_steps=(8, 16, 24), jitter=0.0):
    train_steps = list(range(1, 25)) if train_steps is None else train_steps
    o, u = os.path.join(td, "orig", "logs"), os.path.join(td, "ours", "logs")
    for d, j in ((o, 0.0), (u, jitter)):
        series = {}
        if train_steps:
            series["train/loss"] = {s: 1.0 + 0.01 * s + j for s in train_steps}
        series["validation/loss"] = {s: 0.5 + 0.001 * s + j for s in val_steps}
        write_tb(d, series)
    return dict(ORIG_DIR=o, OURS_DIR=u), [o, u]


VOC_TRAIN = ["training/DmpdlossF", "training/DmpdlossT", "training/DmsdlossF",
             "training/DmsdlossT", "training/Gmpd_feature_loss", "training/Gmpdloss",
             "training/Gmsd_feature_loss", "training/Gmsdloss", "training/aux_mel_loss"]
VOC_VAL = ["validation/stft_loss", "validation/total_loss"]


def voc_fixture(td, n_train=15, rename=False, jitter=0.0, empty=False):
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
        write_tb(d, series)
    return dict(ORIG_LOGS=o, OURS_LOGS=u), [o, u]


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


def run_case(name, mod, make, want, needle, mode="normal"):
    td = tempfile.mkdtemp(prefix="gate1_negctl_")
    try:
        over, paths = make(td)
        now = time.time()
        t0 = now - 5.0
        if mode == "stale":
            for p in paths:
                touch(p, t0 - 86400)
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
        if mode != "no_t0":
            env[G1.T0_ENV] = "%.3f" % t0
        else:
            env.pop(G1.T0_ENV, None)
        p = subprocess.run([sys.executable, "-u", rp, HERE, mod, json.dumps(over)],
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
