# -*- coding: utf-8 -*-
"""`gate1_history_compare` 的阴性对照 —— 每一条错误分支用合成夹具真触发一次。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_history_compare.py --selftest

⛔ 立项理由:它的前身 `TESTING\\s134_f7\\compare_vs_history.py` **一条错误分支都没被执行过**
   (全仓零调用者、无 `--selftest`、无 argparse、`main()` 模块级裸调),而 S140 实测
   它整条链上没有一处能红 —— 三种改法(备份根改名 / live 根改名 / 单档文件名写错)
   **全部干净退 0**。⇒ 新版本的每一条判据都必须在这里被真触发一次。

⛔ 它**只写系统临时目录**,一个字节都不碰仓库 / `TESTING` / 上游树。
   每个场景在**子进程**里跑(退出码必须是真的传出来的)。

⭐ 其中两条是**把断言钉到理由上**的那一对(S139 §6 的血训:「判据红了,但它红的理由
   不是我以为的那个」——本场我自己造的五个洞里三个是这个形状):
     · 「只改 `eta_secs`」**必须绿** ⇒ 证明归一化真的在起作用,而不是碰巧相同;
     · 「只改一个**非 losses** 字段(ckpt 的 path)」**必须红** ⇒ 证明主判据量的是
       **整份 jsonl**,不是只有 losses 那几个数。
   少了任何一条,这台闸都可能在「其实只比了 losses」的状态下打绿。
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
import gate0_guard as G                                          # noqa: E402
import gate1_guard as G1                                         # noqa: E402

_say = G1._say

RUNNER = textwrap.dedent(r'''
    import importlib.util, os, sys
    gt, over_json = sys.argv[1], sys.argv[2]
    sys.path.insert(0, gt)
    import gate1_guard as G1
    spec = importlib.util.spec_from_file_location(
        "t", os.path.join(gt, "gate1_history_compare.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k, v in __import__("json").loads(over_json).items():
        setattr(m, k, v)
    G1.run(m.GATE, lambda: m.main(sys.argv[3:]))
''').strip()

# 合成链:2 步 × 2 分量 = 4 对。⛔ 用**真值**做地板,不是一个与真值无关的常数。
CHAIN = "rvc"
FN = "gate1_ours_steps.jsonl"
STEPS = [0, 1]
KEYS = ["g_total", "mel"]
PAIRS = len(STEPS) * len(KEYS)


def rows(bump=0.0, eta_shift=0, ckpt_path="w/a.pth", nan_at=None, drop_key=None,
         extra_step=False):
    out = []
    for s in STEPS + ([9] if extra_step else []):
        losses = {k: 1.0 + 0.01 * s + 0.001 * i + bump for i, k in enumerate(KEYS)}
        if drop_key:
            losses.pop(drop_key, None)
        if nan_at is not None and s == nan_at:
            losses[KEYS[0]] = float("nan")
        out.append({"type": "step", "step": s, "losses": losses,
                    "eta_secs": 100 + eta_shift, "lr": 1e-4})
    # 末尾那条 forced 收尾 step(真夹具四条链都有),losses 为空 —— 必须被剔除而不是吃掉数据
    out.append({"type": "step", "step": max(STEPS) + 1, "losses": {}, "eta_secs": 0})
    out.append({"type": "ckpt", "kind": "final", "step": max(STEPS), "path": ckpt_path})
    return out


def write_jsonl(path, objs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            # ⚠ 用 `allow_nan=True`(默认)写出裸 `NaN` token —— `json.loads` 读回来是
            #   `float('nan')` 且 `isinstance(nan,(int,float))` 为 True,正是前身穿得过去的那条路。
            f.write(json.dumps(o) + "\n")


def build(td, live_kw=None, base_kw=None):
    live_root = os.path.join(td, "live")
    base_root = os.path.join(td, "base")
    write_jsonl(os.path.join(live_root, FN), rows(**(live_kw or {})))
    write_jsonl(os.path.join(base_root, FN), rows(**(base_kw or {})))
    return live_root, base_root


def over_for(live_root, base_root):
    sha = G.dirhash(base_root, [""], [FN])
    return {
        "LIVE": live_root,
        "JSONL_FILES": [FN],
        "BASELINES": [["base", base_root, sha, "negctl 合成基线"]],
        "CASES": {CHAIN: {"kind": "jsonl", "file": FN, "keys": sorted(KEYS),
                          "pairs": PAIRS, "steps": len(STEPS)}},
    }


def scenario(name, want, needle, live_kw=None, base_kw=None, mutate=None,
             stale=False, no_t0=False, argv=("--chain", CHAIN)):
    td = tempfile.mkdtemp(prefix="gate1_hist_negctl_")
    try:
        live_root, base_root = build(td, live_kw, base_kw)
        now = time.time()
        t0 = now - 5.0
        for p in (os.path.join(live_root, FN),):
            os.utime(p, (t0 - 86400, t0 - 86400) if stale else (now, now))
        over = over_for(live_root, base_root)
        if mutate:
            mutate(live_root, base_root, over)
        rp = os.path.join(td, "_runner.py")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(RUNNER)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env.pop(G1.SKIPPED_ENV, None)
        if no_t0:
            env.pop(G1.T0_ENV, None)
        else:
            env[G1.T0_ENV] = "%.3f" % t0
        p = subprocess.run([sys.executable, "-u", rp, HERE, json.dumps(over)] + list(argv),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=600)
        body = (p.stdout or "") + (p.stderr or "")
        ok = p.returncode == want and needle in body
        if ok:
            _say("  ok   %-46s exit %d  (点名:%s)" % (name, p.returncode, needle))
            return None
        why = []
        if p.returncode != want:
            why.append("exit 应为 %d,实际 %d" % (want, p.returncode))
        if needle not in body:
            why.append("转录里没有 %r" % needle)
        tail = [l for l in body.strip().splitlines() if l.strip()][-5:]
        return "%s:%s\n        %s" % (name, ";".join(why), "\n        ".join(tail))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _rename_base(live_root, base_root, over):
    os.rename(base_root, base_root + "_MOVED")


def _flip_base_byte(live_root, base_root, over):
    p = os.path.join(base_root, FN)
    data = bytearray(open(p, "rb").read())
    data[0] ^= 0x01
    open(p, "wb").write(bytes(data))          # ⚠ 长度不变 ⇒ 字节数那把尺子看不见


def _drop_live(live_root, base_root, over):
    os.remove(os.path.join(live_root, FN))


def main():
    _say("=" * 78)
    _say("gate1_history_compare 的阴性对照(合成夹具,只写系统临时目录)")
    _say("=" * 78)
    fails = []
    cases = [
        # ── 对照臂
        ("健康(两侧逐字节同)", 0, "LOSS-TRACE-IDENTICAL", {}, {}, None, False, False),
        # ⭐ 把断言钉到**理由**上的那一对
        ("只差 eta_secs ⇒ 必须绿(归一化真在起作用)", 0, "LOSS-TRACE-IDENTICAL",
         {"eta_shift": 77}, {}, None, False, False),
        ("只改一个**非 losses** 字段(ckpt path)⇒ 必须红", 1, "仍然不逐字节相同",
         {"ckpt_path": "w/OTHER.pth"}, {}, None, False, False),
        # ── ⛔ 前身在这三种状态下**全部干净退 0**
        ("基线根被改名 ⇒ 不许是绿", 3, "参照物没了", {}, {}, _rename_base, False, False),
        ("live 产物缺席 ⇒ 不许是绿", 3, "不在", {}, {}, _drop_live, False, False),
        ("基线内容被改一位(长度不变)⇒ 3", 3, "冻结参照的内容", {}, {}, _flip_base_byte, False, False),
        # ── ⛔ 那条死掉的新鲜度守卫的替代品
        ("live 是陈货(不是本轮产物)⇒ 3", 3, "不是本轮产物", {}, {}, None, True, False),
        ("缺 GATE1_T0 ⇒ 3", 3, "不可归因", {}, {}, None, False, True),
        # ── ⛔ 地板:一个分量整个消失(前身:rvc 删任一键仍恰好过线)
        ("live 少一个分量键 ⇒ 3", 3, "分量名单与登记的不同",
         {"drop_key": "mel"}, {"drop_key": "mel"}, None, False, False),
        ("live 多一个 step ⇒ 3", 3, "登记的真值", {"extra_step": True}, {}, None, False, False),
        # ── 阳性对照
        ("数值真的漂了 ⇒ 1", 1, "FAIL", {"bump": 0.05}, {}, None, False, False),
        # ── ⛔ NaN:前身对它恒不可见(`if d > worst[0]` 对 NaN 恒为 False)
        ("live 注入 NaN ⇒ 1 并点名", 1, "非有限", {"nan_at": 1}, {}, None, False, False),
        ("参照侧注入 NaN ⇒ 1 并点名", 1, "非有限", {}, {"nan_at": 0}, None, False, False),
    ]
    for name, want, needle, lk, bk, mut, stale, no_t0 in cases:
        r = scenario(name, want, needle, lk, bk, mut, stale, no_t0)
        if r:
            fails.append(r)
    _say("")
    if fails:
        for f in fails:
            _say("  FAIL %s" % f)
        _say("gate1_history_negctl: FAILED(%d/%d)" % (len(fails), len(cases)))
        return G1.EXIT_SELFTEST
    _say("gate1_history_negctl: ALL OK(%d 场景)" % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
