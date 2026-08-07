# -*- coding: utf-8 -*-
"""关卡:S116 §F5-③ⓒ 的续训 checkpoint 守卫。

⛔ 判据要求(照 gate_numerics_guard / gate_loader_budget 的形状):每条断言各问各的,
   而且**必须有一条钉住「全新训练不会被拒」** —— 那是这一刀最容易造出的退化。
"""
import glob
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
REPO_TRAINING = r"D:\MyDev\Utai_v2-dev\training"
if REPO_TRAINING not in sys.path:
    sys.path.insert(0, REPO_TRAINING)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utai_train import ckpt_guard  # noqa: E402
from utai_train.rvc import train_utils as R  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not cond else ""))


class Tiny(nn.Module):
    def __init__(self, n=8, emb=4):
        super().__init__()
        self.a = nn.Linear(n, n)
        self.emb_g = nn.Embedding(emb, n)


tmp = tempfile.mkdtemp(prefix="s116_gate_")


def write(name, model, optimizer, drop=None, reshape=None):
    p = os.path.join(tmp, name)
    R.save_checkpoint(model, optimizer, 1e-4, 4, p)
    if drop or reshape:
        b = torch.load(p, map_location="cpu", weights_only=False)
        if drop:
            b["model"].pop(drop)
        if reshape:
            k, shp = reshape
            b["model"][k] = torch.zeros(*shp)
        torch.save(b, p)
    return p


print("=== C1-C3: check_resume_state_dict ===")
m = Tiny()
o = torch.optim.AdamW(m.parameters(), 1e-4)
good = write("G_2333333.pth", m, o)
saved = torch.load(good, map_location="cpu", weights_only=False)["model"]
try:
    ckpt_guard.check_resume_state_dict(good, m.state_dict(), saved)
    ok = True
except ckpt_guard.ResumeRefused as e:
    ok, err = False, e
check("C1 一份完整的 checkpoint 不许被拒", ok)

bad_missing = dict(saved)
bad_missing.pop("a.weight")
try:
    ckpt_guard.check_resume_state_dict(good, m.state_dict(), bad_missing)
    raised = None
except ckpt_guard.ResumeRefused as e:
    raised = str(e)
check("C2 缺 key 必须抛 ResumeRefused", raised is not None)
check("C2b 消息带稳定 CODE 与那个 key", bool(raised) and ckpt_guard.CODE in raised and "a.weight" in raised, raised)

bad_shape = dict(saved)
bad_shape["emb_g.weight"] = torch.zeros(99, 8)
try:
    ckpt_guard.check_resume_state_dict(good, m.state_dict(), bad_shape)
    raised2 = None
except ckpt_guard.ResumeRefused as e:
    raised2 = str(e)
check("C3 形状不符必须抛,并说清 need/got", raised2 is not None and "need (4, 8)" in raised2 and "got (99, 8)" in raised2, raised2)

print()
print("=== C4-C7: resume_was_intended ===")
d1 = tempfile.mkdtemp(prefix="s116_dir1_")
check("C4 空目录不是续训", not ckpt_guard.resume_was_intended(d1))
open(os.path.join(d1, "G_2333333.pth"), "wb").close()
check("C5 只有 G 不是续训", not ckpt_guard.resume_was_intended(d1))
open(os.path.join(d1, "D_2333333.pth"), "wb").close()
check("C6 G+D 齐 = 续训(RVC 形状)", ckpt_guard.resume_was_intended(d1))

d2 = tempfile.mkdtemp(prefix="s116_dir2_")
open(os.path.join(d2, "G_0.pth"), "wb").close()
open(os.path.join(d2, "D_0.pth"), "wb").close()
check("★C7 sovits 的 step-0 底模【不是】续训 —— 全新训练不许被拒",
      not ckpt_guard.resume_was_intended(d2, seeded_base_is_step_zero=True))
check("C7b 同一个目录在 RVC 口径下仍算续训(两个口径必须能分开)",
      ckpt_guard.resume_was_intended(d2))
open(os.path.join(d2, "G_800.pth"), "wb").close()
open(os.path.join(d2, "D_800.pth"), "wb").close()
check("C8 真的存过一次之后就是续训了", ckpt_guard.resume_was_intended(d2, seeded_base_is_step_zero=True))

print()
print("=== C9-C11: 生产 load_checkpoint 的 resume 开关 ===")
m2, o2 = Tiny(), None
m2 = Tiny()
o2 = torch.optim.AdamW(m2.parameters(), 1e-4)
p_missing = write("G_missing.pth", m, o, drop="a.weight")
before = float(m2.a.weight.abs().sum())
try:
    R.load_checkpoint(p_missing, m2, o2, resume=False)
    tolerant = True
except ckpt_guard.ResumeRefused:
    tolerant = False
check("C9 resume=False 保持上游的宽容(pretrain 语义不许被这一刀改掉)", tolerant)
check("C9b 而且它确实塞进了随机初始值(说明 C10 拦的是真事)",
      abs(float(m2.a.weight.abs().sum()) - before) < 1e-6)

m3 = Tiny()
o3 = torch.optim.AdamW(m3.parameters(), 1e-4)
try:
    R.load_checkpoint(p_missing, m3, o3, resume=True)
    refused = False
except ckpt_guard.ResumeRefused:
    refused = True
check("★C10 resume=True 必须拒", refused)

m4 = Tiny()
o4 = torch.optim.AdamW(m4.parameters(), 1e-4)
try:
    R.load_checkpoint(good, m4, o4, resume=True)
    fine = True
except ckpt_guard.ResumeRefused:
    fine = False
check("C11 完整 checkpoint 在 resume=True 下照常加载", fine)

print()
print("=== P1-P6: plan_load —— 三个分岔的唯一真源(S117)===")
d3 = tempfile.mkdtemp(prefix="s117_plan_")
check("P1 空目录 = pretrain", ckpt_guard.plan_load(d3) == ckpt_guard.LOAD_PRETRAIN)
check("P1b 空目录 + sovits 口径 = pretrain",
      ckpt_guard.plan_load(d3, seeded_base_is_step_zero=True) == ckpt_guard.LOAD_PRETRAIN)
open(os.path.join(d3, "G_0.pth"), "wb").close()
open(os.path.join(d3, "D_0.pth"), "wb").close()
check("★★P2 只有 step-0 底模 + sovits 口径 = seeded_base(**这一条就是 c44dec6 丢掉的分支**)",
      ckpt_guard.plan_load(d3, seeded_base_is_step_zero=True) == ckpt_guard.LOAD_SEEDED_BASE,
      ckpt_guard.plan_load(d3, seeded_base_is_step_zero=True))
check("P3 同一个目录在 RVC 口径下 = resume(RVC 的底模不叫 *_0)",
      ckpt_guard.plan_load(d3) == ckpt_guard.LOAD_RESUME)
open(os.path.join(d3, "G_800.pth"), "wb").close()
open(os.path.join(d3, "D_800.pth"), "wb").close()
check("P4 真存过一次 = resume", ckpt_guard.plan_load(d3, seeded_base_is_step_zero=True) == ckpt_guard.LOAD_RESUME)
d4 = tempfile.mkdtemp(prefix="s117_plan2_")
open(os.path.join(d4, "G_0.pth"), "wb").close()
check("P5 只有半个底模 = pretrain(撕裂的一半不许当底模加载)",
      ckpt_guard.plan_load(d4, seeded_base_is_step_zero=True) == ckpt_guard.LOAD_PRETRAIN)
check("P6 三个返回值互不相同", len({ckpt_guard.LOAD_RESUME, ckpt_guard.LOAD_SEEDED_BASE,
                                   ckpt_guard.LOAD_PRETRAIN}) == 3)

print()
print("=== B1-B8: ★底模【真的被加载了】—— 这是 c44dec6 溜过去的那条断言(S117)===")
print("    (缺陷的形状是「什么都没做」:S116 的 C7 只钉了『全新训练不许被【拒】』,")
print("     没有任何一条钉『底模仍然被【加载】』。)")
from utai_train.sovits import train as SOVITS  # noqa: E402
from utai_train.sovits import utils as SOVITS_UTILS  # noqa: E402
from utai_train.sovits_v2 import train as SOVITS_V2  # noqa: E402


def seed_base_pair(dirpath, base_net, suffix=0, optimizer_none=True):
    """写一对 G_<suffix>/D_<suffix>。真底模的 optimizer 是 None(实测上游 logs/44k 的 G_0.pth:
    iteration=0, optimizer=None)—— 夹具照着造,否则测的是一个现实里不存在的文件。"""
    o = torch.optim.AdamW(base_net.parameters(), 1e-4)
    for name in ("G", "D"):
        p = os.path.join(dirpath, "%s_%d.pth" % (name, suffix))
        SOVITS_UTILS.save_checkpoint(base_net, o, 1e-4, 0, p)
        if optimizer_none:
            b = torch.load(p, map_location="cpu", weights_only=False)
            b["optimizer"] = None
            torch.save(b, p)
    return dirpath


def fresh_pair():
    torch.manual_seed(4242)
    g, d = Tiny(), Tiny()
    return g, d, torch.optim.AdamW(g.parameters(), 1e-4), torch.optim.AdamW(d.parameters(), 1e-4)


def fingerprint(m):
    return float(m.a.weight.detach().abs().sum())


for label, mod, want_step, why_step in (
    ("sovits", SOVITS, 1, "4.x 在 global_step += 1 之前判 eval_interval ⇒ 0 会在第一步就用练过的副本覆盖原始底模"),
    ("sovits_v2", SOVITS_V2, 0, "v2 有显式的 step-0 处理(存档跳过、evaluate 照跑)"),
):
    dbase = tempfile.mkdtemp(prefix="s117_base_%s_" % label)
    torch.manual_seed(7)
    base = Tiny()
    base_fp = fingerprint(base)
    seed_base_pair(dbase, base)

    g, d, og, od = fresh_pair()
    random_fp = fingerprint(g)
    ep, gs = mod.load_start_state(dbase, g, d, og, od, False)
    check("★★B[%s] 全新工作区必须把底模加载进 net_g(而不是留着随机初始值)" % label,
          abs(fingerprint(g) - base_fp) < 1e-6 and abs(random_fp - base_fp) > 1e-6,
          "loaded=%.6f base=%.6f random=%.6f" % (fingerprint(g), base_fp, random_fp))
    check("B[%s] net_d 同样" % label, abs(fingerprint(d) - base_fp) < 1e-6)
    check("B[%s] epoch_str = 1" % label, ep == 1, str(ep))
    check("★B[%s] global_step = %d —— %s" % (label, want_step, why_step), gs == want_step, str(gs))

    # 空目录:什么都不加载,权重必须原样
    dempty = tempfile.mkdtemp(prefix="s117_empty_%s_" % label)
    g2, d2, og2, od2 = fresh_pair()
    ep2, gs2 = mod.load_start_state(dempty, g2, d2, og2, od2, False)
    check("B[%s] 空工作区:不加载任何东西,(1,0)" % label,
          (ep2, gs2) == (1, 0) and abs(fingerprint(g2) - random_fp) < 1e-6)

    # 真续训:严格检查仍然生效
    dres = tempfile.mkdtemp(prefix="s117_res_%s_" % label)
    seed_base_pair(dres, base, suffix=0)
    torch.manual_seed(31)
    trained = Tiny()
    seed_base_pair(dres, trained, suffix=800, optimizer_none=False)
    g3, d3m, og3, od3 = fresh_pair()
    ep3, gs3 = mod.load_start_state(dres, g3, d3m, og3, od3, False)
    check("B[%s] 真续训:加载 step-800 那一对,step = 801" % label,
          gs3 == 801 and abs(fingerprint(g3) - fingerprint(trained)) < 1e-6,
          "step=%s" % gs3)

    # 真续训 + 缺 key ⇒ 必须拒(S116 那一刀不许被本次重构弄丢)
    bad = torch.load(os.path.join(dres, "G_800.pth"), map_location="cpu", weights_only=False)
    bad["model"].pop("a.weight")
    torch.save(bad, os.path.join(dres, "G_800.pth"))
    g4, d4, og4, od4 = fresh_pair()
    try:
        mod.load_start_state(dres, g4, d4, og4, od4, False)
        refused_still = False
    except ckpt_guard.ResumeRefused:
        refused_still = True
    check("★B[%s] 真续训遇到盖不住的存档仍然拒(S116 §F5-③ⓒ 不许被这次重构弄丢)" % label, refused_still)

d_skip = seed_base_pair(tempfile.mkdtemp(prefix="s117_skip_"), base)
g5, d5, og5, od5 = fresh_pair()
ep5, gs5 = SOVITS_V2.load_start_state(d_skip, g5, d5, og5, od5, True)
check("B[sovits_v2] skip_optimizer=True:epoch/step 被压回 (1,0)", (ep5, gs5) == (1, 0), "%s,%s" % (ep5, gs5))
check("★B[sovits_v2] skip_optimizer=True 时底模【仍然】被加载(它是『用底模播种权重』的旋钮,不是『什么都不加载』)",
      abs(fingerprint(g5) - base_fp) < 1e-6,
      "loaded=%.6f base=%.6f" % (fingerprint(g5), base_fp))

print()
print("=== C12: refuse_unreadable 带 CODE ===")
e = ckpt_guard.refuse_unreadable(d1, RuntimeError("boom"))
check("C12 unreadable 的消息带自己的 CODE 与原异常类型",
      ckpt_guard.FAILED_CODE in str(e) and "RuntimeError" in str(e), str(e))

print()
print(f"gate_ckpt_guard: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
