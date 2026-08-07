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
print("=== C12: refuse_unreadable 带 CODE ===")
e = ckpt_guard.refuse_unreadable(d1, RuntimeError("boom"))
check("C12 unreadable 的消息带自己的 CODE 与原异常类型",
      ckpt_guard.FAILED_CODE in str(e) and "RuntimeError" in str(e), str(e))

print()
print(f"gate_ckpt_guard: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
