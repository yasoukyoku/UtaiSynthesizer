# -*- coding: utf-8 -*-
"""关卡:S117 §F2⒜ 的续训状态 sidecar(GradScaler / RNG / 数据集身份)。

⛔ 判据要求(照 gate_ckpt_guard / gate_numerics_guard 的形状):
   * 每条断言各问各的,变异要能各红各的;
   * ★必须有一条**阳性对照**:不恢复 scaler 时轨迹要真的分叉,否则「恢复之后逐位相同」
     这句话不携带任何信息(S115 血训);
   * ★必须有一条钉住「全新训练不受影响」—— 那是这一刀最容易造出的退化。
"""
import copy
import json
import os
import random
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
REPO_TRAINING = r"D:\MyDev\Utai_v2-dev\training"
if REPO_TRAINING not in sys.path:
    sys.path.insert(0, REPO_TRAINING)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utai_train import device as device_shim  # noqa: E402
from utai_train import resume_state as RS  # noqa: E402
from utai_train.rvc.infer_pack import commons  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not cond else ""))


tmp = tempfile.mkdtemp(prefix="s117_rs_gate_")
with open(os.path.join(tmp, "dataset.fingerprint"), "w", encoding="utf-8") as f:
    f.write("aaaabbbbcccc0001|enc=x|loudnorm=0")

print("=== R1-R6: 捕获/写入/读回 ===")
torch.manual_seed(5)
random.seed(6)
np.random.seed(7)
scaler = device_shim.make_scaler("cuda" if torch.cuda.is_available() else "cpu",
                                 torch.cuda.is_available())
if scaler.is_enabled():
    scaler._scale = torch.full((), 1024.0, dtype=torch.float32, device="cuda")
blob = RS.capture(scaler, epoch=4, global_step=1234, exp_dir=tmp,
                  dataset_items=24058, loader_len=6014)
p = os.path.join(tmp, RS.LATEST_NAME)
RS.write(p, blob)
check("R1 sidecar 落盘且是合法 JSON", os.path.isfile(p) and isinstance(json.load(open(p, encoding="utf-8")), dict))
check("R2 写入是原子的(不留 .tmp)", not os.path.exists(p + ".tmp"))
check("R3 记下了数据集身份与两个规模", blob["dataset_fingerprint"].startswith("aaaabbbb")
      and blob["dataset_items"] == 24058 and blob["loader_len"] == 6014)
check("R4 记下了四套 RNG 里的至少 python/torch_cpu/numpy",
      {"python", "torch_cpu", "numpy"} <= set(blob["rng"].keys()), str(sorted(blob["rng"])))
check("★R5 epoch 对不上的 sidecar 必须被拒(kill 在两次写之间会留下陈货)",
      RS.load_for(5, p) is None and RS.load_for(4, p) is not None)
check("R6 schema 不认识就整份丢掉,而不是猜",
      (lambda bad: RS.read(bad) is None)(
          (lambda q: (open(q, "w", encoding="utf-8").write(json.dumps({"schema": 999})), q)[1])(
              os.path.join(tmp, "bad.json"))))

print()
print("=== R7-R9: RNG 逐位恢复 ===")
got = RS.load_for(4, p)
nxt_expected = (float(torch.randn(1)), random.random(), float(np.random.rand(1)[0]))
torch.manual_seed(999)
random.seed(999)
np.random.seed(999)
RS.restore(got, None)
nxt_got = (float(torch.randn(1)), random.random(), float(np.random.rand(1)[0]))
for label, a, b in zip(("torch_cpu", "python", "numpy"), nxt_expected, nxt_got):
    check("R7[%s] 恢复之后抽出来的下一个数逐位相同" % label, a == b, "%r vs %r" % (a, b))

print()
print("=== R10-R12: 漂移描述 ===")
code, lines = RS.describe_drift(got, exp_dir=tmp, dataset_items=24058, loader_len=6014)
check("R10 同一份数据:不报 CODE", code is None, str(code))
check("R10b 但仍然把身份打进日志(下一次报告要靠它)", any("dataset unchanged" in l for l in lines))
with open(os.path.join(tmp, "dataset.fingerprint"), "w", encoding="utf-8") as f:
    f.write("ffff0000dddd9999|enc=x|loudnorm=0")
code2, lines2 = RS.describe_drift(got, exp_dir=tmp, dataset_items=20000, loader_len=5000)
check("★R11 数据换了必须报 CODE_DATASET_CHANGED", code2 == RS.CODE_DATASET_CHANGED, str(code2))
check("R11b 并说清两个规模各变成了多少",
      any("24058 -> 20000" in l for l in lines2) and any("6014 -> 5000" in l for l in lines2),
      str(lines2))
check("R12 没有 sidecar 时说【不知道】,不假装没变",
      RS.describe_drift(None, exp_dir=tmp)[0] is None
      and "unknown" in RS.describe_drift(None, exp_dir=tmp)[1][0])

print()
print("=== R13-R15: scaler 的三种臂 ===")
cpu_scaler = device_shim.make_scaler("cpu", False)
blob_cpu = RS.capture(cpu_scaler, epoch=1, global_step=0, exp_dir=tmp)
check("R13 关掉的 scaler:记成 None + scaler_enabled=False(而不是一个空 dict)",
      blob_cpu["scaler"] is None and blob_cpu["scaler_enabled"] is False)
rep = RS.restore(blob_cpu, device_shim.make_scaler("cuda" if torch.cuda.is_available() else "cpu", False))
check("R14 fp32 存档喂给 fp32 运行:不报错,如实说没有", "not in checkpoint" in (rep.scaler or ""), rep.scaler)
if torch.cuda.is_available():
    rep2 = RS.restore(blob, device_shim.make_scaler("cpu", False))
    check("R15 fp16 存档喂给【关掉 scaler 的】运行:跳过并说明(用户把 fp16 关了)",
          "skipped" in (rep2.scaler or ""), rep2.scaler)
    s_live = device_shim.make_scaler("cuda", True)
    rep3 = RS.restore(blob, s_live)
    # ⚠ 断言 scaler **自己的读数**,不是报告里的那句话。第一版只查了字符串,而变异 N1
    # (删掉 load_state_dict、留下那句 "restored")照样让它绿 —— 报的是文档不是行为。
    check("R15b fp16 存档喂给 fp16 运行:scaler 的 scale 真的变成了存档里那个值",
          s_live.get_scale() == 1024.0, "get_scale()=%r report=%r" % (s_live.get_scale(), rep3.scaler))
    check("R15c 而且报告如实说了 restored", "restored" in (rep3.scaler or ""), rep3.scaler)
else:
    print("  (无 CUDA,R15 系跳过)")

print()
print("=== ★R16-R18: 阳性对照 —— 不恢复 scaler 时轨迹真的分叉吗 ===")
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False   # 与 rvc/train.py:70-71 相同
    torch.backends.cudnn.benchmark = False

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(8, 32, 5, padding=2), nn.LeakyReLU(0.1),
                                     nn.Conv1d(32, 8, 5, padding=2))

        def forward(self, x):
            return self.net(x)

    def make():
        torch.manual_seed(7)
        m = Toy().cuda()
        return m, torch.optim.AdamW(m.parameters(), 1e-4, betas=[0.8, 0.99], eps=1e-9)

    def batch(i):
        g = torch.Generator().manual_seed(1000 + i)
        return torch.randn(4, 8, 128, generator=g).cuda(), torch.randn(4, 8, 128, generator=g).cuda()

    def run(m, o, sc, lo, hi):
        skipped = 0
        for i in range(lo, hi):
            x, y = batch(i)
            w0 = m.net[0].weight.detach().clone()
            with device_shim.autocast("cuda", enabled=True):
                out = m(x)
                with device_shim.autocast("cuda", enabled=False):
                    loss = ((out.float() - y) ** 2).mean() * 800.0
            o.zero_grad()
            sc.scale(loss).backward()
            sc.unscale_(o)
            commons.clip_grad_value_(m.parameters(), None)
            sc.step(o)
            sc.update()
            if torch.equal(w0, m.net[0].weight.detach()):
                skipped += 1
        return skipped

    def fp(m):
        return torch.cat([q.detach().float().flatten().cpu() for q in m.parameters()])

    WARM, AFTER = 40, 25
    mA, oA = make(); sA = device_shim.make_scaler("cuda", True)
    run(mA, oA, sA, 0, WARM); run(mA, oA, sA, WARM, WARM + AFTER)
    fA = fp(mA)
    mA2, oA2 = make(); sA2 = device_shim.make_scaler("cuda", True)
    run(mA2, oA2, sA2, 0, WARM); run(mA2, oA2, sA2, WARM, WARM + AFTER)
    floor = float((fA - fp(mA2)).abs().max())
    check("★R16 噪声地板 = 0(否则下面两条不携带信息)", floor == 0.0, "%.3e" % floor)

    mB, oB = make(); sB = device_shim.make_scaler("cuda", True)
    run(mB, oB, sB, 0, WARM)
    state = RS.capture(sB, epoch=1, global_step=WARM, exp_dir=tmp)
    sd = {k: v.detach().clone() for k, v in mB.state_dict().items()}
    od = {"state": {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                    for k, v in oB.state_dict()["state"].items()},
          "param_groups": oB.state_dict()["param_groups"]}

    def resumed(restore_scaler):
        m, o = make()
        m.load_state_dict(sd)
        # ⛔ deepcopy is LOAD-BEARING, and leaving it out produced a false red once (S117).
        # `Optimizer.load_state_dict` casts with `tensor.to(dtype, device)`, which RETURNS THE
        # SAME OBJECT when nothing has to change — so the optimizer's live `exp_avg`/`exp_avg_sq`
        # become the very tensors in `od`, and `step()` mutates them in place. The first arm would
        # then hand the second arm a state it had already trained on.
        o.load_state_dict(copy.deepcopy(od))
        s = device_shim.make_scaler("cuda", True)
        if restore_scaler:
            RS.restore(state, s)
        sk = run(m, o, s, WARM, WARM + AFTER)
        return float((fA - fp(m)).abs().max()), sk

    dC, skC = resumed(False)
    dD, skD = resumed(True)
    check("★R17 不恢复 scaler:轨迹分叉,而且有若干步被静默跳过",
          dC > floor and skC > 0, "diff=%.3e skipped=%d" % (dC, skC))
    check("★★R18 恢复 scaler:与【从没停过】逐位相同,零步被跳过",
          dD == 0.0 and skD == 0, "diff=%.3e skipped=%d" % (dD, skD))
else:
    print("  (无 CUDA —— 阳性对照【测不了】,不是【通过】)")
    check("R16-R18 需要 CUDA", False, "本机没有 CUDA:这三条是空的,不许当成绿的")

print()
print("=== R19: 全新训练不受影响 ===")
empty = tempfile.mkdtemp(prefix="s117_fresh_")
check("★R19 没有 sidecar 时 load_for 返回 None ⇒ 全新训练一个字节都不受影响",
      RS.load_for(1, os.path.join(empty, RS.LATEST_NAME)) is None)
rep_none = RS.restore(None, device_shim.make_scaler("cpu", False))
check("R19b restore(None) 是无害的空操作", rep_none.scaler is None and rep_none.rng == {})

print()
print("=== ★R20-R28: 可续训的 best 存档(§F2⒜ 的正题)===")
from utai_train import numerics  # noqa: E402
from utai_train.sovits import train as SOVITS  # noqa: E402
from utai_train.sovits import utils as SOVITS_UTILS  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.emb_g = nn.Embedding(4, 8)


ws = tempfile.mkdtemp(prefix="s117_best_")
torch.manual_seed(21)
bg, bd = Tiny(), Tiny()
bog = torch.optim.AdamW(bg.parameters(), 1e-4)
bod = torch.optim.AdamW(bd.parameters(), 1e-4)
for net, opt in ((bg, bog), (bd, bod)):     # 让 AdamW 真的建出动量
    sum(p.float().pow(2).sum() for p in net.parameters()).backward()
    opt.step(); opt.zero_grad(set_to_none=False)

blob_best = RS.capture(None, epoch=7, global_step=1400, exp_dir=ws, dataset_items=10, loader_len=5)
out = RS.save_best_pair(ws, SOVITS_UTILS.save_checkpoint, (bg, bd), (bog, bod), 1e-4,
                        epoch=7, metric=12.5, blob=blob_best)
check("R20 三个文件都写出来了", all(os.path.isfile(os.path.join(out, n))
                                   for n in (RS.BEST_G, RS.BEST_D, RS.BEST_STATE)))
import glob as _glob  # noqa: E402

check("★R21 它【不在】model_dir 的 G_*/D_* 平面上(五个消费点都在那一层:"
      "sovits 从 D 文件名 parse step · clean_checkpoints · 两个 latest_checkpoint_path · resume_was_intended)",
      not _glob.glob(os.path.join(ws, "G_*.pth")) and not _glob.glob(os.path.join(ws, "D_*.pth")))
check("R22 state.json 记下了 metric/epoch/step", (lambda b: b["best_metric"] == 12.5
      and b["epoch"] == 7 and b["global_step"] == 1400)(RS.read_best(ws)))
os.remove(os.path.join(out, RS.BEST_STATE))
check("★R23 少了完成标记就当作【没有】,而不是半份",
      RS.read_best(ws) is None)
RS.write(os.path.join(out, RS.BEST_STATE), dict(blob_best, best_metric=12.5))
os.remove(os.path.join(out, RS.BEST_D))
check("★R23b 有标记但少了 D 也当作【没有】(GAN 只能从一对续)", RS.read_best(ws) is None)

print()
print("--- choose_pair ---")
mdir = tempfile.mkdtemp(prefix="s117_choose_")
torch.manual_seed(3)
latest_net = Tiny()
lo = torch.optim.AdamW(latest_net.parameters(), 1e-4)
for n in ("G_800", "D_800"):
    SOVITS_UTILS.save_checkpoint(latest_net, lo, 1e-4, 5, os.path.join(mdir, "%s.pth" % n))
RS.save_best_pair(mdir, SOVITS_UTILS.save_checkpoint, (bg, bd), (bog, bod), 1e-4,
                  epoch=7, metric=12.5, blob=blob_best)
lf = lambda pat: SOVITS_UTILS.latest_checkpoint_path(mdir, pat)  # noqa: E731
g1, d1, b1, s1 = RS.choose_pair(mdir, RS.PREFER_LATEST, lf)
check("R24 默认取最新那一对,且不带 best 的 blob",
      s1 == RS.PREFER_LATEST and b1 is None and g1.endswith("G_800.pth"))
g2, d2, b2, s2 = RS.choose_pair(mdir, RS.PREFER_BEST, lf)
check("★R25 要 best 就给 best,并把它的 step 一起给出来",
      s2 == RS.PREFER_BEST and b2 is not None and b2["global_step"] == 1400
      and g2.endswith(RS.BEST_G) and d2.endswith(RS.BEST_D))
empty2 = tempfile.mkdtemp(prefix="s117_nobest_")
for n in ("G_800", "D_800"):
    SOVITS_UTILS.save_checkpoint(latest_net, lo, 1e-4, 5, os.path.join(empty2, "%s.pth" % n))
g3, d3, b3, s3 = RS.choose_pair(empty2, RS.PREFER_BEST,
                                lambda pat: SOVITS_UTILS.latest_checkpoint_path(empty2, pat))
check("★R26 要 best 但没有 ⇒ 退回最新,并且**把 source 如实报成 latest**(调用方据此发警告)",
      s3 == RS.PREFER_LATEST and b3 is None)

print()
print("--- 端到端:从 best 续训 ---")
g5, d5 = Tiny(), Tiny()
og5, od5 = torch.optim.AdamW(g5.parameters(), 1e-4), torch.optim.AdamW(d5.parameters(), 1e-4)
ep, gs, side = SOVITS.load_start_state(mdir, g5, d5, og5, od5, False, prefer=RS.PREFER_BEST)
check("★★R27 prefer=best:权重来自 best 那一对,step 用它自己记的 1400+1",
      gs == 1401 and abs(float(g5.a.weight.abs().sum()) - float(bg.a.weight.abs().sum())) < 1e-6,
      "step=%s" % gs)
g6, d6 = Tiny(), Tiny()
og6, od6 = torch.optim.AdamW(g6.parameters(), 1e-4), torch.optim.AdamW(d6.parameters(), 1e-4)
ep6, gs6, _ = SOVITS.load_start_state(mdir, g6, d6, og6, od6, False, prefer=RS.PREFER_LATEST)
check("★R28 prefer=latest(默认)行为不变:走 G_800/D_800,step=801",
      gs6 == 801 and abs(float(g6.a.weight.abs().sum()) - float(latest_net.a.weight.abs().sum())) < 1e-6,
      "step=%s" % gs6)

print()
print("--- optimizer_state_is_safe(实测机制,不是手戳一个 inf 进去)---")
torch.manual_seed(11)
hn = Tiny()
ho = torch.optim.AdamW(hn.parameters(), 1e-4, betas=[0.8, 0.99], eps=1e-9)
for _ in range(5):
    for p in hn.parameters():
        p.grad = torch.full_like(p, 0.05)
    ho.step(); ho.zero_grad(set_to_none=False)
check("R29 健康的动量:通过", numerics.optimizer_state_is_safe((("G", ho),)))
weights_before = float(hn.a.weight.detach().abs().sum())
for p in hn.parameters():                    # 一个「巨大但有限」的梯度,阈值以上
    p.grad = torch.full_like(p, 1e21)
ho.step(); ho.zero_grad(set_to_none=False)
check("★R30 一个 1e21 的【有限】梯度之后:权重仍然全部有限……",
      numerics.best_save_is_safe(hn.state_dict()))
check("★★R31 ……但动量已经是 inf,而 best_save_is_safe 看不见它",
      not numerics.optimizer_state_is_safe((("G", ho),)))

print()
print(f"gate_resume_state: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
