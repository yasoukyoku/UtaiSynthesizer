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
print("=== ★D1-D14: 浅扩散(§F8⒜ —— 把上面这一套推广过去)===")
# ⛔ 全部驱动生产函数:真的 Saver 写存档、真的 load_start_state 选存档、真的 latest_numbered_path
# 扫目录。夹具只负责造一个小模型 —— 用 torch.save 自己拼 checkpoint 就等于在测我自己的布局。
from utai_train.sovits import diff_pipeline as DP  # noqa: E402
from utai_train.sovits.diffusion.logger import utils as DU  # noqa: E402
from utai_train.sovits.diffusion.logger.saver import Saver  # noqa: E402


class TinyDiff(nn.Module):
    def __init__(self, fill):
        super().__init__()
        self.w = nn.Parameter(torch.full((4,), float(fill)))


def make_saver(expdir, step):
    os.makedirs(expdir, exist_ok=True)
    args = DU.DotDict({"env": {"expdir": expdir}, "data": {"sampling_rate": 44100}})
    sv = Saver(args, initial_global_step=step)
    return sv


def warm(net):
    """真的让 AdamW 建出动量 —— 空 optimizer 与热 optimizer 的区别正是这一刀的标的。"""
    o = torch.optim.AdamW(net.parameters(), 1e-4)
    sum(p.pow(2).sum() for p in net.parameters()).backward()
    o.step()
    o.zero_grad(set_to_none=False)
    return o


def fresh_pair(fill):
    """⚠ 返回值第三项是【热身之后】的真实读数,不是 `fill`。

    第一版拿 fill 当期望值,五条断言全红 —— 因为 warm() 真的走了一步 AdamW,盘上的权重
    是 2.9999… 而不是 3.0。那正是「我编的期望 + 生产写下的真值 = 假断言」(S114),
    正解是把期望从夹具的实际状态里取出来,不是从我脑子里取。
    """
    n = TinyDiff(fill)
    o = warm(n)
    return n, o, float(n.w[0])


def start(expdir, prefer=None):
    n = TinyDiff(-1.0)
    o = torch.optim.AdamW(n.parameters(), 1e-4)
    return DP.load_start_state(expdir, n, o, device="cpu", prefer=prefer), n, o


dexp = os.path.join(tempfile.mkdtemp(prefix="s118_diff_"), "diffusion")
sv = make_saver(dexp, 0)
net_p, opt_p, W_P = fresh_pair(3.0)      # 周期存档的内容
sv.global_step = 2000
sv.save_model(net_p, None, postfix="2000")           # 周期存档:上游 save_opt=false ⇒ 无 optimizer
net_b, opt_b, W_B = fresh_pair(7.0)      # best 的内容
sv.global_step = 1400
RS.save_solo_snapshot(dexp, RS.BEST_DIR, lambda p: sv.save_model_to(p, net_b, opt_b),
                      blob=RS.capture(None, epoch=0, global_step=1400, exp_dir=tmp,
                                      dataset_items=12, loader_len=3),
                      metric=0.25)
bdir = RS.snapshot_dir(dexp, RS.BEST_DIR)
check("D1 best 快照:一个 model.pt + 一个 state.json",
      os.path.isfile(os.path.join(bdir, RS.BEST_MODEL)) and os.path.isfile(os.path.join(bdir, RS.BEST_STATE)))
check("★D2 state.json 自己写下了 payload 清单(读的一侧不许再有第二份硬编清单)",
      RS.read_snapshot(dexp, RS.BEST_DIR)["files"] == [RS.BEST_MODEL],
      str(RS.read_snapshot(dexp, RS.BEST_DIR).get("files")))
os.remove(os.path.join(bdir, RS.BEST_STATE))
check("★D3 少了完成标记就当【没有】 —— 而 model.pt 还在盘上",
      RS.read_snapshot(dexp, RS.BEST_DIR) is None and os.path.isfile(os.path.join(bdir, RS.BEST_MODEL)))
RS.save_solo_snapshot(dexp, RS.BEST_DIR, lambda p: sv.save_model_to(p, net_b, opt_b),
                      blob=RS.capture(None, epoch=0, global_step=1400, exp_dir=tmp), metric=0.25)

# ★ 上游那把「最大编号」的尺子不许被子目录动到 —— 它是递归扫的(os.walk)
p_no_snap, s_no_snap = DU.latest_numbered_path(dexp)
check("★D4 快照子目录【不改变】上游扫描的答案(它是递归的,这条不是显然的)",
      s_no_snap == 2000 and os.path.basename(p_no_snap) == "model_2000.pt",
      "%s step=%s" % (os.path.basename(str(p_no_snap)), s_no_snap))
# 阳性对照:目录名短到 5 字符以下时,分隔符被吃掉、文件名尾巴暴露给 isdigit() ⇒ 真的会劫持
short = os.path.join(tempfile.mkdtemp(prefix="s118_short_"), "diffusion")
sv2 = make_saver(short, 500)
sv2.save_model(net_p, None, postfix="500")
os.makedirs(os.path.join(short, "logs"), exist_ok=True)
sv2.save_model_to(os.path.join(short, "logs", "x9999.pt"), net_p, None)
p_hj, s_hj = DU.latest_numbered_path(short)
check("★D4b 阳性对照:4 字符的子目录名(logs/,而 Saver 真的会建它)确实劫持了扫描",
      s_hj == 9999 and not os.path.isfile(p_hj),
      "step=%s 指向不存在的 %s" % (s_hj, os.path.basename(str(p_hj))))
check("★D4c ⇒ 所以两个快照目录名的长度是【判据】,不是排版",
      len(RS.BEST_DIR) >= RS.SNAPSHOT_DIR_MIN_LEN and len(RS.LATEST_DIR) >= RS.SNAPSHOT_DIR_MIN_LEN
      and RS.SNAPSHOT_DIR_MIN_LEN == 6,
      "best=%d latest=%d min=%d" % (len(RS.BEST_DIR), len(RS.LATEST_DIR), RS.SNAPSHOT_DIR_MIN_LEN))

print()
print("--- 谁被选中 ---")
st, n1, o1 = start(dexp)                      # 默认 latest:此时还没有滚动快照
check("★D5 默认 = 上游那一格(编号 2000),而且如实说它【没带 optimizer】",
      st.source == DP.SRC_NUMBERED and st.step == 2000 and st.had_optimizer is False
      and not o1.state and float(n1.w[0]) == W_P,
      "source=%s step=%s had_opt=%s 动量=%s" % (st.source, st.step, st.had_optimizer, bool(o1.state)))
st, n2, o2 = start(dexp, RS.PREFER_BEST)
check("★★D6 要 best 就给 best:权重来自 best(fill=7)、步号来自它自己的 1400、动量真的回来了",
      st.source == DP.SRC_BEST and st.step == 1400 and st.had_optimizer is True
      and float(n2.w[0]) == W_B and bool(o2.state),
      "source=%s step=%s w=%.3f 动量=%s" % (st.source, st.step, float(n2.w[0]), bool(o2.state)))
check("★D6b 而且把 best 的 state.json 交回给了调用方(scaler/RNG/数据集身份要靠它)",
      st.blob is not None and st.blob.get("best_metric") == 0.25)

# 滚动的完整续训点(生产里由 solver.refresh_resume_point 在每次验证/停止/完成时刷新)
net_l, opt_l, W_L = fresh_pair(9.0)
sv.global_step = 2000
RS.save_solo_snapshot(dexp, RS.LATEST_DIR, lambda p: sv.save_model_to(p, net_l, opt_l),
                      blob=RS.capture(None, epoch=0, global_step=2000, exp_dir=tmp))
st, n3, o3 = start(dexp)
check("★★D7 同一步上【完整的那一份赢】:2000 == 2000 ⇒ 取滚动快照,动量不再是空的",
      st.source == DP.SRC_SNAPSHOT and st.step == 2000 and st.had_optimizer is True
      and float(n3.w[0]) == W_L and bool(o3.state),
      "source=%s step=%s w=%.3f" % (st.source, st.step, float(n3.w[0])))
sv.global_step = 4000
sv.save_model(net_p, None, postfix="4000")     # 编号跑到快照前面去了
st, _, _ = start(dexp)
check("★D8 编号超过快照时不许倒退:4000 > 2000 ⇒ 回到编号那一格",
      st.source == DP.SRC_NUMBERED and st.step == 4000, "source=%s step=%s" % (st.source, st.step))

print()
print("--- 退化路径 / 全新训练 ---")
nobest = os.path.join(tempfile.mkdtemp(prefix="s118_nobest_"), "diffusion")
svn = make_saver(nobest, 600)
svn.save_model(net_p, opt_p, postfix="600")
st, _, o4 = start(nobest, RS.PREFER_BEST)
check("★D9 要 best 但没有完整的 ⇒ 退回去,并把 source 如实报成别的(调用方据此发警告)",
      st.source != DP.SRC_BEST and st.step == 600 and st.had_optimizer is True and bool(o4.state),
      "source=%s" % st.source)
empty_exp = os.path.join(tempfile.mkdtemp(prefix="s118_fresh_"), "diffusion")
os.makedirs(empty_exp)
st, n5, o5 = start(empty_exp)
check("★★D10 全新训练:什么都不恢复、step 0、blob 为 None ⇒ 一个字节都不受影响",
      st.source == DP.SRC_FRESH and st.step == 0 and st.blob is None and not o5.state
      and abs(float(n5.w[0]) + 1.0) < 1e-6)
# 编号扫描指到一个不存在的文件(非规范名/被劫持)——今天会死在 torch.load 里
broken = os.path.join(tempfile.mkdtemp(prefix="s118_broken_"), "diffusion")
svb = make_saver(broken, 100)
svb.save_model_to(os.path.join(broken, "model_0100.pt"), net_p, None)   # int('0100') -> 100
net_r, opt_r, W_R = fresh_pair(5.0)
svb.global_step = 100
RS.save_solo_snapshot(broken, RS.LATEST_DIR, lambda p: svb.save_model_to(p, net_r, opt_r),
                      blob=RS.capture(None, epoch=0, global_step=100, exp_dir=tmp))
st, n6, _ = start(broken)
check("★D11 扫描指向一个不存在的文件时,快照能把它救回来(今天是 FileNotFoundError)",
      st.source == DP.SRC_SNAPSHOT and float(n6.w[0]) == W_R,
      "source=%s" % st.source)

print()
print("--- 底模不许在【其实是续训】的时候被重新播种 ---")
onlysnap = os.path.join(tempfile.mkdtemp(prefix="s118_onlysnap_"), "diffusion")
svs = make_saver(onlysnap, 4000)
RS.save_solo_snapshot(onlysnap, RS.LATEST_DIR, lambda p: svs.save_model_to(p, net_l, opt_l),
                      blob=RS.capture(None, epoch=0, global_step=4000, exp_dir=tmp))


class _NoopReporter:
    def stage(self, *a, **k):
        pass


DP._seed_base_model(onlysnap, "", _NoopReporter())
check("★D12 只剩快照(编号存档被清理/手删了)时,不许再落一个 model_0.pt 进去",
      not os.path.isfile(os.path.join(onlysnap, "model_0.pt")))

print()
print("--- ★清扫器:从 best 回退续训时不许走过别人的分支 ---")
import re as _re  # noqa: E402

from utai_train.sovits.diffusion import solver as SOLVER  # noqa: E402


def sweep_case(files, current, written, superseded, force_save):
    """驱动【真的】_sweep_old_checkpoints + 真的 Saver.delete_model,返回幸存的步号。"""
    d = os.path.join(tempfile.mkdtemp(prefix="s118_sweep_"), "diffusion")
    s = make_saver(d, 0)
    for st in files:
        s.global_step = st
        s.save_model(net_p, None, postfix=str(st))
    a = DU.DotDict({"train": {"interval_force_save": force_save}})
    SOLVER._sweep_old_checkpoints(s, a, current, written, superseded_step=superseded)
    return sorted(int(_re.fullmatch(r"model_(\d+)\.pt", n).group(1))
                  for n in os.listdir(d) if _re.fullmatch(r"model_(\d+)\.pt", n))


# 正常续训(从编号那一格接着练):行为与今天一致 —— 被接续的那一格 + 本轮自己的非里程碑存档死掉,
# 更早的运行留下的东西一个都不许动。
norm = sweep_case([0, 1000, 2000, 3000, 4000], current=5000,
                  written={4000}, superseded=3000, force_save=5000)
check("★D15 正常续训:被接续的那格与本轮的非里程碑死掉,更早运行的留着(= 今天的行为)",
      norm == [0, 1000, 2000], str(norm))

# ★回退续训(从 best 快照):被放弃分支在【上方】,而它的尾巴是唯一带 optimizer 的那个文件。
rew = sweep_case([0, 700, 1400, 2100, 3500, 5000], current=5600,
                 written={2100}, superseded=None, force_save=1400)
check("★★D16 从 best 回退:被放弃分支的 3500 / 5000 必须活着(旧的区间写法会把它们逐个删掉)",
      3500 in rew and 5000 in rew and 2100 not in rew, str(rew))
# 阳性对照:同一批文件、同一个当前步,只把「谁写的」改成「本轮写的」⇒ 它们真的会被删
ctl = sweep_case([0, 700, 1400, 2100, 3500, 5000], current=5600,
                 written={2100, 3500, 5000}, superseded=None, force_save=1400)
check("★D16b 阳性对照:同一格清扫器【有能力】删掉它们 ⇒ 上面那条是所有权规则挡住的,不是别的豁免",
      3500 not in ctl and 5000 not in ctl and rew != ctl, "%s vs %s" % (rew, ctl))
check("★D17 里程碑与 model_0(底模)永不被删",
      0 in rew and 1400 in rew and 0 in ctl and 1400 in ctl, "%s / %s" % (rew, ctl))

print()
print("--- ★NaN 的 best metric:写不进去,而且已经在盘上的那份要被修好 ---")
_SOL = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                         "training", "utai_train", "sovits", "diffusion", "solver.py"),
            encoding="utf-8").read()
check("★D18 写入侧:metric 必须有限才许写 best(json.dump 会写出裸 NaN,load 又收得回来)",
      "if math.isfinite(test_loss) and numerics.best_save_is_safe(" in _SOL)
check("★D19 读取侧:已经记进 best_state.json 的 NaN 要被当作【没有】,否则那个工作区永久冻结",
      "if not math.isfinite(best_metric):" in _SOL)
# 行为面:json 真的会把 NaN 原样往返,所以上面那两条不是防一个不存在的问题
_q = os.path.join(tmp, "best_state_nan.json")
json.dump({"metric": float("nan"), "step": 5}, open(_q, "w", encoding="utf-8"))
_back = json.load(open(_q, encoding="utf-8"))
check("★D20 阳性对照:json 的确会把 NaN 原样往返,而 (0.1 < nan) 恒为 False",
      _back["metric"] != _back["metric"] and not (0.1 < _back["metric"]),
      "%r" % (_back["metric"],))

print()
print("=== ★D21-D27:§F8⒡ 毒存档既不许被写成续训点,也不许被续训 ===")


def poison(net):
    with torch.no_grad():
        net.w[0] = float("nan")
    return net


# ⒜ 发布侧:resume_point_is_safe 的两半各自独立失效(实测机制,不是手戳)
ok_net, ok_opt, _ = fresh_pair(1.0)
check("D21 健康的权重 + 健康的动量:允许发布",
      numerics.resume_point_is_safe(ok_net.state_dict(), (("d", ok_opt),)) is True)
bad_net, bad_opt, _ = fresh_pair(1.0)
poison(bad_net)
check("★D21b 权重坏了:拒绝发布(否则它会覆盖掉【最后一个健康的续训点】,而选择器还优先它)",
      numerics.resume_point_is_safe(bad_net.state_dict(), (("d", bad_opt),)) is False)
hn = TinyDiff(1.0)
ho = torch.optim.AdamW(hn.parameters(), 1e-4, betas=[0.8, 0.99], eps=1e-9)
for p2 in hn.parameters():                      # S117 的机制:巨大但【有限】的梯度
    p2.grad = torch.full_like(p2, 1e21)
ho.step()
check("★D21c 权重全有限、只有动量死了:同样拒绝(两半独立失效 —— S117 实测过这一种)",
      numerics.best_save_is_safe(hn.state_dict()) is True
      and numerics.resume_point_is_safe(hn.state_dict(), (("d", ho),)) is False)

# ⒝ 读取侧:毒存档要被跳过,而且【毒动量不许活到下一次尝试】
pexp = os.path.join(tempfile.mkdtemp(prefix="s118_poison_"), "diffusion")
svp = make_saver(pexp, 0)
good_net, good_opt, W_GOOD = fresh_pair(2.0)
svp.global_step = 100
RS.save_solo_snapshot(pexp, RS.LATEST_DIR, lambda p: svp.save_model_to(p, good_net, good_opt),
                      blob=RS.capture(None, epoch=0, global_step=100, exp_dir=tmp))
bad2 = poison(TinyDiff(8.0))
svp.global_step = 200
svp.save_model(bad2, None, postfix="200")        # 毒的、编号更大的、而且不带 optimizer
st, npm, opm = start(pexp)
check("★★D22 毒存档被跳过,退回更早的健康快照,并把跳过的条数报出来",
      st.source == DP.SRC_SNAPSHOT and st.step == 100 and st.poisoned_skipped == 1
      and float(npm.w[0]) == W_GOOD,
      "source=%s step=%s skipped=%s w=%r" % (st.source, st.step, st.poisoned_skipped, float(npm.w[0])))
check("★★D22b 而且【毒的动量没有活下来】—— 退回的那份带 optimizer,所以动量必须是它的",
      bool(opm.state) and all(bool(torch.isfinite(v).all())
                              for g in opm.state.values() for v in g.values()
                              if torch.is_tensor(v)),
      "动量条目=%d" % len(opm.state))
check("D22c 健康路径不受影响:没有毒存档时 poisoned_skipped == 0",
      start(dexp)[0].poisoned_skipped == 0)

# ⒞ 全都是毒的 ⇒ 拒绝,而不是拿 nan 接着练
allbad = os.path.join(tempfile.mkdtemp(prefix="s118_allbad_"), "diffusion")
sva = make_saver(allbad, 0)
sva.global_step = 50
sva.save_model(poison(TinyDiff(3.0)), None, postfix="50")
_err = None
try:
    start(allbad)
except RuntimeError as e:
    _err = str(e)
check("★D23 一个健康的都没有 ⇒ 带着 CODE 响亮拒绝(不许拿 nan 接着练)",
      _err is not None and _err.startswith(RS.CODE_ARCHIVE_POISONED + ":"), str(_err)[:90])

# ⒟ 上游那道 nan 闸:条件逐字保留,消息换成可本地化的 CODE
check("★D24 nan abort 现在带稳定 CODE(原本是裸英文 prose,backendError.ts 认不出来)",
      "numerics.CODE_DIVERGED, saver.global_step" in _SOL and "if torch.isnan(loss):" in _SOL)
check("★D25 而那个【inf 走得过去】的洞由 DivergenceGuard 接住(patience 不是 1 —— fp16 的单次溢出"
      "按设计会自愈,实测同一个 inf 损失权重毫发无伤)",
      "divergence.observe(saver.global_step, {'loss': current_loss})" in _SOL
      and "numerics.DivergenceGuard(((\"diffusion\", model),)" in _SOL)
check("★D26 滚动续训点的发布走的是那个【模块级】谓词,不是闭包里手写的两句",
      "numerics.resume_point_is_safe(" in _SOL)
check("D27 三个 CODE 都是 SCREAMING_SNAKE 且带 TRAINING_ 前缀",
      all(c.startswith("TRAINING_") and c.replace("_", "").isupper()
          for c in (RS.CODE_DATASET_CHANGED, RS.CODE_OPTIMIZER_NOT_RESTORED, RS.CODE_ARCHIVE_POISONED)))

print()
print("--- 那两个 CODE 的形状 ---")
check("D13 两个 CODE 都是 SCREAMING_SNAKE 且带 TRAINING_ 前缀(它们要当 json 键用)",
      all(c.startswith("TRAINING_") and c.replace("_", "").isupper()
          for c in (RS.CODE_DATASET_CHANGED, RS.CODE_OPTIMIZER_NOT_RESTORED)))
check("★D14 solver 的三个存档点都刷新了滚动续训点(验证 / 停止 / 完成)",
      (lambda s: s.count("refresh_resume_point(epoch)") - s.count("def refresh_resume_point(epoch)") == 3)(
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                            "training", "utai_train", "sovits", "diffusion", "solver.py"),
               encoding="utf-8").read()))

print()
print(f"gate_resume_state: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
