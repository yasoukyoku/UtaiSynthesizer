# -*- coding: utf-8 -*-
"""§F7 笔 D · ②号尺子 —— RVC **预处理** wall-clock,我方 vs 上游,**逐 stage**。

⛔⛔ **这一把与①号尺子(`gate_speed_rvc.py`)的判据形状【不同】,不许照抄。**
   ①号问的是「我方是不是不比上游慢」,单边判红。
   ②号**预先声明我方在两段上是【预期更慢】的** —— 因为那是**设计选择**,不是缺陷:

     · 切片:我方**单进程**(`rvc/preprocess.py:127` 单层 for);上游默认起
       `ceil(n_cpu/1.5)` 个 `multiprocessing.Process`(`preprocess.py:122-127`)。
       ⚠ 我方串行的理由写在 `rvc/preprocess.py:4-6` 的头注里,而且**不是**为了防 OOM、
       也**不是**为了正确性:「identical outputs — files were independent」;
       真实代价是两个**用户可见功能** —— **逐文件 JSONL 进度**与**文件之间可强停**。
     · 特征(ContentVec):我方硬编 `CPUExecutionProvider` + fp32 377 MB
       (`rvc/extract_feature.py:36`);上游走 **GPU + fp16** 的 fairseq HuBERT。
       ⛔ 这一条**不是并行问题是 EP 问题**,而且四个出货 lock 全是 `onnxruntime==1.23.2`
       (CPU 版包名)⇒ **搬 GPU 是发版包的事**,不是改一行 provider。
     · f0:我方 rmvpe **循环外只构造一次**;上游 RVC 每进程一次(默认同卡 2 进程)。
       ⇒ 这一段的方向**不预先声明**,让读数说话。

⇒ **判据 = 「方向与声明相符」+「倍数不超过声明的上界」**,而不是「必须更快」。
   一条预先声明为红的判据,如果不给上界,就只是一句自我实现的话。

⛔ **不许复用任何现存的 `*_orig.py` 驱动**(S138 侦察):
   `gate0_sovits_orig.py:41,74,94` 把 `ProcessPoolExecutor` 换成串行 `InlineExecutor`,
   `gate0_sovits_v2_orig.py:12,14` 把 `Pool`/`Process` 双双内联,`gate0_vocoder.py:5` 直调绕开。
   README 把这登记成「零数值影响」—— **对数值真,对速度是把上游最大的优势亲手抹掉**,
   然后得出「我们不比它慢」。⇒ 本闸**自己起上游的默认并行度**。

出口码(与 `gate0_guard.py` 同一套):0 PASS / 1 真红 / 3 不可归因 / 4 自检失败

============ ⭐⭐⭐ 第一次真跑的读数,以及它【推翻了上面那份预登记】 ============
S138 2026-08-12,`gate_dataset` = **3 个 wav / 180 s / 51 个切片**:

    stage      我方(s)   上游(s)    倍数
    slice        1.56     14.66    0.11×   ⇐ 预登记说「我方更慢 ≤8×」,实测**我方快 9.4 倍**
    f0           6.48      8.09    0.80×
    feature      3.39      5.18    0.65×   ⇐ 预登记说「我方更慢 ≤20×」,实测**我方更快**
    合计        11.43     27.94    0.41×   ⇒ **我方整体快 2.4 倍**

⛔ **预登记那两条【错了】,而且方向相反。** 上面的 `EXPECT` **故意保持原样不改** ——
   预登记的价值就在于它写在跑之前;事后改掉它等于把这一条实测抹掉。

★ **机理(实测支持,而不是我编的自洽解释)**:这份数据集只有 **3 个输入文件**,
  而上游默认起 `ceil(16/1.5) = 11` 个 Windows 进程,**每个都要重新 import torch/fairseq**;
  特征那一段上游是 fairseq HuBERT 上 GPU,**模型加载**在 51 个切片上摊不开。
  ⇒ **进程启动与模型加载压倒了实际计算。**

⛔⛔ **因此这条读数【强烈依赖数据集规模】,不许外推**:
  它说的是「**在 51 个切片这个规模上,我方每一段都更快**」,
  它**没有**说「我方在 500 个切片上也更快」——那正好是并行度开始真正付钱的规模。
  ⇒ **§F10(预处理优化)要的下一个读数是【交叉点在哪】**:
     把数据集按 ×4 / ×16 放大重测,找出上游的并行度开始反超的规模。
     ⚠ 在拿到那个读数之前,**「我方预处理更慢」这个前提是没有证据的**,
     而 S134/S138 两轮侦察都把它当成了既定事实(F10/F11 的 `matters` 全建立在它上面)。
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ARENA = r"D:\MyDev\TESTING\s138_f7\arena"
RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"
# ⛔⛔ 上游侧必须用【RVC 整合包自己的 runtime】(py3.9 / torch 2.0.0+cu118),
#    不是我们的 .venv —— 上游 preprocess 依赖 `ffmpeg`(ffmpeg-python)与 `fairseq`,
#    两个包只在那个 runtime 里。本轮实测:用 .venv 跑上游侧,三段全部
#    `ModuleNotFoundError` ⇒ 上游产物 0 件 ⇒ 闸判 UNRUNNABLE(而它拒绝得对)。
#    ⚠ 这也是 `run_gate0_chain.py:25-29` 早写死的口径:gate0 的两侧解释器
#    **天生不同**(原版侧 = 原版时代环境),「⛔ 不许把它修成 gate1 的形状」。
#    ⇒ 含义:②号尺子量的是【环境轴 + 代码轴】的合体,而那正是用户实际经历的东西。
RVC_PY = os.path.join(RVC, "runtime", "python.exe")
DATASET = r"D:\MyDev\TESTING\utai-v2-testing\gate_dataset"

OURS_PREP = os.path.join(ARENA, "prep_ours")
UP_PREP = os.path.join(RVC, "logs", "s138prep")

# 资产:与生产同一份(`assets{rmvpe_pt, contentvec_onnx}`,见 rvc/pipeline.py:13)
RMVPE_PT = os.path.join(REPO, "data", "models", "auxiliary", "rmvpe.pt")
CONTENTVEC = os.path.join(REPO, "data", "models", "auxiliary", "contentvec_768l12.onnx")

EXIT_PASS, EXIT_RED, EXIT_UNRUNNABLE, EXIT_SELFTEST = 0, 1, 3, 4

# ⛔ 预先声明(跑之前写下的,不许事后改):方向 + 上界。
#    上界不是拍的,是从机理推的,并在头注里说明依据;第一次跑完要把实测填进 §F7 并复核。
EXPECT = {
    "slice":   {"dir": "slower", "bound": 8.0,
                "why": "我方单进程 vs 上游 ceil(n_cpu/1.5) 进程 ⇒ 上界取核数量级"},
    "f0":      {"dir": "unknown", "bound": None,
                "why": "我方 rmvpe 循环外构造一次(赢) vs 上游默认同卡 2 进程(赢)—— 让读数说话"},
    "feature": {"dir": "slower", "bound": 20.0,
                "why": "我方 CPU ORT fp32 377MB vs 上游 GPU fp16 ⇒ 量级可能很大,先给一个宽上界"},
}


class Unrunnable(RuntimeError):
    """这一轮的读数不可归因。⛔ 绝不许被读成『通过』,也不许被读成『慢了』。"""


def n_cpu():
    return os.cpu_count() or 4


def upstream_n_p():
    """上游 infer-web.py:1204 的默认值 `int(np.ceil(config.n_cpu / 1.5))`。"""
    return int(math.ceil(n_cpu() / 1.5))


def _clear(d, subs):
    for s in subs:
        p = os.path.join(d, s)
        if os.path.isdir(p):
            shutil.rmtree(p)
    os.makedirs(d, exist_ok=True)


def _count(d, sub):
    p = os.path.join(d, sub)
    return len(os.listdir(p)) if os.path.isdir(p) else 0


# --------------------------------------------------------------- 我方侧
def run_ours(transcript):
    """三段分开计时。⛔ 每段之前把产物删掉 —— 五条链的 f0/特征全是 skip-if-exists,
    不清就是在量『跳过一遍要多久』(S135 记的那条:那正是 gate0 曾经的假 PASS)。"""
    _clear(OURS_PREP, ["0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "training")
    out = {}
    # ⛔ 签名照生产的调用点抄(`rvc/pipeline.py:119-141`),不许凭记忆写 —— 我第一版就写错了三处,
    #    而这正是 `gate_driver_arity.py` 存在的理由(S134:七个调用点漂了四个月没人看见)。
    code = (
        "import sys,time,json,logging\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "logging.basicConfig(level=logging.ERROR)\n"
        "from utai_train.rvc.preprocess import preprocess_trainset\n"
        "from utai_train.rvc.extract_f0 import extract_f0\n"
        "from utai_train.rvc.extract_feature import extract_features\n"
        "POOL=r'%s'; DS=r'%s'; RMVPE=r'%s'; CVEC=r'%s'\n"
        "class R:\n"
        "    def stage(self,*a,**k): pass\n"
        "    def step(self,*a,**k): pass\n"
        "class S:\n"
        "    def requested(self): return False\n"
        "    def check(self): pass\n"
        "res={}\n"
        "t=time.perf_counter(); preprocess_trainset(DS,48000,POOL,3.7,'ffmpeg',R(),S());"
        " res['slice']=time.perf_counter()-t\n"
        # ⛔ device/is_half 照生产:backend=='cuda' 时 is_half=True(rmvpe 的 half 只在 CUDA 上)
        "t=time.perf_counter(); extract_f0(POOL,RMVPE,'cuda',True,'ffmpeg',R(),S());"
        " res['f0']=time.perf_counter()-t\n"
        "t=time.perf_counter(); extract_features(POOL,'v2',CVEC,R(),S());"
        " res['feature']=time.perf_counter()-t\n"
        "print('@@'+json.dumps(res))\n"
    ) % (OURS_PREP, DATASET, RMVPE_PT, CONTENTVEC)
    with open(transcript, "a", encoding="utf-8", errors="backslashreplace") as tf:
        tf.write("\n===== OURS prep\n")
        r = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE,
                           stderr=tf, text=True, encoding="utf-8", errors="backslashreplace",
                           env=env, cwd=REPO)
        tf.write(r.stdout or "")
    line = [l for l in (r.stdout or "").splitlines() if l.startswith("@@")]
    if not line:
        raise Unrunnable("我方侧没吐出读数(rc=%s)⇒ 这一跑没跑起来。转录:%s" % (r.returncode, transcript))
    out = json.loads(line[-1][2:])
    out["counts"] = {s: _count(OURS_PREP, s) for s in ("0_gt_wavs", "2a_f0", "3_feature768")}
    return out


# --------------------------------------------------------------- 上游侧
def _spawn_parts(script_argv_fn, n_part, transcript, label):
    """上游的 f0/特征是【调用方起 n_part 个进程、各做一片】。⛔ 这正是不许复用
    现存 orig 驱动的原因:它们把并行度内联掉了。"""
    procs = []
    with open(transcript, "a", encoding="utf-8", errors="backslashreplace") as tf:
        tf.write("\n===== ORIG %s  n_part=%d\n" % (label, n_part))
        for i in range(n_part):
            procs.append(subprocess.Popen([RVC_PY] + script_argv_fn(n_part, i),
                                          cwd=RVC, stdout=tf, stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
    return [p.returncode for p in procs]


def run_orig(transcript, n_p=None, n_f0=2):
    _clear(UP_PREP, ["0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"])
    n_p = n_p or upstream_n_p()
    res = {}
    env_note = "n_p=%d(= ceil(%d/1.5),上游 infer-web 默认)· f0 分片=%d(上游默认 gpus_rmvpe='0-0')" \
               % (n_p, n_cpu(), n_f0)
    # ① 切片:上游自己起 n_p 个 Process(noparallel=False)
    t = time.perf_counter()
    with open(transcript, "a", encoding="utf-8", errors="backslashreplace") as tf:
        tf.write("\n===== ORIG slice  n_p=%d\n" % n_p)
        subprocess.run([RVC_PY, os.path.join(RVC, "infer", "modules", "train", "preprocess.py"),
                        DATASET, "48000", str(n_p), UP_PREP, "False", "3.7"],
                       cwd=RVC, stdout=tf, stderr=subprocess.STDOUT)
    res["slice"] = time.perf_counter() - t
    # ② f0(rmvpe,GPU)
    t = time.perf_counter()
    _spawn_parts(lambda n, i: [os.path.join(RVC, "infer", "modules", "train", "extract",
                                            "extract_f0_rmvpe.py"),
                               str(n), str(i), "0", UP_PREP, "True"], n_f0, transcript, "f0")
    res["f0"] = time.perf_counter() - t
    # ③ 特征(fairseq HuBERT,GPU fp16)
    t = time.perf_counter()
    _spawn_parts(lambda n, i: [os.path.join(RVC, "infer", "modules", "train",
                                            "extract_feature_print.py"),
                               "cuda:0", str(n), str(i), UP_PREP, "v2", "True"], 1,
                 transcript, "feature")
    res["feature"] = time.perf_counter() - t
    res["counts"] = {s: _count(UP_PREP, s) for s in ("0_gt_wavs", "2a_f0", "3_feature768")}
    res["_env"] = env_note
    return res


# --------------------------------------------------------------- 判读
def judge(ours, orig):
    print()
    print("[并行度] 上游:%s" % orig.get("_env"))
    print("[并行度] 我方:全部单进程(设计选择,理由见 rvc/preprocess.py:4-6)")
    print()
    # ⛔ 产物件数必须两侧相等,否则量的不是同一件事
    for k in ("0_gt_wavs", "2a_f0", "3_feature768"):
        a, b = ours["counts"].get(k, 0), orig["counts"].get(k, 0)
        if a == 0 or b == 0:
            raise Unrunnable("产物 %s 有一侧是 0 件(我方 %d / 上游 %d)⇒ 这一跑没真做事" % (k, a, b))
        if a != b:
            raise Unrunnable("产物 %s 件数不等(我方 %d / 上游 %d)⇒ 两侧不是同一件事,读数不可比"
                             % (k, a, b))
    print("产物件数两侧相等:%s" % ours["counts"])
    print()
    bad = []
    print("%-9s %10s %10s %8s   %s" % ("stage", "我方(s)", "上游(s)", "倍数", "预先声明"))
    for st in ("slice", "f0", "feature"):
        o, u = ours[st], orig[st]
        ratio = o / u if u else float("inf")
        e = EXPECT[st]
        tag = "%s%s" % (e["dir"], "" if e["bound"] is None else " ≤%.0f×" % e["bound"])
        flag = ""
        if e["dir"] == "slower" and ratio < 1.0:
            flag = "  ⭐ 方向与声明相反(我方反而更快)——**这是好消息,但要更新声明**"
        if e["bound"] is not None and ratio > e["bound"]:
            flag = "  ⛔ 超出声明的上界"
            bad.append("%s 慢 %.2f× > 声明上界 %.0f×" % (st, ratio, e["bound"]))
        print("%-9s %10.2f %10.2f %8.2f×   %s%s" % (st, o, u, ratio, tag, flag))
    tot_o = sum(ours[s] for s in ("slice", "f0", "feature"))
    tot_u = sum(orig[s] for s in ("slice", "f0", "feature"))
    print("%-9s %10.2f %10.2f %8.2f×" % ("合计", tot_o, tot_u, tot_o / tot_u if tot_u else 0))
    print()
    print("★ 各段占我方总时间的比例(=【优化先动哪一段】的依据):")
    for st in ("slice", "f0", "feature"):
        print("   %-9s %5.1f%%" % (st, ours[st] / tot_o * 100))
    print()
    if bad:
        print("RESULT: RED —— " + " · ".join(bad))
        return EXIT_RED
    print("RESULT: ALL PASS —— 三段的方向与倍数都在跑之前声明的范围内")
    print("        ⛔ 注意判据的形状:**②号尺子不判「我方必须更快」** —— 它判"
          "「方向与声明相符 且 不超过声明的上界」。切片与特征两段**预先声明为我方更慢**,"
          "那是设计选择不是缺陷(理由见本文件头注)。")
    return EXIT_PASS


def selftest():
    fails = []

    def expect_unrunnable(label, fn):
        try:
            fn()
        except Unrunnable as e:
            print("  ok   %-40s -> %s" % (label, str(e)[:60]))
            return
        except Exception as e:  # noqa: BLE001
            fails.append("%s: 抛了 %s 而不是 Unrunnable" % (label, type(e).__name__))
            return
        fails.append("%s: 没抛" % label)

    base_o = {"slice": 1, "f0": 1, "feature": 1, "counts": {"0_gt_wavs": 4, "2a_f0": 4, "3_feature768": 4}}
    base_u = {"slice": 1, "f0": 1, "feature": 1, "counts": {"0_gt_wavs": 4, "2a_f0": 4, "3_feature768": 4},
              "_env": "selftest"}
    # ⑴ 一侧产物为 0 ⇒ 不可归因(⛔ 这正是空集假 PASS 的入口)
    z = json.loads(json.dumps(base_u)); z["counts"]["2a_f0"] = 0
    expect_unrunnable("一侧产物 0 件", lambda: judge(base_o, z))
    # ⑵ 两侧件数不等 ⇒ 不可归因
    z = json.loads(json.dumps(base_u)); z["counts"]["3_feature768"] = 3
    expect_unrunnable("两侧件数不等", lambda: judge(base_o, z))
    # ⑶ 超上界 ⇒ 红
    slow = json.loads(json.dumps(base_o)); slow["feature"] = 40
    if judge(slow, base_u) != EXIT_RED:
        fails.append("特征慢 40× 应当 RED")
    else:
        print("  ok   %-40s -> RED" % "超出声明上界")
    # ⑷ 在声明内 ⇒ 绿
    ok = json.loads(json.dumps(base_o)); ok["slice"] = 5; ok["feature"] = 10
    if judge(ok, base_u) != EXIT_PASS:
        fails.append("在声明范围内应当 PASS")
    else:
        print("  ok   %-40s -> PASS" % "在声明范围内")
    print()
    if fails:
        for f in fails:
            print("  FAIL %s" % f)
        print("gate_speed_prep_rvc 自检: FAILED(%d)" % len(fails))
        return EXIT_SELFTEST
    print("gate_speed_prep_rvc 自检: ALL OK")
    return EXIT_PASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-p", type=int, default=None, help="上游切片进程数(默认 = 上游自己的默认值)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    bad = [k for k in ("UTAI_DIAGNOSTICS", "CUDA_LAUNCH_BLOCKING") if os.environ.get(k)]
    if bad:
        print("⛔ preflight:环境里有 %s ⇒ 【闸的前提不满足】" % bad)
        return EXIT_UNRUNNABLE
    if not os.path.isdir(DATASET):
        print("⛔ preflight:缺数据集 %s" % DATASET)
        return EXIT_UNRUNNABLE
    tr = os.path.join(ARENA, "gate_speed_prep_transcript.txt")
    os.makedirs(ARENA, exist_ok=True)
    open(tr, "w", encoding="utf-8").close()
    print("[axis] 预处理 wall-clock · 我方 vs 上游 · **上游用它自己的默认并行度 + 它自己的 runtime**")
    print("[axis] ⇒ 这一把量的是【环境轴 + 代码轴】的合体(上游 py3.9/torch 2.0.0+cu118 + fairseq GPU;")
    print("       我方 py3.10/.venv + ContentVec CPU ORT)—— **而那正是用户实际经历的东西**。")
    print("[axis] ⛔ 本闸不复用任何现存 `*_orig.py` 驱动 —— 它们的并行度被内联掉了")
    print("[note] n=1,不配对:两段的预期差是【倍数级】,不是几个百分点 ⇒ 这个精度够用。")
    print("       ⚠ 若某一段实测落在 1.0-1.3× 区间,那一段的结论要按①号尺子那套配对法重测。")
    try:
        ours = run_ours(tr)
        orig = run_orig(tr, args.n_p)
        return judge(ours, orig)
    except Unrunnable as exc:
        print()
        print("RESULT: GATE-UNRUNNABLE ⇒ %s" % exc)
        print("        ⛔ 这**不是**「通过」,也**不是**「慢了」。")
        return EXIT_UNRUNNABLE


if __name__ == "__main__":
    sys.exit(main())
