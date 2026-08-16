# -*- coding: utf-8 -*-
"""逐音配对 Δ —— 音域扩展这条线上唯一还站得住的波形读数形态。

## 为什么绝对读数一律作废(S147 对抗核验,别重推)

上一场造的两把尺子(ASR / EDR)量出「被抱怨的那几个音排在全曲前列」,看着很硬。
核验把它整条判死:**同样那几个音在一条 PSOLA 一次都没跑的渲染上仍占前 1-6 名**
(`UTAI_MG_SHIFT=0` ⇒ `apply_inverse` 提前 return),与生产排名 **Spearman +0.929**;
而且在那条零位移臂上 `corr(读数, |计划位移|) = +0.880` —— **救援计划优先挑的就是这把尺子
读数高的音**。⇒ 它们量的是「音高 × 时长 × 处理剂量」,不是「坏」。

⇒ **唯一可用的形态:同一个音、两条臂、相减。** 音高/音素/乐句/位置全部钉死,共线源全部消掉。

## 窗从哪来(S148 实测,别改回去)

⛔ **不许用 `notes[].frame0` 当窗起点。** 生产有 consonant preroll:音符自己的辅音落在
音符边界**之前**(首个 phone 的 frame0 比 notes[].frame0 早,中位 −3 帧 / min −7 帧 /
537 条非零)。实测窗外能量占比:

| 窗来源 | 最佳 offset | 窗外能量 | 长休止 p99 泄漏 |
|---|---|---|---|
| `notes[]` | −1 帧 | 0.523% | −45.1 dB |
| `phones[]` | **0 帧** | **0.0263%** | **−74.5 dB** |

⭐ 顺带证伪一条:「音频比谱面栅格早约 2 帧」是**用 `notes[]` 量出来的假象** —— 换 `phones[]`
之后最佳 offset 就是 0。那 2 帧整个是 preroll,**别给尺子加全局对齐补偿**。

⭐⭐ **但「起音」的锚点是音符边界,不是窗起点。** preroll 最长 7 帧(140 ms),而辅音爆发落在
preroll 的**后段**;把「前 30 ms」锚在窗起点会量到爆发**之前**那段安静 —— 实测 `[753]` 因此
从 +14.2 dB 变成 **−39.9 dB**。⇒ 起音区 = `[窗起点, 音符边界 + 30 ms]`。

⛔ `phones[].zero_frames` 清的是 **`note_hz`(喂给模型的 f0)**,不是音频:`[753]` 的 `/t/`
在那 5 个 "zeroed" 帧里峰值 **−1.00 dBFS**。别把它当静音跳过。

## 硬规矩

* 两条臂必须**同路渲染**。实测真机 `vocal.wav` 与 CLI 探针 `F_full_s0.wav`(同一首、都不带
  救援)**逐样本相关 0.0097**,而包络相关 0.9965 —— 跨路对拍量到的是路线差,且它长得像信号。
* 两条臂必须**等长**(这里 assert)。长度不同 = 时间栅格变了 ⇒ 全部读数作废。
* **报 Δ 必须同时报地板**,否则不知道多大才算数。地板登记在 `registry.json.note_delta`。
* ⛔ **~1 dB 级的 ΔHNR 已经被盲测证明听不见**(registry `known_blind_spots` 末条)。
  这里的读数同理只能当**描述**,承重的那一问仍然要过耳朵。

## 用法

    $py = "D:\\MyDev\\Utai_v2-dev\\training\\.venv\\Scripts\\python.exe"
    & $py note_delta.py --selfcheck                      # 先跑这个
    & $py note_delta.py --a <armA.wav> --b <armB.wav>    # 逐音 Δ = B − A
"""
import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "registry.json")

FPS = 50.0            # 谱面帧率
ATTACK_MS = 30.0      # 起音区在音符边界之后再延伸多少
BODY_SKIP_MS = 60.0   # 中段从音符边界之后这里开始
LOW_HZ = 1000.0       # low_ratio 的分界(S146f:至今唯一与耳朵同向的那条轴)
EPS = 1e-9

KEYS = ("onset_peak_dbfs", "attack_db", "rms_db", "low_ratio", "ripple_db")
EXIT_OK, EXIT_BAD, EXIT_UNRUNNABLE = 0, 1, 3


def die(msg):
    print(f"UNRUNNABLE: {msg}", file=sys.stderr)
    raise SystemExit(EXIT_UNRUNNABLE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry():
    return json.load(io.open(REGISTRY, encoding="utf-8"))


def load_lane(path):
    lane = json.load(io.open(path, encoding="utf-8"))
    return lane["notes"], lane["phones"], lane["total_frames"]


def note_windows(notes, phones, total_frames, n_samples):
    """{k: (win_a, onset_end, body_start, win_b)},样本下标。"""
    spf = n_samples / total_frames
    sps = spf * FPS
    by_evt = {}
    for p in phones:
        e = p["evt"]
        a, b = p["frame0"], p["frame0"] + p["dur"]
        if e in by_evt:
            by_evt[e] = (min(by_evt[e][0], a), max(by_evt[e][1], b))
        else:
            by_evt[e] = (a, b)
    out = {}
    for n in notes:
        if n["note_num"] <= 0 or n["k"] not in by_evt:
            continue
        fa, fb = by_evt[n["k"]]
        a = max(int(fa * spf), 0)
        b = min(int(fb * spf), n_samples)
        anchor = int(n["frame0"] * spf)
        oe = min(anchor + int(sps * ATTACK_MS / 1000.0), b)
        bs = min(anchor + int(sps * BODY_SKIP_MS / 1000.0), b)
        if b > a and oe > a and b > bs:
            out[n["k"]] = (a, oe, bs, b)
    return out, spf


def measure(x, sr, a, oe, bs, b):
    seg = x[a:b].astype(np.float64)
    onset = x[a:oe].astype(np.float64)
    body = x[bs:b].astype(np.float64)
    if len(onset) < 8 or len(body) < int(sr * 0.02):
        return None
    atk_peak = float(np.max(np.abs(onset)))
    body_rms = float(np.sqrt(np.mean(body ** 2)))
    rms = float(np.sqrt(np.mean(seg ** 2)))
    w = np.hanning(len(seg)) if len(seg) > 16 else np.ones(len(seg))
    sp = np.abs(np.fft.rfft(seg * w)) ** 2
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    tot = float(np.sum(sp)) + EPS
    low = float(np.sum(sp[fr < LOW_HZ]))
    hop = max(int(sr * 0.02), 1)
    ntr = len(seg) // hop
    ripple = (float(np.std(20 * np.log10(np.maximum(
        np.array([np.sqrt(np.mean(seg[i * hop:(i + 1) * hop] ** 2)) for i in range(ntr)]), EPS))))
        if ntr >= 3 else float("nan"))
    return {
        "onset_peak_dbfs": 20 * np.log10(max(atk_peak, EPS)),
        "attack_db": 20 * np.log10(max(atk_peak, EPS) / max(body_rms, EPS)),
        "rms_db": 20 * np.log10(max(rms, EPS)),
        "low_ratio": low / tot,
        "ripple_db": ripple,
    }


def _mono(p):
    if not os.path.exists(p):
        die(f"缺素材: {p}")
    x, sr = sf.read(p, dtype="float32")
    return (x[:, 0] if x.ndim > 1 else x), sr


def paired(path_a, path_b, lane_path, note_filter=None, base_offset=0, full_samples=None):
    """逐音 Δ = B − A。切片臂用 `base_offset`(样本)对回整曲坐标。"""
    xa, sa = _mono(path_a)
    xb, sb = _mono(path_b)
    if sa != sb:
        die(f"采样率不同 {sa} vs {sb} —— 读数不构成比较")
    if len(xa) != len(xb):
        die(f"长度不同 {len(xa)} vs {len(xb)} —— 时间栅格变了,所有读数作废")
    notes, phones, tf = load_lane(lane_path)
    wins, spf = note_windows(notes, phones, tf, full_samples or len(xa))
    rows = []
    for k, (a, oe, bs, b) in sorted(wins.items()):
        if note_filter and not note_filter(k):
            continue
        sh = (a - base_offset, oe - base_offset, bs - base_offset, b - base_offset)
        if sh[0] < 0 or sh[3] > len(xa):
            continue
        ma, mb = measure(xa, sa, *sh), measure(xb, sb, *sh)
        if ma is None or mb is None:
            continue
        rows.append({"k": k,
                     **{f"d_{q}": mb[q] - ma[q] for q in KEYS},
                     **{f"a_{q}": ma[q] for q in KEYS}})
    return rows, spf


def summarize(rows, label):
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"{'尺子':<18}{'|Δ| p50':>10}{'|Δ| p90':>10}{'|Δ| max':>10}{'Δ 中位':>10}")
    for q in KEYS:
        d = np.array([r[f"d_{q}"] for r in rows], dtype=float)
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        ad = np.abs(d)
        print(f"{q:<18}{np.percentile(ad,50):10.4f}{np.percentile(ad,90):10.4f}"
              f"{ad.max():10.4f}{np.median(d):10.4f}")


# ────────────────────────────── selfcheck ──────────────────────────────

def selfcheck():
    """三档,与 `selfcheck.py` 同一套出口码(0 通过 / 1 读数不符 / 3 跑不起来)。"""
    reg = registry()
    nd = reg.get("note_delta")
    if nd is None:
        die("registry.json 里没有 note_delta 块 —— 判据的登记值不存在,读数不构成比较")
    root = os.environ.get(nd["dir_env"], nd["dir_default"])
    bad = 0

    print("① 夹具指纹")
    for name, want in nd["sha256"].items():
        p = os.path.join(root, name)
        if not os.path.exists(p):
            die(f"缺夹具 {p}")
        got = sha256(p)
        if got != want:
            die(f"夹具 {name} 指纹不符\n  登记 {want}\n  实际 {got}\n"
                f"  ⛔ 换过夹具的读数与登记值【不构成比较】(README 硬规矩 6)")
        print(f"   {name}: OK")
    lane = os.path.join(root, nd["lane"])
    if not os.path.exists(lane):
        die(f"缺泳道 {lane}")

    print("\n② 口径闸(尺子的旋钮还是不是登记的那一套)")
    live = {"ATTACK_MS": ATTACK_MS, "BODY_SKIP_MS": BODY_SKIP_MS,
            "LOW_HZ": LOW_HZ, "FPS": FPS}
    for k, want in nd["caliber"].items():
        if live[k] != want:
            print(f"   ⛔ {k}: 登记 {want} 实际 {live[k]}")
            bad += 1
        else:
            print(f"   {k} = {want}: OK")

    print("\n③ 阴性对照:同一个文件与自己(工装不许有随机性)")
    base = os.path.join(root, nd["baseline_arm"])
    rows, _ = paired(base, base, lane)
    worst = max(abs(r[f"d_{q}"]) for r in rows for q in KEYS if np.isfinite(r[f"d_{q}"]))
    if worst != 0.0:
        print(f"   ⛔ 自比不为零: {worst:.3e}")
        bad += 1
    else:
        print(f"   {len(rows)} 个音全部 Δ == 0: OK")

    print("\n④ 地板:整曲同配置两跑的逐音 |Δ|(登记值是实测出来的,不是推的)")
    fl = nd["floor_same_config"]
    rows, _ = paired(os.path.join(root, fl["arm_a"]), os.path.join(root, fl["arm_b"]), lane)
    if len(rows) != fl["n"]:
        print(f"   ⛔ 命中音符数 {len(rows)},登记 {fl['n']}")
        bad += 1
    for q, want in fl["percentiles"].items():
        d = np.abs(np.array([r[f"d_{q}"] for r in rows], dtype=float))
        d = d[np.isfinite(d)]
        # ⛔ 只钉 p50/p90:max 是重尾的(几乎全部来自单个不稳定的短音 [512]),
        #    拿它当判据等于把「模型偶发抖动」写进闸里。
        line, hit = [], True
        for tag in ("p50", "p90"):
            got = float(np.percentile(d, int(tag[1:])))
            w = want[tag]
            # 容差:登记值 ×2 + 一个绝对地板(⛔ 不许拿被测常量算期望值,这里比的是实测字面量)
            good = got <= w * 2.0 + 0.005
            hit &= good
            line.append(f"{tag} {got:.4f}(登记 {w:.4f}){'' if good else ' ⛔'}")
        if not hit:
            bad += 1
        print(f"   {q:<18} " + "  ".join(line))

    print("\n⑤ 分离度(【正】判据 —— ③④ 都挡不住一把「永远返回常数」的尺子)")
    sp = nd["separation"]
    plan = json.load(io.open(os.path.join(root, sp["plan_json"]), encoding="utf-8"))["groups"]
    rescued = set()
    for a, b, _s in plan:
        rescued.update(range(a, b + 1))
    rows, _ = paired(os.path.join(root, sp["ref"]), os.path.join(root, sp["cand"]), lane)
    r_in = np.abs([r["d_rms_db"] for r in rows if r["k"] in rescued])
    r_out = np.abs([r["d_rms_db"] for r in rows if r["k"] not in rescued])
    p_in, p_out = float(np.percentile(r_in, 50)), float(np.percentile(r_out, 50))
    ratio = p_in / max(p_out, 1e-9)
    g = sp["gate"]
    for name, got, cmpf, want in (
        ("被救援音 |Δrms| p50", p_in, lambda a, b: a >= b, g["rescued_rms_p50_min"]),
        ("未救援音 |Δrms| p50", p_out, lambda a, b: a <= b, g["unrescued_rms_p50_max"]),
        ("分离比", ratio, lambda a, b: a >= b, g["ratio_min"]),
    ):
        good = cmpf(got, want)
        bad += 0 if good else 1
        print(f"   {name:<22} {got:10.4f}  (闸 {want})  {'OK' if good else '⛔'}")
    print(f"   n: 被救 {len(r_in)} / 未救 {len(r_out)}  (登记 {sp['measured']['rescued_n']} / "
          f"{sp['measured']['unrescued_n']})")

    print("\n⑥ 窗规则(只看单条臂 —— ③④⑤ 都是两侧同窗的比较,窗错位会相消)")
    wr = nd["window_rule_gate"]
    x, sr_a = _mono(os.path.join(root, wr["arm"]))
    notes, phones, tf = load_lane(lane)
    wins, _ = note_windows(notes, phones, tf, len(x))
    cov = np.zeros(len(x), dtype=bool)
    for (a, _oe, _bs, b) in wins.values():
        cov[a:b] = True
    e = x.astype(np.float64) ** 2
    pct = float(e[~cov].sum() / e.sum()) * 100.0
    good = pct <= wr["gate"]["out_of_window_energy_pct_max"]
    bad += 0 if good else 1
    print(f"   窗外能量占比 {pct:.4f}%  (闸 ≤{wr['gate']['out_of_window_energy_pct_max']}%,"
          f"实测 phones[] {wr['measured']['out_of_window_energy_pct']}% / "
          f"错误的 notes[] {wr['measured']['_alternative_notes_windows_pct']}%)  {'OK' if good else '⛔'}")

    print("\n⑦ 起音锚点(钉一个已知读数 —— 锚错了它会从 +14 变成 −40)")
    ap = nd["anchor_pin"]
    k = ap["gate"]["note"]
    if k not in wins:
        print(f"   ⛔ 夹具里没有 note {k}")
        bad += 1
    else:
        m = measure(x, sr_a, *wins[k])
        got = m["attack_db"]
        good = ap["gate"]["attack_db_min"] <= got <= ap["gate"]["attack_db_max"]
        bad += 0 if good else 1
        print(f"   [{k}] attack_db {got:8.3f}  (闸 [{ap['gate']['attack_db_min']},"
              f"{ap['gate']['attack_db_max']}],实测 {ap['measured']['attack_db']},"
              f"锚在窗起点会读 {ap['measured']['_if_anchored_at_window_start']})  {'OK' if good else '⛔'}")

    print()
    if bad:
        print(f"FAIL — {bad} 条读数不符")
        return EXIT_BAD
    print("ALL PASS")
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--lane")
    ap.add_argument("--notes", help="只看这些音,形如 753-762 或 753,760")
    ap.add_argument("--json", help="把逐音结果写到这个文件")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not (args.a and args.b):
        die("要么 --selfcheck,要么 --a <armA.wav> --b <armB.wav>")
    reg = registry()
    nd = reg["note_delta"]
    root = os.environ.get(nd["dir_env"], nd["dir_default"])
    lane = args.lane or os.path.join(root, nd["lane"])
    flt = None
    if args.notes:
        if "-" in args.notes:
            lo, hi = (int(v) for v in args.notes.split("-"))
            flt = lambda k: lo <= k <= hi  # noqa: E731
        else:
            keep = {int(v) for v in args.notes.split(",")}
            flt = lambda k: k in keep      # noqa: E731
    rows, _ = paired(args.a, args.b, lane, note_filter=flt)
    summarize(rows, f"{os.path.basename(args.b)} − {os.path.basename(args.a)}")
    if args.json:
        json.dump(rows, io.open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n逐音结果 -> {args.json}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
