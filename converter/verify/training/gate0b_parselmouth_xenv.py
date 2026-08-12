# -*- coding: utf-8 -*-
"""gate0b_parselmouth_xenv — parselmouth 跨版本 f0 一次性交叉定审（S40，红队 A13）。

背景：gate0 双侧同 venv 只隔离代码轴;librosa/torch 轴有 S38/S39 既有逐位证据,
唯 parselmouth(to_pitch_ac)是本链全新依赖、项目内零跨版本证据。本脚本在第二环境
(praat-parselmouth 旧版)对同一组切片跑【原版 wav2F0 真码】,与主 venv 结果按
S37 f0 定审口径比对(超 0.5Hz 帧数 / 清浊翻转数)。

⛔⛔ **S139 第一次真跑,而读数买回的第一条是【这条闸的前提写错了】。**
   本文件原来写着两个环境「**内嵌不同 Praat 内核**」—— **实测不是**:
   本机能用的两个环境是 RVC 整合包 runtime 的 **parselmouth 0.4.2** 与 training venv 的
   **0.4.7**,而**两者的 `PRAAT_VERSION` 都是 6.1.38**。
   ⇒ 这一跑跨的是 **wrapper 版本轴**(以及 py3.9/3.10 · numpy 1.23/1.26 · soundfile 0.11/0.14),
     **没有跨 Praat 内核轴** —— 而那正是原始立项理由里最硬的那一条。
   ⇒ 所以 `--compare` 现在**自己把两侧的 PRAAT_VERSION 打出来**,相同就响亮记一条零覆盖:
     绿是真的,但它证明的东西比这个文件名听起来的窄。
   ⚠ **这不是「结论无效」**,是「**别把它记成它并不具备的覆盖面**」(S122 那条纪律)。
   ▶ 真要跨 Praat 内核,得装一个 parselmouth **0.3.x**(Praat 6.0.x)或更新的 0.5.x,
     那是一件独立的事,不在 §F7 第一遍里。

⭐ S139 实测读数:**voiced frames 8094 | >0.5Hz: 0 | uv flips: 0 | max 0.0000Hz ⇒ PASS**
   (9 个切片,`TESTING\smoke_vocoder\ws\slices`;两侧 npz 留在 `TESTING\s139_f7\`)

两阶段(同一脚本,--dump 在任一环境产出 npz;主环境再 --compare):
  1) D:\\MyDev\\RVC\\RVC20240604Nvidia\\runtime\\python.exe  ... --dump out_old.npz   (0.4.2)
  2) training\\.venv\\Scripts\\python.exe                     ... --dump out_new.npz   (0.4.7)
  3) training\\.venv\\Scripts\\python.exe                     ... --compare out_old.npz out_new.npz

出口码:0 PASS · 1 判负 · 3 读数不可归因(两侧同一个 parselmouth / 文件对不上 / 帧数为 0)
"""
import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

ORIG = pathlib.Path(r"D:\MyDev\SingingVocoders")
SLICES = pathlib.Path(r"D:\MyDev\TESTING\smoke_vocoder\ws\slices")
HPARAMS = {"hop_size": 512, "audio_sample_rate": 44100, "f0_min": 65, "f0_max": 1100}


def dump(out_path):
    import importlib.util

    import parselmouth  # noqa: F401  (version report)
    import soundfile as sf

    # load the ORIGINAL wav2F0.py file directly (verbatim code) — a package
    # import would execute utils/__init__.py, which drags in the full
    # lightning stack this minimal second venv deliberately lacks
    spec = importlib.util.spec_from_file_location(
        "wav2F0_solo", str(ORIG / "utils" / "wav2F0.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    get_pitch = mod.get_pitch

    arrs = {}
    for wav in sorted(SLICES.glob("*.wav")):
        data, sr = sf.read(str(wav), dtype="float32")
        assert sr == 44100, wav
        length = (len(data) + HPARAMS["hop_size"] - 1) // HPARAMS["hop_size"]
        f0, uv = get_pitch("parselmouth", data, length=length, hparams=HPARAMS,
                           interp_uv=True)
        arrs[wav.stem + ".f0"] = f0
        arrs[wav.stem + ".uv"] = uv.astype(np.uint8)
    if not arrs:
        print("[UNRUNNABLE] %s 下一个 .wav 都没有 —— 这不是一次比较,是一次空转" % SLICES)
        sys.exit(3)
    # ⛔ S139:把**这一侧的身份**存进 npz。原来只 print 出来,而 print 是汇报不是判据
    #    (`gate0_guard.py:173-175` 的原话)—— 于是 `--compare` 结构上分不出
    #    「两个不同环境」和「同一个环境跑了两次」。
    arrs["_meta.parselmouth"] = np.array(str(parselmouth.VERSION))
    arrs["_meta.praat"] = np.array(str(parselmouth.PRAAT_VERSION))
    arrs["_meta.python"] = np.array(sys.version.split()[0])
    arrs["_meta.executable"] = np.array(sys.executable)
    np.savez(out_path, **arrs)
    print(f"dumped {(len(arrs)-4)//2} slices with parselmouth {parselmouth.VERSION} "
          f"(Praat {parselmouth.PRAAT_VERSION}, py {sys.version.split()[0]}) -> {out_path}")


def _meta(z, key, default="(未记录 —— 这份 npz 是 S139 之前 dump 的)"):
    k = "_meta." + key
    return str(z[k]) if k in z.files else default


def compare(a_path, b_path):
    za, zb = np.load(a_path), np.load(b_path)
    # ⛔ 身份先说,再比数 —— 「两侧到底是不是两个环境」是这条闸的全部前提
    pm_a, pm_b = _meta(za, "parselmouth"), _meta(zb, "parselmouth")
    pr_a, pr_b = _meta(za, "praat"), _meta(zb, "praat")
    print("A: parselmouth %s / Praat %s / py %s" % (pm_a, pr_a, _meta(za, "python")))
    print("B: parselmouth %s / Praat %s / py %s" % (pm_b, pr_b, _meta(zb, "python")))
    if pm_a == pm_b and not pm_a.startswith("("):
        # S134 复发榜第 9 条:**对照臂和被测臂指向同一个东西**。
        print("[UNRUNNABLE] 两侧是**同一个** parselmouth(%s)⇒ 这是一次自比对,不是跨版本定审。"
              % pm_a)
        sys.exit(3)
    if pr_a == pr_b and not pr_a.startswith("("):
        # ⛔ 不判红:结论仍然成立,只是**覆盖面比文件名听起来的窄**。响亮记账(S122 纪律)。
        print("[NO-COVERAGE] 两侧的 **Praat 内核是同一个**(%s)⇒ 这一跑跨的是 wrapper 版本轴"
              "(与 py/numpy/soundfile 轴),**没有跨 Praat 内核轴** —— 而那是本闸原始立项理由里"
              "最硬的一条。要跨它得装 parselmouth 0.3.x / 0.5.x,是独立的一件事。" % pr_a)
    if set(za.files) != set(zb.files):
        only_a = sorted(set(za.files) - set(zb.files))[:5]
        only_b = sorted(set(zb.files) - set(za.files))[:5]
        print("[UNRUNNABLE] 两份 npz 的键集不同:只在 A %s;只在 B %s" % (only_a, only_b))
        sys.exit(3)
    stems = sorted({k[:-3] for k in za.files if k.endswith(".f0")})
    if not stems:
        print("[UNRUNNABLE] 一个切片都没有 —— 空集不是通过")
        sys.exit(3)
    total = bad = flips = 0
    worst = 0.0
    for s in stems:
        fa, fb = za[s + ".f0"], zb[s + ".f0"]
        ua, ub = za[s + ".uv"], zb[s + ".uv"]
        voiced = (~ua.astype(bool)) & (~ub.astype(bool))
        d = np.abs(fa - fb)[voiced]
        total += int(voiced.sum())
        bad += int((d > 0.5).sum())
        flips += int((ua != ub).sum())
        if d.size:
            worst = max(worst, float(d.max()))
    print(f"voiced frames {total} | >0.5Hz: {bad} | uv flips: {flips} | max {worst:.4f}Hz")
    if total == 0:
        # ⛔ 空集守卫:0 帧时 bad==0 且 flips==0 ⇒ 上面那条 ok 恒真 ⇒ 一条零判据的 PASS。
        print("[UNRUNNABLE] 有声帧数是 0 —— 两侧都判成清音(或切片全空)⇒ 没有任何东西被比过")
        sys.exit(3)
    ok = bad == 0 and flips <= max(2, total // 1000)  # S37/S38 axis: ~0.1% edge frames
    print("=== gate0b_parselmouth_xenv:", "PASS" if ok else "FAIL",
          "=== (%d 切片 / %d 有声帧)" % (len(stems), total))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.dump:
        dump(args.dump)
    elif args.compare:
        compare(*args.compare)
