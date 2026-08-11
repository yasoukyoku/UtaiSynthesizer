# -*- coding: utf-8 -*-
"""gate0_vocoder — 声码器微调预处理对拍（S40）。

参照物 = 原版 openvpi/SingingVocoders 仓库代码真实执行（D:/MyDev/SingingVocoders
@4d0889c 的 process.wav2spec 直调——绕开 ProcessPoolExecutor 编排层，S39 教训）。
环境轴说明（README 登记）：SingingVocoders 无钉版依赖（librosa 仅 load/filters.mel、
torch 2.x 兼容、parselmouth 无版本断言）→ 双侧同 training/.venv，对拍面 = 纯代码轴；
parselmouth 的跨版本轴由 gate0b_parselmouth_xenv.py 一次性交叉定审补证。

判定：
  (a) 同一组 44.1k 切片，原版 wav2spec vs vendored process_sv.wav2spec 的 npz
      audio/mel/f0/uv/pe 全字段【逐位相等】（同库同版同代码，任何非零=移植错误）。
  (b) 48k 源正确性（红队 A1 实弹回归）：440Hz 正弦 48k 源经我们的 slice 阶段
      （统一重采样 44100）后，wav2spec 的 f0 中位数 ≈ 440Hz（±1Hz）。同时演示
      上游原始路径（48k 直喂）的 f0 会系统性偏低 ×44100/48000——证明偏离 #9 的
      必要性（该演示仅打印，不计 PASS/FAIL）。

⛔ S135(§F7 笔 2)登记的**覆盖边界**（别把这一关的绿读成它不具备的东西）：
  · (a) 段的两份 wav2spec 是逐字拷贝只改 import 行 ⇒ 它能抓的只有**移植抄写错误**；
    管线层 slice_dataset / process_slices / build_filelists 是我方带登记偏离的代码，
    (a) 段一行都没碰。
  · (b) 段是**自合成 48k 440Hz 正弦的自检**，不是与上游对拍；而且它的上游对照
    (:124-129) 明写「仅打印，不计 PASS/FAIL」，:125 的 `if ok_raw:` 还让上游那条
    失败时什么都不打印、什么都不失败。
  · 这两条链**结构上够不着 pool_dir**（extract_all 没有这个形参，本文件三个调用点
    也都不带）⇒ 它们的绿对 §F2⒝ 的池布局一个字也没说。
  · 退出码此前只有 0/1：`assert slices` 触发、`assert ok_a and ok_b` 触发、import 阶段炸，
    与「被测的东西不对」共用同一个非零 —— 违反 S129 铁律第一条。已改成走 gate0_guard。

用法：training/.venv/Scripts/python.exe converter/verify/training/gate0_vocoder.py
"""
import os
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate0_guard as G  # noqa: E402

APP = pathlib.Path(r"D:\MyDev\Utai_v2-dev")
ORIG = pathlib.Path(r"D:\MyDev\SingingVocoders")
SLICES = pathlib.Path(r"D:\MyDev\TESTING\smoke_vocoder\ws\slices")
OUT = pathlib.Path(r"D:\MyDev\TESTING\gate0_vocoder")
MIN_SLICES = 9      # 声码器冒烟固定切出 9 片

sys.path.insert(0, str(APP / "training"))
sys.path.insert(0, str(ORIG))  # original repo package roots (utils/, process.py)

import process as orig_process  # noqa: E402  (original repo)
from utai_train.vocoder import pipeline as vpipe  # noqa: E402
from utai_train.vocoder import process_sv  # noqa: E402

ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        ok = False


def mel_config():
    cfg = vpipe.build_train_config(
        {"total_steps": 100, "save_every_steps": 50, "batch_size": 2,
         "keep_ckpts": 2, "seed": 1234},
        "X:/unused.ckpt", "X:/unused",
    )
    return cfg


def main():
    # ⛔ S135 二审:本文件此前是唯一一个**不调 read_t0、也不 declare_frozen** 的 gate0
    #    脚本 ⇒ 它对「我读的输入是哪一批」完全无话可说,而它的输入(声码器冒烟切出来的
    #    9 片)gate0/gate1 里**没有任何脚本能重建**。⇒ 那是一份**故意冻结**的输入,
    #    要把它的件数与日期打进转录自证,而不是默不作声。
    t0 = G.read_t0("GATE0 VOCODER")
    G.declare_frozen(
        "输入 smoke_vocoder/ws/slices", str(SLICES), [""], MIN_SLICES,
        "声码器冒烟的产物;gate0/gate1 里没有任何脚本能重建它 ⇒ 故意冻结,不是陈货",
        suffixes=[".wav"],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = mel_config()

    # ---- (a) side-by-side wav2spec on the SAME slices ----
    slices = sorted(SLICES.glob("*.wav"))
    # ⛔ S135:原来这里是裸 assert —— 切片没了给的是 traceback + 非零,与「被测的东西
    # 不对」共用同一个退出码(S129 铁律第一条)。而 SLICES 是声码器冒烟的产物,
    # gate0/gate1 里没有任何脚本能重建它 ⇒ 它没了属于「闸跑不起来」。
    G.require_min("gate0(a) 输入切片", len(slices), MIN_SLICES,
                  "SLICES=%s(声码器冒烟的产物,gate 里没有脚本能重建)" % SLICES)
    print(f"=== gate0(a): {len(slices)} slices, orig vs vendored, bitwise ===")
    worst = {}
    for s in slices:
        a = OUT / (s.stem + ".orig.npz")
        b = OUT / (s.stem + ".ours.npz")
        ok_a, res_a = orig_process.wav2spec(cfg, s, a)
        ok_b, res_b = process_sv.wav2spec(cfg, s, b)
        assert ok_a and ok_b, f"wav2spec failed: {res_a} / {res_b}"
        za, zb = np.load(str(res_a)), np.load(str(res_b))
        for key in ("audio", "mel", "f0", "uv", "pe"):
            same = np.array_equal(za[key], zb[key])
            worst.setdefault(key, True)
            worst[key] &= same
            if not same:
                d = float(np.max(np.abs(za[key].astype(np.float64) - zb[key].astype(np.float64))))
                print(f"  MISMATCH {s.name}.{key}: max_abs {d:.3e}")
    for key, same in worst.items():
        check(same, f"npz field '{key}' bitwise-identical across all slices")

    # ---- (b) 48k source correctness through OUR slice stage ----
    print("=== gate0(b): 48k source f0 correctness (deviation #9 regression) ===")
    import soundfile as sf

    sr48 = 48000
    t = np.arange(int(sr48 * 8.0)) / sr48
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    src_dir = OUT / "src48k"
    src_dir.mkdir(exist_ok=True)
    sf.write(str(src_dir / "tone.wav"), tone, sr48, subtype="FLOAT")

    class _Rep:
        def stage(self, *a, **k):
            pass

    class _Stop:
        def check(self):
            pass

    sl_dir = OUT / "slices48k"
    if sl_dir.exists():
        import shutil

        shutil.rmtree(sl_dir)
    vpipe.slice_dataset(str(src_dir), str(sl_dir), "ffmpeg", _Rep(), _Stop())
    sl = sorted(pathlib.Path(sl_dir).glob("*.wav"))
    check(len(sl) >= 1, f"48k source sliced+resampled ({len(sl)} slices @44100)")
    info = sf.info(str(sl[0]))
    check(info.samplerate == 44100, f"slice sample rate == 44100 (got {info.samplerate})")

    okc, npz = process_sv.wav2spec(cfg, sl[0], OUT / "tone48.npz")
    assert okc, npz
    z = np.load(str(npz))
    f0 = z["f0"][z["f0"] > 0]
    med = float(np.median(f0))
    check(abs(med - 440.0) < 1.0, f"f0 median {med:.2f}Hz ≈ 440Hz through our 44.1k slice stage")

    # demonstration only: the upstream raw-48k path mislabels f0 by 44100/48000
    ok_raw, npz_raw = orig_process.wav2spec(cfg, src_dir / "tone.wav", OUT / "tone48_raw.npz")
    if ok_raw:
        zr = np.load(str(npz_raw))
        f0r = zr["f0"][zr["f0"] > 0]
        print(f"  (demo) upstream raw-48k path f0 median = {np.median(f0r):.2f}Hz "
              f"(expected mislabel ≈ {440 * 44100 / 48000:.2f}Hz — why deviation #9 exists)")

    # ⛔ 同上:走 G.finish 才让 note_uncovered / PASS-WITH-GAPS 这条路闭合
    print("\n=== gate0_vocoder:", "PASS" if ok else "FAIL", "===")
    G.finish("GATE0 VOCODER", [] if ok else ["vocoder 判据"],
             allow_uncovered="--allow-uncovered" in sys.argv)


if __name__ == "__main__":
    G.run("GATE0 VOCODER", main)
