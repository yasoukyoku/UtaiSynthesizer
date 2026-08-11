"""gate0_diff step 4 — compare the --use_diff products, C layer (same 44k
input, both sides fp32 CPU, same aug seed/draw order).

Per slice:
  .vol.npy      expect BIT-EXACT (Volume_Extractor was bit-exact torch 2.0<->2.5
                in the S38 C5 gate)
  .mel.npy      nsf_hifigan-recipe mel — axis: torch 2.0 vs 2.5 stft/matmul
                (librosa mel filterbank 0.9.1 vs 0.11 measured bit-identical);
                PASS line: max_abs <= 1e-4 in ln-mel domain
  .aug_mel.npy  keyshift must match EXACTLY (same random draw — proves the RNG
                alignment). PASS = loud bins (ln-mel > -10) within 1e-5 and
                near-floor bins within 5e-4: the keyshift path runs a
                non-power-of-two FFT (n_fft*2^(k/12)) where the torch 2.0<->2.5
                kernel axis leaves slightly more fp noise, and ln amplifies it
                near the 1e-5 clamp floor (the S36-documented effect; measured
                S39: 12/21632 entries >1e-5, ALL at ln values < -10.1,
                linear-domain rel 1.1e-4) — a real code difference would show
                on loud bins, which must stay at fp-noise level
  .aug_vol.npy  same loudness shift -> expect bit-exact-or-fp-noise (<=1e-6)
Also verifies the (aug_mel, aug_vol) PAIR consistency file-by-file on our side
(both present — the pair-atomic rule).

Run (our venv):
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate0_diff_compare.py

★ S135(§F7 笔 2)修了本文件的**两条真·空判据**,并接入 gate0_guard:
  ⑴ `pair_missing` 恒为 0。它在 :79 检查的两个文件在 :58/:74 已经被 np.load 过 ——
     不存在就先抛 FileNotFoundError,那一段永远走不到。而文件头 :20-21 把它宣传成
     "pair-atomic 规则的验证"。S129 同族:一条从没被执行过的错误分支就是一条空判据。
     ⇒ 改成**加载之前**先查在场,并把"只有一半"与"两个都没有"分开报。
  ⑵ `stats.get("aug_mel_loud", 0.0)`:若没有任何 bin 的 ln-mel > -10(换素材/改 mel
     定义就会),这个键从不被写入,`.get` 返回 0.0 ⇒ **唯一有分辨力的那条判据无条件通过**,
     而头注 :17-18 恰恰写着 "a real code difference would show on loud bins"。
     ⇒ 改成统计响亮 bin 总数,为 0 就判**不可归因**,不许静默放行。
  ⑶ `assert wavs, "orig side empty — run gate0_diff_orig.py first"`:ORIG 里的 .wav 是
     **prepare** 摆进去的,与 gate0_diff_orig 跑没跑毫无关系 ⇒ 这条 assert 永远不会为
     它自己写的那个理由触发;真症状是 :49 一个裸 FileNotFoundError。
  ⑷ 第三条假绿通路(双侧陈货之外的那一档):**orig 全新 + ours 全量陈货**照样绿 ——
     aug_seed 固定 1234,两次独立运行抽出的 keyshift 数列逐个重合,keyshift 这条
     "随机流对齐的证明"在这一档完全失效。prepare 顺序处理 diff_orig/diff_ours,
     两者之间 Ctrl-C 就能造出这个状态。⇒ 靠 require_fresh 兜住。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate0_guard as G  # noqa: E402

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
ORIG = os.path.join(TESTING, "diff_orig", "gate")
OURS = os.path.join(TESTING, "diff_ours", "gate")

MEL_LINE = 1e-4
VOL_LINE = 0.0
AUG_VOL_LINE = 1e-6
AUG_MEL_LOUD_LINE = 1e-5   # bins with ln-mel > NEAR_FLOOR (the meaningful ones)
AUG_MEL_FLOOR_LINE = 5e-4  # near-clamp bins (ln amplification, see header)
NEAR_FLOOR = -10.0
MIN_SLICES = 30            # gate 固定 33 片


def main():
    t0 = G.read_t0("GATE0 DIFF")
    # 双侧都必须是本轮 staged + 本轮算的(见头注 ⑷)
    G.require_fresh("原版侧 diff_orig/gate", ORIG, [""], t0, MIN_SLICES * 4)
    G.require_fresh("我方 diff_ours/gate", OURS, [""], t0, MIN_SLICES * 4)

    wavs = sorted(n for n in os.listdir(ORIG) if n.endswith(".wav"))
    # ⛔ 原来这里是 `assert wavs, "orig side empty — run gate0_diff_orig.py first"`,
    # 而 .wav 是 prepare 摆的,与 orig 跑没跑无关 ⇒ 它永远不为那个理由触发。
    G.require_min("diff 切片件数(prepare 摆的)", len(wavs), MIN_SLICES)
    stats = {"vol": 0.0, "mel": 0.0, "aug_mel": 0.0, "aug_vol": 0.0}
    keyshift_mismatch = 0
    loud_bins = 0         # 响亮 bin 总数 —— 为 0 就说明 LOUD 那条判据零覆盖
    pair_broken = []      # 只有一半 —— 这是 pair-atomic 规则的真违反
    pair_absent = []      # 两个都没有 —— 这是"没跑到",不是规则违反
    for n in wavs:
        # ⛔ 在 np.load 之前查在场,否则这一段永远走不到(原缺陷)
        has_am = os.path.exists(os.path.join(OURS, n + ".aug_mel.npy"))
        has_av = os.path.exists(os.path.join(OURS, n + ".aug_vol.npy"))
        if has_am != has_av:
            pair_broken.append(n)
        elif not has_am:
            pair_absent.append(n)
        vol_a = np.load(os.path.join(ORIG, n + ".vol.npy"))
        vol_b = np.load(os.path.join(OURS, n + ".vol.npy"))
        stats["vol"] = max(stats["vol"], float(np.abs(vol_a - vol_b).max()))

        mel_a = np.load(os.path.join(ORIG, n + ".mel.npy"))
        mel_b = np.load(os.path.join(OURS, n + ".mel.npy"))
        assert mel_a.shape == mel_b.shape, (n, mel_a.shape, mel_b.shape)
        stats["mel"] = max(stats["mel"], float(np.abs(mel_a - mel_b).max()))

        am_a, ks_a = np.load(os.path.join(ORIG, n + ".aug_mel.npy"), allow_pickle=True)
        am_b, ks_b = np.load(os.path.join(OURS, n + ".aug_mel.npy"), allow_pickle=True)
        if float(ks_a) != float(ks_b):
            keyshift_mismatch += 1
            print(f"KEYSHIFT MISMATCH {n}: orig={ks_a} ours={ks_b}")
        am_a = np.array(am_a, dtype=np.float64)
        am_b = np.array(am_b, dtype=np.float64)
        assert am_a.shape == am_b.shape, (n, am_a.shape, am_b.shape)
        d = np.abs(am_a - am_b)
        stats["aug_mel"] = max(stats["aug_mel"], float(d.max()))
        loud = am_a > NEAR_FLOOR
        loud_bins += int(loud.sum())
        if loud.any():
            stats["aug_mel_loud"] = max(
                stats.get("aug_mel_loud", 0.0), float(d[loud].max())
            )

        av_a = np.load(os.path.join(ORIG, n + ".aug_vol.npy"))
        av_b = np.load(os.path.join(OURS, n + ".aug_vol.npy"))
        stats["aug_vol"] = max(stats["aug_vol"], float(np.abs(av_a - av_b).max()))

    print(f"files: {len(wavs)}")
    for k, v in stats.items():
        print(f"max_abs {k}: {v:.3e}")
    print(f"keyshift mismatches: {keyshift_mismatch}")
    print(f"aug pair broken(只有一半): {len(pair_broken)} {pair_broken[:5]}")
    print(f"aug pair absent(两个都没有): {len(pair_absent)} {pair_absent[:5]}")
    print(f"loud bins(ln-mel > {NEAR_FLOOR}): {loud_bins}")

    # ⛔ 没有响亮 bin ⇒ AUG_MEL_LOUD_LINE 这条判据本轮零覆盖,而它是这里唯一
    # "真代码差异会显形"的那条(头注 :17-18)。不许靠 .get 的默认值静默放行。
    if loud_bins == 0:
        raise G.GateUnrunnable(
            "aug_mel 一个响亮 bin 都没有(ln-mel 全 <= %s)⇒ AUG_MEL_LOUD_LINE 这条判据"
            "本轮零覆盖。头注说『真代码差异会显形在响亮 bin 上』——今天没有响亮 bin,"
            "所以这一关对那件事什么都没说。" % NEAR_FLOOR
        )
    if pair_absent:
        raise G.GateUnrunnable(
            "%d 个切片的 aug_mel/aug_vol 两个都不在(不是 pair-atomic 违反,是根本没跑到):%s"
            % (len(pair_absent), pair_absent[:5])
        )

    ok = (
        keyshift_mismatch == 0
        and not pair_broken
        and stats["vol"] <= VOL_LINE
        and stats["mel"] <= MEL_LINE
        and stats["aug_mel_loud"] <= AUG_MEL_LOUD_LINE
        and stats["aug_mel"] <= AUG_MEL_FLOOR_LINE
        and stats["aug_vol"] <= AUG_VOL_LINE
    )
    # ⛔ S135 二审:原来这里直接 raise SystemExit,**绕过了 G.finish** ⇒ 这条链上
    #    `note_uncovered` 是一条结构性空判据(记了账也没人读)。走 finish 才闭合。
    G.finish("GATE0 DIFF", [] if ok else ["diff 判据"],
             allow_uncovered="--allow-uncovered" in sys.argv)


if __name__ == "__main__":
    G.run("GATE0 DIFF", main)
