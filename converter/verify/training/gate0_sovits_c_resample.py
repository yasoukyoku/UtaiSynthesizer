"""SoVITS 关卡0 C1：resample 链**代码轴**定审 —— 在 RVC runtime（librosa 0.9.1，
与原版侧同一环境）里运行**我们的** _resample_chain，与原版 resample.py 的实跑
产物逐位对拍。同环境同库版本 → 任何差异都只能来自移植代码本身。期望逐位 0。

    D:\\MyDev\\RVC\\RVC20240604Nvidia\\runtime\\python.exe ^
        converter\\verify\\training\\gate0_sovits_c_resample.py

⛔ S135(§F7 笔 2):本文件此前在切片目录为空时会打印 `[PASS] C1 ... 0 文件` 并
   **exit 0** —— names=[] ⇒ 循环 0 次 ⇒ worst 停在初值 (0,"") ⇒ ok 恒真。
   于是「为了强制重建把切片清了」这个动作恰好会把最硬的一条判据(resample 移植逐位 0)
   换成一句空话,而且退出码是绿的。现在接 gate0_guard:件数下限 + 产物必须是本轮的。
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate0_guard as G  # noqa: E402

UTAI = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(UTAI, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SLICES = os.path.join(TESTING, "sovits_slices", "gate")
ORIG_44K = os.path.join(TESTING, "sovits_orig", "dataset44k", "gate")

import librosa
import numpy as np

from utai_train.sovits.preprocess import _resample_chain

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


MIN_SLICES = 30      # gate 固定 33 片；缩水 = 有一侧没跑，不是"被测的东西变了"


def main():
    t0 = G.read_t0("GATE0 SOVITS C1")
    G.require_fresh("C1 输入 sovits_slices/gate", SLICES, [""], t0, MIN_SLICES,
                    suffixes=[".wav"])
    G.require_fresh("C1 参照 sovits_orig/dataset44k/gate", ORIG_44K, [""], t0, MIN_SLICES,
                    suffixes=[".wav"])
    names = sorted(n for n in os.listdir(SLICES) if n.endswith(".wav"))
    G.require_min("C1 切片件数", len(names), MIN_SLICES)
    worst = (0, "")
    missing = []
    for n in names:
        # same loader the original used (librosa.load sr=None on the float32 slice)
        wav, sr = librosa.load(os.path.join(SLICES, n), sr=None)
        ours = _resample_chain(wav.astype(np.float32), int(sr), loudnorm=True)
        orig_path = os.path.join(ORIG_44K, n)
        if ours is None:
            # trim swallowed the slice — original would have written silence/empty
            if os.path.exists(orig_path):
                missing.append(n)
            continue
        from scipy.io import wavfile

        sr2, orig = wavfile.read(orig_path)
        assert sr2 == 44100
        if orig.shape != ours.shape:
            print("[FAIL] C1 %s shape %s vs %s" % (n, orig.shape, ours.shape))
            sys.exit(G.EXIT_RED)
        d = int(np.abs(orig.astype(np.int32) - ours.astype(np.int32)).max())
        if d > worst[0]:
            worst = (d, n)
    ok = worst[0] == 0 and not missing
    print(
        "[%s] C1 resample 代码轴（同 librosa 0.9.1）: %d 文件, max_abs_diff=%d @ %s, missing=%s"
        % ("PASS" if ok else "FAIL", len(names), worst[0], worst[1], missing[:5])
    )
    sys.exit(G.EXIT_PASS if ok else G.EXIT_RED)


if __name__ == "__main__":
    G.run("GATE0 SOVITS C1", main)
