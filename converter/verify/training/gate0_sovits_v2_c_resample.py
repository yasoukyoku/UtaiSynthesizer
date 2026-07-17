"""SoVITS 4.0-v2 关卡0 C1：resample 链**代码轴**定审 —— 在 RVC runtime
（librosa 0.9.1，与原版侧同一环境）里运行**我们的** _resample_chain
（trim_top_db=20 + loudnorm=True = v2 resample.py 的组合），与原版 v2
resample.py 的实跑产物逐位对拍。同环境同库版本 → 任何差异都只能来自
移植代码本身。期望逐位 0。

    D:\\MyDev\\RVC\\RVC20240604Nvidia\\runtime\\python.exe ^
        converter\\verify\\training\\gate0_sovits_v2_c_resample.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

UTAI = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(UTAI, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SLICES = os.path.join(TESTING, "sovits_slices", "gate")
ORIG_44K = os.path.join(TESTING, "sovits_v2_orig", "dataset44k", "gate")

import librosa
import numpy as np

from utai_train.sovits.preprocess import _resample_chain

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def main():
    names = sorted(n for n in os.listdir(SLICES) if n.endswith(".wav"))
    worst = (0, "")
    missing = []
    for n in names:
        # same loader the original used (librosa.load sr=None on the float32 slice)
        wav, sr = librosa.load(os.path.join(SLICES, n), sr=None)
        ours = _resample_chain(wav.astype(np.float32), int(sr), loudnorm=True, trim_top_db=20)
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
            sys.exit(1)
        d = int(np.abs(orig.astype(np.int32) - ours.astype(np.int32)).max())
        if d > worst[0]:
            worst = (d, n)
    ok = worst[0] == 0 and not missing
    print(
        "[%s] C1 v2 resample 代码轴（同 librosa 0.9.1, top_db=20）: %d 文件, max_abs_diff=%d @ %s, missing=%s"
        % ("PASS" if ok else "FAIL", len(names), worst[0], worst[1], missing[:5])
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
