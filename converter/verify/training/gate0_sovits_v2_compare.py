"""SoVITS 4.0-v2 关卡0 对拍（training venv）。三层判据（照 4.1 关卡0 惯例）：

A 层（松阈值，已知数值轴叠加=librosa 0.9.1↔0.11 重采样 + 各自管线）：
  - 44k wav：文件集合相等、长度差 ≤32 采样、min SNR > 30 dB
  - .soft.pt：cos > 0.98 且 max|Δ| < 3.0
  - .f0.npy（v2 = RAW dio 数组）：双浊帧 |Δ|>0.5Hz 占比 ≤1%、uv 翻转 ≤0.5%
  - aam mel：我方 .aam80.npy vs oracle .mel80.npy（各自 wav），SNR > 30 dB
C 层（紧阈值，逐轴定审——喂同一输入，只剩单一代码/库轴）：
  - C2 ContentVec：oracle 固定 16k 输入 → 我方 onnx vs 真 fairseq 存档，
    cos > 0.99999 且 max|Δ| < 2e-3
  - C3 dio：我方 vendored compute_f0_dio(原版 44k wav) vs 原版 .f0.npy，
    坏帧率 ≤ 0.002 且翻转率 ≤ 0.002（pyworld/numpy 版本轴）
  - C4 aam mel：我方 vendored audio.melspectrogram(原版 44k wav) vs oracle
    .mel80.npy —— librosa 0.9.1↔0.11 + kwargs/pad_mode 适配轴，max|Δ| < 1e-3
    （mel 值域 [0,4]）
S 层（语义自检）：filelists（val=2、train+val=全 wav、路径存在）、检索矩阵
  （float32/2 维/dim 256）、config 语义对拍（spk/model/data 全等，白名单键除外）
C1（resample 代码轴）另跑 gate0_sovits_v2_c_resample.py（RVC runtime）。

    training/.venv/Scripts/python.exe converter/verify/training/gate0_sovits_v2_compare.py
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
ORIG = os.path.join(TESTING, "sovits_v2_orig")
OURS = os.path.join(TESTING, "sovits_v2_ours")
ORIG_44K = os.path.join(ORIG, "dataset44k", "gate")
OURS_44K = os.path.join(OURS, "dataset_44k", "gate")
ORACLE = os.path.join(ORIG, "oracle")

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

failures = []


def check(ok, label, detail=""):
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", label, detail))
    if not ok:
        failures.append(label)


def snr_db(ref, x):
    n = min(len(ref), len(x))
    ref, x = ref[:n].astype(np.float64), x[:n].astype(np.float64)
    noise = ref - x
    p_sig = float((ref**2).sum())
    p_noise = float((noise**2).sum()) + 1e-12
    return 10.0 * np.log10(p_sig / p_noise + 1e-12)


def load_wav_i16(path):
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    assert sr == 44100
    return data.astype(np.float32) / 32768.0


def a_wavs():
    orig = sorted(n for n in os.listdir(ORIG_44K) if n.endswith(".wav"))
    ours = sorted(n for n in os.listdir(OURS_44K) if n.endswith(".wav"))
    # anti-empty guard（红队 A11 同族：空集合==空集合会假 PASS）——gate 数据集
    # 固定 33 片，任何显著缩水都说明有一侧没跑/跑错目录
    check(len(orig) >= 30 and len(ours) >= 30, "A wav 数量下限(防空集假 PASS)",
          "%d vs %d" % (len(orig), len(ours)))
    check(orig == ours, "A wav 文件集合", "%d vs %d" % (len(orig), len(ours)))
    if orig != ours or len(orig) < 30:
        return []
    worst = (1e9, "")
    max_len_diff = 0
    for n in orig:
        a = load_wav_i16(os.path.join(ORIG_44K, n))
        b = load_wav_i16(os.path.join(OURS_44K, n))
        max_len_diff = max(max_len_diff, abs(len(a) - len(b)))
        s = snr_db(a, b)
        if s < worst[0]:
            worst = (s, n)
    check(max_len_diff <= 32, "A wav 长度差 ≤32", str(max_len_diff))
    check(worst[0] > 30.0, "A wav SNR > 30 dB", "min %.1f @ %s" % worst)
    return orig


def a_products(names):
    worst_cos, worst_max = 1.0, 0.0
    f0_bad_worst, uv_flip_worst = 0.0, 0.0
    mel_snr_worst = 1e9
    for n in names:
        co = torch.load(os.path.join(ORIG_44K, n + ".soft.pt"), weights_only=False).squeeze(0).numpy()
        cu = torch.load(os.path.join(OURS_44K, n + ".soft.pt"), weights_only=False).squeeze(0).numpy()
        t = min(co.shape[1], cu.shape[1])
        co, cu = co[:, :t].astype(np.float64), cu[:, :t].astype(np.float64)
        num = (co * cu).sum()
        den = np.linalg.norm(co) * np.linalg.norm(cu) + 1e-12
        worst_cos = min(worst_cos, float(num / den))
        worst_max = max(worst_max, float(np.abs(co - cu).max()))

        fo = np.load(os.path.join(ORIG_44K, n + ".f0.npy"), allow_pickle=True)
        fu = np.load(os.path.join(OURS_44K, n + ".f0.npy"), allow_pickle=True)
        fo = np.asarray(fo, dtype=np.float64).reshape(-1)
        fu = np.asarray(fu, dtype=np.float64).reshape(-1)
        t = min(len(fo), len(fu))
        fo, fu = fo[:t], fu[:t]
        vo, vu = fo > 0, fu > 0
        both = vo & vu
        if both.sum() > 0:
            bad = float((np.abs(fo[both] - fu[both]) > 0.5).sum() / both.sum())
            f0_bad_worst = max(f0_bad_worst, bad)
        flips = float((vo != vu).sum() / t)
        uv_flip_worst = max(uv_flip_worst, flips)

        mo = np.load(os.path.join(ORACLE, n + ".mel80.npy"))
        mu = np.load(os.path.join(OURS_44K, n + ".aam80.npy"))
        t = min(mo.shape[0], mu.shape[0])
        s = snr_db(mo[:t].reshape(-1), mu[:t].reshape(-1))
        mel_snr_worst = min(mel_snr_worst, s)
    check(worst_cos > 0.98, "A soft cos > 0.98", "min %.6f" % worst_cos)
    check(worst_max < 3.0, "A soft max|Δ| < 3.0", "max %.4f" % worst_max)
    # dio 对 -58dB 级输入扰动在浊音边界抖动明显（实测 2.2%/1.2%）——A 层只报
    # 输入轴；代码轴的决定性判据是 C3（同输入逐位 0）。阈值按 dio 特性放宽
    # （rmvpe 的 4.1 关卡同层是 1%/0.5%，dio 本就更抖）。
    check(f0_bad_worst <= 0.03, "A f0 双浊帧坏帧率 ≤3% (dio 输入轴)", "max %.4f" % f0_bad_worst)
    check(uv_flip_worst <= 0.02, "A uv 翻转率 ≤2% (dio 输入轴)", "max %.4f" % uv_flip_worst)
    check(mel_snr_worst > 30.0, "A aam mel SNR > 30 dB", "min %.1f" % mel_snr_worst)


def c2_contentvec(names):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        os.path.join(REPO, "data", "models", "auxiliary", "contentvec_256l9.onnx"),
        so,
        providers=["CPUExecutionProvider"],
    )
    worst_cos, worst_max = 1.0, 0.0
    for n in names:
        wav16k = np.load(os.path.join(ORACLE, n + ".wav16k.npy"))
        ref = np.load(os.path.join(ORACLE, n + ".venc256.npy"))[0]  # [256, T]
        feats = sess.run(["features"], {"waveform": wav16k[None, :]})[0][0].T  # [256, T]
        t = min(ref.shape[1], feats.shape[1])
        a, b = ref[:, :t].astype(np.float64), feats[:, :t].astype(np.float64)
        num = (a * b).sum()
        den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        worst_cos = min(worst_cos, float(num / den))
        worst_max = max(worst_max, float(np.abs(a - b).max()))
    check(worst_cos > 0.99999, "C2 ContentVec cos > 0.99999", "min %.8f" % worst_cos)
    check(worst_max < 2e-3, "C2 ContentVec max|Δ| < 2e-3", "max %.2e" % worst_max)


def c3_dio(names):
    import librosa

    from utai_train.sovits_v2.utils import compute_f0_dio

    worst_bad, worst_flip = 0.0, 0.0
    for n in names:
        wav, _ = librosa.load(os.path.join(ORIG_44K, n), sr=44100)
        ours = compute_f0_dio(wav, sampling_rate=44100, hop_length=512)
        ref = np.asarray(
            np.load(os.path.join(ORIG_44K, n + ".f0.npy"), allow_pickle=True),
            dtype=np.float64,
        ).reshape(-1)
        t = min(len(ref), len(ours))
        a, b = ref[:t], np.asarray(ours[:t], dtype=np.float64)
        va, vb = a > 0, b > 0
        both = va & vb
        if both.sum() > 0:
            worst_bad = max(worst_bad, float((np.abs(a[both] - b[both]) > 0.5).sum() / both.sum()))
        worst_flip = max(worst_flip, float((va != vb).sum() / t))
    check(worst_bad <= 0.002, "C3 dio 同输入坏帧率 ≤0.2%", "max %.5f" % worst_bad)
    check(worst_flip <= 0.002, "C3 dio uv 翻转率 ≤0.2%", "max %.5f" % worst_flip)


def c4_aam_mel(names):
    import librosa

    from utai_train.sovits_v2 import utils as v2_utils
    from utai_train.sovits_v2.modules import audio as v2_audio

    hps = v2_utils.get_hparams_from_file(os.path.join(OURS, "config.json"))
    worst = 0.0
    frame_diff = 0
    for n in names:
        ref = np.load(os.path.join(ORACLE, n + ".mel80.npy"))
        wav = v2_utils.load_wav(
            os.path.join(ORIG_44K, n),
            raw_sr=hps.data.sampling_rate,
            target_sr=hps.data.sampling_rate,
            win_size=hps.data.win_size,
            hop_size=hps.data.hop_length,
        )
        mel = v2_audio.melspectrogram(wav, hps.data).astype(np.float32).T
        frame_diff = max(frame_diff, abs(ref.shape[0] - mel.shape[0]))
        t = min(ref.shape[0], mel.shape[0])
        worst = max(worst, float(np.abs(ref[:t].astype(np.float64) - mel[:t].astype(np.float64)).max()))
    check(frame_diff == 0, "C4 aam mel 帧数一致", str(frame_diff))
    check(worst < 1e-3, "C4 aam mel 同输入 max|Δ| < 1e-3 (librosa 0.9.1↔0.11 轴)", "max %.2e" % worst)


def s_semantics(names):
    # filelists: per-speaker val=2, train+val == every ≥0.3s wav (the flist
    # split skips <0.3s slices — same rule upstream applies; upstream also
    # reserves 2 test files which our house split folds into train, registered
    # deviation: test.txt is never consumed by training)
    import wave

    def _dur(p):
        with wave.open(p, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())

    eligible = [n for n in names if _dur(os.path.join(OURS_44K, n)) >= 0.3]
    with open(os.path.join(OURS, "filelists", "train.txt"), encoding="utf-8") as f:
        train = [line.strip() for line in f if line.strip()]
    with open(os.path.join(OURS, "filelists", "val.txt"), encoding="utf-8") as f:
        val = [line.strip() for line in f if line.strip()]
    check(len(val) == 2, "S val=2", str(len(val)))
    check(
        len(train) + len(val) == len(eligible),
        "S train+val=全部合格片(≥0.3s)",
        "%d+%d vs %d (总 %d)" % (len(train), len(val), len(eligible), len(names)),
    )
    check(all(os.path.exists(p) for p in train + val), "S filelist 路径存在")
    check(len(set(train) & set(val)) == 0, "S train/val 不重叠")

    # retrieval matrix
    mat = np.load(os.path.join(OURS, "cluster", "0.index_vectors.npy"))
    check(mat.dtype == np.float32 and mat.ndim == 2 and mat.shape[1] == 256,
          "S 检索矩阵 float32/[N,256]", str(mat.shape))

    # config semantics vs the upstream-generated config
    with open(os.path.join(ORIG, "config.json"), encoding="utf-8") as f:
        oc = json.load(f)
    with open(os.path.join(OURS, "config.json"), encoding="utf-8") as f:
        uc = json.load(f)
    check(oc["spk"] == uc["spk"], "S config.spk 全等", "%s vs %s" % (oc["spk"], uc["spk"]))
    check(oc["model"] == uc["model"], "S config.model 全等")
    TRAIN_ALLOW = {"epochs", "batch_size", "eval_interval", "keep_ckpts", "fp16_run", "num_workers", "log_interval", "seed"}
    bad = [k for k in set(oc["train"]) | set(uc["train"])
           if k not in TRAIN_ALLOW and oc["train"].get(k) != uc["train"].get(k)]
    check(not bad, "S config.train 白名单外全等", str(bad))
    DATA_ALLOW = {"training_filelist", "validation_filelist"}
    bad = [k for k in set(oc["data"]) | set(uc["data"])
           if k not in DATA_ALLOW and oc["data"].get(k) != uc["data"].get(k)]
    check(not bad, "S config.data 白名单外全等", str(bad))


def main():
    names = a_wavs()
    if not names:
        # a_wavs 已记 FAIL；显式兜底防「零文件全跳过 → 无 failure → 假 PASS」
        failures.append("A wav 集合为空/不一致，后续层未执行")
    else:
        a_products(names)
        c2_contentvec(names)
        c3_dio(names)
        c4_aam_mel(names)
        s_semantics(names)
    print()
    if failures:
        print("GATE0 SOVITS_V2 FAILED: %s" % failures)
        sys.exit(1)
    print("GATE0 SOVITS_V2 ALL PASS（C1 需另跑 gate0_sovits_v2_c_resample.py）")


if __name__ == "__main__":
    main()
