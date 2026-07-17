"""SoVITS 关卡0 对拍：utai_train SoVITS 预处理产物 vs 原版 so-vits-svc 4.1-Stable
脚本在「原版时代环境」（RVC runtime：librosa 0.9.1 / fairseq 0.12.2 / torch 2.0
CPU fp32）的实跑产物。

    training/.venv/Scripts/python.exe converter/verify/training/gate0_sovits_compare.py

三层判据（方法论同 RVC 关卡0，README 有全部读数）：
  A  端到端 vs 原版实跑（松阈值）。已知数值轴：
     - 44k 重采样 + 16k 重采样：librosa 0.9.1 (kaiser_best) vs 0.11 (soxr_hq)；
     - 特征：fairseq torch2.0 vs 我们 ContentVec onnx；
     - spec/vol/f0：torch 2.0 vs 2.5。
  C  定审（紧阈值，逐轴剥离）：
     C1 resample 代码轴（同 librosa 0.9.1 跑我们的链）→ gate0_sovits_c_resample.py
     C2 ContentVec 768/256：同一 16k 输入（原版侧 oracle 存盘）喂我们 onnx vs
        真 fairseq vencoder CPU fp32
     C3 f0：同一 44k 输入，原版 RMVPEF0Predictor(torch2.0) vs 我们 vendored
        (torch2.5)，双方 fp32 CPU —— 只剩 torch 版本轴
     C4 spec / C5 vol：同一 44k 输入，torch 版本轴
  S  我方 filelist / config / 检索库语义自检 + config 与原版逐键语义对拍。
"""
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
ORIG = os.path.join(TESTING, "sovits_orig")
ORIG_44K = os.path.join(ORIG, "dataset44k", "gate")
ORACLE = os.path.join(ORIG, "oracle")
OURS = os.path.join(TESTING, "sovits_ours")
OURS_44K = os.path.join(OURS, "dataset_44k", "gate")
AUX = os.path.join(REPO, "data", "models", "auxiliary")

failures = []


def check(label, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}: {detail}")
    if not ok:
        failures.append(label)


def wav_names(root):
    return {n for n in os.listdir(root) if n.endswith(".wav")}


def snr(orig, ours):
    err = orig.astype(np.float64) - ours.astype(np.float64)
    p_err = float((err**2).mean())
    if p_err == 0:
        return 999.0
    return 10 * np.log10(float((orig.astype(np.float64) ** 2).mean()) / p_err)


def a_wavs():
    from scipy.io import wavfile

    a, b = wav_names(ORIG_44K), wav_names(OURS_44K)
    check("A/44k 文件集合", a == b, f"orig={len(a)} ours={len(b)} 差集={sorted(a ^ b)[:6]}")
    if a != b:
        return
    snr_min = (1e99, "")
    for n in sorted(a):
        sr1, x = wavfile.read(os.path.join(ORIG_44K, n))
        sr2, y = wavfile.read(os.path.join(OURS_44K, n))
        assert sr1 == sr2 == 44100
        m = min(len(x), len(y))
        # length may differ by a few samples across resampler implementations
        if abs(len(x) - len(y)) > 32:
            check(f"A/44k 长度 {n}", False, f"{len(x)} vs {len(y)}")
            return
        s = snr(x[:m], y[:m])
        if s < snr_min[0]:
            snr_min = (s, n)
    check("A/44k 波形接近(librosa版本轴)", snr_min[0] > 30.0, f"min SNR={snr_min[0]:.1f} dB @ {snr_min[1]}")


def a_products():
    names = sorted(wav_names(ORIG_44K) & wav_names(OURS_44K))

    # .soft.pt — [1, 768, T]
    worst_cos, worst_max = (1.0, ""), (0.0, "")
    for n in names:
        co = torch.load(os.path.join(ORIG_44K, n + ".soft.pt"), map_location="cpu", weights_only=True).numpy()
        cu = torch.load(os.path.join(OURS_44K, n + ".soft.pt"), map_location="cpu", weights_only=True).numpy()
        tmin = min(co.shape[-1], cu.shape[-1])
        if abs(co.shape[-1] - cu.shape[-1]) > 2:
            check(f"A/soft 帧数 {n}", False, f"{co.shape} vs {cu.shape}")
            return
        x, y = co[0, :, :tmin].ravel(), cu[0, :, :tmin].ravel()
        cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        d = float(np.abs(x - y).max())
        if cos < worst_cos[0]:
            worst_cos = (cos, n)
        if d > worst_max[0]:
            worst_max = (d, n)
    check(
        "A/soft(ContentVec)",
        worst_cos[0] > 0.98 and worst_max[0] < 3.0,
        f"min_cos={worst_cos[0]:.9f} @ {worst_cos[1]}, max|Δ|={worst_max[0]:.3e} @ {worst_max[1]}",
    )

    # .f0.npy — (f0, uv)。so-vits 的后处理在清音区做线性插值填充：uv 边界帧随
    # 输入(-52dB 重采样轴)漂移一帧，就会把插值锚点差异扩散到整段清音区 —— 填充
    # 值对训练无实义（uv=0 标注在案），判据只看双方都判浊音的帧 + uv 翻转率；
    # 填充扩散只报告不判分。C3 已证同输入下 f0 代码 0 帧差异。
    tot = tot_v = bad_v = flips = bad_all = 0
    worst_v = 0.0
    for n in names:
        fo, uo = np.load(os.path.join(ORIG_44K, n + ".f0.npy"), allow_pickle=True)
        fu, uu = np.load(os.path.join(OURS_44K, n + ".f0.npy"), allow_pickle=True)
        m = min(len(fo), len(fu))
        fo = np.asarray(fo[:m], dtype=np.float64)
        fu = np.asarray(fu[:m], dtype=np.float64)
        uo = np.asarray(uo[:m])
        uu = np.asarray(uu[:m])
        tot += m
        d = np.abs(fo - fu)
        bad_all += int((d > 0.5).sum())
        flips += int((uo != uu).sum())
        vmask = (uo > 0.5) & (uu > 0.5)
        tot_v += int(vmask.sum())
        dv = d[vmask]
        bad_v += int((dv > 0.5).sum())
        if dv.size:
            worst_v = max(worst_v, float(dv.max()))
    check(
        "A/f0(Hz)+uv",
        tot_v > 0 and bad_v / tot_v <= 0.01 and flips / tot <= 0.005,
        f"浊帧={tot_v} |Δ|>0.5Hz={bad_v}({100.0*bad_v/max(1,tot_v):.4f}%) "
        f"uv翻转={flips}({100.0*flips/max(1,tot):.4f}%) max浊帧|Δ|={worst_v:.4f}Hz "
        f"[信息: 全帧含清音插值填充 {bad_all}/{tot}={100.0*bad_all/max(1,tot):.2f}%]",
    )

    # .spec.pt
    snr_min = (1e99, "")
    for n in names:
        so = torch.load(os.path.join(ORIG_44K, n.replace(".wav", ".spec.pt")), map_location="cpu", weights_only=True).numpy()
        su = torch.load(os.path.join(OURS_44K, n.replace(".wav", ".spec.pt")), map_location="cpu", weights_only=True).numpy()
        t = min(so.shape[-1], su.shape[-1])
        s = snr(so[:, :t], su[:, :t])
        if s < snr_min[0]:
            snr_min = (s, n)
    check("A/spec", snr_min[0] > 30.0, f"min SNR={snr_min[0]:.1f} dB @ {snr_min[1]}")

    # .vol.npy
    snr_min = (1e99, "")
    for n in names:
        vo = np.load(os.path.join(ORIG_44K, n + ".vol.npy"))
        vu = np.load(os.path.join(OURS_44K, n + ".vol.npy"))
        t = min(len(vo), len(vu))
        s = snr(vo[:t], vu[:t])
        if s < snr_min[0]:
            snr_min = (s, n)
    check("A/vol", snr_min[0] > 30.0, f"min SNR={snr_min[0]:.1f} dB @ {snr_min[1]}")


def c2_contentvec():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = {
        768: ort.InferenceSession(os.path.join(AUX, "contentvec_768l12.onnx"), so, providers=["CPUExecutionProvider"]),
        256: ort.InferenceSession(os.path.join(AUX, "contentvec_256l9.onnx"), so, providers=["CPUExecutionProvider"]),
    }
    for dim in (768, 256):
        worst_cos, worst_max, count = (1.0, ""), (0.0, ""), 0
        for n in sorted(os.listdir(ORACLE)):
            if not n.endswith(".wav16k.npy"):
                continue
            count += 1
            wav16k = np.load(os.path.join(ORACLE, n))
            ref = np.load(os.path.join(ORACLE, n.replace(".wav16k.npy", ".venc%d.npy" % dim)))  # [1, dim, T]
            feats = sess[dim].run(["features"], {"waveform": wav16k[None, :]})[0][0]  # [T, dim]
            x, y = ref[0].T.ravel(), feats.ravel()
            m = min(len(x), len(y))
            x, y = x[:m], y[:m]
            cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
            d = float(np.abs(x - y).max())
            if cos < worst_cos[0]:
                worst_cos = (cos, n)
            if d > worst_max[0]:
                worst_max = (d, n)
        check(
            f"C2 ContentVec {dim}（同16k输入，onnx vs 真fairseq fp32 CPU）",
            count > 0 and worst_cos[0] > 0.99999 and worst_max[0] < 2e-3,
            f"{count} 文件, min_cos={worst_cos[0]:.10f}, max|Δ|={worst_max[0]:.3e} @ {worst_max[1]}",
        )


def c3_f0():
    import librosa

    from utai_train.sovits.f0.RMVPEF0Predictor import RMVPEF0Predictor

    pred = RMVPEF0Predictor(
        hop_length=512, sampling_rate=44100, dtype=torch.float32, device="cpu",
        threshold=0.05,
        model_path=os.path.join(REPO, "data", "models", "training", "sovits", "rmvpe.pt"),
    )
    tot = bad = flips = 0
    worst = 0.0
    for n in sorted(wav_names(ORIG_44K)):
        fo, uo = np.load(os.path.join(ORIG_44K, n + ".f0.npy"), allow_pickle=True)
        wav, _ = librosa.load(os.path.join(ORIG_44K, n), sr=44100)
        fu, uu = pred.compute_f0_uv(wav)
        m = min(len(fo), len(fu))
        tot += m
        d = np.abs(np.asarray(fo[:m], dtype=np.float64) - np.asarray(fu[:m], dtype=np.float64))
        bad += int((d > 0.5).sum())
        worst = max(worst, float(d.max()) if m else 0.0)
        flips += int((np.asarray(uo[:m]) != np.asarray(uu[:m])).sum())
    check(
        "C3 f0 定审（同44k输入，双方fp32 CPU，torch 2.0 vs 2.5 轴）",
        tot > 0 and bad / tot <= 0.002 and flips / tot <= 0.002,
        f"帧={tot} |Δ|>0.5Hz={bad}({100.0*bad/max(1,tot):.4f}%) uv翻转={flips} max|Δ|={worst:.4f}Hz",
    )


def c4_c5_spec_vol():
    import librosa

    from utai_train.sovits.modules.mel_processing import spectrogram_torch
    from utai_train.sovits.utils import Volume_Extractor

    vex = Volume_Extractor(512)
    worst_spec, worst_vol = (0.0, ""), (0.0, "")
    for n in sorted(wav_names(ORIG_44K)):
        wav, _ = librosa.load(os.path.join(ORIG_44K, n), sr=44100)
        audio_norm = torch.FloatTensor(wav).unsqueeze(0)
        so = torch.load(os.path.join(ORIG_44K, n.replace(".wav", ".spec.pt")), map_location="cpu", weights_only=True)
        su = torch.squeeze(spectrogram_torch(audio_norm, 2048, 44100, 512, 2048, center=False), 0)
        rel = float((torch.abs(so - su) / (torch.abs(so) + 1e-3)).max())
        if rel > worst_spec[0]:
            worst_spec = (rel, n)
        vo = np.load(os.path.join(ORIG_44K, n + ".vol.npy"))
        vu = vex.extract(audio_norm).numpy()
        d = float(np.abs(vo - vu).max())
        if d > worst_vol[0]:
            worst_vol = (d, n)
    check("C4 spec 定审（同44k输入，torch 2.0 vs 2.5 轴）", worst_spec[0] < 1e-3, f"max_rel={worst_spec[0]:.3e} @ {worst_spec[1]}")
    check("C5 vol 定审（同44k输入）", worst_vol[0] < 1e-5, f"max|Δ|={worst_vol[0]:.3e} @ {worst_vol[1]}")


def s_layer():
    # filelists
    train = open(os.path.join(OURS, "filelists", "train.txt"), encoding="utf-8").read().splitlines()
    val = open(os.path.join(OURS, "filelists", "val.txt"), encoding="utf-8").read().splitlines()
    n_wavs = len(wav_names(OURS_44K))
    missing = [p for p in train + val if not os.path.exists(p)]
    check(
        "S/filelists",
        len(val) == 2 and len(train) + len(val) == n_wavs and not missing,
        f"train={len(train)} val={len(val)} wavs={n_wavs} 缺失={missing[:3]}",
    )

    # retrieval matrix
    mat = np.load(os.path.join(OURS, "cluster", "0.index_vectors.npy"))
    frames = 0
    for n in sorted(os.listdir(OURS_44K)):
        if n.endswith(".soft.pt"):
            frames += torch.load(os.path.join(OURS_44K, n), map_location="cpu", weights_only=True).shape[-1]
    check(
        "S/检索库",
        mat.dtype == np.float32 and mat.ndim == 2 and mat.shape[1] == 768 and (mat.shape[0] == frames or frames > 2e5),
        f"rows={mat.shape[0]} dim={mat.shape[1]} dtype={mat.dtype} (特征总帧={frames})",
    )

    # config semantic diff vs the original-generated config
    with open(os.path.join(ORIG, "config.json"), encoding="utf-8") as f:
        co = json.load(f)
    with open(os.path.join(OURS, "config.json"), encoding="utf-8") as f:
        cu = json.load(f)
    diffs = []
    ALLOWED_TRAIN = {"epochs", "batch_size", "all_in_mem"}  # deliberate gate params
    for k in set(co["train"]) | set(cu["train"]):
        if co["train"].get(k) != cu["train"].get(k) and k not in ALLOWED_TRAIN:
            diffs.append(f"train.{k}: {co['train'].get(k)} vs {cu['train'].get(k)}")
    for k in set(co["data"]) | set(cu["data"]):
        if k in ("training_files", "validation_files"):
            continue
        if co["data"].get(k) != cu["data"].get(k):
            diffs.append(f"data.{k}: {co['data'].get(k)} vs {cu['data'].get(k)}")
    for k in set(co["model"]) | set(cu["model"]):
        if co["model"].get(k) != cu["model"].get(k):
            diffs.append(f"model.{k}: {co['model'].get(k)} vs {cu['model'].get(k)}")
    if co.get("spk") != cu.get("spk"):
        diffs.append(f"spk: {co.get('spk')} vs {cu.get('spk')}")
    check("S/config 语义对拍", not diffs, "; ".join(diffs) if diffs else "model/data/spk 全等，train 仅关卡参数差异")


def main():
    print("== A: 端到端 vs 原版时代环境实跑（librosa/torch/提取器数值轴，松阈值）==")
    a_wavs()
    a_products()
    print("== C: 定审（紧阈值，逐轴剥离；C1 见 gate0_sovits_c_resample.py 输出）==")
    c2_contentvec()
    c3_f0()
    c4_c5_spec_vol()
    print("== S: 我方产物语义自检 ==")
    s_layer()
    print()
    if failures:
        print("GATE0 SOVITS: FAILURES:", failures)
        sys.exit(1)
    print("GATE0 SOVITS: ALL PASS (C1 需另跑 gate0_sovits_c_resample.py)")


if __name__ == "__main__":
    main()
