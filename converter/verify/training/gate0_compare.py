"""关卡0 对拍：utai_train RVC 预处理产物 vs 原版 RVC 训练脚本产物。

    training/.venv/Scripts/python.exe converter/verify/training/gate0_compare.py

三层判据（调查过程与全部读数见本目录 README.md）：
  A  端到端 vs 原版 NVIDIA 整合包实跑产物（rvc_orig）。已知两条数值轴叠加：
     - 16k 重采样：原版 runtime librosa 0.9.1 (kaiser_best) vs 我们 0.11
       (soxr_hq)，同一行 librosa.resample 调用（代码同构），实测 ~39dB；
     - 特征：原版脚本永远把模型 .to("cuda")（cudnn TF32 卷积噪声 ~1e-2），
       f0 的 is_half 是字符串恒真值（NVIDIA 上事实恒 half）。
     → A 层用松阈值，验的是"整条链在真实原版面前没有结构性错误"。
  C  提取器定审（紧阈值，数值轴已剥离）：
     - f0: 原版 extract_f0_print 的 rmvpe-CPU-fp32 分支（rvc_B2_orig）vs 我们
       fp32 CPU —— 实测 0 帧超 0.5Hz、0 清浊翻转、max 0.24 mHz；
     - 特征: 真 fairseq 0.12.2 extract_features fp32 CPU（rvc_fairseq_fp32，由
       README 里的参照命令生成）vs 我们的 ContentVec onnx —— 实测全 51 文件
       max 7.7e-4 / min cos 1-1e-9。
  S  我方 filelist / index 产物语义自检。

⚠️ Windows 陷阱备忘：`CUDA_VISIBLE_DEVICES=`（空值）在 Windows 上等于**删除**
该变量（Windows 不存在空环境变量）→ 照样看见所有 GPU。要禁用 CUDA 用 `-1`。
"""
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
ORIG = os.path.join(TESTING, "rvc_orig")
OURS = os.path.join(TESTING, "rvc_ours")
F0_FP32_REF = os.path.join(TESTING, "rvc_B2_orig")       # 原版 rmvpe CPU fp32
FEAT_FP32_REF = os.path.join(TESTING, "rvc_fairseq_fp32")  # 真 fairseq fp32 CPU

failures = []


def check(label, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}: {detail}")
    if not ok:
        failures.append(label)


def names(root, sub):
    return {n for n in os.listdir(os.path.join(root, sub)) if not n.endswith(".spec.pt")}


def compare_wav_dir(sub, exact):
    a, b = names(ORIG, sub), names(OURS, sub)
    check(f"A/{sub} 文件集合", a == b, f"orig={len(a)} ours={len(b)} 差集={sorted(a ^ b)[:6]}")
    if a != b:
        return
    worst = (0.0, "")
    snr_min = (1e99, "")
    for n in sorted(a):
        sr1, x = wavfile.read(os.path.join(ORIG, sub, n))
        sr2, y = wavfile.read(os.path.join(OURS, sub, n))
        assert sr1 == sr2 and x.shape == y.shape, (n, sr1, sr2, x.shape, y.shape)
        err = x.astype(np.float64) - y.astype(np.float64)
        d = float(np.abs(err).max())
        if d > worst[0]:
            worst = (d, n)
        p_err = float((err**2).mean())
        snr = 10 * np.log10(float((x.astype(np.float64) ** 2).mean()) / p_err) if p_err > 0 else 999.0
        if snr < snr_min[0]:
            snr_min = (snr, n)
    if exact:
        check(f"A/{sub} 波形逐位", worst[0] == 0.0, f"max_abs_diff={worst[0]:.3e} @ {worst[1]}")
    else:
        check(
            f"A/{sub} 波形接近(librosa版本轴)",
            snr_min[0] > 30.0,
            f"min SNR={snr_min[0]:.1f} dB @ {snr_min[1]}, max_abs={worst[0]:.3e}",
        )


def compare_f0(tag, orig_root, frac_thr, flip_thr, coarse_thr, ours_root=None):
    ours_root = ours_root or OURS
    a, b = names(orig_root, "2b-f0nsf"), names(ours_root, "2b-f0nsf")
    check(f"{tag} f0 文件集合", a == b, f"orig={len(a)} ours={len(b)}")
    if a != b:
        return
    tot = bad = flips = 0
    mx = 0.0
    for n in sorted(a):
        x = np.load(os.path.join(orig_root, "2b-f0nsf", n))
        y = np.load(os.path.join(ours_root, "2b-f0nsf", n))
        assert x.shape == y.shape, (n, x.shape, y.shape)
        d = np.abs(x - y)
        tot += len(x)
        flips += int(((x == 0) != (y == 0)).sum())
        bad += int((d > 0.5).sum())
        mx = max(mx, float(d.max()))
    check(
        f"{tag} f0(Hz) 全局",
        (bad / tot) <= frac_thr and (flips / tot) <= flip_thr,
        f"帧={tot} |Δ|>0.5Hz={bad}({bad/tot:.5%}) 清浊翻转={flips} max|Δ|={mx:.4f}Hz",
    )
    # coarse 是 256 桶 mel 量化：亚 0.5Hz 的差落在桶边界也会翻相邻桶，
    # 失配率天然高于 f0 判据 → 阈值单列
    tot = bad = 0
    for n in sorted(a):
        x = np.load(os.path.join(orig_root, "2a_f0", n))
        y = np.load(os.path.join(ours_root, "2a_f0", n))
        tot += len(x)
        bad += int((x != y).sum())
    check(f"{tag} coarse 全局", (bad / tot) <= coarse_thr, f"帧={tot} 失配={bad}({bad/tot:.5%})")


def compare_feat(tag, ref_dir, max_thr, cos_thr):
    a, b = names(ref_dir, "3_feature768"), names(OURS, "3_feature768")
    check(f"{tag} 特征文件集合", a == b, f"ref={len(a)} ours={len(b)}")
    if a != b:
        return
    mx = 0.0
    cmin = 1.0
    worst = ""
    for n in sorted(a):
        x = np.load(os.path.join(ref_dir, "3_feature768", n)).astype(np.float64)
        y = np.load(os.path.join(OURS, "3_feature768", n)).astype(np.float64)
        assert x.shape == y.shape, (n, x.shape, y.shape)
        d = float(np.abs(x - y).max())
        if d > mx:
            mx = d
            worst = n
        cmin = min(cmin, float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y))))
    check(f"{tag} 特征", mx < max_thr and cmin > cos_thr, f"max|Δ|={mx:.3e} @ {worst}, min_cos={cmin:.9f}")


def selfcheck_filelist_index():
    fl = os.path.join(OURS, "filelist.txt")
    with open(fl, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l]
    bad = [l for l in lines if len(l.split("|")) != 5]
    check("S/filelist 字段数", not bad, f"{len(lines)} 行, 非5字段 {len(bad)}")
    missing = [p for l in lines for p in l.split("|")[:4] if not os.path.exists(p)]
    check("S/filelist 路径存在", not missing, f"缺失 {len(missing)}: {missing[:3]}")
    mute = [l for l in lines if "/mute/" in l]
    check("S/filelist mute 行", len(mute) == 2, f"mute 行数={len(mute)}")

    fea_dir = os.path.join(OURS, "3_feature768")
    frames = sum(np.load(os.path.join(fea_dir, n)).shape[0] for n in os.listdir(fea_dir))
    idx = np.load(os.path.join(OURS, "total_fea.npy"))
    ok = idx.dtype == np.float32 and (idx.shape[0] == frames or frames > 2e5)
    check("S/index 矩阵", ok, f"rows={idx.shape[0]} dtype={idx.dtype} (特征总帧={frames})")


def main():
    print("== A: 端到端 vs 原版整合包实跑（含 librosa/CUDA-TF32/half 数值轴，松阈值）==")
    compare_wav_dir("0_gt_wavs", exact=True)
    compare_wav_dir("1_16k_wavs", exact=False)
    compare_f0("A", ORIG, frac_thr=0.01, flip_thr=0.005, coarse_thr=0.03)
    # A 特征的输入音频本身就带 16k 重采样轴（他们 kaiser_best/我们 soxr_hq）→
    # cos≈0.985 是输入差异的传导，结构性错误会掉到 0.9x 以下；定审看 C
    compare_feat("A", ORIG, max_thr=3.0, cos_thr=0.98)

    print("== C: 提取器定审（fp32 CPU 参照，紧阈值）==")
    # f0 定审：双方都 fp32 CPU（我方产物 = rvc_B2_ours，由 README 命令生成）
    OURS_F0_FP32 = os.path.join(TESTING, "rvc_B2_ours")
    if os.path.isdir(os.path.join(F0_FP32_REF, "2b-f0nsf")) and os.path.isdir(
        os.path.join(OURS_F0_FP32, "2b-f0nsf")
    ):
        compare_f0(
            "C", F0_FP32_REF, frac_thr=0.0005, flip_thr=0.0001, coarse_thr=0.0005,
            ours_root=OURS_F0_FP32,
        )
    else:
        check("C f0 参照", False, f"缺 {F0_FP32_REF} 或 {OURS_F0_FP32}（生成命令见 README）")
    if os.path.isdir(os.path.join(FEAT_FP32_REF, "3_feature768")):
        compare_feat("C", FEAT_FP32_REF, max_thr=2e-3, cos_thr=0.99999)
    else:
        check("C 特征参照", False, f"缺 {FEAT_FP32_REF}（生成命令见 README）")

    selfcheck_filelist_index()
    print()
    if failures:
        print("GATE0: FAIL —", ", ".join(failures))
        sys.exit(1)
    print("GATE0: ALL PASS")


if __name__ == "__main__":
    main()
