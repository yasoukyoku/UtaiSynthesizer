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

★ S135(§F7 笔 2)接入 `gate0_guard`：本文件此前**两侧目录都为空时会打印
  `max|Δ|=0.000e+00, min_cos=1.000000000` 并 ALL PASS**（集合 set()==set() 通过、
  循环 0 次、worst/cmin 停在初值）⇒ 删目录 = 正确地红，清空目录 = 假 PASS，
  两种清法后果相反。而且全文零个 mtime/时间戳检查 ⇒ 它分不出读到的是今天的产物
  还是七月的。现在：我方侧必须 `require_fresh`，冻结参照必须 `declare_frozen`
  并把日期打进转录，两侧都冻结的判据必须 `note_uncovered` 响亮记账。
  ⛔ 需要环境变量 `GATE0_T0`（本轮起始 epoch 秒，由 run_gate0_chain.py 设）。
"""
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate0_guard as G  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
ORIG = os.path.join(TESTING, "rvc_orig")
OURS = os.path.join(TESTING, "rvc_ours")
F0_FP32_REF = os.path.join(TESTING, "rvc_B2_orig")       # 原版 rmvpe CPU fp32
FEAT_FP32_REF = os.path.join(TESTING, "rvc_fairseq_fp32")  # 真 fairseq fp32 CPU
OURS_F0_FP32 = os.path.join(TESTING, "rvc_B2_ours")      # C 层 f0 的我方 fp32 CPU 产物

# gate 数据集固定切出 51 片。任何显著缩水都说明有一侧没跑 / 跑错目录，
# 而不是"被测的东西变了"——所以它走 UNRUNNABLE 不走 FAIL。
MIN_SLICES = 45
OURS_PRODUCT_DIRS = ["0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"]

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
    # 防空集(S68 f599e76 已判 [major],补丁形状见 gate0_sovits_v2_compare.py:72-75)：
    # 没有这两行，两侧都空时下面每一条都会 PASS 而什么都没比。
    G.require_min(f"A/{sub} orig 件数", len(a), MIN_SLICES)
    G.require_min(f"A/{sub} ours 件数", len(b), MIN_SLICES)
    check(f"A/{sub} 文件集合", a == b, f"orig={len(a)} ours={len(b)} 差集={sorted(a ^ b)[:6]}")
    if a != b:
        return
    worst = (0.0, "")
    snr_min = (1e99, "")
    for n in sorted(a):
        sr1, x = wavfile.read(os.path.join(ORIG, sub, n))
        sr2, y = wavfile.read(os.path.join(OURS, sub, n))
        # 形状/采样率不一致是**被测的东西不对**（真红），不是闸跑不起来
        # ⇒ 走 check 拿一条带标签的 FAIL，别再甩 traceback（S129 铁律）
        if sr1 != sr2 or x.shape != y.shape:
            check(f"A/{sub} 形状一致", False, f"{n}: sr {sr1} vs {sr2}, shape {x.shape} vs {y.shape}")
            return
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
    G.require_min(f"{tag} f0 orig 件数", len(a), MIN_SLICES)
    G.require_min(f"{tag} f0 ours 件数", len(b), MIN_SLICES)
    check(f"{tag} f0 文件集合", a == b, f"orig={len(a)} ours={len(b)}")
    if a != b:
        return
    tot = bad = flips = 0
    mx = 0.0
    for n in sorted(a):
        x = np.load(os.path.join(orig_root, "2b-f0nsf", n))
        y = np.load(os.path.join(ours_root, "2b-f0nsf", n))
        if x.shape != y.shape:
            check(f"{tag} f0 形状一致", False, f"{n}: {x.shape} vs {y.shape}")
            return
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
    G.require_min(f"{tag} 特征 ref 件数", len(a), MIN_SLICES)
    G.require_min(f"{tag} 特征 ours 件数", len(b), MIN_SLICES)
    check(f"{tag} 特征文件集合", a == b, f"ref={len(a)} ours={len(b)}")
    if a != b:
        return
    mx = 0.0
    cmin = 1.0
    worst = ""
    for n in sorted(a):
        x = np.load(os.path.join(ref_dir, "3_feature768", n)).astype(np.float64)
        y = np.load(os.path.join(OURS, "3_feature768", n)).astype(np.float64)
        if x.shape != y.shape:
            check(f"{tag} 特征形状一致", False, f"{n}: {x.shape} vs {y.shape}")
            return
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
    t0 = G.read_t0("GATE0 RVC")

    print("== 0: 产物身份（本轮算的 vs 冻结参照）==")
    # 我方侧：必须是**本轮**算出来的。⛔ 别用 stage 的 done 计数——三处 reporter.stage
    # 全在 skip 的 continue 之前（extract_f0.py:86 vs :91，:99 还无条件报满），
    # 全跳过时它照样走满。唯一可用的是产物 mtime。
    for sub in OURS_PRODUCT_DIRS:
        G.require_fresh(f"我方 rvc_ours/{sub}", OURS, [sub], t0, MIN_SLICES)
    G.require_fresh(
        "我方 rvc_ours/{filelist.txt,total_fea.npy}", OURS, [""], t0, 2,
        suffixes=["filelist.txt", "total_fea.npy"],
    )
    # 原版侧：**故意**冻结，不是陈货。它是「不变输入(gate_dataset，2026-07-05 起未动)
    # × 不变上游代码(RVC 整合包 2024-06-04)」的函数；而且 rvc_orig/3_feature768 是
    # CUDA 跑的（它自己的 extract_f0_feature.log 写着 'move model to cuda'，本文件
    # 头注也记了 cudnn TF32 噪声 ~1e-2）⇒ **重算反而会换掉参照物**，README 的 A 层
    # 读数从此对不上原件。
    # ⛔ `expect_sha` 是 S135 二审补的(M11):在那之前 `declare_frozen` **只判件数**、
    # 然后把 mtime **打印**出来 —— 而打印是汇报不是判据。参照侧被重跑、被从别的备份还原、
    # 被指到另一个目录,三种情况一律照常通过,唯一差别是那一行的日期变了。
    # 下面三串是 2026-08-11 登记的值。⛔ 它们变了要**先弄清为什么**,再决定改不改这里 ——
    # 改这三行等于宣布「我接受一个新的参照物」。
    G.declare_frozen(
        "原版侧 rvc_orig", ORIG, OURS_PRODUCT_DIRS, MIN_SLICES,
        "不变输入 × 不变上游代码的函数；3_feature768 是 CUDA/TF32 产物、不逐位可复现，重算即换参照",
        expect_sha="65f171f76d76e0e20d8ce7896b5c594acd64a3b111cde21bf3bfcb376e06ccdb",
    )
    G.declare_frozen(
        "C 参照 rvc_B2_orig", F0_FP32_REF, ["2a_f0", "2b-f0nsf"], MIN_SLICES,
        "原版 rmvpe CPU fp32 参照（README 的 ② 命令生成）",
        expect_sha="704092e7e46e47791fef873d5fd28799c910954bb147a4953b0046e1c56923a0",
    )
    G.declare_frozen(
        "C 参照 rvc_fairseq_fp32", FEAT_FP32_REF, ["3_feature768"], MIN_SLICES,
        "真 fairseq 0.12.2 fp32 CPU 参照；重建 = CUDA_VISIBLE_DEVICES=-1 跑上游 extract_feature_print",
        expect_sha="a53d5f9f90983d129e6dfdd850db92d1d43c4726f9144466f758dea88f9f09e8",
    )

    print("== A: 端到端 vs 原版整合包实跑（含 librosa/CUDA-TF32/half 数值轴，松阈值）==")
    compare_wav_dir("0_gt_wavs", exact=True)
    compare_wav_dir("1_16k_wavs", exact=False)
    compare_f0("A", ORIG, frac_thr=0.01, flip_thr=0.005, coarse_thr=0.03)
    # A 特征的输入音频本身就带 16k 重采样轴（他们 kaiser_best/我们 soxr_hq）→
    # cos≈0.985 是输入差异的传导，结构性错误会掉到 0.9x 以下；定审看 C
    compare_feat("A", ORIG, max_thr=3.0, cos_thr=0.98)

    print("== C: 提取器定审（fp32 CPU 参照，紧阈值）==")
    # ⛔ S135 查明：这一条**两侧都在 rvc_B2_\***，而 gate0_run_ours.py 只写 rvc_ours、
    # 全仓没有任何脚本写 rvc_B2_ours ⇒ 除非本轮显式重算过它，这条判据是一次
    # 「七月比七月」的同义反复：今天把 f0 改坏它照样打印七月那组数并 PASS。
    # 而它是我方 f0 提取器**唯一的紧阈值判据**（frac 5e-4 / flip 1e-4，A 层是 1e-2 / 5e-3）
    # ⇒ 一个 1% 帧内、0.5Hz 内的 f0 退化 A 层吸收得掉，全靠这一条。
    b2 = G.collect(OURS_F0_FP32, ["2b-f0nsf"], [".npy"])
    if len(b2) < MIN_SLICES:
        raise G.GateUnrunnable(
            "C 层 f0 我方侧 rvc_B2_ours/2b-f0nsf 只有 %d 件（下限 %d）" % (len(b2), MIN_SLICES)
        )
    if all(m >= t0 for _p, m in b2):
        G.require_fresh("我方 rvc_B2_ours(C 层 fp32 CPU)", OURS_F0_FP32,
                        ["2a_f0", "2b-f0nsf"], t0, MIN_SLICES)
        compare_f0(
            "C", F0_FP32_REF, frac_thr=0.0005, flip_thr=0.0001, coarse_thr=0.0005,
            ours_root=OURS_F0_FP32,
        )
    else:
        G.note_uncovered(
            "C f0 定审",
            "两侧都是冻结产物（rvc_B2_orig × rvc_B2_ours，本轮都没重算）"
            "⇒ 这条判据对今天的 f0 代码零覆盖。要救活它：清 rvc_B2_ours/{2a_f0,2b-f0nsf}，"
            "把本轮的 rvc_ours/1_16k_wavs 同步进 rvc_B2_ours/1_16k_wavs，"
            "再用今天的代码跑一次 fp32 CPU 的 extract_f0（见 gate0_rebuild_b2_ours.py）",
        )
    compare_feat("C", FEAT_FP32_REF, max_thr=2e-3, cos_thr=0.99999)

    selfcheck_filelist_index()
    G.finish("GATE0", failures, allow_uncovered="--allow-uncovered" in sys.argv)


if __name__ == "__main__":
    G.run("GATE0", main)
