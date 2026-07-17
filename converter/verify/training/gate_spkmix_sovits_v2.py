"""SoVITS 4.0-v2 多歌手 spk_mix E2E 补验（S68 批4；S56 方法论移植）。

对象 = 批4 冒烟产出的真 2 歌手 v2 模型（TESTING\\smoke_sovits_v2）。四层：
  ① emb 位恒等：one-hot @ emb_spk.weight == emb_spk.weight[i]（torch.equal，
     真训练权重逐位 —— S56 的 one-hot==gather 硬验同款）
  ② torch(sid) vs ORT(spk_mix one-hot) 确定档全图对拍（noise/phase/f0d_cond
     全零 + 固定 c/f0）：两歌手各一，max|Δ| < 1e-4（fp32 ORT 轴，批1 det tier
     同量级）
  ③ 歌手响应：ORT one-hot0 vs one-hot1 输出实质不同（欠训练模型也必须过 ——
     emb 行不同 → 条件不同）
  ④ 混合 sanity：spk_mix=[.5,.5]（归一后铺进 [1,n_spk]）输出有限且异于两端
companion .f0.onnx：spk_mix one-hot 两歌手响应差 + 有限性。

    converter/.venv/Scripts/python.exe converter/verify/training/gate_spkmix_sovits_v2.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CONV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, CONV)

import json

import numpy as np
import torch

from architectures import sovits_v4v2 as v2

T = 120
TESTING = r"D:\MyDev\TESTING\smoke_sovits_v2"
CKPT = os.path.join(TESTING, "ws_multi", "weights", "smokev2m.pth")
CFG = os.path.join(TESTING, "ws_multi", "weights", "config.json")
ONNX = os.path.join(TESTING, "converted", "smokev2m.onnx")
F0_ONNX = os.path.join(TESTING, "converted", "smokev2m.f0.onnx")
SIDECAR = os.path.join(TESTING, "converted", "smokev2m.json")

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

failures = []


def check(ok, label, detail=""):
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", label, detail))
    if not ok:
        failures.append(label)


def main():
    import onnxruntime as ort

    with open(SIDECAR, encoding="utf-8") as f:
        sc = json.load(f)
    n_spk = sc["spk_mix"]["n_spk"]
    check(sc["spk_mix"]["available"] and "spk_mix" in sc["inputs"], "sidecar spk_mix 块", str(n_spk))
    check(list(sc["speakers"].values()) == [0, 1], "sidecar speakers 双歌手 id 序", str(sc["speakers"]))

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    with open(CFG, encoding="utf-8") as f:
        cfg = json.load(f)
    model, meta = v2.build_from_checkpoint(ckpt, cfg)

    # ① emb 位恒等（真权重，S56 one-hot==gather）
    W = model.emb_spk.weight.detach()
    for i in (0, 1):
        oh = torch.zeros(1, W.shape[0])
        oh[0, i] = 1.0
        check(torch.equal(oh @ W, W[i : i + 1]), "① one-hot@W == W[%d] 逐位" % i)

    # deterministic shared inputs
    rng = np.random.RandomState(1234)
    c = rng.randn(1, T, meta["c_dim"] if "c_dim" in meta else 256).astype(np.float32) * 0.1
    f0 = np.full((1, T), 220.0, dtype=np.float32)
    noise = np.zeros((1, 192, T), dtype=np.float32)
    phase = np.zeros((1, (meta.get("n_fft", 2048) // 2 + 1), T), dtype=np.float32)
    f0d = np.zeros((1, 192, T), dtype=np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])

    def run_ort(mix_vec):
        feeds = {
            "c": c,
            "f0": f0,
            "noise": noise,
            "phase": phase,
            "spk_mix": mix_vec,
            "f0d_cond": f0d,
        }
        return sess.run(["audio"], feeds)[0][0, 0]

    def one_hot(i):
        m = np.zeros((1, n_spk), dtype=np.float32)
        m[0, i] = 1.0
        return m

    # ② torch(sid) vs ORT(one-hot)
    model.export_spk_mix = False
    outs_torch = []
    with torch.no_grad():
        for i in (0, 1):
            a = model(
                torch.from_numpy(c),
                torch.from_numpy(f0),
                torch.from_numpy(noise),
                torch.from_numpy(phase),
                torch.LongTensor([i]),
                torch.from_numpy(f0d),
            )[0, 0].numpy()
            outs_torch.append(a)
    outs_ort = [run_ort(one_hot(0)), run_ort(one_hot(1))]
    for i in (0, 1):
        d = float(np.abs(outs_torch[i] - outs_ort[i]).max())
        check(d < 1e-4, "② torch(sid=%d) vs ORT(one-hot) max|Δ| < 1e-4" % i, "%.2e" % d)

    # ③ 歌手响应
    diff01 = float(np.abs(outs_ort[0] - outs_ort[1]).max())
    rms = float(np.sqrt((outs_ort[0] ** 2).mean()) + 1e-12)
    check(diff01 > 1e-3 * max(rms, 1e-3), "③ 歌手 0 vs 1 输出实质不同", "max|Δ|=%.2e rms=%.2e" % (diff01, rms))

    # ④ 混合 sanity
    mix = np.zeros((1, n_spk), dtype=np.float32)
    mix[0, 0] = 0.5
    mix[0, 1] = 0.5
    out_mix = run_ort(mix)
    check(np.isfinite(out_mix).all(), "④ blend 输出有限")
    check(
        float(np.abs(out_mix - outs_ort[0]).max()) > 0 and float(np.abs(out_mix - outs_ort[1]).max()) > 0,
        "④ blend ≠ 两端",
    )

    # companion .f0.onnx（spk_mix 输入）
    fsess = ort.InferenceSession(F0_ONNX, so, providers=["CPUExecutionProvider"])
    uv = np.ones((1, T), dtype=np.float32)
    f0p = []
    for i in (0, 1):
        p = fsess.run(["f0_pred"], {"c": c, "f0": f0, "uv": uv, "spk_mix": one_hot(i)})[0]
        check(np.isfinite(p).all(), "companion f0 有限 (spk %d)" % i)
        f0p.append(p)
    check(float(np.abs(f0p[0] - f0p[1]).max()) > 0, "companion f0 歌手响应差")

    print()
    if failures:
        print("SPKMIX V2 E2E FAILED: %s" % failures)
        sys.exit(1)
    print("SPKMIX V2 E2E ALL PASS")


if __name__ == "__main__":
    main()
