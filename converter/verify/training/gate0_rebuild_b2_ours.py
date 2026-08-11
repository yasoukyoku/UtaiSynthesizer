# -*- coding: utf-8 -*-
"""gate0 · 把 RVC 的 **C 层 f0 定审**从一条冻结的自比对救活(S135,§F7 笔 2)。

⛔ 为什么需要它:`gate0_compare.py` 的 C 层 f0 两侧是
      F0_FP32_REF  = TESTING/rvc_B2_orig   (原版 rmvpe CPU fp32,2026-07-05)
      OURS_F0_FP32 = TESTING/rvc_B2_ours   (我方 fp32 CPU,        2026-07-07)
  而 `gate0_run_ours.py` 只写 `rvc_ours`,**全仓没有任何脚本写 rvc_B2_ours**
  (README:23 只给了 orig 侧那条手拼命令;compare:161 那句"由 README 命令生成"是假的)。
  ⇒ 这条判据是一次「七月比七月」的同义反复:今天把 f0 改坏,它照样打印七月那组
  「0/10534 帧超 0.5Hz、max 0.24 mHz」并 PASS。
  而它是我方 f0 提取器**唯一的紧阈值判据**(frac 5e-4 / flip 1e-4),
  A 层是 frac 1e-2 / flip 5e-3 —— 一个 1% 帧内、0.5Hz 内的 f0 退化 A 层吸收得掉。
  ⚠ 而 `rmvpe.py` 恰恰被 `4e525ef` 动过(torch.load 加了 weights_only=False)。

做法(只重算**我方**那一侧,参照侧原封不动):
  1. 要求 `rvc_ours/1_16k_wavs` 是**本轮**产物(否则救活的还是陈货)。
  2. **逐字节**核对 `rvc_ours/1_16k_wavs` 与 `rvc_B2_orig/1_16k_wavs`:
     参照侧是"同一份 16k 输入喂上游 rmvpe CPU fp32"的产物,输入一旦变了它就不再同源。
     · 全等 ⇒ 参照仍然有效,只重算我方侧。
     · 不等 ⇒ ⛔ **拒绝**,并说明必须先按 README:23 在 RVC runtime 里重建 rvc_B2_orig,
              否则救活的是一条拿新输入比旧参照的假判据。
  3. 删掉 `rvc_B2_ours/{2a_f0,2b-f0nsf}`,把本轮 16k wav 同步进 `rvc_B2_ours/1_16k_wavs`,
     用**今天的** `utai_train.rvc.extract_f0` 以 device="cpu" / is_half=False 重算。

运行(our venv,必须带 GATE0_T0):
    training/.venv/Scripts/python.exe converter/verify/training/gate0_rebuild_b2_ours.py
"""
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate0_guard as G  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
OURS = os.path.join(TESTING, "rvc_ours")
B2_OURS = os.path.join(TESTING, "rvc_B2_ours")
B2_ORIG = os.path.join(TESTING, "rvc_B2_orig")
RVC_REPO = r"D:\MyDev\RVC\RVC20240604Nvidia"
MIN_SLICES = 45

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def dirhash(d):
    """目录内容的可复算指纹:按文件名排序,喂 名字 + 字节。"""
    h = hashlib.sha256()
    if not os.path.isdir(d):
        return None, 0
    names = sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)))
    for n in names:
        h.update(n.encode("utf-8"))
        with open(os.path.join(d, n), "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
    return h.hexdigest(), len(names)


def main():
    t0 = G.read_t0("GATE0 RVC C-f0 rebuild")

    src16k = os.path.join(OURS, "1_16k_wavs")
    G.require_fresh("本轮 rvc_ours/1_16k_wavs", src16k, [""], t0, MIN_SLICES, suffixes=[".wav"])

    h_ours, n_ours = dirhash(src16k)
    h_ref, n_ref = dirhash(os.path.join(B2_ORIG, "1_16k_wavs"))
    print("[HASH] rvc_ours/1_16k_wavs      %s (%d 件)" % (h_ours, n_ours))
    print("[HASH] rvc_B2_orig/1_16k_wavs   %s (%d 件)" % (h_ref, n_ref))
    if h_ref is None:
        raise G.GateUnrunnable("rvc_B2_orig/1_16k_wavs 不在 —— 参照侧的输入没了,无从判定同源")
    if h_ours != h_ref:
        raise G.GateUnrunnable(
            "本轮 16k wav 与参照侧 rvc_B2_orig 的输入**不逐字节相同** ⇒ 参照不再同源。\n"
            "       ⛔ 拒绝在这种状态下重算 rvc_B2_ours —— 那会造出一条『拿新输入比旧参照』的假判据。\n"
            "       要继续必须先按 README 的 ② 在 RVC runtime 里用本轮 16k wav 重建 rvc_B2_orig。"
        )
    print("[SAME-SOURCE] 本轮 16k wav 与参照侧输入逐字节相同 ⇒ rvc_B2_orig 仍然有效,只重算我方侧")

    # 同步输入(逐字节相同,所以这一步在正常情况下是 no-op;写出来是为了让脚本对
    # 「参照侧输入被替换过」这种历史状态也自洽)
    dst16k = os.path.join(B2_OURS, "1_16k_wavs")
    os.makedirs(dst16k, exist_ok=True)
    for n in sorted(os.listdir(src16k)):
        shutil.copy2(os.path.join(src16k, n), os.path.join(dst16k, n))

    # ⛔ 删【目录本身】而不是清空内容 —— compare 的守卫只看 isdir,清空会变成假 PASS,
    #    删掉才会正确地红。这里两个目录随后由 extract_f0:67-68 自己 makedirs 重建。
    for sub in ("2a_f0", "2b-f0nsf"):
        p = os.path.join(B2_OURS, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print("[CLEARED] %s" % p)

    from utai_train.protocol import Reporter
    from utai_train.rvc.extract_f0 import extract_f0
    from utai_train.stopfile import StopFlag

    print("[RUN] extract_f0(pool_dir=rvc_B2_ours, device='cpu', is_half=False) —— 今天的代码")
    extract_f0(
        B2_OURS,
        os.path.join(REPO, "data", "models", "auxiliary", "rmvpe.pt"),
        "cpu",
        False,
        os.path.join(RVC_REPO, "ffmpeg.exe"),
        Reporter(throttle_secs=1.0),
        StopFlag(os.path.join(B2_OURS, "stop.flag.never")),
    )

    # 自证:重算出来的东西真的是本轮的
    G.require_fresh("重算后的 rvc_B2_ours/{2a_f0,2b-f0nsf}", B2_OURS,
                    ["2a_f0", "2b-f0nsf"], t0, MIN_SLICES * 2)
    print("B2-OURS-REBUILT")
    sys.exit(G.EXIT_PASS)


if __name__ == "__main__":
    G.run("GATE0 RVC C-f0 rebuild", main)
