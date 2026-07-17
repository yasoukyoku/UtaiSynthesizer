"""SoVITS 4.0-v2 关卡1 prepare：双侧工作区就位（training venv）。

前置：gate0_sovits_v2_run_ours.py 已跑完（复用其 dataset_44k 产物 + filelists +
config —— batch 组成由 filelist 行序 + seed 决定）。
  - 原版侧 = <v2 repo>\\logs\\gate1_sovits_v2（train.py 硬编码 ./logs/<name>）
  - 我方侧 = TESTING\\gate1_sovits_v2_ours
两侧各拷入同一对官方底模 G_0/D_0（~1GB/侧）。
gate 配置：epochs=2 / batch=4 / log_interval=1 / eval_interval=1000（唯一命中的
边界 = 上游必然触发的 step-0 evaluate，RNG 流对拍点）/ num_workers=0（双侧）/
fp16_run=False（v2 恒 fp32）。
★ 上游 data_utils 懒生成 `.mel.npy` —— 我方产物名是 `.aam80.npy`（防 diff 池
同名冲突的登记偏差）→ 此处把每个 .aam80.npy 复制为同目录 .mel.npy，让上游直接
命中缓存（两侧消费**逐字节相同**的 mel 文件，mel 生成轴已由关卡0 C4 单独定审）。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_v2_prepare.py
"""
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SOVITS_V2 = r"D:\MyDev\TESTING\SoVITS-4.0_v2\src\so-vits-svc"
GATE0_EXP = os.path.join(TESTING, "sovits_v2_ours")
ORIG_EXP = os.path.join(SOVITS_V2, "logs", "gate1_sovits_v2")
OURS_EXP = os.path.join(TESTING, "gate1_sovits_v2_ours")
GATE_CFG = os.path.join(TESTING, "gate1_sovits_v2_config.json")
BASE_DIR = os.path.join(REPO, "data", "models", "training", "sovits_v2")

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def main():
    with open(os.path.join(GATE0_EXP, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["train"]["fp16_run"] is False
    assert cfg["train"]["epochs"] == 2 and cfg["train"]["batch_size"] == 4
    assert cfg["train"]["num_workers"] == 0, "gate 需要双侧 num_workers=0"
    cfg["train"]["log_interval"] = 1
    cfg["train"]["eval_interval"] = 1000
    with open(GATE_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # filelist sanity: absolute paths must exist (batch composition = line order + seed)
    for key in ("training_filelist", "validation_filelist"):
        with open(cfg["data"][key], encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                assert not p or os.path.exists(p), "filelist 路径缺失: %s" % p

    # upstream lazy-mel cache: duplicate .aam80.npy -> .mel.npy (see header)
    spk_dir = os.path.join(GATE0_EXP, "dataset_44k", "gate")
    dup = 0
    for n in os.listdir(spk_dir):
        if n.endswith(".aam80.npy"):
            dst = os.path.join(spk_dir, n.replace(".aam80.npy", ".mel.npy"))
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(spk_dir, n), dst)
                dup += 1
    print("duplicated %d aam80 -> mel.npy for the upstream lazy cache" % dup)

    for exp in (ORIG_EXP, OURS_EXP):
        if os.path.isdir(exp):
            shutil.rmtree(exp)
        os.makedirs(exp, exist_ok=True)
        for n in ("G_0.pth", "D_0.pth"):
            shutil.copyfile(os.path.join(BASE_DIR, n), os.path.join(exp, n))
    # ours reads config from its exp dir (train() contract)
    shutil.copyfile(GATE_CFG, os.path.join(OURS_EXP, "config.json"))

    print("GATE1 SOVITS_V2 PREPARE DONE")
    print("  orig exp:", ORIG_EXP)
    print("  ours exp:", OURS_EXP)


if __name__ == "__main__":
    main()
