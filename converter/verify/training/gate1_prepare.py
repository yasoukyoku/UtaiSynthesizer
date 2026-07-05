"""关卡1 准备：把同一份预处理产物布置给 原版 train.py 与 我们的 train()，
保证两侧数据集合与顺序完全一致（batch 组成由 filelist 顺序+seed 决定，与路径
字符串无关）。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_prepare.py

布置：
  原版侧: D:\\MyDev\\RVC\\RVC20240604Nvidia\\logs\\gate1  （train.py -e gate1 要求 cwd 相对 logs/）
  我方侧: D:\\MyDev\\TESTING\\utai-v2-testing\\gate1_ours
两侧 config.json 相同：v2/48k 模板 + fp16_run=false（CPU 确定性）+ log_interval=1。
filelist：以我方 gate0 产物 rvc_ours/filelist.txt 的行序为准，逐行改写路径前缀。
"""
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SRC = os.path.join(TESTING, "rvc_ours")
RVC = r"D:\MyDev\RVC\RVC20240604Nvidia"
ORIG_EXP = os.path.join(RVC, "logs", "gate1")
OURS_EXP = os.path.join(TESTING, "gate1_ours")

SUBDIRS = ["0_gt_wavs", "2a_f0", "2b-f0nsf", "3_feature768", "mute"]


def copy_artifacts(dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for sub in SUBDIRS:
        shutil.copytree(os.path.join(SRC, sub), os.path.join(dst, sub))


def rewrite_filelist(dst_exp):
    src_prefix = SRC.replace("\\", "/")
    dst_prefix = dst_exp.replace("\\", "/")
    with open(os.path.join(SRC, "filelist.txt"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = [l.replace(src_prefix, dst_prefix) for l in lines if l]
    with open(os.path.join(dst_exp, "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return len(out)


def write_config(dst_exp):
    with open(os.path.join(SRC, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    config["train"]["fp16_run"] = False
    config["train"]["log_interval"] = 1
    with open(os.path.join(dst_exp, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")


def main():
    for exp in (ORIG_EXP, OURS_EXP):
        copy_artifacts(exp)
        n = rewrite_filelist(exp)
        write_config(exp)
        print(f"prepared {exp}: {n} filelist entries")
    # 抽查两侧行序一致（样本名序列必须逐行相同）
    def sample_names(exp):
        with open(os.path.join(exp, "filelist.txt"), encoding="utf-8") as f:
            return [l.split("|")[0].rsplit("/", 1)[-1] for l in f.read().splitlines() if l]

    assert sample_names(ORIG_EXP) == sample_names(OURS_EXP), "两侧样本顺序不一致"
    print("sample order identical:", len(sample_names(ORIG_EXP)), "entries")


if __name__ == "__main__":
    main()
