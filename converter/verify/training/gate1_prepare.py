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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402
# ⚠ `gate1_guard` 在 import 时就把 stdout/stderr 钉成 UTF-8(见 gate0_guard 头注第四条)——
#   这一行原来是 `sys.stdout.reconfigure(...)`,只钉了 stdout,而这个文件的断言走的是 stderr。

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
    # ⛔⛔ S139:**全部前置断言 + 输入身份,都要跑在第一句 rmtree 之前。**
    #    `copy_artifacts` 的 `rmtree(dst)` 一执行,`RVC\logs\gate1`(1.39 GB)与
    #    `gate1_ours`(2.93 GB)就没了 —— 而原来只缺一个子目录就会在 copytree 中途
    #    FileNotFoundError,留下「参照侧已经删了、只重建了一半」的残局(S139 实测:
    #    只缺 `3_feature768` ⇒ 退 1,而原版侧落点已残留 12 件 / 5 个子目录只建了 3 个)。
    #    ⚠ 五个 prepare 里只有 `gate1_diff_prepare.py` 原本就把断言放在 rmtree 之前。
    ident = G1.src_identity("gate1/rvc 的输入(= gate0 的 rvc_ours)", SRC, SUBDIRS,
                            min_files=200)
    for extra in ("filelist.txt", "config.json"):
        if not os.path.isfile(os.path.join(SRC, extra)):
            raise G1.GateUnrunnable("gate1/rvc 的输入缺 %s:%s" % (extra, SRC))

    for exp in (ORIG_EXP, OURS_EXP):
        copy_artifacts(exp)
        n = rewrite_filelist(exp)
        write_config(exp)
        # ⛔ 记下**这一轮的输入是哪一棵 gate0 树** —— 见 gate1_guard 的「输入身份」一段:
        #    `rvc_ours` 在 2026-08-11 20:23 被整棵重写过,而 S134 那次 gate1 是当天 09:04 跑的,
        #    此前**没有任何东西记录或检查这一点**。
        G1.write_input_identity(exp, ident)
        print(f"prepared {exp}: {n} filelist entries")

    # 抽查两侧行序一致（样本名序列必须逐行相同）
    def sample_names(exp):
        with open(os.path.join(exp, "filelist.txt"), encoding="utf-8") as f:
            return [l.split("|")[0].rsplit("/", 1)[-1] for l in f.read().splitlines() if l]

    if sample_names(ORIG_EXP) != sample_names(OURS_EXP):
        # ⛔ 原来是裸 assert:AssertionError 走 stderr、消息是中文、本机 locale 是 cp932
        #    ⇒ 在重定向/管道下它自己会 UnicodeEncodeError,而退出码同样是 1 = 真红的码。
        raise G1.GateUnrunnable("两侧样本顺序不一致 —— 这是【闸没准备好】,不是被测的东西不对")
    print("sample order identical:", len(sample_names(ORIG_EXP)), "entries")


if __name__ == "__main__":
    G1.run("gate1_prepare (RVC)", main)
