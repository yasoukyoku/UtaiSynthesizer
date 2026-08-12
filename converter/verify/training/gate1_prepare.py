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

# ⛔⛔ S140:身份/断言那一侧**必须用展开过的这一份**,不能直接喂 SUBDIRS。
#    `mute` 是**目录套目录**(mute/0_gt_wavs/mute48k.wav 等四件全在下一层),而
#    `gate0_guard.collect` 是**非递归**的(:186-190 `if not os.path.isfile(p): continue`)
#    ⇒ `src_identity` 的 `empty_subs` 会把 `mute` 判成空子目录并抛 GateUnrunnable
#    ⇒ rvc 这条链在 prepare 的第一句断言上退 3,跑器退 6。**实测复现过。**
#    而拷贝端 `copy_artifacts` 用的是**递归**的 `shutil.copytree` ⇒ 两边口径不一致。
#    ⚠ 这条分支自 S139 写下起从没在真夹具上执行过(S139 §7-A 亲口写着本场没跑任何 prepare),
#      而它的自检夹具是**扁平**目录,结构上盖不到这一形。
#    ⛔ **不许把 `mute` 从名单里删掉**:它是 filelist 53 条里的 2 条,删掉等于让输入身份
#      对**真参与训练的样本**装瞎。⛔ 也不许改 `collect`(它是 gate0 dirhash 的地基)。
ID_SUBS = ["0_gt_wavs", "2a_f0", "2b-f0nsf", "3_feature768",
           "mute/0_gt_wavs", "mute/2a_f0", "mute/2b-f0nsf", "mute/3_feature768"]
# 真值(2026-08-12 实测):51×4 + 1×4 = 208。⛔ 地板必须是**这条链的真值**,
# 不是一个与真值无关的常数 —— 那正是 `gate1_guard.py:93-96` 立的纪律,而 S139
# 自己在这五个调用点上写的是 200/10/10/10/5(另外四条比真值低 4~26 倍)。
ID_MIN_FILES = 208


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
    ident = G1.src_identity("gate1/rvc 的输入(= gate0 的 rvc_ours)", SRC, ID_SUBS,
                            min_files=ID_MIN_FILES)
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
