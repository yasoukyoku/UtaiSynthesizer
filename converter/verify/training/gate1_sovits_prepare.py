"""SoVITS 关卡1 布置：两侧共用关卡0 我方侧的预处理产物（sovits_ours 的 filelists
指向绝对路径，双方直接读同一批文件 —— batch 组成由 filelist 行序 + seed 决定）。

  原版侧 model_dir = <so-vits repo>/logs/gate1_sovits（train.py 硬编码 ./logs/<name>）
  我方侧 exp_dir   = TESTING/gate1_sovits_ours

两侧各放同一份 config（log_interval=1 逐步记录）+ 同一对底模 G_0/D_0。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_prepare.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate1_guard as G1                                        # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SOVITS = r"D:\MyDev\so-vits-svc\so-vits-svc"
OURS_G0 = os.path.join(REPO, "data", "models", "training", "sovits", "vec768")

SRC_CFG = os.path.join(TESTING, "sovits_ours", "config.json")
ORIG_DIR = os.path.join(SOVITS, "logs", "gate1_sovits")
OURS_DIR = os.path.join(TESTING, "gate1_sovits_ours")
GATE_CFG = os.path.join(TESTING, "gate1_sovits_config.json")


def main():
    with open(SRC_CFG, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["train"]["all_in_mem"] is True, "gate 需要 all_in_mem（双侧 num_workers=0）"
    assert cfg["train"]["fp16_run"] is False
    assert cfg["train"]["epochs"] == 2 and cfg["train"]["batch_size"] == 4
    cfg["train"]["log_interval"] = 1  # 原版侧 TB 逐 step 记录
    with open(GATE_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ⛔⛔ S139:这条「filelist 里的路径必须都在」的断言原来排在 **rmtree 之后**(:53 vs :39)
    #    ⇒ 一次失败留下的是「两侧 expdir 已经删了、只铺了底模」的残局。全部前移。
    #    顺带白拿一份**输入身份** —— 见 gate1_guard 的「输入身份」一段:
    #    这两条链的输入是 gate0 我方侧产物,而 gate0 重跑一次就换了一棵树,
    #    此前没有任何东西记录或检查这一点。
    paths = []
    for lst in ("training_files", "validation_files"):
        for line in open(cfg["data"][lst], encoding="utf-8"):
            p = line.strip()
            if p:
                paths.append(p)
    # ⛔ S140:地板 10 是一个**与真值无关的常数**,而真值是 33(31 train + 2 val,实测)——
    #    正是 `gate1_guard.py:93-96` 立的纪律在它自己同一场的新代码里被违反。
    #    一棵掉了 70% 文件的 gate0 树照样过这道断言 ⇒ prepare 照常 rmtree 3.34 GB 再拿残树重建。
    # ⚠ 这道断言只覆盖 filelist 里那 31+2 个 **.wav**;真正喂训练的 `.spec.pt`/`.soft.pt`
    #   (同目录 132 件)**不在指纹里** —— 而 S140 实测正是那 66 件在 08-11 20:25 被 gate0
    #   重算过、而 wav 一个没变。⇒ 这条身份对今天唯一真实的输入漂移是**瞎的**,别当它不瞎。
    #   (改成对整个 dataset_44k/gate 算指纹是对的做法,但那会同时改掉 sha 的定义 ⇒ 单独一笔。)
    ident = G1.src_identity_files("gate1/sovits 的输入(filelist 指向的 gate0 产物 .wav)",
                                  paths, min_files=33)
    for n in ("G_0.pth", "D_0.pth"):
        if not os.path.isfile(os.path.join(OURS_G0, n)):
            raise G1.GateUnrunnable("底模缺 %s:%s" % (n, OURS_G0))

    for d in (ORIG_DIR, OURS_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        shutil.copyfile(os.path.join(OURS_G0, "G_0.pth"), os.path.join(d, "G_0.pth"))
        shutil.copyfile(os.path.join(OURS_G0, "D_0.pth"), os.path.join(d, "D_0.pth"))
        G1.write_input_identity(d, ident)
    # ours side reads exp_dir/config.json
    shutil.copyfile(GATE_CFG, os.path.join(OURS_DIR, "config.json"))

    print("prepared:", ORIG_DIR, "and", OURS_DIR)


if __name__ == "__main__":
    G1.run("gate1_sovits_prepare", main)
