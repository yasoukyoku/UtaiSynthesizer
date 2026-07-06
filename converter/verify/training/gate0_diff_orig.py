"""gate0_diff（原版侧，ground truth）：在 RVC 整合包 runtime（= S38 钉死的
「原版时代环境」，torch 2.0.0 / torchaudio 2.0.1+cpu / librosa 0.9.1）里对
gate0_diff_prepare staged 的 44k 切片逐文件直调 so-vits-svc 4.1-Stable 的
preprocess_hubert_f0.process_one(diff=True)，产出 --use_diff 全部产物
（.vol.npy / .mel.npy / .aug_mel.npy / .aug_vol.npy）。

soft/f0/spec 已由 prepare 预置（skip-if-exists 跳过）→ 本关只走 vol/mel/aug
数学，原版侧不需要 fairseq / GPU。

运行（cwd 任意）：
    D:\\MyDev\\RVC\\RVC20240604Nvidia\\runtime\\python.exe ^
        converter\\verify\\training\\gate0_diff_orig.py

Harness 补丁（零数值影响，全部登记）：
  - loguru 桩（RVC runtime 无 loguru；纯日志）
  - configs/config.json + configs/diffusion.yaml 快照/恢复（preprocess_hubert_f0
    在模块顶层就读它们；diffusion.yaml 的 vocoder.ckpt 指向本项目训练资产 =
    双方同一份 NSF-HiFiGAN 权重，对拍只剩代码轴）
  - 逐文件直调 process_one 而非 __main__（绕开 spawn 执行器 + shuffle：spawn
    子进程不继承随机种子、shuffle 先消耗随机流 —— 两者都属编排非数学；draw
    顺序改由本 driver 的 random.seed(1234) + sorted 文件序钉死，与我方
    extract 的 random.Random(1234) 逐 draw 对齐）
  - CUDA_VISIBLE_DEVICES=-1（本机有 GPU；process_one 的 mel 走显式 cpu 参数，
    但 belt-and-suspenders）
"""
import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SOVITS = r"D:\MyDev\so-vits-svc\so-vits-svc"
UTAI = r"D:\MyDev\Utai_v2-dev"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SHIM = os.path.join(TESTING, "pyshim")
SPK_DIR = os.path.join(TESTING, "diff_orig", "gate")
NSF_MODEL = os.path.join(UTAI, "data", "models", "training", "sovits", "nsf_hifigan", "model")
OURS_CONFIG = os.path.join(TESTING, "sovits_ours", "config.json")
SEED = 1234

sys.path.insert(0, SHIM)
sys.path.insert(0, SOVITS)


def main():
    os.chdir(SOVITS)
    assert os.path.isdir(SPK_DIR), "run gate0_diff_prepare.py first"
    assert os.path.isfile(NSF_MODEL), "missing nsf_hifigan training asset"

    # snapshot repo configs; preprocess_hubert_f0 reads BOTH at import time
    cfg_dir = os.path.join(SOVITS, "configs")
    snaps = {}
    for n in ("config.json", "diffusion.yaml"):
        p = os.path.join(cfg_dir, n)
        snaps[n] = open(p, "rb").read() if os.path.exists(p) else None

    try:
        # config.json = the S38 gate's ours-side config (vec768 / vol_embedding)
        shutil.copyfile(OURS_CONFIG, os.path.join(cfg_dir, "config.json"))
        # diffusion.yaml: template + our vocoder ckpt (module-level dconfig load)
        with open(
            os.path.join(SOVITS, "configs_template", "diffusion_template.yaml"),
            encoding="utf-8",
        ) as f:
            dtxt = f.read()
        dtxt = dtxt.replace(
            "ckpt: 'pretrain/nsf_hifigan/model'",
            "ckpt: '%s'" % NSF_MODEL.replace("\\", "/"),
        )
        with open(os.path.join(cfg_dir, "diffusion.yaml"), "w", encoding="utf-8") as f:
            f.write(dtxt)

        import random

        import preprocess_hubert_f0 as pp
        from diffusion.vocoder import Vocoder

        mel_extractor = Vocoder("nsf-hifigan", NSF_MODEL, device="cpu")
        random.seed(SEED)
        wavs = sorted(
            os.path.join(SPK_DIR, n)
            for n in os.listdir(SPK_DIR)
            if n.endswith(".wav")
        )
        for i, wav in enumerate(wavs):
            pp.process_one(wav, None, "rmvpe", "cpu", diff=True, mel_extractor=mel_extractor)
            print("processed %d/%d %s" % (i + 1, len(wavs), os.path.basename(wav)))
        print(json.dumps({"n": len(wavs)}))
    finally:
        for n, data in snaps.items():
            p = os.path.join(cfg_dir, n)
            if data is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                with open(p, "wb") as f:
                    f.write(data)
        print("repo configs restored")

    print("GATE0 DIFF ORIG SIDE DONE")


if __name__ == "__main__":
    main()
