"""gate0_diff（我方侧）：对 prepare staged 的同一份 44k 切片跑
utai_train.sovits.extract.extract_all(diff_mode=True)，CPU fp32，aug 种子
与原版侧 harness 相同（random.Random(1234) vs 原版 random.seed(1234)，同一
MT19937 流 + 同 sorted 文件序 + 每文件同 2 次 uniform draw = 逐 draw 对齐）。

运行（our venv）：
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate0_diff_run_ours.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

UTAI = r"D:\MyDev\Utai_v2-dev"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
D44K_ROOT = os.path.join(TESTING, "diff_ours")
OURS_CONFIG = os.path.join(TESTING, "sovits_ours", "config.json")
SEED = 1234

sys.path.insert(0, os.path.join(UTAI, "training"))


class _Stop:
    def requested(self):
        return False

    def check(self):
        pass


class _Reporter:
    def stage(self, stage, done=None, total=None, message=None):
        if done is not None and total is not None and done >= total:
            print("stage %s %s/%s" % (stage, done, total))


def main():
    from utai_train.sovits import utils
    from utai_train.sovits.extract import extract_all

    hps = utils.get_hparams_from_file(OURS_CONFIG)
    extract_all(
        D44K_ROOT,
        hps,
        os.path.join(UTAI, "data", "models", "auxiliary", "contentvec_768l12.onnx"),
        os.path.join(UTAI, "data", "models", "training", "sovits", "rmvpe.pt"),
        "cpu",
        _Reporter(),
        _Stop(),
        diff_mode=True,
        nsf_hifigan_model=os.path.join(
            UTAI, "data", "models", "training", "sovits", "nsf_hifigan", "model"
        ),
        aug_seed=SEED,
    )
    print("GATE0 DIFF OURS SIDE DONE")


if __name__ == "__main__":
    main()
