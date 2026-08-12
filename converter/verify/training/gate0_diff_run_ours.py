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
    """⛔ S140:形参**逐字照抄** `training/utai_train/protocol.py` 的 Reporter
    (:60/:78/:98/:110/:125/:128)。此前仓内四个桩类有**三种**写法(`*a, **k` / 窄签名缺
    `force` / 少五个方法),而 `gate_driver_arity` **结构上看不见类方法**(它只解析模块级
    `def`,:97-99 与 :184-186)⇒ 桩与生产 Reporter 的签名漂移是一条**全仓无人看守**的面,
    而它的 `VERDICT ALL-BIND` 那条绿对这一面是空的。
    ⚠ 生产里 `stage`/`step` 各有 8 处与 4 处 `force=True` 的调用点 ⇒ 窄签名不只在 stage 上会炸。
    """

    def stage(self, stage, done=None, total=None, message=None, force=False):
        # ⚠ 这一行是**转录装饰,不是判据** —— `gate0_guard.py:214-216` 与
        #   `gate0_compare.py:188-190` 都明写「别拿 stage 的 done 计数当判据」
        #   (它在 skip 之前汇报,全跳过也走满)。
        if done is not None and total is not None and done >= total:
            print("stage %s %s/%s" % (stage, done, total))

    def step(self, step, total_steps, epoch, total_epochs, lr, losses, force=False):
        pass

    def ckpt(self, kind, path, step, epoch, metric=None):
        pass

    def warn(self, code):
        pass

    def done(self, reason, summary=None):
        pass

    def error(self, message):
        pass


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
