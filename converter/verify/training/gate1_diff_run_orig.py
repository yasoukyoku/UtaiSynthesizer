"""gate1_diff（原版侧，ground truth）：在**我们的 venv**（torch 2.5.1，与我方侧
同一解释器 —— 隔离代码轴，S37/S38 关卡1 同款）里 runpy 原样执行 so-vits-svc
4.1-Stable 的 train_diff.py，不改仓库任何文件。

Harness 补丁（只动执行环境，零数值影响，全部登记）：
  - random.seed(1234) + torch.manual_seed(1234)（上游无种子；我方驱动在同一
    位置播种 —— 模型构造/DataLoader shuffle/t 采样/noise/test() 的随机流
    完全对齐）
  - loguru 桩（我们的 venv 没有 loguru；纯日志）
  - faiss 桩（repo 根 utils.py 顶层 import faiss；train_diff 链从不触及检索
    —— S38 gate1_sovits 同款桩）
  - librosa.get_duration(filename=..., sr=...) -> path=...（librosa 0.11 删了
    filename kw；0.9.1 的 filename 路径同样只读文件头、忽略 sr —— 环境轴）
  - CUDA_VISIBLE_DEVICES=-1（yaml device=cpu，belt-and-suspenders）

运行（our venv）：
    training\\.venv\\Scripts\\python.exe converter\\verify\\training\\gate1_diff_run_orig.py
"""
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SOVITS = r"D:\MyDev\so-vits-svc\so-vits-svc"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
YAML = os.path.join(TESTING, "gate1_diff_orig.yaml")
SEED = 1234

sys.path.insert(0, SOVITS)


def install_loguru_stub():
    mod = types.ModuleType("loguru")

    class _Logger:
        def _log(self, level, msg, *a, **k):
            try:
                sys.stderr.write("[%s] %s\n" % (level, msg))
            except Exception:
                pass

        def info(self, msg, *a, **k):
            self._log("INFO", msg)

        def warning(self, msg, *a, **k):
            self._log("WARN", msg)

        def error(self, msg, *a, **k):
            self._log("ERROR", msg)

        def debug(self, msg, *a, **k):
            self._log("DEBUG", msg)

    mod.logger = _Logger()
    sys.modules["loguru"] = mod


def install_faiss_stub():
    # repo-root utils.py does `import faiss` at module top; the train_diff code
    # path never touches retrieval — a bare module object satisfies the import
    sys.modules["faiss"] = types.ModuleType("faiss")


def install_get_duration_shim():
    import librosa

    orig = librosa.get_duration

    def shim(*a, **kw):
        if "filename" in kw:
            kw["path"] = kw.pop("filename")
            kw.pop("sr", None)  # ignored for file-path input in 0.9.1 too
        return orig(*a, **kw)

    librosa.get_duration = shim


def main():
    import random

    import runpy

    import torch

    assert os.path.isfile(YAML), "run gate1_diff_prepare.py first"
    os.chdir(SOVITS)
    install_loguru_stub()
    install_faiss_stub()
    install_get_duration_shim()

    random.seed(SEED)
    torch.manual_seed(SEED)

    sys.argv = ["train_diff.py", "-c", YAML]
    runpy.run_path(os.path.join(SOVITS, "train_diff.py"), run_name="__main__")
    print("GATE1 DIFF ORIG SIDE DONE")


if __name__ == "__main__":
    main()
