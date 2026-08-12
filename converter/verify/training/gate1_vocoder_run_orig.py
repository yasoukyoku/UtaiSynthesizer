# -*- coding: utf-8 -*-
"""gate1_vocoder_run_orig — 原版侧：openvpi/SingingVocoders train.py 真实执行。

参照 repo 代码零改动；执行环境 shim（S38「只动执行环境」先例，均为环境轴而非数学轴）：
  1. utils.training_utils.get_strategy -> "auto"（上游深走 lightning≤2.5 私有
     accelerator_connector API，lightning 2.6 已移除——单设备语义等价，我方
     pipeline 的登记偏离同款）
  2. GanBaseTask.train/val_dataloader 的 persistent_workers/prefetch_factor 按
     num_workers==0 条件化（torch 硬性 ValueError；= vendored A2 补丁的镜像，
     双侧同轴）
  3. CUDA 屏蔽（CUDA_VISIBLE_DEVICES=-1）——fp32 CPU 对拍（库内共识）

用法：cd D:/MyDev/SingingVocoders 后以 training/.venv python 运行本脚本。
"""
import os
import pathlib
import runpy
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ORIG = pathlib.Path(r"D:\MyDev\SingingVocoders")
GATE = pathlib.Path(r"D:\MyDev\TESTING\gate1_vocoder")

os.chdir(ORIG)  # upstream resolves configs/work_dir relative to the repo root
sys.path.insert(0, str(ORIG))

import torch.utils.data  # noqa: E402

import utils.training_utils as tu  # noqa: E402  (original repo)
import training.base_task_gan as btg  # noqa: E402


# ---- shim 1: get_strategy (lightning 2.6 private-API removal) ----
tu.get_strategy = lambda *a, **k: "auto"

# ---- shim 2: workers==0 dataloader legality (mirror of vendored A2) ----
def _train_dl(self):
    nw = self.config["ds_workers"]
    return torch.utils.data.DataLoader(
        self.train_dataset,
        collate_fn=self.train_dataset.collater,
        batch_size=self.config["batch_size"],
        num_workers=nw,
        prefetch_factor=self.config["dataloader_prefetch_factor"] if nw > 0 else None,
        pin_memory=True,
        persistent_workers=nw > 0,
    )


def _val_dl(self):
    nw = self.config["ds_workers"]
    return torch.utils.data.DataLoader(
        self.valid_dataset,
        collate_fn=self.valid_dataset.collater,
        batch_size=1,
        num_workers=nw,
        prefetch_factor=self.config["dataloader_prefetch_factor"] if nw > 0 else None,
        shuffle=False,
    )


btg.GanBaseTask.train_dataloader = _train_dl
btg.GanBaseTask.val_dataloader = _val_dl

# work_dir stays INSIDE the repo tree (gitignored /experiments/ — upstream's
# own default layout): DsModelCheckpoint logs paths via relative_to(cwd) and
# hard-crashes on out-of-tree work dirs (the very 红队 A4 bug our vendored
# side patches; the参照 repo stays code-untouched, so we satisfy its cwd
# assumption instead). The compare script reads from here.
import shutil  # noqa: E402

work = ORIG / "experiments" / "gate1_voc"
if work.exists():
    shutil.rmtree(work)

# ⛔⛔ S140:把 prepare 记下的**输入身份**搬到这条臂**真正的** expdir 里。
#    此前它只写在 `TESTING\gate1_vocoder\orig\` —— 而那个目录**原版臂从头到尾不碰**
#    (实测今天 0 件),原版臂的 expdir 是上面这个 `work`,compare 的 ORIG_LOGS 也指那里。
#    ⇒ 那条 `[INPUT-ID] orig` 行挂在一个与被读数据**没有任何因果链**的空壳上,
#      而「一个指着无关目录的身份比没有身份更坏」。
#    ⚠ 必须在 rmtree **之后**写(否则连同 expdir 一起被自己删掉)。
_ident_src = GATE / "orig" / "gate1_input.identity.json"
work.mkdir(parents=True, exist_ok=True)
if _ident_src.is_file():
    shutil.copy2(str(_ident_src), str(work / "gate1_input.identity.json"))
else:
    sys.stderr.write(
        "[NO-INPUT-ID] gate1_vocoder_run_orig:%s 不在 ⇒ 这条臂的 expdir 里不会有输入身份"
        "(本轮没跑 prepare?)\n" % _ident_src)

sys.argv = [
    "train.py",
    "--config", str(GATE / "gate_config.yaml"),
    "--exp_name", "gate1_voc",
]
runpy.run_path(str(ORIG / "train.py"), run_name="__main__")
print("orig side done ->", work)
