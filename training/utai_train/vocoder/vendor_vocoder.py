# -*- coding: utf-8 -*-
"""Vendoring tool for the SingingVocoders nsf_HiFigan chain (S40) — provenance
record + regeneration aid. NOT imported by the pipeline.

Discipline (S37/S38/S39): copy-verbatim; the ONLY mechanical diffs are
  (1) a provenance header prepended to each file
  (2) import-line rewrites to the utai_train.vocoder package namespace
Post-copy REGISTERED deviations were applied manually on top (each carries a
header note at its site):
  - training/base_task_gan.py: dataloader persistent/prefetch conditional on
    num_workers==0 (红队 A2) + load_pre_train_model map_location="cpu"
  - utils/training_utils.py: DsModelCheckpoint relative_to → _display_path
    fallback (红队 A4)
Re-running this script would OVERWRITE those deviations — diff before/after.

Audit readout (S40): 28 diff lines total vs upstream = exactly the import
rewrites (16 after the indented-import fix below); everything else verbatim.
"""
import re
import pathlib

SRC = pathlib.Path(r"D:\MyDev\SingingVocoders")
DST = pathlib.Path(__file__).resolve().parent
COMMIT = "4d0889c (2026-03-08)"

FILES = [
    ("models/nsf_HiFigan/models.py",  "models/nsf_HiFigan/models.py"),
    ("training/base_task_gan.py",     "training/base_task_gan.py"),
    ("training/nsf_HiFigan_task.py",  "training/nsf_HiFigan_task.py"),
    ("modules/loss/HiFiloss.py",      "modules/loss/HiFiloss.py"),
    ("modules/loss/stft_loss.py",     "modules/loss/stft_loss.py"),
    ("utils/wav2mel.py",              "utils/wav2mel.py"),
    ("utils/wav2F0.py",               "utils/wav2F0.py"),
    ("utils/config_utils.py",         "utils/config_utils.py"),
    ("utils/training_utils.py",       "utils/training_utils.py"),
    ("utils/__init__.py",             "utils/__init__.py"),
    ("process.py",                    "process_sv.py"),
]

PKG = "utai_train.vocoder"
# ⚠️ 教训（S40 冒烟实锤）：import 重写不能只锚行首——base_task_gan.build_optimizer/
# build_scheduler 里有【函数体内缩进的】`from utils import ...`，行首锚定会漏网
# （运行期 ModuleNotFoundError）。改为允许前导空白。
REWRITES = [
    (re.compile(r"^(\s*)from models\.", re.M),   rf"\1from {PKG}.models."),
    (re.compile(r"^(\s*)from modules\.", re.M),  rf"\1from {PKG}.modules."),
    (re.compile(r"^(\s*)from training\.", re.M), rf"\1from {PKG}.training."),
    (re.compile(r"^(\s*)from utils\.", re.M),    rf"\1from {PKG}.utils."),
    (re.compile(r"^(\s*)from utils import", re.M), rf"\1from {PKG}.utils import"),
    (re.compile(r"^import utils$", re.M),        f"import {PKG}.utils as utils"),
]

HEADER = """\
# vendored: openvpi/SingingVocoders @ {commit} -- {rel}
# 逐字拷贝;仅改动 = 本头注 + import 行重写到 utai_train.vocoder 命名空间(见 vendor_vocoder.py)。
# 数学/RNG/控制流与上游逐字一致;偏离一律另文件实现,不改本文件。
"""


def main():
    for rel, dst_rel in FILES:
        text = (SRC / rel).read_text(encoding="utf-8")
        n_rw = 0
        for pat, repl in REWRITES:
            text, n = pat.subn(repl, text)
            n_rw += n
        dst_p = DST / dst_rel
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        dst_p.write_text(HEADER.format(commit=COMMIT, rel=rel) + text,
                         encoding="utf-8", newline="\n")
        print(f"{rel:35s} -> {dst_rel:35s} rewrites={n_rw}")
    print("done — now RE-APPLY the registered deviations listed in the module header!")


if __name__ == "__main__":
    main()
