# -*- coding: utf-8 -*-
"""gate1_vocoder_run_ours — 我方侧：utai_train.vocoder pipeline._train 直驱。

与生产唯一的差异 = gate 小型化 config（prepare 单源）+ 桩 reporter/stop——
_train 内部（seed→task→Trainer→fit）与生产逐语句同路径。CPU（CUDA 屏蔽）。
"""
import json
import os
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

APP = pathlib.Path(r"D:\MyDev\Utai_v2-dev")
GATE = pathlib.Path(r"D:\MyDev\TESTING\gate1_vocoder")

sys.path.insert(0, str(APP / "training"))

import yaml  # noqa: E402

from utai_train.vocoder import pipeline as vpipe  # noqa: E402
from utai_train.rvc.train_utils import get_logger  # noqa: E402


class _Rep:
    def stage(self, *a, **k):
        pass

    def step(self, *a, **k):
        pass

    def ckpt(self, *a, **k):
        pass


class _Stop:
    def check(self):
        pass

    def requested(self):
        return False


def main():
    exp_dir = GATE / "ours" / "gate1_voc"
    exp_dir.mkdir(parents=True, exist_ok=True)
    get_logger(str(exp_dir))  # root logger BEFORE vendored imports (protocol hygiene)

    with open(GATE / "gate_config.yaml", encoding="utf8") as f:
        config = yaml.safe_load(f)

    cfg = {"total_steps": 15, "save_every_steps": 5, "seed": 1234}
    summary = vpipe._train(cfg, str(exp_dir), config, _Rep(), _Stop())
    print("ours summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
