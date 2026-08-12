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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ⛔ S139: **运行期**确认 CPU-only 真的生效 —— 十个 gate1 跑器里此前只有 `gate1_run_orig.py`
#    有这条硬拒绝，而那是唯一一个**不是被测对象**的臂。只写 stderr（stdout 是协议流）。
import gate1_guard as _G1  # noqa: E402
_G1.assert_cpu_only(__file__)

APP = pathlib.Path(r"D:\MyDev\Utai_v2-dev")
GATE = pathlib.Path(r"D:\MyDev\TESTING\gate1_vocoder")

sys.path.insert(0, str(APP / "training"))

import yaml  # noqa: E402

from utai_train.vocoder import pipeline as vpipe  # noqa: E402
from utai_train.rvc.train_utils import get_logger  # noqa: E402


class _Rep:
    """⛔⛔ S140:此前三个方法**全是 `pass`** ⇒ 这条链上「尺子收到了多少条读数」这件事
    **今天等于零判据**:reporter 被调了十几次,一条都没落地,而 `gate1_vocoder_compare`
    唯一的点数下限量的是 **TB writer 写了几个点**,不是 reporter 收到几条。
    ⇒ 五条链里,声码器是**唯一一条协议/reporter 面完全不在被比较面上**的。

    ⛔ 交接说的「断言 `summary["steps"]==15` 即可覆盖」——**数字对,结论错**:
       `summary` 是 `vocoder/pipeline.py:983-990` 用 `trainer.global_step` 与 `protocol_cb`
       **现算**的,六个键**没有一处读 reporter** ⇒ 三个方法保持 `pass`,那条断言照样绿。
       (那条断言仍然值得加,但它覆盖的是 global→real 的折半,**不是** reporter 通道。)

    ⇒ 改成**计数桩**:三个通道各自记账,跑完 dump 成 `reporter_tally.json`,
      由 compare 拿登记值对拍。⛔ 方法签名**逐字照抄** `protocol.Reporter` 的六个
      (stage/step/ckpt/warn/done/error,含 `force=` 形参)—— 此前仓内四个桩类有**三种**
      写法,而 `gate_driver_arity` 结构上看不见类方法(它只解析模块级 `def`),
      所以桩与生产 Reporter 的签名漂移是一条**全仓无人看守**的面。
    """

    def __init__(self):
        self.n_stage = 0
        self.n_step = 0
        self.ckpts = []
        self.n_warn = self.n_done = self.n_error = 0

    # ⛔ 形参**逐字照抄** `training/utai_train/protocol.py` 的 Reporter(:60/:78/:98/:110/:125/:128)。
    #    ⚠ `step` 的顺序是 `(step, total_steps, epoch, total_epochs, lr, losses, force=False)` ——
    #      **`lr` 在 `losses` 前面**,而生产调用点全是位置参数
    #      (`harness.py:285` / `pipeline.py:980`)⇒ 顺序写反了就是一条静默绑错的桩,
    #      而 `gate_driver_arity` **看不见类方法**(它只解析模块级 `def`)⇒ 全仓无人看守。
    def stage(self, stage, done=None, total=None, message=None, force=False):
        self.n_stage += 1

    def step(self, step, total_steps, epoch, total_epochs, lr, losses, force=False):
        self.n_step += 1

    def ckpt(self, kind, path, step, epoch, metric=None):
        self.ckpts.append(kind)

    def warn(self, code):
        self.n_warn += 1

    def done(self, reason, summary=None):
        self.n_done += 1

    def error(self, message):
        self.n_error += 1

    def tally(self):
        return {"n_stage": self.n_stage, "n_step": self.n_step,
                "ckpt_kinds": list(self.ckpts), "n_ckpt": len(self.ckpts),
                "n_warn": self.n_warn, "n_done": self.n_done, "n_error": self.n_error}


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
    # S134: 3rd positional `pool_dir` (vocoder/pipeline.py:800 `_train(cfg, run_dir, pool_dir,
    # config, reporter, stop)`) — it sits BEFORE `config`, so appending would bind
    # pool_dir=config / config=_Rep() / reporter=_Stop() and still not raise TypeError.
    rep = _Rep()
    summary = vpipe._train(cfg, str(exp_dir), str(exp_dir), config, rep, _Stop())
    # ⛔ S140:reporter 的记账落盘,由 `gate1_vocoder_compare` 拿登记值对拍。
    #    ⚠ 落在 expdir 里 ⇒ prepare 的 rmtree 会把它一起清掉,所以它**永远是本轮的**。
    tally = rep.tally()
    tally["summary_steps"] = summary.get("steps")
    tmp = exp_dir / "reporter_tally.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tally, f, ensure_ascii=False, sort_keys=True, indent=2)
    os.replace(str(tmp), str(exp_dir / "reporter_tally.json"))
    # ⚠ 这条链 `ours_stdout=None`(没有协议 JSONL 流)⇒ stdout 全进 3_ours.log,
    #   这里 print 是安全的。⛔ 但**一旦哪天给它接上真 Reporter + ours_stdout**,
    #   这一行必须先改成 `sys.stderr.write` —— `protocol.py` 头注写死 stdout 是协议流的独占通道。
    print("ours summary:", json.dumps(summary, ensure_ascii=False))
    print("ours reporter tally:", json.dumps(tally, ensure_ascii=False))


if __name__ == "__main__":
    main()
