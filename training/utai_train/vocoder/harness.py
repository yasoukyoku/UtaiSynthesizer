"""Lightning-side harness for the vocoder backend (S40): the protocol bridge
Callback, the task subclass exposing per-component losses, and the weights/
snapshot writer ({'generator': sd} + config.json — the exact pair
converter/export_nsf_hifigan.py consumes; export_ckpt.py semantics).

STEP UNITS (the one place this is allowed to be subtle): lightning manual-opt
GAN counts BOTH optimizer steps — trainer.global_step advances 2 per batch
(upstream README: "实际的步数是它显示步数的一半"). Everything user-facing
(protocol step/total_steps, weights snapshot names, run.json
total_steps/save_every_steps) uses REAL steps = global_step // 2;
lightning-internal artifacts (model_ckpt_steps_<N>.ckpt names, TB x-axis,
Trainer max_steps, log_interval) stay in global units verbatim.
"""
import json
import math
import os

import lightning.pytorch as pl
import torch

from .training.nsf_HiFigan_task import nsf_HiFigan


class UtaiNsfTask(nsf_HiFigan):
    """training_step override = upstream nsf_HiFigan/GanBaseTask.training_step
    verbatim (base_task_gan.py:335-345) + keep the component-loss dict on the
    module for the protocol callback (upstream only bar-logs the sum and TB-logs
    every log_interval). Zero math / RNG change."""

    def training_step(self, sample, batch_idx):
        log_outputs = self._training_step(sample, batch_idx)
        self.log_dict(
            {"loss": sum(log_outputs.values())},
            prog_bar=True, logger=False, on_step=True, on_epoch=False,
        )
        if self.global_step % self.config["log_interval"] == 0:
            tb_log = {f"training/{k}": v for k, v in log_outputs.items()}
            self.logger.log_metrics(tb_log, step=self.global_step)
        self._utai_losses = {k: float(v) for k, v in log_outputs.items()}

    def print_arch(self):
        # upstream setup() prints the module tree with a raw print() — stdout
        # is protocol-owned here, route it to the file/stderr log instead
        import logging

        logging.getLogger(__name__).info("| model Arch: %s", self)

    def setup(self, stage):
        # idempotent setup (tail-validation deviation depends on this):
        # trainer.validate() re-enters setup, and the upstream implementation
        # unconditionally REBUILDS the model — post-fit that would evaluate
        # freshly initialized weights (and even skip the finetune seed, since
        # the workspace already holds checkpoints). fit calls setup exactly
        # once, so the gate1 parity surface is untouched.
        # ⚠️ sentinel is generator, NOT model: upstream build_model() has no
        # return statement, so GanBaseTask.setup leaves self.model = None
        # forever (a self.model guard silently never fires — caught by the
        # 0.88-vs-0.36 tail-val readout in the off-grid smoke).
        if self.generator is not None:
            return
        super().setup(stage)


def export_config_json(pristine_config):
    """export_ckpt.py:40-61 verbatim mapping: model_args + the mel geometry +
    pc_aug/mini_nsf/noise_sigma defaults. Built from the PRISTINE config dict
    (build_model mutates config['model_args'] in place; upstream reads the
    yaml dumped before task construction — same pre-mutation values)."""
    new_config = dict(pristine_config["model_args"])
    new_config["sampling_rate"] = pristine_config["audio_sample_rate"]
    new_config["num_mels"] = pristine_config["audio_num_mel_bins"]
    new_config["hop_size"] = pristine_config["hop_size"]
    new_config["n_fft"] = pristine_config["fft_size"]
    new_config["win_size"] = pristine_config["win_size"]
    new_config["fmin"] = pristine_config["fmin"]
    new_config["fmax"] = pristine_config["fmax"]
    new_config["pc_aug"] = pristine_config.get("pc_aug", False)
    if "mini_nsf" not in new_config:
        new_config["mini_nsf"] = False
    if "noise_sigma" not in new_config:
        new_config["noise_sigma"] = 0.0
    return new_config


def save_weights_snapshot(weights_dir, filename, pl_module, pristine_config):
    """workspace/weights/<filename> = {'generator': sd} deploy-format snapshot
    (weight_norm params kept raw — remove_weight_norm happens at LOAD time,
    exactly like the community checkpoints) + a config.json beside it.
    Atomic writes; the .tmp+replace changes the zip archive name embedded by
    torch.save, so cross-run comparisons must compare tensors, not bytes
    (S39 lesson)."""
    os.makedirs(weights_dir, exist_ok=True)
    snap = {
        "generator": {
            k: v.detach().cpu().clone()
            for k, v in pl_module.generator.state_dict().items()
        }
    }
    path = os.path.join(weights_dir, filename)
    tmp = path + ".tmp"
    torch.save(snap, tmp)
    os.replace(tmp, path)

    cfg_path = os.path.join(weights_dir, "config.json")
    if not os.path.exists(cfg_path):
        cfg_tmp = cfg_path + ".tmp"
        with open(cfg_tmp, "w", encoding="utf-8") as f:
            json.dump(export_config_json(pristine_config), f, indent=1)
        os.replace(cfg_tmp, cfg_path)
    return path


class UtaiProtocolCallback(pl.Callback):
    """Protocol bridge: per-batch step messages + stop-flag polling; per-val
    periodic/best weights snapshots. best = TRUE validation loss (the task's
    full-length log10-STFT L1 — S39 precedent: a real val loss beats the EMA
    heuristic GANs are usually forced into), tracked across resumes via
    workspace/best_state.json."""

    def __init__(self, reporter, stop, total_steps_real, workspace, pristine_config):
        self.reporter = reporter
        self.stop = stop
        self.total_steps = int(total_steps_real)
        self.workspace = workspace
        self.weights_dir = os.path.join(workspace, "weights")
        self.pristine_config = pristine_config
        self.best_file = os.path.join(workspace, "best_state.json")
        self.best_val = self._load_best()
        self.stop_requested = False
        self.initial_global = None
        # tail-validation bookkeeping: which global step last got a validation
        # (graceful stops get one from lightning itself — observed; natural
        # off-grid completion does NOT, the pipeline back-fills it post-fit)
        self.last_val_global = None
        self.last_val_value = None

    # ---- best bookkeeping (survives resumes independently of lightning state) ----
    def _load_best(self):
        try:
            with open(self.best_file, encoding="utf-8") as f:
                v = json.load(f).get("best_val")
            return float(v) if v is not None and math.isfinite(float(v)) else None
        except Exception:
            return None

    def _save_best(self, val, real_step):
        tmp = self.best_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"best_val": val, "step": real_step}, f)
        os.replace(tmp, self.best_file)

    # ---- hooks ----
    def on_train_start(self, trainer, pl_module):
        if self.initial_global is None:
            self.initial_global = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        real = trainer.global_step // 2
        losses = getattr(pl_module, "_utai_losses", {})
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else 0.0
        # total_epochs = 0 sentinel: the vocoder run is step-based, the UI hides
        # epoch displays (S39 diffusion precedent)
        self.reporter.step(real, self.total_steps, trainer.current_epoch, 0, lr, losses)
        if not self.stop_requested and self.stop.requested():
            self.stop_requested = True
            trainer.should_stop = True

    def on_validation_end(self, trainer, pl_module):
        # fires after the sanity check too — no snapshot before any training
        if trainer.sanity_checking:
            return
        real = trainer.global_step // 2
        # periodic snapshot = the convert-ready import candidate (S38: the
        # protocol must reference weights/, never the keep_ckpts-cleaned
        # workspace lightning checkpoints)
        path = save_weights_snapshot(
            self.weights_dir, f"vocoder_{real}.ckpt", pl_module, self.pristine_config
        )
        self.reporter.ckpt("periodic", path, real, trainer.current_epoch)

        self.last_val_global = trainer.global_step
        val = trainer.callback_metrics.get("val_loss")
        if val is None:
            return
        v = float(val)
        self.last_val_value = v
        if math.isfinite(v) and (self.best_val is None or v < self.best_val):
            self.best_val = v
            best_path = save_weights_snapshot(
                self.weights_dir, "vocoder_best.ckpt", pl_module, self.pristine_config
            )
            self._save_best(v, real)
            self.reporter.ckpt("best", best_path, real, trainer.current_epoch, metric=v)
