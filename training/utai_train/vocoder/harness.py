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
import logging
import math
import os

import lightning.pytorch as pl
import torch

from .. import numerics
from .. import resume_state
from .training.nsf_HiFigan_task import nsf_HiFigan
from .utils.training_utils import DsModelCheckpoint

logger = logging.getLogger(__name__)


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


class UtaiDsModelCheckpoint(DsModelCheckpoint):
    """DsModelCheckpoint + "say WHICH file you just wrote" (★S119 §F8⒝).

    The upstream callback is the only thing that writes the periodic
    ``model_ckpt_steps_<global>.ckpt`` grid, and until §F8⒝ nothing needed to know which of them
    is the newest — ``max(step)`` answered that. 「从最佳存档继续」 breaks that equivalence
    (see :func:`utai_train.resume_state.save_pointer`), so the rolling pointer has to be
    refreshed by whoever actually writes a file, AFTER it exists.

    ⛔ It has to be HERE and not in :class:`UtaiProtocolCallback`. Lightning force-reorders
    Checkpoint-class callbacks to the END of the list
    (``callback_connector.py``: ``tuner_callbacks + other_callbacks + checkpoint_callbacks``),
    so a plain callback's ``on_validation_end`` runs BEFORE this save — a pointer written there
    would name a file that does not exist yet.
    """

    def __init__(self, *args, on_saved=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_saved = on_saved

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if self._on_saved is not None:
            # The parent flattens every path to `dirpath/<basename>` (training_utils.py), so the
            # basename is the whole truth about where the file landed.
            self._on_saved(trainer, os.path.basename(str(filepath)))


class UtaiProtocolCallback(pl.Callback):
    """Protocol bridge: per-batch step messages + stop-flag polling; per-val
    periodic/best weights snapshots. best = TRUE validation loss (the task's
    full-length log10-STFT L1 — S39 precedent: a real val loss beats the EMA
    heuristic GANs are usually forced into), tracked across resumes via
    workspace/best_state.json.

    ★S119 §F8⒝ added three jobs to it, all of which belong to the same moment (a validation
    boundary is where this backend decides anything):

    * write a RESUMABLE archive of the best point — ``weights/vocoder_best.ckpt`` has always been
      quality-selected, but it is ``{'generator': sd}`` only, so the best point was a dead end for
      續訓 exactly like the three GANs before §F2⒜;
    * refresh the rolling POINTER (which flat checkpoint is this branch's tip) and carry the RNG
      + dataset identity that lightning's own checkpoint does not store (measured: its 8 keys
      hold no RNG at any precision);
    * watch the reported losses, because this backend had no divergence guard at all — measured
      (S119 recon), resuming from a nan-poisoned checkpoint ran to `completed`, exit 0, and
      exported a 100%-nan vocoder as a success.
    """

    def __init__(self, reporter, stop, total_steps_real, workspace, pristine_config,
                 resumed=None, start_step=0, dataset_items=None):
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
        # ★S119 §F8⒝ — the sidecar of the archive this run continues from (None when fresh);
        # `restore_report` is filled at on_fit_start so the run can LOG what it managed to
        # restore rather than assume it (resume_state.RestoreReport exists for that reason).
        self.resumed = resumed
        self.restore_report = None
        # The flat checkpoint this branch last wrote — the pointer's payload. Seeded with the
        # file we resumed FROM so that a run killed before its first save still leaves a pointer
        # describing the live branch rather than none at all.
        self.tip_name = None
        self.tip_step = int(start_step)
        self.dataset_items = dataset_items
        self.guard = None

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

    # ---- resume state (★S119 §F8⒝) ----
    def capture(self, trainer):
        """The part of a resume lightning's own checkpoint does NOT carry.

        Measured (S119): a lightning 2.6.5 checkpoint from this chain has exactly 8 top-level
        keys and holds NO RNG at any precision, and there is no GradScaler to lose because
        ``pl_trainer_precision`` is pinned to ``"32-true"`` — so ``scaler=None`` here is a
        structural fact, not an omission.
        """
        return resume_state.capture(
            None,
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            exp_dir=self.workspace,
            dataset_items=self.dataset_items,
        )

    def note_saved(self, trainer, basename):
        """A flat checkpoint just landed — remember it and refresh the rolling pointer."""
        self.tip_name = basename
        self.tip_step = int(trainer.global_step)
        resume_state.save_pointer(self.workspace, basename, blob=self.capture(trainer))

    def on_fit_start(self, trainer, pl_module):
        """Put the captured RNG back — and this is the LAST hook where that still does anything.

        ⛔★ Measured (S119 recon, real hook trace): the train DataLoader's ``_base_seed`` is drawn
        in ``_FitLoop.setup_data()``, which runs BEFORE ``on_train_start``. Every worker's python
        / numpy / torch seed derives from that one number
        (``pl_worker_init_function``: ``base_seed = torch.initial_seed() - worker_id``), and the
        crop offset + volume augmentation are drawn in the WORKERS. So restoring at
        ``on_train_start`` would be a silent no-op — the exact "the verification is empty" shape.

        ⚠ WHAT THIS DOES AND DOES NOT BUY, measured end-to-end, because the honest ceiling here is
        lower than for the GANs. Without it a resumed vocoder run does not merely diverge from an
        uninterrupted one, it REPLAYS: ``seed_everything`` re-runs at every launch, nothing
        between it and the iterator differs between fresh and resume, so the resumed run redraws
        the ORIGINAL run's opening batches — same records, byte-identical crops. Restoring the
        CPU generator changes ``_base_seed`` and ends the replay. It does NOT restore the DATA
        POSITION: lightning truncates the interrupted epoch to the right NUMBER of batches but
        feeds them from the start of the dataloader (it warns about this itself: "your dataloader
        is not resumable"), so the head of the filelist is still seen twice and its tail skipped
        once per resume. Fixing that needs a stateful dataloader — a deviation on the gate1
        parity surface, deliberately not taken here.
        """
        if self.resumed is None:
            return
        self.restore_report = resume_state.restore(self.resumed, None, logger)

    # ---- hooks ----
    def on_train_start(self, trainer, pl_module):
        if self.initial_global is None:
            self.initial_global = trainer.global_step
        if self.guard is None:
            # ⛔★S119 — this backend had NO divergence guard. Measured: resuming from a
            # nan-poisoned checkpoint trained to `completed` with exit 0 and exported a
            # 100%-nan vocoder as a success. `discriminator` is a ModuleDict (msd+mpd); both
            # halves are scanned through it.
            self.guard = numerics.DivergenceGuard(
                (("G", pl_module.generator), ("D", pl_module.discriminator)), logger=logger
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        real = trainer.global_step // 2
        losses = getattr(pl_module, "_utai_losses", {})
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else 0.0
        # total_epochs = 0 sentinel: the vocoder run is step-based, the UI hides
        # epoch displays (S39 diffusion precedent)
        self.reporter.step(real, self.total_steps, trainer.current_epoch, 0, lr, losses)
        if self.guard is not None and losses:
            self.guard.observe(real, losses)
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

        val = trainer.callback_metrics.get("val_loss")
        if val is None:
            # ⛔★S119 — `last_val_global` is committed only WITH a value. It used to be set one
            # line earlier, unconditionally, while `last_val_value` kept an older step's number:
            # `pipeline._train` pairs them (`last_val_value if last_val_global == final_global`)
            # precisely to prove the metric belongs to that checkpoint, and the split assignment
            # made that pair able to disagree with itself — the final checkpoint reported with
            # another checkpoint's score. Latent today (the only non-logging validation path is
            # gated on `skip_immediate_validation`, whose setter is commented out upstream), so
            # this is closing the shape, not a live bug.
            return
        self.last_val_global = trainer.global_step
        v = float(val)
        self.last_val_value = v
        if math.isfinite(v) and (self.best_val is None or v < self.best_val):
            self.best_val = v
            best_path = save_weights_snapshot(
                self.weights_dir, "vocoder_best.ckpt", pl_module, self.pristine_config
            )
            self._save_best(v, real)
            self.reporter.ckpt("best", best_path, real, trainer.current_epoch, metric=v)
            self._save_resumable_best(trainer, pl_module, v)

    def _save_resumable_best(self, trainer, pl_module, metric):
        """★S119 §F8⒝ — the RESUMABLE archive of the best point.

        ``weights/vocoder_best.ckpt`` beside it has always been quality-selected on a real
        validation loss, but it is ``{'generator': sd}``: no discriminator, no optimizer moments,
        no loop state. So until now the best point was a DEAD END for 續訓 — the only thing a user
        could continue from was the newest checkpoint, i.e. the degraded state — the same defect
        §F2⒜ removed for the three GANs and §F8⒜ for shallow diffusion.

        ⛔ Why it needs its own file rather than "protect the flat checkpoint at the best step":
        the flat grid is pruned by ``DsModelCheckpoint(monitor="step", mode="max", save_top_k=N)``,
        i.e. strictly the newest N, so the best point's checkpoint is deleted whenever the best
        validation falls outside the last ``keep_ckpts`` validations (measured). That sweeper is
        upstream bookkeeping keyed on the step number; teaching it about quality is a bigger and
        more fragile change than writing one archive.

        ⛔ ``trainer.save_checkpoint`` — never the checkpoint callback: ``DsModelCheckpoint``
        flattens every path to ``dirpath/<basename>`` (measured), so handing it a nested path
        writes to the workspace ROOT and its remove path deletes the ROOT file of that name.

        The write is guarded on BOTH halves, because they fail independently (S117/S118): weights
        can be nan while the moments are fine, and the moments can be permanently dead
        (``exp_avg_sq = inf``) while every weight is finite. Refusing leaves the previous best
        archive in place, which is the whole point of having one.
        """
        if not numerics.resume_point_is_safe(
            pl_module.state_dict(),
            [("G", trainer.optimizers[0]), ("D", trainer.optimizers[1])]
            if len(trainer.optimizers) >= 2
            else [],
            logger,
        ):
            return
        try:
            resume_state.save_solo_snapshot(
                self.workspace,
                resume_state.BEST_DIR,
                lambda p: trainer.save_checkpoint(p),
                blob=self.capture(trainer),
                metric=metric,
                payload_name=resume_state.BEST_CKPT,
            )
        except Exception as e:  # a full disk must not take the training run down with it
            logger.error("failed to write the resumable best archive: %s: %s", type(e).__name__, e)
