# Vendored from so-vits-svc 4.1-Stable train.py (@ 730930d).
# The training math is UNCHANGED: same dataset/collate (data_utils.py verbatim),
# same model construction (SynthesizerTrn/MultiPeriodDiscriminator from the
# verbatim models.py), same AdamW pair / ExponentialLR(last_epoch=epoch_str-2) /
# warmup, same GradScaler/autocast(half_type) step order (D then G), same loss
# composition (gen + fm + mel*c_mel + kl*c_kl + lf0), same clip_grad_value_(None),
# same resume semantics (latest G_*/D_* in the workspace; the base models are
# copied in as G_0/D_0 by train_prep exactly like upstream's logs/44k drop-in,
# so global_step numbering matches upstream), same eval_interval save cadence +
# clean_checkpoints(keep_ckpts, *_0.pth exempt), same evaluate() tensorboard
# audio/mel dump.
# Deviations (deliberate; verified by the loss-trajectory gate vs upstream):
#   - single-process, no mp.spawn/DDP (upstream wraps even 1 GPU in DDP; the
#     gradients are identical at world_size 1), .cuda(rank) -> device moves with
#     a CPU fallback (upstream hard-asserts CUDA; CPU is needed for the
#     determinism gate and as a last-resort mode — fp16_run is forced off there)
#   - stdout JSONL protocol per step (raw loss values; upstream printed 3-digit
#     stdout logs — those go to the file log instead)
#   - graceful stop: stop flag checked every step -> save latest G/D + a release
#     snapshot, report, exit cleanly (upstream could only be killed)
#   - best tracking: EMA(loss_mel) checked at epoch end -> weights/<name>_best.pth
#     (heuristic — GAN losses are not comparable across steps, mel is the closest
#     perceptual proxy; upstream has no best mechanism and its eval computes no
#     scalar loss; diffusion/vocoder objects get true val-loss best instead)
#   - on natural completion also saves latest G/D + weights/<name>.pth (upstream
#     never saves at the end — it loses everything after the last eval_interval)
#   - release snapshots (weights/*.pth) follow compress_model.removeOptimizer
#     semantics: drop enc_q.*, fp16 weights, {'model',...} wrapper — but without
#     the dummy fresh-optimizer state (the converter only reads ckpt['model']);
#     every eval_interval save ALSO exports one (the UI import list must not
#     point at workspace G_<step>.pth resume files — clean_checkpoints deletes
#     those, and their neighboring config keys the speaker as the slug)
#   - DataLoader gets persistent_workers when num_workers > 0 (Windows spawn
#     would otherwise re-import torch in every worker at EVERY epoch; math
#     unchanged, only the volume-aug RNG stream differs from a per-epoch respawn)
import json
import logging
import math
import multiprocessing
import os
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .. import device as device_shim  # aliased: train() has a local `device = torch.device(...)` that would shadow a bare `device` import
from . import utils
from .data_utils import TextAudioCollate, TextAudioSpeakerLoader
from .models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from .modules import commons
from .modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True

# EMA over ~100 steps; same policy as the RVC trainer (see its header)
BEST_EMA_ALPHA = 2.0 / (100.0 + 1.0)


def export_release(state_dict, path, epoch, learning_rate):
    """compress_model.removeOptimizer semantics: no enc_q, fp16 weights."""
    sd = {k: v.half() for k, v in state_dict.items() if "enc_q" not in k}
    tmp = path + ".tmp"
    torch.save(
        {
            "model": sd,
            "iteration": epoch,
            "optimizer": None,
            "learning_rate": learning_rate,
        },
        tmp,
    )
    os.replace(tmp, path)
    return path


def train(cfg, exp_dir, reporter, stop):
    hps = utils.get_hparams_from_file(os.path.join(exp_dir, "config.json"))
    hps.model_dir = exp_dir
    name = cfg["model_slug"]

    # backend = effective device (cuda|xpu|cpu), single source (shim). Same
    # two-gate split as the RVC trainer: use_cuda gates DEVICE PLACEMENT (true for
    # any accelerator, cuda or xpu — the 8x `.to(device)` below follow it), while
    # `backend == "cuda"` gates cuda-only ops (set_device) + the fp16 precision
    # gate. xpu runs fp32 and selects its device via ZE_AFFINITY_MASK (setup_visibility).
    backend = device_shim.resolve_backend(cfg)
    use_cuda = backend != "cpu"
    amp_backend = backend
    if backend != "cuda":
        # fp16 GradScaler/autocast are CUDA-only; cpu AND xpu run fp32 (design §4.5).
        # NB `!= "cuda"` (never `== "cpu"`) — xpu must also force fp16 off.
        hps.train.fp16_run = False

    global_step = 0

    torch.manual_seed(hps.train.seed)
    if backend == "cuda":
        torch.cuda.set_device(0)
    device = device_shim.torch_device(backend)

    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0)
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem, vol_aug=False)
    eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                             batch_size=1, pin_memory=False,
                             drop_last=False, collate_fn=collate_fn,
                             persistent_workers=True)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        ckpt_name = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step = int(ckpt_name[ckpt_name.rfind("_") + 1:ckpt_name.rfind(".")]) + 1
    except Exception:
        logger.warning("load old checkpoint failed, starting from scratch")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = device_shim.make_scaler(amp_backend, hps.train.fp16_run)
    half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16

    weights_dir = os.path.join(exp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    _write_release_config(exp_dir, weights_dir, cfg)

    total_epochs = int(hps.train.epochs)
    total_steps = total_epochs * max(1, len(train_loader))
    keep_ckpts = getattr(hps.train, "keep_ckpts", 0)

    ema_mel = None
    best_metric = None
    best_step = None
    best_path = os.path.join(weights_dir, "%s_best.pth" % name)
    # best survives resume (RVC trainer policy — see its header)
    best_state_path = os.path.join(exp_dir, "best_state.json")
    if os.path.exists(best_state_path):
        try:
            with open(best_state_path, encoding="utf-8") as f:
                prev = json.load(f)
            best_metric = float(prev["metric"])
            best_step = int(prev["step"])
        except Exception:
            logger.warning("best_state.json unreadable, starting best tracking fresh")

    stopped = False
    steps_this_run = 0
    final_path = None
    last_epoch = epoch_str - 1

    def save_best(epoch):
        export_release(net_g.state_dict(), best_path, epoch, hps.train.learning_rate)
        with open(best_state_path, "w", encoding="utf-8") as f:
            json.dump({"metric": best_metric, "step": best_step}, f)
        reporter.ckpt("best", best_path, best_step, epoch, metric=best_metric)

    def save_gd(epoch, step):
        # step = the number of the last EXECUTED update: upstream saves inside
        # the iteration (pre-increment), and resume continues at parsed+1 — the
        # stop/final paths run post-increment and must pass global_step-1 or a
        # step index would be skipped on every stop/resume cycle
        g_path = os.path.join(hps.model_dir, "G_{}.pth".format(step))
        d_path = os.path.join(hps.model_dir, "D_{}.pth".format(step))
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)
        if keep_ckpts > 0:
            utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
        return g_path

    start_time = time.time()

    for epoch in range(epoch_str, total_epochs + 1):
        last_epoch = epoch
        # set up warm-up learning rate (upstream verbatim)
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch

        net_g.train()
        net_d.train()
        for batch_idx, items in enumerate(train_loader):
            # stop BEFORE the first step of this run counts as "nothing trained"
            if stop.requested():
                stopped = True
                logger.info("stop requested before step %s", global_step)
                break
            c, f0, spec, y, spk, lengths, uv, volume = items
            g = spk.to(device, non_blocking=True)
            spec, y = spec.to(device, non_blocking=True), y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            f0 = f0.to(device, non_blocking=True)
            uv = uv.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            if volume is not None:
                volume = volume.to(device, non_blocking=True)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)

            with device_shim.autocast(amp_backend, enabled=hps.train.fp16_run, dtype=half_type):
                y_hat, ids_slice, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                    spec_lengths=lengths, vol=volume)

                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

                with device_shim.autocast(amp_backend, enabled=False, dtype=half_type):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with device_shim.autocast(amp_backend, enabled=hps.train.fp16_run, dtype=half_type):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with device_shim.autocast(amp_backend, enabled=False, dtype=half_type):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_lf0 = F.mse_loss(pred_lf0, lf0) if net_g.use_automatic_f0_prediction else 0
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            raw_mel = float(loss_mel)
            # a single non-finite step must not poison the EMA forever
            if math.isfinite(raw_mel):
                ema_mel = raw_mel if ema_mel is None else (
                    BEST_EMA_ALPHA * raw_mel + (1 - BEST_EMA_ALPHA) * ema_mel
                )
            lr = optim_g.param_groups[0]['lr']
            reporter.step(
                global_step,
                total_steps,
                epoch,
                total_epochs,
                lr,
                {
                    "g_total": float(loss_gen_all),
                    "d_total": float(loss_disc_all),
                    "gen": float(loss_gen),
                    "fm": float(loss_fm),
                    "mel": raw_mel,
                    "kl": float(loss_kl),
                    "lf0": float(loss_lf0),
                },
            )

            if global_step % hps.train.log_interval == 0:
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if net_g.use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                            pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                 norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, device, global_step)
                save_gd(epoch, global_step)
                # the ckpt surfaced to the UI is a weights/ release snapshot:
                # workspace G_<step>.pth files are resume state managed (and
                # DELETED) by clean_checkpoints, and their sidecar config keys
                # the speaker as the slug — weights/ has the display-name config
                path = export_release(
                    net_g.state_dict(),
                    os.path.join(weights_dir, "%s_e%s_s%s.pth" % (name, epoch, global_step)),
                    epoch,
                    hps.train.learning_rate,
                )
                reporter.ckpt("periodic", path, global_step, epoch)

            global_step += 1
            steps_this_run += 1

            if stop.requested():
                stopped = True
                logger.info("stop requested at epoch %s step %s", epoch, global_step)
                break
        # /steps

        if stopped:
            if steps_this_run > 0:
                save_gd(epoch, global_step - 1)
                path = export_release(
                    net_g.state_dict(),
                    os.path.join(weights_dir, "%s_e%s_s%s.pth" % (name, epoch, global_step)),
                    epoch,
                    hps.train.learning_rate,
                )
                reporter.ckpt("stop", path, global_step, epoch)
                if (
                    ema_mel is not None
                    and math.isfinite(ema_mel)
                    and (best_metric is None or ema_mel < best_metric)
                ):
                    best_metric = ema_mel
                    best_step = global_step
                    save_best(epoch)
                final_path = path
            break

        if (
            ema_mel is not None
            and math.isfinite(ema_mel)
            and (best_metric is None or ema_mel < best_metric)
        ):
            best_metric = ema_mel
            best_step = global_step
            save_best(epoch)

        now = time.time()
        logger.info('====> Epoch: %s, cost %.2f s', epoch, now - start_time)
        start_time = now

        # save the true final state before leaving (upstream's resume point was
        # the last eval_interval boundary, losing the tail); break BEFORE the
        # scheduler step so a later resume sees the same optimizer lr
        if epoch >= total_epochs:
            logger.info("Training is done. The program is closed.")
            save_gd(epoch, global_step - 1)
            final_path = export_release(
                net_g.state_dict(),
                os.path.join(weights_dir, "%s.pth" % name),
                epoch,
                hps.train.learning_rate,
            )
            reporter.ckpt("final", final_path, global_step, epoch)
            break

        scheduler_g.step()
        scheduler_d.step()

    # emit the last step un-throttled so the UI progress reaches the end (EMPTY
    # losses — the EMA is not a raw data point and must not land on the curve)
    if steps_this_run > 0:
        reporter.step(
            global_step,
            total_steps,
            last_epoch,
            total_epochs,
            optim_g.param_groups[0]["lr"],
            {},
            force=True,
        )
    writer.close()
    writer_eval.close()

    return {
        "stopped": stopped,
        "final_weight": final_path,
        "best_weight": best_path if best_metric is not None else None,
        "best_metric": best_metric,
        "best_step": best_step,
        "steps": global_step,
        "epochs": last_epoch,
        "weights_dir": weights_dir,
    }


def _write_release_config(exp_dir, weights_dir, cfg):
    """The converter auto-discovers config.json NEXT TO the .pth it is given.
    The workspace config keys the speaker as the ASCII slug (the data pipeline
    resolves speaker ids from directory names); the release copy in weights/
    carries the real display name so the exported sidecar's speakers map (and a
    kmeans centers file derived from it) shows the user's model name."""
    with open(os.path.join(exp_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    display = cfg.get("model_name") or cfg["model_slug"]
    config["spk"] = {display: 0}
    dst = os.path.join(weights_dir, "config.json")
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, dst)


def evaluate(hps, generator, eval_loader, writer_eval, device, global_step):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv, volume = items
            g = spk[:1].to(device)
            spec, y = spec[:1].to(device), y[:1].to(device)
            c = c[:1].to(device)
            f0 = f0[:1].to(device)
            uv = uv[:1].to(device)
            if volume is not None:
                volume = volume[:1].to(device)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_hat, _ = generator.infer(c, f0, uv, g=g, vol=volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()
