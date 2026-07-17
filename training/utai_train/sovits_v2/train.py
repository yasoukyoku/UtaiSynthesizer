# Vendored from so-vits-svc 4.0-v2 train.py (@ cf5a8fb) — VISinger2 training
# loop. The training math is UNCHANGED: same dataset/collate (data_utils.py
# vendored, DatasetConstructor with DistributedSampler(num_replicas=1, rank=0)
# + set_epoch, exactly the upstream cpurun/run wiring so the shuffle and every
# RNG consumer stay in upstream order), same model construction
# (SynthesizerTrn/Discriminator from the vendored models.py), same AdamW pair /
# ExponentialLR(last_epoch=epoch_str-2), same D-then-G step order with
# accumulation_steps gating the G step, same loss composition
# (gen/2 + fm/2 + 45*mel + 45*mel_ddsp + 45*spec_ddsp + mel_am + kl + lf0,
# kl/lf0/mel_am computed inside the model forward), same clip_grad_value_(None),
# same eval_interval cadence INCLUDING the step-0 boundary (evaluate() runs at
# global_step 0 on a fresh run and consumes RNG — the loss-trajectory gate
# depends on that), same evaluate() semantics (valid_loader, batch_idx==8 cap).
# v2 is pure fp32: upstream has no amp/GradScaler anywhere (fp16_run is a dead
# config key) — no autocast here either; the Rust layer forces fp16=false.
# Deviations (deliberate; verified by the loss-trajectory gate vs upstream):
#   - single-process: no mp.spawn/DDP/dist (upstream CUDA path wraps rank 0 in
#     DDP with find_unused_parameters — world-size-1 gradients are identical;
#     upstream's own cpurun path is already DDP-free and is the gate reference),
#     .cuda(rank) -> .to(device) with the device shim (cpu/xpu fallback)
#   - stdout JSONL protocol per step (upstream printed to stdout; that goes to
#     the file log), tensorboard kept (scalars at log_interval like upstream)
#   - graceful stop: stop flag checked every step -> save latest G/D + release
#     snapshot, report, exit cleanly
#   - resume: global_step parses from the latest D_*.pth name (0 stays 0 = the
#     fresh/seeded case matching upstream's (epoch_str-1)*len numbering; N>0
#     resumes at N+1). Upstream recomputes (epoch-1)*len and REPLAYS the whole
#     epoch — ours continues precisely (same policy as the 4.x trainer). The
#     cfg "skip_optimizer" knob (default False = upstream run()'s semantics of
#     inheriting the base ckpt's optimizer state) exists for the determinism
#     gate, whose reference is upstream's native cpurun (skip_optimizer=True,
#     forced epoch 1 / step 0) — the knob mirrors that exactly.
#   - at the step-0 eval boundary evaluate() still runs (RNG parity, above) but
#     the G_0/D_0 save + release export are SKIPPED: upstream overwrites the
#     seeded base G_0.pth with a 1-step-trained copy + optimizer, which both
#     churns 400MB and destroys the pristine base (saves consume no RNG).
#     The stop/final paths skip the save for the same step-0 edge (a run
#     stopped after exactly one update would otherwise write a REAL G_0/D_0
#     that the resume rule misreads as the fresh base): the release snapshot
#     still captures those weights, the base stays pristine, and resume
#     restarts step 0 on the untouched base = clean fresh continuation
#   - best tracking: EMA(loss_mel) at epoch end -> weights/<name>_best.pth
#     (same heuristic as the 4.x GAN trainer; upstream has no best mechanism)
#   - on stop/natural completion also saves latest G/D + weights release
#     snapshot (upstream loses everything after the last eval_interval)
#   - release snapshots (weights/*.pth): upstream v2 has no removeOptimizer —
#     ours drops the optimizer + the training-only posterior_encoder.* keys and
#     halves the weights (the converter re-floats and tolerates the stripped
#     posterior via its has_posterior flag; f0_decoder is KEPT — the auto-f0
#     companion export needs it), 424MB -> ~180MB
import json
import logging
import math
import os
import time

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .. import device as device_shim  # aliased: train() has a local `device`
from . import utils
from .data_utils import DatasetConstructor
from ..sovits.flist import resolve_speakers
from .models import (
    Discriminator,
    SynthesizerTrn,
)
from .modules import commons
from .modules.losses import discriminator_loss, feature_loss, generator_loss
from .modules.mel_processing import mel_spectrogram_torch, spectrogram_torch

logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True

# EMA over ~100 steps; same policy as the 4.x trainers
BEST_EMA_ALPHA = 2.0 / (100.0 + 1.0)


def export_release(state_dict, path, epoch, learning_rate):
    """v2 release snapshot: drop optimizer + training-only posterior_encoder.*,
    fp16 weights (see header; f0_decoder stays — auto-f0 companion needs it)."""
    sd = {
        k: v.half() if v.is_floating_point() else v
        for k, v in state_dict.items()
        if not k.startswith("posterior_encoder.")
    }
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

    backend = device_shim.resolve_backend(cfg)
    use_cuda = backend != "cpu"

    torch.manual_seed(hps.train.seed)
    if backend == "cuda":
        torch.cuda.set_device(0)
    device = device_shim.torch_device(backend)

    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # upstream cpurun/run wiring verbatim: DatasetConstructor owns the loaders
    # (DistributedSampler num_replicas=1 rank=0 shuffle, worker counts from
    # train.num_workers — see data_utils header)
    dataset_constructor = DatasetConstructor(hps, num_replicas=1, rank=0)
    train_loader = dataset_constructor.get_train_loader()
    valid_loader = dataset_constructor.get_valid_loader()

    net_g = SynthesizerTrn(hps).to(device)
    net_d = Discriminator(hps, hps.model.use_spectral_norm).to(device)

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

    # skip_optimizer: gate knob mirroring upstream cpurun (see header)
    skip_optimizer = bool(cfg.get("skip_optimizer", False))
    global_step = 0
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        ckpt_name = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        parsed = int(ckpt_name[ckpt_name.rfind("_") + 1:ckpt_name.rfind(".")])
        # fresh seeded base (D_0) -> 0, matching upstream's (epoch-1)*len; a
        # real save at step N (pre-increment) resumes at N+1 (header)
        global_step = 0 if parsed == 0 else parsed + 1
    except Exception:
        logger.warning("load old checkpoint failed, starting from scratch")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        # upstream cpurun verbatim: optimizer fresh, epoch/step forced to start
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    weights_dir = os.path.join(exp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    _write_release_config(exp_dir, weights_dir, cfg)

    total_epochs = int(hps.train.epochs)
    total_steps = total_epochs * max(1, len(train_loader))
    keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
    accumulation_steps = int(getattr(hps.train, "accumulation_steps", 1))

    ema_mel = None
    best_metric = None
    best_step = None
    best_path = os.path.join(weights_dir, "%s_best.pth" % name)
    # best survives resume (4.x trainer policy)
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
        # step = the number of the last EXECUTED update (stop/final paths run
        # post-increment and pass global_step-1; the in-loop periodic save is
        # pre-increment) — same numbering contract as the 4.x trainer
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
        train_loader.sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()
        for batch_idx, data_dict in enumerate(train_loader):
            # stop BEFORE the first step of this run counts as "nothing trained"
            if stop.requested():
                stopped = True
                logger.info("stop requested before step %s", global_step)
                break

            c = data_dict["c"]
            mel = data_dict["mel"]
            f0 = data_dict["f0"]
            uv = data_dict["uv"]
            wav = data_dict["wav"]
            spkid = data_dict["spkid"]

            c_lengths = data_dict["c_lengths"]
            mel_lengths = data_dict["mel_lengths"]

            if use_cuda:
                c, c_lengths = c.to(device, non_blocking=True), c_lengths.to(device, non_blocking=True)
                mel, mel_lengths = mel.to(device, non_blocking=True), mel_lengths.to(device, non_blocking=True)
                wav = wav.to(device, non_blocking=True)
                f0 = f0.to(device, non_blocking=True)
                spkid = spkid.to(device, non_blocking=True)
                uv = uv.to(device, non_blocking=True)

            # forward (upstream verbatim: kl/lf0/mel_am terms come out of the model)
            y_hat, ids_slice, LF0, y_ddsp, kl_div, predict_mel, mask, \
                pred_lf0, loss_f0, norm_f0 = net_g(c, c_lengths, f0, uv, mel, mel_lengths, spk_id=spkid)
            y_ddsp = y_ddsp.unsqueeze(1)

            # Discriminator
            y = commons.slice_segments(wav, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
            y_ddsp_mel = mel_spectrogram_torch(
                y_ddsp.squeeze(1),
                hps.data.n_fft,
                hps.data.acoustic_dim,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_size,
                hps.data.fmin,
                hps.data.fmax
            )

            y_logspec = torch.log(spectrogram_torch(
                y.squeeze(1),
                hps.data.n_fft,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_size
            ) + 1e-7)

            y_ddsp_logspec = torch.log(spectrogram_torch(
                y_ddsp.squeeze(1),
                hps.data.n_fft,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_size
            ) + 1e-7)

            y_mel = mel_spectrogram_torch(
                y.squeeze(1),
                hps.data.n_fft,
                hps.data.acoustic_dim,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_size,
                hps.data.fmin,
                hps.data.fmax
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.n_fft,
                hps.data.acoustic_dim,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_size,
                hps.data.fmin,
                hps.data.fmax
            )

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

            optim_d.zero_grad()
            loss_disc_all.backward()
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            optim_d.step()

            # Generator loss (upstream verbatim composition + accumulation gate)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

            loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
            loss_mel_dsp = F.l1_loss(y_mel, y_ddsp_mel) * 45
            loss_spec_dsp = F.l1_loss(y_logspec, y_ddsp_logspec) * 45

            loss_mel_am = F.mse_loss(mel * mask, predict_mel * mask)

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)

            loss_fm = loss_fm / 2
            loss_gen = loss_gen / 2
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_mel_dsp + kl_div + loss_mel_am + loss_spec_dsp + \
                loss_f0

            loss_gen_all = loss_gen_all / accumulation_steps

            loss_gen_all.backward()
            if (global_step + 1) % accumulation_steps == 0:
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                optim_g.step()
                optim_g.zero_grad()

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
                    "adv": float(loss_gen),
                    "fm": float(loss_fm),
                    "mel": raw_mel,
                    "mel_ddsp": float(loss_mel_dsp),
                    "spec_ddsp": float(loss_spec_dsp),
                    "mel_am": float(loss_mel_am),
                    "kl": float(kl_div),
                    "lf0": float(loss_f0),
                },
            )

            if global_step % hps.train.log_interval == 0:
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(
                    "Losses: g_total=%s mel=%s, step: %s, lr: %s",
                    float(loss_gen_all), raw_mel, global_step, lr,
                )

                # upstream scalar tags verbatim (the loss-trajectory gate reads
                # the ORIGINAL side's tensorboard with these same tags)
                scalar_dict = {"loss/total": loss_gen_all,
                               "loss/mel": loss_mel,
                               "loss/adv": loss_gen,
                               "loss/fm": loss_fm,
                               "loss/mel_ddsp": loss_mel_dsp,
                               "loss/spec_ddsp": loss_spec_dsp,
                               "loss/mel_am": loss_mel_am,
                               "loss/kl_div": kl_div,
                               "loss/lf0": loss_f0,
                               "learning_rate": lr,
                               "grad_norm_d": grad_norm_d}
                image_dict = {
                    "train/lf0": utils.plot_data_to_numpy(LF0[0, 0, :].cpu().numpy(),
                                                          pred_lf0[0, 0, :].detach().cpu().numpy()),
                    "train/norm_lf0": utils.plot_data_to_numpy(LF0[0, 0, :].cpu().numpy(),
                                                               norm_f0[0, 0, :].detach().cpu().numpy()),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                    images=image_dict)

            if global_step % hps.train.eval_interval == 0:
                # evaluate ALWAYS runs on the boundary (upstream order/RNG); the
                # ckpt work is skipped at step 0 (header: pristine seeded base)
                evaluate(hps, net_g, valid_loader, writer_eval, device, use_cuda, global_step)
                if global_step > 0:
                    save_gd(epoch, global_step)
                    # the ckpt surfaced to the UI is a weights/ release snapshot
                    # (workspace G_<step>.pth resume files get DELETED by
                    # clean_checkpoints; S38 lesson)
                    path = export_release(
                        net_g.state_dict(),
                        os.path.join(weights_dir, "%s_e%s_s%s.pth" % (name, epoch, global_step)),
                        epoch,
                        hps.train.learning_rate,
                    )
                    reporter.ckpt("periodic", path, global_step, epoch)
                net_g.train()

            global_step += 1
            steps_this_run += 1

            if stop.requested():
                stopped = True
                logger.info("stop requested at epoch %s step %s", epoch, global_step)
                break
        # /steps

        if stopped:
            if steps_this_run > 0:
                # step-0 edge (stop after exactly ONE executed update): saving
                # would overwrite the pristine seeded G_0/D_0 AND the resume
                # rule would misread the trained state as the fresh base —
                # skip the workspace save (the release snapshot below still
                # captures the weights); resume restarts step 0 on the
                # UNTOUCHED base = a clean fresh continuation, zero drift
                if global_step - 1 > 0:
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

        # save the true final state before leaving; break BEFORE the scheduler
        # step so a later resume sees the same optimizer lr (4.x policy)
        if epoch >= total_epochs:
            logger.info("Training is done. The program is closed.")
            if global_step - 1 > 0:  # same step-0 edge as the stop path above
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
    kmeans centers file derived from it) shows the user's model name. Same
    policy + id ordering (resolve_speakers single source) as the 4.x trainer."""
    with open(os.path.join(exp_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    speakers = resolve_speakers(cfg)
    config["spk"] = {sp["name"]: i for i, sp in enumerate(speakers)}
    dst = os.path.join(weights_dir, "config.json")
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, dst)


def evaluate(hps, generator, eval_loader, writer_eval, device, use_cuda, global_step):
    """Upstream evaluate() verbatim semantics: batch cap 8, first-item slicing,
    infer(predict_f0 default False) — it CONSUMES torch RNG (z_p noise + phase
    rand), so the call must stay on the eval_interval boundary (gate parity)."""
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(eval_loader):
            if batch_idx == 8:
                break
            c = data_dict["c"]
            mel = data_dict["mel"]
            f0 = data_dict["f0"]
            uv = data_dict["uv"]
            wav = data_dict["wav"]
            spkid = data_dict["spkid"]

            wav_lengths = data_dict["wav_lengths"]

            if use_cuda:
                c = c.to(device)
                wav = wav.to(device)
                mel = mel.to(device)
                f0 = f0.to(device)
                uv = uv.to(device)
                spkid = spkid.to(device)
            c = c[:1]
            wav = wav[:1]
            mel = mel[:1]
            f0 = f0[:1]
            spkid = spkid[:1]
            y_hat, y_harm, y_noise, _ = generator.infer(c, f0=f0, uv=uv, g=spkid)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.n_fft,
                hps.data.acoustic_dim,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_size,
                hps.data.fmin,
                hps.data.fmax
            )
            image_dict.update({
                "gen/mel_%d" % batch_idx: utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                "gt/mel_%d" % batch_idx: utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
            })
            audio_dict.update({
                "gen/audio_%d" % batch_idx: y_hat[0, :, :],
                # upstream verbatim: the harm/noise decomposition tracks (keys
                # constant across batches like upstream — last batch survives)
                "gen/harm": y_harm[0, :, :],
                "gen/noise": y_noise[0, :, :],
                "gt/audio_%d" % batch_idx: wav[0, :, :wav_lengths[0]],
            })

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()
