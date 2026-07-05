# Vendored from RVC 20240604 infer/modules/train/train.py.
# The training math is UNCHANGED: same dataset/bucket-sampler/collate, same model
# construction (is_half=fp16_run, sr string), same AdamW/ExponentialLR(last_epoch=
# epoch_str-2), same GradScaler/autocast step order (D then G), same loss
# composition (gen + fm + mel*c_mel + kl*c_kl), same clip_grad_value_(None), same
# resume-over-pretrained priority, same per-epoch save cadence and G_2333333
# latest-naming, same GPU batch cache (if_cache_data_in_gpu).
# Deviations (deliberate; verified by the loss-trajectory gate vs upstream):
#   - single-process single-GPU: no mp.spawn/DDP (CUDA_VISIBLE_DEVICES is set by
#     the runner, so cuda:0 is the selected card; multi-GPU DDP is a backlog item)
#   - no ipex/XPU path, no nof0 branch (nof0 is CLOSED project-wide)
#   - stdout JSONL protocol per step (raw un-clamped loss values; the original's
#     <=75/<=9 clamps were display-only and are kept for tensorboard exactly)
#   - graceful stop: stop flag checked every step -> save latest G/D + a small
#     model snapshot, report, and exit cleanly (upstream could only be killed)
#   - best tracking: EMA(loss_mel) checked at epoch end -> <name>_best.pth
#     (heuristic — GAN losses are not comparable across steps, mel is the closest
#     perceptual proxy; diffusion/vocoder objects get true val-loss best instead)
#   - on natural completion also saves latest G/D (upstream's resume point was
#     the last save_every_epoch boundary, losing tail epochs)
#   - normal termination emits the protocol "done" (upstream: os._exit(2333333))
import datetime
import json
import logging
import math
import os
from random import shuffle
from time import time as ttime

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import train_utils as utils
from .infer_pack import commons
from .data_utils import (
    DistributedBucketSampler,
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .process_ckpt import savee

logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# EMA over ~100 steps; mel is logged every step so this smooths GAN noise enough
# to compare epochs without lagging real improvements too far behind
BEST_EMA_ALPHA = 2.0 / (100.0 + 1.0)


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def build_hps(cfg, exp_dir):
    hps = utils.get_hparams_from_file(os.path.join(exp_dir, "config.json"))
    hps.model_dir = hps.experiment_dir = exp_dir
    hps.name = cfg["model_slug"]
    hps.save_every_epoch = int(cfg.get("save_every_epoch", 5))
    hps.total_epoch = int(cfg["total_epoch"])
    hps.pretrainG = cfg.get("pretrain_g", "") or ""
    hps.pretrainD = cfg.get("pretrain_d", "") or ""
    hps.version = cfg["version"]
    hps.train.batch_size = int(cfg["batch_size"])
    hps.sample_rate = cfg["sample_rate"]  # "32k"|"40k"|"48k" string, as upstream
    hps.if_f0 = 1
    hps.if_latest = 1 if cfg.get("keep_only_latest", True) else 0
    hps.save_every_weights = "1" if cfg.get("save_every_weights", True) else "0"
    hps.if_cache_data_in_gpu = 1 if cfg.get("cache_gpu", False) else 0
    hps.data.training_files = os.path.join(exp_dir, "filelist.txt")
    if not torch.cuda.is_available():
        # CPU fallback (RVC only): amp/GradScaler are CUDA-only
        hps.train.fp16_run = False
    return hps


def train(cfg, exp_dir, reporter, stop):
    hps = build_hps(cfg, exp_dir)
    global_step = 0

    if hps.version == "v1":
        from .infer_pack.models import MultiPeriodDiscriminator
        from .infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    else:
        from .infer_pack.models import MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator
        from .infer_pack.models import SynthesizerTrnMs768NSFsid as RVC_Model_f0

    torch.manual_seed(hps.train.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)

    writer = SummaryWriter(log_dir=hps.model_dir)

    train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=1,
        rank=0,
        shuffle=True,
    )
    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    net_g = RVC_Model_f0(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        is_half=hps.train.fp16_run,
        sr=hps.sample_rate,
    )
    if use_cuda:
        net_g = net_g.cuda()
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if use_cuda:
        net_d = net_d.cuda()
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        logger.info("loaded D")
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # 如果首次不能加载，加载pretrain
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            logger.info("loaded pretrained %s" % (hps.pretrainG))
            logger.info(
                net_g.load_state_dict(
                    torch.load(hps.pretrainG, map_location="cpu")["model"]
                )
            )
        if hps.pretrainD != "":
            logger.info("loaded pretrained %s" % (hps.pretrainD))
            logger.info(
                net_d.load_state_dict(
                    torch.load(hps.pretrainD, map_location="cpu")["model"]
                )
            )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    weights_dir = os.path.join(exp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    total_steps = hps.total_epoch * len(train_loader)
    ema_mel = None
    best_metric = None
    best_step = None
    best_path = os.path.join(weights_dir, "%s_best.pth" % hps.name)
    # best survives resume — otherwise the first improvement of a new run would
    # overwrite a better <name>_best.pth from the previous run
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
    cache = []

    def save_best(epoch):
        save_small("%s_best" % hps.name, epoch)
        with open(best_state_path, "w", encoding="utf-8") as f:
            json.dump({"metric": best_metric, "step": best_step}, f)
        reporter.ckpt("best", best_path, best_step, epoch, metric=best_metric)

    def save_gd(epoch):
        if hps.if_latest == 0:
            g_path = os.path.join(hps.model_dir, "G_{}.pth".format(global_step))
            d_path = os.path.join(hps.model_dir, "D_{}.pth".format(global_step))
        else:
            g_path = os.path.join(hps.model_dir, "G_{}.pth".format(2333333))
            d_path = os.path.join(hps.model_dir, "D_{}.pth".format(2333333))
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)

    def save_small(name, epoch):
        path = os.path.join(weights_dir, "%s.pth" % name)
        savee(net_g.state_dict(), hps.sample_rate, hps.if_f0, path, epoch, hps.version, hps)
        return path

    for epoch in range(epoch_str, hps.total_epoch + 1):
        last_epoch = epoch
        train_sampler.set_epoch(epoch)
        net_g.train()
        net_d.train()

        # Prepare data iterator (upstream GPU-cache branch preserved verbatim)
        if hps.if_cache_data_in_gpu == True:
            data_iterator = cache
            if cache == []:
                for batch_idx, info in enumerate(train_loader):
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                    if use_cuda:
                        phone = phone.cuda(0, non_blocking=True)
                        phone_lengths = phone_lengths.cuda(0, non_blocking=True)
                        pitch = pitch.cuda(0, non_blocking=True)
                        pitchf = pitchf.cuda(0, non_blocking=True)
                        sid = sid.cuda(0, non_blocking=True)
                        spec = spec.cuda(0, non_blocking=True)
                        spec_lengths = spec_lengths.cuda(0, non_blocking=True)
                        wave = wave.cuda(0, non_blocking=True)
                        wave_lengths = wave_lengths.cuda(0, non_blocking=True)
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
            else:
                shuffle(cache)
        else:
            data_iterator = enumerate(train_loader)

        epoch_recorder = EpochRecorder()
        for batch_idx, info in data_iterator:
            # stop BEFORE the first step of this run counts as "nothing trained"
            if stop.requested():
                stopped = True
                logger.info("stop requested before step %s", global_step)
                break
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
            if (hps.if_cache_data_in_gpu == False) and use_cuda:
                phone = phone.cuda(0, non_blocking=True)
                phone_lengths = phone_lengths.cuda(0, non_blocking=True)
                pitch = pitch.cuda(0, non_blocking=True)
                pitchf = pitchf.cuda(0, non_blocking=True)
                sid = sid.cuda(0, non_blocking=True)
                spec = spec.cuda(0, non_blocking=True)
                spec_lengths = spec_lengths.cuda(0, non_blocking=True)
                wave = wave.cuda(0, non_blocking=True)

            with autocast(enabled=hps.train.fp16_run):
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                with autocast(enabled=False):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                if hps.train.fp16_run == True:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments(
                    wave, ids_slice * hps.data.hop_length, hps.train.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            # protocol carries the RAW values; the <=75/<=9 clamps below stay
            # display-only exactly like upstream
            raw_mel = float(loss_mel)
            # a single non-finite step must not poison the EMA forever
            if math.isfinite(raw_mel):
                ema_mel = raw_mel if ema_mel is None else (
                    BEST_EMA_ALPHA * raw_mel + (1 - BEST_EMA_ALPHA) * ema_mel
                )
            lr = optim_g.param_groups[0]["lr"]
            reporter.step(
                global_step,
                total_steps,
                epoch,
                hps.total_epoch,
                lr,
                {
                    "g_total": float(loss_gen_all),
                    "d_total": float(loss_disc),
                    "gen": float(loss_gen),
                    "fm": float(loss_fm),
                    "mel": raw_mel,
                    "kl": float(loss_kl),
                },
            )

            if global_step % hps.train.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9
                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl}
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
            global_step += 1
            steps_this_run += 1

            if stop.requested():
                stopped = True
                logger.info("stop requested at epoch %s step %s", epoch, global_step)
                break
        # /Run steps

        if stopped:
            if steps_this_run > 0:
                save_gd(epoch)
                path = save_small("%s_e%s_s%s" % (hps.name, epoch, global_step), epoch)
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

        if epoch % hps.save_every_epoch == 0:
            save_gd(epoch)
            if hps.save_every_weights == "1":
                path = save_small("%s_e%s_s%s" % (hps.name, epoch, global_step), epoch)
                reporter.ckpt("periodic", path, global_step, epoch)

        if (
            ema_mel is not None
            and math.isfinite(ema_mel)
            and (best_metric is None or ema_mel < best_metric)
        ):
            best_metric = ema_mel
            best_step = global_step
            save_best(epoch)

        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))

        # upstream exits before the scheduler step on the final epoch — preserve
        # that so a later "continue training" resume sees the same optimizer lr
        if epoch >= hps.total_epoch:
            logger.info("Training is done. The program is closed.")
            save_gd(epoch)
            final_path = save_small(hps.name, epoch)
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
            hps.total_epoch,
            optim_g.param_groups[0]["lr"],
            {},
            force=True,
        )
    writer.close()

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
