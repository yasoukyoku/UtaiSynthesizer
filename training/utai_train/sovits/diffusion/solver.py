# Vendored from so-vits-svc 4.1-Stable diffusion/solver.py (@ 730930d).
# The training math is UNCHANGED: same loop order (global_step_increment at
# batch start -> zero_grad -> device moves -> forward loss (fp32 vs autocast)
# -> nan raise -> backward/step -> scheduler.step()), same interval_val save
# cadence with delete-previous-non-milestone, same test() (full-inference RTF
# pass + per-val-sample loss loop averaged over batch_size draws + TB
# spec/audio dumps — its torch/python RNG consumption is part of the verbatim
# trajectory, incl. the LAZY NsfHifiGAN weight-norm Generator construction on
# the FIRST vocoder.infer call; do NOT "optimize" that into a preload).
# Deviations (deliberate; loss trajectory gated against the unmodified
# upstream solver):
#   - print() -> logging (stdout belongs exclusively to the JSONL protocol)
#   - protocol integration: reporter.step per training step (throttled
#     Rust-side display only), reporter.ckpt for milestone/best/stop/final
#   - graceful stop: the stop flag is polled at every batch start; on stop the
#     current state is saved (WITH optimizer — resume must not lose the AdamW
#     moments) and the loop exits cleanly (upstream could only be killed)
#   - completion = total_steps (our UI thinks in steps like upstream's
#     intervals do; the yaml epochs stays the upstream 100000 sentinel):
#     when global_step reaches total_steps the loop saves a final checkpoint
#     (WITH optimizer — 加练 resumes from it) and returns
#   - periodic checkpoints follow upstream save_opt (template false, no
#     optimizer) and are reported to the UI ONLY at milestone steps
#     (step % interval_force_save == 0, the survivors of upstream's
#     delete-previous rule; interval_force_save is normalized Rust-side to a
#     multiple of interval_val so the milestone grid == the survivor grid) —
#     reporting the in-between saves would hand the UI paths that the very
#     next save deletes
#   - the delete-previous rule is widened to a sweep: every val-save deletes
#     ALL non-milestone numbered model_<step>.pt below the current step
#     (except model_0.pt) — upstream only deletes step-interval_val exactly,
#     which strands stop/final saves from earlier runs on disk forever
#   - best tracking on the REAL validation loss (test_loss): model_best.pt +
#     diffusion best_state.json survive resume (load_model's numeric scan
#     ignores the _best postfix, so it never hijacks resume)
import json
import logging
import os
import re
import time

import librosa
import numpy as np
import torch
from torch import autocast
from torch.cuda.amp import GradScaler

from .logger import utils
from .logger.saver import Saver

logger = logging.getLogger(__name__)


def is_milestone(step, interval_force_save):
    """THE survivor predicate — shared by the delete sweep and the protocol
    report so they can never drift apart."""
    return step % interval_force_save == 0


def _sweep_old_checkpoints(saver, args, current_step, floor_step):
    """Delete THIS RUN's superseded non-milestone checkpoints: floor_step <=
    step < current_step (floor = the resume source's step — stale once the
    first new save lands). Files below the floor belong to EARLIER runs and
    are never touched: they were kept under whatever interval_force_save grid
    those runs used, and re-judging them by the current grid would silently
    delete historical milestones when the user changes the interval between
    runs (review F2). model_0.pt (the seeded base) and the un-numbered
    model_best.pt are always exempt."""
    for name in os.listdir(saver.expdir):
        m = re.fullmatch(r"model_(\d+)\.pt", name)
        if not m:
            continue
        step = int(m.group(1))
        if step == 0 or step < floor_step or step >= current_step:
            continue
        if is_milestone(step, args.train.interval_force_save):
            continue
        saver.delete_model(postfix=str(step))


def test(args, model, vocoder, loader_test, saver):
    logger.info(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0].split("/")[-1]
            speaker = data['name'][0].split("/")[-2]
            logger.info('--------')
            logger.info('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            logger.info('>> %s', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                    data['units'],
                    data['f0'],
                    data['volume'],
                    data['spk_id'],
                    gt_spec=None if model.k_step_max == model.timesteps else data['mel'],
                    infer=True,
                    infer_speedup=args.infer.speedup,
                    method=args.infer.method,
                    k_step=model.k_step_max
                    )
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()

            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            logger.info('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)

            # loss
            for i in range(args.train.batch_size):
                loss = model(
                    data['units'],
                    data['f0'],
                    data['volume'],
                    data['spk_id'],
                    gt_spec=data['mel'],
                    infer=False,
                    k_step=model.k_step_max)
                test_loss += loss.item()

            # log mel
            saver.log_spec(f"{speaker}_{fn}.wav", data['mel'], mel)

            # log audi
            path_audio = data['name_ext'][0]
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({f"{speaker}_{fn}_gt.wav": audio,f"{speaker}_{fn}_pred.wav": signal})
    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches

    # check
    logger.info(' [test_loss] test_loss: %s', test_loss)
    logger.info(' Real Time Factor %s', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, scheduler, vocoder,
          loader_train, loader_test,
          reporter=None, stop=None, total_steps=None, best_state=None):
    """Deviation surface (see file header): reporter/stop/total_steps/
    best_state are OUR harness hooks; passing None for all of them runs the
    loop with upstream semantics (the loss-trajectory gate does exactly that
    aside from total_steps as the cutoff)."""
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # best-so-far validation loss, survives resume (deviation)
    best_metric = None
    best_step = None
    best_state_path = None
    if best_state is not None:
        best_state_path = os.path.join(args.env.expdir, 'best_state.json')
        if os.path.exists(best_state_path):
            try:
                with open(best_state_path, encoding='utf-8') as f:
                    prev = json.load(f)
                best_metric = float(prev['metric'])
                best_step = int(prev['step'])
            except Exception:
                logger.warning('best_state.json unreadable, starting best tracking fresh')

    stopped = False
    finished = False
    final_path = None
    steps_this_run = 0
    last_test_loss = None

    def report_step(epoch, force=False, extra_losses=None, empty=False):
        if reporter is None:
            return
        losses = {} if empty else {'loss': float(current_loss)}
        if extra_losses:
            losses.update(extra_losses)
        reporter.step(
            saver.global_step,
            int(total_steps) if total_steps else 0,
            epoch,
            0,  # diffusion epochs are a sentinel unit — the UI hides them (A8)
            float(optimizer.param_groups[0]['lr']),
            losses,
            force=force,
        )

    def save_with_optimizer(postfix):
        # stop/final = resume state: ALWAYS carry the optimizer (deviation)
        return saver.save_model(model, optimizer, postfix=postfix)

    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    saver.log_info("epoch|batch_idx/num_batches|output_dir|batch/s|lr|time|step")
    current_loss = 0.0
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            # graceful stop BEFORE the step counts (deviation)
            if stop is not None and stop.requested():
                stopped = True
                logger.info('stop requested at step %s', saver.global_step)
                break
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)

            # forward
            if dtype == torch.float32:
                loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'],
                                aug_shift = data['aug_shift'], gt_spec=data['mel'].float(), infer=False, k_step=model.k_step_max)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    loss = model(data['units'], data['f0'], data['volume'], data['spk_id'],
                                    aug_shift = data['aug_shift'], gt_spec=data['mel'], infer=False, k_step=model.k_step_max)

            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

            current_loss = loss.item()
            steps_this_run += 1
            report_step(epoch)

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        current_loss,
                        saver.get_total_time(),
                        saver.global_step
                    )
                )

                saver.log_value({
                    'train/loss': current_loss
                })

                saver.log_value({
                    'train/lr': current_lr
                })

            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None

                # save latest
                path = saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                # widened upstream delete-previous rule (see header)
                _sweep_old_checkpoints(saver, args, saver.global_step, initial_global_step)

                # run testing set
                test_loss = test(args, model, vocoder, loader_test, saver)
                last_test_loss = test_loss
                model.train()

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })

                if reporter is not None and is_milestone(saver.global_step, args.train.interval_force_save):
                    reporter.ckpt('periodic', path, saver.global_step, epoch, metric=test_loss)

                # best on the REAL validation loss (deviation)
                if best_state is not None and (best_metric is None or test_loss < best_metric):
                    best_metric = test_loss
                    best_step = saver.global_step
                    best_path = saver.save_model(model, None, postfix='best')
                    with open(best_state_path, 'w', encoding='utf-8') as f:
                        json.dump({'metric': best_metric, 'step': best_step}, f)
                    if reporter is not None:
                        reporter.ckpt('best', best_path, best_step, epoch, metric=best_metric)

                # surface the val loss on the live curve
                report_step(epoch, force=True, extra_losses={'val': float(test_loss)})

            # completion by total_steps (deviation, see header)
            if total_steps is not None and saver.global_step >= int(total_steps):
                finished = True
                logger.info('reached total_steps %s at step %s', total_steps, saver.global_step)
                break
        # /batches

        if stopped or finished:
            break

    if stopped and steps_this_run > 0:
        final_path = save_with_optimizer(str(saver.global_step))
        if reporter is not None:
            reporter.ckpt('stop', final_path, saver.global_step, 0, metric=last_test_loss)
    elif steps_this_run > 0:
        # finished by total_steps — OR the 100000-epoch sentinel ran out first
        # (tiny datasets: batches/epoch * 100000 < total_steps). Either way the
        # run trained everything it could and must leave a resumable final
        # checkpoint WITH optimizer, not silently skip it (review F4/F7).
        if not finished:
            logger.warning(
                'epoch sentinel (%s) exhausted at step %s before total_steps %s',
                args.train.epochs, saver.global_step, total_steps,
            )
        final_path = save_with_optimizer(str(saver.global_step))
        if reporter is not None:
            reporter.ckpt('final', final_path, saver.global_step, 0, metric=last_test_loss)

    # emit the last step un-throttled so the UI progress reaches the end
    # (EMPTY losses — a duplicate same-step data point must not land on the
    # curve; same policy as the SoVITS trainer)
    if reporter is not None and steps_this_run > 0:
        report_step(0, force=True, empty=True)

    saver.writer.close()
    return {
        'stopped': stopped,
        'steps': saver.global_step,
        'steps_this_run': steps_this_run,
        'final_weight': final_path,
        'best_weight': (os.path.join(args.env.expdir, 'model_best.pt')
                        if best_metric is not None else None),
        'best_metric': best_metric,
        'best_step': best_step,
        'last_val_loss': last_test_loss,
    }
