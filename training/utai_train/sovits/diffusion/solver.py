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
#     ignores the _best postfix, so it never hijacks resume — measured S118: it
#     is not merely ignored, it is UNREACHABLE, and an expdir holding only
#     model_best.pt dies with FileNotFoundError)
#   - ★S118 §F8⒜, the S117 §F2⒜ generalization: (a) the run state no checkpoint
#     holds — GradScaler scale, RNG, dataset identity — is captured into the
#     snapshots' state.json and restored on resume; (b) the BEST point gets a
#     RESUMABLE snapshot (resume_best/model.pt, WITH optimizer) beside the
#     inference-only model_best.pt, guarded by numerics.best_save_is_safe +
#     optimizer_state_is_safe; (c) a rolling resume_latest/model.pt is refreshed
#     at every validation and at stop/final, because upstream's save_opt: false
#     makes every periodic checkpoint optimizer-less and load_model restores
#     such a file in total silence; (d) diagnostic mode also drives this loop's
#     log_interval and prints the scaler's scale + skip flag.
#     The numbered model_<step>.pt grid is byte-identical to before.
import json
import logging
import math
import os
import re
import time

import librosa
import numpy as np
import torch

from ... import diag
from ... import device as device_shim  # aliased for consistency + collision-safety (see rvc/sovits trainers)
from ... import numerics
from ... import resume_state
from .logger import utils
from .logger.saver import Saver

logger = logging.getLogger(__name__)


def is_milestone(step, interval_force_save):
    """THE survivor predicate — shared by the delete sweep and the protocol
    report so they can never drift apart."""
    return step % interval_force_save == 0


def _sweep_old_checkpoints(saver, args, current_step, written, superseded_step=None):
    """Delete the superseded non-milestone checkpoints THIS RUN is responsible for.

    Files this run did not write belong to EARLIER runs and are never touched: they were kept
    under whatever interval_force_save grid those runs used, and re-judging them by the current
    grid would silently delete historical milestones when the user changes the interval between
    runs (review F2). model_0.pt (the seeded base) and the un-numbered model_best.pt are always
    exempt.

    ⛔★S118 §F8⒜ — "responsible for" USED to be expressed as the RANGE
    ``floor_step <= step < current_step``, and that was only equivalent to "ours" while
    ``initial_global_step`` was guaranteed to be the HIGHEST step on disk. Resuming from the best
    snapshot breaks exactly that guarantee, and the range form then eats the abandoned branch one
    file at a time as the new run marches past their numbers — measured by driving THIS function:
    with the resume at 1400 and 0/700/1400/2100/2800/3500/5000 on disk under
    interval_force_save=1400, model_2100 died at the 2800 save, model_3500 at 4200 and
    model_5000 — the abandoned tip, and the only file there carrying an optimizer — at 5600.
    So the set is now EXPLICIT: the steps this run wrote, plus ``superseded_step``.

    ``superseded_step`` = the numbered save this run's own first save replaces, i.e. the step we
    resumed AT when we resumed from the grid or from the rolling snapshot (deleting it was and
    remains deliberate: our newer save carries strictly more). It is None on a REWIND, because
    then the files at and above that step describe a branch we are walking away from, and
    destroying someone else's branch is not this sweep's job.
    """
    ours = set(int(s) for s in (written or ()))
    if superseded_step:
        ours.add(int(superseded_step))
    for name in os.listdir(saver.expdir):
        m = re.fullmatch(r"model_(\d+)\.pt", name)
        if not m:
            continue
        step = int(m.group(1))
        if step == 0 or step >= current_step or step not in ours:
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
          reporter=None, stop=None, total_steps=None, best_state=None,
          resumed=None, ws_dir=None, superseded_step=None):
    """Deviation surface (see file header): reporter/stop/total_steps/
    best_state/resumed/ws_dir are OUR harness hooks; passing None for all of
    them runs the loop with upstream semantics (the loss-trajectory gate does
    exactly that aside from total_steps as the cutoff).

    `resumed` is the `state.json` of the snapshot this run continued from, or None
    (`diff_pipeline.load_start_state` decides); `ws_dir` is the WORKSPACE, which is
    the expdir's parent and the only place `dataset.fingerprint` lives;
    `superseded_step` is None on a REWIND and otherwise the resumed step — see
    `_sweep_old_checkpoints`, where getting this wrong destroys an earlier branch."""
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
    #: The numbered saves THIS run wrote — the sweep's authority over what it may delete.
    written_steps = set()

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
    dataset_items = None
    try:
        dataset_items = len(loader_train.dataset)
    except (AttributeError, TypeError):
        pass
    model.train()
    saver.log_info('======= start training =======')
    scaler = device_shim.make_scaler(args.device, True)

    # ★S118 §F8⒜ — everything a resume needs that NO diffusion checkpoint has ever carried:
    # the GradScaler's loss scale, the RNG streams, and which dataset the checkpoint was
    # trained on. Measured on the GAN side (S117): losing the scale alone makes the resumed
    # trajectory diverge by 8.35e-04 against a noise floor of exactly 0, and silently skips the
    # handful of updates it takes for the scale to climb back down from 65536.
    # ⚠ HONEST BOUND: this does NOT make a diffusion resume bit-identical to never having
    # stopped, and it cannot — upstream's loop restarts `for epoch in range(...)` from zero, so
    # the DATA order begins again regardless. What it fixes is that every resume used to replay
    # the SAME augmentation/dropout/timestep noise stream from `seed`, over and over.
    def capture_state(epoch):
        return resume_state.capture(
            scaler, epoch=epoch, global_step=saver.global_step,
            exp_dir=ws_dir, dataset_items=dataset_items, loader_len=num_batches,
        )

    def refresh_resume_point(epoch):
        """Rewrite the rolling COMPLETE resume point — see `resume_state.LATEST_DIR` for why the
        numbered grid cannot be one (upstream's `save_opt: false`)."""
        resume_state.save_solo_snapshot(
            args.env.expdir, resume_state.LATEST_DIR,
            lambda path: saver.save_model_to(path, model, optimizer),
            blob=capture_state(epoch),
        )

    resume_state.restore(resumed, scaler, logger)
    if resumed is not None:
        resume_state.report_drift(
            resumed, reporter, logger,
            exp_dir=ws_dir, dataset_items=dataset_items, loader_len=num_batches,
        )
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
                with device_shim.autocast(args.device, dtype=dtype):
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
            if saver.global_step % diag.log_interval(args.train.interval_log) == 0:
                # ★S118 §4/§F8⒜ — diagnostic mode collapses the sampling blind spot to zero here
                # too, and prints the one thing no log has ever carried: the GradScaler's scale
                # and whether torch SKIPPED this step (a skipped step trains nothing while
                # emitting a perfectly normal, finite loss line). Empty string when the mode is
                # off, so a normal run's log stays byte-identical.
                # ⚠ No gradient norms here, unlike the three GAN trainers: this loop never calls
                # `clip_grad_value_`, so there is no norm already computed to print — and adding
                # a per-parameter `.item()` would be a real per-step cost, not a free read.
                if diag.enabled():
                    logger.info("diag step=%s%s", saver.global_step, diag.scaler_note(scaler))
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
                _sweep_old_checkpoints(saver, args, saver.global_step, written_steps,
                                       superseded_step=superseded_step)
                written_steps.add(saver.global_step)

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
                    # ★S118 §F8⒜ — the two guards the three GAN trainers have had since S114/S117.
                    # ⛔ The metric must NOT advance when the write is refused: the file on disk
                    # still describes the OLD metric, so advancing without writing would make
                    # best_state.json lie in the other direction.
                    # ⚠ This loop is NOT the stale-EMA shape the guard was written for (its metric
                    # is the live test_loss, so a poisoned model normally yields a nan that loses
                    # the comparison) — but `torch.isnan(loss)` above does not catch -inf, and a
                    # -inf metric WINS every comparison. One write destroys the file.
                    # ⛔ …and the METRIC has to be finite too, which is a separate hole from the
                    # weights. `json.dump` writes a bare `NaN`, `json.load` accepts it back, and
                    # every later `test_loss < nan` is False — so ONE nan validation on a
                    # workspace that had no best yet (`best_metric is None` wins the comparison
                    # above) persists `{"metric": NaN}` and FREEZES best tracking for that
                    # workspace forever, in this run and every future one, with a
                    # well-formed-looking file and not one error. Measured S118.
                    if math.isfinite(test_loss) and numerics.best_save_is_safe(model.state_dict(), logger):
                        best_metric = test_loss
                        best_step = saver.global_step
                        best_path = saver.save_model(model, None, postfix='best')
                        # …and a RESUMABLE snapshot beside it. `model_best.pt` is written with
                        # `optimizer=None`, i.e. it is an inference-only dead end — the exact
                        # thing tproject.rs already says about it ("never a resume point … would
                        # rewind thousands of steps AND zero the AdamW momentum"). The extra
                        # guard is not decoration: a finite-but-huge gradient can leave the
                        # moments at inf while every weight is still finite (S117, measured), and
                        # such a checkpoint is worse than useless as a rollback target.
                        if numerics.optimizer_state_is_safe((("diffusion", optimizer),), logger):
                            resume_state.save_solo_snapshot(
                                args.env.expdir, resume_state.BEST_DIR,
                                lambda path: saver.save_model_to(path, model, optimizer),
                                blob=capture_state(epoch), metric=best_metric,
                            )
                        with open(best_state_path, 'w', encoding='utf-8') as f:
                            json.dump({'metric': best_metric, 'step': best_step}, f)
                        if reporter is not None:
                            reporter.ckpt('best', best_path, best_step, epoch, metric=best_metric)

                # ★S118 §F8⒜ — refresh the rolling resume point, AFTER test(): a resume from here
                # continues with the batch that would have followed the validation, and test()
                # itself draws from the RNG (diffusion sampling + the per-sample loss draws), so
                # capturing before it would restore a stream the never-stopped run had already
                # advanced past.
                refresh_resume_point(epoch)

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
        # ★S118 §F8⒜ — the numbered stop save DOES carry the optimizer, but nothing else carries
        # the scale/RNG/dataset identity, so the rolling snapshot has to be refreshed here too.
        # It lands on the same step, and the chooser breaks that tie towards the snapshot
        # precisely because it is the one with the complete state.
        refresh_resume_point(epoch)
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
        refresh_resume_point(epoch)   # ★S118 §F8⒜ — same reason as the stop branch above
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
