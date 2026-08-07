"""The part of a resume that `save_checkpoint` does NOT carry — one source for all three trainers.

WHY THIS EXISTS (measured, S117 §F2⒜; the open question was
`project_v2_resume_divergence_open` §3-1).

S116 established that a healthy resume restores the model weights, every Adam moment, ``step``,
``param_groups``, ``lr`` and ``initial_lr`` bit-identically. That list is complete for what
``save_checkpoint`` writes — and that is exactly the hole: under fp16 the run also carries a
``GradScaler`` whose loss scale appears in no checkpoint, so **every** resume restarts it at
torch's ``init_scale=65536``.

Measured on this machine (RTX 3080 Ti, torch 2.5.1+cu121) driving the real
``device.make_scaler`` / ``train_utils.save_checkpoint`` / ``load_checkpoint`` /
``commons.clip_grad_value_`` in ``rvc/train.py``'s step order, 60 warm-up + 40 compared steps:

===========================================  ==============  ==================
arm                                          max |Δweight|   updates skipped
===========================================  ==============  ==================
same arm run twice (the noise floor)         0.0             0
resume as shipped today (scaler lost)        8.35e-04        5
resume with the scaler state restored        0.0             0
===========================================  ==============  ==================

The floor being exactly 0 is what makes the middle row a signal rather than a rounding story,
and the third row is the positive control: restoring the scaler removes the divergence exactly.

**The skipped updates are the concrete damage.** While the scale climbs back down from 65536 it
overflows; ``GradScaler.step`` then skips the optimizer, so those batches train nothing. Nothing
reports it: the reported LOSSES stay finite through the whole recalibration (the overflow is in
the gradients, not in the forward pass), so ``numerics.DivergenceGuard`` — which observes
reported losses — neither fires nor notices. That answers both halves of §3-1: no false positive,
and no visibility. `` grad_norm`` does go non-finite on exactly those steps, and today it is
computed and thrown away except for a tensorboard scalar.

RNG is captured for the same reason and with much weaker evidence: it changes the trajectory
(sovits' ``__getitem__`` draws from the global ``random``, dropout draws from the CUDA
generator), not the stability. It is restored because "resume == never stopped" is the property
worth having, and it costs ~11 KB.

⚠ WHY A SIDECAR FILE AND NOT A KEY INSIDE THE CHECKPOINT. Putting the blob in the ``G_*.pth``
dict would make it atomic with the checkpoint for free, but reading it back means either a second
``torch.load`` of a 400 MB file or changing the return signature of three vendored
``load_checkpoint`` copies. A sidecar also has to be readable by Rust — the archive inventory
(`tproject::scan_project_ckpts`) must be able to see and describe a resume point, or it becomes
"a file nothing in the UI can see or reclaim", which is the exact failure that inventory exists
to end. So: plain JSON, written LAST (its presence is the completion marker for the pair before
it), and validated against the checkpoint's own epoch on the way back in — a sidecar left behind
by a crash between the two writes is IGNORED, loudly, never trusted.
"""

import json
import os
import random

#: Bumped when the meaning of a field changes. An unknown (higher) schema is ignored rather than
#: guessed at — a resume that silently half-understands its own state file is worse than one that
#: falls back to today's behaviour and says so.
SCHEMA = 1

#: The sidecar that belongs to the rolling `G_*/D_*` pair in the model dir.
LATEST_NAME = "resume_state.json"

#: Directory holding the resumable snapshot of the BEST point. It is a SUBDIRECTORY on purpose:
#: five separate consumers walk the model dir looking for `G_*`/`D_*` and every one of them would
#: mis-handle a best pair parked there —
#:   1. `sovits/train.py` parses global_step out of the D filename → `D_best.pth` = ValueError
#:      → the whole resume is refused;
#:   2. `sovits/utils.py:clean_checkpoints` deletes all but the newest N `G*`/`D*` → it would
#:      delete the best pair;
#:   3. `rvc/train_utils.py:latest_checkpoint_path` sorts by (mtime, digits) → a best written
#:      after the periodic save becomes "the latest" and the next resume silently continues from
#:      it instead;
#:   4. `sovits/utils.py:latest_checkpoint_path` sorts by the digits of the WHOLE PATH → a name
#:      with no digits contributes only the directory's;
#:   5. `ckpt_guard.resume_was_intended` globs `G_*.pth`/`D_*.pth` → a leftover best pair alone
#:      would make an otherwise fresh directory look resumable.
BEST_DIR = "resume_best"
BEST_G = "G.pth"
BEST_D = "D.pth"
BEST_STATE = "state.json"

#: The shallow-diffusion payload: ONE file, because that trainer's checkpoint IS one file
#: (``{'global_step', 'model', 'optimizer'}``) instead of a G+D pair (S118 §F8⒜).
BEST_MODEL = "model.pt"

#: Directory holding the ROLLING complete resume point — shallow diffusion only, and it exists
#: because that trainer is the only one whose periodic checkpoint is NOT a resume point:
#: ``diffusion_template.yaml`` ships upstream's ``save_opt: false``, so every ``interval_val``
#: save is ``{global_step, model}`` with no optimizer (measured: 210.7 MB against 631.9 MB for
#: one that has it), and ``logger/utils.load_model`` restores such a file in TOTAL silence —
#: ``optim.state`` simply stays ``{}``. Our stop/final saves do carry the optimizer, so the hole
#: is exactly "the run ended ungracefully": a kill, an OOM, a power loss. Flipping ``save_opt``
#: would put 421 MB of moments into every kept MILESTONE instead (20 milestones = 12.6 GB), so
#: the resume point gets its own rolling file and the numbered archive grid stays byte-identical
#: to today. ★ User decision 2026-08-07: pay the ~632 MB, no switch — same call as §F2⒜'s.
LATEST_DIR = "resume_latest"

#: ⛔ BOTH directory names MUST be at least 6 characters, and this is not cosmetic.
#: ``logger/utils.load_model`` finds candidates with a RECURSIVE ``os.walk`` and then slices each
#: path with ``s[len(expdir + "/model_"):]`` before asking ``str.isdigit()``. Measured (S118,
#: `TESTING/s118_f8a/probe_nested_pt_arithmetic.py`): with a subdirectory name of 6+ characters
#: the discarded 7-char prefix ends INSIDE the name, so the remainder still contains the path
#: separator and can never be all-digits — the nested file contributes step 0 and is ignored.
#: With a name of 5 or fewer the separator is consumed and the FILENAME's tail is exposed:
#: ``logs/x9999.pt`` slices to ``"9999"``, and ``load_model`` then rebuilds a flat
#: ``model_9999.pt`` that does not exist and dies with ``FileNotFoundError``. ``logs/`` is 4
#: characters and IS created in production (``Saver`` puts tensorboard there) — it is only
#: dormant because tensorboard writes no ``.pt``. `gate_resume_state` pins the length.
SNAPSHOT_DIR_MIN_LEN = 6

#: Stable CODE (→ `utai_train.runner` → `backendError.ts` → all three locales) for "you are
#: resuming onto a dataset that is not the one this checkpoint was trained on". It is a WARNING,
#: never a refusal: `resume_lock.rs` classifies `dataset` as Costly, i.e. changing it is
#: legitimate and merely re-runs preprocessing. Refusing here would contradict that table.
CODE_DATASET_CHANGED = "TRAINING_RESUME_DATASET_CHANGED"

#: Stable CODE for "this resume could not restore the optimizer's momentum, because the archive it
#: continues from does not contain any" (shallow diffusion only — the three GANs always write it).
#: Also a WARNING, never a refusal: continuing without the moments is far better than refusing to
#: continue at all, and it is the ONLY option a user with a pre-S118 workspace has. It earns a
#: user-visible line because the cost is real and measurable (S118: the first post-resume update
#: is ±lr for every parameter at once, 4.7x-25.5x the warm optimizer's, and the second moment
#: takes ~1700-2000 steps to recover) — and because it should now be RARE: `LATEST_DIR` exists so
#: that a normal run never lands here.
CODE_OPTIMIZER_NOT_RESTORED = "TRAINING_RESUME_OPTIMIZER_MISSING"


class RestoreReport:
    """What `restore` actually managed to do — every field is reported, none is assumed.

    The trainer logs this verbatim. A resume that silently failed to restore half of its state
    is precisely the class of bug this module exists to close, so "I did not restore X" has to
    be as visible as "I did".
    """

    def __init__(self):
        self.scaler = None      # None = nothing to restore | "restored" | "<why not>"
        self.rng = {}           # generator name -> "restored" | "<why not>"
        self.notes = []         # free-form, user-visible through the log

    def line(self):
        rng = ", ".join("%s=%s" % (k, v) for k, v in sorted(self.rng.items())) or "none"
        return "resume state: scaler=%s; rng[%s]%s" % (
            self.scaler if self.scaler is not None else "absent",
            rng,
            ("; " + "; ".join(self.notes)) if self.notes else "",
        )


def _hex(byte_tensor):
    """A torch uint8 RNG-state tensor as hex. Both torch generators hand back exactly that."""
    return byte_tensor.cpu().numpy().tobytes().hex()


def _unhex(text):
    import numpy as np
    import torch

    return torch.from_numpy(np.frombuffer(bytes.fromhex(text), dtype=np.uint8).copy())


def _capture_rng():
    import torch

    out = {"torch_cpu": _hex(torch.get_rng_state())}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # device 0 only: the runner masks visibility to exactly one card (device.setup_visibility),
        # so `get_rng_state_all` would record a list whose length is a property of the machine.
        out["torch_cuda"] = _hex(torch.cuda.get_rng_state())
    st = random.getstate()
    out["python"] = [st[0], list(st[1]), st[2]]
    try:
        import numpy as np

        ns = np.random.get_state()  # ("MT19937", uint32[624], pos, has_gauss, cached_gaussian)
        out["numpy"] = [
            ns[0],
            ns[1].astype("<u4").tobytes().hex(),
            int(ns[2]),
            int(ns[3]),
            float(ns[4]),
        ]
    except Exception:
        pass
    return out


def _restore_rng(blob, report):
    import torch

    for key, fn in (
        ("torch_cpu", lambda v: torch.set_rng_state(_unhex(v))),
        ("torch_cuda", lambda v: torch.cuda.set_rng_state(_unhex(v))),
    ):
        if key not in blob:
            continue
        try:
            fn(blob[key])
            report.rng[key] = "restored"
        except Exception as e:  # a checkpoint from a CUDA box resumed on CPU, etc.
            report.rng[key] = "%s: %s" % (type(e).__name__, e)
    if "python" in blob:
        try:
            v = blob["python"]
            random.setstate((v[0], tuple(int(x) for x in v[1]), v[2]))
            report.rng["python"] = "restored"
        except Exception as e:
            report.rng["python"] = "%s: %s" % (type(e).__name__, e)
    if "numpy" in blob:
        try:
            import numpy as np

            v = blob["numpy"]
            keys = np.frombuffer(bytes.fromhex(v[1]), dtype="<u4").astype(np.uint32).copy()
            np.random.set_state((v[0], keys, int(v[2]), int(v[3]), float(v[4])))
            report.rng["numpy"] = "restored"
        except Exception as e:
            report.rng["numpy"] = "%s: %s" % (type(e).__name__, e)


def read_dataset_fingerprint(exp_dir):
    """The fingerprint `utai_train.cache.invalidate_extract_caches` last wrote for this slot.

    ⚠ It always describes the dataset as it is RIGHT NOW — preprocessing overwrites it on the way
    in. That is why it has to be copied into the checkpoint's sidecar: comparing the file against
    itself would always match.
    """
    if not exp_dir:
        # The upstream loss-trajectory gate drives the diffusion solver with no workspace at all;
        # "we do not know" has to be an answer, not a TypeError inside a resume header.
        return None
    p = os.path.join(exp_dir, "dataset.fingerprint")
    try:
        with open(p, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def capture(scaler=None, *, epoch, global_step, exp_dir=None, dataset_items=None, loader_len=None):
    """Everything a resume needs that the `G_*/D_*` pair does not already carry.

    ⛔ `global_step` is recorded for DIAGNOSIS ONLY — do not restore the step counter from it.
    A resume re-runs epoch `epoch_str` from its beginning (upstream's semantics, preserved), so
    the counter has to be the step at the START of that epoch, which is what each trainer already
    derives. Restoring a mid-epoch save's counter instead would put the reported step ahead of
    the batches actually being trained.
    """
    blob = {
        "schema": SCHEMA,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng": _capture_rng(),
    }
    if scaler is not None:
        st = scaler.state_dict()
        # A disabled GradScaler (cpu / xpu — see device.make_scaler) returns {} and refuses to
        # load one back. Record the emptiness rather than an empty dict, so the reader never has
        # to guess whether "no scaler" means "fp32 run" or "we forgot".
        blob["scaler"] = st if st else None
        blob["scaler_enabled"] = bool(st)
    if exp_dir is not None:
        blob["dataset_fingerprint"] = read_dataset_fingerprint(exp_dir)
    if dataset_items is not None:
        blob["dataset_items"] = int(dataset_items)
    if loader_len is not None:
        blob["loader_len"] = int(loader_len)
    return blob


def snapshot_dir(exp_dir, sub_dir=BEST_DIR):
    return os.path.join(exp_dir, sub_dir)


def best_dir(exp_dir):
    return snapshot_dir(exp_dir, BEST_DIR)


def save_snapshot(exp_dir, sub_dir, write_payload, *, blob, metric=None):
    """Write a resumable snapshot into ``<exp_dir>/<sub_dir>/``, marker LAST. Returns the dir.

    ``write_payload(dir)`` writes the checkpoint file(s) and returns the list of names it wrote.
    Those names go INTO the marker, so :func:`read_snapshot` verifies exactly what the writer
    produced instead of consulting a second hard-coded list that can drift from it — the two
    shapes in this repo (a GAN's G+D pair, shallow diffusion's single file) already differ.

    ⛔ ORDER: payload first, ``state.json`` LAST, and the old marker is removed BEFORE the
    payload is touched. The marker's presence is what says the files beside it are complete, so a
    kill anywhere in between must read as "there is no snapshot", never as a half-written one.
    """
    d = snapshot_dir(exp_dir, sub_dir)
    os.makedirs(d, exist_ok=True)
    state_path = os.path.join(d, BEST_STATE)
    # Remove the marker FIRST: while the payload is being rewritten the old marker would describe
    # files that no longer exist.
    try:
        os.remove(state_path)
    except OSError:
        pass
    names = list(write_payload(d))
    out = dict(blob)
    out["files"] = names
    if metric is not None:
        out["best_metric"] = float(metric)
    write(state_path, out)
    return d


def save_best_pair(exp_dir, save_checkpoint, nets, optims, learning_rate, *, epoch, metric, blob):
    """Write the resumable snapshot OF THE BEST POINT (the GAN shape: a G+D pair).

    `save_checkpoint` is the trainer's own vendored function, passed in rather than imported: the
    three copies differ (argument order, sovits' `clean_checkpoints` tail) and this module must
    not become a fourth opinion about how a checkpoint is written.

    WHY THIS EXISTS. `weights/<name>_best.pth` is an inference-only export — the generator alone,
    no discriminator, no optimizer. `tproject.rs` already says the same thing about the diffusion
    twin: "a BEST SNAPSHOT, never a resume point … offering it would rewind thousands of steps
    AND zero the AdamW momentum". So when a run degrades past its best, the user's only resume
    point is the LATEST one — which is the degraded state. The best was a dead end.

    ⛔ NOT in `model_dir`, and the reason is five separate consumers — see BEST_DIR.
    """
    (net_g, net_d), (optim_g, optim_d) = nets, optims

    def payload(d):
        save_checkpoint(net_g, optim_g, learning_rate, epoch, os.path.join(d, BEST_G))
        save_checkpoint(net_d, optim_d, learning_rate, epoch, os.path.join(d, BEST_D))
        return [BEST_G, BEST_D]

    return save_snapshot(exp_dir, BEST_DIR, payload, blob=blob, metric=metric)


def save_solo_snapshot(exp_dir, sub_dir, save_one, *, blob, metric=None):
    """The single-model shape (shallow diffusion): ``save_one(path)`` writes ONE checkpoint file.

    Self-contained on purpose. Storing only the optimizer here and reusing the weights already
    sitting in ``model_<step>.pt`` / ``model_best.pt`` would save ~211 MB, but it would invent a
    cross-file invariant nothing else in this repo has — and the archive cleanup CAN delete a
    numbered checkpoint, which would leave the moments behind as an unusable orphan.
    """

    def payload(d):
        save_one(os.path.join(d, BEST_MODEL))
        return [BEST_MODEL]

    return save_snapshot(exp_dir, sub_dir, payload, blob=blob, metric=metric)


#: `cfg["resume_from"]` — which archive a 续训 continues from. Default is the historical behaviour.
PREFER_LATEST = "latest"
PREFER_BEST = "best"


def choose_pair(exp_dir, prefer, latest):
    """Which (G, D) a resume should read. Returns `(g_path, d_path, best_blob, source)`.

    `latest(pattern)` is the trainer's OWN `latest_checkpoint_path` already bound to its model
    dir — the three copies sort differently (rvc by `(mtime, digits)`, the so-vits pair by the
    digits of the whole path) and this module must not become a fourth opinion about "latest".

    `best_blob` is non-None only when the best snapshot was chosen; it carries the step number,
    which is the ONE thing a `resume_best/G.pth` cannot state the way a `G_<n>.pth` does.

    ⚠ Falling back is loud, never silent: asking for the best point and quietly getting the
    latest one is the exact shape of failure this whole feature exists to remove.
    """
    if prefer == PREFER_BEST:
        blob = read_best(exp_dir)
        if blob is not None:
            d = best_dir(exp_dir)
            return os.path.join(d, BEST_G), os.path.join(d, BEST_D), blob, PREFER_BEST
    return latest("G_*.pth"), latest("D_*.pth"), None, PREFER_LATEST


def read_snapshot(exp_dir, sub_dir=BEST_DIR):
    """A snapshot's state, or None when there is not a COMPLETE one in ``<exp_dir>/<sub_dir>/``.

    The payload names come out of the marker itself (``files``), so this cannot disagree with
    whatever :func:`save_snapshot` actually wrote. A marker from before S118 lists none, and for
    those the payload was always the GAN pair — hence the fallback.
    """
    d = snapshot_dir(exp_dir, sub_dir)
    blob = read(os.path.join(d, BEST_STATE))
    if blob is None:
        return None
    names = blob.get("files") or (BEST_G, BEST_D)
    if not all(os.path.isfile(os.path.join(d, n)) for n in names):
        return None
    return blob


def read_best(exp_dir):
    """The best snapshot's state, or None when there is not a complete one."""
    return read_snapshot(exp_dir, BEST_DIR)


def write(path, blob):
    """Atomic write. Callers MUST write this AFTER both halves of the pair — its presence is what
    says the pair beside it is complete."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    os.replace(tmp, path)


def read(path):
    """The sidecar, or None when it is absent / unreadable / from a schema we do not know."""
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict) or blob.get("schema") != SCHEMA:
        return None
    return blob


def load_for(checkpoint_epoch, path, logger=None):
    """The sidecar at `path`, but only if it belongs to the checkpoint that was just loaded.

    A crash between `save_checkpoint` and `write` leaves a sidecar one save old. Trusting it
    would restore a stale RNG/scale — small, but silent, and this module exists because silent
    is how the last one hid. The epoch the trainer just read out of the checkpoint is the join
    key; a mismatch falls back to today's behaviour and SAYS SO.
    """
    blob = read(path)
    if blob is None:
        return None
    if int(blob.get("epoch", -1)) != int(checkpoint_epoch):
        if logger is not None:
            logger.warning(
                "resume state sidecar is stale (it describes epoch %s, the checkpoint is epoch %s)"
                " — ignoring it; the GradScaler and RNG will restart from defaults",
                blob.get("epoch"),
                checkpoint_epoch,
            )
        return None
    return blob


def restore(blob, scaler=None, logger=None):
    """Put the captured state back. Every piece is independent: one failure never costs another."""
    report = RestoreReport()
    if blob is None:
        return report
    if scaler is not None:
        st = blob.get("scaler")
        if not st:
            report.scaler = "not in checkpoint (fp32 run)" if blob.get("scaler_enabled") is False else "absent"
        elif not scaler.is_enabled():
            # fp16 was turned OFF between runs (filelist.py lets the toggle through on purpose,
            # so that turning it off after a blowup actually takes effect). Loading a scale into
            # a disabled scaler is a no-op in torch, but saying nothing would hide the switch.
            report.scaler = "skipped: this run has the scaler disabled"
        else:
            try:
                scaler.load_state_dict(st)
                report.scaler = "restored (scale=%s)" % st.get("scale")
            except Exception as e:
                report.scaler = "%s: %s" % (type(e).__name__, e)
    _restore_rng(blob.get("rng") or {}, report)
    if logger is not None:
        logger.info("%s", report.line())
    return report


def report_drift(blob, reporter, logger, *, exp_dir, dataset_items=None, loader_len=None):
    """Log the resume header and raise the warning CODE if the dataset moved. ONE call per
    trainer — the three of them must not each grow their own version of this.

    Returns the CODE (or None), so a caller that wants to assert on it can.
    """
    code, lines = describe_drift(
        blob, exp_dir=exp_dir, dataset_items=dataset_items, loader_len=loader_len
    )
    for line in lines:
        logger.info("%s", line)
    if code and reporter is not None:
        reporter.warn(code)
    return code


def describe_drift(blob, *, exp_dir, dataset_items=None, loader_len=None):
    """What changed between the checkpoint's world and this run's. Returns (code, lines).

    `code` is `CODE_DATASET_CHANGED` when the dataset identity itself moved, else None. `lines`
    is always populated — it is the resume header from
    `project_v2_resume_divergence_open` §4-3, and its job is to make the NEXT bug report
    answerable: a resumed run whose loader length changed is a different training problem from a
    resumed run whose data is identical, and today's logs cannot tell them apart.
    """
    lines = []
    code = None
    if blob is None:
        return None, ["resume: no state sidecar — dataset identity at checkpoint time is unknown"]
    now_fp = read_dataset_fingerprint(exp_dir)
    was_fp = blob.get("dataset_fingerprint")
    if was_fp and now_fp and was_fp != now_fp:
        code = CODE_DATASET_CHANGED
        lines.append("resume: DATASET CHANGED since this checkpoint (%s -> %s)" % (was_fp[:12], now_fp[:12]))
    elif was_fp and now_fp:
        lines.append("resume: dataset unchanged (%s)" % was_fp[:12])
    else:
        lines.append("resume: dataset identity unavailable (was=%s now=%s)" % (bool(was_fp), bool(now_fp)))
    for label, old, new in (
        ("dataset items", blob.get("dataset_items"), dataset_items),
        ("batches per epoch", blob.get("loader_len"), loader_len),
    ):
        if old is not None and new is not None and int(old) != int(new):
            lines.append("resume: %s changed %s -> %s" % (label, old, new))
        elif new is not None:
            lines.append("resume: %s = %s" % (label, new))
    return code, lines
