"""S172 backstop: keep the RNN usable when MIOpen cannot compile its kernels.

WHAT BREAKS. The AMD runtime pack ships no C++ standard library. MIOpen JIT-compiles its
kernels through hipRTC at run time, and those sources reach `#include <type_traits>` via
vector_types.hpp -> miopen_type_traits.hpp. On a machine that happens to have Microsoft's
MSVC headers installed the include resolves and everything works; on a machine without them
it does not, and the compile dies:

    MIOpen(HIP): Error [Compile] hiprtcCompileProgram ... MIOpenNeuron.cpp: HIPRTC_ERROR_COMPILATION (6)
    RuntimeError: miopenStatusUnknownError        <- out of _VF.gru, i.e. here

Two community machines (RX 7700S/gfx1102 and RX 7900 XT/gfx1100) hit it; one of them had
Visual Studio and UNINSTALLED it, so a user can LOSE the ability to train by removing an
unrelated program. rmvpe's BiGRU is the first real GPU workload of four of the five training
chains, so the run dies in preprocessing with zero checkpoints written.

WHY A BACKSTOP AT ALL, when the real fix is to ship the headers: because that fix is a
shipped-payload bet, and any pack rebuild or ROCm bump can quietly break it again. This layer
costs ~nothing when the headers are there and turns a dead run into a working one when they
are not.

WHY REACTIVE (try, then degrade) RATHER THAN ALWAYS-BYPASS:
  * measured: the HIP context SURVIVES miopenStatusUnknownError — the same process, same
    module, recovers on the very next call (cold retry 69.2 ms, then 31.06 ms steady). That
    is the fact that makes reactive deliverable at all.
  * an always-bypass emits no signal. Reactive makes this a SENSOR as well as a crutch: the
    warning tells us, in every future community log, whether that machine could compile
    MIOpen kernels — which is exactly how we will tell "the header fix worked" from "it
    failed and the crutch hid it".

WHAT IT GUARDS. The WHOLE rmvpe network forward, not just the RNN — see the call sites in
rvc/rmvpe.py::mel2hidden and sovits/f0/rmvpe/inference.py::mel2hidden. Guarding only the RNN
was wrong and measurably so: the SoVITS lanes run fp32 (sovits/extract.py passes
dtype=torch.float32), and in fp32 torch hands BatchNorm to MIOpen too, so those lanes die in
the DeepUnet's `self.bn(x)` BEFORE they ever reach the BiGRU — with an RNN-only guard their
traceback is byte-identical to having no guard at all. RVC only looked safe because it runs
fp16, where BatchNorm does not go to MIOpen.

WHY IT DEGRADES TO GPU-WITHOUT-MIOpen (measured on the reproduction):
  * it is about an order of magnitude faster than a CPU pass. ⚠ Do not quote a single
    ratio: three runs of the same comparison gave 7.5x / 9.7x / 12.7x, because MIOpen's
    algorithm choice moves between processes.
  * numerics, on REAL material (10,534 frames, perf DB shared as production shares it):
    mean 0.025 cents, median 0.003, p99.9 0.051, max 0.28 — with 2 voiced-flag flips and
    4 coarse_f0 bins moved per ~10k frames. That is far under the 5-10 cent perceptual JND
    and under the ~12-cent width of one coarse bin. ⇒ we do NOT invalidate pools and do NOT
    add an f0-lineage token to the pool identity.
    ⛔ An earlier version of this comment claimed "0/201 bins moved, below the fast path's
    own noise floor". Both halves were artefacts: the 201-frame sample was a synthetic
    220 Hz tone spanning ~70 cents (3 of 256 bins), so it could not move a bin whatever the
    code did — a negative control used as an alibi — and the "noise floor" came from giving
    each arm a fresh MIOPEN_USER_DB_PATH, which production does not do.

⛔ THIS IS NOT the S67 silent GPU->CPU degradation. It stays on the GPU, and it is LOUD:
   one `reporter.warn(CODE_BYPASSED)` per run through the normal warning channel.

⚠ `torch.backends.cudnn.enabled` is PROCESS-global, not thread-local, and it is what gates
  torch's MIOpen RNN descent (`ATen/native/RNN.cpp::use_miopen()` ends in
  `userEnabledCuDNN()`). Every f0 lane is single-threaded today, so the scope below is safe.
  If f0 extraction is ever moved onto worker threads this must be revisited: a concurrent
  thread could re-enable it mid-GRU.
⚠ We save/restore ONLY `enabled`. `torch.backends.cudnn.flags(enabled=False)` looks tidier
  but also resets `benchmark` and `deterministic` to False inside the block (measured), which
  would silently change conv behaviour if the scope ever grew — sovits/train.py sets
  `benchmark = True` on purpose.
"""
import contextlib
import logging
import sys
import threading

logger = logging.getLogger(__name__)

#: Raised once per run when the bypass engages. Plain top-level literal on purpose — the
#: cross-language CODE gate parses it as the single source (same contract as
#: device.py::AMD_GPU_NOT_FOUND_CODE).
CODE_BYPASSED = "TRAINING_MIOPEN_KERNEL_BYPASSED"

_lock = threading.Lock()
_bypass = False
_reporter = None
_is_rocm = None


def install(reporter):
    """Give this module the run's reporter so the degrade can be announced. Called once from
    runner.py after the accelerator guard, before any lane dispatches. Safe to skip: without
    a reporter the bypass still works, it just logs instead of raising a UI warning."""
    global _reporter
    _reporter = reporter


def reset_for_tests():
    """Test seam: the module-level latch is per-process, and a gate that exercises the
    degrade twice in one process would otherwise see the second run already bypassed."""
    global _bypass, _reporter
    with _lock:
        _bypass = False
    _reporter = None


def _rocm():
    global _is_rocm
    if _is_rocm is None:
        import torch

        # non-None ONLY on the two AMD packs; None on cpu / nv-cu130 / xpu and on the dev
        # venv (2.5.1+cu121). Verified across every torch install in this repo.
        _is_rocm = getattr(torch.version, "hip", None) is not None
    return _is_rocm


def _is_miopen_kernel_failure(exc):
    """MIOpen surfaces a failed kernel build as a bare `miopenStatusUnknownError`; the reason
    (the hipRTC compiler text) goes to stderr, never into the exception. Match on the status
    name rather than on the compiler text, which we may not be able to see at all."""
    return "miopen" in ("%s" % exc).lower()


@contextlib.contextmanager
def _without_miopen():
    import torch

    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = prev


def _announce(exc):
    msg = (
        "MIOpen could not build its RNN kernel on this GPU (%s); running the RNN without "
        "MIOpen for the rest of this run. Still on the GPU — f0 quality is unaffected "
        "(measured below this path's own run-to-run noise)." % exc
    )
    logger.error("%s", msg)
    # stderr as well: the app log captures the sidecar's stderr, and a future bug report may
    # arrive as a log file without the warning list attached.
    print("[miopen] %s %s" % (CODE_BYPASSED, msg), file=sys.stderr, flush=True)
    if _reporter is not None:
        try:
            _reporter.warn(CODE_BYPASSED)
        except Exception:  # a broken reporter must never turn the rescue into a failure
            logger.exception("could not report %s", CODE_BYPASSED)


def run(fn):
    """Call `fn()`, degrading off MIOpen's RNN path once if MIOpen cannot build its kernel.

    On every non-ROCm runtime this is one boolean test and a direct call — no try frame, no
    behaviour change, so the NVIDIA / Intel / CPU lanes stay byte-identical.
    """
    global _bypass

    if not _rocm():
        return fn()
    if _bypass:                     # already degraded: no throw, no retry, once per run
        with _without_miopen():
            return fn()
    try:
        return fn()
    except RuntimeError as exc:
        if not _is_miopen_kernel_failure(exc):
            raise
        with _lock:
            first = not _bypass
            _bypass = True
        if first:
            _announce(exc)
        with _without_miopen():
            return fn()
