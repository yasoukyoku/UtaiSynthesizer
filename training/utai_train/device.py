"""Single-source device / backend shim (S42 Phase B — design §4.1).

Replaces the scattered `torch.cuda.amp.*` imports and ad-hoc
`torch.cuda.is_available()` two-value checks with ONE place that knows about every
backend. The `torch.cuda.amp` namespace is deprecated on the torch-2.11 axis (and
slated for removal); routing autocast/GradScaler through here migrates every
trainer to `torch.amp.*` in one shot.

Split into two halves by the torch-import seam:
  - `setup_visibility(cfg)` is PURE os.environ and is called by runner.py BEFORE
    torch is imported anywhere (visibility env vars must be set pre-import). It must
    therefore never import torch — the top level of this module imports only `os`,
    and every torch-touching helper imports torch lazily.
  - `resolve_backend` / `autocast` / `make_scaler` / `torch_device` are called from
    inside the train loops, where torch is already imported.

Backends:
  - "cuda"  NVIDIA, and ROCm/torch-hip which exposes the whole torch.cuda.* API
            (so AMD rides this branch transparently — design §4.1).
  - "xpu"   Intel Arc (Phase C). bf16 without a GradScaler (Arc A-series lacks fp64,
            so a real GradScaler crashes — design §1.3).
  - "cpu"   fp32, no autocast, no scaler.

Phase B exercises only cuda/cpu on real hardware; the xpu branches are wired now so
Phase C is a detection change, not a scatter of new if-elses. Numeric contract:
on cuda+enabled this is byte-identical to the old torch.cuda.amp path; on cpu
autocast is a no-op and the scaler never scales, so gate1's CPU fp32 trace stays
bitwise-equal to the pre-migration trace.
"""
import os

BACKENDS = ("cuda", "xpu", "cpu")


def _device_type(backend):
    """Accept a bare backend ('cuda'/'xpu'/'cpu') OR a device string carrying an
    index ('cuda:0'). torch.amp wants the TYPE, and — the important half — an
    unrecognized string must NEVER silently fall through to a disabled scaler on a
    real GPU (Stage-1 review finding): 'cuda:0' -> 'cuda' keeps loss scaling on."""
    return str(backend).split(":")[0]


def setup_visibility(cfg):
    """Set the device-visibility env var for the requested backend. Called from
    runner.py BEFORE any torch import.

    Preserves the Windows invariant: an EMPTY env var equals a DELETED var (all GPUs
    visible), so CPU mode is the explicit "-1" sentinel, never an empty string. The
    cuda branch is byte-identical to the old inline runner.py behaviour, so a run.json
    without `device_backend` (pre-Phase-B format) behaves exactly as before.
    """
    backend = cfg.get("device_backend", "cuda")
    gpu = cfg.get("gpu")
    if gpu is None or str(gpu) == "":
        return
    gpu = str(gpu)
    if backend == "xpu":
        # Intel: ZE_AFFINITY_MASK selects the device; enable CPU fallback so a
        # missing op is a slow op, not a hard crash (design §1.3). "-1" = CPU: leave
        # the mask unset (fallback + no cuda visibility → cpu).
        if gpu != "-1":
            os.environ["ZE_AFFINITY_MASK"] = gpu
        os.environ["PYTORCH_ENABLE_XPU_FALLBACK"] = "1"
    else:
        # cuda AND rocm/hip both honour CUDA_VISIBLE_DEVICES; "-1" hides all GPUs.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def require_wanted_accelerator(cfg):
    """LOUD guard against silent GPU→CPU degradation (S67, community bug). The user
    picked a GPU (gpu != "-1") and the resolved runtime targets an accelerator
    backend, so torch MUST actually see one: a masked/mismatched CUDA_VISIBLE_DEVICES
    (the old WMI-index-as-CUDA-ordinal bug) or a broken driver used to fall through
    resolve_backend's cpu fallback without a word — hours of slow training with zero
    hint. Raises with a stable CODE the frontend maps to i18n (TRAIN_GPU_UNAVAILABLE);
    the runner turns it into a protocol error, failing the run up front instead.
    Explicit CPU ("-1" / backend "cpu") keeps the silent-cpu semantics — it's correct
    there. Called by runner.py AFTER setup_visibility, so importing torch here is safe.
    """
    backend = cfg.get("device_backend", "cuda")
    gpu = str(cfg.get("gpu", ""))
    if backend == "cpu" or gpu == "-1":
        return
    import torch

    if backend == "cuda" and torch.cuda.is_available():
        return
    if backend == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return
    count = torch.cuda.device_count() if backend == "cuda" else 0
    raise RuntimeError(
        "TRAINING_GPU_UNAVAILABLE: backend=%s selected_gpu=%r CUDA_VISIBLE_DEVICES=%r torch_devices=%d"
        % (backend, gpu, os.environ.get("CUDA_VISIBLE_DEVICES"), count)
    )


def resolve_backend(cfg):
    """The EFFECTIVE backend after checking real availability. Default "cuda" keeps
    pre-Phase-B run.json compatible. Falls back to cpu when the wanted accelerator
    isn't actually available (e.g. CUDA_VISIBLE_DEVICES=-1 → cuda unavailable), which
    reproduces the old `torch.cuda.is_available()`-driven CPU fallback exactly."""
    import torch

    want = cfg.get("device_backend", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    if want == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def torch_device(backend):
    import torch

    bt = _device_type(backend)
    if bt == "cuda":
        return torch.device("cuda:0")
    if bt == "xpu":
        return torch.device("xpu:0")
    return torch.device("cpu")


def autocast(backend, enabled=True, dtype=None):
    """torch.amp.autocast for `backend` — the single replacement for the deprecated
    torch.cuda.amp.autocast. On cuda+enabled it is byte-identical to the old cuda
    autocast; on cpu the callers pass enabled=False (fp16 is forced off upstream), so
    it is a no-op regardless of device_type and gate1's CPU trace is unchanged."""
    import torch

    return torch.amp.autocast(_device_type(backend), enabled=enabled, dtype=dtype)


def make_scaler(backend, enabled):
    """GradScaler for `backend`. Only cuda ever actually scales: xpu runs bf16
    without a scaler (Arc A-series lacks fp64 → GradScaler crashes, design §1.3) and
    cpu is fp32, so both are built disabled — `scaler.scale()/step()/update()` become
    transparent pass-throughs, identical math to the old CPU path."""
    import torch

    bt = _device_type(backend)
    if bt == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.amp.GradScaler(bt if bt in BACKENDS else "cpu", enabled=False)
