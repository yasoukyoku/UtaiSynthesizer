"""Runtime-pack self-test (S42).

    python -m utai_train.envtest --out <pack_dir>/envtest.json [--device cpu|cuda|xpu]

Spawned by the Rust pyenv command layer with cwd = <app>/training (package importable),
UTF-8 env forced by util::python_command. stdout = JSONL progress ({"type":"item",...}
per check, {"type":"done","overall":...} last); the full report is written to --out.

DESIGN RULE (s42 设计 §4.5, anti self-verification): every check asserts NUMBERS
against explicit tolerances — "imported without raising" is never a pass criterion
for anything that computes. This is the gate we ship to machines we cannot test on
(Intel/AMD tiers run the same items on-device against a same-machine CPU reference).

Items are independent: one failure never stops the run — the report shows the full
picture (a missing praat is a different repair than a broken torch).
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

# stdout must be protocol-clean UTF-8 exactly like the training sidecar; Reporter's
# constructor owns that reconfigure (single source) — we borrow it and emit our own
# line shapes through the same cleaned-JSON path.
from .protocol import Reporter, _clean

_reporter = None  # instantiated in main() for the stdout reconfigure side-effect


def _emit(obj):
    sys.stdout.write(json.dumps(_clean(obj), ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ─── §4.5(c) XPU CPU-fallback capture ──────────────────────────────────────
# On an Intel-GPU machine, PYTORCH_ENABLE_XPU_FALLBACK lets an unimplemented xpu op
# run on the CPU instead of hard-crashing, and PYTORCH_DEBUG_XPU_FALLBACK makes torch
# WARN once per fallen-back op. We record those warnings during the on-device checks
# and surface any HOT op (conv/stft/fft — the HiFi-GAN / mel throughput path) as an
# explicit WARN: a silent CPU fallback that tanks training must never read as a green
# pass (design §4.5). Inert on cpu/cuda — no xpu fallback path exists there, so the
# accumulator stays empty and the cuda/cpu verdict is byte-unchanged.
import contextlib  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402

_FALLBACK_OPS = set()
# torch's fallback WARN text (same shape as the MPS fallback message); tolerate a few
# phrasings so a torch-version wording tweak doesn't silently stop the capture.
_FALLBACK_SIG = re.compile(
    r"fall\s*back to run on the CPU"
    r"|not currently supported on the XPU"
    r"|falling back to (?:the )?CPU",
    re.I,
)
_OP_QUOTED = re.compile(r"operator '([^']+)'")
_OP_ATEN = re.compile(r"(aten::[A-Za-z0-9_.]+)")
_HOT_SUBSTRINGS = ("conv", "stft", "istft", "fft")


class _Warn:
    """A check returns _Warn(detail) to report status='warn' — visible but non-fatal:
    it does NOT flip report['overall'] (so the pass/fail gate the Rust side reads is
    unchanged and a tolerable cold-op fallback never blocks a pack); the amber UI
    treatment keys off report['warnings_present'] instead."""

    __slots__ = ("detail",)

    def __init__(self, detail):
        self.detail = detail


def _extract_fallback_op(msg):
    m = _OP_QUOTED.search(msg) or _OP_ATEN.search(msg)
    return m.group(1) if m else msg.strip()[:60]


def _classify_hot(ops):
    return {op for op in ops if any(s in op.lower() for s in _HOT_SUBSTRINGS)}


@contextlib.contextmanager
def _capture_fallbacks(sink):
    """Record torch's XPU→CPU fallback warnings emitted inside the block into `sink`.
    try/finally so a check that raises part-way still contributes what it triggered."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for wm in caught:
                msg = str(getattr(wm, "message", wm))
                if _FALLBACK_SIG.search(msg):
                    sink.add(_extract_fallback_op(msg))


def _tone(sr=44100, freq=220.0, secs=1.0, amp=0.6):
    import numpy as np

    t = np.arange(int(sr * secs), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _vowel_tone(sr=44100, freq=220.0, secs=0.5):
    """Vibrato (±4 Hz @5.5 Hz) + 3 harmonics + light noise — a stationary vowel-ish
    stimulus. Needed specifically for pw.harvest: on a PERFECTLY periodic constant
    tone its voicing decision marks ~everything unvoiced (probed on this exact env:
    pure sine → 1/101 voiced frames, this signal → 101/101 at 220.33 Hz), while
    dio+stonemask voices the pure tone fine. Real audio is never perfectly periodic,
    so the natural-ish stimulus is the honest probe of the production f0 branch."""
    import numpy as np

    n = int(sr * secs)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.RandomState(3)
    f_inst = freq + 4.0 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    x = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    return (x + 0.003 * rng.randn(n)).astype(np.float64)


# ─── checks (each returns a detail string on pass, raises on fail) ──────────

def check_python_info(ctx):
    import platform

    return "%s %s (%s)" % (platform.python_implementation(), sys.version.split()[0], sys.executable)


REQUIRED_MODULES = [
    "torch", "torchaudio", "numpy", "scipy", "librosa", "soundfile", "sklearn",
    "matplotlib", "yaml", "onnxruntime", "onnx", "onnxconverter_common", "faiss",
    "parselmouth", "pyworld", "click", "lightning", "torchmetrics", "tensorboard",
    "numba",
]


def check_imports(ctx):
    import warnings

    versions = {}
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            m = __import__(mod)
            with warnings.catch_warnings():
                # click 8.4 deprecation-warns on __version__ — harmless here, keep
                # the protocol stderr quiet.
                warnings.simplefilter("ignore", DeprecationWarning)
                versions[mod] = str(getattr(m, "__version__", "?"))
        except Exception as e:
            missing.append("%s (%s)" % (mod, type(e).__name__))
    ctx["versions"] = versions
    if missing:
        # Stable CODE (S74b): the panel renders this detail to the user, so every check whose
        # failure the user must ACT on carries a code the frontend localizes. Remedy = reinstall
        # the runtime pack. The detail after the code stays raw/technical by convention.
        raise RuntimeError("ENVTEST_PACKAGES_BROKEN: " + ", ".join(missing))
    return ", ".join("%s %s" % kv for kv in sorted(versions.items()))


def _nvidia_driver_major():
    """Driver major via nvidia-smi — WITHOUT touching torch.cuda. None = unreadable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode != 0:
            return None
        lines = (out.stdout or "").strip().splitlines()
        return int(lines[0].split(".")[0]) if lines else None
    except Exception:
        return None


def check_cuda_driver(ctx):
    """S68f: the driver floor is checked BEFORE anything touches torch.cuda — probing
    the CUDA runtime on a driver older than the wheel's CUDA major can ACCESS-VIOLATE
    inside the runtime (community RTX 4070 Laptop: cudaErrorNotSupported warning, then
    0xC0000005 with no report ever written). CUDA 13 wheels need an r580+ driver.
    On failure the RUN SHORT-CIRCUITS (main() skips the remaining checks) so the
    verdict arrives as a normal report instead of a crash. Unknown driver fails open —
    check_torch_backend's graceful is_available() probe still covers plain-unavailable."""
    if ctx["device"] != "cuda":
        return None  # 该档位不适用
    import torch  # importing torch does NOT initialize CUDA

    # S172: a ROCm pack also runs on the "cuda" tier (HIP masquerades as torch.cuda),
    # where torch.version.cuda is None — the old text then read "cuda ? (no driver floor
    # required)", which names the wrong vendor on an AMD box and establishes nothing.
    # Say which runtime was actually seen; the CUDA 13 driver floor genuinely does not
    # apply to HIP, so this stays a pass either way.
    if getattr(torch.version, "hip", None):
        return "rocm/hip %s (no CUDA driver floor applies)" % torch.version.hip
    cu = torch.version.cuda or ""
    if not cu.startswith("13"):
        return "cuda %s (no driver floor required)" % (cu or "?")
    drv = _nvidia_driver_major()
    if drv is None:
        return "driver version unreadable (nvidia-smi) - not gating; later checks decide"
    if drv < 580:
        # Reuses the CODE the nv-cu130 DOWNLOAD gate already emits (backend.RUNTIME_DRIVER_TOO_OLD,
        # trilingual) — same condition, same remedy, one localized sentence for both.
        raise RuntimeError("RUNTIME_DRIVER_TOO_OLD: driver %d < 580, cuda %s" % (drv, cu))
    return "driver %d (>= 580)" % drv


# S169 "probe ran; no device this pack can drive" — a plain top-level literal on
# purpose: the Rust cross-language gate parses it as the single source (the same
# contract as device.py's AMD_GPU_NOT_FOUND_CODE / ckpt_guard.py's CODE).
NO_COVERED_GPU_CODE = "ENVTEST_AMD_NO_COVERED_GPU"


def check_amd_device_pick(ctx):
    """S169: on the cuda tier of a ROCm pack, pick the GPU by kernel arch instead of
    trusting HIP device 0. HIP's device order is not the app's DXGI order, and on a
    mixed-arch laptop device 0 can be an iGPU the pack carries no kernels for (field
    case: gfx1035 iGPU first, the supported RX 7700S/gfx1102 second — every default-
    device touch died, one of them as a native 0xC0000005).

    Skips on non-cuda tiers and when the caller passed no --gfx-targets (NVIDIA
    packs, pre-0.12.2 callers) — behaviour is then byte-identical to before.

    Placement is load-bearing: AFTER cuda_driver, BEFORE torch_backend. torch.cuda
    is first touched in check_torch_backend and torch imports are lazy inside every
    check, so setting the visibility env here still precedes HIP init. A FAIL
    short-circuits the remaining checks exactly like cuda_driver's (main() break):
    they would touch the uncovered device 0 and can crash natively — one decisive
    fail + a clean report beats a 0xC0000005 with no report.

    Two distinct reds by design (closed-gate rule): the enum helper raises
    TRAINING_AMD_ENUM_FAILED ("the probe could not run"), this check raises
    ENVTEST_AMD_NO_COVERED_GPU ("probe ran; no device this pack can drive")."""
    targets = ctx.get("gfx_targets") or []
    if ctx["device"] != "cuda" or not targets:
        return None
    from .hipenum import (
        VISIBILITY_VARS,
        base_arch,
        describe_devices,
        enumerate_hip_devices,
        pick_hip_index,
    )

    devices = enumerate_hip_devices()
    idx = None
    for t in targets:
        idx = pick_hip_index(devices, t)
        if idx is not None:
            break
    if idx is None:
        raise RuntimeError(
            "%s: pack kernels cover [%s]; visible: %s"
            % (NO_COVERED_GPU_CODE, ",".join(targets), describe_devices(devices))
        )
    for k in VISIBILITY_VARS:
        os.environ.pop(k, None)  # delete, never blank: on Windows empty == unset
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    picked = next((d for d in devices if int(d.get("index", -1)) == idx), {})
    return "picked hip_index=%d name=%s arch=%s (devices: %s)" % (
        idx,
        picked.get("name", "?"),
        base_arch(picked.get("arch", "?")),
        describe_devices(devices),
    )


def check_torch_backend(ctx):
    import torch

    dev = ctx["device"]
    info = "torch %s" % torch.__version__
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ENVTEST_TORCH_NO_CUDA: torch.cuda.is_available()=False (torch %s, cuda %s)"
                % (torch.__version__, torch.version.cuda)
            )
        info += "; cuda %s; %s" % (torch.version.cuda, torch.cuda.get_device_name(0))
    elif dev == "xpu":
        # S74 (community report): a pre-Arc Intel box (Iris Xe) installed the xpu pack and got a
        # self-test failure naming only the CHECK ("torch_backend") — a guessing game. Raise a
        # STABLE ENGLISH CODE (i18n hard rule: the localized explanation lives in the frontend's
        # backend.XPU_NO_DEVICE, the detail here stays raw/technical) so the UI can say WHICH
        # Intel GPUs torch-XPU actually supports and that DirectML still covers inference.
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            raise RuntimeError("XPU_NO_DEVICE: no torch.xpu namespace in torch %s" % torch.__version__)
        if not xpu.is_available():
            raise RuntimeError(
                "XPU_NO_DEVICE: torch.xpu.is_available()=False (torch %s)" % torch.__version__
            )
        info += "; xpu %s" % xpu.get_device_name(0)
    return info


def check_stft_roundtrip(ctx):
    """istft(stft(x)) reconstruction under COLA (hann, 75% overlap) — the numeric
    heart of every vocoder/mel path. fp32 reference error is ~1e-6; 1e-4 is the
    loud-failure line, not a tuning knob."""
    import torch

    torch.manual_seed(1234)
    x = torch.randn(1, 44100)
    win = torch.hann_window(2048)
    spec = torch.stft(x, 2048, hop_length=512, window=win, center=True, return_complex=True)
    y = torch.istft(spec, 2048, hop_length=512, window=win, center=True, length=x.shape[-1])
    err = (y - x).abs().max().item()
    if not (err < 1e-4):
        raise RuntimeError("reconstruction error too large: max_abs=%.3e (limit 1e-4)" % err)
    if spec.abs().sum().item() <= 0:
        raise RuntimeError("stft output is all zeros")
    return "max_abs=%.2e, spec %s" % (err, tuple(spec.shape))


def check_resample(ctx):
    """torchaudio.transforms.Resample — the rmvpe-f0 chain's numeric-sensitive
    dependency (requirements.txt: same-version pair with torch is mandatory)."""
    import numpy as np
    import torch
    import torchaudio

    x = torch.from_numpy(_tone(44100, 220.0, 1.0)).unsqueeze(0)
    y = torchaudio.transforms.Resample(44100, 16000)(x)
    if y.shape[-1] != 16000:
        raise RuntimeError("output length %d != 16000" % y.shape[-1])
    if not torch.isfinite(y).all():
        raise RuntimeError("output contains NaN/Inf")
    rms_in = float(x.pow(2).mean().sqrt())
    rms_out = float(y.pow(2).mean().sqrt())
    ratio = rms_out / max(rms_in, 1e-9)
    if not (0.7 < ratio < 1.4):
        raise RuntimeError("energy ratio out of range %.3f (resampler numerically broken?)" % ratio)
    del np
    return "len 16000, rms ratio %.3f" % ratio


def check_librosa_mel(ctx):
    """librosa is a lazy_loader shell in 0.11 — `import librosa` proves nothing.
    filters.mel is the REAL production call (rvc/sovits mel_processing build their
    filterbanks with it); assert actual numbers out of it."""
    import librosa
    import numpy as np

    basis = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128)
    if basis.shape != (128, 1025):
        raise RuntimeError("mel basis has unexpected shape %s" % (basis.shape,))
    if not np.isfinite(basis).all() or basis.min() < 0:
        raise RuntimeError("mel basis contains invalid values")
    zero_rows = int((basis.sum(axis=1) <= 0).sum())
    if zero_rows:
        raise RuntimeError("%d empty mel filters" % zero_rows)
    return "mel basis (128,1025), 0 empty filters"


def check_numba_jit(ctx):
    """numba/llvmlite native codegen — a transitive librosa dependency whose DLLs
    are classic antivirus-quarantine bait; compile and RUN a kernel, don't just
    import."""
    import numpy as np
    from numba import njit

    @njit(cache=False)
    def _acc(arr):
        total = 0.0
        for i in range(arr.shape[0]):
            total += arr[i] * 0.5
        return total

    x = np.arange(1000, dtype=np.float64)
    got = _acc(x)
    want = float(x.sum() * 0.5)
    if abs(got - want) > 1e-6:
        raise RuntimeError("JIT result wrong: %.6f != %.6f" % (got, want))
    return "njit compile+run OK (%.1f)" % got


def check_soundfile_roundtrip(ctx):
    import numpy as np
    import soundfile as sf

    tmp = ctx["tmp_dir"]
    path = os.path.join(tmp, "roundtrip_f32.wav")
    data = _tone(44100, 220.0, 0.5)
    sf.write(path, data, 44100, subtype="FLOAT")
    back, sr = sf.read(path, dtype="float32")
    if sr != 44100 or back.shape[0] != data.shape[0]:
        raise RuntimeError("read-back shape/samplerate mismatch")
    if not np.array_equal(back, data):
        raise RuntimeError("float32 wav round-trip is not bit-exact")
    return "float32 bit-exact (%d samples)" % data.shape[0]


def check_parselmouth(ctx):
    """PSOLA 增强引擎 (S41) 的宿主 — a 220 Hz tone must pitch-track to 220 Hz.
    Uses a 0.5 s source: praat raises on <50 ms audio (S41 landmine), stay far away."""
    import numpy as np
    import parselmouth

    snd = parselmouth.Sound(_tone(44100, 220.0, 0.5).astype(np.float64), 44100)
    pitch = snd.to_pitch_ac()
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if voiced.size < 10:
        raise RuntimeError("almost no voiced frames detected (%d)" % voiced.size)
    med = float(np.median(voiced))
    if not (212.0 < med < 228.0):
        raise RuntimeError("f0 median %.2f Hz deviates from 220 Hz" % med)
    ver = getattr(parselmouth, "PRAAT_VERSION", "?")
    return "praat %s, f0 median %.2f Hz (%d voiced)" % (ver, med, voiced.size)


def check_pyworld(ctx):
    import numpy as np
    import pyworld as pw

    x = _vowel_tone(44100, 220.0, 0.5)
    f0, _t = pw.harvest(x, 44100, f0_floor=80.0, f0_ceil=800.0)
    voiced = f0[f0 > 0]
    if voiced.size < f0.size // 2:
        raise RuntimeError("harvest found too few voiced frames (%d/%d)" % (voiced.size, f0.size))
    med = float(np.median(voiced))
    if not (210.0 < med < 230.0):
        raise RuntimeError("harvest f0 median %.2f Hz deviates from 220 Hz" % med)
    return "f0 median %.2f Hz (%d/%d voiced)" % (med, voiced.size, f0.size)


def check_onnxruntime(ctx):
    """A REAL session on a minimal graph — import-only would miss broken native DLLs."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    node = helper.make_node("Add", ["a", "b"], ["c"])
    graph = helper.make_graph(
        [node],
        "envtest_add",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3]),
        ],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((2, 3), dtype=np.float32)
    (c,) = sess.run(None, {"a": a, "b": b})
    if not np.array_equal(c, a + b):
        raise RuntimeError("Add produced a wrong result")
    return "ort %s CPU EP OK" % ort.__version__


def check_faiss(ctx):
    import faiss
    import numpy as np

    rng = np.random.RandomState(7)
    base = rng.randn(200, 32).astype(np.float32)
    idx = faiss.IndexFlatL2(32)
    idx.add(base)
    d, i = idx.search(base[:5], 1)
    if not (i[:, 0] == np.arange(5)).all():
        raise RuntimeError("self nearest-neighbour search misaligned: %s" % i[:, 0].tolist())
    if float(d.max()) > 1e-4:
        raise RuntimeError("self distance is non-zero: %.3e" % float(d.max()))
    return "IndexFlatL2 self-NN 5/5"


def check_sklearn_kmeans(ctx):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.RandomState(11)
    pts = np.concatenate([rng.randn(80, 8) + 4.0, rng.randn(80, 8) - 4.0]).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=3, batch_size=64).fit(pts)
    centers = km.cluster_centers_
    if not np.isfinite(centers).all():
        raise RuntimeError("cluster centers contain NaN/Inf")
    spread = float(np.abs(centers).mean())
    if not (2.0 < spread < 6.0):
        raise RuntimeError("cluster centers out of range (|mean|=%.2f, expected ~4)" % spread)
    return "2 centers, |mean| %.2f" % spread


def check_lightning(ctx):
    """A REAL 1-batch Trainer.fit — the vocoder finetune chain is the only Lightning
    consumer and a torch×lightning integration break (the design's §3.1-5 open item)
    must show up HERE, not at the user's first training launch. CPUAccelerator
    .is_available() is a constant True — worthless as a check.
    stdout stays protocol-clean: everything lightning prints is rerouted to stderr."""
    import contextlib

    import lightning
    import torch

    class _Probe(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(8, 1)
            self.seen_loss = None

        def training_step(self, batch, _idx):
            (x,) = batch
            loss = self.layer(x).pow(2).mean()
            self.seen_loss = float(loss.detach())
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    torch.manual_seed(3)
    ds = torch.utils.data.TensorDataset(torch.randn(16, 8))
    dl = torch.utils.data.DataLoader(ds, batch_size=8)
    probe = _Probe()
    with contextlib.redirect_stdout(sys.stderr):
        trainer = lightning.Trainer(
            fast_dev_run=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(probe, dl)
    import math

    if probe.seen_loss is None or not math.isfinite(probe.seen_loss):
        raise RuntimeError("Trainer.fit did not run training_step, or loss was non-finite")
    return "lightning %s Trainer.fit 1 step OK (loss %.4f)" % (
        lightning.__version__, probe.seen_loss)


def check_tiny_gan(ctx):
    """A miniature GAN with Conv1d + ConvTranspose1d — front/backward/optimizer on the
    op family HiFi-GAN training lives on (and the exact family with known MIOpen /
    XPU risk on non-NVIDIA backends). Asserts the L1 term actually LEARNS (robust
    decrease), gradients flow AND are finite on BOTH nets (§4.5(b): non-zero non-NaN —
    a partially-NaN grad set must fail loudly, not masquerade as 'no gradient'),
    everything stays finite. Runs under the fallback capture at its call site, so a
    conv silently falling to CPU on xpu is recorded, not hidden."""
    import torch
    import torch.nn as nn

    dev = torch.device(ctx["device"] if ctx["device"] != "cpu" else "cpu")
    torch.manual_seed(7)
    g = nn.Sequential(
        nn.ConvTranspose1d(8, 16, 8, stride=2, padding=3),
        nn.LeakyReLU(0.1),
        nn.ConvTranspose1d(16, 8, 8, stride=2, padding=3),
        nn.LeakyReLU(0.1),
        nn.Conv1d(8, 1, 7, padding=3),
        nn.Tanh(),
    ).to(dev)
    d = nn.Sequential(
        nn.Conv1d(1, 16, 15, stride=4, padding=7),
        nn.LeakyReLU(0.1),
        nn.Conv1d(16, 32, 15, stride=4, padding=7),
        nn.LeakyReLU(0.1),
        nn.Conv1d(32, 1, 3, padding=1),
    ).to(dev)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-3)
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-3)
    z = torch.randn(4, 8, 64, device=dev)
    target = torch.sin(torch.linspace(0, 50, 256, device=dev)).repeat(4, 1, 1)

    l1_hist = []
    grad_seen = False
    for _step in range(30):
        fake = g(z)
        # D step
        opt_d.zero_grad(set_to_none=True)
        loss_d = torch.mean((d(target) - 1.0) ** 2) + torch.mean(d(fake.detach()) ** 2)
        loss_d.backward()
        for p in d.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError("D gradients contain NaN/Inf (step %d)" % _step)
        opt_d.step()
        # G step (L1 dominates so the learning check is deterministic-ish)
        opt_g.zero_grad(set_to_none=True)
        l1 = (fake - target).abs().mean()
        loss_g = l1 + 0.1 * torch.mean((d(fake) - 1.0) ** 2)
        loss_g.backward()
        for p in g.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise RuntimeError("G gradients contain NaN/Inf (step %d)" % _step)
            if float(p.grad.abs().sum()) > 0:
                grad_seen = True
        opt_g.step()
        if not (torch.isfinite(loss_d) and torch.isfinite(loss_g)):
            raise RuntimeError("loss became NaN/Inf (step %d)" % _step)
        l1_hist.append(float(l1.detach()))
    first = sum(l1_hist[:3]) / 3.0
    last = sum(l1_hist[-3:]) / 3.0
    if not grad_seen:
        raise RuntimeError("G never received a non-zero gradient")
    if not (last < first * 0.8):
        raise RuntimeError("L1 did not converge (first %.4f -> last %.4f, expected >20%% drop)" % (first, last))
    return "L1 %.4f → %.4f (30 steps, %s)" % (first, last, dev)


def check_dataloader_spawn(ctx):
    """THE embedded-python probe: DataLoader workers on Windows re-launch
    sys.executable via multiprocessing spawn — exactly what breaks on the
    python.org embeddable distribution (and what our standalone base must survive)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.arange(64, dtype=torch.float32).view(16, 4))
    dl = DataLoader(ds, batch_size=4, num_workers=1, persistent_workers=False)
    total = 0.0
    batches = 0
    for (batch,) in dl:
        total += float(batch.sum())
        batches += 1
    expected = float(sum(range(64)))
    if batches != 4 or abs(total - expected) > 1e-3:
        raise RuntimeError("spawn worker produced wrong data (batches=%d, sum=%.1f != %.1f)" % (batches, total, expected))
    return "1 spawn worker, 4 batches, sum OK (executable=%s)" % os.path.basename(sys.executable)


# GPU tiers (Phase B: cuda; Phase C: xpu) — same-machine CPU reference comparisons.
def check_gpu_stft_vs_cpu(ctx):
    import torch

    dev = ctx["device"]
    if dev == "cpu":
        return None  # skip marker
    torch.manual_seed(99)
    x = torch.randn(1, 44100)
    win = torch.hann_window(2048)
    ref = torch.stft(x, 2048, hop_length=512, window=win, center=True, return_complex=True)
    xd = x.to(dev)
    wd = win.to(dev)
    sd = torch.stft(xd, 2048, hop_length=512, window=wd, center=True, return_complex=True)
    err = (sd.cpu() - ref).abs().max().item()
    scale = ref.abs().max().item()
    if not (err < 1e-3 * max(scale, 1.0)):
        raise RuntimeError("GPU stft deviates from the CPU reference: max_abs=%.3e (scale %.2f)" % (err, scale))
    # §4.5(a) reconstruction half: run istft ON DEVICE and compare to the CPU roundtrip
    # elementwise — exercises the inverse-FFT kernel (the other half of every vocoder
    # path), not just the forward transform. fp32 device↔cpu error is ~1e-6; 1e-3 loud.
    y_ref = torch.istft(ref, 2048, hop_length=512, window=win, center=True, length=x.shape[-1])
    yd = torch.istft(sd, 2048, hop_length=512, window=wd, center=True, length=x.shape[-1])
    err_i = (yd.cpu() - y_ref).abs().max().item()
    if not (err_i < 1e-3):
        raise RuntimeError("GPU istft deviates from the CPU reference: max_abs=%.3e (limit 1e-3)" % err_i)
    return "stft %.2e / istft %.2e vs CPU (scale %.2f)" % (err, err_i, scale)


def check_gpu_amp_step(ctx):
    import torch

    dev = ctx["device"]
    if dev == "cpu":
        return None
    if dev == "cuda":
        # fp16 + GradScaler — the production RVC/SoVITS mixed-precision shape.
        model = torch.nn.Conv1d(4, 4, 3, padding=1).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(dev)
        x = torch.randn(2, 4, 128, device=dev)
        w0 = model.weight.detach().clone()
        with torch.amp.autocast(dev, dtype=torch.float16):
            y = model(x)
            loss = y.pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if not torch.isfinite(loss):
            raise RuntimeError("amp loss is non-finite")
        # A weight MUST move: catches a silently no-op autocast/scaler (grads all
        # zeroed / step skipped on a spurious inf) that a loss-finite check alone
        # would pass green (§4.5 anti-self-deception).
        delta = float((model.weight.detach() - w0).abs().max())
        if not (delta > 0):
            raise RuntimeError("no weight moved after one amp step (autocast/scaler silently doing nothing?)")
        return "fp16 autocast + GradScaler step OK (dw %.2e)" % delta
    if dev == "xpu":
        # bf16 WITHOUT GradScaler — Arc A 系无 fp64，scaler 必崩（设计 §4.1）。
        model = torch.nn.Conv1d(4, 4, 3, padding=1).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(2, 4, 128, device=dev)
        w0 = model.weight.detach().clone()
        with torch.amp.autocast(dev, dtype=torch.bfloat16):
            loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError("bf16 loss is non-finite")
        delta = float((model.weight.detach() - w0).abs().max())
        if not (delta > 0):
            raise RuntimeError("no weight moved after one bf16 step (autocast silently doing nothing?)")
        return "bf16 autocast (no scaler) step OK (dw %.2e)" % delta
    return None


# ─── S172: the hole this tier had ────────────────────────────────────────────
# MEASURED: the 22 checks above return overall=pass, 22/22, on a machine where every
# single f0 slice fails and no training chain can start. The reason is structural, not a
# matter of degree — MIOpen resolves convolution from PRECOMPILED objects, so tiny_gan /
# gpu_stft_vs_cpu / gpu_amp_step never once ask the run-time kernel compiler to build
# anything. The only op in utai_train that does is rmvpe's BiGRU, and it was untested.
HIPRTC_HEADERS_CODE = "ENVTEST_HIPRTC_CXX_HEADERS_MISSING"
HIPRTC_PAYLOAD_CODE = "ENVTEST_HIPRTC_CXX_HEADERS_NOT_SHIPPED"
HIPRTC_PROBE_CODE = "ENVTEST_HIPRTC_PROBE_FAILED"
GPU_RNN_CODE = "ENVTEST_GPU_RNN_BROKEN"

_PROBE_KERNEL = b"__global__ void utai_probe(float* o){ *o = 0.f; }\n"
_PROBE_SRC = (
    b"#include <type_traits>\n"
    b'static_assert(std::is_same<float,float>::value,"");\n' + _PROBE_KERNEL
)


def _pack_cxx_include_dirs():
    """The C++ standard-library headers we ship for hipRTC, as RELATIVE paths.

    ⛔ ONE SOURCE with the Rust launcher: `util::hiprtc_cxx_include_options` builds
    `HIPRTC_COMPILE_OPTIONS_APPEND` from this same layout and both spawn sites set cwd to
    `<app_dir>/training`. If the directory moves, both move — otherwise this gate stops
    describing what the app actually does, which is worse than having no gate.
    [] = a build that ships none (every pack up to 0.12.3), where the this-machine arm
    below is the whole verdict.
    """
    override = os.environ.get("UTAI_HIPRTC_INCLUDE_DIRS")  # harness/manual override only
    if override:
        return [p for p in override.split(os.pathsep) if p]
    root = "rocm_cxx_headers"
    subs = [os.path.join(root, d) for d in ("v1", "clangres")
            if os.path.isdir(os.path.join(root, d))]
    return subs


def _hiprtc_lib():
    """ctypes handle on the hipRTC the PACK ships (never a system ROCm install)."""
    import ctypes

    try:
        import _rocm_sdk_core  # only exists inside a ROCm pack

        bindir = os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "bin")
        names = [n for n in sorted(os.listdir(bindir))
                 if n.lower().startswith("hiprtc") and n.lower().endswith(".dll")
                 and "builtin" not in n.lower()]
        if hasattr(os, "add_dll_directory") and os.path.isdir(bindir):
            os.add_dll_directory(bindir)  # amd_comgr.dll sits next to it
        if not names:
            raise RuntimeError("no hiprtc*.dll under %s" % bindir)
        lib = ctypes.CDLL(os.path.join(bindir, names[0]))
    except Exception as e:
        # "the probe could not run" — a DIFFERENT red from "the compiler cannot find the
        # headers" (closed-gate iron rule: a tool that failed to START must never read as
        # either a pass or as the failure it exists to detect).
        raise RuntimeError("%s: %s: %s" % (HIPRTC_PROBE_CODE, type(e).__name__, e))
    lib.hiprtcCreateProgram.restype = ctypes.c_int
    lib.hiprtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
    lib.hiprtcCompileProgram.restype = ctypes.c_int
    lib.hiprtcCompileProgram.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.hiprtcGetProgramLogSize.restype = ctypes.c_int
    lib.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.hiprtcGetProgramLog.restype = ctypes.c_int
    lib.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hiprtcDestroyProgram.restype = ctypes.c_int
    lib.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    return os.path.basename(lib._name), lib


def _hiprtc_try(lib, src, opts):
    """(rc, log) for one hipRTC compile. rc 0 = compiled."""
    import ctypes

    prog = ctypes.c_void_p()
    rc0 = lib.hiprtcCreateProgram(ctypes.byref(prog), src, b"utai_probe.cpp", 0, None, None)
    if rc0 != 0:
        return rc0, "hiprtcCreateProgram rc=%d" % rc0
    try:
        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = lib.hiprtcCompileProgram(prog, len(opts), arr)
        n = ctypes.c_size_t()
        lib.hiprtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(max(n.value, 1))
        lib.hiprtcGetProgramLog(prog, buf)
        return rc, buf.value.decode("utf-8", "replace").strip()
    finally:
        try:
            lib.hiprtcDestroyProgram(ctypes.byref(prog))
        except Exception:
            pass


def _first_error_line(log):
    for ln in (log or "").splitlines():
        if "error" in ln.lower():
            return ln.strip()[:240]
    txt = (log or "").strip()
    return txt.splitlines()[0][:240] if txt else "(the compiler printed nothing)"


def check_hiprtc_cxx_headers(ctx):
    """ROCm packs only (~2 s). MIOpen does not ship every kernel precompiled: several are
    JIT-compiled through hipRTC at RUN TIME, and those sources reach
    `#include <type_traits>` (miopen_type_traits.hpp:151 <- vector_types.hpp:8). AMD's
    comgr is supposed to embed the libc++ headers for exactly that and our pinned nightly
    ships the payload EMPTY (2 of ~180), so on a machine with no Microsoft C++ toolchain
    NOTHING satisfies that include and every such kernel dies as
    HIPRTC_ERROR_COMPILATION(6) -> miopenStatusUnknownError. First production victim:
    rmvpe's BiGRU, i.e. f0 extraction — 4 of the 5 training chains die before step 1
    (S172 community reports). Ask the SHIPPED compiler directly instead of inferring it
    from a symptom, and ask it in BOTH the shipping-regression form and the
    this-machine form."""
    if ctx["device"] != "cuda":
        return None
    import torch

    if not getattr(torch.version, "hip", None):
        return None  # an NVIDIA cuda-tier pack has no hipRTC to ask
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    name, lib = _hiprtc_lib()
    oa = ("--offload-arch=%s" % arch).encode()
    std = b"-std=c++17"

    # (1) can this compiler build ANYTHING here? Otherwise a red below would be
    #     misattributed to the headers.
    rc_bare, log_bare = _hiprtc_try(lib, _PROBE_KERNEL, [oa, std])
    if rc_bare != 0:
        raise RuntimeError("%s: the shipped hipRTC cannot compile even a bare kernel for %s "
                           "(rc=%d) — %s" % (HIPRTC_PROBE_CODE, arch, rc_bare,
                                             _first_error_line(log_bare)))
    # (2) NEGATIVE CONTROL: a bogus arch MUST fail, or this probe cannot see failure at
    #     all and its green means nothing.
    rc_neg, _ = _hiprtc_try(lib, _PROBE_KERNEL, [b"--offload-arch=gfx9999", std])
    if rc_neg == 0:
        raise RuntimeError("%s: the negative control COMPILED (gfx9999 is not a real arch) — "
                           "this probe cannot detect a compile failure, so its verdict is "
                           "worthless" % HIPRTC_PROBE_CODE)
    # (3) THE SHIPPING-REGRESSION ASSERTION. -nostdinc drops the system include dirs, so
    #     this passes only if the headers WE ship are self-sufficient — a developer's
    #     installed MSVC can no longer mask a payload that stopped shipping.
    shipped = _pack_cxx_include_dirs()
    ship_note = "this build ships no C++ headers"
    if shipped:
        # -idirafter, concatenated, v1 first — the SAME form the launcher injects
        # (util::hiprtc_cxx_include_options). A gate that tested `-I` would be testing
        # something the app never does, and `-I` is exactly the form that breaks a
        # Visual-Studio machine by shadowing its C++ library (S172, measured).
        opts = [oa, std, b"-nostdinc"] + [b"-idirafter" + d.encode() for d in shipped]
        rc_s, log_s = _hiprtc_try(lib, _PROBE_SRC, opts)
        if rc_s != 0:
            raise RuntimeError(
                "%s: the C++ headers this build ships do not satisfy <type_traits> on their "
                "own (rc=%d, %d dir(s): %s). %s"
                % (HIPRTC_PAYLOAD_CODE, rc_s, len(shipped), os.pathsep.join(shipped),
                   _first_error_line(log_s)))
        ship_note = "%d shipped dir(s) self-sufficient under -nostdinc" % len(shipped)
        for d in shipped:
            if " " in d:
                # HIPRTC_COMPILE_OPTIONS_APPEND is split on whitespace and honours no
                # quoting ('error reading "-IC:\\Program": invalid argument'), so a path
                # with a space cannot be injected at all — catch it here, not in the field.
                raise RuntimeError("%s: shipped include dir contains a space and therefore "
                                   "cannot be injected: %s" % (HIPRTC_PAYLOAD_CODE, d))

    # (4) THIS MACHINE, as the app actually runs it — the launcher's injected
    #     HIPRTC_COMPILE_OPTIONS_APPEND is honoured for this call too (measured).
    rc_tt, log_tt = _hiprtc_try(lib, _PROBE_SRC, [oa, std])
    if rc_tt != 0:
        raise RuntimeError(
            "%s: MIOpen compiles kernels at run time and they #include <type_traits>; the "
            "compiler in this pack cannot find it on this machine (arch %s, rc=%d). %s"
            % (HIPRTC_HEADERS_CODE, arch, rc_tt, _first_error_line(log_tt)))
    ctx["hiprtc_headers_ok"] = True
    return "%s resolves <type_traits> for %s; %s; gfx9999 control failed as required" % (
        name, arch, ship_note)


# Device-vs-CPU limits, RELATIVE to max|reference| (S172, measured):
#   fp32  worst measured 5.81e-04 — NVIDIA/cuDNN with TF32 ON (torch's default); the same
#         GRU on ROCm/MIOpen measures 5.83e-07 and on cuDNN with TF32 off 7.02e-06.
#         Limit 1e-2 = 17x the worst real reading. ⛔ Do NOT "tighten it to match AMD":
#         1e-4 would redden every Ampere+ NVIDIA box in the field.
#   fp16  worst measured 1.31e-03 (ROCm 780M; cuDNN 6.94e-04). Limit 2e-2 = 15x.
# Mutation control (measured): all-zeros, one direction dropped, directions swapped, time
# reversed, one frame in 128 zeroed, +1% noise and a uniform 2x gain all go RED; +0.1%
# noise and a 1.01x gain do not — that is this check's sensitivity floor, stated rather
# than implied.
RNN_REL_TOL = {"fp32": 1e-2, "fp16": 2e-2}
RNN_MIN_COS = 0.999


def check_gpu_rnn_vs_cpu(ctx):
    """The production shape of rmvpe's BiGRU — rvc/rmvpe.py:161-173 (fp16) and
    sovits/f0/rmvpe/seq.py:6-12 (fp32) — i.e. the f0 extractor that 4 of the 5 training
    chains run BEFORE step 1, and the ONLY live RNN in utai_train. It is also the one op
    family the rest of this tier structurally cannot cover: tiny_gan / gpu_stft /
    gpu_amp are convolution+FFT, and MIOpen resolves convolution from precompiled
    objects, so none of them ever asks the run-time compiler to build a kernel (measured).
    Asserts NUMBERS against a same-machine CPU reference built from the SAME weights, in
    both production dtypes: a kernel that returns shaped garbage must fail here, not only
    one that raises."""
    import torch

    dev = ctx["device"]
    if dev == "cpu":
        return None
    torch.manual_seed(2172)
    net = torch.nn.GRU(128, 96, num_layers=1, batch_first=True, bidirectional=True).eval()
    x = torch.randn(1, 128, 128)
    with torch.no_grad():
        ref = net(x)[0]
    scale = float(ref.abs().max())
    if not (scale > 0.1):
        raise RuntimeError("the CPU reference itself is degenerate (max|ref|=%.3e) — this "
                           "check cannot conclude anything" % scale)
    try:
        with torch.no_grad():
            g = net.to(dev)
            outs = {"fp32": g(x.to(dev))[0].float().cpu()}
            outs["fp16"] = g.half()(x.to(dev).half())[0].float().cpu()
    except RuntimeError as e:
        # The raise shape on a ROCm box with no C++ headers: RuntimeError out of _VF.gru
        # carrying "miopenStatusUnknownError" and NOTHING else — the reason lives only in
        # MIOpen's Warning-level log. Say what it means here.
        hint = ("" if ctx.get("hiprtc_headers_ok") else
                " — see hiprtc_cxx_headers above: this pack's run-time kernel compiler "
                "could not find the C++ headers it needs on this machine, which is what "
                "makes every RNN kernel fail")
        raise RuntimeError("%s: the GPU RNN kernel did not run (%s: %s)%s"
                           % (GPU_RNN_CODE, type(e).__name__, e, hint))
    parts = []
    for tag in ("fp32", "fp16"):
        y = outs[tag]
        if not torch.isfinite(y).all():
            raise RuntimeError("%s: %s GPU RNN output contains NaN/Inf" % (GPU_RNN_CODE, tag))
        rel = float((y - ref).abs().max()) / scale
        cos = float(torch.nn.functional.cosine_similarity(
            y.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item())
        # Both assertions are load-bearing and neither subsumes the other (measured):
        # cosine is blind to a uniform gain (2x scores 1.000000); rel is the coarser of
        # the two on a single-frame dropout.
        if not (rel < RNN_REL_TOL[tag]):
            raise RuntimeError("%s: %s GPU RNN deviates from the CPU reference: rel=%.3e "
                               "(limit %.0e, |ref|max %.3f)"
                               % (GPU_RNN_CODE, tag, rel, RNN_REL_TOL[tag], scale))
        if not (cos > RNN_MIN_COS):
            raise RuntimeError("%s: %s GPU RNN output does not track the CPU reference "
                               "(cosine %.6f, floor %.3f)"
                               % (GPU_RNN_CODE, tag, cos, RNN_MIN_COS))
        parts.append("%s rel %.2e cos %.6f" % (tag, rel, cos))
    return "BiGRU(128->96x2, 128 frames) vs CPU: " + ", ".join(parts)


def check_fallback_selftest(ctx):
    """Validates the CPU-fallback capture+classify harness itself — the ONE piece of
    §4.5(c) reachable WITHOUT an Intel GPU. Emit a synthetic PyTorch-shaped fallback
    warning, assert the parser extracts the op AND flags it hot, and assert an
    unrelated warning is NOT captured (no false-green from noise). The real trigger
    only fires on xpu silicon; here we prove the reporting logic is correct so the
    first xpu run only has to exercise the trigger, not debug the plumbing.
    Uses a LOCAL sink so nothing leaks into the global _FALLBACK_OPS."""
    local = set()
    with _capture_fallbacks(local):
        warnings.warn(
            "The operator 'aten::_fft_r2c' is not currently supported on the XPU "
            "backend and will fall back to run on the CPU. This may have performance "
            "implications.",
            UserWarning,
            stacklevel=1,
        )
    if "aten::_fft_r2c" not in local:
        raise RuntimeError("fallback capture failed: no operator extracted from the warning (got %s)" % sorted(local))
    if "aten::_fft_r2c" not in _classify_hot(local):
        raise RuntimeError("fallback classification failed: _fft_r2c was not flagged as a hot operator")
    noise = set()
    with _capture_fallbacks(noise):
        warnings.warn("some unrelated deprecation notice", DeprecationWarning, stacklevel=1)
    if noise:
        raise RuntimeError("unrelated warning wrongly captured: %s" % sorted(noise))
    return "capture+classify OK (synthetic aten::_fft_r2c -> hot; noise not captured)"


def check_fallback_ops(ctx):
    """§4.5(c): surface any aten op that silently ran on CPU because the xpu backend
    lacks it (accumulated by _capture_fallbacks around the on-device checks). Empty on
    cpu/cuda → skip/pass. A HOT op (conv/stft/fft) that fell back tanks training
    throughput → explicit WARN (visible, not a silent green); a cold op is tolerable
    (still WARN, but flagged as limited impact). Never a hard FAIL — a fallback is
    'neither green nor a crash' (design §4.5)."""
    # S172: skip on every NON-xpu tier, not just cpu. _FALLBACK_SIG only matches the
    # XPU-specific warning text, so on cuda/ROCm the sink is empty BY CONSTRUCTION —
    # and the old code turned that structural emptiness into the green sentence "every
    # operator ran natively on the device", which is a claim about something this check
    # never measured. An empty detector must report "not applicable", never "clean".
    if ctx["device"] != "xpu":
        return None
    ops = sorted(_FALLBACK_OPS)
    if not ops:
        return "no CPU fallbacks (every operator ran natively on the xpu device)"
    hot = sorted(_classify_hot(_FALLBACK_OPS))
    if hot:
        return _Warn("ENVTEST_OP_FALLBACK_HOT: %s | all fallbacks: %s"
                     % (", ".join(hot), ", ".join(ops)))
    return _Warn("ENVTEST_OP_FALLBACK_COLD: %s" % ", ".join(ops))


CHECKS = [
    ("python_info", check_python_info),
    ("imports", check_imports),
    ("cuda_driver", check_cuda_driver),
    # S169: must sit between cuda_driver and torch_backend — the first torch.cuda
    # touch is in check_torch_backend, so the arch-picked mask set here still
    # precedes HIP init. Its FAIL short-circuits like cuda_driver's (see main()).
    ("amd_device_pick", check_amd_device_pick),
    ("torch_backend", check_torch_backend),
    ("stft_roundtrip", check_stft_roundtrip),
    ("resample", check_resample),
    ("librosa_mel", check_librosa_mel),
    ("numba_jit", check_numba_jit),
    ("soundfile_roundtrip", check_soundfile_roundtrip),
    ("parselmouth_praat", check_parselmouth),
    ("pyworld_harvest", check_pyworld),
    ("onnxruntime_session", check_onnxruntime),
    ("faiss_search", check_faiss),
    ("sklearn_kmeans", check_sklearn_kmeans),
    ("lightning", check_lightning),
    ("tiny_gan", check_tiny_gan),
    ("dataloader_spawn", check_dataloader_spawn),
    ("gpu_stft_vs_cpu", check_gpu_stft_vs_cpu),
    ("gpu_amp_step", check_gpu_amp_step),
    # S172 — the two checks that close this tier's structural hole. Order matters: the
    # cause (can the run-time kernel compiler find its C++ headers?) runs BEFORE the
    # symptom (does an on-device RNN produce the right numbers?), so when both go red the
    # second one can point at the first instead of reporting a bare
    # `miopenStatusUnknownError`. Neither subsumes the other — measured, a WARM MIOpen
    # kernel cache makes the RNN pass on a machine whose compiler is broken, and a
    # payload that stopped shipping makes the header check fail on a machine whose RNN
    # is fine because it has Visual Studio.
    ("hiprtc_cxx_headers", check_hiprtc_cxx_headers),
    ("gpu_rnn_vs_cpu", check_gpu_rnn_vs_cpu),
    # last: after every on-device check has run under the fallback capture, so the
    # accumulator is fully populated before we summarize it.
    ("fallback_selftest", check_fallback_selftest),
    ("fallback_ops", check_fallback_ops),
]

# on-device checks run inside _capture_fallbacks so a hot op silently falling to CPU
# on xpu is recorded (not the selftest — it uses its own local sink — nor fallback_ops).
_FALLBACK_TIER = {"tiny_gan", "gpu_stft_vs_cpu", "gpu_amp_step", "gpu_rnn_vs_cpu"}


def main():
    global _reporter
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="report json path (pack_dir/envtest.json)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"])
    # S169: comma list of gfx archs the pack under test carries kernels for (AMD
    # packs only; Rust derives it from the pack's version). Empty = no arch-keyed
    # device pick, byte-identical to the pre-0.12.2 behaviour.
    ap.add_argument("--gfx-targets", default="", help="comma list, e.g. gfx1100,gfx1101,gfx1102,gfx1103")
    args = ap.parse_args()

    # Design §4.1/§4.5: enable the xpu CPU-fallback net + its debug logging BEFORE any
    # torch import (torch is imported lazily inside the checks, and protocol.py pulls in
    # no torch, so this is still pre-import). A missing xpu op then becomes a loud,
    # diagnosable CPU fallback (captured by check_fallback_ops) instead of a hard crash.
    # Harmless/ignored on cpu & cuda. setdefault: never clobber an explicit caller value
    # (the Rust launcher sets these too; doing it here makes a manual
    # `python -m utai_train.envtest --device xpu` behave identically, not hard-crash).
    os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_DEBUG_XPU_FALLBACK", "1")

    _reporter = Reporter()  # stdout/stderr UTF-8 reconfigure (single source in protocol.py)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    tmp_dir = os.path.join(out_dir, "envtest_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ctx = {
        "device": args.device,
        "tmp_dir": tmp_dir,
        "gfx_targets": [t for t in args.gfx_targets.split(",") if t],
    }

    items = []
    started = datetime.datetime.now().isoformat(timespec="seconds")
    for name, fn in CHECKS:
        t0 = time.monotonic()
        try:
            if name in _FALLBACK_TIER:
                with _capture_fallbacks(_FALLBACK_OPS):
                    detail = fn(ctx)
            else:
                detail = fn(ctx)
            if isinstance(detail, _Warn):
                status = "warn"
                detail = detail.detail
            elif detail is None:
                status = "skip"
                detail = "not applicable to this tier"
            else:
                status = "pass"
        except Exception as e:
            status = "fail"
            # The detail is what the UI shows the user (S74: the Settings pack list renders every
            # failed check's detail). Message + SOURCE LOCATION — the old form appended
            # traceback[-1], which IS "Type: message" verbatim, so every failure read as the same
            # sentence twice.
            tb = traceback.format_exc().strip().splitlines()
            loc = ""
            for ln in tb:
                s = ln.strip()
                if s.startswith("File "):
                    loc = s
            # "@" (not "|") separates the location: the Rust side joins MULTIPLE failed checks
            # with " | ", and a detail carrying the same separator makes the summary unreadable.
            detail = "%s: %s" % (type(e).__name__, e) + (" @ %s" % loc if loc else "")
        ms = int((time.monotonic() - t0) * 1000)
        item = {"name": name, "status": status, "detail": str(detail), "ms": ms}
        items.append(item)
        _emit({"type": "item", **item})
        if name in ("cuda_driver", "amd_device_pick") and status == "fail":
            # S68f / S169: every later GPU check would touch torch.cuda — on a
            # below-floor driver, or with no arch-covered HIP device to mask to,
            # that can crash the interpreter outright (see the two checks' docs).
            # One decisive fail + a clean report beats a 0xC0000005 with no report.
            break

    shutil.rmtree(tmp_dir, ignore_errors=True)
    failed = [i["name"] for i in items if i["status"] == "fail"]
    warned = [i["name"] for i in items if i["status"] == "warn"]
    report = {
        "schema": 1,
        "device": args.device,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "items": items,
        "failed_names": failed,
        # §4.5(c): warns do NOT flip overall — the Rust pass/fail gate stays unchanged
        # and a tolerable fallback never blocks a pack; the UI ambers off warnings_present.
        "warned_names": warned,
        "warnings_present": bool(warned),
        "fallback_ops": sorted(_FALLBACK_OPS),
        "overall": "fail" if failed else "pass",
        "versions": ctx.get("versions", {}),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    _emit({"type": "done", "overall": report["overall"], "failed": failed, "warned": warned})
    # Same hard-exit posture as runner.py — a lingering spawn worker must not hang us.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
