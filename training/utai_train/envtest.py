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
        raise RuntimeError("缺失/损坏的包: " + ", ".join(missing))
    return ", ".join("%s %s" % kv for kv in sorted(versions.items()))


def check_torch_backend(ctx):
    import torch

    dev = ctx["device"]
    info = "torch %s" % torch.__version__
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available()=False —— 驱动过旧或不匹配？"
                "（cu130 需 NVIDIA 驱动 ≥ 580.88）"
            )
        info += "; cuda %s; %s" % (torch.version.cuda, torch.cuda.get_device_name(0))
    elif dev == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError("torch.xpu 不可用 —— Intel 显卡驱动未装或不被支持")
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
        raise RuntimeError("重建误差过大: max_abs=%.3e (阈 1e-4)" % err)
    if spec.abs().sum().item() <= 0:
        raise RuntimeError("stft 输出全零")
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
        raise RuntimeError("输出长度 %d ≠ 16000" % y.shape[-1])
    if not torch.isfinite(y).all():
        raise RuntimeError("输出含 NaN/Inf")
    rms_in = float(x.pow(2).mean().sqrt())
    rms_out = float(y.pow(2).mean().sqrt())
    ratio = rms_out / max(rms_in, 1e-9)
    if not (0.7 < ratio < 1.4):
        raise RuntimeError("能量比异常 %.3f（重采样器数值损坏？）" % ratio)
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
        raise RuntimeError("mel 基形状异常 %s" % (basis.shape,))
    if not np.isfinite(basis).all() or basis.min() < 0:
        raise RuntimeError("mel 基含非法值")
    zero_rows = int((basis.sum(axis=1) <= 0).sum())
    if zero_rows:
        raise RuntimeError("%d 个空 mel 滤波器" % zero_rows)
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
        raise RuntimeError("JIT 结果错误 %.6f != %.6f" % (got, want))
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
        raise RuntimeError("读回形状/采样率不符")
    if not np.array_equal(back, data):
        raise RuntimeError("float32 wav 读写不是逐位一致")
    return "float32 逐位一致 (%d samples)" % data.shape[0]


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
        raise RuntimeError("几乎没有浊音帧被检出 (%d)" % voiced.size)
    med = float(np.median(voiced))
    if not (212.0 < med < 228.0):
        raise RuntimeError("f0 中位数 %.2f Hz 偏离 220 Hz" % med)
    ver = getattr(parselmouth, "PRAAT_VERSION", "?")
    return "praat %s, f0 median %.2f Hz (%d voiced)" % (ver, med, voiced.size)


def check_pyworld(ctx):
    import numpy as np
    import pyworld as pw

    x = _vowel_tone(44100, 220.0, 0.5)
    f0, _t = pw.harvest(x, 44100, f0_floor=80.0, f0_ceil=800.0)
    voiced = f0[f0 > 0]
    if voiced.size < f0.size // 2:
        raise RuntimeError("harvest 浊音帧过少 (%d/%d)" % (voiced.size, f0.size))
    med = float(np.median(voiced))
    if not (210.0 < med < 230.0):
        raise RuntimeError("harvest f0 中位数 %.2f Hz 偏离 220 Hz" % med)
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
        raise RuntimeError("Add 结果错误")
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
        raise RuntimeError("自身最近邻检索错位: %s" % i[:, 0].tolist())
    if float(d.max()) > 1e-4:
        raise RuntimeError("自身距离非零: %.3e" % float(d.max()))
    return "IndexFlatL2 self-NN 5/5"


def check_sklearn_kmeans(ctx):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.RandomState(11)
    pts = np.concatenate([rng.randn(80, 8) + 4.0, rng.randn(80, 8) - 4.0]).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=3, batch_size=64).fit(pts)
    centers = km.cluster_centers_
    if not np.isfinite(centers).all():
        raise RuntimeError("聚类中心含 NaN/Inf")
    spread = float(np.abs(centers).mean())
    if not (2.0 < spread < 6.0):
        raise RuntimeError("聚类中心异常（|mean|=%.2f，期望 ~4）" % spread)
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
        raise RuntimeError("Trainer.fit 未执行 training_step 或 loss 非有限")
    return "lightning %s Trainer.fit 1 step OK (loss %.4f)" % (
        lightning.__version__, probe.seen_loss)


def check_tiny_gan(ctx):
    """A miniature GAN with Conv1d + ConvTranspose1d — front/backward/optimizer on the
    op family HiFi-GAN training lives on (and the exact family with known MIOpen /
    XPU risk on non-NVIDIA backends). Asserts the L1 term actually LEARNS (robust
    decrease), gradients flow, everything stays finite."""
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
        opt_d.step()
        # G step (L1 dominates so the learning check is deterministic-ish)
        opt_g.zero_grad(set_to_none=True)
        l1 = (fake - target).abs().mean()
        loss_g = l1 + 0.1 * torch.mean((d(fake) - 1.0) ** 2)
        loss_g.backward()
        for p in g.parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                grad_seen = True
                break
        opt_g.step()
        if not (torch.isfinite(loss_d) and torch.isfinite(loss_g)):
            raise RuntimeError("loss 出现 NaN/Inf (step %d)" % _step)
        l1_hist.append(float(l1.detach()))
    first = sum(l1_hist[:3]) / 3.0
    last = sum(l1_hist[-3:]) / 3.0
    if not grad_seen:
        raise RuntimeError("G 从未收到非零梯度")
    if not (last < first * 0.8):
        raise RuntimeError("L1 未收敛（首 %.4f → 尾 %.4f，期望降 >20%%）" % (first, last))
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
        raise RuntimeError("spawn worker 数据错误 (batches=%d, sum=%.1f≠%.1f)" % (batches, total, expected))
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
    sd = torch.stft(xd, 2048, hop_length=512, window=win.to(dev), center=True, return_complex=True)
    err = (sd.cpu() - ref).abs().max().item()
    scale = ref.abs().max().item()
    if not (err < 1e-3 * max(scale, 1.0)):
        raise RuntimeError("GPU stft 偏离 CPU 参考: max_abs=%.3e (scale %.2f)" % (err, scale))
    return "max_abs=%.2e vs CPU (scale %.2f)" % (err, scale)


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
        with torch.amp.autocast(dev, dtype=torch.float16):
            y = model(x)
            loss = y.pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if not torch.isfinite(loss):
            raise RuntimeError("amp loss 非有限")
        return "fp16 autocast + GradScaler 一步 OK"
    if dev == "xpu":
        # bf16 WITHOUT GradScaler — Arc A 系无 fp64，scaler 必崩（设计 §4.1）。
        model = torch.nn.Conv1d(4, 4, 3, padding=1).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(2, 4, 128, device=dev)
        with torch.amp.autocast(dev, dtype=torch.bfloat16):
            loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError("bf16 loss 非有限")
        return "bf16 autocast (no scaler) 一步 OK"
    return None


CHECKS = [
    ("python_info", check_python_info),
    ("imports", check_imports),
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
]


def main():
    global _reporter
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="report json path (pack_dir/envtest.json)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"])
    args = ap.parse_args()

    _reporter = Reporter()  # stdout/stderr UTF-8 reconfigure (single source in protocol.py)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    tmp_dir = os.path.join(out_dir, "envtest_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ctx = {"device": args.device, "tmp_dir": tmp_dir}

    items = []
    started = datetime.datetime.now().isoformat(timespec="seconds")
    for name, fn in CHECKS:
        t0 = time.monotonic()
        try:
            detail = fn(ctx)
            status = "skip" if detail is None else "pass"
            detail = detail or "（该档位不适用）"
        except Exception as e:
            status = "fail"
            tb = traceback.format_exc().strip().splitlines()
            detail = "%s: %s | %s" % (type(e).__name__, e, tb[-1] if tb else "")
        ms = int((time.monotonic() - t0) * 1000)
        item = {"name": name, "status": status, "detail": str(detail), "ms": ms}
        items.append(item)
        _emit({"type": "item", **item})

    shutil.rmtree(tmp_dir, ignore_errors=True)
    failed = [i["name"] for i in items if i["status"] == "fail"]
    report = {
        "schema": 1,
        "device": args.device,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "items": items,
        "failed_names": failed,
        "overall": "fail" if failed else "pass",
        "versions": ctx.get("versions", {}),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    _emit({"type": "done", "overall": report["overall"], "failed": failed})
    # Same hard-exit posture as runner.py — a lingering spawn worker must not hang us.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
