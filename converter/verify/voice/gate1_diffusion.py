"""Gate 1 — SoVITS shallow-diffusion equivalence: ORIGINAL (so-vits-svc
4.1-Stable diffusion/{unit2mel,diffusion,wavenet}.py) vs our port
(architectures/sovits_diffusion.py) and its ONNX export (export_diffusion.py).

a.  REAL weights (东雪莲 Sovits4.1 扩散模型.pt, vec768l12/n_spk1/k_step_max0):
    original Unit2Mel.forward FULL SAMPLING vs our port, strict=True both
    sides, fixed-noise monkeypatch (torch.randn/randn_like -> one seeded
    generator; identical call order on both sides => identical noise), REAL
    gt_spec mel via the original Vocoder.extract on a real vocal snippet.
    ALL SIX methods: naive (method=None, k=20), ddim/pndm/dpm-solver/
    dpm-solver++/unipc (k=20 sp=5, exercising lower_order_final steps<10),
    PLUS dpm-solver++ k=100/sp=10 (template default, steps=10 => NO
    lower_order_final) PLUS an only-diffusion run (gt_spec=None, t=1000,
    sp=100). PASS: mel max_abs_diff < 1e-5. dpm/unipc on OUR side run through
    the ORIGINAL solver modules (we never port the solvers — Rust does).
b.  intermediate bisect helpers: cond (original embedding sum captured by a
    decoder stub vs our EncoderExport) and single denoiser calls at INT t=55
    (original int64 vs our f32 — the time-input deviation) and FLOAT t=13.7.
c.  our torch vs ORT: encoder + denoiser separately, export T=100 < 1e-4
    (denoiser at several float t incl. non-integer), dynamic-T sweep
    200/57/20/7 < 5e-4 (same rationale as gate1_sovits: ORT/torch fp32 conv
    rounding).
d.  shipping export via the export_diffusion.py CLI (yaml auto-resolved: the
    single *.yaml in the Chinese-named directory) -> sidecar == contract §2.3
    EXACTLY (every key incl. spec_min/spec_max from the ckpt buffers, resolved
    k_step_max, no extra keys) + schedule-check line in stdout + ORT load from
    bytes + shape sanity.
e.  refusals: unknown/log10 vocoder type, exotic encoder, config/weight
    mismatches, --out-dims mismatch, unknown schedule (mutated betas), yaml
    resolution (none/ambiguous/config.yaml/same-stem precedence), wrong
    model.type, CLI exit code.
f.  random-weight transplants for structures the real ckpt lacks:
    n_spk=8 — spk one-hot MatMul vs ORIGINAL forward's sid path AND
    spk_mix_dict fractional path (tol 1e-6), full sampling with sid, export
    with spk_mix input + ORT parity + Chinese speaker names in the sidecar;
    k_step_max=100 (timesteps=200) — shallow-only guards both sides, shallow
    sampling parity, export still succeeds and sidecar records k_step_max=100.

Run:  converter/.venv/Scripts/python.exe converter/verify/voice/gate1_diffusion.py
"""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Windows console default codepage (cp932/936) chokes on the report glyphs.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOVITS_REPO = r"D:\MyDev\so-vits-svc\so-vits-svc"
CONVERTER = r"D:\MyDev\Utai_v2-dev\converter"
DIFF_PT = r"D:\MyDev\TESTING\Sovits-SVC\东雪莲\Sovits4.1东雪莲扩散模型.pt"
DIFF_YAML = r"D:\MyDev\TESTING\Sovits-SVC\东雪莲\Sovits4.1东雪莲扩散配置文件.yaml"
NSF_HIFIGAN_CKPT = r"D:\MyDev\so-vits-svc\so-vits-svc\pretrain\nsf_hifigan\model"
VOCAL_WAV = r"D:\MyDev\TESTING\ikanaiteyo\vocal.wav"
SCRATCH = (r"C:\Users\admin\AppData\Local\Temp\claude\D--MyDev-Utai-v2-dev"
           r"\86733fc5-aabc-40de-8c29-9c294ceb0706\scratchpad\gate1_diffusion")
TEST_OUTPUT = os.path.join(CONVERTER, "test_output")

TOL_TORCH = 1e-5   # torch-vs-torch full sampling (gates a, f)
TOL_INTER = 1e-6   # torch-vs-torch intermediates / spk one-hot (gates b, f)
TOL_ONNX = 1e-4    # torch-vs-ORT fp32 at the export T (gate c, mandated)
TOL_ONNX_SWEEP = 5e-4
SWEEP_T = (200, 57, 20, 7)

sys.path.insert(0, SOVITS_REPO)
sys.path.insert(0, CONVERTER)

import yaml                                   # noqa: E402
import soundfile as sf                        # noqa: E402
from diffusion.unit2mel import (              # noqa: E402 — ORIGINAL repo
    Unit2Mel as OrigUnit2Mel,
    load_model_vocoder,
)
import utils as so_utils                      # noqa: E402 — ORIGINAL utils.py
from architectures import sovits_diffusion   # noqa: E402 — our port
import export_diffusion as export_mod        # noqa: E402

EXPORT_T = export_mod.EXPORT_T  # 100 — sweep values must differ from it
assert EXPORT_T not in SWEEP_T

os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(TEST_OUTPUT, exist_ok=True)
torch.set_grad_enabled(False)

failures = []


def check(cond, label):
    tag = "PASS" if cond else "FAIL"
    print(f"    [{tag}] {label}")
    if not cond:
        failures.append(label)


def mad(a, b):
    return (a - b).abs().max().item()


# --- noise patching --------------------------------------------------------
# Diffusion randomness: q_sample's randn_like (shallow init), forward's randn
# (only-diffusion init), p_sample's noise_like->randn (naive per-step). The
# DPM/UniPC/ddim/pndm inner loops are deterministic. Both sides run the SAME
# algorithm => identical randn/randn_like call order => one seeded generator
# reproduces identical noise on both sides.

class FixedNoise:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.gen = torch.Generator().manual_seed(self.seed)
        self._randn, self._randn_like = torch.randn, torch.randn_like
        gen, real_randn = self.gen, self._randn

        def fixed_randn(*size, **kw):
            if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
                size = tuple(size[0])
            dtype = kw.get("dtype")
            return real_randn(*size, generator=gen, dtype=dtype)

        def fixed_randn_like(t, **kw):
            return real_randn(tuple(t.shape), generator=gen, dtype=t.dtype)

        torch.randn, torch.randn_like = fixed_randn, fixed_randn_like
        return self

    def __exit__(self, *exc):
        torch.randn, torch.randn_like = self._randn, self._randn_like


class CaptureCond(torch.nn.Module):
    """Stands in for unit2mel.decoder: returns the condition tensor the
    decoder would receive ([B,T,n_hidden], pre-transpose) untouched."""

    def forward(self, x, **kwargs):
        return x


def orig_cond_of(orig_model, *fwd_args, **fwd_kwargs):
    """Runs the ORIGINAL Unit2Mel.forward with its decoder swapped for
    CaptureCond -> the original's exact embedding sum."""
    saved = orig_model.decoder
    orig_model.decoder = CaptureCond()
    try:
        x = orig_model(*fwd_args, **fwd_kwargs)
    finally:
        orig_model.decoder = saved
    return x


def make_cond_inputs(t, dim, seed, unvoiced=True):
    """units/f0/volume in the ORIGINAL forward layout ([1,T,dim], [1,T,1] Hz,
    [1,T,1]); 2-D views feed our export graphs."""
    gen = torch.Generator().manual_seed(seed)
    units = torch.randn(1, t, dim, generator=gen)
    tt = torch.linspace(0, 2 * math.pi, t)
    f0 = 220.0 * (2 ** (0.3 * torch.sin(tt)))[None, :]
    if unvoiced and t >= 20:
        f0[:, t // 5:t // 5 + 8] = 0.0  # unvoiced stretch: log(1+0/700)=0 path
    volume = (0.5 * torch.rand(1, t, generator=gen)) ** 2
    return units, f0[:, :, None], volume[:, :, None]


# ===========================================================================
print("=== gate (a) — real weights: 东雪莲扩散模型, full-sampling all methods ===")

# The model's own yaml points vocoder.ckpt at a MISSING finetuned path; the
# original loader needs a real vocoder (mel extraction), so patch a temp yaml
# with an ABSOLUTE path to the repo's community pretrain.
cfg_real = yaml.safe_load(Path(DIFF_YAML).read_text(encoding="utf-8"))
cfg_patched = copy.deepcopy(cfg_real)
cfg_patched["vocoder"]["ckpt"] = NSF_HIFIGAN_CKPT
patched_yaml = os.path.join(SCRATCH, "diff_config_patched.yaml")
Path(patched_yaml).write_text(yaml.safe_dump(cfg_patched), encoding="utf-8")

orig, vocoder, _args = load_model_vocoder(DIFF_PT, device="cpu",
                                          config_path=patched_yaml)
orig.eval()
print("    orig strict load OK (load_model_vocoder)")

ck = torch.load(DIFF_PT, map_location="cpu", weights_only=False)
ours, meta = sovits_diffusion.build_from_checkpoint(ck, cfg_real)
print("    ours strict=True load OK")
check(meta["encoder"] == "vec768l12" and meta["encoder_out_channels"] == 768
      and meta["timesteps"] == 1000 and meta["k_step_max"] == 1000
      and meta["n_spk"] == 1 and meta["use_pitch_aug"]
      and meta["spec_min"] == [-12.0] and meta["spec_max"] == [2.0]
      and meta["infer_method"] == "dpm-solver++" and meta["infer_speedup"] == 10
      and meta["speakers"] == {"AzumaVocal": 0}
      and meta["unit_interpolate_mode"] == "nearest",
      f"meta: vec768l12/1000/k_step_max1000/n_spk1/aug/spec[-12,2] (got {meta})")
sc = meta["schedule_check"]
check(sc["betas_max_abs_diff"] < 1e-7 and sc["alphas_cumprod_max_abs_diff"] < 1e-7,
      f"schedule recompute vs ckpt buffers: betas {sc['betas_max_abs_diff']:.3e}, "
      f"alphas_cumprod {sc['alphas_cumprod_max_abs_diff']:.3e} < 1e-7")

# REAL gt_spec: original Vocoder.extract mel of a real vocal snippet.
raw, sr = sf.read(VOCAL_WAV, dtype="float32", always_2d=True)
mono = raw.mean(axis=1)
if sr != 44100:
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(44100, int(sr))
    mono = resample_poly(mono, 44100 // g, int(sr) // g).astype(np.float32)
seg_len = int(1.5 * 44100) // 512 * 512
best_off, best_rms = 0, -1.0
for off_s in (5.0, 10.0, 20.0, 40.0, 0.0):
    off = int(off_s * 44100)
    if off + seg_len > len(mono):
        continue
    rms = float(np.sqrt(np.mean(mono[off:off + seg_len] ** 2)))
    if rms > best_rms:
        best_off, best_rms = off, rms
    if rms > 0.02:
        break
seg = torch.from_numpy(mono[best_off:best_off + seg_len].copy())[None, :]
gt_mel = vocoder.extract(seg, 44100)  # [1, T, 128] via ORIGINAL nvSTFT
T_REAL = gt_mel.shape[1]
check(gt_mel.shape == (1, seg_len // 512, 128) and torch.isfinite(gt_mel).all()
      and best_rms > 1e-3,
      f"real gt_spec mel: shape {tuple(gt_mel.shape)}, offset {best_off / 44100:.1f}s, "
      f"rms {best_rms:.3f}")

units, f0, vol3 = make_cond_inputs(T_REAL, 768, 1234)
# real volume from the ORIGINAL Volume_Extractor on the same snippet
vol_real = so_utils.Volume_Extractor(512).extract(seg)
check(vol_real.shape == (T_REAL,), f"Volume_Extractor frames == mel frames ({T_REAL})")
vol3 = vol_real[None, :, None]

# (method, infer_speedup, k_step, gt_spec?) — cheap-but-nontrivial params.
RUNS = [
    ("naive(None) k=20", None, 1, 20, True),
    ("ddim k=20 sp=5", "ddim", 5, 20, True),
    ("pndm k=20 sp=5", "pndm", 5, 20, True),
    ("dpm-solver k=20 sp=5", "dpm-solver", 5, 20, True),
    ("dpm-solver++ k=20 sp=5", "dpm-solver++", 5, 20, True),
    ("unipc k=20 sp=5", "unipc", 5, 20, True),
    ("dpm-solver++ k=100 sp=10 (template default)", "dpm-solver++", 10, 100, True),
    ("only-diffusion dpm-solver++ sp=100 (gt_spec=None, t=1000)",
     "dpm-solver++", 100, 100, False),
]
for i, (label, method, speedup, k_step, with_gt) in enumerate(RUNS):
    gt = gt_mel if with_gt else None
    seed = 9000 + i
    with FixedNoise(seed):
        o_mel = orig(units, f0, vol3, gt_spec=gt, infer=True,
                     infer_speedup=speedup, method=method, k_step=k_step,
                     use_tqdm=False)
    with FixedNoise(seed):
        m_mel = ours(units, f0, vol3, gt_spec=gt, infer=True,
                     infer_speedup=speedup, method=method, k_step=k_step,
                     use_tqdm=False)
    d = mad(o_mel, m_mel)
    check(d < TOL_TORCH and o_mel.shape == (1, T_REAL, 128),
          f"(a) {label}: mel max_abs_diff {d:.3e} < {TOL_TORCH:.0e}")

# ===========================================================================
print("=== gate (b) — intermediates: cond + single denoiser steps ===")
enc_ours = sovits_diffusion.EncoderExport(ours)
den_ours = sovits_diffusion.DenoiserExport(ours)

cond_orig = orig_cond_of(orig, units, f0, vol3, gt_spec=gt_mel, k_step=20,
                         use_tqdm=False).transpose(1, 2)
cond_ours = enc_ours(units, f0[:, :, 0], vol3[:, :, 0])
d = mad(cond_orig, cond_ours)
check(d < TOL_INTER, f"(b) cond (orig embedding sum vs EncoderExport): "
                     f"{d:.3e} < {TOL_INTER:.0e}")

gen = torch.Generator().manual_seed(4321)
x_step = torch.randn(1, 1, 128, T_REAL, generator=gen)
for t_orig, t_ours, label in (
        (torch.tensor([55], dtype=torch.long), torch.tensor([55.0]), "int t=55 (orig int64 vs ours f32)"),
        (torch.tensor([13.7]), torch.tensor([13.7]), "float t=13.7"),
        (torch.tensor([456.789]), torch.tensor([456.789]), "float t=456.789")):
    o_eps = orig.decoder.denoise_fn(x_step, t_orig, cond_orig)
    m_eps = den_ours(x_step, t_ours, cond_ours)
    d = mad(o_eps, m_eps)
    check(d < TOL_INTER, f"(b) denoiser single step {label}: {d:.3e} < {TOL_INTER:.0e}")

# ===========================================================================
print("=== gate (c) — our torch vs ORT (encoder & denoiser) ===")
import onnxruntime as ort  # noqa: E402


def make_session(path):
    """ORT session from BYTES — python onnxruntime cannot open Chinese
    session paths on Windows (Rust's ort uses wide strings, unaffected)."""
    return ort.InferenceSession(Path(path).read_bytes(),
                                providers=["CPUExecutionProvider"])


export_dir = os.path.join(SCRATCH, "export_test")
export_mod.export_diffusion_assets(DIFF_PT, export_dir, config_path=DIFF_YAML)
enc_sess = make_session(os.path.join(export_dir, "encoder.onnx"))
den_sess = make_session(os.path.join(export_dir, "denoiser.onnx"))
check([i.name for i in enc_sess.get_inputs()] == ["units", "f0", "volume"],
      "encoder.onnx inputs == [units, f0, volume] (n_spk=1: no spk_mix)")
check([i.name for i in den_sess.get_inputs()] == ["x", "time", "cond"]
      and den_sess.get_inputs()[1].type == "tensor(float)",
      "denoiser.onnx inputs == [x, time, cond], time is f32")


def enc_diff(t, seed):
    u, f, v = make_cond_inputs(t, 768, seed)
    t_out = enc_ours(u, f[:, :, 0], v[:, :, 0])
    o_out = enc_sess.run(None, {"units": u.numpy(), "f0": f[:, :, 0].numpy(),
                                "volume": v[:, :, 0].numpy()})[0]
    return mad(t_out, torch.from_numpy(o_out))


def den_diff(t, t_val, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, 128, t, generator=g)
    cond = torch.randn(1, 256, t, generator=g)
    tv = torch.tensor([t_val], dtype=torch.float32)
    t_out = den_ours(x, tv, cond)
    o_out = den_sess.run(None, {"x": x.numpy(), "time": tv.numpy(),
                                "cond": cond.numpy()})[0]
    return mad(t_out, torch.from_numpy(o_out))


d = enc_diff(EXPORT_T, 42)
check(d < TOL_ONNX, f"(c) encoder T={EXPORT_T}: {d:.3e} < {TOL_ONNX:.0e}")
for t_val in (0.0, 1.0, 13.7, 55.0, 456.789, 999.0):
    d = den_diff(EXPORT_T, t_val, 7000 + int(t_val))
    check(d < TOL_ONNX, f"(c) denoiser T={EXPORT_T} t={t_val}: {d:.3e} < {TOL_ONNX:.0e}")
print(f"    dynamic-T sweep (graphs exported at T={EXPORT_T}):")
for t in SWEEP_T:
    d = enc_diff(t, 100 + t)
    check(d < TOL_ONNX_SWEEP, f"(c) encoder T={t}: {d:.3e} < {TOL_ONNX_SWEEP:.0e}")
    d = den_diff(t, 13.7, 200 + t)
    check(d < TOL_ONNX_SWEEP, f"(c) denoiser T={t} t=13.7: {d:.3e} < {TOL_ONNX_SWEEP:.0e}")

# ===========================================================================
print("=== gate (d) — shipping export via export_diffusion.py CLI ===")
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
ship_dir = os.path.join(TEST_OUTPUT, "Sovits4.1东雪莲扩散模型.diffusion")
proc = subprocess.run(
    [sys.executable, "export_diffusion.py", "--input", DIFF_PT,
     "--outdir", ship_dir],
    cwd=CONVERTER, capture_output=True, env=env)
stdout_txt = proc.stdout.decode("utf-8", errors="replace")
print("    " + "\n    ".join(stdout_txt.strip().splitlines()[-4:]))
check(proc.returncode == 0, f"CLI exit code {proc.returncode}")
if proc.returncode != 0:
    print(proc.stderr.decode("utf-8", errors="replace"))
check("schedule check" in stdout_txt and "max_abs_diff" in stdout_txt,
      "CLI stdout contains the schedule recompute check")

EXPECT_SIDECAR = {
    "type": "sovits_diffusion",
    "encoder": "vec768l12", "encoder_out_channels": 768,
    "sample_rate": 44100, "block_size": 512,
    "n_layers": 20, "n_chans": 512, "n_hidden": 256,
    "timesteps": 1000, "k_step_max": 1000,          # yaml 0 -> timesteps
    "schedule": "linear", "max_beta": 0.02,
    "spec_min": [-12.0], "spec_max": [2.0],          # from the ckpt buffers
    "n_spk": 1, "speakers": {"AzumaVocal": 0},
    "use_pitch_aug": True,
    "infer_method": "dpm-solver++", "infer_speedup": 10,
    "unit_interpolate_mode": "nearest",
    "files": {"encoder": "encoder.onnx", "denoiser": "denoiser.onnx"},
}
sidecar = json.loads(Path(ship_dir, "diffusion.json").read_text(encoding="utf-8"))
for k, v in EXPECT_SIDECAR.items():
    check(sidecar.get(k) == v, f"sidecar {k} == {v!r} (got {sidecar.get(k)!r})")
check(set(sidecar.keys()) == set(EXPECT_SIDECAR.keys()),
      f"sidecar schema EXACT — no missing/extra keys "
      f"(extra: {set(sidecar) - set(EXPECT_SIDECAR)}, "
      f"missing: {set(EXPECT_SIDECAR) - set(sidecar)})")

for name, feed_shapes in (("encoder.onnx", None), ("denoiser.onnx", None)):
    sess = make_session(os.path.join(ship_dir, name))
    t = 77
    if name == "encoder.onnx":
        u, f, v = make_cond_inputs(t, 768, 555)
        out = sess.run(None, {"units": u.numpy(), "f0": f[:, :, 0].numpy(),
                              "volume": v[:, :, 0].numpy()})[0]
        ok = out.shape == (1, 256, t) and np.isfinite(out).all()
    else:
        g = torch.Generator().manual_seed(556)
        out = sess.run(None, {
            "x": torch.randn(1, 1, 128, t, generator=g).numpy(),
            "time": np.array([13.7], dtype=np.float32),
            "cond": torch.randn(1, 256, t, generator=g).numpy()})[0]
        ok = out.shape == (1, 1, 128, t) and np.isfinite(out).all()
    check(ok, f"shipping {name}: ORT-from-bytes T={t} shape/finite OK ({out.shape})")

# ===========================================================================
print("=== gate (e) — refusals ===")


def expect_refusal(fn, expect_substr, label):
    try:
        fn()
        check(False, f"{label}: refused")
    except ValueError as e:
        check(expect_substr in str(e), f"{label}: {e}")


def mutated_cfg(mutator):
    cfg2 = copy.deepcopy(cfg_real)
    mutator(cfg2)
    return cfg2


expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(
    ck, mutated_cfg(lambda c: c["vocoder"].update(type="hifigan"))),
    "暂不支持 vocoder type", "unknown vocoder type")
expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(
    ck, mutated_cfg(lambda c: c["vocoder"].update(type="nsf-hifigan-log10"))),
    "暂不支持 vocoder type", "nsf-hifigan-log10 refused")
expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(
    ck, mutated_cfg(lambda c: c["data"].update(encoder="whisper-ppg"))),
    "暂不支持 encoder", "exotic encoder")
expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(
    ck, mutated_cfg(lambda c: c["model"].update(n_layers=18))),
    "配置文件与模型不匹配", "config/weights n_layers mismatch")
expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(
    ck, mutated_cfg(lambda c: c["data"].update(encoder="vec256l9",
                                               encoder_out_channels=256))),
    "配置文件与模型不匹配", "encoder dim mismatch (vec256l9 on a 768 model)")
expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(ck, cfg_real, out_dims=80),
               "mel 维度", "--out-dims mismatch")
expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(ck, None),
               "必须带配置", "missing config dict")

sd_bad = dict(ck["model"])
sd_bad["decoder.betas"] = torch.linspace(1e-4, 0.05, 1000)
expect_refusal(lambda: sovits_diffusion.build_from_checkpoint(
    {"model": sd_bad}, cfg_real),
    "未知的扩散调度", "unknown schedule (mutated betas)")

# yaml resolution matrix (load_diffusion_config never opens the .pt)
MINIMAL_CFG = {"model": {"type": "Diffusion"}, "data": {}, "vocoder": {}}


def yaml_dir(name, files):
    d = Path(SCRATCH) / name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.yaml"):
        old.unlink()
    for fname in files:
        (d / fname).write_text(yaml.safe_dump(MINIMAL_CFG), encoding="utf-8")
    return d / "模型.pt"


expect_refusal(lambda: sovits_diffusion.load_diffusion_config(
    yaml_dir("yres_none", [])), "未找到扩散模型的配置文件", "no yaml in dir")
expect_refusal(lambda: sovits_diffusion.load_diffusion_config(
    yaml_dir("yres_ambig", ["a.yaml", "b.yaml"])),
    "无法确定", "ambiguous yamls")
_, p = sovits_diffusion.load_diffusion_config(
    yaml_dir("yres_cfg", ["a.yaml", "config.yaml"]))
check(p.name == "config.yaml", f"config.yaml beats ambiguity (got {p.name})")
_, p = sovits_diffusion.load_diffusion_config(
    yaml_dir("yres_stem", ["模型.yaml", "config.yaml", "other.yaml"]))
check(p.name == "模型.yaml", f"same-stem yaml wins (got {p.name})")
expect_refusal(lambda: sovits_diffusion.load_diffusion_config(
    DIFF_PT, explicit_config=os.path.join(SCRATCH, "missing.yaml")),
    "不存在", "--config path missing")

wrongtype = Path(SCRATCH) / "wrongtype.yaml"
wrongtype.write_text(yaml.safe_dump(
    {"model": {"type": "RectifiedFlow"}, "data": {}, "vocoder": {}}),
    encoding="utf-8")
expect_refusal(lambda: sovits_diffusion.load_diffusion_config(
    DIFF_PT, explicit_config=str(wrongtype)),
    "不是扩散模型的配置文件", "model.type != Diffusion")

# CLI error path: bad yaml -> exit 1 + Chinese stderr
badvoc = Path(SCRATCH) / "badvoc.yaml"
badvoc.write_text(yaml.safe_dump(mutated_cfg(
    lambda c: c["vocoder"].update(type="hifigan"))), encoding="utf-8")
proc = subprocess.run(
    [sys.executable, "export_diffusion.py", "--input", DIFF_PT,
     "--config", str(badvoc), "--outdir", os.path.join(SCRATCH, "refused_out")],
    cwd=CONVERTER, capture_output=True, env=env)
stderr_txt = proc.stderr.decode("utf-8", errors="replace")
check(proc.returncode == 1 and "错误" in stderr_txt and "暂不支持 vocoder" in stderr_txt,
      f"CLI refusal: exit {proc.returncode}, stderr carries the Chinese error")

# ===========================================================================
print("=== gate (f) — random-weight transplants: n_spk=8 + k_step_max<timesteps ===")

# ---- n_spk=8 (spk_embed exists; real ckpt is single-speaker) ----
torch.manual_seed(20260705)
orig_spk = OrigUnit2Mel(256, 8, use_pitch_aug=False, out_dims=16, n_layers=2,
                        n_chans=32, n_hidden=24, timesteps=200, k_step_max=0)
# WaveNet output_projection weights are zero-init — randomize so the denoiser
# math is exercised (a wrong wiring cannot hide behind a constant output).
orig_spk.decoder.denoise_fn.output_projection.weight.data.normal_(0, 0.05)
orig_spk.eval()
CFG_SPK = {
    "model": {"type": "Diffusion", "n_spk": 8, "n_layers": 2, "n_chans": 32,
              "n_hidden": 24, "timesteps": 200, "k_step_max": 0,
              "use_pitch_aug": False},
    "data": {"encoder": "vec256l9", "encoder_out_channels": 256,
             "sampling_rate": 44100, "block_size": 512,
             "unit_interpolate_mode": "left"},
    "vocoder": {"type": "nsf-hifigan", "ckpt": "unused"},
    "infer": {"method": "unipc", "speedup": 5},
    "spk": {f"歌手{i}": i for i in range(8)},
}
ck_spk = {"model": orig_spk.state_dict()}
ours_spk, meta_spk = sovits_diffusion.build_from_checkpoint(ck_spk, CFG_SPK, out_dims=16)
print("    ours strict=True transplant load OK "
      f"({len(ck_spk['model'])} tensors, spk_embed[8,24])")
check(meta_spk["n_spk"] == 8 and meta_spk["k_step_max"] == 200
      and meta_spk["out_dims"] == 16 and not meta_spk["use_pitch_aug"],
      f"meta: n_spk 8, k_step_max 200 (yaml 0), out_dims 16, no aug")

T_SPK = 40
u_s, f_s, v_s = make_cond_inputs(T_SPK, 256, 777)
enc_spk = sovits_diffusion.EncoderExport(ours_spk)
for k in (0, 3, 7):
    x_o = orig_cond_of(orig_spk, u_s, f_s, v_s,
                       spk_id=torch.LongTensor([[k]]), use_tqdm=False)
    one_hot = torch.zeros(T_SPK, 8)
    one_hot[:, k] = 1.0
    c_m = enc_spk(u_s, f_s[:, :, 0], v_s[:, :, 0], one_hot)
    d = mad(x_o.transpose(1, 2), c_m)
    check(d < TOL_INTER,
          f"(f) spk one-hot MatMul vs ORIGINAL sid=[[{k}]]: {d:.3e} < {TOL_INTER:.0e}")

mix_dict = {0: 0.5, 3: 0.3, 7: 0.2}
x_o = orig_cond_of(orig_spk, u_s, f_s, v_s, spk_mix_dict=mix_dict, use_tqdm=False)
mix_rows = torch.zeros(T_SPK, 8)
for k, v in mix_dict.items():
    mix_rows[:, k] = v
c_m = enc_spk(u_s, f_s[:, :, 0], v_s[:, :, 0], mix_rows)
d = mad(x_o.transpose(1, 2), c_m)
check(d < TOL_INTER,
      f"(f) fractional spk_mix vs ORIGINAL spk_mix_dict: {d:.3e} < {TOL_INTER:.0e}")

# full-sampling parity through the sid path of BOTH Unit2Mel ports
gt_small = -12.0 + 14.0 * torch.rand(1, T_SPK, 16,
                                     generator=torch.Generator().manual_seed(778))
for label, method, speedup in (("ddim", "ddim", 5), ("naive(None)", None, 1)):
    with FixedNoise(31337):
        o_mel = orig_spk(u_s, f_s, v_s, spk_id=torch.LongTensor([[3]]),
                         gt_spec=gt_small, infer_speedup=speedup, method=method,
                         k_step=20, use_tqdm=False)
    with FixedNoise(31337):
        m_mel = ours_spk(u_s, f_s, v_s, spk_id=torch.LongTensor([[3]]),
                         gt_spec=gt_small, infer_speedup=speedup, method=method,
                         k_step=20, use_tqdm=False)
    d = mad(o_mel, m_mel)
    check(d < TOL_TORCH, f"(f) n_spk=8 full sampling {label} sid=3: {d:.3e} < {TOL_TORCH:.0e}")

# export the transplant: spk_mix input + Chinese speakers in the sidecar
spk_pt = os.path.join(SCRATCH, "移植_spk8.pt")
spk_yaml = os.path.join(SCRATCH, "移植_spk8.yaml")
torch.save(ck_spk, spk_pt)
Path(spk_yaml).write_text(yaml.safe_dump(CFG_SPK, allow_unicode=True),
                          encoding="utf-8")
spk_outdir = os.path.join(SCRATCH, "移植_spk8.diffusion")
spk_sidecar = export_mod.export_diffusion_assets(spk_pt, spk_outdir,
                                                 config_path=spk_yaml, out_dims=16)
check(spk_sidecar["n_spk"] == 8 and spk_sidecar["speakers"]["歌手7"] == 7
      and spk_sidecar["k_step_max"] == 200 and spk_sidecar["infer_method"] == "unipc",
      f"spk sidecar: n_spk/speakers(中文)/k_step_max/infer_method "
      f"(got n_spk={spk_sidecar['n_spk']}, spk={spk_sidecar['speakers']})")
spk_enc_sess = make_session(os.path.join(spk_outdir, "encoder.onnx"))
check([i.name for i in spk_enc_sess.get_inputs()] == ["units", "f0", "volume", "spk_mix"],
      "transplant encoder.onnx has the spk_mix input")
one_hot = torch.zeros(T_SPK, 8)
one_hot[:, 3] = 1.0
ort_cond = spk_enc_sess.run(None, {
    "units": u_s.numpy(), "f0": f_s[:, :, 0].numpy(),
    "volume": v_s[:, :, 0].numpy(), "spk_mix": one_hot.numpy()})[0]
x_o = orig_cond_of(orig_spk, u_s, f_s, v_s, spk_id=torch.LongTensor([[3]]),
                   use_tqdm=False)
d = mad(x_o.transpose(1, 2), torch.from_numpy(ort_cond))
check(d < TOL_ONNX,
      f"(f) ORT spk one-hot vs ORIGINAL sid path: {d:.3e} < {TOL_ONNX:.0e}")

# ---- k_step_max=100 < timesteps=200 (shallow-only model) ----
torch.manual_seed(20260706)
orig_ks = OrigUnit2Mel(256, 1, use_pitch_aug=True, out_dims=16, n_layers=2,
                       n_chans=32, n_hidden=24, timesteps=200, k_step_max=100)
orig_ks.decoder.denoise_fn.output_projection.weight.data.normal_(0, 0.05)
orig_ks.eval()
CFG_KS = copy.deepcopy(CFG_SPK)
CFG_KS["model"].update(n_spk=1, k_step_max=100, use_pitch_aug=True)
CFG_KS["spk"] = {"solo": 0}
ck_ks = {"model": orig_ks.state_dict()}
ours_ks, meta_ks = sovits_diffusion.build_from_checkpoint(ck_ks, CFG_KS, out_dims=16)
check(meta_ks["k_step_max"] == 100 and meta_ks["timesteps"] == 200,
      f"meta: shallow-only k_step_max 100 / timesteps 200 (got {meta_ks['k_step_max']})")

# original raises on gt_spec=None; ours must too (guard parity)
try:
    orig_ks(u_s, f_s, v_s, gt_spec=None, method="ddim", infer_speedup=5,
            k_step=20, use_tqdm=False)
    check(False, "(f) ORIGINAL shallow-only guard fired on gt_spec=None")
except Exception as e:
    check("can not infer alone" in str(e),
          f"(f) ORIGINAL shallow-only guard: {e}")
expect_refusal(lambda: ours_ks(u_s, f_s, v_s, gt_spec=None, method="ddim",
                               infer_speedup=5, k_step=20, use_tqdm=False),
               "仅支持浅扩散", "(f) ours shallow-only guard on gt_spec=None")
expect_refusal(lambda: ours_ks(u_s, f_s, v_s, gt_spec=gt_small, method="ddim",
                               infer_speedup=5, k_step=150, use_tqdm=False),
               "k_step 超过", "(f) ours k_step > k_step_max guard")

with FixedNoise(555):
    o_mel = orig_ks(u_s, f_s, v_s, gt_spec=gt_small, method="ddim",
                    infer_speedup=5, k_step=20, use_tqdm=False)
with FixedNoise(555):
    m_mel = ours_ks(u_s, f_s, v_s, gt_spec=gt_small, method="ddim",
                    infer_speedup=5, k_step=20, use_tqdm=False)
d = mad(o_mel, m_mel)
check(d < TOL_TORCH, f"(f) shallow-only model ddim k=20: {d:.3e} < {TOL_TORCH:.0e}")

# export still succeeds (shallow-only is a legal shipping asset) — via CLI
ks_pt = os.path.join(SCRATCH, "移植_ks100.pt")
torch.save(ck_ks, ks_pt)
ks_yaml = os.path.join(SCRATCH, "移植_ks100.yaml")
Path(ks_yaml).write_text(yaml.safe_dump(CFG_KS, allow_unicode=True),
                         encoding="utf-8")
ks_outdir = os.path.join(SCRATCH, "移植_ks100.diffusion")
proc = subprocess.run(
    [sys.executable, "export_diffusion.py", "--input", ks_pt,
     "--config", ks_yaml, "--outdir", ks_outdir, "--out-dims", "16"],
    cwd=CONVERTER, capture_output=True, env=env)
check(proc.returncode == 0, f"(f) shallow-only CLI export exit {proc.returncode}")
if proc.returncode != 0:
    print(proc.stderr.decode("utf-8", errors="replace"))
else:
    ks_sidecar = json.loads(Path(ks_outdir, "diffusion.json")
                            .read_text(encoding="utf-8"))
    check(ks_sidecar["k_step_max"] == 100 and ks_sidecar["timesteps"] == 200
          and ks_sidecar["use_pitch_aug"] is True,
          f"(f) shallow-only sidecar k_step_max == 100 (got {ks_sidecar['k_step_max']})")

# ===========================================================================
print()
if failures:
    print(f"GATE FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL DIFFUSION GATES PASSED")
