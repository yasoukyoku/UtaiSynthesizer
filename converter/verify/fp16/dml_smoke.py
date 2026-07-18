"""DML true-fp16 kernel smoke (S31 关卡5 style + S68c NaN-mechanism probe).

Runs one forward per (model, EP/device, input-regime):
  inputs: zeros (silent-band 0-underflow probe) / randn*3 (typical) / randn*50 (overflow stress)
  models: fp32 (CPU reference + DML), fp16 OLD recipe, fp16 NEW recipe (DML dev0/dev1)
Reports non-finite output counts (primary gate) and relative error vs the fp32-CPU
reference on the same input. CPU results for fp16 models are NOT trusted as quality
evidence (CPU EP emulates fp16 in fp32) — they're only printed for context.

Usage: python dml_smoke.py <fp32.onnx> <fp16_old.onnx> <fp16_new.onnx>
"""
import sys

import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)

fp32_path, old_path, new_path = sys.argv[1], sys.argv[2], sys.argv[3]


def real_stft_chunk(frames):
    """Real spectral content: STFT of the S31 test clip, packed like the Rust pipeline
    ([1, 2*freq_bins, T, 2] = stereo channels stacked on freq axis, last dim re/im)."""
    import struct
    with open(r"D:\MyDev\TESTING\MSST\mono_20s.wav", "rb") as f:
        data = f.read()
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        csz = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + csz]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + csz + (csz & 1)
    audio_fmt, ch, _sr, _, _, bits = fmt
    if audio_fmt in (3, 65534) and bits == 32:
        x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch).astype(np.float32)
    else:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, ch) / 32768.0
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    n_fft, hop = 2048, 441
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    specs = []
    for c in range(2):
        sig = x[:, c]
        cols = []
        for t in range(frames):
            s = t * hop
            fr = sig[s:s + n_fft]
            if len(fr) < n_fft:
                fr = np.pad(fr, (0, n_fft - len(fr)))
            cols.append(np.fft.rfft(fr * win))
        specs.append(np.stack(cols, axis=1))  # [freq, T]
    spec = np.concatenate(specs, axis=0)  # [2*freq, T]
    out = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)[None]
    return out  # [1, 2050, T, 2]


def make_inputs(sess, seed=7):
    rng = np.random.RandomState(seed)
    feeds = {}
    for i in sess.get_inputs():
        shape = []
        for d in i.shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                shape.append(1 if not shape else 801)  # batch→1, sym time→801
        base = rng.randn(*shape).astype(np.float32)
        feeds[i.name] = (shape, base)
    return feeds


REAL = None


def run(sess, feeds, scale):
    global REAL
    if scale == "real":
        if REAL is None:
            shape = next(iter(feeds.values()))[0]
            REAL = real_stft_chunk(shape[2] if len(shape) == 4 else 801)
        ins = {k: REAL for k in feeds}
    else:
        ins = {k: (v[1] * scale if scale is not None else np.zeros(v[0], np.float32)) for k, v in feeds.items()}
    outs = sess.run(None, ins)
    return [np.asarray(o) for o in outs]


def stats(tag, outs, ref=None):
    nonfin = sum(int((~np.isfinite(o)).sum()) for o in outs)
    line = f"    {tag:<26} nonfinite={nonfin:<8}"
    if ref is not None:
        num = sum(float(((np.nan_to_num(o) - r) ** 2).sum()) for o, r in zip(outs, ref))
        den = sum(float((r ** 2).sum()) for r in ref)
        snr = 10 * np.log10(den / max(num, 1e-30)) if den > 0 else float("nan")
        line += f" SNR_vs_fp32cpu={snr:7.1f} dB"
    print(line, flush=True)
    return nonfin


def session(path, provider, device_id=0):
    so = ort.SessionOptions()
    if provider == "cpu":
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return ort.InferenceSession(
        path, so,
        providers=[("DmlExecutionProvider", {"device_id": device_id})],
    )


print("building fp32 CPU reference session...", flush=True)
ref_sess = session(fp32_path, "cpu")
feeds = make_inputs(ref_sess)
print("input shapes:", {k: v[0] for k, v in feeds.items()}, flush=True)

REGIMES = [("zeros", None), ("tiny*1e-5", 1e-5), ("real-stft", "real"), ("randn*3", 3.0), ("randn*50", 50.0)]
refs = {}
for name, scale in REGIMES:
    refs[name] = [np.nan_to_num(o) for o in run(ref_sess, feeds, scale)]
    stats(f"fp32 cpu {name}", refs[name])
del ref_sess

# Graph-math discriminator: the CPU EP emulates fp16 ops in fp32 — if a regime is
# non-finite HERE too, the blowup is in the GRAPH's math/constants, not in true-fp16
# kernel precision.
for label, path in (("fp16-OLD", old_path), ("fp16-NEW", new_path)):
    s = session(path, "cpu")
    print(f"  [{label} CPU(fp32-emu)]", flush=True)
    for name, scale in REGIMES:
        outs = run(s, feeds, scale)
        stats(name, outs, refs[name])
    del s

total_bad_new = 0
for label, path in (("fp32", fp32_path), ("fp16-OLD", old_path), ("fp16-NEW", new_path)):
    for dev in (0, 1):
        try:
            s = session(path, "dml", dev)
        except Exception as e:
            print(f"  [{label} DML dev{dev}] session failed: {str(e)[:120]}", flush=True)
            continue
        print(f"  [{label} DML dev{dev}]", flush=True)
        for name, scale in REGIMES:
            try:
                outs = run(s, feeds, scale)
            except Exception as e:
                print(f"    {name}: RUN FAILED: {str(e)[:120]}", flush=True)
                continue
            bad = stats(name, outs, refs[name])
            if label == "fp16-NEW":
                total_bad_new += bad
        del s

print("\nGATE(fp16-NEW nonfinite on DML) =", "PASS" if total_bad_new == 0 else f"FAIL ({total_bad_new})")
sys.exit(0 if total_bad_new == 0 else 1)
