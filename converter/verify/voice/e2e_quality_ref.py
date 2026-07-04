"""关卡 2 — SoVITS 4.x QUALITY-PATH E2E python reference (S36: shallow diffusion /
only_diffusion / second_encoding / NSF-HiFiGAN enhancer / auto-f0).

EXTENDS e2e_sovits_ref.py (imports its certified helpers — ZeroNoise, RmvpeFront,
sovits_post_process, contentvec_session, build_orig_synth, pad_array_center) with a
faithful transcription of the ORIGINAL Svc.infer quality-path body
(D:\\MyDev\\so-vits-svc\\so-vits-svc\\inference\\infer_tool.py:256-340) around ONE
padded segment, driving:
  * the ORIGINAL torch SynthesizerTrn (real .pth weights) — z/SineGen noise zeroed
    via ZeroNoise == Rust det main onnx + noise_scale=0;
  * the ORIGINAL torch diffusion stack (diffusion/unit2mel.py load_model_vocoder with
    a temp yaml whose vocoder.ckpt points at the repo pretrain — the model's own yaml
    references a missing finetuned path, same patch as gate1_diffusion.py), ZeroNoise
    zeroing q_sample noise / only-diffusion initial randn / naive per-step noise ==
    Rust debug_zero_noise;
  * the ORIGINAL torch NSF-HiFiGAN generator (vdecoder/nsf_hifigan load_model) under
    ZeroNoise (rand_ini + uv noise = 0) == our deterministic vocoder onnx export;
  * gt mel via the ORIGINAL Vocoder.extract / nvSTFT (no resample at 44.1k).

RESAMPLER POLICY (S35 variant-B judged path): every pipeline-internal resample
(44.1k→16k for f0+hubert, second_encoding 44.1k→16k, enhancer adaptive in/out) is
scipy.signal.resample_poly — the exact house resampler of the Rust side
(features::resample). DEVIATION vs the true original (torchaudio sinc Resample):
established S35 attribution — waveform SNR collapses through NSF phase integration
while mel/listening stays equivalent, so the judged reference uses the Rust resampler.

second_encoding (svcFlow.md RISK 4): infer_tool.py:313-314 misses the c.unsqueeze(0)
that get_unit_f0 does — this script runs the EXACT original lines (2-D c [H, n_frames])
AND the sane batched layout, reports what actually happens and their max_abs_diff.

Run: converter\\.venv\\Scripts\\python.exe converter\\verify\\voice\\e2e_quality_ref.py
        --input <44.1k mono f32 wav> --outdir <dir> [--variants v1,v2,...]
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e2e_sovits_ref as base  # noqa: E402  (inserts SOVITS repo + converter on sys.path)

so_utils = base.so_utils
sovits_v4 = base.sovits_v4

from diffusion.unit2mel import load_model_vocoder            # noqa: E402  ORIGINAL
from vdecoder.nsf_hifigan.models import load_model as load_nsf_model  # noqa: E402

torch.set_grad_enabled(False)

TESTING = r"D:\MyDev\TESTING\Sovits-SVC"
AZUMA_PTH = os.path.join(TESTING, "东雪莲", "Sovits4.1东雪莲主模型.pth")
AZUMA_CFG = os.path.join(TESTING, "东雪莲", "Sovits4.1东雪莲主配置文件.json")
DIFF_PT = os.path.join(TESTING, "东雪莲", "Sovits4.1东雪莲扩散模型.pt")
DIFF_YAML = os.path.join(TESTING, "东雪莲", "Sovits4.1东雪莲扩散配置文件.yaml")
AKIKO_PTH = os.path.join(TESTING, "MinamiyaAkiko-Sovits4.0", "akiko_320000.pth")
AKIKO_CFG = os.path.join(TESTING, "MinamiyaAkiko-Sovits4.0", "config.json")
NSF_CKPT = r"D:\MyDev\so-vits-svc\so-vits-svc\pretrain\nsf_hifigan\model"

PAD_SECONDS = base.PAD_SECONDS      # 0.5
NOISE_SCALE = base.NOISE_SCALE      # 0.4
RMVPE_THRED = base.RMVPE_THRED      # 0.05

# (model, quality-path knobs) per variant — mirrors the Rust harness UTAI_VOICE_OPTS.
VARIANTS = {
    "v1_shallow_dpmpp": dict(model="azuma", shallow=True, method="dpm-solver++",
                             speedup=10, k_step=100),
    "v2_shallow_naive": dict(model="azuma", shallow=True, method=None,
                             speedup=1, k_step=100),
    "v3_shallow_unipc": dict(model="azuma", shallow=True, method="unipc",
                             speedup=10, k_step=100),
    "v4_only_dpmpp": dict(model="azuma", only=True, method="dpm-solver++",
                          speedup=10, k_step=100),
    "v5_second_enc": dict(model="azuma", shallow=True, method="dpm-solver++",
                          speedup=10, k_step=100, second=True),
    "v6a_autof0_azuma": dict(model="azuma", auto_f0=True),
    "v6b_autof0_akiko": dict(model="akiko", auto_f0=True),
    "v7_enhance_k0": dict(model="azuma", enhance=True, adaptive_key=0),
    "v8_enhance_k4": dict(model="azuma", enhance=True, adaptive_key=4),
}
DEFAULTS = dict(shallow=False, only=False, second=False, method=None, speedup=1,
                k_step=100, auto_f0=False, enhance=False, adaptive_key=0)


def load_synth(pth, cfg_path):
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    cfg, cfg_src = sovits_v4.load_sovits_config(Path(pth), Path(cfg_path))
    _, meta = sovits_v4.build_from_checkpoint(ck, cfg)
    synth = base.build_orig_synth(pth, cfg)
    print(f"[qref] {Path(pth).name}: v{meta['version']} dim={meta['features_dim']} "
          f"hop={meta['hop_size']} sr={meta['sample_rate']} vol={meta['vol_embedding']} "
          f"mode={meta['unit_interpolate_mode']} (config {cfg_src})")
    return synth, meta


def load_diffusion():
    """ORIGINAL load_model_vocoder with the vocoder.ckpt patched to the repo pretrain
    (the 东雪莲 yaml points at a missing finetuned path — gate1_diffusion precedent)."""
    cfg_real = yaml.safe_load(Path(DIFF_YAML).read_text(encoding="utf-8"))
    cfg_patched = copy.deepcopy(cfg_real)
    cfg_patched["vocoder"]["ckpt"] = NSF_CKPT
    patched = HERE.parent.parent / "test_output" / "e2e_diff_config_patched.yaml"
    patched.parent.mkdir(parents=True, exist_ok=True)
    patched.write_text(yaml.safe_dump(cfg_patched), encoding="utf-8")
    model, vocoder, args = load_model_vocoder(DIFF_PT, device="cpu",
                                              config_path=str(patched))
    model.eval()
    print(f"[qref] diffusion: timesteps={model.timesteps} k_step_max={model.k_step_max} "
          f"n_spk={model.n_spk} (yaml infer {args.infer.method}/sp{args.infer.speedup})")
    return model, vocoder


def enhance_ref(gen, stft, audio_t, sample_rate, f0_cur, hop_size, adaptive_key):
    """modules/enhancer.py Enhancer.enhance transcription, silence_front=0, with the
    REFPOLY policy for the two adaptive resamples (original: torchaudio sinc)."""
    enhancer_sr = 44100
    enh_hop = 512
    adaptive_factor = 2 ** (-adaptive_key / 12)
    adaptive_sample_rate = 100 * int(np.round(enhancer_sr / adaptive_factor / 100))
    real_factor = enhancer_sr / adaptive_sample_rate

    a = audio_t.numpy()
    if sample_rate != adaptive_sample_rate:
        audio_res = resample_poly(a, adaptive_sample_rate, sample_rate).astype(np.float32)
    else:
        audio_res = a

    n_frames = int(len(audio_res) // enh_hop + 1)
    f0_np = f0_cur.squeeze(0).cpu().numpy() * real_factor        # enhancer.py:56 (f32)
    time_org = (hop_size / sample_rate) * np.arange(len(f0_np)) / real_factor
    time_frame = (enh_hop / enhancer_sr) * np.arange(n_frames)
    f0_res = np.interp(time_frame, time_org, f0_np,
                       left=f0_np[0], right=f0_np[-1]).astype(np.float32)

    mel = stft.get_mel(torch.from_numpy(audio_res.copy())[None, :])   # [1,128,T]
    f0_t = torch.from_numpy(f0_res)[None, :]
    enhanced = gen(mel, f0_t[:, :mel.size(-1)]).view(-1)              # enhancer.py:106
    out = enhanced.numpy()
    if adaptive_sample_rate != enhancer_sr:                           # original: always-true
        out = resample_poly(out, enhancer_sr, adaptive_sample_rate).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def run_variant(name, o, ctx):
    synth, meta = ctx["synths"][o["model"]]
    diff_model, vocoder = ctx["diffusion"]
    rmvpe = ctx["rmvpe"]
    input_audio = ctx["input"]

    model_sr = meta["sample_rate"]
    hop = meta["hop_size"]
    dim = meta["features_dim"]
    ssl_mode = meta["unit_interpolate_mode"]
    vol_emb = meta["vol_embedding"]

    pad = int(model_sr * PAD_SECONDS)
    wav_m = np.concatenate([np.zeros(pad, np.float32),
                            input_audio.astype(np.float32),
                            np.zeros(pad, np.float32)])
    n_frames = len(wav_m) // hop
    wav16k = resample_poly(wav_m, 16000, model_sr).astype(np.float32)

    # f0/uv (our rmvpe onnx + ORIGINAL post_process — S35-certified shim)
    f0_100 = rmvpe.f0_100fps(wav16k, RMVPE_THRED)
    if np.all(f0_100 == 0):
        f0 = np.zeros(n_frames, np.float32)
        uv = np.zeros(n_frames, np.float32)
    else:
        f0, uv = base.sovits_post_process(f0_100, n_frames, hop, model_sr)

    # content (our contentvec onnx == Rust) → repeat_expand
    c_raw = base.contentvec_session(dim).run(None, {"waveform": wav16k[None]})[0]
    c_hub = torch.from_numpy(c_raw[0].T.copy())                       # [ssl, T]
    c = so_utils.repeat_expand_2d(c_hub, n_frames, ssl_mode).unsqueeze(0)

    f0t = torch.from_numpy(f0)[None].float()
    uvt = torch.from_numpy(uv)[None].float()
    sid = torch.LongTensor([[0]])
    vol = None
    if vol_emb:
        vol = so_utils.Volume_Extractor(hop).extract(
            torch.FloatTensor(wav_m)[None, :])[None, :]

    dbg = {"f0_src": f0.copy(), "uv": uv.copy()}
    se_note = None

    with base.ZeroNoise():
        if not o["only"]:
            audio_t, f0_ret = synth.infer(c, f0t, uvt, g=sid, noice_scale=NOISE_SCALE,
                                          predict_f0=o["auto_f0"], vol=vol)
            audio_t = audio_t[0, 0].data.float()
            f0_cur = f0_ret.detach().float()          # == input f0 unless predict_f0
            audio_mel = vocoder.extract(audio_t[None, :], model_sr) if o["shallow"] else None
        else:
            audio_t = torch.from_numpy(wav_m.copy())  # infer_tool.py:301 — SOURCE wav
            f0_cur = f0t
            audio_mel = None

        if o["shallow"] or o["only"]:
            # infer_tool.py:308 — vol_embedding reuses the source vol, else extract
            # from `audio` (= VITS output for shallow, source wav for only_diffusion)
            if vol is None:
                vol_d = so_utils.Volume_Extractor(hop).extract(
                    audio_t[None, :])[None, :, None]
            else:
                vol_d = vol[:, :, None]

            c_use = c
            if o["shallow"] and o["second"]:
                # audio16k: REFPOLY (original: torchaudio Resample(target,16000))
                audio16k = resample_poly(audio_t.numpy(), 16000, model_sr).astype(np.float32)
                c2_raw = base.contentvec_session(dim).run(
                    None, {"waveform": audio16k[None]})[0]            # [1,T,ssl]
                c2 = torch.from_numpy(c2_raw[0].T.copy())[None]       # [1,ssl,T] == hubert.encoder layout
                # ==== EXACT ORIGINAL LINES (infer_tool.py:313-314): NO unsqueeze ====
                c_verbatim = so_utils.repeat_expand_2d(
                    c2.squeeze(0), f0_cur.shape[1], ssl_mode)         # 2-D [ssl, n_frames]
                c_use = c_verbatim
                se_note = {"verbatim_shape": tuple(c_verbatim.shape)}

            f0_d = f0_cur[:, :, None]                                 # infer_tool.py:315
            cond_in = c_use.transpose(-1, -2)                         # infer_tool.py:316
            mel_out = diff_model(cond_in, f0_d, vol_d, spk_id=sid, spk_mix_dict=None,
                                 gt_spec=audio_mel, infer=True,
                                 infer_speedup=o["speedup"], method=o["method"],
                                 k_step=o["k_step"], use_tqdm=False)

            if se_note is not None:
                # empirical RISK-4 probe: sane batched layout [1,ssl,nf]→[1,nf,ssl]
                mel_batched = diff_model(
                    c_use.unsqueeze(0).transpose(-1, -2), f0_d, vol_d, spk_id=sid,
                    spk_mix_dict=None, gt_spec=audio_mel, infer=True,
                    infer_speedup=o["speedup"], method=o["method"],
                    k_step=o["k_step"], use_tqdm=False)
                se_note["mel_shape_verbatim"] = tuple(mel_out.shape)
                se_note["mel_shape_batched"] = tuple(mel_batched.shape)
                se_note["mel_max_abs_diff"] = float(
                    (mel_out - mel_batched).abs().max().item())

            dbg["mel_out"] = mel_out[0].numpy().copy()
            if audio_mel is not None:
                dbg["mel_gt"] = audio_mel[0].numpy().copy()
            dbg["vol_d"] = vol_d[0, :, 0].numpy().copy()
            dbg["f0_d"] = f0_cur[0].numpy().copy()

            audio_t = vocoder.infer(mel_out, f0_d).squeeze()          # lazy-loads generator

        if o["enhance"]:
            audio_t = enhance_ref(ctx["nsf_gen"], vocoder.vocoder.stft, audio_t,
                                  model_sr, f0_cur, hop, o["adaptive_key"])

    # loudness_envelope_adjustment == 1 → change_rms skipped (original guard)
    out = audio_t.numpy()
    trimmed = out[pad:len(out) - pad]
    per_length = int(np.ceil(len(input_audio) / model_sr * model_sr))
    y = base.pad_array_center(trimmed.astype(np.float32), per_length)
    return y, dbg, se_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="44.1k mono f32 wav (both sides)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--variants", default=None, help="comma list, default all")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    assert sr == 44100, f"input must be 44100 Hz (got {sr})"
    print(f"[qref] input n={len(audio)} rms={np.sqrt(np.mean(audio**2)):.4f}")

    wanted = list(VARIANTS) if args.variants is None else [
        v.strip() for v in args.variants.split(",")]
    models_needed = {VARIANTS[v]["model"] for v in wanted}

    ctx = {"input": audio, "rmvpe": base.RmvpeFront(), "synths": {}}
    if "azuma" in models_needed:
        ctx["synths"]["azuma"] = load_synth(AZUMA_PTH, AZUMA_CFG)
    if "akiko" in models_needed:
        ctx["synths"]["akiko"] = load_synth(AKIKO_PTH, AKIKO_CFG)
    ctx["diffusion"] = load_diffusion()
    need_enh = any(VARIANTS[v].get("enhance") for v in wanted)
    ctx["nsf_gen"] = load_nsf_model(NSF_CKPT, device="cpu")[0] if need_enh else None

    for name in wanted:
        o = {**DEFAULTS, **VARIANTS[name]}
        t0 = time.time()
        y, dbg, se_note = run_variant(name, o, ctx)
        wav_path = outdir / f"ref_{name}.wav"
        sf.write(wav_path, y, 44100, subtype="FLOAT")
        np.savez(outdir / f"ref_{name}_dbg.npz", **dbg)
        print(f"[qref] {name}: n={len(y)} peak={np.abs(y).max():.4f} "
              f"rms={np.sqrt(np.mean(y**2)):.4f} ({time.time()-t0:.1f}s) -> {wav_path}")
        if se_note is not None:
            print(f"[qref] second_encoding VERBATIM probe: c shape {se_note['verbatim_shape']} "
                  f"(2-D, no unsqueeze) ran WITHOUT crash; mel verbatim "
                  f"{se_note['mel_shape_verbatim']} vs batched {se_note['mel_shape_batched']}, "
                  f"max_abs_diff = {se_note['mel_max_abs_diff']:.3e}")
    print("[qref] done")


if __name__ == "__main__":
    main()
