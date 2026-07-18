"""Model converter: .pth/.ckpt → .onnx

Converts RVC, SoVITS, BSRoformer, MelBandRoformer, MDX23C, HTDemucs, and
UVR VR-arch checkpoints to ONNX format for inference via ONNX Runtime in the
Rust backend; legacy MDX-Net models are already ONNX and get a passthrough
(validate + write the sidecar json).

Requires PyTorch (CPU is sufficient for conversion; mdx_net needs no torch).

Usage:
    python convert.py --input model.pth  --output model.onnx --type rvc
    python convert.py --input model.pth  --output model.onnx --type sovits
    python convert.py --input model.ckpt --output model.onnx --type bs_roformer [--config model.yaml]
    python convert.py --input model.ckpt --output model.onnx --type mel_band_roformer [--config model.yaml]
    python convert.py --input model.ckpt --output model.onnx --type mdx23c --config model.yaml
    python convert.py --input model.th   --output model.onnx --type htdemucs [--config model.yaml]
    python convert.py --input model.pth  --output model.onnx --type uvr_vr
    python convert.py --input model.onnx --output model.onnx --type mdx_net

uvr_vr/mdx_net take no --config: their params come from embedded registries
keyed by UVR's tail-hash (architectures/uvr_vr.py, architectures/mdx_net.py);
unknown hashes are refused, not guessed.

sovits reads the so-vits config.json next to the .pth (or --config); missing
config falls back to weight-shape inference with warnings. Cluster / feature-
retrieval companion assets (.pt kmeans / .pkl faiss) convert via
export_cluster.py, not this script.

Separation types accept --precision fp32|fp16|both (default fp32). fp16 keeps
the shared <stem>.json and writes <stem>.fp16.onnx (deleting the fp32 .onnx);
both keeps both files. rvc/sovits ignore --precision. fp16 is numerically
verified (45 dB gate) for the four FP16_VERIFIED_TYPES only — uvr_vr/mdx_net
are refused until they pass their own CUDA-EP gate (verify/README.md 关卡 3).
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

from architectures.rvc_v2 import (
    build_from_checkpoint as build_rvc_v2,
    sr2sr as RVC_SR2SR,
)
from architectures.sovits_v4 import (
    build_from_checkpoint as build_sovits_v4,
    load_sovits_config,
    set_deterministic as sovits_set_deterministic,
    F0PredictorWrapper,
)
from architectures.sovits_v4v2 import (
    build_from_checkpoint as build_sovits_v4v2,
    is_v2_state_dict,
    F0PredictorWrapperV2,
)
from architectures.bs_roformer import (
    load_from_checkpoint as load_bs_roformer,
    detect_config,
    MAX_ROPE_SEQ,
)
from architectures.mel_band_roformer import (
    load_from_checkpoint as load_mel_band_roformer,
    detect_config as detect_mel_config,
)
from architectures.mdx23c import (
    load_from_checkpoint as load_mdx23c,
    detect_config as detect_mdx23c_config,
)
from architectures.htdemucs import (
    load_from_checkpoint as load_htdemucs,
    detect_config as detect_htdemucs_config,
)
from architectures.uvr_vr import (
    load_from_checkpoint as load_uvr_vr,
    detect_config as detect_uvr_vr_config,
    VrMaskModel,
    WINDOW_SIZE as VR_WINDOW_SIZE,
    NON_ACCOM_STEMS,
)
from architectures.msst_yaml import load_msst_yaml, stem_fields, stem_fields_from_yaml

# --precision applies to these (rvc/sovits ignore it); fp16 is numerically
# verified (>45 dB vs fp32) for the roformer archs + mdx23c + htdemucs ONLY —
# refuse the rest (uvr_vr/mdx_net must pass their own CUDA-EP gate first).
# S68c re-gate (norm-stats fp32 protection recipe, onnx_fp16.py): CUDA E2E
# BS 72.8/67.9 dB, MelBand 68.6/63.5 dB (Δ vs old recipe ≤0.1 dB — protection
# is numerically free on healthy kernels); mdx23c byte-identical (md5, empty
# block list); htdemucs untouched (own S31 path). DML smoke: old recipe's
# fp16 Clip floor is 1e-7 (weakened 1e5×) + fp16 ReduceL2 stats — NaN cascade
# on drivers without fp32 accumulation; new recipe keeps stats + the 1e-12
# floor in fp32.
SEPARATION_TYPES = {"bs_roformer", "mel_band_roformer", "mdx23c", "htdemucs",
                    "uvr_vr", "mdx_net"}
FP16_VERIFIED_TYPES = {"bs_roformer", "mel_band_roformer", "mdx23c", "htdemucs"}


# The attention rel-pos branches bake at trace time for the generic long-T
# path; the graph is then correct for every T >= window_size + 2 (= 12 for
# RVC's window_size 10). RVC_EXPORT_T must stay well above that so the generic
# path is the one baked; RVC_MIN_FRAMES is verified + gated by
# verify/voice/gate1_rvc.py and documented in the sidecar json.
RVC_EXPORT_T = 200
RVC_MIN_FRAMES = 12
RVC_WINDOW_SIZE = 10  # attentions.Encoder default; mirrors architectures/rvc_v2.py


def convert_rvc(input_path: Path, output_path: Path, deterministic: bool = False):
    """Convert RVC v1/v2 (f0) .pth model to ONNX.

    Export signature (matches the official models_onnx contract):
      phone[1,T,dim] f32, phone_lengths[1] i64, pitch[1,T] i64, pitchf[1,T] f32,
      sid[1] i64, rnd[1,inter_channels,T] f32 -> audio[1,1,L].
    rnd is the caller's N(0,1) noise ALREADY multiplied by noise_scale
    (original default 0.66666); zeros -> deterministic z_p.
    `deterministic` additionally zeroes SineGen's in-graph randomness
    (gate builds only — shipping exports keep the original noise semantics).
    """
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    config = checkpoint.get("config")
    if config is None or not isinstance(config, (list, tuple)) or len(config) < 18:
        print("ERROR: checkpoint has no usable 'config' list", file=sys.stderr)
        sys.exit(1)

    version = checkpoint.get("version", "v1")
    if checkpoint.get("f0", 1) != 1:
        # User-facing: RVC nof0 models have a different graph (no pitch inputs,
        # plain HiFi-GAN dec) and are deliberately unsupported.
        print("ERROR: 暂不支持无音高(nof0)的 RVC 模型", file=sys.stderr)
        sys.exit(1)

    # build_from_checkpoint re-validates version/f0/feature-dim and patches the
    # speaker count from emb_g.weight (checkpoints lie about it).
    model = build_rvc_v2(checkpoint, deterministic=deterministic)

    # ①c: a GENUINE multi-speaker RVC co-train embeds an ordered `speakers` name list in the .pth
    # (savee, len > 1). This is the ONLY trustworthy multi signal — n_speakers = emb_g table size
    # (109) even for a single-speaker import, so it CANNOT gate this (would flip every single-
    # speaker export to spk_mix and break byte-identity). Multi → export a spk_mix [1, n_spk] blend
    # input replacing scalar sid so emb_g can be interpolated at inference. Mirrors convert_sovits.
    rvc_speakers = checkpoint.get("speakers")
    export_spk_mix = isinstance(rvc_speakers, (list, tuple)) and len(rvc_speakers) > 1
    model.export_spk_mix = export_spk_mix

    features_dim = 256 if version == "v1" else 768
    inter_channels = config[2]
    sample_rate = config[-1]
    if isinstance(sample_rate, str):
        sample_rate = RVC_SR2SR[sample_rate]
    n_speakers = int(checkpoint["weight"]["emb_g.weight"].shape[0])
    upp = model.dec.upp  # samples per frame (= sample_rate // 100, 10 ms)

    seq_len = RVC_EXPORT_T
    phone = torch.randn(1, seq_len, features_dim)
    phone_lengths = torch.tensor([seq_len], dtype=torch.int64)
    pitch = torch.randint(1, 256, (1, seq_len), dtype=torch.int64)
    # Plausible f0 contour with an unvoiced stretch (exercises the uv path).
    pitchf = 150.0 + 100.0 * torch.rand(1, seq_len)
    pitchf[:, 40:60] = 0.0
    # speaker input: scalar sid (single) OR a normalized spk_mix [1, n_speakers] blend (multi) —
    # the emb_g gather vs `spk_mix @ emb_g.weight` matmul is bit-identical for a one-hot row.
    # n_speakers (emb_g rows) is FIXED, so spk_mix has no dynamic axis.
    if export_spk_mix:
        spk = torch.rand(1, n_speakers)
        spk = spk / spk.sum(dim=1, keepdim=True)
        spk_name = "spk_mix"
    else:
        spk = torch.zeros(1, dtype=torch.int64)
        spk_name = "sid"
    rnd = torch.randn(1, inter_channels, seq_len)

    torch.onnx.export(
        model,
        (phone, phone_lengths, pitch, pitchf, spk, rnd),
        str(output_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", spk_name, "rnd"],
        output_names=["audio"],
        dynamic_axes={
            "phone": {1: "seq_len"},
            "pitch": {1: "seq_len"},
            "pitchf": {1: "seq_len"},
            "rnd": {2: "seq_len"},
            "audio": {2: "audio_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}", file=sys.stderr)

    # ORT sanity at the export T AND at a different T — the second run proves
    # the trace stayed dynamic in seq_len (a baked graph fails right here at
    # export time instead of at runtime in Rust).
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    for t in (seq_len, 137):
        feeds = {
            "phone": np.random.randn(1, t, features_dim).astype(np.float32),
            "phone_lengths": np.array([t], dtype=np.int64),
            "pitch": np.random.randint(1, 256, (1, t)).astype(np.int64),
            "pitchf": (150.0 + 100.0 * np.random.rand(1, t)).astype(np.float32),
            "rnd": np.random.randn(1, inter_channels, t).astype(np.float32),
        }
        if export_spk_mix:
            m = np.random.rand(1, n_speakers).astype(np.float32)
            feeds["spk_mix"] = m / m.sum(axis=1, keepdims=True)
        else:
            feeds["sid"] = np.zeros(1, dtype=np.int64)
        audio = sess.run(None, feeds)[0]
        if audio.shape != (1, 1, t * upp) or not np.isfinite(audio).all():
            raise ValueError(
                f"ORT sanity run failed at T={t}: shape {audio.shape} "
                f"(expected (1, 1, {t * upp})), finite={np.isfinite(audio).all()}"
            )
        print(f"ORT sanity run passed: T={t} -> audio (1, 1, {t * upp})")

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "rvc",
        "version": version,
        "features_dim": features_dim,
        "sample_rate": int(sample_rate),
        "hop_ms": 10,
        "n_speakers": n_speakers,
        "noise": {
            # rnd must be N(0,1) * noise_scale, shaped [1, inter_channels, T].
            "rnd_input": [1, int(inter_channels), "T"],
            "default_scale": 0.66666,
        },
        "inputs": ["phone", "phone_lengths", "pitch", "pitchf", spk_name, "rnd"],
        # Shortest supported input: the traced attention rel-pos branch is only
        # valid for T >= window_size + 2 (gated in verify/voice/gate1_rvc.py).
        "min_frames": RVC_MIN_FRAMES,
    }
    # ①c: for a genuine multi-speaker model, add the speaker NAME→id map (RVC .pth is otherwise
    # nameless — names come from the embedded `speakers` list, index = spk_id = emb_g row) and the
    # spk_mix block. Appended ONLY for multi so single-speaker sidecars stay byte-identical (the
    # "inputs" list already carries "spk_mix" vs "sid").
    if export_spk_mix:
        config_data["speakers"] = {str(name): i for i, name in enumerate(rvc_speakers)}
        config_data["spk_mix"] = {"available": True, "n_spk": n_speakers}
    # utf-8 + ensure_ascii=False: CJK speaker names must survive verbatim. For single-speaker
    # (ASCII-only content) this is byte-identical to the old default write_text/ensure_ascii=True.
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print(f"Converted RVC {version}: {input_path} -> {output_path}"
          + (" [deterministic gate build]" if deterministic else ""))
    print(f"Config saved: {config_path}")
    print(f"Sample rate: {sample_rate}, Features: {features_dim}, "
          f"Speakers: {n_speakers}")


# SoVITS attention rel-pos branches bake at trace time exactly like RVC's;
# window_size is 4 (vs RVC's 10), so the generic path is valid for every
# T >= 6 (= min_frames in the sidecar, gated by verify/voice/gate1_sovits.py).
SOVITS_EXPORT_T = 200
SOVITS_NOISE_SCALE = 0.4  # original inference default (infer_tool.py noice_scale)


def convert_sovits(input_path: Path, output_path: Path,
                   config_json: Path = None, deterministic: bool = False):
    """Convert SoVITS 4.0/4.1 .pth model to ONNX.

    Export signature (official onnxexport formulation minus mel2ph/speaker-mix):
      c[1,T,ssl_dim] f32 (ALREADY expanded to the f0 frame count — expansion
      stays Rust-side), f0[1,T] f32 (Hz; f0_to_coarse runs in-graph),
      uv[1,T] f32, noise[1,inter,T] f32, sid[1] i64
      [, vol[1,T] f32 — IFF the model has vol_embedding]  -> audio[1,1,T*hop].
    noise is the caller's N(0,1) ALREADY multiplied by noice_scale (original
    default 0.4); zeros -> deterministic z_p.
    `deterministic` additionally zeroes SineGen's in-graph randomness
    (gate builds only — shipping exports keep the original noise semantics).

    When the checkpoint carries f0_decoder weights (both 4.0/4.1 eras do), a
    companion auto-f0 predictor graph is exported next to the main onnx as
    <stem>.f0.onnx (c/f0/uv/sid[/vol] -> f0_pred[1,T] Hz) and the sidecar
    gains an "auto_f0" object; otherwise "auto_f0": {"available": false}.

    Reads the config.json next to the .pth (or --config); a missing config
    falls back to weight-shape inference with warnings."""
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    # 4.0-v2 (VISinger2) checkpoints share the .pth container + the "sovits"
    # import type but are a COMPLETELY different graph ("模型完全不通用") —
    # route by state-dict namespace (disjoint: text_encoder/dec_harm/dec_noise
    # vs enc_p/emb_g). The user imports through the same SoVITS entry.
    if is_v2_state_dict(checkpoint.get("model") or {}):
        return convert_sovits_v2(input_path, output_path, checkpoint,
                                 config_json, deterministic)

    config, config_src = load_sovits_config(input_path, config_json)
    if config_src is not None:
        print(f"Using config: {config_src}")

    # build_from_checkpoint validates the 排期项 flags (depthwise/transformer
    # flow/exotic encoders/vocoders), strict=True-loads, and returns the
    # sidecar meta (weights are the truth for every tensor-shaped hyperparam).
    model, meta = build_sovits_v4(checkpoint, config)
    sovits_set_deterministic(model, deterministic)

    # ①c: a GENUINE multi-speaker model (len(speakers) > 1 — the real trained
    # count, NOT emb_g's table size n_speakers, which is large even for a
    # single-speaker import) exports a spk_mix [1, n_spk] f32 blend in place of
    # the scalar sid, so speaker embeddings can be blended at inference. Every
    # single-speaker export keeps sid → BYTE-IDENTICAL to pre-①c.
    export_spk_mix = len(meta.get("speakers") or {}) > 1
    n_spk = meta["n_speakers"]
    model.export_spk_mix = export_spk_mix

    ssl_dim = meta["features_dim"]     # detected — NOT hardcoded (old 256 bug)
    inter_channels = meta["inter_channels"]
    hop_size = meta["hop_size"]
    vol_embedding = meta["vol_embedding"]

    seq_len = SOVITS_EXPORT_T
    torch.manual_seed(20260704)
    c = torch.randn(1, seq_len, ssl_dim)
    # Plausible f0 contour with an unvoiced stretch (exercises the uv path).
    f0 = 150.0 + 250.0 * torch.rand(1, seq_len)
    f0[:, 40:60] = 0.0
    uv = (f0 > 0).float()
    noise = torch.randn(1, inter_channels, seq_len) * SOVITS_NOISE_SCALE
    # speaker input: scalar sid (single-speaker) OR a normalized spk_mix
    # [1, n_spk] (multi) — the emb_g gather vs `spk_mix @ emb_g.weight` matmul is
    # bit-identical for a one-hot row. n_spk is FIXED, so spk_mix has no dyn axis.
    if export_spk_mix:
        spk = torch.rand(1, n_spk)
        spk = spk / spk.sum(dim=1, keepdim=True)
        spk_name = "spk_mix"
    else:
        spk = torch.zeros(1, dtype=torch.int64)
        spk_name = "sid"

    input_names = ["c", "f0", "uv", "noise", spk_name]
    dynamic_axes = {
        "c": {1: "seq_len"},
        "f0": {1: "seq_len"},
        "uv": {1: "seq_len"},
        "noise": {2: "seq_len"},
        "audio": {2: "audio_len"},
    }
    inputs = (c, f0, uv, noise, spk)
    if vol_embedding:
        vol = torch.rand(1, seq_len) * 0.1
        input_names.append("vol")
        dynamic_axes["vol"] = {1: "seq_len"}
        inputs = inputs + (vol,)

    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=["audio"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}", file=sys.stderr)

    # ORT sanity at the export T AND at a different T — the second run proves
    # the trace stayed dynamic in seq_len.
    import numpy as np
    import onnxruntime as ort
    # Load from BYTES, not path: the python onnxruntime build cannot open
    # session paths containing Chinese characters on Windows (locale ACP
    # issue, measured 2026-07-04) — and SoVITS model files routinely have
    # Chinese names (feedback_v2_unicode_paths). Rust's ort crate uses proper
    # wide-string paths and is unaffected.
    sess = ort.InferenceSession(output_path.read_bytes())
    for t in (seq_len, 137):
        f0_np = (150.0 + 250.0 * np.random.rand(1, t)).astype(np.float32)
        f0_np[:, t // 5:t // 5 + 10] = 0.0
        feeds = {
            "c": np.random.randn(1, t, ssl_dim).astype(np.float32),
            "f0": f0_np,
            "uv": (f0_np > 0).astype(np.float32),
            "noise": (np.random.randn(1, inter_channels, t)
                      * SOVITS_NOISE_SCALE).astype(np.float32),
        }
        if export_spk_mix:
            m = np.random.rand(1, n_spk).astype(np.float32)
            feeds["spk_mix"] = m / m.sum(axis=1, keepdims=True)
        else:
            feeds["sid"] = np.zeros(1, dtype=np.int64)
        if vol_embedding:
            feeds["vol"] = (0.1 * np.random.rand(1, t)).astype(np.float32)
        audio = sess.run(None, feeds)[0]
        if audio.shape != (1, 1, t * hop_size) or not np.isfinite(audio).all():
            raise ValueError(
                f"ORT sanity run failed at T={t}: shape {audio.shape} "
                f"(expected (1, 1, {t * hop_size})), finite={np.isfinite(audio).all()}"
            )
        print(f"ORT sanity run passed: T={t} -> audio (1, 1, {t * hop_size})")

    # ── auto-f0 companion graph (<stem>.f0.onnx) ────────────────────────────
    # Both 4.0- and 4.1-era checkpoints carry f0_decoder weights; the weights
    # are the truth (meta["has_f0_decoder"] — 4.0 configs lack the key). The
    # predictor graph shares the synthesizer's nn.Parameters at the python
    # level and contains NO randomness, so `deterministic` is irrelevant here
    # (a shipping export is already reproducible). Gated by
    # verify/voice/gate_autof0.py against models.py infer(predict_f0=True).
    auto_f0_path = None
    f0_input_names = None
    if meta["has_f0_decoder"]:
        auto_f0_path = output_path.with_name(output_path.stem + ".f0.onnx")
        predictor = F0PredictorWrapper(model)
        predictor.eval()
        f0_input_names = ["c", "f0", "uv", spk_name] + (["vol"] if vol_embedding else [])
        f0_dynamic_axes = {
            "c": {1: "seq_len"},
            "f0": {1: "seq_len"},
            "uv": {1: "seq_len"},
            "f0_pred": {1: "seq_len"},
        }
        f0_inputs = (c, f0, uv, spk)
        if vol_embedding:
            f0_dynamic_axes["vol"] = {1: "seq_len"}
            f0_inputs = f0_inputs + (vol,)
        torch.onnx.export(
            predictor,
            f0_inputs,
            str(auto_f0_path),
            input_names=f0_input_names,
            output_names=["f0_pred"],
            dynamic_axes=f0_dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(str(auto_f0_path)))
        print(f"ONNX check passed: {auto_f0_path}", file=sys.stderr)

        f0_sess = ort.InferenceSession(auto_f0_path.read_bytes())
        for t in (seq_len, 137):
            f0_np = (150.0 + 250.0 * np.random.rand(1, t)).astype(np.float32)
            f0_np[:, t // 5:t // 5 + 10] = 0.0
            feeds = {
                "c": np.random.randn(1, t, ssl_dim).astype(np.float32),
                "f0": f0_np,
                "uv": (f0_np > 0).astype(np.float32),
            }
            if export_spk_mix:
                m = np.random.rand(1, n_spk).astype(np.float32)
                feeds["spk_mix"] = m / m.sum(axis=1, keepdims=True)
            else:
                feeds["sid"] = np.zeros(1, dtype=np.int64)
            if vol_embedding:
                feeds["vol"] = (0.1 * np.random.rand(1, t)).astype(np.float32)
            f0_pred = f0_sess.run(None, feeds)[0]
            if f0_pred.shape != (1, t) or not np.isfinite(f0_pred).all():
                raise ValueError(
                    f"auto-f0 ORT sanity run failed at T={t}: shape "
                    f"{f0_pred.shape} (expected (1, {t})), "
                    f"finite={np.isfinite(f0_pred).all()}"
                )
            print(f"ORT sanity run passed: T={t} -> f0_pred (1, {t})")

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "sovits",
        "version": meta["version"],
        "features_dim": ssl_dim,
        "speech_encoder": meta["speech_encoder"],
        "sample_rate": meta["sample_rate"],
        "hop_size": hop_size,
        "vol_embedding": vol_embedding,
        "unit_interpolate_mode": meta["unit_interpolate_mode"],
        "n_speakers": meta["n_speakers"],
        "speakers": meta["speakers"],
        "noise": {
            # noise must be N(0,1) * noice_scale, shaped [1, inter_channels, T].
            "noise_input": [1, inter_channels, "T"],
            "default_scale": SOVITS_NOISE_SCALE,
        },
        "inputs": input_names,
        # Shortest supported input: the traced attention rel-pos branch is only
        # valid for T >= window_size + 2 (gated in verify/voice/gate1_sovits.py).
        "min_frames": meta["min_frames"],
        # Appended AFTER every pre-existing key: older sidecar consumers parse
        # the fields above and must see them byte-compatibly (ModelConfig keeps
        # unknown keys in `extra`, exposing auto_f0 to the frontend verbatim).
        "auto_f0": (
            {"available": True, "file": auto_f0_path.name,
             "inputs": f0_input_names}
            if auto_f0_path is not None else {"available": False}
        ),
    }
    # ①c: multi-speaker models take a spk_mix [1, n_spk] f32 blend (rows sum to 1)
    # in place of sid — appended ONLY for multi so single-speaker sidecars stay
    # byte-identical (the "inputs" list already carries "spk_mix" vs "sid").
    if export_spk_mix:
        config_data["spk_mix"] = {"available": True, "n_spk": n_spk}
    # utf-8 + ensure_ascii=False: Chinese speaker names must survive verbatim.
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print(f"Converted SoVITS {meta['version']}: {input_path} -> {output_path}"
          + (" [deterministic gate build]" if deterministic else ""))
    print(f"Config saved: {config_path}")
    print(f"Sample rate: {meta['sample_rate']}, SSL dim: {ssl_dim} "
          f"({meta['speech_encoder']}), Hop: {hop_size}, "
          f"Speakers: {meta['n_speakers']}, vol_embedding: {vol_embedding}, "
          f"auto_f0: {auto_f0_path.name if auto_f0_path is not None else False}")


SOVITS_V2_EXPORT_T = 200  # window_size 4 → generic rel-pos path, same as sovits


def convert_sovits_v2(input_path: Path, output_path: Path, checkpoint,
                      config_json: Path = None, deterministic: bool = False):
    """Convert a SoVITS 4.0-v2 (VISinger2) .pth model to ONNX. Reached from
    convert_sovits via state-dict detection — same CLI surface.

    Export signature (see architectures/sovits_v4v2.py header for the full
    contract + documented deviations):
      c[1,T,256] f32 (ALREADY expanded — expansion stays Rust-side),
      f0[1,T] f32 (Hz), noise[1,inter,T] f32 (caller N(0,1) * noice_scale,
      original default 0.4), phase[1,n_fft//2+1,T] f32 (caller
      uniform*2*3.14-3.14 — Generator_Noise's random phase made explicit),
      sid[1] i64 [or spk_mix[1,n_spk] f32]  ->  audio[1,1,T*hop].
    The main graph has NO uv input (v2 only reads uv in the predict_f0 branch
    = the <stem>.f0.onnx companion) and NO vol input (no vol_embedding in the
    v2 architecture). `deterministic` is a no-op: with noise+phase explicit
    there is no in-graph randomness left (det build == shipping build)."""
    config, config_src = load_sovits_config(input_path, config_json)
    if config_src is not None:
        print(f"Using config: {config_src}")

    model, meta = build_sovits_v4v2(checkpoint, config)

    # ①c gating — identical convention to convert_sovits: the REAL trained
    # speaker count from the config spk map, NOT emb_spk's table size (200
    # even for single-speaker models).
    export_spk_mix = len(meta.get("speakers") or {}) > 1
    n_spk = meta["n_speakers"]
    model.export_spk_mix = export_spk_mix

    ssl_dim = meta["features_dim"]
    inter_channels = meta["inter_channels"]
    prior_hidden = meta["prior_hidden"]
    hop_size = meta["hop_size"]
    n_fft = meta["n_fft"]
    phase_bins = n_fft // 2 + 1

    seq_len = SOVITS_V2_EXPORT_T
    torch.manual_seed(20260717)
    c = torch.randn(1, seq_len, ssl_dim)
    # Plausible f0 contour with an unvoiced stretch (f0=0 exercises the
    # remove_above_nyquist / sin-bank zero-pitch path).
    f0 = 150.0 + 250.0 * torch.rand(1, seq_len)
    f0[:, 40:60] = 0.0
    noise = torch.randn(1, inter_channels, seq_len) * SOVITS_NOISE_SCALE
    phase = torch.rand(1, phase_bins, seq_len) * 2 * 3.14 - 3.14
    # deviation 7: trace with a NON-zero f0d_cond so the add cannot be folded
    # away; runtime manual mode feeds zeros (bit-exact no-op).
    f0d_cond = torch.randn(1, prior_hidden, seq_len) * 0.1
    if export_spk_mix:
        spk = torch.rand(1, n_spk)
        spk = spk / spk.sum(dim=1, keepdim=True)
        spk_name = "spk_mix"
    else:
        spk = torch.zeros(1, dtype=torch.int64)
        spk_name = "sid"

    input_names = ["c", "f0", "noise", "phase", spk_name, "f0d_cond"]
    dynamic_axes = {
        "c": {1: "seq_len"},
        "f0": {1: "seq_len"},
        "noise": {2: "seq_len"},
        "phase": {2: "seq_len"},
        "f0d_cond": {2: "seq_len"},
        "audio": {2: "audio_len"},
    }
    inputs = (c, f0, noise, phase, spk, f0d_cond)

    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=["audio"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}", file=sys.stderr)

    # ORT sanity at the export T AND at a different T — the second run proves
    # the trace stayed dynamic in seq_len (the VISinger2 decoder slices with
    # traced sizes; a baked constant would fail loudly here).
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path.read_bytes())  # bytes: ACP paths
    for t in (seq_len, 137):
        f0_np = (150.0 + 250.0 * np.random.rand(1, t)).astype(np.float32)
        f0_np[:, t // 5:t // 5 + 10] = 0.0
        feeds = {
            "c": np.random.randn(1, t, ssl_dim).astype(np.float32),
            "f0": f0_np,
            "noise": (np.random.randn(1, inter_channels, t)
                      * SOVITS_NOISE_SCALE).astype(np.float32),
            "phase": (np.random.rand(1, phase_bins, t) * 2 * 3.14 - 3.14
                      ).astype(np.float32),
            "f0d_cond": np.zeros((1, prior_hidden, t), dtype=np.float32),
        }
        if export_spk_mix:
            m = np.random.rand(1, n_spk).astype(np.float32)
            feeds["spk_mix"] = m / m.sum(axis=1, keepdims=True)
        else:
            feeds["sid"] = np.zeros(1, dtype=np.int64)
        audio = sess.run(None, feeds)[0]
        if audio.shape != (1, 1, t * hop_size) or not np.isfinite(audio).all():
            raise ValueError(
                f"ORT sanity run failed at T={t}: shape {audio.shape} "
                f"(expected (1, 1, {t * hop_size})), finite={np.isfinite(audio).all()}"
            )
        print(f"ORT sanity run passed: T={t} -> audio (1, 1, {t * hop_size})")

    # ── auto-f0 companion graph (<stem>.f0.onnx) ────────────────────────────
    # Same mechanism as convert_sovits; the wrapper shares the synthesizer's
    # nn.Parameters and contains no randomness. Gated by gate1_sovits_v2.py
    # against models.py infer(predict_f0=True) (normalize_f0 factor pinned).
    auto_f0_path = None
    f0_input_names = None
    if meta["has_f0_decoder"]:
        auto_f0_path = output_path.with_name(output_path.stem + ".f0.onnx")
        predictor = F0PredictorWrapperV2(model)
        predictor.eval()
        f0_input_names = ["c", "f0", "uv", spk_name]
        f0_dynamic_axes = {
            "c": {1: "seq_len"},
            "f0": {1: "seq_len"},
            "uv": {1: "seq_len"},
            "f0_pred": {1: "seq_len"},
            "f0d_cond": {2: "seq_len"},
        }
        uv = (f0 > 0).float()
        f0_inputs = (c, f0, uv, spk)
        torch.onnx.export(
            predictor,
            f0_inputs,
            str(auto_f0_path),
            input_names=f0_input_names,
            # deviation 7: f0d_cond = the alias side-effect term, fed back as
            # the main graph's f0d_cond input in auto-f0 mode
            output_names=["f0_pred", "f0d_cond"],
            dynamic_axes=f0_dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(str(auto_f0_path)))
        print(f"ONNX check passed: {auto_f0_path}", file=sys.stderr)

        f0_sess = ort.InferenceSession(auto_f0_path.read_bytes())
        for t in (seq_len, 137):
            f0_np = (150.0 + 250.0 * np.random.rand(1, t)).astype(np.float32)
            f0_np[:, t // 5:t // 5 + 10] = 0.0
            feeds = {
                "c": np.random.randn(1, t, ssl_dim).astype(np.float32),
                "f0": f0_np,
                "uv": (f0_np > 0).astype(np.float32),
            }
            if export_spk_mix:
                m = np.random.rand(1, n_spk).astype(np.float32)
                feeds["spk_mix"] = m / m.sum(axis=1, keepdims=True)
            else:
                feeds["sid"] = np.zeros(1, dtype=np.int64)
            outs = f0_sess.run(None, feeds)
            f0_pred, f0d = outs[0], outs[1]
            if (f0_pred.shape != (1, t) or not np.isfinite(f0_pred).all()
                    or f0d.shape != (1, prior_hidden, t) or not np.isfinite(f0d).all()):
                raise ValueError(
                    f"auto-f0 ORT sanity run failed at T={t}: f0_pred {f0_pred.shape} "
                    f"(expected (1, {t})), f0d_cond {f0d.shape} "
                    f"(expected (1, {prior_hidden}, {t}))"
                )
            print(f"ORT sanity run passed: T={t} -> f0_pred (1, {t}) + "
                  f"f0d_cond (1, {prior_hidden}, {t})")

    config_path = output_path.with_suffix(".json")
    config_data = {
        "type": "sovits",
        "version": meta["version"],          # "4.0-v2" — the UI badge verbatim
        "features_dim": ssl_dim,
        "speech_encoder": meta["speech_encoder"],
        "sample_rate": meta["sample_rate"],
        "hop_size": hop_size,
        "vol_embedding": False,
        "unit_interpolate_mode": meta["unit_interpolate_mode"],
        "n_speakers": meta["n_speakers"],
        "speakers": meta["speakers"],
        "noise": {
            # noise must be N(0,1) * noice_scale, shaped [1, inter_channels, T].
            "noise_input": [1, inter_channels, "T"],
            "default_scale": SOVITS_NOISE_SCALE,
        },
        # v2-only: Generator_Noise's random phase, uniform*2*3.14-3.14
        # (upstream's literal 3.14), shaped [1, n_fft//2+1, T]; zeros = det.
        "phase": {
            "phase_input": [1, phase_bins, "T"],
        },
        # v2-only (deviation 7): the auto-f0 alias side-effect term — zeros in
        # manual mode, the companion's 2nd output in auto mode.
        "f0d_cond": {
            "input": [1, prior_hidden, "T"],
        },
        "inputs": input_names,
        "min_frames": meta["min_frames"],
        "auto_f0": (
            {"available": True, "file": auto_f0_path.name,
             "inputs": f0_input_names,
             "outputs": ["f0_pred", "f0d_cond"]}
            if auto_f0_path is not None else {"available": False}
        ),
    }
    if export_spk_mix:
        config_data["spk_mix"] = {"available": True, "n_spk": n_spk}
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print(f"Converted SoVITS {meta['version']}: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Sample rate: {meta['sample_rate']}, SSL dim: {ssl_dim} "
          f"({meta['speech_encoder']}), Hop: {hop_size}, "
          f"Speakers: {meta['n_speakers']}, "
          f"auto_f0: {auto_f0_path.name if auto_f0_path is not None else False}")


def _inference_params_from_yaml(config_yaml, fallback_chunk: int, fallback_overlap: int):
    """Original MSST inference recipe: audio.chunk_size + inference.num_overlap.
    Roformer separation quality depends on the inference chunk length matching the
    training dim_t — do NOT shrink these for speed (S16 did: 131584/2 vs the trained
    352800/4, a real audible SDR hit). The app's overlap slider is the speed knob.
    Hard-fails on unparseable yaml (load_msst_yaml) — silent fallback would bake
    wrong params."""
    y = load_msst_yaml(config_yaml)
    chunk = (y.get("audio") or {}).get("chunk_size", fallback_chunk)
    overlap = (y.get("inference") or {}).get("num_overlap", fallback_overlap)
    return chunk, overlap


def _write_model_json(output_path: Path, config_data: dict) -> Path:
    config_path = output_path.with_suffix(".json")
    config_path.write_text(json.dumps(config_data, indent=2))
    return config_path


def _roformer_chunk_params(config: dict, config_yaml,
                           fallback_chunk: int, fallback_overlap: int):
    """Resolve chunk/overlap BEFORE the expensive export and hard-fail if the
    chunk's STFT frame count exceeds the exported RoPE cos/sin cache
    (MAX_ROPE_SEQ) — the model would export cleanly and then die at runtime
    with a cryptic out-of-range read inside com.microsoft::RotaryEmbedding."""
    chunk_size, num_overlap = _inference_params_from_yaml(
        config_yaml, fallback_chunk, fallback_overlap)
    hop = config.get("stft_hop_length", 441)
    frames = chunk_size // hop + 1
    if frames > MAX_ROPE_SEQ:
        raise ValueError(
            f"chunk_size {chunk_size} @ hop {hop} gives {frames} STFT frames, "
            f"exceeding the exported rotary cache MAX_ROPE_SEQ={MAX_ROPE_SEQ} "
            f"(architectures/bs_roformer.py). Raise MAX_ROPE_SEQ and re-export."
        )
    return chunk_size, num_overlap


def _write_roformer_json(output_path: Path, model_type: str, config: dict,
                         config_yaml, chunk_size: int, num_overlap: int):
    """Shared bs_roformer/mel_band_roformer .json tail (ONE source of truth —
    the melband copy had drifted and lost dynamic_batch)."""
    n_fft = config.get("stft_n_fft", 2048)
    num_stems = config.get("num_stems", 1)
    config_data = {
        "type": model_type,
        "sample_rate": config.get("sample_rate", 44100),
        "stereo": config.get("stereo", False),
        "num_stems": num_stems,
        "n_fft": n_fft,
        "hop_length": config.get("stft_hop_length", 441),
        "win_length": config.get("stft_win_length", n_fft),
        "freq_bins": n_fft // 2 + 1,
        "chunk_size": chunk_size,
        "num_overlap": num_overlap,
        "batch_size": 1,
        "dynamic_batch": True,
    }
    config_data.update(stem_fields_from_yaml(load_msst_yaml(config_yaml), num_stems))
    config_path = _write_model_json(output_path, config_data)
    return config_path


def convert_bs_roformer(input_path: Path, output_path: Path, config_yaml: Path = None):
    """Convert BSRoformer .ckpt model to ONNX."""
    config = detect_config(str(input_path))
    # Fallback = viperx BSRoformer family (ep_317/ep_368 yamls both say 352800/4).
    chunk_size, num_overlap = _roformer_chunk_params(config, config_yaml, 352800, 4)
    model = load_bs_roformer(str(input_path), config)

    stereo = config.get("stereo", False)
    audio_channels = 2 if stereo else 1
    n_fft = config.get("stft_n_fft", 2048)
    freq_bins = n_fft // 2 + 1
    fs = freq_bins * audio_channels

    dummy = torch.randn(1, fs, 256, 2)

    with torch.no_grad():
        test_out = model(dummy)
    print(f"Test inference OK: input {list(dummy.shape)} -> output {list(test_out.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["stft_repr"],
        output_names=["mask"],
        dynamic_axes={
            "stft_repr": {0: "batch", 2: "time_frames"},
            "mask": {0: "batch", 3: "time_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"stft_repr": dummy.numpy()})
    diff = abs(test_out.numpy() - ort_out[0]).max()
    print(f"ORT verification: max diff = {diff:.6e}")
    # Batch-axis self-check: a B=2 batch must equal two independent B=1 runs, item-for-item. Proves the
    # dynamic batch axis exported correctly (constant-folding didn't bake batch=1) AND that the Rust
    # unbind offset (j*N + stem) matches the real [B, N, ...] output layout. If this asserts, the batched
    # Rust path would silently corrupt output — fail loudly here at export time instead.
    dummy2 = torch.cat([dummy, dummy + 0.01], dim=0).numpy()
    ort_b2 = sess.run(None, {"stft_repr": dummy2})[0]
    s0 = sess.run(None, {"stft_repr": dummy.numpy()})[0]
    s1 = sess.run(None, {"stft_repr": (dummy + 0.01).numpy()})[0]
    b2_diff = max(abs(ort_b2[0] - s0[0]).max(), abs(ort_b2[1] - s1[0]).max())
    assert b2_diff < 1e-3, f"BATCH-AXIS EXPORT BROKEN: B=2 != 2x(B=1) (diff {b2_diff:.3e})"
    print(f"Batch-axis check passed: B=2 == 2x(B=1), diff = {b2_diff:.3e}")

    config_path = _write_roformer_json(
        output_path, "bs_roformer", config, config_yaml, chunk_size, num_overlap)

    print(f"Converted BSRoformer: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stereo: {stereo}, Stems: {config.get('num_stems', 1)}, "
          f"FFT: {n_fft}, Hop: {config.get('stft_hop_length', 441)}")


def convert_mel_band_roformer(input_path: Path, output_path: Path,
                              config_yaml: Path = None):
    """Convert MelBandRoformer .ckpt model to ONNX."""
    yaml_str = str(config_yaml) if config_yaml else None
    config = detect_mel_config(str(input_path), yaml_str)
    # Fallback = melband_roformer_inst_v2 yaml (485100/4); pass --config for other melband models.
    chunk_size, num_overlap = _roformer_chunk_params(config, config_yaml, 485100, 4)
    model = load_mel_band_roformer(str(input_path), config=config, yaml_path=yaml_str)

    stereo = config.get("stereo", False)
    audio_channels = 2 if stereo else 1
    n_fft = config.get("stft_n_fft", 2048)
    freq_bins = n_fft // 2 + 1
    fs = freq_bins * audio_channels

    dummy = torch.randn(1, fs, 256, 2)

    with torch.no_grad():
        test_out = model(dummy)
    print(f"Test inference OK: input {list(dummy.shape)} -> output {list(test_out.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["stft_repr"],
        output_names=["mask"],
        dynamic_axes={
            "stft_repr": {0: "batch", 2: "time_frames"},
            "mask": {0: "batch", 3: "time_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"stft_repr": dummy.numpy()})
    diff = abs(test_out.numpy() - ort_out[0]).max()
    print(f"ORT verification: max diff = {diff:.6e}")
    # Batch-axis self-check: a B=2 batch must equal two independent B=1 runs, item-for-item. Proves the
    # dynamic batch axis exported correctly (constant-folding didn't bake batch=1) AND that the Rust
    # unbind offset (j*N + stem) matches the real [B, N, ...] output layout. If this asserts, the batched
    # Rust path would silently corrupt output — fail loudly here at export time instead.
    dummy2 = torch.cat([dummy, dummy + 0.01], dim=0).numpy()
    ort_b2 = sess.run(None, {"stft_repr": dummy2})[0]
    s0 = sess.run(None, {"stft_repr": dummy.numpy()})[0]
    s1 = sess.run(None, {"stft_repr": (dummy + 0.01).numpy()})[0]
    b2_diff = max(abs(ort_b2[0] - s0[0]).max(), abs(ort_b2[1] - s1[0]).max())
    assert b2_diff < 1e-3, f"BATCH-AXIS EXPORT BROKEN: B=2 != 2x(B=1) (diff {b2_diff:.3e})"
    print(f"Batch-axis check passed: B=2 == 2x(B=1), diff = {b2_diff:.3e}")

    config_path = _write_roformer_json(
        output_path, "mel_band_roformer", config, config_yaml, chunk_size, num_overlap)

    print(f"Converted MelBandRoformer: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stereo: {stereo}, Stems: {config.get('num_stems', 1)}, "
          f"FFT: {n_fft}, Hop: {config.get('stft_hop_length', 441)}")


def convert_mdx23c(input_path: Path, output_path: Path, config_yaml: Path = None):
    """Convert MDX23C (TFC-TDF-v3) .ckpt model to ONNX."""
    yaml_str = str(config_yaml) if config_yaml else None
    config = detect_mdx23c_config(str(input_path), yaml_str)
    model = load_mdx23c(str(input_path), config=config, yaml_path=yaml_str)

    dim_f = config["dim_f"]
    num_stems = config["num_target_instruments"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    sample_rate = config.get("sample_rate", 44100)

    # Chunk/overlap from the model's own yaml — the old hardcoded 261120 broke
    # models trained with other geometries (e.g. chunk 130560 @ hop 512).
    chunk_size, num_overlap = _inference_params_from_yaml(config_yaml, 261120, 4)

    # The U-Net downsamples time by 2 per scale; the decoder skip-cat hard-fails
    # at runtime unless the frame count divides 2^num_scales. Check BEFORE export.
    num_scales = config["num_scales"]
    frames = chunk_size // hop_length + 1
    if frames % (2 ** num_scales) != 0:
        raise ValueError(
            f"MDX23C chunk/hop geometry is invalid for this model: chunk_size "
            f"{chunk_size} @ hop {hop_length} gives {frames} STFT frames, which is "
            f"not divisible by 2^{num_scales}={2 ** num_scales} (the U-Net's total "
            f"time downsampling). Inference WOULD hard-fail in the decoder skip-cat. "
            f"Pass the model's training yaml via --config so the trained chunk_size "
            f"is used instead of the 261120 default."
        )

    # CaC input: [B, 4, dim_f, T]
    dummy = torch.randn(1, 4, dim_f, 256)

    with torch.no_grad():
        test_out = model(dummy)
    print(f"Test inference OK: input {list(dummy.shape)} -> output {list(test_out.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["stft_repr"],
        output_names=["separated"],
        dynamic_axes={
            "stft_repr": {3: "time_frames"},
            "separated": {4: "time_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"stft_repr": dummy.numpy()})
    diff = abs(test_out.numpy() - ort_out[0]).max()
    print(f"ORT verification: max diff = {diff:.6e}")

    config_data = {
        "type": "mdx23c",
        "sample_rate": sample_rate,
        "stereo": True,
        "num_stems": num_stems,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "dim_f": dim_f,
        "num_subbands": config["num_subbands"],
        "chunk_size": chunk_size,
        "num_overlap": num_overlap,
        "batch_size": 1,
    }
    config_data.update(stem_fields_from_yaml(load_msst_yaml(config_yaml), num_stems))
    config_path = _write_model_json(output_path, config_data)

    print(f"Converted MDX23C: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Stems: {num_stems}, dim_f: {dim_f}, FFT: {n_fft}, Hop: {hop_length}")


def convert_htdemucs(input_path: Path, output_path: Path, config_yaml: Path = None):
    """Convert HTDemucs .th/.ckpt model to ONNX."""
    yaml_str = str(config_yaml) if config_yaml else None
    yaml_config = load_msst_yaml(yaml_str)
    config = detect_htdemucs_config(str(input_path), yaml_str)
    model = load_htdemucs(str(input_path), config=config)

    nfft = config.get("nfft", 4096)
    samplerate = config.get("samplerate", 44100)
    # segment: ckpt kwargs > yaml training.segment > 11 (resolved by detect_config;
    # MSST runs its htdemucs vocal models at 11s, not the demucs default 10s)
    segment = config.get("segment", 11)
    sources = config.get("sources", 4)
    num_stems = len(sources) if isinstance(sources, (list, tuple)) else int(sources)
    freq_bins = nfft // 2

    # Fixed segment size (HTDemucs requires fixed input for ONNX)
    segment_samples = int(segment * samplerate)
    hop = nfft // 4
    # Spec frames: ceil(segment_samples / hop)
    le = int(math.ceil(segment_samples / hop))

    dummy_spec = torch.randn(1, 4, freq_bins, le)
    dummy_mix = torch.randn(1, 2, segment_samples)

    with torch.no_grad():
        freq_out, time_out = model(dummy_spec, dummy_mix)
    print(f"Test inference OK:")
    print(f"  CaC input: {list(dummy_spec.shape)}, Mix input: {list(dummy_mix.shape)}")
    print(f"  Freq out: {list(freq_out.shape)}, Time out: {list(time_out.shape)}")

    torch.onnx.export(
        model,
        (dummy_spec, dummy_mix),
        str(output_path),
        input_names=["cac_spec", "mix"],
        output_names=["freq_out", "time_out"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"cac_spec": dummy_spec.numpy(), "mix": dummy_mix.numpy()})
    diff_f = abs(freq_out.numpy() - ort_out[0]).max()
    diff_t = abs(time_out.numpy() - ort_out[1]).max()
    print(f"ORT verification: freq max diff = {diff_f:.6e}, time max diff = {diff_t:.6e}")

    # Fallback 2 (NOT 4): official demucs weights ship signature-only yamls (no inference
    # section), and the model authors' own tooling (demucs apply_model) runs overlap=0.25
    # ≈ effective coverage 1.33 with triangular crossfades — ov4 is 2.95x that compute for
    # no proven quality gain (MSST's blanket 4 is an app-wide convention, not htdemucs
    # tuning). An MSST yaml that explicitly says num_overlap still wins (faithful).
    # Mirror: msst-catalog.ts MSST_DEFAULT_NUM_OVERLAP.htdemucs.
    num_overlap = (yaml_config.get("inference") or {}).get("num_overlap", 2)

    config_data = {
        "type": "htdemucs",
        "sample_rate": samplerate,
        "stereo": True,
        "num_stems": num_stems,
        "n_fft": nfft,
        "hop_length": hop,
        "win_length": nfft,
        "segment_samples": segment_samples,
        "processing_mode": "hybrid",
        "num_overlap": num_overlap,
        "batch_size": 1,
    }
    # Stem labels: ckpt kwargs['sources'] (official demucs) beats yaml training.
    if isinstance(sources, (list, tuple)):
        config_data.update(stem_fields(sources, None, num_stems))
    else:
        config_data.update(stem_fields_from_yaml(yaml_config, num_stems))
    config_path = _write_model_json(output_path, config_data)

    print(f"Converted HTDemucs: {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Sources: {sources}, FFT: {nfft}, Segment: {segment}s ({segment_samples} samples)")


def convert_uvr_vr(input_path: Path, output_path: Path):
    """Convert UVR VR-arch .pth model to ONNX (mask predictor, fixed T window)."""
    config = detect_uvr_vr_config(str(input_path))
    net = load_uvr_vr(str(input_path), config)
    model = VrMaskModel(net)
    model.eval()

    mp = config["model_params"]
    bins = mp["bins"]

    # Magnitude domain input ([0,1) after the pipeline's global max-norm) —
    # torch.rand keeps the parity check in-domain.
    dummy = torch.rand(1, 2, bins + 1, VR_WINDOW_SIZE)

    with torch.no_grad():
        test_out = model(dummy)
    print(f"Test inference OK: input {list(dummy.shape)} -> output {list(test_out.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["mag"],
        output_names=["mask"],
        # T is deliberately static (=WINDOW_SIZE crops, DML-friendly); only batch is dynamic.
        dynamic_axes={
            "mag": {0: "batch"},
            "mask": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    ort_out = sess.run(None, {"mag": dummy.numpy()})
    diff = abs(test_out.numpy() - ort_out[0]).max()
    print(f"ORT verification: max diff = {diff:.6e}")
    # Batch-axis self-check (same rationale as the roformers): a B=2 batch must
    # equal two independent B=1 runs — proves the dynamic batch axis survived
    # tracing (the v5.1 LSTM reshape is the risky spot) before Rust batches windows.
    dummy2 = torch.cat([dummy, torch.rand(1, 2, bins + 1, VR_WINDOW_SIZE)], dim=0)
    ort_b2 = sess.run(None, {"mag": dummy2.numpy()})[0]
    s0 = sess.run(None, {"mag": dummy2[0:1].numpy()})[0]
    s1 = sess.run(None, {"mag": dummy2[1:2].numpy()})[0]
    b2_diff = max(abs(ort_b2[0] - s0[0]).max(), abs(ort_b2[1] - s1[0]).max())
    assert b2_diff < 1e-3, f"BATCH-AXIS EXPORT BROKEN: B=2 != 2x(B=1) (diff {b2_diff:.3e})"
    print(f"Batch-axis check passed: B=2 == 2x(B=1), diff = {b2_diff:.3e}")

    band_keys = sorted(mp["band"].keys())
    top = mp["band"][band_keys[-1]]
    bands = []
    for d in band_keys:
        bp = mp["band"][d]
        band = {"sr": bp["sr"], "hl": bp["hl"], "n_fft": bp["n_fft"],
                "crop_start": bp["crop_start"], "crop_stop": bp["crop_stop"]}
        # Optional per-band keys pass through verbatim (Rust mirrors the original
        # presence/threshold checks, e.g. top-band hpf only applies when > 0).
        for k in ("hpf_start", "hpf_stop", "lpf_start", "lpf_stop", "convert_channels"):
            if k in bp:
                band[k] = bp[k]
        bands.append(band)

    config_data = {
        "type": "uvr_vr",
        "sample_rate": mp["sr"],
        "stereo": True,
        "num_stems": 2,
        # Schema-required STFT trio = TOP band's values; the VR pipeline reads `bands`.
        "n_fft": top["n_fft"],
        "hop_length": top["hl"],
        "win_length": top["n_fft"],
        "window_size": VR_WINDOW_SIZE,
        "offset": net.offset,
        "bins": bins,
        "is_v51": bool(config["is_v51"]),
        "pre_filter_start": mp["pre_filter_start"],
        "pre_filter_stop": mp["pre_filter_stop"],
        # v5.0 GLOBAL waveform-domain channel transforms (v5.1 uses per-band
        # convert_channels in `bands` instead; both default off).
        "reverse": bool(mp.get("reverse", False)),
        "mid_side": bool(mp.get("mid_side", False)),
        "mid_side_b2": bool(mp.get("mid_side_b2", False)),
        "bands": bands,
        "aggr_split_bin": mp["band"][1]["crop_stop"],
        "primary_non_accom": config["primary_stem"] in NON_ACCOM_STEMS,
        "batch_size": 4,
        "dynamic_batch": True,
        "stem_names": list(config["stem_names"]),
    }
    config_path = _write_model_json(output_path, config_data)

    print(f"Converted UVR VR ({config['name']}, "
          f"{'v5.1 CascadedNet' if config['is_v51'] else 'v5.0 CascadedASPPNet'}): "
          f"{input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"Bins: {bins}, Bands: {len(bands)}, Window: {VR_WINDOW_SIZE}, "
          f"Offset: {net.offset}, Stems: {config['stem_names']}")


def convert_mdx_net(input_path: Path, output_path: Path):
    """Legacy MDX-Net passthrough: validate the existing .onnx + write the json."""
    from architectures.mdx_net import detect_config as detect_mdx_net_config, MDX_HOP
    config = detect_mdx_net_config(str(input_path))
    dim_f, dim_t = config["dim_f"], config["dim_t"]

    if input_path.resolve() != output_path.resolve():
        shutil.copyfile(input_path, output_path)

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    # Cross-check the graph's static F/T dims against the registry — a registry
    # typo or a mislabeled file must fail HERE, not as garbage audio at runtime.
    in0 = onnx_model.graph.input[0]
    dims = [d.dim_value for d in in0.type.tensor_type.shape.dim]
    if dims[1:] != [4, dim_f, dim_t]:
        raise ValueError(
            f"ONNX input shape {dims} does not match registry "
            f"[batch, 4, {dim_f}, {dim_t}] for {config['name']}")
    print(f"ONNX check passed: {output_path} (input [batch, 4, {dim_f}, {dim_t}])")

    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    smoke = sess.run(None, {"input": np.zeros((1, 4, dim_f, dim_t), dtype=np.float32)})[0]
    if smoke.shape != (1, 4, dim_f, dim_t) or not np.isfinite(smoke).all():
        raise ValueError(f"ORT smoke run failed: output shape {smoke.shape}")
    print("ORT smoke run passed")

    config_data = {
        "type": "mdx_net",
        "sample_rate": 44100,
        "stereo": True,
        "num_stems": 1,
        "n_fft": config["n_fft"],
        "hop_length": MDX_HOP,
        "win_length": config["n_fft"],
        "dim_f": dim_f,
        "dim_t": dim_t,
        "chunk_size": MDX_HOP * (dim_t - 1),
        "num_overlap": 2,
        "compensate": config["compensate"],
        "batch_size": 1,
        "stem_names": list(config["stem_names"]),
        "residual_name": config["residual_name"],
    }
    config_path = _write_model_json(output_path, config_data)

    print(f"Converted MDX-Net ({config['name']}): {input_path} -> {output_path}")
    print(f"Config saved: {config_path}")
    print(f"n_fft: {config['n_fft']}, compensate: {config['compensate']}, "
          f"stem: {config['stem_names'][0]} (+residual {config['residual_name']})")


def main():
    parser = argparse.ArgumentParser(description="Convert .pth/.ckpt models to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Input .pth/.ckpt file")
    parser.add_argument("--output", type=str, required=True, help="Output .onnx file")
    parser.add_argument("--type", type=str, required=True,
                        choices=["rvc", "sovits", "bs_roformer", "mel_band_roformer",
                                 "mdx23c", "htdemucs", "uvr_vr", "mdx_net"],
                        help="Model type")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional MSST training YAML (STFT params for mel_band_roformer / "
                             "mdx23c; inference chunk_size/num_overlap; hyperparams + segment "
                             "for htdemucs; stem labels for all separation models). For sovits: "
                             "path to the so-vits config.json (default: auto-detect next to "
                             "the .pth)")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "both"],
                        help="Separation types only (rvc/sovits ignore it). fp32: <stem>.onnx "
                             "as today. fp16: convert to <stem>.fp16.onnx and delete the fp32 "
                             ".onnx (the shared <stem>.json is kept). both: keep both files.")
    parser.add_argument("--deterministic", action="store_true",
                        help="rvc/sovits only: zero SineGen's in-graph randomness for "
                             "numerical gate builds. Shipping conversions must NOT use this.")
    args = parser.parse_args()

    # Refuse unverified archs BEFORE touching any file — the torch export is
    # expensive and the result would be numerically unvalidated.
    if args.precision != "fp32" and args.type in SEPARATION_TYPES - FP16_VERIFIED_TYPES:
        print(f"Error: fp16 not yet numerically verified for {args.type}; use fp32",
              file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.type == "rvc":
        convert_rvc(input_path, output_path, deterministic=args.deterministic)
    elif args.type == "sovits":
        config_json = Path(args.config) if args.config else None
        convert_sovits(input_path, output_path, config_json,
                       deterministic=args.deterministic)
    elif args.type == "bs_roformer":
        config_yaml = Path(args.config) if args.config else None
        convert_bs_roformer(input_path, output_path, config_yaml)
    elif args.type == "mel_band_roformer":
        config_yaml = Path(args.config) if args.config else None
        convert_mel_band_roformer(input_path, output_path, config_yaml)
    elif args.type == "mdx23c":
        config_yaml = Path(args.config) if args.config else None
        convert_mdx23c(input_path, output_path, config_yaml)
    elif args.type == "htdemucs":
        config_yaml = Path(args.config) if args.config else None
        convert_htdemucs(input_path, output_path, config_yaml)
    elif args.type == "uvr_vr":
        convert_uvr_vr(input_path, output_path)
    elif args.type == "mdx_net":
        convert_mdx_net(input_path, output_path)

    if args.precision != "fp32" and args.type in SEPARATION_TYPES:
        # The torch export above MUST stay fp32 — it is the intermediate the
        # fp16 conversion consumes. Lazy import: rvc/sovits-only environments
        # don't need onnxconverter-common.
        from onnx_fp16 import convert_onnx_to_fp16, default_fp16_path
        fp16_path = default_fp16_path(output_path)
        convert_onnx_to_fp16(output_path, fp16_path)
        print(f"Converted to fp16: {fp16_path}")
        if args.precision == "fp16":
            # Keep the .json — it is shared by both precisions.
            output_path.unlink()
            print(f"Removed fp32 intermediate: {output_path}")


if __name__ == "__main__":
    main()
