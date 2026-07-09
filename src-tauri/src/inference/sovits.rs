//! so-vits-svc 4.x inference — faithful port of the ORIGINAL slice_inference
//! (D:\MyDev\so-vits-svc\so-vits-svc\inference\infer_tool.py: Svc.slice_inference + the
//! `inference\slicer.py` it uses + Svc.infer/get_unit_f0):
//!   native mono → silence-slice (slicer.cut / Slicer.slice, db threshold) into ordered
//!   (silent | non-silent) chunks tiling the whole signal → clip_seconds-window each
//!   non-silent chunk into ≤ CLIP_SECONDS pieces (the VRAM bound) → per piece run
//!   `infer_segment` (== the original per-slice infer path) → silent chunks become
//!   length-matched zeros → concat → final pad_array to the whole-input per_length.
//!
//! `infer_segment` (ONE non-silent piece, == the original per-slice infer):
//!   zero-pad 0.5 s both ends (at NATIVE sr, like slice_inference) → resample to model sr
//!   (wav_m) and 16 kHz (wav16k) → RMVPE f0 (thred 0.05) → RMVPEF0Predictor.post_process
//!   (resize + uv + gap interp) → f0 shift → ContentVec → repeat_expand to the hop grid →
//!   optional cluster/feature retrieval → optional Volume_Extractor vol → ONNX (explicit
//!   noise input) → change_rms (on the PADDED signals, like the original) →
//!   trim 0.5 s·model_sr. The caller then pad_arrays each piece to its per_length.
//!
//! WHY chunk: the whole-segment path ran the WHOLE song through the voice model in one
//! forward pass — a 126 s vocal peaked ~11.5 GB (the synthesizer's O(T²) rel-pos attention
//! + O(samples) decoder). Bounding each forward to ≤ CLIP_SECONDS caps that activation
//! (RVC's rvc.rs already does this via opt_ts silence-seek + t_max windows; this mirrors
//! the discipline with so-vits' own slicer semantics).
//!
//! DOCUMENTED deviations:
//!   - both resamples are scipy-exact resample_poly (original: torchaudio sinc Resample;
//!     the original even uses TWO different 16 k signals — RMVPE resamples 44.1k→16k
//!     internally with lowpass_filter_width=128 while hubert gets the default-width
//!     resample — we feed ONE wav16k to both)
//!   - noise is an explicit graph input (N(0,1)·noise_scale), seeded per-segment via
//!     seg_rng(seed, seg_idx) (the seed splitmixed with the piece index, like rvc.rs) so
//!     each piece gets independent-but-reproducible noise; a single-segment input reduces
//!     to seed_from_u64(seed), i.e. byte-identical to the old whole-segment path
//!   - slice_inference's lg_num crossfade (default 0) is NOT ported — pieces butt-join at
//!     silence / clip boundaries exactly like the original with lg_num=0
//!   - cluster assets come from converter-emitted .npy files (integration point below)

use std::sync::Arc;

use ndarray::{Array2, Array4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::diffusion::{self, DiffusionSchedule, SamplerMethod};
use super::engine::{InputTensor, OnnxEngine};
use super::features::{
    change_rms, contentvec_extract, knn_blend, librosa_rms, nearest_cluster_centers,
    reflect_pad_np, repeat_expand_2d, resample, KnnIndex,
};
use super::nsf_hifigan::{self, VocoderConfig};
use super::{SovitsOptions, SynthesisResult};
use crate::audio::AudioBuffer;
use crate::{Result, UtaiError};

/// slice_inference default pad_seconds (per NON-silent piece).
const PAD_SECONDS: f64 = 0.5;
/// slice_inference clip_seconds — hard-splits a long non-silent run into ≤ this many
/// seconds so a sustained vocal with no silence still bounds the voice-model forward.
/// 30 s ≈ 2.7k frames @44.1k/512 vs the whole-song ~10.8k — cuts the O(T²) attention ~16×.
const CLIP_SECONDS: f64 = 30.0;
/// slice_inference slice_db — silence threshold fed to slicer.cut. -40 dB is the common
/// so-vits inference default; more-negative = only genuinely silent gaps become zeros.
const SLICE_DB: f32 = -40.0;

/// Per-speaker cluster asset (so-vits cluster_infer_ratio path). Two kinds exist upstream:
/// kmeans centers (cluster/__init__.py get_cluster_center_result: nearest-center replace)
/// and feature-retrieval faiss indexes (same top-8 weighting as RVC).
pub enum ClusterAsset {
    KmeansCenters(KnnIndex),
    FeatureIndex(KnnIndex),
}

/// Shallow-diffusion runtime (resolved by the command layer from `<stem>.diffusion/`,
/// validated against the main model there — see commands/inference.rs).
pub struct DiffusionRuntime {
    pub encoder_session: String,
    pub denoiser_session: String,
    pub schedule: DiffusionSchedule,
    /// Parsed ONCE by the command layer (resolve_sovits_quality) — the pipeline never
    /// re-parses the wire string.
    pub method: SamplerMethod,
    /// n_hidden of the condition (WaveNet conditioner width, diffusion.json n_hidden).
    pub n_hidden: usize,
    /// diffusion.json n_spk — the encoder graph has a `spk_mix [T, n_spk]` input iff > 1.
    pub n_spk: usize,
    /// diffusion.json unit_interpolate_mode — used for the ContentVec repeat_expand ONLY
    /// under only_diffusion (original infer_tool.py:156; shallow uses the MAIN model's).
    pub unit_interpolate_mode: String,
}

/// NSF-HiFiGAN vocoder runtime (the aux nsf_hifigan.onnx + its slaney filterbank),
/// shared by shallow diffusion (mel→audio) and the enhancer.
pub struct VocoderRuntime {
    pub session: String,
    pub mel_filters: Arc<Array2<f32>>,
    pub cfg: VocoderConfig,
}

/// Session handles + model facts (resolved by the command layer from the sidecar config).
pub struct SovitsModel<'a> {
    pub engine: &'a OnnxEngine,
    pub voice_session: &'a str,
    pub contentvec_session: &'a str,
    pub rmvpe_session: &'a str,
    pub mel_filters: &'a Array2<f32>,
    pub cluster: Option<&'a ClusterAsset>,
    /// Present iff a diffusion mode is REQUESTED this run (command layer resolves+validates).
    pub diffusion: Option<DiffusionRuntime>,
    /// Present iff diffusion OR the enhancer is requested (both vocode through it).
    pub vocoder: Option<VocoderRuntime>,
    /// Present iff auto_f0 is requested (`<stem>.f0.onnx` session).
    pub f0_predictor_session: Option<String>,
    pub sample_rate: u32,
    pub hop_size: usize,
    pub features_dim: usize,
    /// Whether the exported graph HAS a vol input — decided by the sidecar "inputs"
    /// array (fallback: vol_embedding bool). The vol tensor is only fed when true.
    pub vol_embedding: bool,
    /// ①c: `Some(n_spk)` iff the graph HAS a "spk_mix" input (genuine multi-speaker export,
    /// n_spk = emb_g table width) — then a dense [1, n_spk] blend is fed in place of scalar `sid`.
    /// `None` = single-speaker / pre-①c export → the `sid` i64 path (byte-identical).
    pub spk_mix: Option<usize>,
    /// config.unit_interpolate_mode, default 'left' (infer_tool.py line 142).
    pub unit_interpolate_mode: String,
    /// inter_channels of the noise input (192 for sovits 4.0/4.1).
    pub noise_channels: usize,
    /// Minimum frame count the exported graph accepts (sidecar "min_frames", 6 for SoVITS).
    /// Now checked PER piece; the 0.5 s zero-pad each side already guarantees ≥ ~172 frames
    /// so this only trips on a sub-~10 ms clip remainder (zero-filled, not errored).
    pub min_frames: usize,
}

pub fn run_pipeline(
    m: &SovitsModel,
    audio: &AudioBuffer,
    options: &SovitsOptions,
    progress: &dyn Fn(f32),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<SynthesisResult> {
    if audio.samples.is_empty() {
        return Err(UtaiError::Audio("输入音频为空".into()));
    }
    progress(0.02);

    // ── mono-ize once at native sr; the whole pipeline downstream is native-sr samples ──
    let mono = crate::audio::resample::to_mono(audio);
    let native_sr = mono.sample_rate;
    let total_in = mono.samples.len();

    // resampled (model-sr) length of a native-sr span — slice_inference's
    // `int(ceil(len(data)/audio_sr*target_sample))`.
    let out_len = |len_native: usize| -> usize {
        (len_native as f64 / native_sr as f64 * m.sample_rate as f64).ceil() as usize
    };

    // ── silence slicing (slicer.cut + chunks2audio semantics) ──
    let slices = silence_slices(&mono.samples, native_sr, SLICE_DB);
    // clip_seconds window in native samples (the memory bound); .max(1) guards a div/loop.
    let per_size = ((native_sr as f64 * CLIP_SECONDS) as usize).max(1);

    // total non-silent pieces = infer_segment calls = ceil(len/per_size) summed (progress).
    let total_windows: usize = slices
        .iter()
        .filter(|s| !s.silent)
        .map(|s| ((s.end - s.start) + per_size - 1) / per_size)
        .sum();

    let mut audio_out: Vec<f32> = Vec::with_capacity(out_len(total_in) + per_size);
    let mut seg_idx: u64 = 0;
    let mut done_windows: usize = 0;

    for s in &slices {
        let seg_len = s.end - s.start;
        if s.silent {
            // slice_inference: silent chunk → zeros of the resampled length (no model run).
            audio_out.resize(audio_out.len() + out_len(seg_len), 0.0);
            continue;
        }
        // split_list_by_n(data, per_size): butt-joined clip windows, no overlap (lg_num=0).
        let mut w = s.start;
        while w < s.end {
            let w_end = (w + per_size).min(s.end);
            let piece_in = &mono.samples[w..w_end];
            let per_length = out_len(w_end - w);

            // coarse per-piece progress band inside [0.02, 0.98]; infer_segment sub-reports.
            let span = total_windows.max(1) as f32;
            let band_lo = 0.02 + 0.96 * (done_windows as f32) / span;
            let band_hi = 0.02 + 0.96 * ((done_windows + 1) as f32) / span;
            let report = |f: f32| progress(band_lo + (band_hi - band_lo) * f.clamp(0.0, 1.0));

            if cancel() {
                return Err(UtaiError::Inference("已取消".into()));
            }
            let trimmed = infer_segment(m, piece_in, native_sr, seg_idx, options, &report, cancel)?;
            // pad_array(_audio, per_length) — center zero-pad up (never truncates), exactly
            // like the original; a zero-filled degenerate piece becomes zeros(per_length).
            let piece = pad_array_center(trimmed, per_length);
            audio_out.extend_from_slice(&piece);

            seg_idx += 1;
            done_windows += 1;
            w = w_end;
        }
    }

    // ── end-to-end length contract: pad the FULL output up to the whole-input per_length
    //    (matches the old single-segment run_pipeline's final pad_array_center). For
    //    native_sr == model_sr this is a no-op — the per-piece pads already sum exactly. ──
    audio_out = pad_array_center(audio_out, out_len(total_in));
    progress(1.0);

    Ok(SynthesisResult {
        audio: audio_out,
        sample_rate: m.sample_rate,
    })
}

/// ONE non-silent piece → trimmed model-sr audio (pad removed; caller pad_arrays to
/// per_length). This is the original slice_inference per-slice infer path verbatim
/// (Svc.infer + get_unit_f0 around a 0.5 s-padded segment). `report` receives a local
/// [0,1] fraction that the caller maps into this piece's global progress band.
///
/// A degenerate piece (n_frames < min_frames, or too short to survive the 0.5 s trim)
/// returns an EMPTY vec — pad_array_center then yields zeros(per_length), matching the
/// original's `_audio[pad:-pad]` collapsing to empty on a tiny remainder. This can only
/// happen for a sub-~10 ms clip remainder, never for a real slice (the 0.5 s pad each side
/// alone guarantees ≥ ~172 frames @44.1k/512 ≫ min_frames=6).
fn infer_segment(
    m: &SovitsModel,
    seg: &[f32],
    native_sr: u32,
    seg_idx: u64,
    options: &SovitsOptions,
    report: &dyn Fn(f32),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<Vec<f32>> {
    // Stage fractions: the plain path keeps its S35 milestones; a diffusion run re-bands
    // so the sampler loop owns a visible 0.5→0.9 sweep (per denoise eval).
    let has_diff = m.diffusion.is_some();
    let (p_f0, p_cv, p_vits) = if has_diff { (0.2, 0.35, 0.5) } else { (0.4, 0.65, 0.95) };
    // slice_inference pad: zeros at the NATIVE sr, then resample (original order).
    let pad_native = (native_sr as f64 * PAD_SECONDS) as usize; // int(audio_sr * pad_seconds)
    let mut padded = vec![0.0f32; pad_native];
    padded.extend_from_slice(seg);
    padded.extend(std::iter::repeat(0.0).take(pad_native));

    let wav_m = resample(&padded, native_sr, m.sample_rate);
    let wav16k = resample(&wav_m, m.sample_rate, super::f0::RMVPE_SR);

    let n_frames = wav_m.len() / m.hop_size;
    let pad_out = (m.sample_rate as f64 * PAD_SECONDS) as usize;
    // Degenerate-piece guard. The onnx contract emits exactly n_frames*hop_size samples,
    // so n_frames*hop_size <= 2*pad_out means the 0.5 s trim would leave nothing.
    if n_frames < m.min_frames || n_frames * m.hop_size <= 2 * pad_out {
        report(1.0);
        return Ok(Vec::new());
    }

    // ── f0 + uv (RMVPEF0Predictor.compute_f0_uv semantics) ──
    let f0_raw = super::f0::rmvpe_detect(
        m.engine,
        m.rmvpe_session,
        m.mel_filters,
        &wav16k,
        super::f0::SOVITS_RMVPE_THRESHOLD,
    )?;
    // compute_f0_uv: torch.all(f0 == 0) short-circuits BEFORE post_process
    let (mut f0, uv) = if f0_raw.iter().all(|&v| v == 0.0) {
        (vec![0.0f32; n_frames], vec![0.0f32; n_frames])
    } else {
        super::f0::sovits_f0_postprocess(&f0_raw, n_frames, m.hop_size, m.sample_rate)
    };
    // get_unit_f0: f0 = f0 * 2^(tran/12) AFTER the predictor post-process; uv untouched
    let ratio = 2.0f32.powf(options.f0_shift / 12.0);
    f0.iter_mut().for_each(|v| *v *= ratio);
    report(p_f0); // f0 done

    if cancel() {
        return Err(UtaiError::Inference("已取消".into()));
    }

    // ── content features: ContentVec (50 fps) → repeat_expand to the hop grid.
    //    only_diffusion uses the DIFFUSION yaml's interpolate mode (original
    //    infer_tool.py:156 — the whole Svc is configured from the diffusion yaml there);
    //    every other path uses the main model's (line 142). ──
    let expand_mode: &str = match (&m.diffusion, options.only_diffusion) {
        (Some(diff), true) => &diff.unit_interpolate_mode,
        _ => &m.unit_interpolate_mode,
    };
    let c_raw = contentvec_extract(m.engine, m.contentvec_session, &wav16k, m.features_dim)?;
    let mut c = repeat_expand_2d(&c_raw, n_frames, expand_mode)?;
    report(p_cv); // content features done

    // ── cluster / feature retrieval (get_unit_f0 cluster_infer_ratio path) ──
    if options.cluster_ratio > 0.0 {
        match m.cluster {
            Some(ClusterAsset::FeatureIndex(index)) => {
                // identical weighting to RVC retrieval (original squared-L2 metric —
                // the cosine option is RVC-only); blend handled inside knn_blend
                c = knn_blend(&c, index, options.cluster_ratio, false);
            }
            Some(ClusterAsset::KmeansCenters(centers)) => {
                let cluster_c = nearest_cluster_centers(&c, centers);
                let r = options.cluster_ratio;
                c.zip_mut_with(&cluster_c, |orig, &cl| *orig = r * cl + (1.0 - r) * *orig);
            }
            None => {
                tracing::warn!(
                    "cluster_ratio={} 但该模型没有可用的聚类/检索资产——跳过（与原版缺文件时的行为一致）",
                    options.cluster_ratio
                );
            }
        }
    }

    // ── vol (utils.Volume_Extractor on the padded model-sr signal, iff vol_embedding) ──
    let vol = if m.vol_embedding {
        let v = extract_volume(&wav_m, m.hop_size);
        if v.len() != n_frames {
            return Err(UtaiError::Inference(format!(
                "vol 帧数异常：{} != {}",
                v.len(),
                n_frames
            )));
        }
        Some(v)
    } else {
        None
    };

    let t = n_frames as i64;

    // ── ①c speaker feed: a multi-speaker graph (m.spk_mix = Some(n_spk)) takes a dense
    //    spk_mix [1, n_spk] f32 blend in place of the scalar sid; both the main graph and the
    //    f0 companion share the SAME vector (matching the export, which renamed the input in
    //    both). None → the sid i64 path (single-speaker / pre-①c export, byte-identical). ──
    let spk_mix_dense: Option<Vec<f32>> = m
        .spk_mix
        .map(|n_spk| super::build_spk_mix_dense(&options.spk_mix, options.speaker_id, n_spk));

    // ── auto f0 (models.py:523-527 via `<stem>.f0.onnx`): the predictor consumes the SAME
    //    inputs the main graph is about to get (post-cluster c, shifted f0, uv, sid[, vol])
    //    and its output REPLACES f0 everywhere downstream — enc_p (coarse in-graph), the NSF
    //    source, the diffusion condition, the vocoder and the enhancer (infer_tool.py:297:
    //    the returned f0 IS the predicted one). uv stays the source-detected mask. ──
    if let Some(f0p_sid) = &m.f0_predictor_session {
        let mut p_inputs = vec![
            (
                "c",
                InputTensor::F32 {
                    data: c.iter().copied().collect(),
                    shape: vec![1, t, m.features_dim as i64],
                },
            ),
            (
                "f0",
                InputTensor::F32 {
                    data: f0.clone(),
                    shape: vec![1, t],
                },
            ),
            (
                "uv",
                InputTensor::F32 {
                    data: uv.clone(),
                    shape: vec![1, t],
                },
            ),
        ];
        if let Some(mix) = &spk_mix_dense {
            p_inputs.push((
                "spk_mix",
                InputTensor::F32 {
                    data: mix.clone(),
                    shape: vec![1, mix.len() as i64],
                },
            ));
        } else {
            p_inputs.push((
                "sid",
                InputTensor::I64 {
                    data: vec![options.speaker_id.unwrap_or(0) as i64],
                    shape: vec![1],
                },
            ));
        }
        if let Some(v) = &vol {
            p_inputs.push((
                "vol",
                InputTensor::F32 {
                    data: v.clone(),
                    shape: vec![1, t],
                },
            ));
        }
        let outs = m.engine.run(f0p_sid, p_inputs)?;
        let f0_pred = outs
            .into_iter()
            .next()
            .ok_or_else(|| UtaiError::Inference("自动音高预测器没有返回输出".into()))?;
        if f0_pred.len() != n_frames {
            return Err(UtaiError::Inference(format!(
                "自动音高预测帧数异常：{} != {}",
                f0_pred.len(),
                n_frames
            )));
        }
        f0 = f0_pred;
    }

    // ── VITS main graph (skipped entirely in only_diffusion — the original never even
    //    loads net_g there; audio for the diffusion vol basis is the SOURCE wav) ──
    let vits_out: Option<Vec<f32>> = if options.only_diffusion {
        None
    } else {
        // per-piece seeded noise (independent reproducible pieces)
        let noise = seg_noise(m.noise_channels, n_frames, options.seed, seg_idx, options.noise_scale);

        let mut inputs = vec![
            (
                "c",
                InputTensor::F32 {
                    data: c.iter().copied().collect(),
                    shape: vec![1, t, m.features_dim as i64],
                },
            ),
            (
                "f0",
                InputTensor::F32 {
                    data: f0.clone(),
                    shape: vec![1, t],
                },
            ),
            (
                "uv",
                InputTensor::F32 {
                    data: uv.clone(),
                    shape: vec![1, t],
                },
            ),
            (
                "noise",
                InputTensor::F32 {
                    data: noise,
                    shape: vec![1, m.noise_channels as i64, t],
                },
            ),
        ];
        if let Some(mix) = &spk_mix_dense {
            inputs.push((
                "spk_mix",
                InputTensor::F32 {
                    data: mix.clone(),
                    shape: vec![1, mix.len() as i64],
                },
            ));
        } else {
            inputs.push((
                "sid",
                InputTensor::I64 {
                    data: vec![options.speaker_id.unwrap_or(0) as i64],
                    shape: vec![1],
                },
            ));
        }
        if let Some(v) = &vol {
            inputs.push((
                "vol",
                InputTensor::F32 {
                    data: v.clone(),
                    shape: vec![1, t],
                },
            ));
        }

        let outputs = m.engine.run(m.voice_session, inputs)?;
        let out = outputs
            .into_iter()
            .next()
            .ok_or_else(|| UtaiError::Inference("SoVITS 模型没有返回输出".into()))?;
        Some(out)
    };
    report(p_vits);

    // ── diffusion block (infer_tool.py:307-328): shallow refines the VITS mel, only_diffusion
    //    generates from pure noise; both vocode through NSF-HiFiGAN with the CURRENT f0 ──
    let mut out: Vec<f32> = match (&m.diffusion, &m.vocoder) {
        (Some(diff), Some(voc)) => {
            if cancel() {
                return Err(UtaiError::Inference("已取消".into()));
            }
            // vol for the DIFFUSION condition (line 308): vol_embedding → the source-wav vol
            // (reused); else extracted from `audio` = VITS output (shallow) / source (only).
            let vol_d: Vec<f32> = match &vol {
                Some(v) => v.clone(),
                None => {
                    let basis: &[f32] = match &vits_out {
                        Some(a) => a,
                        None => &wav_m,
                    };
                    extract_volume(basis, m.hop_size)
                }
            };
            // second_encoding (lines 309-314, shallow only): re-extract ContentVec from the
            // VITS output resampled to 16 k, repeat_expand with the MAIN model's mode (the
            // diffusion yaml's mode only applies to only_diffusion upstream). DEVIATION note:
            // the original misses c.unsqueeze(0) here (latent shape bug) — we keep the sane
            // batched layout; the E2E gate pins the original's actual behavior.
            if options.second_encoding && vits_out.is_some() {
                let out16k = resample(
                    vits_out.as_deref().unwrap(),
                    m.sample_rate,
                    super::f0::RMVPE_SR,
                );
                let c2 = contentvec_extract(m.engine, m.contentvec_session, &out16k, m.features_dim)?;
                c = repeat_expand_2d(&c2, n_frames, &m.unit_interpolate_mode)?;
            }
            run_diffusion(
                m, diff, voc, &c, &f0, &vol_d, options, seg_idx,
                vits_out.as_deref(), n_frames, report, cancel,
            )?
        }
        _ => vits_out.ok_or_else(|| {
            UtaiError::Inference("内部错误：既无 VITS 输出也无扩散配置".into())
        })?,
    };

    // ── NSF-HiFiGAN enhancer (infer_tool.py:329-335) — plain VITS path only; the command
    //    layer already force-disabled it under any diffusion mode (original mutual exclusion).
    if options.nsf_enhance && !has_diff {
        let voc = m.vocoder.as_ref().ok_or_else(|| {
            UtaiError::Inference("内部错误：增强器开启但声码器未装载".into())
        })?;
        out = nsf_hifigan::enhance(
            m.engine,
            &voc.session,
            &voc.mel_filters,
            &voc.cfg,
            &out,
            m.sample_rate,
            &f0,
            m.hop_size,
            options.enhancer_adaptive_key,
        )?;
    }
    report(0.95);

    // ── loudness envelope: original applies change_rms INSIDE infer(), i.e. on the
    //    still-PADDED input/output, BEFORE slice_inference trims — mirror that order ──
    if options.loudness_envelope != 1.0 {
        change_rms(&wav_m, m.sample_rate, &mut out, m.sample_rate, options.loudness_envelope);
    }

    // ── slice_inference output handling: trim int(target_sample·pad_seconds). (The
    //    per_length pad_array is the caller's — matches _audio = pad_array(_audio, per).) ──
    if out.len() <= 2 * pad_out {
        // Unreachable given the guard above (kept defensive), treat as degenerate.
        report(1.0);
        return Ok(Vec::new());
    }
    let trimmed = out[pad_out..out.len() - pad_out].to_vec();
    report(1.0);
    Ok(trimmed)
}

/// Domain-separation constant XORed into the user seed for the DIFFUSION noise stream —
/// keeps it independent from (but as reproducible as) the VITS z-noise stream of the
/// same piece.
const DIFF_RNG_DOMAIN: u64 = 0xD1FF_05ED_5EED_0001;

/// The per-piece diffusion pass (GaussianDiffusion.forward infer branch, unit2mel.py:131-166
/// condition build): encoder.onnx → cond; shallow = norm_spec(gt mel)+q_sample(k_step-1),
/// only = randn full-depth; host-side sampler loop calling denoiser.onnx per step; denorm →
/// NSF-HiFiGAN vocode with the (possibly auto-predicted) f0. Reports 0.5→0.9 across denoise
/// evals, 0.95 handled by the caller.
#[allow(clippy::too_many_arguments)]
fn run_diffusion(
    m: &SovitsModel,
    diff: &DiffusionRuntime,
    voc: &VocoderRuntime,
    c: &Array2<f32>,
    f0: &[f32],
    vol: &[f32],
    options: &SovitsOptions,
    seg_idx: u64,
    vits_out: Option<&[f32]>,
    n_frames: usize,
    report: &dyn Fn(f32),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<Vec<f32>> {
    let t = n_frames as i64;
    let num_mels = voc.cfg.num_mels;

    // ── condition (encoder.onnx == Unit2Mel embeds; spk one-hot MatMul when multi-spk) ──
    let mut enc_inputs = vec![
        (
            "units",
            InputTensor::F32 {
                data: c.iter().copied().collect(),
                shape: vec![1, t, m.features_dim as i64],
            },
        ),
        (
            "f0",
            InputTensor::F32 {
                data: f0.to_vec(),
                shape: vec![1, t],
            },
        ),
        (
            "volume",
            InputTensor::F32 {
                data: vol.to_vec(),
                shape: vec![1, t],
            },
        ),
    ];
    let spk_mix; // keep alive for the input borrow
    if diff.n_spk > 1 {
        let spk = options.speaker_id.unwrap_or(0) as usize;
        let mut mix = vec![0.0f32; n_frames * diff.n_spk];
        for row in 0..n_frames {
            mix[row * diff.n_spk + spk] = 1.0;
        }
        spk_mix = mix;
        enc_inputs.push((
            "spk_mix",
            InputTensor::F32 {
                data: spk_mix,
                shape: vec![t, diff.n_spk as i64],
            },
        ));
    }
    let cond = m
        .engine
        .run(&diff.encoder_session, enc_inputs)?
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("扩散条件编码器没有返回输出".into()))?;
    if cond.len() != diff.n_hidden * n_frames {
        return Err(UtaiError::Inference(format!(
            "扩散条件形状异常：{} != {}×{}",
            cond.len(),
            diff.n_hidden,
            n_frames
        )));
    }

    // ── denoiser closure: one ε(x, t, cond) eval per call, cancellable per step ──
    let denoise = |x: &Array4<f32>, step_t: f32| -> Result<Array4<f32>> {
        if cancel() {
            return Err(UtaiError::Inference("已取消".into()));
        }
        let outs = m.engine.run(
            &diff.denoiser_session,
            vec![
                (
                    "x",
                    InputTensor::F32 {
                        data: x.iter().copied().collect(),
                        shape: vec![1, 1, num_mels as i64, t],
                    },
                ),
                (
                    "time",
                    InputTensor::F32 {
                        data: vec![step_t],
                        shape: vec![1],
                    },
                ),
                (
                    "cond",
                    InputTensor::F32 {
                        data: cond.clone(),
                        shape: vec![1, diff.n_hidden as i64, t],
                    },
                ),
            ],
        )?;
        let eps = outs
            .into_iter()
            .next()
            .ok_or_else(|| UtaiError::Inference("扩散去噪网络没有返回输出".into()))?;
        Array4::from_shape_vec((1, 1, num_mels, n_frames), eps)
            .map_err(|e| UtaiError::Inference(format!("扩散去噪输出形状异常：{}", e)))
    };

    // ── sampler (host-side; the solvers rebuild their schedule over betas[..t]) ──
    let method = diff.method;
    let mut rng = seg_rng(options.seed ^ DIFF_RNG_DOMAIN, seg_idx);
    let mut noise_src = if options.debug_zero_noise {
        diffusion::NoiseSource::Zero
    } else {
        diffusion::NoiseSource::Rng(&mut rng)
    };
    let speedup = options.diffusion_speedup as usize;
    let k_step = options.k_step as usize;
    // per-eval fraction from the sampler → the piece's 0.5..0.9 band
    let prog = |f: f32| report(0.5 + 0.4 * f.min(1.0));

    let x_out: Array4<f32> = match vits_out {
        Some(vits) => {
            // shallow: gt mel from the VITS output. Output length is exactly n_frames·hop,
            // so the nvSTFT frame count MUST equal n_frames — assert instead of the
            // original's silent tolerance (a mismatch means broken geometry upstream).
            let mel_gt = super::mel::nsf_mel(vits, &voc.mel_filters);
            if mel_gt.ncols() != n_frames || mel_gt.nrows() != num_mels {
                return Err(UtaiError::Inference(format!(
                    "浅扩散 mel 形状异常：[{},{}] != [{},{}]",
                    mel_gt.nrows(),
                    mel_gt.ncols(),
                    num_mels,
                    n_frames
                )));
            }
            // [M,T] → [1,1,M,T] is a zero-copy reshape (nsf_mel output is standard layout)
            let gt4 = mel_gt
                .insert_axis(ndarray::Axis(0))
                .insert_axis(ndarray::Axis(0));
            let x0_norm = diff.schedule.norm_spec(&gt4);
            diffusion::sample_shallow(
                &diff.schedule,
                &x0_norm,
                k_step,
                speedup,
                method,
                &mut noise_src,
                denoise,
                &prog,
            )?
        }
        None => diffusion::sample_full(
            &diff.schedule,
            num_mels,
            n_frames,
            speedup,
            method,
            &mut noise_src,
            denoise,
            &prog,
        )?,
    };

    // ── sampler output is already denormed [1,1,M,T] → [M,T] → NSF-HiFiGAN
    //    (Vocoder.infer trims f0 to the mel frame count) ──
    let mel_out =
        Array2::from_shape_fn((num_mels, n_frames), |(mm, tt)| x_out[[0, 0, mm, tt]]);
    report(0.9);
    nsf_hifigan::vocode(m.engine, &voc.session, &mel_out, f0)
}

/// Deterministic per-piece RNG: user seed splitmixed with the piece index (mirrors
/// rvc.rs chunk_rng). seg_idx=0 reduces to seed_from_u64(seed) → byte-identical to the
/// old single whole-segment noise draw.
fn seg_rng(seed: u64, seg_idx: u64) -> StdRng {
    StdRng::seed_from_u64(seed ^ seg_idx.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// The net_g explicit `noise` input: N(0,1)·scale, `channels·n_frames` values row-major (the ONNX
/// input is `[1, channels, T]`), drawn from the per-piece seg_rng. Extracted so the cover path
/// (infer_segment) and the S48 score path (score2svc) build the SAME noise byte-for-byte — the
/// S35 export moved net_g's internal randn OUT to this graph input, so reproducibility hinges on an
/// identical draw (seed + seg_idx + channel×frame count + scale, in this exact order).
pub(crate) fn seg_noise(
    channels: usize,
    n_frames: usize,
    seed: u64,
    seg_idx: u64,
    scale: f32,
) -> Vec<f32> {
    let mut rng = seg_rng(seed, seg_idx);
    (0..channels * n_frames)
        .map(|_| {
            let n: f32 = rng.sample(StandardNormal);
            n * scale
        })
        .collect()
}

/// One slice of the input: (silent?, start_sample, end_sample) in NATIVE-sr samples.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Slice {
    silent: bool,
    start: usize,
    end: usize,
}

/// Port of inference/slicer.py — `cut(path, db_thresh)` defaults (min_len=5000) →
/// `Slicer.slice` → `chunks2audio`, over a native-sr mono signal. Returns ordered slices
/// that tile [0, len) (zero-length dropped, exactly like chunks2audio's `tag[0]!=tag[1]`).
/// The RMS envelope reuses features::librosa_rms (the SAME center=True zero-pad
/// librosa.feature.rms verified there).
fn silence_slices(samples: &[f32], sr: u32, db_thresh: f32) -> Vec<Slice> {
    // Slicer.__init__ (min_length=5000, min_interval=300, hop_size=20, max_sil_kept=5000 ms)
    let threshold = 10f32.powf(db_thresh / 20.0);
    let hop = (sr as f64 * 20.0 / 1000.0).round() as usize; // samples
    let min_interval_samp = sr as f64 * 300.0 / 1000.0; // samples (float)
    let win_size = (min_interval_samp.round() as usize).min(4 * hop);
    let min_length = (sr as f64 * 5000.0 / 1000.0 / hop as f64).round() as usize; // frames
    let min_interval = (min_interval_samp / hop as f64).round() as usize; // frames
    let max_sil_kept = (sr as f64 * 5000.0 / 1000.0 / hop as f64).round() as usize; // frames

    let n = samples.len();
    let single = || vec![Slice { silent: false, start: 0, end: n }];
    // Slicer.slice: `samples.shape[0] <= min_length` → whole signal as one non-silent chunk
    // (verbatim — the original compares a SAMPLE count to a FRAME count; harmless guard).
    if n == 0 || hop == 0 || n <= min_length {
        return single();
    }

    let rms = librosa_rms(samples, win_size, hop);
    if rms.is_empty() {
        return single();
    }

    // absolute index of the FIRST minimum in rms[lo..hi] (np.argmin ties → first).
    let argmin_range = |lo: usize, hi: usize| -> usize {
        let hi = hi.min(rms.len());
        let lo = lo.min(hi.saturating_sub(1));
        let mut best = lo;
        for i in (lo + 1)..hi {
            if rms[i] < rms[best] {
                best = i;
            }
        }
        best
    };

    // ── Slicer.slice state machine (sil_tags = (start_frame, end_frame) silence spans) ──
    let mut sil_tags: Vec<(usize, usize)> = Vec::new();
    let mut silence_start: Option<usize> = None;
    let mut clip_start: usize = 0;
    let total_frames = rms.len();
    for i in 0..total_frames {
        if rms[i] < threshold {
            if silence_start.is_none() {
                silence_start = Some(i);
            }
            continue;
        }
        let ss = match silence_start {
            None => continue,
            Some(ss) => ss,
        };
        let is_leading_silence = ss == 0 && i > max_sil_kept;
        let need_slice_middle =
            i - ss >= min_interval && i - clip_start >= min_length;
        if !is_leading_silence && !need_slice_middle {
            silence_start = None;
            continue;
        }
        if i - ss <= max_sil_kept {
            let pos = argmin_range(ss, i + 1);
            if ss == 0 {
                sil_tags.push((0, pos));
            } else {
                sil_tags.push((pos, pos));
            }
            clip_start = pos;
        } else if i - ss <= max_sil_kept * 2 {
            let pos = argmin_range(i - max_sil_kept, ss + max_sil_kept + 1);
            let pos_l = argmin_range(ss, ss + max_sil_kept + 1);
            let pos_r = argmin_range(i - max_sil_kept, i + 1);
            if ss == 0 {
                sil_tags.push((0, pos_r));
                clip_start = pos_r;
            } else {
                sil_tags.push((pos_l.min(pos), pos_r.max(pos)));
                clip_start = pos_r.max(pos);
            }
        } else {
            let pos_l = argmin_range(ss, ss + max_sil_kept + 1);
            let pos_r = argmin_range(i - max_sil_kept, i + 1);
            if ss == 0 {
                sil_tags.push((0, pos_r));
            } else {
                sil_tags.push((pos_l, pos_r));
            }
            clip_start = pos_r;
        }
        silence_start = None;
    }
    // trailing silence
    if let Some(ss) = silence_start {
        if total_frames - ss >= min_interval {
            let silence_end = total_frames.min(ss + max_sil_kept);
            let pos = argmin_range(ss, silence_end + 1);
            sil_tags.push((pos, total_frames + 1));
        }
    }

    if sil_tags.is_empty() {
        return single();
    }

    // ── build chunks (Slicer.slice tail) then chunks2audio (drop empty/reversed) ──
    let clamp = |frame: usize| (frame * hop).min(n);
    let mut chunks: Vec<Slice> = Vec::new();
    let push = |silent: bool, start: usize, end: usize, out: &mut Vec<Slice>| {
        if start < end {
            out.push(Slice { silent, start, end });
        }
    };
    // 第一段静音并非从头开始，补上有声片段
    if sil_tags[0].0 != 0 {
        push(false, 0, clamp(sil_tags[0].0), &mut chunks);
    }
    for idx in 0..sil_tags.len() {
        if idx != 0 {
            push(false, clamp(sil_tags[idx - 1].1), clamp(sil_tags[idx].0), &mut chunks);
        }
        push(true, clamp(sil_tags[idx].0), clamp(sil_tags[idx].1), &mut chunks);
    }
    // 最后一段静音并非结尾，补上结尾片段
    let last_end = clamp(sil_tags[sil_tags.len() - 1].1);
    if last_end < n {
        push(false, last_end, n, &mut chunks);
    }

    if chunks.is_empty() {
        return single();
    }
    chunks
}

/// infer_tool.pad_array: center zero-pad up to target_length (unchanged if already ≥).
fn pad_array_center(arr: Vec<f32>, target_length: usize) -> Vec<f32> {
    let current = arr.len();
    if current >= target_length {
        return arr;
    }
    let pad_width = target_length - current;
    let pad_left = pad_width / 2;
    let pad_right = pad_width - pad_left;
    let mut out = vec![0.0f32; pad_left];
    out.extend(arr);
    out.extend(std::iter::repeat(0.0).take(pad_right));
    out
}

/// utils.Volume_Extractor.extract port:
///   n_frames = len // hop; pad audio² reflect (hop//2, (hop+1)//2);
///   per-hop-window mean → sqrt. (unfold produces len//hop+1 windows; the original
///   truncates to n_frames — window k covers padded[k·hop .. k·hop+hop).)
pub fn extract_volume(audio: &[f32], hop_size: usize) -> Vec<f32> {
    let n_frames = audio.len() / hop_size;
    let sq: Vec<f32> = audio.iter().map(|&v| v * v).collect();
    let padded = reflect_pad_np(&sq, hop_size / 2, (hop_size + 1) / 2);
    (0..n_frames)
        .map(|k| {
            let start = k * hop_size;
            let mean = padded[start..start + hop_size]
                .iter()
                .map(|&v| v as f64)
                .sum::<f64>()
                / hop_size as f64;
            mean.sqrt() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // python (original utils.Volume_Extractor.extract, ast-exec, hop_size=8):
    //   rng=np.random.RandomState(11); audio=(randn(50)*0.4).f32 → volume [6]
    #[test]
    fn volume_extractor_matches_original() {
        const VOLUME_IN: &[f32] = &[6.997818947e-01, -1.144291982e-01, -1.938260496e-01, -1.061327457e+00, -3.313851776e-03, -1.278525442e-01, -2.146517485e-01, 1.261610687e-01, 1.684202850e-01, -4.262411892e-01, -3.544958532e-01, -1.902934015e-01, 2.758729160e-01, 2.244768739e-01, -5.222194195e-01, -4.477901161e-01, 2.947349548e-01, 6.298536062e-01, -1.243003551e-02, -2.733786404e-01, 4.382518828e-01, -1.238306537e-01, 2.903008759e-01, 6.196286678e-01, 2.520319223e-01, 2.939729393e-02, 2.929085493e-01, -2.570301592e-01, -7.123727351e-02, -2.295818180e-01, -8.175012469e-02, -1.945980340e-01, -7.431013137e-02, -1.522145718e-01, 3.559105471e-02, 2.546866424e-02, 1.185388416e-01, 5.611084700e-01, -6.187450290e-01, 5.182474256e-01, -9.490017593e-02, -4.929384887e-01, -6.896790862e-02, 3.673534840e-02, 4.270233810e-01, -4.246537685e-01, 8.693928272e-02, 4.712780192e-02, -6.736443639e-01, -4.743021131e-01];
        const VOLUME_OUT: &[f32] = &[5.962238312e-01, 2.372432947e-01, 3.804930151e-01, 3.338894248e-01, 1.289092153e-01, 3.934495747e-01];
        let got = extract_volume(VOLUME_IN, 8);
        assert_eq!(got.len(), VOLUME_OUT.len());
        for (i, (g, w)) in got.iter().zip(VOLUME_OUT.iter()).enumerate() {
            assert!((g - w).abs() < 2e-6, "vol[{}]: {} vs {}", i, g, w);
        }
    }

    // infer_tool.pad_array semantics: center zero-pad, left = width//2
    #[test]
    fn pad_array_center_matches_original() {
        assert_eq!(
            pad_array_center(vec![1.0, 2.0, 3.0], 8),
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        );
        // already long enough → unchanged
        assert_eq!(pad_array_center(vec![1.0, 2.0], 2), vec![1.0, 2.0]);
        assert_eq!(pad_array_center(vec![1.0, 2.0, 3.0], 2), vec![1.0, 2.0, 3.0]);
    }

    // seg_rng: same seed+piece reproduces; different piece differs (mirrors rvc chunk_rng).
    #[test]
    fn seg_rng_is_deterministic_and_segment_distinct() {
        let draw = |seed: u64, idx: u64| -> Vec<f32> {
            let mut r = seg_rng(seed, idx);
            (0..8).map(|_| r.sample(StandardNormal)).collect()
        };
        assert_eq!(draw(42, 0), draw(42, 0), "same seed+piece must reproduce");
        assert_ne!(draw(42, 0), draw(42, 1), "different pieces must differ");
        // seg_idx=0 must equal the old whole-segment draw: seed_from_u64(seed)
        let old: Vec<f32> = {
            let mut r = StdRng::seed_from_u64(42);
            (0..8).map(|_| r.sample(StandardNormal)).collect()
        };
        assert_eq!(draw(42, 0), old, "piece 0 must match the pre-chunking noise draw");
    }

    // seg_noise must be byte-identical to the old inline draw (channels·frames values, row-major,
    // ·scale, from seg_rng) — the refactor that shares it with the score path must not drift the
    // S35 cover-path noise.
    #[test]
    fn seg_noise_matches_inline_draw() {
        let (channels, n_frames, seed, seg_idx, scale) = (3usize, 4usize, 42u64, 1u64, 0.4f32);
        let got = seg_noise(channels, n_frames, seed, seg_idx, scale);
        let want: Vec<f32> = {
            let mut rng = seg_rng(seed, seg_idx);
            (0..channels * n_frames)
                .map(|_| {
                    let n: f32 = rng.sample(StandardNormal);
                    n * scale
                })
                .collect()
        };
        assert_eq!(got.len(), channels * n_frames);
        assert_eq!(got, want, "seg_noise must reproduce the inline per-piece draw exactly");
    }

    // A pure-loud signal has no silence → one non-silent slice covering [0, n].
    #[test]
    fn silence_slices_no_silence_single_slice() {
        let sr = 16000u32;
        let samples = vec![0.5f32; 4000]; // > min_length (250) samples, all ≫ -40 dB
        let slices = silence_slices(&samples, sr, SLICE_DB);
        assert_eq!(slices, vec![Slice { silent: false, start: 0, end: 4000 }]);
    }

    // Signal shorter than min_length → single non-silent slice (verbatim early return).
    #[test]
    fn silence_slices_short_input_single_slice() {
        let slices = silence_slices(&vec![0.5f32; 100], 16000, SLICE_DB);
        assert_eq!(slices, vec![Slice { silent: false, start: 0, end: 100 }]);
    }

    // Loud then a long trailing silence → slices tile [0,n], first non-silent, a silent
    // slice exists, endpoints/contiguity hold (exact cut frame left to the state machine).
    #[test]
    fn silence_slices_trailing_silence_tiles() {
        let sr = 16000u32;
        let mut samples = vec![0.5f32; 4000]; // loud
        samples.extend(std::iter::repeat(0.0f32).take(8000)); // long silence (≫ min_interval)
        let n = samples.len();
        let slices = silence_slices(&samples, sr, SLICE_DB);
        assert!(slices.len() >= 2, "expected a loud + silent split, got {:?}", slices);
        assert_eq!(slices.first().unwrap().start, 0);
        assert!(!slices.first().unwrap().silent, "first slice should be non-silent");
        assert_eq!(slices.last().unwrap().end, n);
        assert!(slices.iter().any(|s| s.silent), "expected at least one silent slice");
        for w in slices.windows(2) {
            assert_eq!(w[0].end, w[1].start, "slices must tile without gaps/overlap");
        }
    }
}
