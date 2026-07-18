//! RVC inference — faithful port of the ORIGINAL pipeline
//! (D:\MyDev\RVC\RVC20240604Nvidia\infer\modules\vc\pipeline.py, Pipeline.pipeline + vc):
//!   16 kHz mono → butter(5,48,'high') filtfilt → opt_ts silence-seek chunking (fp32
//!   config: x_pad=1 x_query=6 x_center=38 x_max=41) → per chunk: ContentVec (50 fps) →
//!   optional KNN retrieval → optional L2 norm (our extra) → 2x nearest upsample →
//!   protect blend → ONNX (new converter signature with explicit rnd) → trim t_pad_tgt →
//!   concat → rms mix → optional output resample.
//!
//! DOCUMENTED deviations from the original (rationale in the task spec / code):
//!   - resampling is scipy-exact resample_poly (original: ffmpeg swr at load time)
//!   - audio stays f32 after the (f64) filtfilt — the original carries float64 to the
//!     encoder input where it casts to f32 anyway; difference is fp32 noise floor
//!   - KNN is EXACT brute-force top-8 (original: faiss IVF nprobe=1, approximate) with a
//!     1e-9 squared-distance clamp (original NaNs on an exact match)
//!   - rnd noise is an explicit graph input, seeded from options.seed and mixed with the
//!     chunk index (original: unseeded torch.randn inside net_g.infer)
//!   - NO int16 quantization/normalize at the end — we stay f32 for the DAW
//!   - f0_to_coarse rounds half-away-from-zero (original np.rint = half-to-even); only
//!     differs on exact .5 mel boundaries, measure-zero on real f0

use ndarray::{s, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::path::Path;

use super::engine::{InputTensor, OnnxEngine};
use super::features::{
    change_rms, contentvec_extract, highpass_48hz_16k, knn_blend, reflect_pad_np, resample,
    upsample_2x_nearest, KnnIndex,
};
use super::{RvcOptions, SynthesisResult};
use crate::audio::AudioBuffer;
use crate::{Result, UtaiError};

const SR: usize = 16000;
const WINDOW: usize = 160;
// fp32 config values (config.py: x_pad/x_query/x_center/x_max = 1/6/38/41 when not half)
const X_PAD: usize = 1;

/// S67c: silence-seek chunking tier. Upstream RVC itself selects these by device memory
/// (config.py device_config: 3/10/60/65 fp16, 1/6/38/41 fp32 — our former constants —
/// and 1/5/30/32 for ≤4 GB cards), so picking a smaller tier on a memory-starved
/// DirectML box ADOPTS the upstream mechanism rather than inventing numbers. x_pad stays
/// 1 in every tier (matching both upstream low tiers), so the seam mechanics (t_pad
/// reflect-pad + t_pad_tgt trim) are byte-identical across tiers — a tier only moves
/// WHERE the silence-seek cuts land, the same difference two upstream users with
/// different cards always had. Different cuts ⇒ different outputs, but net_g is
/// graph-stochastic run-to-run anyway (S54) — there is no fixed-output contract to break.
struct ChunkTier {
    x_query: usize,
    x_center: usize,
    x_max: usize,
    /// Minimum system-wide available commit (MB) to afford this tier on DirectML.
    /// Calibrated against the tier's WORST-CASE chunk, not x_max: consecutive opt_ts cuts
    /// can land t_query early and t_query late, so a mid-song chunk reaches
    /// x_center + 2·x_query (+ 2 s pads) — review round 2 caught the original x_max-based
    /// table under-providing by 20-40%. Formula: measured lengv2.3 ticket curve
    /// (ticket_size_by_t, ~98 MB/s linear: 43 s→4206 MB, 34 s→3361, 19 s→1704, 10 s→820)
    /// extrapolated to worst_t = x_center + 2·x_query + 2, × 1.3 model-variance headroom
    /// + 1 GB cushion. This table is a first line, not an absolute no-OOM guarantee —
    /// mid-run exhaustion is layered off to the S67b growth-recycle loop, the engine's
    /// pre-shape floor, and the Auto CPU fallback.
    need_mb: u64,
}

const CHUNK_TIERS: &[ChunkTier] = &[
    ChunkTier { x_query: 6, x_center: 38, x_max: 41, need_mb: 7800 }, // upstream fp32/5G (default); worst 52 s
    ChunkTier { x_query: 5, x_center: 30, x_max: 32, need_mb: 6500 }, // upstream ≤4GB tier; worst 42 s
    ChunkTier { x_query: 4, x_center: 17, x_max: 19, need_mb: 4550 }, // S67c sub-tier; worst 27 s
    ChunkTier { x_query: 2, x_center: 9, x_max: 10, need_mb: 3000 },  // S67c smallest; worst 15 s
];

/// Pick the chunking tier for this run. Non-DirectML devices always get tier 0 — the
/// former hardcoded constants, byte-identical behavior (CUDA's BFC arena reuses across
/// shapes, S31; CPU pays no shape ticket at all). On DirectML the first tier whose
/// `need_mb` fits the CURRENT system available commit wins; if even the smallest doesn't
/// fit, the smallest is used anyway — the engine's INFERENCE_LOW_MEMORY floor (and the
/// Auto-device CPU fallback in load_voice) guard the truly hopeless cases.
fn pick_chunk_tier(engine: &OnnxEngine, voice_session: &str) -> &'static ChunkTier {
    // A concurrently evicted session resolves to None (reload-on-miss rebuilds it inside
    // run_typed, AFTER this pick) — fall back to the global preference so an explicit
    // DirectML choice never silently loses the tiering (review round 2). Auto stays
    // conservative-false: the window is one eviction race wide, and guessing DML on a
    // CUDA box would needlessly shorten its chunks.
    let is_dml = engine
        .resolved_device(voice_session)
        .map(|d| d.contains("DirectML"))
        .unwrap_or_else(|| {
            matches!(engine.device(), super::engine::DeviceConfig::DirectMl { .. })
        });
    if !is_dml {
        return &CHUNK_TIERS[0];
    }
    let (_, avail) = super::engine::system_memory_mb();
    if avail == 0 {
        return &CHUNK_TIERS[0]; // measurement failed — keep the default tier
    }
    let tier = CHUNK_TIERS
        .iter()
        .find(|t| avail >= t.need_mb)
        .unwrap_or(CHUNK_TIERS.last().expect("tiers non-empty"));
    if tier.x_max != CHUNK_TIERS[0].x_max {
        tracing::info!(
            "RVC chunk tier lowered to x_max={} s (system available commit {} MB; DirectML \
             first-shape pool scales with chunk length — upstream low-memory tiering)",
            tier.x_max, avail
        );
    }
    tier
}

/// RVC retrieval index loaded from .npy [N, dim] — raw vectors + precomputed |v|²
/// (the old cosine-normalized copy is gone: faiss semantics are squared-L2, and dropping
/// the copy halves index RAM).
pub struct RvcIndex {
    pub knn: KnnIndex,
}

impl RvcIndex {
    pub fn load(path: &Path) -> Result<Self> {
        let raw: Array2<f32> = ndarray_npy::read_npy(path).map_err(|e| {
            UtaiError::Model(format!("INDEX_LOAD_FAILED: '{}': {}", path.display(), e))
        })?;
        tracing::info!(
            "Loaded RVC index: {} vectors x {} dim",
            raw.nrows(),
            raw.ncols()
        );
        Ok(Self {
            knn: KnnIndex::new(raw),
        })
    }
}

/// Session handles + model facts the pipeline needs (all resolved by the command layer).
pub struct RvcModel<'a> {
    pub engine: &'a OnnxEngine,
    pub voice_session: &'a str,
    pub contentvec_session: &'a str,
    pub rmvpe_session: &'a str,
    pub mel_filters: &'a Array2<f32>,
    pub index: Option<&'a RvcIndex>,
    pub sample_rate: u32,
    pub features_dim: usize,
    /// ①c (α′): `Some(n_spk)` iff the graph HAS a "spk_mix" input (genuine multi-speaker RVC
    /// export, n_spk = emb_g table width) — then a dense [1, n_spk] blend replaces scalar `sid`.
    /// `None` = single-speaker / pre-①c export → the `sid` i64 path (byte-identical).
    pub spk_mix: Option<usize>,
    /// inter_channels of the rnd input (sidecar "noise.rnd_input"[1]; 192 for v1/v2).
    pub noise_channels: usize,
    /// Minimum frame count the exported graph accepts (sidecar "min_frames", 12 for RVC).
    /// Chunks always carry ≥ 2 s of pad context (≥ ~200 frames), so this only trips on
    /// degenerate inputs — guarded with a clear error rather than padding.
    pub min_frames: usize,
}

pub fn run_pipeline(
    m: &RvcModel,
    audio: &AudioBuffer,
    options: &RvcOptions,
    // S60-2 音域扩展 (cover path) — see sovits::run_pipeline; None ⇒ byte-identical pipeline.
    range: Option<super::vocal_range::SpeakerRange>,
    progress: &dyn Fn(f32),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<SynthesisResult> {
    if audio.samples.is_empty() {
        return Err(UtaiError::Audio("AUDIO_EMPTY_INPUT".into()));
    }
    progress(0.03);
    if m.sample_rate % 100 != 0 {
        return Err(UtaiError::Model(format!(
            "RVC_SR_NOT_100FPS: sample_rate={}",
            m.sample_rate
        )));
    }

    // ── input: mono → 16 kHz → 48 Hz high-pass (filtfilt) ──
    let mut mono = crate::audio::resample::to_mono(audio);
    // S68c: scrub NaN/Inf BEFORE the FIR resample + filtfilt high-pass — both smear a single
    // poisoned sample across the buffer, and all-NaN features then crashed the KNN retrieval
    // (the 0.5.0 20% abort). Covers stems already poisoned on disk by older builds, too.
    let bad = crate::audio::sanitize_non_finite(&mut mono.samples);
    if bad > 0 {
        tracing::warn!("RVC input contained {} non-finite sample(s) (NaN/Inf) — zeroed before feature extraction", bad);
    }
    let wav16k = resample(&mono.samples, mono.sample_rate, SR as u32);
    let audio_f = highpass_48hz_16k(&wav16k)?;

    let tier = pick_chunk_tier(m.engine, m.voice_session);
    let t_pad = SR * X_PAD;
    let t_pad_tgt = m.sample_rate as usize * X_PAD;
    let t_pad2 = t_pad * 2;
    let t_query = SR * tier.x_query;
    let t_center = SR * tier.x_center;
    let t_max = SR * tier.x_max;

    // ── opt_ts: silence-seek cut points (original lines 319-333) ──
    // audio_pad = pad(audio, window//2, 'reflect'); audio_sum[j] = Σ_{i<160}|audio_pad[j+i]|
    // (len == len(audio)); every t_center, cut at the min-|sum| sample within ±t_query.
    let mut opt_ts: Vec<usize> = Vec::new();
    if audio_f.len() + WINDOW > t_max {
        let apad = reflect_pad_np(&audio_f, WINDOW / 2, WINDOW / 2);
        // rolling |x| sum via f64 prefix sums (original adds 160 shifted f64 arrays; only
        // the argmin of a near-silent region consumes this — summation order is immaterial)
        let mut prefix = vec![0.0f64; apad.len() + 1];
        for (i, &v) in apad.iter().enumerate() {
            prefix[i + 1] = prefix[i] + v.abs() as f64;
        }
        let audio_sum: Vec<f64> = (0..audio_f.len())
            .map(|j| prefix[j + WINDOW] - prefix[j])
            .collect();
        let mut t = t_center;
        while t < audio_f.len() {
            let lo = t - t_query;
            let hi = (t + t_query).min(audio_sum.len());
            let mut best = (f64::INFINITY, lo);
            for (j, &v) in audio_sum[lo..hi].iter().enumerate() {
                if v < best.0 {
                    best = (v, lo + j); // strict < keeps the FIRST minimum (np.where[0][0])
                }
            }
            opt_ts.push(best.1);
            t += t_center;
        }
    }

    // ── full-signal pad + f0 (RMVPE @100fps on the padded signal) ──
    let audio_pad = reflect_pad_np(&audio_f, t_pad, t_pad);
    let p_len = audio_pad.len() / WINDOW;

    // Stage logging (S67b): a community 16 GB machine died mid-pipeline with NOTHING in the
    // log between "Aux model cached" and process death — every stage transition below leaves
    // a breadcrumb so the next silent crash points at its stage. INFO = per-run milestones,
    // DEBUG = per-chunk (a song is ~1 line per ~40 s of input; the file layer records both).
    let t_run = std::time::Instant::now();
    tracing::info!(
        "RVC pipeline: {:.1}s @16k, {} chunk(s); f0 (RMVPE, chunked) starting ({})",
        audio_f.len() as f32 / SR as f32,
        opt_ts.len() + 1,
        super::engine::memory_stamp()
    );

    // S66: chunk-bounded (60 s windows + 2 s discarded overlap) — this whole-song pass was the
    // last unbounded GPU feed under gpu_extract (a 4-min song OOM'd 12 GB cards); ≤64 s songs
    // take the original single forward bit-for-bit (rmvpe_detect_chunked short-input path).
    let mut f0 = super::f0::rmvpe_detect_chunked(
        m.engine,
        m.rmvpe_session,
        m.mel_filters,
        &audio_pad,
        super::f0::RVC_RMVPE_THRESHOLD,
    )?;
    // f0 *= 2^(f0_up_key/12) — applied to the raw Hz track BEFORE coarse quantization
    // (unvoiced zeros stay zero under the multiply, like the original)
    let ratio = 2.0f32.powf(options.f0_shift / 12.0);
    f0.iter_mut().for_each(|v| *v *= ratio);
    if f0.len() < p_len {
        return Err(UtaiError::Inference(format!(
            "RVC_F0_FRAMES_SHORT: {} < p_len {}",
            f0.len(),
            p_len
        )));
    }
    let pitchf: Vec<f32> = f0[..p_len].to_vec();
    let pitch: Vec<i64> = pitchf.iter().map(|&v| f0_to_coarse(v)).collect();
    progress(0.2); // f0 (the one whole-signal RMVPE pass) done
    tracing::info!(
        "RVC pipeline: f0 done ({} frames); converting chunks ({})",
        p_len,
        super::engine::memory_stamp()
    );

    // ── chunk loop (original lines 371-441) ──
    // f0 (0.2) → chunks span [0.2, 0.95] → tail + post (0.95 → 1.0)
    let total_chunks = (opt_ts.len() + 1) as f32;
    let sid = options.speaker_id.unwrap_or(0) as i64;
    // ①c (α′): a multi-speaker RVC graph (m.spk_mix = Some(n_spk)) takes a dense [1, n_spk] blend
    // in place of scalar sid; built once and re-fed each chunk. None → the sid path (byte-identical).
    let spk_mix_dense: Option<Vec<f32>> = m
        .spk_mix
        .map(|n_spk| super::build_spk_mix_dense(&options.spk_mix, options.speaker_id, n_spk));
    let mut audio_opt: Vec<f32> = Vec::new();
    let mut s_ix = 0usize;
    let mut chunk_idx: u64 = 0;
    // S60-2/S60c 音域扩展: ONE tier decision over the WHOLE fed pitch track (v1 whole-render
    // semantics — a per-chunk shift gave each chunk its own formant coloration and the seams
    // between colorations were the dominant audible artifact, §user实测). Every chunk renders
    // at the same shift (coarse re-binned from the shifted Hz) and its UNtrimmed output is
    // TD-PSOLA'd back before append_trimmed, so the constant-ratio seams stay inside the
    // trimmed-away pads/silence. shift 0 ⇒ the exact original vc_chunk calls (byte-identical).
    let range_shift = range
        .map(|r| {
            // loudness weighting: RMS on the same 100 fps grid pitchf was detected on — quiet
            // phantom highs (reverb tails / harmony bleed) stop driving whole-song shifts
            let rms = super::vocal_range::frame_rms(&audio_pad, WINDOW, pitchf.len());
            super::vocal_range::piece_range_shift(&pitchf, Some(&rms), &r)
        })
        .unwrap_or(0);
    if range_shift != 0 {
        tracing::info!("range-extend(cover/rvc): whole signal rendered {range_shift:+} st into comfort");
    }
    let ranged_chunk = |pitch_sl: &[i64],
                        pitchf_sl: &[f32],
                        chunk: &[f32],
                        chunk_idx: u64|
     -> Result<Vec<f32>> {
        if range_shift == 0 {
            return vc_chunk(m, chunk, pitch_sl, pitchf_sl, sid, spk_mix_dense.as_deref(), options, chunk_idx);
        }
        let k = 2.0f32.powf(range_shift as f32 / 12.0);
        let pf: Vec<f32> = pitchf_sl.iter().map(|&v| v * k).collect();
        let pc: Vec<i64> = pf.iter().map(|&v| f0_to_coarse(v)).collect();
        let out = vc_chunk(m, chunk, &pc, &pf, sid, spk_mix_dense.as_deref(), options, chunk_idx)?;
        Ok(super::vocal_range::psola_inverse_hop(
            out,
            &pf,
            m.sample_rate as usize / 100,
            m.sample_rate,
            range_shift,
        ))
    };
    for &ot in &opt_ts {
        if cancel() {
            return Err(UtaiError::Inference("CANCELLED".into()));
        }
        let t = ot / WINDOW * WINDOW;
        // Clamp to buffer length: Python's `audio_pad[s : t+t_pad2+window]` TRUNCATES, but Rust
        // slicing PANICS. When the last silence-seek cut lands in the final partial <WINDOW window
        // (song ~3-6s past a t_center multiple, ending on a quiet passage, L not a WINDOW multiple),
        // t+t_pad2+WINDOW can exceed audio_pad.len(). vc_chunk re-derives p_len from the (shorter)
        // chunk, so a truncated tail chunk is handled correctly — matching the original.
        let chunk = &audio_pad[s_ix..(t + t_pad2 + WINDOW).min(audio_pad.len())];
        let pl = s_ix / WINDOW;
        let ph = (t + t_pad2) / WINDOW;
        let t_chunk = std::time::Instant::now();
        let out = ranged_chunk(&pitch[pl..ph], &pitchf[pl..ph], chunk, chunk_idx)?;
        append_trimmed(&mut audio_opt, &out, t_pad_tgt)?;
        s_ix = t;
        chunk_idx += 1;
        tracing::debug!(
            "RVC chunk {}/{} done ({:.1}s in, {:.0} ms, {})",
            chunk_idx,
            opt_ts.len() + 1,
            chunk.len() as f32 / SR as f32,
            t_chunk.elapsed().as_secs_f64() * 1000.0,
            super::engine::memory_stamp()
        );
        progress(0.2 + 0.75 * (chunk_idx as f32 / total_chunks));
    }
    // final chunk: audio_pad[t:] with the remaining pitch tail (t=None → whole signal)
    if cancel() {
        return Err(UtaiError::Inference("CANCELLED".into()));
    }
    let chunk = &audio_pad[s_ix..];
    let t_chunk = std::time::Instant::now();
    let out = ranged_chunk(&pitch[s_ix / WINDOW..], &pitchf[s_ix / WINDOW..], chunk, chunk_idx)?;
    append_trimmed(&mut audio_opt, &out, t_pad_tgt)?;
    tracing::debug!(
        "RVC chunk {}/{} done (final, {:.1}s in, {:.0} ms, {})",
        chunk_idx + 1,
        opt_ts.len() + 1,
        chunk.len() as f32 / SR as f32,
        t_chunk.elapsed().as_secs_f64() * 1000.0,
        super::engine::memory_stamp()
    );

    // ── rms mix (original: change_rms(audio, 16000, audio_opt, tgt_sr, rate) if rate != 1) ──
    if options.rms_mix_rate != 1.0 {
        change_rms(&audio_f, SR as u32, &mut audio_opt, m.sample_rate, options.rms_mix_rate);
    }

    // ── optional output resample (original guard: tgt_sr != resample_sr >= 16000) ──
    let mut final_sr = m.sample_rate;
    if options.resample_sr >= 16000 && options.resample_sr != m.sample_rate {
        audio_opt = resample(&audio_opt, m.sample_rate, options.resample_sr);
        final_sr = options.resample_sr;
    }
    // ② node-level 共振腔/formant (scalar): warp the WHOLE final signal (mono at final_sr) after the optional
    // resample. ratio = 2^(semi/12); ≈1 passes through verbatim → formant=0 is a near-lossless no-op.
    if options.formant.abs() > 1e-6 {
        audio_opt = utai_dsp::formant_warp(&audio_opt, |_| 2.0_f32.powf(options.formant / 12.0));
    }
    // NO int16 quantization (original's audio_max/max_int16 normalize skipped — we stay f32).
    tracing::info!(
        "RVC pipeline done in {:.1}s ({:.1}s audio out @{}Hz; {})",
        t_run.elapsed().as_secs_f32(),
        audio_opt.len() as f32 / final_sr.max(1) as f32,
        final_sr,
        super::engine::memory_stamp()
    );
    progress(1.0);

    Ok(SynthesisResult {
        audio: audio_opt,
        sample_rate: final_sr,
    })
}

fn append_trimmed(dst: &mut Vec<f32>, out: &[f32], t_pad_tgt: usize) -> Result<()> {
    if out.len() <= 2 * t_pad_tgt {
        return Err(UtaiError::Inference("RVC_CHUNK_TOO_SHORT".into()));
    }
    dst.extend_from_slice(&out[t_pad_tgt..out.len() - t_pad_tgt]);
    Ok(())
}

/// Pipeline.vc port: one padded chunk → model audio (UNtrimmed; caller strips t_pad_tgt).
fn vc_chunk(
    m: &RvcModel,
    chunk: &[f32],
    pitch: &[i64],
    pitchf: &[f32],
    sid: i64,
    // ①c: Some = dense spk_mix [n_spk] blend fed in place of scalar sid (multi-speaker export)
    spk_mix: Option<&[f32]>,
    options: &RvcOptions,
    chunk_idx: u64,
) -> Result<Vec<f32>> {
    // ContentVec @ 50 fps
    let feats = contentvec_extract(m.engine, m.contentvec_session, chunk, m.features_dim)?;
    vc_decode(m, feats, pitch, pitchf, sid, spk_mix, options, chunk_idx, chunk.len() / WINDOW)
}

/// Decode one chunk's already-extracted ContentVec (50 fps) through the RVC net_g → UNtrimmed wav.
/// Verbatim tail of `vc_chunk` (feats0/protect basis → index blend → 2x upsample → protect blend →
/// rnd noise → net_g), pulled out so the ② score render (`render_score_rvc`) drives the SAME
/// retrieval/protect path. `windows_bound` = the caller's frame cap (cover: chunk.len()/WINDOW;
/// score: usize::MAX = feats/pitch bound only). Byte-identical to the inlined version — proven at the
/// source (scratchpad/verbatim_check.py) since the net_g graph is stochastic (RandomNormalLike/Uniform).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vc_decode(
    m: &RvcModel,
    mut feats: Array2<f32>,
    pitch: &[i64],
    pitchf: &[f32],
    sid: i64,
    spk_mix: Option<&[f32]>,
    options: &RvcOptions,
    chunk_idx: u64,
    windows_bound: usize,
) -> Result<Vec<f32>> {
    // feats0 clone happens BEFORE retrieval (original line 221-222)
    let feats0 = if options.protect < 0.5 {
        Some(feats.clone())
    } else {
        None
    };
    if options.index_ratio > 0.0 {
        if let Some(index) = m.index {
            // l2_normalize = cosine NEIGHBOR METRIC only (S36 fix — normalizing the
            // blended model input itself muffled the audio; see knn_blend docs).
            feats = knn_blend(&feats, &index.knn, options.index_ratio, options.l2_normalize);
        }
    }
    // 2x nearest upsample 50 → 100 fps (both copies, original lines 247-251)
    let feats = upsample_2x_nearest(&feats);
    let feats0 = feats0.map(|f| upsample_2x_nearest(&f));

    // p_len = min(windows_bound, feats_T, pitch_len). `windows_bound` is the caller's frame cap:
    // the cover path passes chunk_len//window (feats_T is < it in practice); the ② score path passes
    // usize::MAX (no source-audio window — bounded by the upsampled cv / pitch length only).
    let mut p_len = windows_bound;
    if feats.nrows() < p_len {
        p_len = feats.nrows();
    }
    let p_len = p_len.min(pitch.len());
    if p_len < m.min_frames {
        return Err(UtaiError::Inference(format!(
            "RVC_MIN_FRAMES: {} < {}",
            p_len, m.min_frames
        )));
    }
    let pitch = &pitch[..p_len];
    let pitchf = &pitchf[..p_len];
    let mut feats = feats.slice(s![..p_len, ..]).to_owned();

    // protect blend: pitchff = (pitchf < 1 ? protect : 1); feats = feats·w + feats0·(1-w)
    // (original sets 1 where >0 THEN protect where <1 — net effect: <1 → protect)
    if let Some(f0s) = feats0 {
        let f0s = f0s.slice(s![..p_len, ..]);
        for (i, mut row) in feats.rows_mut().into_iter().enumerate() {
            let w = if pitchf[i] < 1.0 { options.protect } else { 1.0 };
            for (j, v) in row.iter_mut().enumerate() {
                *v = *v * w + f0s[[i, j]] * (1.0 - w);
            }
        }
    }

    // rnd: N(0,1)·noise_scale, [1, inter_channels, T]. Seeded; the chunk index is mixed in
    // so chunks get independent (but reproducible) noise like the original's fresh randn.
    let rnd = chunk_noise(m.noise_channels, p_len, options.seed, chunk_idx, options.noise_scale);

    let t = p_len as i64;
    let phone_data: Vec<f32> = feats.iter().copied().collect();
    let mut inputs = vec![
        (
            "phone",
            InputTensor::F32 {
                data: phone_data,
                shape: vec![1, t, m.features_dim as i64],
            },
        ),
        (
            "phone_lengths",
            InputTensor::I64 {
                data: vec![t],
                shape: vec![1],
            },
        ),
        (
            "pitch",
            InputTensor::I64 {
                data: pitch.to_vec(),
                shape: vec![1, t],
            },
        ),
        (
            "pitchf",
            InputTensor::F32 {
                data: pitchf.to_vec(),
                shape: vec![1, t],
            },
        ),
    ];
    // ①c (α′): dense spk_mix [1, n_spk] blend (multi-speaker export) OR scalar sid i64 (single /
    // pre-①c: byte-identical). The graph renamed the input in the export, so the name must match.
    if let Some(mix) = spk_mix {
        inputs.push((
            "spk_mix",
            InputTensor::F32 {
                data: mix.to_vec(),
                shape: vec![1, mix.len() as i64],
            },
        ));
    } else {
        inputs.push((
            "sid",
            InputTensor::I64 {
                data: vec![sid],
                shape: vec![1],
            },
        ));
    }
    inputs.push((
        "rnd",
        InputTensor::F32 {
            data: rnd,
            shape: vec![1, m.noise_channels as i64, t],
        },
    ));

    let outputs = m.engine.run(m.voice_session, inputs)?;
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| UtaiError::Inference("RVC_NO_OUTPUT".into()))
}

/// Deterministic per-chunk RNG: user seed splitmixed with the chunk index.
fn chunk_rng(seed: u64, chunk_idx: u64) -> StdRng {
    StdRng::seed_from_u64(seed ^ chunk_idx.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// The net_g explicit `rnd` input: N(0,1)·scale, `channels·t` values row-major (ONNX `[1, channels, T]`),
/// drawn from the per-chunk chunk_rng. Extracted so the cover path (vc_chunk) and the S48 score path
/// (score2svc) build the SAME noise byte-for-byte — the export moved net_g's internal randn out to this
/// input, so reproducibility hinges on an identical draw (seed + chunk_idx + channel×frame count + scale).
pub(crate) fn chunk_noise(
    channels: usize,
    t: usize,
    seed: u64,
    chunk_idx: u64,
    scale: f32,
) -> Vec<f32> {
    let mut rng = chunk_rng(seed, chunk_idx);
    (0..channels * t)
        .map(|_| {
            let n: f32 = rng.sample(StandardNormal);
            n * scale
        })
        .collect()
}

/// RVC f0 → coarse 1..255 (pipeline.py get_f0 mel-scale quantization).
/// Formula verified against the original; kept from the previous implementation
/// (tests below are the original verification set).
pub fn f0_to_coarse(f0: f32) -> i64 {
    let f0_mel = 1127.0_f32 * (1.0 + f0 / 700.0).ln();
    if f0_mel <= 0.0 {
        return 1;
    }
    // f0_mel_min = 1127*ln(1+50/700) ≈ 77.74, f0_mel_max = 1127*ln(1+1100/700) ≈ 1064.42
    let normalized = (f0_mel - 77.74) / (1064.42 - 77.74) * 254.0 + 1.0;
    (normalized.round() as i64).clamp(1, 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f0_to_coarse_matches_original_rvc() {
        // Exact values verified against Python original:
        // f0_mel_min = 1127*ln(1+50/700), f0_mel_max = 1127*ln(1+1100/700)
        // coarse = round((mel - mel_min) / (mel_max - mel_min) * 254 + 1), clamp [1, 255]
        assert_eq!(f0_to_coarse(0.0), 1);
        assert_eq!(f0_to_coarse(50.0), 1);
        assert_eq!(f0_to_coarse(100.0), 20);
        assert_eq!(f0_to_coarse(220.0), 60);
        assert_eq!(f0_to_coarse(440.0), 122);
        assert_eq!(f0_to_coarse(880.0), 217);
        assert_eq!(f0_to_coarse(1100.0), 255);
        assert_eq!(f0_to_coarse(2000.0), 255);

        // Monotonicity
        assert!(f0_to_coarse(220.0) < f0_to_coarse(440.0));
        assert!(f0_to_coarse(440.0) < f0_to_coarse(880.0));
    }

    #[test]
    fn chunk_rng_is_deterministic_and_chunk_distinct() {
        let a: Vec<f32> = {
            let mut r = chunk_rng(42, 0);
            (0..8).map(|_| r.sample(StandardNormal)).collect()
        };
        let a2: Vec<f32> = {
            let mut r = chunk_rng(42, 0);
            (0..8).map(|_| r.sample(StandardNormal)).collect()
        };
        let b: Vec<f32> = {
            let mut r = chunk_rng(42, 1);
            (0..8).map(|_| r.sample(StandardNormal)).collect()
        };
        assert_eq!(a, a2, "same seed+chunk must reproduce");
        assert_ne!(a, b, "different chunks must differ");
    }
}
