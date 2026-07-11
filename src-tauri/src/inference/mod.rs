pub mod diffusion;
pub mod engine;
pub mod f0;
pub mod features;
pub mod g2p;
mod g2p_tables;
#[cfg(test)]
mod g2p_golden_ref;
pub mod mel;
pub mod midi_extract;
pub mod nsf_hifigan;
pub mod rvc;
pub mod score2cv;
mod score2cv_tables;
#[cfg(test)]
mod score2cv_cv_ref;
pub mod score2svc;
#[cfg(test)]
mod score2svc_ref;
pub mod sovits;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::Result;

/// Wire-contract options for run_rvc — mirrored 1:1 by src\lib\workflow\voiceDefaults.ts
/// (THE frontend source of truth). Struct-level serde default: any absent key falls back
/// to the Default impl below.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RvcOptions {
    pub f0_shift: f32,
    pub speaker_id: Option<u32>,
    /// ①c speaker blend — empty = use speaker_id (single-speaker / pre-①c: byte-identical).
    /// Consumed by α′ (multi-speaker RVC export); see SovitsOptions::spk_mix.
    pub spk_mix: Vec<SpkMixEntry>,
    pub index_ratio: f32,
    pub protect: f32,
    pub noise_scale: f32,
    pub rms_mix_rate: f32,
    pub l2_normalize: bool,
    pub resample_sr: u32,
    pub seed: u64,
    /// Run the aux extractors (ContentVec + RMVPE) on the global GPU device instead of the
    /// S35 forced-CPU default. Faster, costs VRAM (see ensure_aux_loaded_on rationale).
    pub gpu_extract: bool,
    /// ② 共振腔/formant — node-level SCALAR in semitones (post-decode formant_warp ratio = 2^(semi/12)).
    /// 0 = no shift (a ratio-1 pass-through, near-lossless). Higher = brighter/younger. Neutralized in the
    /// self-sing render (render_vocal_segment) — the vocal editor owns its own formant lane.
    pub formant: f32,
}

impl Default for RvcOptions {
    fn default() -> Self {
        Self {
            f0_shift: 0.0,
            speaker_id: None,
            spk_mix: Vec::new(),
            index_ratio: 0.75,
            protect: 0.33,
            noise_scale: 0.66666,
            rms_mix_rate: 0.25,
            l2_normalize: false,
            resample_sr: 0,
            seed: 0,
            gpu_extract: false,
            formant: 0.0,
        }
    }
}

/// ①c speaker-blend entry (voiceDefaults.ts `SpkMixEntry` mirror): emb_g row `id` weighted by
/// `weight` (≥0). Consumed ONLY by a genuine multi-speaker SoVITS export (the ONNX graph carries
/// a "spk_mix" input); normalized to sum 1 and expanded to a dense [1, n_spk] f32 vector.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpkMixEntry {
    pub id: u32,
    pub weight: f32,
}

/// ①c: build the dense `spk_mix` [n_spk] f32 blend a multi-speaker graph consumes in place of the
/// scalar `sid` (SHARED by the RVC + SoVITS pipelines). Weights are clamped ≥0 and normalized to
/// sum 1; out-of-range ids are dropped. An empty / all-zero / all-out-of-range stack degrades to a
/// ONE-HOT on the selected speaker (fallback 0) — a one-hot row through `spk_mix @ emb_g.weight` is
/// bit-identical to `emb_g(id)`, so a multi-speaker model with no blend behaves exactly like
/// picking that single speaker.
pub(crate) fn build_spk_mix_dense(
    spk_mix: &[SpkMixEntry],
    speaker_id: Option<u32>,
    n_spk: usize,
) -> Vec<f32> {
    let mut dense = vec![0.0f32; n_spk];
    for e in spk_mix {
        let id = e.id as usize;
        if id < n_spk && e.weight > 0.0 {
            dense[id] += e.weight;
        }
    }
    let sum: f32 = dense.iter().sum();
    if sum > 0.0 {
        for w in &mut dense {
            *w /= sum;
        }
    } else {
        let spk = (speaker_id.unwrap_or(0) as usize).min(n_spk.saturating_sub(1));
        dense[spk] = 1.0;
    }
    dense
}

/// ①c: the dominant (max-weight) speaker id of a blend, else `speaker_id` (fallback 0). Used to
/// pick the per-speaker retrieval/cluster asset when a blend is active. Ties → first max in stack
/// order.
pub(crate) fn dominant_speaker(spk_mix: &[SpkMixEntry], speaker_id: Option<u32>) -> u32 {
    spk_mix
        .iter()
        .filter(|e| e.weight > 0.0)
        .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
        .map(|e| e.id)
        .unwrap_or_else(|| speaker_id.unwrap_or(0))
}

#[cfg(test)]
mod spk_mix_tests {
    // S56: the ①c multi-speaker blend math had NO automated coverage (verified one-off in S46/S47);
    // these lock its invariants — the tests/{voice_pipeline,audition_render}.rs E2E tools use
    // single-speaker models (spk_mix: None), so this is the only persistent gate on the Some path.
    use super::{build_spk_mix_dense, dominant_speaker, SpkMixEntry};

    fn e(id: u32, weight: f32) -> SpkMixEntry {
        SpkMixEntry { id, weight }
    }

    #[test]
    fn one_hot_equals_gather() {
        // A single full-weight entry → a one-hot row, which through `spk_mix @ emb_g.weight` is
        // bit-identical to `emb_g(id)` (the doc-comment's whole correctness claim).
        assert_eq!(build_spk_mix_dense(&[e(2, 1.0)], None, 5), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        // any positive weight normalizes to the same one-hot.
        assert_eq!(build_spk_mix_dense(&[e(0, 0.3)], None, 3), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn normalizes_to_sum_one() {
        let d = build_spk_mix_dense(&[e(0, 1.0), e(1, 3.0)], None, 4);
        assert_eq!(d, vec![0.25, 0.75, 0.0, 0.0]);
        assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn duplicate_ids_accumulate() {
        // Two entries on the same id add before normalization (not last-wins).
        let d = build_spk_mix_dense(&[e(2, 1.0), e(2, 1.0), e(0, 2.0)], None, 3);
        assert_eq!(d, vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn out_of_range_and_nonpositive_dropped_then_fallback() {
        // id ≥ n_spk and weight ≤ 0 are dropped; an all-dropped stack degrades to a one-hot on
        // speaker_id (fallback 0), which the min() clamps into range.
        assert_eq!(build_spk_mix_dense(&[e(99, 1.0), e(0, -1.0), e(1, 0.0)], Some(1), 3), vec![0.0, 1.0, 0.0]);
        assert_eq!(build_spk_mix_dense(&[], None, 3), vec![1.0, 0.0, 0.0]); // empty → speaker 0
        assert_eq!(build_spk_mix_dense(&[], Some(99), 3), vec![0.0, 0.0, 1.0]); // clamp to n_spk-1
    }

    #[test]
    fn dominant_is_max_weight_then_fallback() {
        assert_eq!(dominant_speaker(&[e(0, 0.2), e(5, 0.7), e(3, 0.1)], None), 5);
        assert_eq!(dominant_speaker(&[], Some(4)), 4); // empty → speaker_id
        assert_eq!(dominant_speaker(&[e(2, 0.0)], Some(1)), 1); // all zero-weight → fallback
    }
}

/// Wire-contract options for run_sovits (voiceDefaults.ts mirror).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SovitsOptions {
    pub f0_shift: f32,
    pub speaker_id: Option<u32>,
    /// ①c speaker blend — empty = use speaker_id (single-speaker / pre-①c models: byte-identical).
    pub spk_mix: Vec<SpkMixEntry>,
    pub noise_scale: f32,
    pub cluster_ratio: f32,
    pub loudness_envelope: f32,
    pub seed: u64,
    // ── S36 quality path (original so-vits-svc CLI semantics; validation in the command
    //    layer — see commands/inference.rs resolve_sovits_quality) ──
    /// VITS output → mel → q_sample(k_step-1) → sampler → NSF-HiFiGAN. Needs `.diffusion/`.
    pub shallow_diffusion: bool,
    /// Diffusion depth (original -ks default 100). Must be ≤ the diffusion model's
    /// k_step_max. IGNORED in only_diffusion (original: t = k_step_max unconditionally).
    pub k_step: u32,
    /// "naive" | "ddim" | "pndm" | "dpm-solver" | "dpm-solver++" | "unipc".
    pub diffusion_method: String,
    /// Solver steps ≈ k_step / speedup; ≤1 → the plain DDPM loop (original semantics).
    pub diffusion_speedup: u32,
    /// Skip VITS entirely; full-depth from-noise generation (k_step_max == timesteps only).
    pub only_diffusion: bool,
    /// Re-extract ContentVec from the VITS output before diffusing (shallow only).
    pub second_encoding: bool,
    /// NSF-HiFiGAN enhancer — mutually exclusive with any diffusion mode (original
    /// force-disable, infer_tool.py:183-184).
    pub nsf_enhance: bool,
    /// Enhancer high-range adaptation in semitones (original enhancer_adaptive_key).
    pub enhancer_adaptive_key: i32,
    /// Automatic f0 prediction via `<stem>.f0.onnx` (predict_f0=True semantics).
    pub auto_f0: bool,
    /// See RvcOptions::gpu_extract.
    pub gpu_extract: bool,
    /// S40 vocoder resource: registry NAME of an installed NSF-HiFiGAN vocoder
    /// (models/nsf_hifigan/) to use for shallow diffusion + the enhancer.
    /// None / "" = the aux default vocoder (S36 behavior, byte-identical path).
    /// An unknown name is a LOUD error, never a silent fallback.
    pub vocoder_name: Option<String>,
    /// ② 共振腔/formant — node-level SCALAR in semitones (post-decode formant_warp ratio = 2^(semi/12));
    /// 0 = no shift. Neutralized in the self-sing render (render_vocal_segment). See RvcOptions::formant.
    pub formant: f32,
    /// TEST-ONLY (E2E gates): zero every diffusion noise draw (q_sample / initial randn /
    /// naive per-step noise) to mirror the python reference's ZeroNoise monkeypatch.
    /// Deliberately NOT in voiceDefaults.ts — never reachable from the UI.
    pub debug_zero_noise: bool,
}

impl Default for SovitsOptions {
    fn default() -> Self {
        Self {
            f0_shift: 0.0,
            speaker_id: None,
            spk_mix: Vec::new(),
            noise_scale: 0.4,
            cluster_ratio: 0.0,
            loudness_envelope: 1.0,
            seed: 0,
            shallow_diffusion: false,
            k_step: 100,
            diffusion_method: "dpm-solver++".into(),
            diffusion_speedup: 10,
            only_diffusion: false,
            second_encoding: false,
            nsf_enhance: false,
            enhancer_adaptive_key: 0,
            auto_f0: false,
            gpu_extract: false,
            vocoder_name: None,
            formant: 0.0,
            debug_zero_noise: false,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SynthesisResult {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Clone, Debug)]
pub enum VoiceBackendType {
    Rvc,
    SoVits,
}

struct LoadedVoice {
    _backend_type: VoiceBackendType,
    /// Also the key for evicting companion sessions (`<stem>.f0.onnx` / `<stem>.diffusion/`)
    /// on unload — see unload_voice.
    model_path: PathBuf,
    session_id: String,
    sample_rate: u32,
    index: Option<Arc<rvc::RvcIndex>>,
}

/// Cheap cloneable view of a loaded voice for the pipelines.
pub struct VoiceHandle {
    pub session_id: String,
    pub sample_rate: u32,
    pub index: Option<Arc<rvc::RvcIndex>>,
}

pub struct InferenceManager {
    pub engine: engine::OnnxEngine,
    /// Voice sessions keyed by VOICE NAME — model delete/reimport calls unload_voice(name).
    loaded_voices: RwLock<HashMap<String, LoadedVoice>>,
    /// Aux model sessions (ContentVec variants, RMVPE) keyed by (path, on_gpu) — the same
    /// file can hold a CPU and a GPU session simultaneously (the engine key embeds the
    /// device too, so they never collide there either).
    aux_sessions: RwLock<HashMap<(PathBuf, bool), String>>,
    /// Small .npy cache (RMVPE mel filters, so-vits cluster assets) keyed by path.
    npy_cache: RwLock<HashMap<PathBuf, Arc<ndarray::Array2<f32>>>>,
    /// Voice-run cancel, epoch-based. `cancel_voice` cancels every voice run STARTED at or
    /// before the moment of the click (cancel_epoch = current epoch); runs started later
    /// (epoch > cancel_epoch) are unaffected. This kills two races a single reset-on-start
    /// bool had: (a) a cancel aimed at run A being swallowed by run B's start clearing the
    /// flag, and (b) a cancel during A's session-loading phase getting lost. Scope is still
    /// app-global like cancel_separation — cancelling one pane can abort another segment's
    /// CONCURRENT voice run (rare: needs two simultaneous voice renders); a per-run handle
    /// registry is deliberately out of scope, matching the separation cancel's semantics.
    voice_epoch: std::sync::atomic::AtomicU64,
    voice_cancel_epoch: std::sync::atomic::AtomicU64,
}

impl InferenceManager {
    pub fn new() -> Self {
        Self {
            engine: engine::OnnxEngine::new(),
            loaded_voices: RwLock::new(HashMap::new()),
            aux_sessions: RwLock::new(HashMap::new()),
            npy_cache: RwLock::new(HashMap::new()),
            voice_epoch: std::sync::atomic::AtomicU64::new(0),
            voice_cancel_epoch: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Arm a new voice run: returns this run's epoch (pass it to voice_cancelled). Call at
    /// the very START of the command — before session loading — so a cancel during the
    /// multi-second load phase is honored at the first pipeline poll.
    pub fn begin_voice_run(&self) -> u64 {
        self.voice_epoch
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1
    }

    /// Cancel every voice run started at or before now (later runs unaffected).
    pub fn cancel_voice(&self) {
        let now = self.voice_epoch.load(std::sync::atomic::Ordering::SeqCst);
        self.voice_cancel_epoch
            .store(now, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn voice_cancelled(&self, run_epoch: u64) -> bool {
        self.voice_cancel_epoch
            .load(std::sync::atomic::Ordering::SeqCst)
            >= run_epoch
    }

    /// Load (or reuse) an aux ONNX session (ContentVec / RMVPE), cached by (path, device).
    ///
    /// Default (`on_gpu=false`): CPU — aux extractors are one-shot passes whose fp32
    /// activations were the dominant VRAM consumer (a 2-min song peaked ~9 GB on GPU,
    /// S35), and CPU is numerically MORE faithful than the TF32 GPU path (the E2E gate
    /// ran on CPU). `on_gpu=true` (the per-node GPU特征提取 toggle) follows the global
    /// device instead — viable since S35's 30 s chunking bounded the activations, but
    /// still a deliberate user opt-in. mem_pattern stays OFF either way (dynamic T).
    pub fn ensure_aux_loaded_on(&self, path: &Path, on_gpu: bool) -> Result<String> {
        let key = (path.to_path_buf(), on_gpu);
        {
            let cached = self.aux_sessions.read();
            if let Some(sid) = cached.get(&key) {
                if self.engine.is_loaded(sid) {
                    return Ok(sid.clone());
                }
            }
        }
        let sid = if on_gpu {
            self.engine.load_model_with(&path.to_path_buf(), false)?
        } else {
            self.engine
                .load_model_on(&path.to_path_buf(), false, engine::DeviceConfig::Cpu)?
        };
        self.aux_sessions.write().insert(key, sid.clone());
        tracing::info!(
            "Aux model cached: {} ({})",
            path.display(),
            if on_gpu { "global device" } else { "CPU" }
        );
        Ok(sid)
    }

    /// Back-compat wrapper: CPU-forced aux session (the S35 default).
    pub fn ensure_aux_loaded(&self, path: &Path) -> Result<String> {
        self.ensure_aux_loaded_on(path, false)
    }

    /// Load a .npy as Array2<f32>, cached by path (mel filters / cluster assets).
    pub fn load_npy(&self, path: &Path) -> Result<Arc<ndarray::Array2<f32>>> {
        if let Some(arr) = self.npy_cache.read().get(path) {
            return Ok(arr.clone());
        }
        let arr: ndarray::Array2<f32> = ndarray_npy::read_npy(path).map_err(|e| {
            crate::UtaiError::Model(format!("加载 npy 失败 '{}': {}", path.display(), e))
        })?;
        let arr = Arc::new(arr);
        self.npy_cache
            .write()
            .insert(path.to_path_buf(), arr.clone());
        Ok(arr)
    }

    pub fn is_voice_loaded(&self, name: &str) -> bool {
        let voices = self.loaded_voices.read();
        if let Some(voice) = voices.get(name) {
            self.engine.is_loaded(&voice.session_id)
        } else {
            false
        }
    }

    pub fn load_voice(
        &self,
        name: &str,
        model_path: &PathBuf,
        backend_type: VoiceBackendType,
        sample_rate: u32,
        index_path: Option<&PathBuf>,
    ) -> Result<()> {
        if self.is_voice_loaded(name) {
            return Ok(());
        }
        self.unload_voice(name);
        // mem_pattern OFF: voice models run per-chunk with varying T (RVC) / whole-segment T
        // (SoVITS) — dynamic shapes where ORT's memory pattern over-reserves VRAM.
        let session_id = self.engine.load_model_with(model_path, false)?;

        let index = match (&backend_type, index_path) {
            (VoiceBackendType::Rvc, Some(path)) if path.exists() => {
                match rvc::RvcIndex::load(path) {
                    Ok(idx) => Some(Arc::new(idx)),
                    Err(e) => {
                        tracing::warn!("Failed to load index, continuing without: {}", e);
                        None
                    }
                }
            }
            _ => None,
        };

        let voice = LoadedVoice {
            _backend_type: backend_type,
            model_path: model_path.clone(),
            session_id,
            sample_rate,
            index,
        };
        self.loaded_voices.write().insert(name.to_string(), voice);
        Ok(())
    }

    pub fn voice_handle(&self, name: &str) -> Result<VoiceHandle> {
        let voices = self.loaded_voices.read();
        let voice = voices.get(name).ok_or_else(|| {
            crate::UtaiError::Inference(format!("模型 '{}' 尚未加载", name))
        })?;
        Ok(VoiceHandle {
            session_id: voice.session_id.clone(),
            sample_rate: voice.sample_rate,
            index: voice.index.clone(),
        })
    }

    /// S40 vocoder resource hygiene (设计红队 A18): a same-name re-import
    /// REPLACES files on disk while the engine caches sessions BY PATH and
    /// npy assets by path — evict both BEFORE the file swap so the next run
    /// cannot serve the old graph/filterbank (a live ORT session would also
    /// hold a Windows file lock and fail the copy).
    pub fn unload_model_file(&self, onnx_path: &Path) {
        self.engine.unload_paths_with_prefix(onnx_path);
        // blunt like unload_voice below: npy reloads are cheap
        self.npy_cache.write().clear();
    }

    pub fn unload_voice(&self, name: &str) {
        let mut voices = self.loaded_voices.write();
        if let Some(voice) = voices.remove(name) {
            self.engine.unload_model(&voice.session_id);
            // The engine caches companion sessions BY PATH — on a same-name re-import the
            // replaced `<stem>.f0.onnx` / `<stem>.diffusion/*.onnx` files keep their paths,
            // so without this the next run would silently serve the OLD graphs.
            let p = &voice.model_path;
            if let (Some(dir), Some(stem)) = (p.parent(), p.file_stem()) {
                let stem = stem.to_string_lossy();
                self.engine
                    .unload_paths_with_prefix(&dir.join(format!("{}.f0.onnx", stem)));
                self.engine
                    .unload_paths_with_prefix(&dir.join(format!("{}.diffusion", stem)));
            }
        }
        // Model files may be replaced on reimport — drop cached npy assets (cheap reloads)
        // so a stale cluster index / retrieval asset can't outlive its file.
        self.npy_cache.write().clear();
    }
}
