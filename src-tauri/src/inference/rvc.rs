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

/// S165 -- `UTAI_COVER_FLATTEN_MS=<ms>`: lift the quiet stretches of the input envelope
/// with a running-RMS window before the model sees it. **Off by default — it was rendered,
/// measured, and it makes things worse.**
///
/// It shipped on for a few hours on the strength of a comparison that turned out to be
/// worthless: the baseline arm happened to render at a 19 s chunk tier and the treated arm
/// at 32 s, and *that* was the entire "123 bad segments down to 27". See the tier lock in
/// `tier_for_this_song` for why the tier moved on its own.
///
/// Redone with all four arms pinned to the same tier, twice each:
///     off  5.88%  5.85%      on  6.33%  6.34%      (bad-frame rate, whole song)
/// Two renders of the same config land within 0.01-0.03 pp of each other, so the +0.45 pp
/// the lift costs is roughly fifteen times the noise floor, and it reproduces. The user had
/// said it plainly after listening — "basically no improvement over the octave-fix build" —
/// while every ruler I had was still reporting a large win.
///
/// Kept reachable rather than deleted so the arm stays falsifiable, and so the next person
/// who has this idea finds the measurement instead of re-running it.
fn cover_flatten_ms() -> Option<f32> {
    parse_flatten_ms(std::env::var("UTAI_COVER_FLATTEN_MS").ok().as_deref())
}

/// The parsing half of [`cover_flatten_ms`], split out so it can be pinned by a test
/// without touching process-wide env (which would race the rest of the suite).
fn parse_flatten_ms(raw: Option<&str>) -> Option<f32> {
    /// ⛔ 0 = off. Measured worse than off (see [`cover_flatten_ms`]); do not flip back
    /// without a same-tier A/B that beats the 0.03 pp noise floor.
    const DEFAULT_MS: f32 = 0.0;
    match raw {
        Some(v) => {
            let t = v.trim();
            if t == "0" {
                return None;
            }
            t.parse::<f32>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 10.0 && *v <= 1000.0)
        }
        None => (DEFAULT_MS > 0.0).then_some(DEFAULT_MS),
    }
}

/// S165 -- read one `f32` knob with a default. Used by the flattener's three shape
/// parameters so a single build can sweep them (they are exploratory, not shipped).
fn cover_flatten_knob(name: &str, default: f32, lo: f32, hi: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= lo && *v <= hi)
        .unwrap_or(default)
}

/// S165 -- `UTAI_COVER_FLATTEN_BOTH=1` restores the two-sided normalisation that was
/// measured and REJECTED (see the call site). Default is one-sided: lift only.
fn cover_flatten_two_sided() -> bool {
    std::env::var("UTAI_COVER_FLATTEN_BOTH").map(|v| v.trim() == "1").unwrap_or(false)
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
/// S165 §103 —— 每个 session 记住它选定的 chunk tier。见 [`pick_chunk_tier`] 里的理由。
/// ⚠ 跨 session 的状态,所以是进程级的;`UTAI_RVC_TIER_MEMO=0` 可关(安全阀)。
static TIER_MEMO: std::sync::Mutex<Option<(String, &'static ChunkTier)>> =
    std::sync::Mutex::new(None);

fn tier_memo_enabled() -> bool {
    !std::env::var("UTAI_RVC_TIER_MEMO").map(|v| v.trim() == "0").unwrap_or(false)
}

fn tier_memo_get(session_id: &str) -> Option<&'static ChunkTier> {
    TIER_MEMO
        .lock()
        .ok()?
        .as_ref()
        .and_then(|(id, t)| (id == session_id).then_some(*t))
}

fn tier_memo_set(session_id: &str, tier: &'static ChunkTier) {
    if let Ok(mut g) = TIER_MEMO.lock() {
        *g = Some((session_id.to_string(), tier));
    }
}

fn tier_memo_clear() {
    if let Ok(mut g) = TIER_MEMO.lock() {
        *g = None;
    }
}

thread_local! {
    /// S165 §100 —— 这一首歌选定的 chunk tier。见 [`tier_for_this_song`]。
    static LOCKED_TIER: std::cell::Cell<Option<&'static ChunkTier>> =
        const { std::cell::Cell::new(None) };
}

/// 释放 [`LOCKED_TIER`] 的 RAII 哨兵;只有加锁的那一层(最外层)真的清。
struct TierLock(bool);
impl Drop for TierLock {
    fn drop(&mut self) {
        if self.0 {
            LOCKED_TIER.with(|c| c.set(None));
        }
    }
}

/// 整首歌用同一个 chunk tier —— **最外层选一次,donor 递归照用**。
///
/// ⛔ 为什么必须锁住(S165 §100,一整天的 A/B 全毁在这上面):
/// tier 是按**当下可用 commit** 选的,而 WDDM 把显存算进 commit,一次渲染就占掉 7 GB 上下。
/// donor 死区是自递归调用 `run_pipeline` 的,于是每渲一段就重选一次 tier(实测一条臂里 61 次)。
/// 后面几十次都看见 commit 已经很紧 ⇒ 降 tier ⇒ **而降 tier 就是换一个新输入 shape,
/// 要再付一张 DirectML first-shape ticket(单张 1.7-3.4 GB)** ⇒ commit 更紧 ⇒ 再降。
/// **越紧越降,越降越紧**,一条臂里从 32 s 一路掉到 10 s。
///
/// 代价不是省内存而是质量:同一份素材、同一个二进制,
/// **tier 32 s 的臂坏帧率 6.2-6.4 %,tier 19 s 的 13.0-13.6 %** —— 整整翻倍。
/// 用户社区报的"内存问题"、以及跨臂 A/B 莫名其妙对不上,都是这一条。
///
/// 锁住之后三件事同时成立:降级螺旋断掉、shape 只有一种(只付一张 ticket)、
/// 同一次渲染里各段可比。⚠ 锁的是**一首歌**,不是进程 —— 下一首重新按当时的内存选。
fn tier_for_this_song(engine: &OnnxEngine, voice_session: &str) -> (&'static ChunkTier, TierLock) {
    tier_for_this_song_with(|| pick_chunk_tier(engine, voice_session))
}

/// [`tier_for_this_song`] 的锁那一半,与引擎解耦以便判据能直接钉住嵌套行为。
fn tier_for_this_song_with(
    pick: impl FnOnce() -> &'static ChunkTier,
) -> (&'static ChunkTier, TierLock) {
    LOCKED_TIER.with(|c| {
        if let Some(t) = c.get() {
            return (t, TierLock(false)); // donor 递归:照用外层定好的
        }
        let t = pick();
        c.set(Some(t));
        (t, TierLock(true))
    })
}

/// ⭐⭐ S165 —— `UTAI_RVC_CHUNK_MAX_S=<秒>` 把 chunk tier **钉死**在不大于它的第一档。
///
/// ⛔⛔ 为什么必须有它:tier 是按【渲染那一刻的可用 commit】选的,**不是配置的一部分**
///    ⇒ 同一份代码、同一条命令,两条臂可以跑出不同的 tier;而实测 19 s 臂的坏率
///    **13.0-13.6 %**、32 s 臂 **6.2-6.4 %**(整整翻倍)⇒ **任何 A/B 都可能是这个在说话**,
///    而不是被测的那把刀。S165 这一天里它坏掉过三次对照。
///    ⚠ 光靠「渲染前等 commit 回到 7900 MB」拦不住:等到的那一刻与 tier 真正被选的那一刻
///    之间隔着模型加载,中间还会被别的进程吃掉。
/// ⚙ 不设 = 今天的自动选择(**逐位不变**);设了就绕过自动选择与 session memo。
/// ⛔ 它**不保证**这一档跑得起来 —— 钉高了就该 OOM,那是使用者的选择,不是这里的判断。
fn chunk_max_s() -> Option<f32> {
    chunk_max_s_from(std::env::var("UTAI_RVC_CHUNK_MAX_S").ok().as_deref())
}

/// env 解析写成纯函数 —— 判据不许去碰进程状态（同 `parse_infrasonic_ms`）。
fn chunk_max_s_from(v: Option<&str>) -> Option<f32> {
    let v: f32 = v?.trim().parse().ok()?;
    (v > 0.0).then_some(v)
}
fn pick_chunk_tier(engine: &OnnxEngine, voice_session: &str) -> &'static ChunkTier {
    // A concurrently evicted session resolves to None (reload-on-miss rebuilds it inside
    // run_typed, AFTER this pick) — fall back to the global preference so an explicit
    // DirectML choice never silently loses the tiering (review round 2). Auto stays
    // conservative-false: the window is one eviction race wide, and guessing DML on a
    // CUDA box would needlessly shorten its chunks.
    // ⭐ 钉死的 tier 绕过自动选择与 session memo。见 [`chunk_max_s`]。
    if let Some(pin) = chunk_max_s() {
        let t = CHUNK_TIERS
            .iter()
            .find(|t| (t.x_max as f32) <= pin)
            .unwrap_or(CHUNK_TIERS.last().expect("tiers non-empty"));
        tracing::info!(
            "RVC chunk tier lowered to x_max={} s (pinned by UTAI_RVC_CHUNK_MAX_S={pin}) (S165)",
            t.x_max
        );
        return t;
    }
    let resolved = engine.resolved_device(voice_session);
    let is_dml = resolved
        .as_deref()
        .map(|d| d.contains("DirectML"))
        .unwrap_or_else(|| {
            matches!(engine.device(), super::engine::DeviceConfig::DirectMl { .. })
        });
    if !is_dml {
        return &CHUNK_TIERS[0];
    }
    // S165 §103 -- a loaded session has ALREADY paid for its shape.
    //
    // The tier decides the chunk length, the chunk length decides the input shape, and a new
    // input shape is what costs a DirectML first-shape ticket (1.7-3.4 GB). So the question
    // this function is really asking is "can I afford a ticket", and once the session is up
    // holding one, the answer for THAT shape is yes -- the memory is already committed. Asking
    // again per song and seeing a low number gets it exactly backwards: it switches to a
    // shorter chunk, which is a NEW shape, which buys a SECOND ticket. That is the loop that
    // walked this machine from 32 s down to 10 s, at twice the defect rate.
    //
    // So: remember the tier per session, and reuse it for as long as that session is alive.
    // Nothing is being relaxed here -- reuse allocates nothing, it declines to allocate again.
    // When the session is gone (evicted, device switched) the ticket went with it and the
    // next pick starts fresh from whatever memory actually exists.
    if tier_memo_enabled() {
        if resolved.is_some() {
            if let Some(t) = tier_memo_get(voice_session) {
                tracing::info!(
                    "RVC chunk tier reused (x_max={} s) — this session already holds its                      first-shape ticket; re-asking would only buy a second one (S165)",
                    t.x_max
                );
                return t;
            }
        } else {
            tier_memo_clear();
        }
    }
    let (_, avail) = super::engine::system_memory_mb();
    if avail == 0 {
        return &CHUNK_TIERS[0]; // measurement failed — keep the default tier
    }
    let tier = CHUNK_TIERS
        .iter()
        .find(|t| avail >= t.need_mb)
        .unwrap_or(CHUNK_TIERS.last().expect("tiers non-empty"));
    if tier_memo_enabled() {
        tier_memo_set(voice_session, tier);
    }
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

/// S159q —— `UTAI_COVER_DONOR_PAD_MS=<ms>`:donor 切片两侧各加多少真实上下文(0 = 关 = 出厂)。
/// 机理与出处写在使用点上(`run_pipeline` 里 donor 那一段)。⛔ 出厂 0 ⇒ 输出逐位不变。
fn cover_donor_pad_ms() -> f32 {
    std::env::var("UTAI_COVER_DONOR_PAD_MS")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= 5000.0)
        .unwrap_or(0.0)
}

/// S160 探针 —— `UTAI_COVER_F0_IN` 的作用域。
///
/// ⛔ `run_pipeline` 的每个 donor 切片会**自递归**调用它自己(`range = None`,音频只有零点几秒),
/// 并且**在同一份源音频上重跑一次 RMVPE** ⇒ donor 继承同一批音高错误,而 `apply_inverse`
/// 只做常数位移 ⇒ **救援会把错的音高原样搬回来**。所以这个探针必须**整条链都顶**:
/// 最外层顶整曲,donor 顶它自己那一段(按整曲时间轴的帧偏移取,再乘它自己的 `f0_shift`)。
/// ⛔ 出厂不设 env ⇒ `SONG` 恒为 None ⇒ 一个分支都不走,输出逐位不变。
/// ⚠ donor 渲染是**顺序**的(`apply_dead_only_windows` 里没有 rayon),thread_local 成立。
mod f0_probe {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    thread_local! {
        /// 整曲时间轴上的 f0(100 fps,**未乘任何 f0_shift**)。
        static SONG: RefCell<Option<Rc<Vec<f32>>>> = const { RefCell::new(None) };
        /// 当前这一遍在整曲时间轴上的起始帧(最外层 = 0)。
        static OFFSET: Cell<usize> = const { Cell::new(0) };
        static DEPTH: Cell<usize> = const { Cell::new(0) };
    }
    pub struct Scope;
    impl Scope {
        pub fn enter() -> Self {
            DEPTH.with(|d| d.set(d.get() + 1));
            Scope
        }
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        }
    }
    pub fn is_outermost() -> bool {
        DEPTH.with(|d| d.get()) <= 1
    }
    pub fn set_song(v: Vec<f32>) {
        SONG.with(|c| *c.borrow_mut() = Some(Rc::new(v)));
    }
    pub fn song() -> Option<Rc<Vec<f32>>> {
        SONG.with(|c| c.borrow().clone())
    }
    pub fn offset() -> usize {
        OFFSET.with(|c| c.get())
    }
    /// RAII:donor 递归期间把「整曲帧偏移」换成切片自己的。
    pub struct SliceOffset(usize);
    impl SliceOffset {
        pub fn set(off: usize) -> Self {
            let prev = OFFSET.with(|c| c.replace(off));
            SliceOffset(prev)
        }
    }
    impl Drop for SliceOffset {
        fn drop(&mut self) {
            OFFSET.with(|c| c.set(self.0));
        }
    }
}
/// S165 §106 -- fill stretches where RMVPE dropped a held note on the floor.
///
/// RMVPE does not merely mis-read pitch, it sometimes stops reporting one. On this record the
/// worst case is 730 ms in the middle of a single sustained note (4:04.80-4:05.53): `pitchf`
/// is 0 there, RVC synthesises those frames as unvoiced, and what comes out is broadband
/// noise. It is the longest single defect left in the song -- 640 ms of the 15 remaining bad
/// segments, more than the next four put together.
///
/// The two cases separate cleanly on waveform autocorrelation, which is why this is safe:
///     inside that hole    peak 0.962, f0 709.6 Hz with a 2.3 Hz spread over 37 frames
///     genuinely unvoiced  peak p50 0.230, p90 0.396, p99 0.673      (n = 122 frames)
/// A 0.75 gate therefore sits above every unvoiced frame measured and far below the hole.
/// The pitch written in is the autocorrelation reading itself rather than an interpolation
/// across the gap: RMVPE's own first frame after the hole reads 714 Hz against the 712 that
/// autocorrelation measures inside it, which is what says the hole is the same held note and
/// not a gap between two.
///
/// ⚠ Runs BEFORE the octave repair on purpose, so anything filled here goes through the same
/// Viterbi as the rest. Filling afterwards would hand the octave logic a run of frames it
/// never had a chance to judge -- and a wrongly-octaved fill is worse than a hole.
fn fill_voiced_holes_inplace(f0: &mut [f32], audio16k: &[f32], hop: usize) -> usize {
    let mut marks = vec![false; f0.len()];
    fill_voiced_holes_marked(f0, audio16k, hop, &mut marks)
}

/// 同上,外加把**这一刀写过的帧**记在 `marks` 里
/// —— [`drop_inconsistent_fills_inplace`] 要它。
fn fill_voiced_holes_marked(
    f0: &mut [f32],
    audio16k: &[f32],
    hop: usize,
    marks: &mut [bool],
) -> usize {
    let min_run = min_hole_frames();
    let mut filled = 0usize;
    let mut i = 0usize;
    while i < f0.len() {
        if f0[i] > 20.0 {
            i += 1;
            continue;
        }
        // Measure the whole hole before deciding whether it is ours. Scattered 1-3 frame
        // gaps already belong to `fill_isolated_uv_inplace`; measuring the first version of
        // this pass showed why the division matters -- of 240 runs it filled, 231 (96 %) were
        // 10-30 ms crumbs and exactly one was the 730 ms dropout it exists for. Sprinkling
        // patches across the song is how an unvoiced consonant gap acquires a pitch.
        let start = i;
        while i < f0.len() && f0[i] <= 20.0 {
            i += 1;
        }
        if i - start < min_run {
            continue;
        }
        // ⛔⛔ 标记的是**整个洞**,不是「写成功的那些帧」。
        //    S165 实测:`fill_one` 可能只写洞里的一部分(4:28.650 那个洞它只写了末帧 552),
        //    剩下的空档随后被 `fill_isolated_uv_inplace` **插值**填上
        //    (835 = (1118 + 552) / 2,一个字都不差)。
        //    若只标写成功的帧,那一帧的左右紧邻就都是「洞里没被填的无声帧」,
        //    而 [`run_anchor`] 一碰到无声就停 ⇒ 取不到锚 ⇒ 整段被跳过。
        //    ⇒ 标整个洞,「一段」才等于「一个洞」。
        for j in start..i {
            if j < marks.len() {
                marks[j] = true;
            }
            if fill_one(f0, audio16k, hop, j) {
                filled += 1;
            }
        }
    }
    filled
}

/// 一段 `[start, end)` 的**锚**:两侧各最多 [`FILL_ANCHOR_FRAMES`] 个真浊音帧的中位。
/// `skip` 为真的帧不算锚——**补洞自己写的那些不能给自己当参照**。
/// 碰到真无声就停（再往外是另一个音）；取不到 ⇒ `0.0` = 没有锚。
fn run_anchor(f0: &[f32], skip: &[bool], start: usize, end: usize) -> f32 {
    let side = |mut k: usize, back: bool| -> Option<f32> {
        let mut v: Vec<f32> = Vec::new();
        loop {
            if back {
                if k == 0 {
                    break;
                }
                k -= 1;
            } else {
                if k >= f0.len() {
                    break;
                }
            }
            if !skip.get(k).copied().unwrap_or(false) {
                if f0[k] > 20.0 {
                    v.push(f0[k]);
                } else {
                    break; // 洞外的真无声 ⇒ 再往外是另一个音了,停
                }
            }
            if v.len() >= FILL_ANCHOR_FRAMES {
                break;
            }
            if !back {
                k += 1;
            }
        }
        if v.is_empty() {
            return None;
        }
        v.sort_by(f32::total_cmp);
        Some(v[v.len() / 2])
    };
    // ⛔ 两侧各取中位再平均 —— 直接把两边的帧混在一起取中位,会被帧数多的那一侧拖过去
    //    (实测 4:28.310 那个洞:混着取 v[5] 读到 1199,而两侧中位分别是 1058 / 1199)。
    let l = side(start, true);
    let r = side(end, false);
    match (l, r) {
        (Some(a), Some(b)) => 0.5 * (a + b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => 0.0,
    }
}

/// 两侧各取几帧当锚。
const FILL_ANCHOR_FRAMES: usize = 5;

/// ⭐ 补进来的值相对锚的容许比（倍）。落在 `[anchor/RATIO, anchor*RATIO]`
/// 之外 ⇒ 交给谱证据裁决。
///
/// ⚙ **1.4**：实测 163 段有锚的填充里,`填值/锚` 的 p10 = 0.818、中位 1.002、
///    p90 = 1.196 ⇒ 1.4 把**正常那一坨整个放行**,同时把 0.5 与 2.0 两个八度各留
///    **1.43 × 的余量**。
const FILL_ANCHOR_RATIO: f32 = 1.4;

/// ⭐⭐⭐ S165 —— 把**与自己两侧对不上**的补洞帧撤掉（或搬到对得上的那个八度）。
///
/// ⛔ 病因：[`fill_one`] 在 `[F_MIN, F_MAX]` 里**自己找自相关峰、完全不看洞的两侧**,
///    而在衰减尾/音节间隙上,两倍周期的自相关常常比真周期还高 ⇒ 锁到次谐波。
///    实测这首歌 163 段有锚的填充：`填值/锚` 有 **119 段（73 %）落在 0.95-1.05**（完全对）,
///    却拖着一条低尾巴（0.41 / 0.49 / 0.52 / 0.54 / **0.55 = 4:28.310** / **0.60 = 4:28.650** …）,
///    **它们的锚全在 985-1206 Hz**,即贴着或高于 `F_MAX` 能表示的 1143 Hz。
///    用户 2026-08-26 报的 4:28.3 与 4:28.638 两条尾巴就是它：补洞把音的释放段写成浊音、
///    而且写的是次谐波,PSOLA 照着合成 ⇒ f0 以下冒出一团（关掉补洞的对照臂上它们干净消失）。
///
/// ⛔⛔ **为什么必须跑在 [`fix_octave_inplace`] 之后**（承重；S165 第一版就栽在这）：
///    补洞跑在八度修复**之前**,那时的 `f0` 本身就可能整段偏低一个八度
///    ——「ぴゃ」那一带 RMVPE 读的是 ~710,是后面 Viterbi 整段抬成 1412 的。
///    在那一层取锚 = 拿一个自己都不可靠的参照去做八度判断，
///    实测**把 ぴゃ 打坏了 16 帧**（1391→0、1455→364）。
///    放到修复之后,那一带的填充值（1412）与邻居（1417）自然吻合 ⇒ **一帧都不动**。
///
/// 逐段处理被 `marks` 标记的填充：落在锚的 ±[`FILL_ANCHOR_RATIO`] 内 ⇒ 留；
/// 否则试它的整数倍/整数分之一,**哪个八度在音频里真的站得住就搬到哪个**
/// （共用 [`OctaveSpec`] 与 [`OCTAVE_OUTLIER_MIN_DB`]）；一个都站不住 ⇒ **撤掉**。
///
/// 返回（撤掉的帧数, 搬了八度的帧数）。
fn drop_inconsistent_fills_inplace(
    f0: &mut [f32],
    audio16k: &[f32],
    hop: usize,
    marks: &[bool],
) -> (usize, usize) {
    if !anchor_enabled() || f0.is_empty() {
        return (0, 0);
    }
    let spec = OctaveSpec::new();
    let ratio = fill_anchor_ratio();
    let (mut dropped, mut moved) = (0usize, 0usize);
    let n = f0.len().min(marks.len());
    let mut i = 0usize;
    while i < n {
        if !marks[i] {
            i += 1;
            continue;
        }
        let a = i;
        while i < n && marks[i] {
            i += 1;
        }
        let anchor = run_anchor(f0, marks, a, i);
        if anchor <= 20.0 {
            continue; // 没有锚 ⇒ 无从判断,不动
        }
        let in_band = |f: f32| f > anchor / ratio && f < anchor * ratio;
        // ⭐⭐ 判的是**这一段整体**,不是逐帧。
        //    ⛔ 逐帧会让一条单调下滑的坡从带边溜过去:4:28.310 那个洞填的是
        //    `856 667 619 571 593` —— 整条都是自相关在衰减尾上一路走低的产物,
        //    而 856 恰好卡在带内 ⇒ 逐帧判会把它留下,坡还在。
        let mut vals: Vec<f32> = (a..i).map(|k| f0[k]).filter(|v| *v > 20.0).collect();
        if vals.is_empty() {
            continue;
        }
        vals.sort_by(f32::total_cmp);
        if in_band(vals[vals.len() / 2]) {
            continue; // 整段与两侧对得上 ⇒ 一帧不动
        }
        for k in a..i {
            let v = f0[k];
            if v <= 20.0 {
                continue;
            }
            let Some(mag) = spec.frame_mag(audio16k, k, hop) else {
                continue;
            };
            let p_v = spec.prominence_db(&mag, f64::from(v));
            let mut best: Option<(f32, f64)> = None;
            for m in [2.0f32, 3.0, 0.5, 1.0 / 3.0] {
                let f = v * m;
                if !in_band(f) || f64::from(f) >= OCTAVE_SR / 2.0 * 0.9 {
                    continue;
                }
                let d = spec.prominence_db(&mag, f64::from(f)) - p_v;
                if d >= f64::from(OCTAVE_OUTLIER_MIN_DB) && best.is_none_or(|(_, bd)| d > bd) {
                    best = Some((f, d));
                }
            }
            match best {
                Some((f, _)) => {
                    f0[k] = f;
                    moved += 1;
                }
                None => {
                    f0[k] = 0.0;
                    dropped += 1;
                }
            }
        }
    }
    (dropped, moved)
}

/// ⛔ 它**不看洞的两侧** —— 那道一致性闸在 [`drop_inconsistent_fills_inplace`],
/// 跑在**八度修复之后**。⛔⛔ 不许搬回这一层：S165 试过,把那个 730 ms 的
/// 「ぴゃ」漏唱打坏了 16 帧 —— 因为此刻的 `f0` 本身就可能整段偏低一个八度。
fn fill_one(f0: &mut [f32], audio16k: &[f32], hop: usize, i: usize) -> bool {
    /// Above every unvoiced frame measured (p99 = 0.673), far below the hole (0.962).
    const PEAK_GATE: f32 = 0.75;
    /// Below 200 Hz a 40 ms window holds too few periods to trust.
    const F_MIN: f32 = 200.0;
    /// Capped at what `f0_to_coarse` can carry. Raising it was tried and MEASURED WORSE.
    ///
    /// The argument for raising it sounded solid: at the 730 ms dropout the true fundamental
    /// is ~1420 Hz, and with the cap at 1100 autocorrelation locks onto a sub-multiple and
    /// writes 470 Hz — a twelfth low. Reading the truth should also let the frame trip the
    /// `RVC_COARSE_MAX_HZ` death test and get rescued down properly. So: cap to 1600, render,
    /// measure.
    ///
    /// What that missed is that the filled value is a CONDITION handed to the model, not the
    /// output pitch. Measured at the dropout (spectral peaks, source median 1392 Hz):
    ///     cap 1100   output 1384 Hz   -10 cents      whole-song bad frames 4.32 %
    ///     cap 1600   output 1426 Hz   +42 cents                            4.56 %
    /// Writing 470 Hz does not make the model sing a twelfth low — those frames simply skip
    /// the death test and the model tracks the source anyway. Writing 1454 Hz does trip it,
    /// and the -15 semitone rescue that follows costs both pitch accuracy and cleanliness.
    /// Both arms of the comparison lose, so the cap stays where it was.
    const F_MAX: f32 = 1100.0;
    /// 40 ms at 16 kHz -- the window the separation above was measured with.
    const WIN: usize = 640;

    let sr = 16_000.0f32;
    let lag_lo = (sr / F_MAX) as usize;
    let lag_hi = (sr / F_MIN) as usize;
    if lag_hi + 2 >= WIN {
        return false;
    }
    let c = i * hop;
    if c + WIN > audio16k.len() {
        return false;
    }
    let w = &audio16k[c..c + WIN];
    let mean = w.iter().sum::<f32>() / WIN as f32;
    let e0: f32 = w.iter().map(|v| (v - mean) * (v - mean)).sum();
    if e0 <= 1e-9 {
        return false;
    }
    let (mut best_lag, mut best) = (0usize, 0.0f32);
    for lag in lag_lo..=lag_hi {
        let mut acc = 0.0f32;
        for k in 0..WIN - lag {
            acc += (w[k] - mean) * (w[k + lag] - mean);
        }
        let v = acc / e0;
        if v > best {
            best = v;
            best_lag = lag;
        }
    }
    if best >= PEAK_GATE && best_lag > 0 {
        f0[i] = sr / best_lag as f32;
        return true;
    }
    false
}

/// ⚙ 出厂开;`UTAI_COVER_FILL_ANCHOR=0` 关掉 ⇒ 逐位回到 S165 §106 那一版。
fn anchor_enabled() -> bool {
    !matches!(std::env::var("UTAI_COVER_FILL_ANCHOR").as_deref(), Ok("0"))
}

/// ⚙ `UTAI_COVER_FILL_ANCHOR_RATIO=<倍>` 可扫;见 [`FILL_ANCHOR_RATIO`]。
fn fill_anchor_ratio() -> f32 {
    std::env::var("UTAI_COVER_FILL_ANCHOR_RATIO")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| *v > 1.0 && *v < 4.0)
        .unwrap_or(FILL_ANCHOR_RATIO)
}

/// S165 §106 —— 归这一刀管的最小洞长(帧,100 fps)。
/// 出厂 5 帧 = 50 ms:比清辅音的下限(50-120 ms 是**有声辅音**的量级,而 1-3 帧的缝是
/// 跟踪器抖动)高,又远低于那个 730 ms 的漏唱。`UTAI_COVER_MIN_HOLE_FRAMES` 可扫。
fn min_hole_frames() -> usize {
    std::env::var("UTAI_COVER_MIN_HOLE_FRAMES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1 && *n <= 200)
        .unwrap_or(5)
}


/// 八度判据用的分析采样率。⛔ 与 RVC 的 16 kHz f0 链一致,不是渲染采样率。
const OCTAVE_SR: f64 = 16_000.0;

/// ⭐ S165 —— 八度判据的**唯一一份**谱算子(40 ms 汉宁窗 + 谐波突出度)。
///
/// ⛔ 为什么要有它:[`fix_octave_inplace`] 与 [`fix_octave_outliers_inplace`] 必须用
///    **同一把尺子**。两遍各写一份 = 两把会各自漂移的尺子,而第二遍的门槛
///    ([`OCTAVE_OUTLIER_MIN_DB`])是**照着第一遍量出来的读数**定的 —— 尺子一漂,
///    那个门槛就悄悄失去含义。
struct OctaveSpec {
    win: usize,
    nfft: usize,
    w: Vec<f64>,
}

impl OctaveSpec {
    fn new() -> Self {
        let win = (0.040 * OCTAVE_SR) as usize; // 40 ms
        let nfft = win.next_power_of_two();
        // 汉宁窗(一次算好)
        let w: Vec<f64> = (0..win)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / win as f64).cos())
            .collect();
        Self { win, nfft, w }
    }

    /// 第 `i` 帧(`hop` 栅,窗以帧心为中心)的幅度谱;窗越界 ⇒ `None`。
    fn frame_mag(&self, audio16k: &[f32], i: usize, hop: usize) -> Option<Vec<f64>> {
        let c = i * hop;
        let a = c.saturating_sub(self.win / 2);
        if a + self.win > audio16k.len() {
            return None;
        }
        let mut re: Vec<f64> = (0..self.nfft)
            .map(|k| if k < self.win { f64::from(audio16k[a + k]) * self.w[k] } else { 0.0 })
            .collect();
        let mut im = vec![0.0f64; self.nfft];
        octave_fft(&mut re, &mut im);
        Some((0..=self.nfft / 2).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()).collect())
    }

    /// `f` 处的谐波突出度(dB)= `f` 附近的峰 / `0.5f` 与 `1.5f` 两处的背景。
    fn prominence_db(&self, mag: &[f64], f: f64) -> f64 {
        let nfft = self.nfft;
        let bin = |x: f64| -> f64 { x * nfft as f64 / OCTAVE_SR };
        let peak = {
            let (lo, hi) = (bin(f * 0.88).floor().max(0.0) as usize, bin(f * 1.12).ceil() as usize);
            mag[lo.min(nfft / 2)..=hi.min(nfft / 2)].iter().copied().fold(0.0f64, f64::max)
        };
        let bg = {
            let mut v = Vec::new();
            for k in [0.5f64, 1.5] {
                let (lo, hi) =
                    (bin(f * k * 0.92).floor().max(0.0) as usize, bin(f * k * 1.08).ceil() as usize);
                let sl = &mag[lo.min(nfft / 2)..=hi.min(nfft / 2)];
                if !sl.is_empty() {
                    v.push(sl.iter().sum::<f64>() / sl.len() as f64);
                }
            }
            if v.is_empty() { 1e-12 } else { v.iter().sum::<f64>() / v.len() as f64 }
        };
        20.0 * (peak.max(1e-12) / bg.max(1e-12)).log10()
    }
}

/// ⭐ 孤立八度离群帧的**谱证据**门槛(dB)。见 [`fix_octave_outliers_inplace`]。
///
/// ⚙ **8 dB**:实测这首歌全部四个候选,`prom(2f) − prom(f)` 分成泾渭分明的两族 ——
///    真的错:`4:25.960` **+21.8** · `4:28.640` **+15.4**
///    真的低音起音:`4:26.910` **+0.7** · `4:34.180` **−6.0**
///    ⇒ 空档是 `0.65 … 15.41`,取中点 **8.03** ⇒ 两边各留 ≥7 dB 余量。
///    ⛔ 别调成「差不多就行」的小数:这道门是这一遍**唯一**的防误伤手段。
const OCTAVE_OUTLIER_MIN_DB: f32 = 8.0;

/// 孤立游程的最大长度(帧)。⛔ 超过它的就是**成段**的错,归 [`fix_octave_inplace`] 的
/// Viterbi 管 —— 那里有时间连续性,这里没有。
const OCTAVE_OUTLIER_MAX_RUN: usize = 2;

/// 修完之后的 f0 至少要有这么高(Hz),否则这一帧不动。
const OCTAVE_OUTLIER_MIN_HZ: f64 = 50.0;

/// ⭐⭐⭐ S165 —— 【孤立八度离群帧】的补刀。
///
/// ⛔⛔ **为什么 [`fix_octave_inplace`] 结构上修不动这一类**(这才是这个函数存在的理由):
///    那一遍是 Viterbi,换档罚 [`OCTAVE_SWITCH_PENALTY_DB`] = 12 dB **每次**。
///    一个**单帧**的八度错要「换上去、再换回来」⇒ 付 **两次 = 24 dB**,
///    而实测证据只有 15.4 dB(`4:28.640` 那一帧:`prom(559 Hz) = 10.17`,
///    `prom(1118 Hz) = 25.58`)⇒ **它永远赢不了**,和罚值定得对不对无关。
///    ⚠ 岛**末**帧更糟:下一帧无声,`obs = [0, 1e9, 1e9]` 把状态硬拽回 `×1`,
///    连「换上去就不用换回来」这条退路都没有。
///
/// ⭐ 症状:喂给 PSOLA 的 f0 在那一帧只有真值的一半 ⇒ 颗粒按**两倍周期**摆
///    ⇒ 输出多出一条 `f0/2` 的次谐波。用户 2026-08-26 点名的 `4:28.630` 就是它:
///    1150/2350/3550 那一摞谐波突然断掉,同时 ~575 Hz 冒出一团,源与纯 base 都没有。
///    全曲统计:岛**末**帧掉半 0.8 %、岛**首**帧 0.8 %,而岛**内**只有 0.01 %
///    ⇒ 边界处富集 **142 ×**。
///
/// ⛔ 它**不是**「把 [`OCTAVE_SWITCH_PENALTY_DB`] 调小」的替代品:调小罚值会放松**全曲**的
///    成段判据(S160 已判负逐帧独立版),而这一遍只在**孤立短游程**上开口 ——
///    成段的错它一帧都碰不到(见 [`OCTAVE_OUTLIER_MAX_RUN`])。
///
/// 返回改动的帧数。
fn fix_octave_outliers_inplace(pitchf: &mut [f32], audio16k: &[f32], hop: usize) -> usize {
    if pitchf.is_empty() || audio16k.len() < 4 * hop {
        return 0;
    }
    let spec = OctaveSpec::new();
    let n = pitchf.len();
    let voiced = |v: f32| v > 20.0;
    let mut fixed = 0usize;
    let mut i = 0usize;
    while i < n {
        if !voiced(pitchf[i]) {
            i += 1;
            continue;
        }
        // 这个浊音岛 `[a, b)`。⛔ 邻居只在**岛内**取:隔着无声的两个音本来就可以差一个八度。
        let a = i;
        let mut b = i;
        while b < n && voiced(pitchf[b]) {
            b += 1;
        }
        i = b;
        let mut s = a;
        while s < b {
            let mut hit = 0usize;
            'run: for l in 1..=OCTAVE_OUTLIER_MAX_RUN {
                if s + l > b {
                    break;
                }
                let left = (s > a).then(|| f64::from(pitchf[s - 1]));
                let right = (s + l < b).then(|| f64::from(pitchf[s + l]));
                if left.is_none() && right.is_none() {
                    // 整个岛都在游程里 ⇒ 没有参照,不动(那是成段的错,归 Viterbi)。
                    break;
                }
                // `up = true` ⇒ 这一段被读成了**一半**,要 ×2;`false` ⇒ 被读成两倍,要 ×0.5。
                for up in [true, false] {
                    let (lo_r, hi_r) = if up { (0.42, 0.62) } else { (1.65, 2.40) };
                    let shaped = (s..s + l).all(|k| {
                        let v = f64::from(pitchf[k]);
                        [left, right].iter().flatten().all(|&nb| {
                            let r = v / nb;
                            r > lo_r && r < hi_r
                        })
                    });
                    if !shaped {
                        continue;
                    }
                    // 谱证据:游程里**每一帧**都要过门,不许拿一帧的证据去改两帧。
                    let proven = (s..s + l).all(|k| {
                        let f = f64::from(pitchf[k]);
                        let tgt = if up { 2.0 * f } else { 0.5 * f };
                        if tgt < OCTAVE_OUTLIER_MIN_HZ || tgt >= OCTAVE_SR / 2.0 * 0.9 {
                            return false;
                        }
                        let Some(mag) = spec.frame_mag(audio16k, k, hop) else {
                            return false;
                        };
                        spec.prominence_db(&mag, tgt) - spec.prominence_db(&mag, f)
                            >= f64::from(OCTAVE_OUTLIER_MIN_DB)
                    });
                    if !proven {
                        continue;
                    }
                    for k in s..s + l {
                        pitchf[k] *= if up { 2.0 } else { 0.5 };
                        fixed += 1;
                    }
                    hit = l;
                    break 'run;
                }
            }
            s += hit.max(1);
        }
    }
    fixed
}

/// ⭐⭐⭐⭐⭐ S165 —— **修 RMVPE 的八度错**(把真基频报成 1/2)。
///
/// # ⛔ 它为什么必须存在
/// 用户 2026-08-29 点名翻唱轨 **10 段**「大面积的灾难,包括但不限于**八度跳变**、哑音、失声、噪声」。
/// 逐帧量下来(cover 成品 vs 源,`1200·log2(cover/src) < −900` 的帧占比):
/// **10 段全部八度偏低 14.1%-76.5%**,而**用户没点名的 6 段是 0.0%-0.5%** —— 差 30-150 倍。
/// 哑/噪/失声**都挂在八度错上**,是后果不是并列症状。
///
/// ⛔⛔ **而记忆里写着 S160「已修:整条链 Viterbi 修 f0」—— 代码里从来没有。**
/// S160 做的是 `UTAI_COVER_F0_IN` 这个**探针**(doc 逐字写着「出厂不设 env ⇒ 一个分支都不走」),
/// 它证明的是「**f0 准了会怎样**」,不是「已经准了」。⇒ 生产路径上这个错**一直都在**。
///
/// # ⭐ 判据:**光看 f0 序列判不了八度,必须看音频**
/// 如果 RMVPE 报的 `fc` 是真基频,源音频在 `fc` 处**必然有谱峰**;
/// 如果真基频是 `2·fc`,`fc` 处**什么都没有**(那里是谐波之间)。
/// ⇒ 量 `fc` 处相对**谐波间背景**(`0.5·fc` 与 `1.5·fc`)的突出度。
///
/// 实测(真实的 RMVPE 输出 + 源音频,`prom(fc)` dB):
/// | | prom(fc) | 判为八度错的帧% |
/// |---|---|---|
/// | 用户点名的最坏三段 | **1.7 / 2.7 / −3.4** | 50.0 / 29.3 / 43.6 |
/// | **阴性对照(没点名的 5 段)** | **41.0-48.1** | **0.0-0.8** |
/// ⇒ 分得干干净净。
///
/// # ⛔ 必须带时间连续性
/// S160 登记过「**逐帧独立版判负**」:单帧的判断会在一个长音内部来回翻,
/// 造出比八度错本身更难听的东西。⇒ 这里用 **Viterbi**(两状态:保持 / 翻倍),
/// 转移代价 [`OCTAVE_SWITCH_PENALTY_DB`] 把翻转压到**成段**发生。
///
/// # ⚠ 只往上修,不往下;候选 `{fc, 2·fc, 3·fc}`
/// RMVPE 的已知失败模式是**把真基频报成它的 1/2 或 1/3**;没有观测到「报成 2 倍」。
/// ⇒ 候选不含 `fc/2` —— 少一个状态就少一族误伤。
///
/// ⭐ **`3·fc` 是实测逼出来的**:用户点名的「ぴゃ」(244.29-245.76 s,全曲最高音)
/// 源实测真 f0 **1422.6 Hz**,而 RMVPE 在那里报 **471/473**(= 1422.6 **/ 3**)、
/// 少数帧报 **705**(= / 2)、**61 % 的帧直接报无声**。
/// 只有 `{fc, 2fc}` 时那一段被抬到 942 ——**还是错的,只是换了个错法**,
/// 实测该段八度错率从 88.6 % 只降到 73.8 %(其余九段降到 0-40 %)。
/// ⛔ 那 61 % 的无声是**另一族**(RMVPE 在极高音上完全失效),这把刀够不着。
fn fix_octave_inplace(pitchf: &mut [f32], audio16k: &[f32], hop: usize) -> usize {
    if pitchf.is_empty() || audio16k.len() < 4 * hop {
        return 0;
    }
    let spec = OctaveSpec::new();
    // 每帧三个状态(×1 / ×2 / ×3)的观测代价
    let mut obs: Vec<[f64; 3]> = Vec::with_capacity(pitchf.len());
    for (i, &fc) in pitchf.iter().enumerate() {
        if fc <= 20.0 || 2.0 * f64::from(fc) >= OCTAVE_SR / 2.0 * 0.9 {
            obs.push([0.0, 1e9, 1e9]); // 无声/太高 ⇒ 只能保持
            continue;
        }
        let Some(mag) = spec.frame_mag(audio16k, i, hop) else {
            obs.push([0.0, 1e9, 1e9]);
            continue;
        };
        let f = f64::from(fc);
        let p1 = spec.prominence_db(&mag, f);
        let p2 = spec.prominence_db(&mag, 2.0 * f);
        // ⭐ 3 倍那一支:超出 Nyquist 就直接不可用(而不是读到一个折叠回来的假峰)。
        let p3 = if 3.0 * f < OCTAVE_SR / 2.0 * 0.9 {
            spec.prominence_db(&mag, 3.0 * f)
        } else {
            -1e9
        };
        // 代价 = 负的突出度(越突出越便宜);⭐ `×3` 另加先验代价,见 [`OCTAVE_TRIPLE_PRIOR_DB`]。
        let p3c = if p3 <= -1e8 { -p3 } else { -p3 + f64::from(OCTAVE_TRIPLE_PRIOR_DB) };
        obs.push([-p1, -p2, p3c]);
    }
    // Viterbi(三状态:×1 / ×2 / ×3)
    let n = obs.len();
    let mut cost = obs[0];
    let mut back: Vec<[u8; 3]> = vec![[0, 0, 0]; n];
    let pen = f64::from(OCTAVE_SWITCH_PENALTY_DB);
    for i in 1..n {
        let mut next = [0.0f64; 3];
        for s in 0..3 {
            // 从哪个前一状态过来最便宜(保持不罚,换档罚 `pen`)
            let mut best = (0usize, f64::INFINITY);
            for pv in 0..3 {
                let c = cost[pv] + if pv == s { 0.0 } else { pen };
                if c < best.1 {
                    best = (pv, c);
                }
            }
            next[s] = best.1 + obs[i][s];
            back[i][s] = best.0 as u8;
        }
        cost = next;
    }
    let mut st = (0..3).min_by(|&a, &b| cost[a].total_cmp(&cost[b])).unwrap_or(0);
    let mut path = vec![0u8; n];
    for i in (0..n).rev() {
        path[i] = st as u8;
        st = back[i][st] as usize;
    }
    let mut fixed = 0usize;
    for (i, &s) in path.iter().enumerate() {
        if s > 0 && pitchf[i] > 20.0 {
            pitchf[i] *= (s + 1) as f32;
            fixed += 1;
        }
    }
    fixed
}

/// 就地 radix-2 FFT。⛔ 只给 [`fix_octave_inplace`] 用。
fn octave_fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                    re[i + k + len / 2] * ci + im[i + k + len / 2] * cr,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// ⭐ 八度翻转的转移代价(dB)。⛔ 它是「**成段翻转**」与「逐帧乱跳」之间的唯一防线
/// (S160 登记过逐帧独立版判负)。
/// ⚙ **12 dB**:实测点名段的 `prom(fc)` 与 `prom(2fc)` 差 **20-30 dB**(真的错时),
/// 而对照段差 **35-40 dB**(反方向)⇒ 12 dB 的门槛只有真的成段错时才跨得过去。
const OCTAVE_SWITCH_PENALTY_DB: f32 = 12.0;

/// ⭐⭐⭐⭐ S165 —— **`×3` 那一支的先验代价(dB)**。
///
/// # ⛔ 它为什么必须存在:`×3` 会大面积误伤,而 `×2` 不会
/// `3·fc` 落在更高频,那一带**谐波密集、背景估不准** ⇒ `prom(3·fc)` 容易虚高。
/// 实测(离线复刻整条链,先验 = 0):**阴性对照段被抬 8.7 / 18.6 / 23.9 %,
/// 而且几乎全部来自 `×3`**;同一批段在只有 `{fc,2fc}` 时是 **0.0-0.8 %**。
///
/// # ⚙ 扫描(点名段收益 vs 阴性误伤)
/// | 先验 | 点名段均值 | **阴性最大** | ぴゃ | 4:50 |
/// |---|---|---|---|---|
/// | 0 | 41.8 | **23.9** | 100.0 | 83.9 |
/// | 9 | 41.5 | **11.7** | 100.0 | 83.9 |
/// | **12** | 41.5 | **1.5** | 100.0 | 83.9 |
/// | 24 | 41.5 | 0.8 | 100.0 | 83.9 |
/// ⇒ **12 dB 是拐点**(阴性 23.9 → 1.5),而**点名段一点收益都没丢**。
/// ⚙ 取 **18**:在拐点的安全侧、离饱和点近(阴性 1.4 %),`×3` 从 1850 帧收到 **221 帧**。
///
/// # ⭐⭐ 承重的验收不是「抬了多少帧」,是「抬完之后对不对」
/// 逐帧对**源音频实测的真 f0**算误差:
/// | | ぴゃ | 4:50 | **全曲 `|误差|>900 音分` 的帧** |
/// |---|---|---|---|
/// | 修前 | 1900 音分 | 1218 音分 | **8.6 %** |
/// | `×3` 无先验 | → 21 | → 11 | **10.3 %**(⛔ **反而更糟**)|
/// | **`×3` + 先验 18** | → **21** | → **11** | **8.6 % → 2.2 %** |
/// ⇒ ⛔⛔ **只看点名段会被骗**:无先验版把两个点名音修到 11-21 音分,
///   却让**全曲**错误率不降反升。**任何「修好了 X」都要配一个全曲口径。**
const OCTAVE_TRIPLE_PRIOR_DB: f32 = 18.0;

/// S160d —— 填掉 `pitchf` 里**长度恰好 1 帧**、两侧都是浊音的 0(线性插值 = 两侧均值)。返回填了几个。
///
/// ## ⛔ 为什么
/// 用户 2026-08-24 点名 0:58.052「跟着清辅音一起进的咔哒」。逐处量出来:`pitchf` 在
/// 58.03/58.05 是 1583/1599 Hz,而 **58.04 恰好是 0** —— 一个孤立单帧洞。后果两条:
/// ⑴ NSF 被告知那 10 ms 无声 ⇒ 谐波栈断;⑵ PSOLA 的浊音岛被劈成两半,多出两条岛边
/// (S159zi 机理③:岛边 = 单样本宽带阶跃)。
///
/// ## ✅ 实测(LPC 残差尖峰相对源的尖度,同一二进制 `2f68e47670f5`)
/// 58.050 **×67.1 → ×1.9**;252.950(另一次「そしたら」,同一个假名、同一个洞)**×21.2 → ×1.9**。
/// ⭐ **同参两跑地板**:那两处两跑差只有 **0.2**,而全曲尖度 >5 的帧两跑 `|Δ|` p90 3.8 / max 49.4
/// ⇒ 这两处是全曲**唯一**稳稳超过地板的咔哒。⛔ 57.600 那处(51.0 vs 再跑 8.2)是**渲染噪声**,别追。
///
/// ## ⭐ 谱面轨早就这么做了
/// `score2svc.rs` 的 `fill_isolated_uv_max`(S159zp),理由逐字适用于这里:
/// **真实音频里的清辅音给出的是一串 0(50-120 ms);孤立单帧 0 是跟踪器抖动,不是音位事件。**
/// 这条路以前从来没被应用到 cover。⚠ 规模:用户那份素材全曲 **19 个**(2 帧 ×25、3 帧 ×31,一个不碰)。
///
/// ⛔ **先收集再写**:边扫边写会让连续的 0 被逐个「补」成浊音,越过「长度恰好 1」这条界。
fn fill_isolated_uv_inplace(pitchf: &mut [f32]) -> usize {
    if pitchf.len() < 3 {
        return 0;
    }
    let holes: Vec<usize> = (1..pitchf.len() - 1)
        .filter(|&i| pitchf[i] == 0.0 && pitchf[i - 1] > 0.0 && pitchf[i + 1] > 0.0)
        .collect();
    for &i in &holes {
        pitchf[i] = 0.5 * (pitchf[i - 1] + pitchf[i + 1]);
    }
    holes.len()
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
    let _depth = f0_probe::Scope::enter();
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
    let mut audio_f = highpass_48hz_16k(&wav16k)?;

    // S165 -- flatten the INPUT envelope before the model sees it (factory OFF).
    //
    // WHY. The cover break-ups were traced (S165 §88-§90) to one thing: RVC cannot synthesise a
    // regular glottal pulse train while the energy is moving fast. Measured over the whole song
    // (160 ms windows, fully-voiced only, n=3675), bad-window rate by in-window energy slope:
    //     -12..-8 dB -> 10.3%   -8..-5 -> 4.7%   -2..+2 (flat) -> 0.2% (n=1093)   +2..+5 -> 0.4%
    // i.e. moving-energy windows are 5-50x worse than flat ones, decay side worst.
    //
    // The score lane never hits this because its loudness is a CONSTANT placeholder
    // (`VOCAL_FLAT_VOL`, real per-frame loudness deferred to §10.5): |slope|>5 dB in only 2.8% of
    // its windows vs 29.6% for the cover source. It is not stronger -- it never gets that input.
    // ⇒ The user's original architecture call ("do loudness as POST-processing") is what saves it.
    //
    // So do the same for cover: flatten going IN, and let the existing `change_rms` (which already
    // runs AFTER decoding, rvc.rs ~:793) put the source envelope back. Offline measurement of this
    // exact filter (100 ms RMS window, +/-12 dB cage):
    //     windows with |slope|>5 dB : 29.6% -> 9.7%      windows below -8 dB : 9.9% -> 2.5%
    //     (score lane reference: 2.8% / 1.8%)   cost to the source's own periodicity: -0.021
    //
    // ⚠ Factory OFF until an A/B render is heard: this changes what the model is fed everywhere,
    // not just at the 25 defect segments.
    if let Some(win_ms) = cover_flatten_ms() {
        // Shape knobs. One-sided by default: LIFT the quiet stretches, never push the loud
        // ones down. The two-sided version was rendered and rejected in S165 -- it fixed
        // nothing at the 25 defect segments (p50 0.542 -> 0.544/0.560, inside the render
        // noise floor), made the whole song WORSE (p50 0.804 -> 0.792, bad-window rate
        // 2.2% -> 3.4%), and collapsed how well the output still tracks the source's own
        // envelope (r 0.882 -> 0.61) -- because the restore on the way out (`change_rms`,
        // rms_mix_rate = 0.25) only gives back a quarter of what was taken away.
        //
        // One-sided is the right shape because BOTH symptoms only ever need a lift:
        //   * dropout: a quiet stretch that the model reads as "this is all there is"
        //   * break-up: energy falling FAST -- lifting the bottom of the fall flattens the
        //     slope just as well as pressing the top down would have
        // and because the loud stretches keep their original gain, `change_rms` has an
        // order of magnitude less to undo.
        let ref_pct = cover_flatten_knob("UTAI_COVER_FLATTEN_REF_PCT", 40.0, 1.0, 99.0);
        let cap_db = cover_flatten_knob("UTAI_COVER_FLATTEN_CAP_DB", 9.0, 0.0, 24.0);
        // Below this (relative to the reference) a window is breath, sibilance or room
        // tone, not an under-sung note. ⚠ The two-sided version had no such floor: its
        // gain for a silent window ran straight into the cage (env -> 0 => g = 4.0), so it
        // lifted the noise bed by 12 dB. That is a likely part of why its bad-window rate
        // went UP. One-sided makes the omission worse, not better, so the floor is not
        // optional here.
        let floor_db = cover_flatten_knob("UTAI_COVER_FLATTEN_FLOOR_DB", -24.0, -60.0, 0.0);
        // dB per second the gain may move; 0 = uncapped (the S165 §95 behaviour).
        let slew_db_s = cover_flatten_knob("UTAI_COVER_FLATTEN_SLEW_DB_S", 0.0, 0.0, 2000.0);
        let two_sided = cover_flatten_two_sided();

        let n = ((win_ms / 1000.0) * SR as f32) as usize;
        if n >= 2 && audio_f.len() > n {
            // running RMS via prefix sums
            let mut pre = vec![0.0f64; audio_f.len() + 1];
            for (i, &v) in audio_f.iter().enumerate() {
                pre[i + 1] = pre[i] + f64::from(v) * f64::from(v);
            }
            let half = n / 2;
            let mut env = vec![0.0f32; audio_f.len()];
            for i in 0..audio_f.len() {
                let a = i.saturating_sub(half);
                let b = (i + half).min(audio_f.len());
                env[i] = (((pre[b] - pre[a]) / (b - a).max(1) as f64).sqrt() as f32).max(1e-6);
            }
            // Reference = a percentile of the VOICED-level envelope. The two-sided version
            // used the median; one-sided wants it LOWER, because everything under the
            // reference gets lifted and the median would lift half the song.
            let mut lv: Vec<f32> = env.iter().copied().filter(|&v| v > 1e-4).collect();
            if !lv.is_empty() {
                lv.sort_by(f32::total_cmp);
                let idx = (((ref_pct / 100.0) * lv.len() as f32) as usize).min(lv.len() - 1);
                let refv = lv[idx];
                let hi = 10f32.powf(cap_db / 20.0);
                let lo = if two_sided { 1.0 / hi } else { 1.0 };
                let floor = refv * 10f32.powf(floor_db / 20.0);
                let mut gain = vec![1.0f32; audio_f.len()];
                for (i, g) in gain.iter_mut().enumerate() {
                    let e = env[i];
                    let mut v = (refv / e).clamp(lo, hi);
                    // Fade the lift out below the floor instead of cutting it off, so the
                    // gain stays continuous across the boundary (a step here would be a
                    // new transient -- exactly the thing this filter exists to remove).
                    if e < floor {
                        let t = (e / floor.max(1e-9)).clamp(0.0, 1.0);
                        v = 1.0 + (v - 1.0) * t;
                    }
                    *g = v;
                }
                // S165 §99 -- cap how fast the gain itself may move.
                //
                // Measured need: the first (unlimited) version cut spiky frames across the
                // song by 60% but GREW 19 new spiky spots of its own, one of which the user
                // picked out by ear (0:50.10 -- source crest 9.58, off 7.65, lifted 12.21).
                // The cause is this filter's own doing: where the envelope turns sharply the
                // gain turns sharply with it, and a fast gain move IS a transient -- the very
                // thing the filter exists to remove.
                //
                // Two passes taking the min, so the cap holds in both directions and the
                // curve picks up no delay (a one-way pass would smear every lift later in
                // time, which would smear it onto the wrong syllable).
                if slew_db_s > 0.0 {
                    let step = 10f32.powf(slew_db_s / (SR as f32) / 20.0);
                    for i in 1..gain.len() {
                        gain[i] = gain[i].min(gain[i - 1] * step);
                    }
                    for i in (0..gain.len() - 1).rev() {
                        gain[i] = gain[i].min(gain[i + 1] * step);
                    }
                }
                let mut moved = 0usize;
                let mut lifted_db_sum = 0.0f64;
                for (i, v) in audio_f.iter_mut().enumerate() {
                    let g = gain[i];
                    if (g - 1.0).abs() > 1e-3 {
                        moved += 1;
                        lifted_db_sum += f64::from(20.0 * g.log10());
                    }
                    *v *= g;
                }
                let mean_db =
                    if moved > 0 { lifted_db_sum / moved as f64 } else { 0.0 };
                tracing::info!(
                    "RVC cover: input envelope {} ({win_ms} ms window, ref p{ref_pct:.0} {refv:.5}, cap {cap_db:.1} dB, floor {floor_db:.1} dB, slew {slew_db_s} dB/s)                      -- {moved}/{} samples scaled, mean {mean_db:+.2} dB (S165)",
                    if two_sided { "flattened BOTH ways (rejected arm)" } else { "lifted (one-sided)" },
                    audio_f.len()
                );
            }
        }
    }
    let audio_f = audio_f;

    let (tier, _tier_lock) = tier_for_this_song(m.engine, m.voice_session);
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
    // ⭐⭐⭐⭐⭐ S165 —— **修 RMVPE 的八度错**(见 [`fix_octave_inplace`])。
    //
    // ⛔ 位置是承重的:必须在 `*= ratio` **之前** —— 判据要拿 `f0` 去对**原始音频**上的谱峰,
    //    乘过 `f0_shift` 之后两者就对不上了。
    // ⛔ 也必须在 `f0` 上(而不是下面截断后的 `pitchf`)—— donor 那一遍会重跑这整段,
    //    自递归继承的是这里的结果。
    // ⚙ 出厂开;`UTAI_COVER_OCTAVE_FIX=0` 关掉 ⇒ 逐位回到今天。
    // ⭐ S165 §106 —— **先补 RMVPE 漏掉的长音**(见 [`fill_voiced_holes_inplace`])。
    // ⛔ 必须在八度修复**之前**:补回来的帧要和其余帧一起过 Viterbi。
    // ⚙ 出厂开;`UTAI_COVER_FILL_HOLES=0` 关掉 ⇒ 逐位回到今天。
    let mut fill_marks = vec![false; f0.len()];
    if !matches!(std::env::var("UTAI_COVER_FILL_HOLES").as_deref(), Ok("0")) {
        let k = fill_voiced_holes_marked(&mut f0, &audio_pad, WINDOW, &mut fill_marks);
        if k > 0 {
            tracing::info!(
                "RVC f0: filled {k} frame(s) RMVPE dropped mid-note (autocorr peak >= 0.75) (S165)"
            );
        }
    }
    if !matches!(std::env::var("UTAI_COVER_OCTAVE_FIX").as_deref(), Ok("0")) {
        let k = fix_octave_inplace(&mut f0, &audio_pad, WINDOW);
        if k > 0 {
            tracing::info!(
                "RVC f0: octave-repair lifted {k} / {} frame(s) ({:.1}% of voiced) (S165)",
                f0.len(),
                100.0 * k as f32 / f0.iter().filter(|v| **v > 20.0).count().max(1) as f32
            );
        } else {
            tracing::info!("RVC f0: octave-repair found nothing to lift (S165)");
        }
        // ⭐⭐ S165 —— Viterbi 结构上够不着的【孤立单帧】那一类
        // （机理与为什么必须单独一遍，见 [`fix_octave_outliers_inplace`]）。
        // ⚙ 出厂开；`UTAI_COVER_OCTAVE_OUTLIER=0` 关掉 ⇒ 逐位回到只有 Viterbi 那一遍。
        if !matches!(std::env::var("UTAI_COVER_OCTAVE_OUTLIER").as_deref(), Ok("0")) {
            let o = fix_octave_outliers_inplace(&mut f0, &audio_pad, WINDOW);
            if o > 0 {
                tracing::info!(
                    "RVC f0: octave-outlier pass fixed {o} isolated frame(s) Viterbi cannot reach (S165)"
                );
            }
        }
    }
    // ⭐⭐⭐ S165 —— 把与自己两侧对不上的补洞帧撤掉/搬八度。
    // ⛔⛔ 位置是承重的：必须在八度修复**之后** —— 机理与实测见
    //    [`drop_inconsistent_fills_inplace`]（放在之前会把「ぴゃ」打坏 16 帧）。
    if fill_marks.iter().any(|m| *m) {
        let (d, mv) = drop_inconsistent_fills_inplace(&mut f0, &audio_pad, WINDOW, &fill_marks);
        if d > 0 || mv > 0 {
            tracing::info!(
                "RVC f0: fill-anchor gate dropped {d} and re-octaved {mv} filled frame(s) (S165)"
            );
        }
    }
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
    let mut pitchf: Vec<f32> = f0[..p_len].to_vec();
    // ── S160d —— 孤立单帧无声洞:见 [`fill_isolated_uv_inplace`]。
    // ⛔ 放在 `UTAI_COVER_F0_IN` **之前**:探针注入的那条轨要被逐位照用,不能再被这一刀改。
    if !matches!(std::env::var("UTAI_COVER_FILL_UV").as_deref(), Ok("0")) {
        let k = fill_isolated_uv_inplace(&mut pitchf);
        if k > 0 {
            tracing::info!("RVC f0: filled {k} isolated single-frame unvoiced hole(s) (S160d)");
        }
    }
    // S160 探针 —— `UTAI_COVER_F0_IN=<裸 f32 路径>`:**用外部一条 f0 顶掉 RMVPE 那条**。
    //
    // ⛔ 为什么需要它:S160 在用户那份 +7 素材上看见 `pitchf`(= RMVPE)在副歌上成段
    //   报成真基频的 **1/2**(275.25-277.25 s:源是一个 2 秒的 1180 Hz 长音,pitchf 整整两秒
    //   停在 590 Hz,而 `off.wav` 在 590 Hz 上长出一根源里根本没有的基频),
    //   并在全曲最高那个音(244.79-245.41 s,「ぴゃ」,源实测 ~1415 Hz)上**直接报无声 630 ms**。
    //   要判定「这条 f0 是不是那个『炸』」,唯一干净的台面是**只换 f0、其它一个字节不动**。
    // ⚠ 文件语义 = `UTAI_RANGE_DUMP_COVER_F0` 落的那一份的同一段:裸 LE f32,100 fps,
    //   长度 = 最外层的 `out_frames`,对齐**输出时间轴**,**未乘 f0_shift**。
    // ⭐ 它**整条链都顶**(见 `f0_probe` 的 doc):donor 会重跑 RMVPE 并继承同一批错误。
    // ⛔ 出厂不设 ⇒ 整段跳过 ⇒ 输出逐位不变。
    if f0_probe::is_outermost() {
        if let Ok(p) = std::env::var("UTAI_COVER_F0_IN") {
            let want = audio_f.len() / WINDOW;
            let bytes = std::fs::read(&p)
                .unwrap_or_else(|e| panic!("UTAI_COVER_F0_IN={p:?} 读不了: {e}"));
            assert_eq!(bytes.len() % 4, 0, "UTAI_COVER_F0_IN={p:?} 不是 f32 的整数倍");
            let got = bytes.len() / 4;
            assert_eq!(
                got, want,
                "UTAI_COVER_F0_IN={p:?} 帧数 {got} ≠ 期望 {want}(= 最外层 audio_f.len()/WINDOW,\
                 与 UTAI_RANGE_DUMP_COVER_F0 落的那一份同长)"
            );
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert!(
                v.iter().all(|x| x.is_finite() && *x >= 0.0),
                "UTAI_COVER_F0_IN 里有非有限/负值"
            );
            f0_probe::set_song(v);
        }
    }
    // ⛔ 「臂开着」与「臂做了事」必须分开可查(S129 铁律):下面这条 warn 每一遍都打,
    //    最外层与每一个 donor 各一行,`changed` 为 0 也照打。
    if let Some(song) = f0_probe::song() {
        let ratio2 = 2.0f32.powf(options.f0_shift / 12.0);
        let off = f0_probe::offset();
        let pad_f = t_pad / WINDOW;
        let want = audio_f.len() / WINDOW;
        let hi = (pad_f + want).min(pitchf.len());
        let (mut wrote, mut changed, mut missing) = (0usize, 0usize, 0usize);
        for (k, i) in (pad_f..hi).enumerate() {
            match song.get(off + k) {
                Some(&raw) => {
                    let v = raw * ratio2;
                    if (v - pitchf[i]).abs() > 1e-6 {
                        changed += 1;
                    }
                    pitchf[i] = v;
                    wrote += 1;
                }
                None => missing += 1,
            }
        }
        tracing::warn!(
            "RVC f0 OVERRIDE (depth {}, song frame offset {off}, f0_shift {:+}): \
             wrote {wrote}/{want} frames, {changed} differ, {missing} past end of track",
            if f0_probe::is_outermost() { "outer" } else { "donor" },
            options.f0_shift,
        );
    }
    let pitchf = pitchf;
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
    // S85 dead-only(cover;memory S85 七轮): 整曲平移退役——只有模型「连音高都发不出」的
    // 持续区域被局部救援,深度=该区域自己的最小落点(与 score 同一套判据/搜索/拼接器)。
    // 计划在未 pad 网格上出(与输出时间轴对齐;pitchf 已含用户 f0_shift=模型将唱的音高);
    // 死区 donor 在装配+后处理完成后以 run_pipeline 自递归渲染(与 base 全链同构)。
    let out_frames = audio_f.len() / WINDOW;
    let pad_f = t_pad / WINDOW;
    let range_jobs: Vec<super::vocal_range::DeadJob> = match &range {
        Some(r) => {
            let pf_out = &pitchf[pad_f..(pad_f + out_frames).min(pitchf.len())];
            // S159l —— `UTAI_RANGE_DUMP_COVER_F0=<path>`:把**计划器真正看到的那条 f0**落一份裸 f32。
            // ⛔ 为什么需要它:S159l 的「边外扩到清音帧」在真素材上 96 段里有 68 段**一动没动**,
            // 而我用声学方法量出来的「边到最近清音」是 p50 10 ms —— 两个数对不上,
            // 说明**计划器眼里的清音**(`pitchf == 0`)与我量的不是一回事。没有这条出口就只能猜。
            if let Ok(p) = std::env::var("UTAI_RANGE_DUMP_COVER_F0") {
                let mut bytes = Vec::with_capacity(pf_out.len() * 4);
                for v in pf_out {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                match std::fs::write(&p, &bytes) {
                    Ok(()) => tracing::info!("cover f0 dump: {} frames -> {p}", pf_out.len()),
                    Err(e) => tracing::warn!("cover f0 dump {p} failed: {e}"),
                }
            }
            let (jobs, unfixable) = super::vocal_range::cover_dead_plan(pf_out, 100.0, r);
            // 审计恒打印(S83 承诺):无死区也是一个判决;无解区域响亮带位置。
            if jobs.is_empty() && unfixable.is_empty() {
                tracing::info!(
                    "range-extend(cover/rvc, dead-only): no dead regions (usable [{:.0},{:.0}]) — rendering untouched",
                    r.usable.0, r.usable.1
                );
            } else {
                tracing::info!(
                    "range-extend(cover/rvc, dead-only): {} dead region(s), {} unfixable (usable [{:.0},{:.0}])",
                    jobs.len(), unfixable.len(), r.usable.0, r.usable.1
                );
                for j in &jobs {
                    tracing::info!(
                        "range-extend(cover/rvc, dead-only):   region {:.2}s..{:.2}s renders at {:+} st",
                        j.start as f32 / 100.0, j.end as f32 / 100.0, j.shift
                    );
                }
                for &(a, b) in &unfixable {
                    tracing::warn!(
                        "range-extend(cover/rvc, dead-only):   region {:.2}s..{:.2}s has NO landing within ±24 st — rendered broken as-is",
                        a as f32 / 100.0, b as f32 / 100.0
                    );
                }
            }
            jobs
        }
        None => Vec::new(),
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
        let out = vc_chunk(m, chunk, &pitch[pl..ph], &pitchf[pl..ph], sid, spk_mix_dense.as_deref(), options, chunk_idx)?;
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
    let out = vc_chunk(m, chunk, &pitch[s_ix / WINDOW..], &pitchf[s_ix / WINDOW..], sid, spk_mix_dense.as_deref(), options, chunk_idx)?;
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
    // S85 dead-only(cover):死区局部救援——donor 只渲死区邻域切片(S85e 窗口化:每侧
    // DONOR_PAD_SECONDS 真实上下文,同 shift 相邻切片合并;RVC 本就按 ~x_max 秒静音寻点
    // 切块独立转换,切片边界与 base 自身的块边界同类同量,拼接只读窗内、窗深居切片 ≥pad
    // 处)。每切片=整管线自递归(f0_shift+s、range=None、后处理链全同构)+ 切片逆变换回
    // 原音高(fed=移调 pf 同帧段,S82 流式 formant base 保持),嵌回零底全长缓冲交给共享
    // 拼接器。代价从「每 distinct shift 一次整曲」降为「死区占比×pad 展宽」。
    // donor 折进 [0.95, 0.99] 进度带(审查 S85d:曾 noop=长曲 UI 停 95% 读作卡死)。
    if !range_jobs.is_empty() {
        if final_sr % 100 != 0 {
            // resample_sr 理论路径(现无写入者,恒 0):非 100 整除采样率下 fed hop 的整除
            // 截断会让 formant base 时轴漂移——响亮留痕(审查 S85d)。
            tracing::warn!(
                "range-extend(cover/rvc, dead-only): output sr {final_sr} not %100 — formant base grid may drift"
            );
        }
        let pf_out: Vec<f32> =
            pitchf[pad_f..(pad_f + out_frames).min(pitchf.len())].to_vec();
        let base_len = audio_opt.len();
        let spf = base_len as f64 / out_frames.max(1) as f64;
        let pad = (super::vocal_range::DONOR_PAD_SECONDS * final_sr as f32) as usize;
        let in_sr = mono.sample_rate;
        let k_total = {
            let mut s: Vec<i64> = range_jobs.iter().map(|j| j.shift).collect();
            s.sort_unstable();
            s.dedup();
            s.len().max(1)
        };
        let pass = std::cell::Cell::new(0usize);
        // donor 缓冲底=base 快照(审查 S85e):RVC 切片渲染可比跨度短 10-30ms(p_len 帧栅
        // floor+ContentVec 行数上限),曲尾夹紧窗的缺口若垫零会把数字零拼进窗内(咔哒+
        // 静默)——垫 base 则未覆盖样本自动回退旧契约「donor 短了保留 base」。
        let base_snapshot = audio_opt.clone();
        // ⭐⭐⭐⭐⭐ S166c —— **翻唱轨的落点打分**(`UTAI_COVER_SCORE=1`,出厂关)。
        //    ⛔ `notes` 空 ⇒ `scoring == false` ⇒ 候选/打分/修补遍/天花板闸对 cover 全是死代码。
        //    见 [`cover_note_spans`] 与 [`cover_scoring_enabled`]。
        let cover_spans: Vec<super::vocal_range::NoteSpan> =
            if super::vocal_range::cover_scoring_enabled() {
                super::vocal_range::cover_note_spans(&pf_out)
            } else {
                Vec::new()
            };
        if !cover_spans.is_empty() {
            tracing::info!(
                "range-extend(cover/rvc): scoring ON — {} note span(s) from f0 ({} sung)",
                cover_spans.len(),
                cover_spans.iter().filter(|n| n.sung).count()
            );
        }
        super::vocal_range::apply_dead_only_windows_alts(
            &mut audio_opt,
            final_sr,
            out_frames as i64,
            &range_jobs,
            &[],
            &cover_spans,
            // ⭐ 天花板闸要它;`None` ⇒ 那道闸不存在 ⇒ 逐位回到今天。
            if cover_spans.is_empty() { None } else { range },
            false, // cover 无逐渲归一——电平=模型对移调的真实响应,不全局拉平
            // ⚠ 同 sovits cover 臂:第二个参数用不上,cover 要的是样本域带 pad 的合并跨度。
            |s, _own| {
                let spans =
                    super::vocal_range::donor_slice_spans(&range_jobs, s, spf, base_len, pad);
                tracing::info!(
                    "range-extend(cover/rvc, dead-only): donor {s:+} st — {} slice(s), {:.1}s of {:.1}s",
                    spans.len(),
                    spans.iter().map(|(a, b)| b - a).sum::<usize>() as f32 / final_sr as f32,
                    base_len as f32 / final_sr as f32
                );
                let band = pass.get() as f32;
                pass.set(pass.get() + 1);
                let k = 2.0f32.powf(s as f32 / 12.0);
                let pf_shift: Vec<f32> = pf_out.iter().map(|&v| v * k).collect();
                let mut buf = base_snapshot.clone();
                let n_spans = spans.len().max(1);
                for (i, &(a, b)) in spans.iter().enumerate() {
                    let dp = move |p: f32| {
                        let f = (i as f32 + p.clamp(0.0, 1.0)) / n_spans as f32;
                        progress(0.95 + 0.04 * ((band + f) / k_total as f32))
                    };
                    // 输出跨度 → 输入(in_sr)跨度
                    let ia = (a as f64 * in_sr as f64 / final_sr as f64) as usize;
                    let ib = ((b as f64 * in_sr as f64 / final_sr as f64).ceil() as usize)
                        .min(mono.samples.len());
                    if ib <= ia {
                        continue;
                    }
                    // S159q —— `UTAI_COVER_DONOR_PAD_MS=<ms>`(0 = 关 = 今天):给 donor 的切片
                    // **两侧各加一段真实上下文**,渲完再切回来。
                    //
                    // ⛔ 为什么怀疑这里:用户 2026-08-21 指出「谱面轨 +7 渲得出来 ⇒ 模型做得到,
                    // 不存在硬阻碍」。两条车道在 donor 上的差别正是:谱面轨**整曲**移调重渲
                    // (完整上下文),cover **逐小片**递归跑整条 `run_pipeline` —— 那一片里会
                    // **重新 16k 重采样 + 重新抽 ContentVec + 重新跑 RMVPE**,而实测切片中位只有
                    // 0.50 s、53 段短于 0.5 s。RMVPE / ContentVec 在那种长度上远不如整曲可靠。
                    // ⚠ 这条能同时解释用户报的三件事:谐波碎 · 「ぴゃ」糊成一团 · **跳八度**。
                    // ⛔ 出厂 0 = 逐位不变;先量,别在量出来之前翻默认。
                    let pad_in = ((cover_donor_pad_ms() as f64 / 1000.0) * in_sr as f64) as usize;
                    let ja = ia.saturating_sub(pad_in);
                    let jb = (ib + pad_in).min(mono.samples.len());
                    let slice_in = crate::audio::AudioBuffer::new_mono(
                        mono.samples[ja..jb].to_vec(),
                        in_sr,
                    );
                    let mut donor_opts = options.clone();
                    donor_opts.f0_shift += s as f32;
                    // S160 探针:告诉这一遍它在整曲时间轴上从哪一帧开始(见 `f0_probe`)。
                    // ⛔ RAII,`donor` 渲完就还原 —— 出厂不设 env 时它什么也不影响。
                    let _slice_off =
                        f0_probe::SliceOffset::set((ja as f64 * 100.0 / in_sr as f64).round() as usize);
                    let donor = run_pipeline(m, &slice_in, &donor_opts, None, &dp, cancel)?;
                    drop(_slice_off);
                    // 余量渲完就切掉:输出侧的前缀 = (ia − ja) 换算到 `final_sr`。
                    let donor = if pad_in == 0 {
                        donor
                    } else {
                        let r = donor.sample_rate as f64 / in_sr as f64;
                        let cut = ((ia - ja) as f64 * r).round() as usize;
                        let want = ((ib - ia) as f64 * r).round() as usize;
                        let lo = cut.min(donor.audio.len());
                        let hi = (cut + want).min(donor.audio.len());
                        SynthesisResult { audio: donor.audio[lo..hi].to_vec(), ..donor }
                    };
                    // fed f0 取同帧段(100fps;±1 帧取整偏移 ≪ 100ms sticky base 窗)
                    let fa = ((a as f64 / spf) as usize).min(pf_shift.len());
                    let fb = (((b as f64 / spf).ceil() as usize) + 1).min(pf_shift.len());
                    let inv = super::vocal_range::apply_inverse(
                        donor.audio,
                        donor.sample_rate,
                        s,
                        options.range_formant_follow,
                        Some((&pf_shift[fa..fb], (donor.sample_rate as usize / 100).max(1))),
                        // ⛔ S162 —— **cover 车道不吃谱倾斜(`tilt = 0`)**。
                        // 这条引擎两条车道共用,而 tilt 的表是在**谱面轨**素材上拟的、
                        // **cover 上一个读数都没有**;而 cover 的深救援反而更重
                        // (S160 的计划输出:|s|≥8 占救援总时长 **78.1%**,最深 **−18**,
                        //  已超出表的范围)⇒ 按「证明有效果才翻默认」的规矩,不跟着翻。
                        0.0,
                    )
                    .map_err(UtaiError::Inference)?;
                    let n = inv.len().min(base_len - a);
                    buf[a..a + n].copy_from_slice(&inv[..n]);
                }
                Ok(buf)
            },
        )?;
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
        // cover 路不做逐渲归一(电平差 = 模型对移调的真实响应)⇒ 没有这个峰。
        pre_norm_peak: None,
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
    // index_weights = None: the cover path stays byte-identical to pre-S84 retrieval.
    vc_decode(m, feats, pitch, pitchf, sid, spk_mix, options, chunk_idx, chunk.len() / WINDOW, None)
}

/// Decode one chunk's already-extracted ContentVec (50 fps) through the RVC net_g → UNtrimmed wav.
/// Verbatim tail of `vc_chunk` (feats0/protect basis → index blend → 2x upsample → protect blend →
/// rnd noise → net_g), pulled out so the ② score render (`render_score_rvc`) drives the SAME
/// retrieval/protect path. `windows_bound` = the caller's frame cap (cover: chunk.len()/WINDOW;
/// score: usize::MAX = feats/pitch bound only). Byte-identical to the inlined version — proven at the
/// source (scratchpad/verbatim_check.py) since the net_g graph is stochastic (RandomNormalLike/Uniform).
///
/// `index_weights` (S84 B 刀, ② score path only): per-50fps-frame retrieval weight — the S84
/// measurement showed retrieval PULLS transitional fast-run cv toward wrong neighbours (ま's /a/
/// F1 619 with retrieval vs 938 without, closed-vowel直拉; the ko1 dropout amplifier). Effective
/// per-row ratio = index_ratio·w, applied as an exact post-lerp (blended = orig + ratio·(ret−orig)
/// ⇒ orig + w·(blended−orig) = orig + w·ratio·(ret−orig)). `None` (the cover path and every
/// pre-S84 caller) = byte-identical to the unweighted path. Rows beyond the slice default to 1.
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
    index_weights: Option<&[f32]>,
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
            let orig = index_weights.map(|_| feats.clone());
            feats = knn_blend(&feats, &index.knn, options.index_ratio, options.l2_normalize);
            if let (Some(w), Some(orig)) = (index_weights, orig) {
                for (r, mut row) in feats.rows_mut().into_iter().enumerate() {
                    let wr = w.get(r).copied().unwrap_or(1.0).clamp(0.0, 1.0);
                    if wr < 1.0 {
                        for (x, &ov) in row.iter_mut().zip(orig.row(r).iter()) {
                            *x = ov + wr * (*x - ov);
                        }
                    }
                }
            }
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

    // S165 -- `UTAI_COVER_DUMP_TENSORS=<dir>`: drop the decoder's INPUTS for this chunk.
    //
    // Why this exists: ten hypotheses about the cover break-ups were all killed by their own
    // negative controls (S165 §86). The only fact left is that the envelope periodicity of the
    // 18 break-up segments is 0.49 while the source is 0.83 and every normal cover segment is
    // 0.82 -- same lane, same parameters. So the cause has to be in what those 18 positions
    // feed the decoder, and the only way to see that is to dump it instead of guessing.
    //
    // Files (raw little-endian f32): `<dir>/c<chunk>_feats.f32` [T, dim],
    // `<dir>/c<chunk>_pitchf.f32` [T], `<dir>/c<chunk>_meta.txt` (T, dim, offsets).
    // Unset => not a single branch runs => byte-identical.
    if let Ok(dir) = std::env::var("UTAI_COVER_DUMP_TENSORS") {
        let d = std::path::PathBuf::from(&dir);
        if std::fs::create_dir_all(&d).is_ok() {
            let w = |name: &str, v: &[f32]| {
                let mut b = Vec::with_capacity(v.len() * 4);
                for x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                let _ = std::fs::write(d.join(format!("c{chunk_idx}_{name}.f32")), &b);
            };
            w("feats", &feats.iter().copied().collect::<Vec<f32>>());
            w("pitchf", pitchf);
            let _ = std::fs::write(
                d.join(format!("c{chunk_idx}_meta.txt")),
                format!("T={} dim={} p_len={}
", feats.nrows(), feats.ncols(), p_len),
            );
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
    /// S165 §107/§109 —— 补洞的搜索上限**保持在 `f0_to_coarse` 装得下的范围内**。
    ///
    /// ⛔ 抬高它试过,**实测更差,两条都输**。当时的推理听着很硬:那处漏唱的真基频约
    /// 1420 Hz,上限 1100 时自相关锁到次谐波、写进 470 Hz(低了十二度);读出真值还能让
    /// 这一帧撞 `RVC_COARSE_MAX_HZ` 那条死因、被正经救援下来。于是抬到 1600、渲、量——
    ///
    /// 漏掉的是:**补进去的值是喂给模型的【条件】,不是输出音高本身**。在那处实测
    /// (谱峰法,源中位 1392 Hz):
    ///     上限 1100   输出 1384 Hz  −10 音分     全曲坏帧 4.32 %
    ///     上限 1600   输出 1426 Hz  +42 音分              4.56 %
    /// 写 470 Hz 并不会让模型唱低十二度——那些帧只是跳过死因判定,模型照样跟着源唱;
    /// 写 1454 Hz 反而触发 −15 度深救援,音准和干净度**一起变差**。
    ///
    /// ⚠ 这条判据钉的是**上限本身**,以及「读出来的东西装得进 coarse 表」这个约束。
    #[test]
    fn the_fill_search_stays_inside_what_coarse_can_carry() {
        const SR: f32 = 16_000.0;
        const HOP: usize = 160;
        let n = HOP * 60 + 1600;
        // 700 Hz:装得下,必须读准
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 700.0 * i as f32 / SR).sin() * 0.3)
            .collect();
        let mut f0 = vec![0.0f32; 50];
        assert!(super::fill_voiced_holes_inplace(&mut f0, &tone, HOP) >= 40);
        let mut got: Vec<f32> = f0.iter().copied().filter(|v| *v > 20.0).collect();
        got.sort_by(f32::total_cmp);
        let med = got[got.len() / 2];
        assert!((med - 700.0).abs() < 25.0, "装得下的音必须读准,读到 {med:.1}");
        // ⭐ 承重:凡是写进 f0 的值,都必须落在 coarse 表装得下的范围内 —— 否则这一帧会被
        //    判死并拖去深救援,而实测那样音准和干净度一起变差。
        let ceiling = super::super::vocal_range::RVC_COARSE_MAX_HZ;
        let high: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1454.0 * i as f32 / SR).sin() * 0.3)
            .collect();
        let mut f0h = vec![0.0f32; 50];
        super::fill_voiced_holes_inplace(&mut f0h, &high, HOP);
        for v in f0h.iter().copied().filter(|v| *v > 20.0) {
            assert!(
                v <= ceiling,
                "补进去的 {v:.1} Hz 超过了 coarse 上限 {ceiling:.0} ——                  那会把这一帧推进深救援,实测音准与干净度一起变差(§109)"
            );
        }
    }

    /// S165 §106 —— **短洞不归这一刀管**,交给 `fill_isolated_uv_inplace`。
    ///
    /// 第一版没有这条界,实测在全曲补出 240 段:其中 **231 段(96 %)是 10-30 ms 的碎渣**,
    /// 而它真正为之存在的那个漏唱只有 1 段(730 ms)。往全曲撒碎补丁,正是让一个清辅音
    /// 间隙凭空得到音高的方式 —— 用户报的 0:57.540 在那一版里从 26 % 退到 30 %。
    #[test]
    fn short_gaps_are_left_to_the_isolated_hole_pass() {
        const SR: f32 = 16_000.0;
        const HOP: usize = 160;
        let n = HOP * 80 + 1600;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 700.0 * i as f32 / SR).sin() * 0.3)
            .collect();
        // 出厂门槛 5 帧:4 帧的洞不许碰,5 帧的要补 —— 两边都钉住,否则门槛可以随便滑。
        // ⛔ S165 —— 轨上的值必须与音频**一致**（都是 700 Hz）。
        //    第一版写的是 440,而音频是 700 —— 差一个五度。那时无所谓（补洞不看邻居）,
        //    加了锚闸（[`hole_anchor`]）之后它就是一个**洞里的音频与两侧对不上**的夹具
        //    ⇒ 闸拒填是对的,错的是夹具。这条判据要钉的是**洞长门槛**,不是锚闸。
        let hole_of = |len: usize| {
            let mut f0 = vec![700.0f32; 60];
            for v in f0.iter_mut().skip(20).take(len) {
                *v = 0.0;
            }
            f0
        };
        let mut short = hole_of(4);
        assert_eq!(
            super::fill_voiced_holes_inplace(&mut short, &tone, HOP),
            0,
            "4 帧的洞归 fill_isolated_uv_inplace 管,这一刀不许碰"
        );
        assert!(short[20..24].iter().all(|v| *v == 0.0), "短洞必须原样留着");
        let mut long = hole_of(5);
        assert!(
            super::fill_voiced_holes_inplace(&mut long, &tone, HOP) > 0,
            "刚好够门槛(5 帧 = 50 ms)的洞必须补"
        );
        // 门槛本身也钉住,免得默认值被悄悄挪走
        assert_eq!(super::min_hole_frames(), 5, "出厂最小洞长应当是 5 帧 = 50 ms");
    }

    /// S165 §106 —— **RMVPE 把长音整段漏掉时,用自相关把它补回来**;而真清音一帧都不许碰。
    ///
    /// 这两件事必须同时成立才安全,所以判据把两边都钉上:补错了会在清辅音上凭空造出音高,
    /// 比留着洞更糟。分离度是量出来的:洞里自相关峰 0.962,真清音帧 p90 才 0.396(n=122)。
    #[test]
    fn a_dropped_held_note_is_filled_and_real_silence_is_left_alone() {
        const SR: f32 = 16_000.0;
        const HOP: usize = 160; // 100 fps
        let n = HOP * 60 + 1600;
        // ⑴ 一段干净的 700 Hz 持续音,而 f0 全是 0(= RMVPE 整段漏掉)
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 700.0 * i as f32 / SR).sin() * 0.3)
            .collect();
        let mut f0 = vec![0.0f32; 50];
        let k = super::fill_voiced_holes_inplace(&mut f0, &tone, HOP);
        assert!(k >= 40, "整段周期音必须被补回来,只补了 {k} 帧");
        let got: Vec<f32> = f0.iter().copied().filter(|v| *v > 20.0).collect();
        let med = {
            let mut v = got.clone();
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        };
        assert!(
            (med - 700.0).abs() < 25.0,
            "补进去的音高要贴着真值(700 Hz),读到 {med:.1}"
        );
        // ⑵ ⭐⭐ 阴性对照:**贴着闸**的半周期信号,一帧都不许补。
        //
        // ⛔ 第一版用纯白噪,结果把闸从 0.75 一路放到 0.2 它都不被补 —— 那条对照
        //    离闸太远,**测不出闸有没有用**。真实的清擦音自相关峰 p90 = 0.396、
        //    p99 = 0.673(n=122 帧实测),所以对照必须落在那个量级上才有意义。
        let mut rng = 0x1234_5678u32;
        let mut white = || {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (rng >> 16) as f32 / 32_768.0 - 1.0
        };
        // 正弦 : 噪声 ≈ 1.2 : 1 ⇒ 自相关峰落在 0.5-0.7,恰好在闸下面一点
        let halfp: Vec<f32> = (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 700.0 * i as f32 / SR).sin() * 0.30
                    + white() * 0.25
            })
            .collect();
        // 先自证夹具:它的峰确实贴着闸,否则这条对照又是空的
        let peak = {
            let w = &halfp[..640];
            let mean = w.iter().sum::<f32>() / 640.0;
            let e0: f32 = w.iter().map(|v| (v - mean) * (v - mean)).sum();
            let mut best = 0.0f32;
            for lag in 14..80 {
                let acc: f32 =
                    (0..640 - lag).map(|k| (w[k] - mean) * (w[k + lag] - mean)).sum();
                best = best.max(acc / e0);
            }
            best
        };
        assert!(
            (0.45..0.72).contains(&peak),
            "夹具本身要贴着闸(实测清音 p90 0.396 / p99 0.673),现在是 {peak:.3} ——              离闸太远的对照测不出闸有没有用"
        );
        let mut f0n = vec![0.0f32; 50];
        let kn = super::fill_voiced_holes_inplace(&mut f0n, &halfp, HOP);
        assert_eq!(kn, 0, "周期性只到 {peak:.3} 的信号不许被补,却补了 {kn} 帧");
        // ⑶ 已经有音高的帧不许被改写(它只补洞,不重估)
        // ⛔ S165 —— 这一帧的值要与音频**自洽**（690 贴着 700）。
        //    第一版写 123 Hz —— 与 700 Hz 的音频差 5.7 倍 ⇒ 锚闸会把两侧的洞**整段拒掉**,
        //    于是这条判据退化成「什么都没填所以也没被改写」= **空判据**。
        //    ⇒ 下面多钉一条：周围必须真的被填了。
        let mut f0k = vec![0.0f32; 50];
        f0k[10] = 690.0;
        let kk = super::fill_voiced_holes_inplace(&mut f0k, &tone, HOP);
        assert!((f0k[10] - 690.0).abs() < f32::EPSILON, "已有音高的帧不许动");
        assert!(kk >= 30, "周围必须真的被填了（只填了 {kk} 帧）—— 否则上一条断言是空的");
        // ⑷ 静音不许补(能量为零时自相关没有意义)
        let mut f0s = vec![0.0f32; 50];
        let silence = vec![0.0f32; n];
        assert_eq!(
            super::fill_voiced_holes_inplace(&mut f0s, &silence, HOP),
            0,
            "静音里不许凭空造音高"
        );
    }

    /// S165 §103 —— **session 还活着就复用它那一档 tier**,别再去问"内存够不够"。
    ///
    /// tier 决定 chunk 长度,chunk 长度决定输入 shape,而**新 shape 才要付 DirectML
    /// first-shape ticket(1.7-3.4 GB)**。session 已经举着一张了 ⇒ 那个 shape 不用再买。
    /// 每首歌重问一次、看见内存紧就换更短的 chunk,恰恰是**再买一张** —— 这就是把这台机器
    /// 从 32 s 一路推到 10 s、缺陷率翻倍的那个循环。
    ///
    /// ⚠ 复用**不分配任何东西**,它只是拒绝再分配一次,所以这里没有放宽任何门槛。
    /// session 没了(被驱逐 / 换设备)⇒ ticket 也没了 ⇒ 必须重新按当时真实的内存选。
    #[test]
    fn a_live_session_keeps_the_tier_it_already_paid_for() {
        let sid = "test-session-A";
        super::tier_memo_clear();
        assert!(super::tier_memo_get(sid).is_none(), "清空后不该记得任何东西");

        // ⑴ 第一次:选定并记住
        super::tier_memo_set(sid, &super::CHUNK_TIERS[1]);
        assert_eq!(
            super::tier_memo_get(sid).map(|t| t.x_max),
            Some(super::CHUNK_TIERS[1].x_max),
            "同一个 session 必须拿回同一档"
        );

        // ⑵ 换了 session ⇒ 不许串用(不同模型/不同设备是不同的 ticket)
        assert!(
            super::tier_memo_get("test-session-B").is_none(),
            "别的 session 不许命中这条记忆"
        );

        // ⑶ session 没了 ⇒ 记忆必须跟着没
        super::tier_memo_clear();
        assert!(
            super::tier_memo_get(sid).is_none(),
            "session 被驱逐之后还记着旧 tier,就会拿一张早就退掉的票去下结论"
        );

        // ⑷ 安全阀存在(出厂开;=0 关)
        assert!(super::tier_memo_enabled(), "出厂应当是开的");
    }

    /// S165 §100 —— **一首歌只选一次 chunk tier**。
    ///
    /// 这条判据挡的是一整天被毁掉的 A/B:donor 死区自递归调用 `run_pipeline`,
    /// 从前每递归一次就重选一次 tier(实测一条臂里 61 次)。而 WDDM 把显存算进 commit、
    /// 一次渲染占掉 7 GB,于是后面几十次全看见"内存紧"⇒ 降 tier ⇒ 换新输入 shape
    /// ⇒ 再付一张 1.7-3.4 GB 的 DirectML first-shape ticket ⇒ 更紧 ⇒ 再降。
    /// 实测后果:tier 32s 的臂坏帧率 6.2-6.4%,tier 19s 的 13.0-13.6% —— **翻倍**。
    ///
    /// ⚠ 第三段(锁必须还回去)不是形式主义:锁不释放的话,**下一首歌会沿用上一首的 tier**,
    /// 那等于把这个 bug 换了个方向再犯一次。
    #[test]
    fn one_song_picks_its_chunk_tier_exactly_once() {
        use std::cell::Cell;
        let calls = Cell::new(0usize);
        let pick_small = || {
            calls.set(calls.get() + 1);
            &super::CHUNK_TIERS[3] // 最小的那档,和默认档明显不同
        };
        {
            // ⑴ 最外层:真的选一次
            let (outer, _g0) = super::tier_for_this_song_with(pick_small);
            assert_eq!(calls.get(), 1, "最外层应当选一次");
            assert_eq!(outer.x_max, super::CHUNK_TIERS[3].x_max);
            // ⑵ donor 递归(可以嵌很多层):一次都不许再选,而且拿到同一个 tier
            for depth in 0..5 {
                let (inner, _g) = super::tier_for_this_song_with(|| {
                    panic!("donor 递归第 {depth} 层又去选 tier 了 —— 降级螺旋会回来");
                });
                assert_eq!(
                    inner.x_max, outer.x_max,
                    "donor 递归必须照用最外层定好的 tier"
                );
            }
            assert_eq!(calls.get(), 1, "整首歌自始至终只许选一次");
        }
        // ⑶ 整首歌渲完 ⇒ 锁必须还回去,下一首重新按当时的内存选
        let big = || {
            calls.set(calls.get() + 1);
            &super::CHUNK_TIERS[0]
        };
        let (next_song, _g) = super::tier_for_this_song_with(big);
        assert_eq!(calls.get(), 2, "下一首必须重新选,而不是沿用上一首的");
        assert_eq!(
            next_song.x_max,
            super::CHUNK_TIERS[0].x_max,
            "下一首应当拿到它自己选的那一档"
        );
    }


    /// S165 §102 —— 钉住输入包络提升**出厂是关的**。
    ///
    /// 它曾经开了几个小时,依据是一组**假对照**:基线臂恰好跑在 19 s chunk tier、
    /// 处理臂恰好跑在 32 s,那个「坏段 123→27」全是 tier 差出来的(见 `tier_for_this_song`)。
    /// 四条臂钉死在同一个 tier、开关各渲两遍之后:
    ///     关 5.88% / 5.85%      开 6.33% / 6.34%
    /// 同配置两遍只差 0.01-0.03 pp,而这一刀稳定要 +0.45 pp ⇒ **十五倍于噪声底,且可复现**。
    /// ⭐ 用户当时听完就说了「跟八度修复那版比基本没什么明显进步」——**耳朵是对的,尺子是脏的**。
    ///
    /// ⚠ 第二条断言(显式 `0` 必须真的关掉)不是形式主义:阴性臂要是被默认值悄悄吃掉,
    /// 以后就再也没法证伪这把刀了 —— S129「一条从没被执行过的错误分支就是一条空判据」。
    #[test]
    fn the_input_envelope_lift_ships_off_after_measuring_worse() {
        // ⑴ 没设环境变量 = 出厂 ⇒ **关**(同 tier 四臂实测:开 6.33/6.34% vs 关 5.88/5.85%,
        //    而同配置两遍只差 0.01-0.03 pp ⇒ 这一刀稳定地把结果推坏 0.45 pp)
        assert_eq!(
            super::parse_flatten_ms(None),
            None,
            "出厂必须是关的 —— 想翻回去先拿一组【同 tier】的 A/B 打赢 0.03 pp 的噪声底"
        );
        // ⑵ 显式 0 = 关(阴性臂必须可达)
        assert_eq!(super::parse_flatten_ms(Some("0")), None, "显式 0 必须真的关掉");
        assert_eq!(super::parse_flatten_ms(Some(" 0 ")), None, "带空白的 0 也是关");
        // ⑶ 合法值原样生效
        assert_eq!(super::parse_flatten_ms(Some("250")), Some(250.0));
        assert_eq!(super::parse_flatten_ms(Some("10")), Some(10.0), "下界含在内");
        assert_eq!(super::parse_flatten_ms(Some("1000")), Some(1000.0), "上界含在内");
        // ⑷ 越界/垃圾 ⇒ 关,而**不是**悄悄回落到默认值
        //    (回落会让一个手滑的越界值伪装成「我设了它」,正是我们最怕的那种静默)
        for bad in ["5000", "9", "-1", "abc", "", "NaN", "inf"] {
            assert_eq!(
                super::parse_flatten_ms(Some(bad)),
                None,
                "越界或垃圾值 {bad:?} 必须落到关,不许回落成默认值"
            );
        }
    }

    use super::*;


    /// ⭐⭐⭐ S165 —— 补洞的【锚闸】([`drop_inconsistent_fills_inplace`])。
    ///
    /// ⛔ 病因:`fill_one` 在 `[F_MIN, F_MAX]` 里**自己找自相关峰、完全不看洞的两侧**,
    ///    而在衰减尾/音节间隙上,两倍周期的自相关常常比真周期还高 ⇒ 它锁到次谐波。
    ///    实测这首歌 163 段有锚的填充:`填值/锚` 有 **119 段(73 %)落在 0.95-1.05**,
    ///    却拖着一条低尾巴(0.41 / 0.49 / 0.52 / 0.54 / **0.55 = 4:28.310** / **0.60 = 4:28.650** …),
    ///    **它们的锚全在 985-1206 Hz**,即贴着或高于 `F_MAX` 能表示的 1143 Hz。
    ///    用户 2026-08-26 报的那两条尾巴就是它。
    ///
    /// ⛔⛔ **承重的一半是「必须跑在八度修复之后」。** 第一版把闸写在 `fill_one` 里,
    ///    结果**把「ぴゃ」那个 730 ms 漏唱打坏了 16 帧**(1391→0、1455→364)——
    ///    因为补洞跑在八度修复之前,那时的 `f0` 本身就整段偏低一个八度
    ///    (那一带 RMVPE 读 ~710,靠 Viterbi 整段抬成 1412)。
    ///    ⇒ 这条判据必须**同时**钉住「修得对」和「ぴゃ 一帧不动」。
    ///
    /// 五条:
    /// ⑴ **正常洞**(音频与两侧一致)—— 一帧都不许被撤。
    /// ⑵ **「ぴゃ」形**(真基频高于搜索上限,整段被读低一个八度)—— 一帧都不许被撤。
    /// ⑶ **次谐波填充**(两侧 1100,填 619)—— 必须撤掉或搬回对得上的八度。
    /// ⑷ **锚不算自己**:`run_anchor` 不许把补洞自己写的帧当参照。
    /// ⑸ 旋钮关掉 ⇒ 一帧不动。
    #[test]
    fn a_fill_that_disagrees_with_its_own_edges_is_dropped_but_pika_is_untouched() {
        const SR: f32 = 16_000.0;
        const HOP: usize = 160;
        let n = HOP * 60 + 1600;
        let tone = |f: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR;
                    let mut v = 0.0f32;
                    for k in 1..=5 {
                        let kf = f * k as f32;
                        if kf < SR / 2.0 * 0.9 {
                            v += (2.0 * std::f32::consts::PI * kf * t).sin() / k as f32;
                        }
                    }
                    v * 0.3
                })
                .collect()
        };
        // 帧 20..30 是被补出来的;两侧是真浊音。
        let make = |edge: f32, fill: f32| -> (Vec<f32>, Vec<bool>) {
            let mut f0 = vec![edge; 50];
            let mut m = vec![false; 50];
            for k in 20..30 {
                f0[k] = fill;
                m[k] = true;
            }
            (f0, m)
        };

        // ⑴ 正常:两侧 700,补的也是 700 ⇒ 一帧不动。
        let a700 = tone(700.0);
        let (mut f0, m) = make(700.0, 700.0);
        let before = f0.clone();
        let (d, mv) = drop_inconsistent_fills_inplace(&mut f0, &a700, HOP, &m);
        assert_eq!((d, mv), (0, 0), "正常洞被动了({d} 撤 / {mv} 搬)");
        assert_eq!(f0, before);

        // ⑵ ⛔ 「ぴゃ」形:真基频 1454 高于搜索上限 ⇒ **整段**(补的和两侧)都是 727,
        //    自洽 ⇒ 一帧都不许被撤。这一条不成立,这一整刀就是净损失。
        let ahigh = tone(1454.0);
        let (mut pika, mp) = make(727.0, 727.0);
        let pb = pika.clone();
        let (dp, mvp) = drop_inconsistent_fills_inplace(&mut pika, &ahigh, HOP, &mp);
        assert_eq!(
            (dp, mvp),
            (0, 0),
            "「ぴゃ」形被动了({dp} 撤 / {mvp} 搬)—— 整段一致偏低一个八度是**对的**,\
             那是后面 Viterbi 的活;在这里动它就是 S165 第一版那个 16 帧的回归"
        );
        assert_eq!(pika, pb);

        // ⑶ 次谐波填充:两侧是真的 1100,补进来的是 550(一半)⇒ 必须被处理掉。
        let a1100 = tone(1100.0);
        let (mut bad, mb) = make(1100.0, 550.0);
        let (db, mvb) = drop_inconsistent_fills_inplace(&mut bad, &a1100, HOP, &mb);
        assert_eq!(db + mvb, 10, "10 帧次谐波填充只处理了 {db}+{mvb} 帧");
        for k in 20..30 {
            assert!(
                bad[k] <= 20.0 || (bad[k] > 1100.0 / 1.4 && bad[k] < 1100.0 * 1.4),
                "第 {k} 帧留下了 {v:.0} Hz —— 既没撤掉也没搬到对得上的八度",
                v = bad[k]
            );
        }
        assert!(bad[..20].iter().all(|v| (*v - 1100.0).abs() < f32::EPSILON), "两侧不许被动");

        // ⑷ ⛔ 锚不许把补洞自己写的帧算进去 —— 否则 550 会给 550 当参照,闸永远不开。
        let (f0b, mb2) = make(1100.0, 550.0);
        assert!(
            (run_anchor(&f0b, &mb2, 20, 30) - 1100.0).abs() < 1.0,
            "锚读到 {a:.0},说明它把补洞自己写的帧当成参照了",
            a = run_anchor(&f0b, &mb2, 20, 30)
        );

        // ⑸ ⛔⛔ **锚要能跨过洞里【没被填】的那些无声帧**。
        //    S165 实测:4:28.650 那个洞 `fill_one` 只写了末帧 552,左边紧邻是同一个洞里
        //    没被填的空档 ⇒ 第一版的 `run_anchor` 一碰无声就停 ⇒ 取不到锚 ⇒ 整段被跳过,
        //    那条尾巴原封不动留着。⇒ 标记的是**整个洞**,锚要跳过洞内的无声继续往外找。
        let mut f0h = vec![1100.0f32; 50];
        let mut mh = vec![false; 50];
        for k in 20..30 {
            mh[k] = true;
            f0h[k] = 0.0; // 洞里大部分没被填
        }
        f0h[29] = 550.0; // 只有末帧写上了(而且是次谐波)
        assert!(
            (run_anchor(&f0h, &mh, 20, 30) - 1100.0).abs() < 1.0,
            "洞里有没填的空档时锚读到 {a:.0} —— 它应当跳过洞内的无声,继续往外找真浊音",
            a = run_anchor(&f0h, &mh, 20, 30)
        );
        let (dh, mvh) = drop_inconsistent_fills_inplace(&mut f0h, &a1100, HOP, &mh);
        assert_eq!(dh + mvh, 1, "洞里那一帧次谐波没被处理({dh} 撤 / {mvh} 搬)");

        // ⑹ ⛔ 判的是**这一段整体**:一条单调下滑的坡,不许让卡在带边的第一帧溜过去。
        //    实测 4:28.310 那个洞填的是 856 667 619 571 593(锚 ≈ 1128)——
        //    逐帧判会把 856 留下,坡还在。
        let mut ramp = vec![1100.0f32; 50];
        let mut mr = vec![false; 50];
        for (j, v) in [856.0f32, 667.0, 619.0, 571.0, 593.0].iter().enumerate() {
            ramp[20 + j] = *v;
            mr[20 + j] = true;
        }
        let (dr, mvr) = drop_inconsistent_fills_inplace(&mut ramp, &a1100, HOP, &mr);
        assert_eq!(dr + mvr, 5, "下滑坡只处理了 {dr}+{mvr}/5 帧 —— 856 大概率从带边溜过去了");
        assert!(
            ramp[20] <= 20.0 || (ramp[20] > 1100.0 / 1.4 && ramp[20] < 1100.0 * 1.4),
            "坡的第一帧留下了 {v:.0} Hz",
            v = ramp[20]
        );

        // ⑸ 门槛写字面量,不许拿常量自己跟自己比。
        assert!(
            FILL_ANCHOR_RATIO > 1.2 && FILL_ANCHOR_RATIO < 1.7,
            "锚带 {FILL_ANCHOR_RATIO} 掉出了实测区间:p10=0.818 / p90=1.196 要放行,\
             而 0.5 与 2.0 两个八度要拦住"
        );
        assert_eq!(FILL_ANCHOR_FRAMES, 5);
    }

    /// ⭐⭐⭐ S165 —— 【孤立八度离群帧】的补刀([`fix_octave_outliers_inplace`])。
    ///
    /// ⛔⛔ 这条判据的**承重点是「Viterbi 修不动」那一半** —— 没有它,这一整遍看起来
    ///    只是 [`fix_octave_inplace`] 的重复,下一个人会顺手把它删掉。
    ///    机理:换档罚 12 dB 要付**两次**(换上去、再换回来),而单帧的证据够不到 24 dB;
    ///    岛**末**帧还更糟——下一帧无声,`obs = [0, 1e9, 1e9]` 把状态硬拽回 `×1`。
    ///
    /// ⛔⛔ **夹具必须先复现真实条件**(第一版就栽在这):纯合成音上证据差 **73.7 dB**,
    ///    Viterbi 一下就修了 ⇒ 那个夹具证明不了任何事,只证明「合成音太干净」。
    ///    真实的 `4:28.640` 是 `prom(559) = 10.17` / `prom(1118) = 25.58` / 差 **15.41**。
    ///    ⇒ 这里掺 `NOISE_AMP` 的确定性噪声底把差压到 **≈15 dB**,并**在判据里当场量一遍**
    ///    确认它落在 `(12, 24)` —— 12 = 一次换档罚,24 = 单帧要付的两次。
    ///
    /// 五条:
    /// ⑴ **夹具自检**:证据差真的落在 `(12, 24)`。
    /// ⑵ **阳性(岛末)**:真基频 400、只有**最后一帧**报成 200 ⇒ 补刀遍抬回 400。
    /// ⑶ ⛔ **同一份输入上,只跑 Viterbi 抬不动它**。
    /// ⑷ **阴性(真的低音)**:那一帧的音频**本来就是** 200 Hz ⇒ 一帧都不许动。
    /// ⑸ **成段的错不归它管**:5 帧连着报成一半 ⇒ 超过 `OCTAVE_OUTLIER_MAX_RUN` ⇒ 不碰。
    #[test]
    fn an_isolated_halved_frame_is_lifted_where_viterbi_structurally_cannot() {
        const SR: f32 = 16_000.0;
        const HOP: usize = 160; // 100 fps
        // 把证据差压进真实区间的噪声幅度(标定见 doc)。
        const NOISE_AMP: f32 = 0.35;
        let n_frames = 60usize;
        let n = n_frames * HOP + 4 * HOP;
        // 确定性噪声底(LCG;⛔ 不许用真随机 —— 判据要能逐位复现)
        let noise: Vec<f32> = {
            let mut st: u32 = 12345;
            (0..n)
                .map(|_| {
                    st = st.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (st as f32 / 4_294_967_296.0) * 2.0 - 1.0
                })
                .collect()
        };
        // 真基频 `f` 的多谐波音 + 噪声底;`swap` 指定的样本区间换成另一个基频。
        let tone = |f: f32, swap: Option<(usize, usize, f32)>| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR;
                    let f = match swap {
                        Some((a, b, g)) if i >= a && i < b => g,
                        _ => f,
                    };
                    let mut v = 0.0f32;
                    for k in 1..=6 {
                        let kf = f * k as f32;
                        if kf < SR / 2.0 * 0.9 {
                            v += (2.0 * std::f32::consts::PI * kf * t).sin() / k as f32;
                        }
                    }
                    v * 0.3 + NOISE_AMP * noise[i]
                })
                .collect()
        };
        let audio400 = tone(400.0, None);
        let last = n_frames - 1;

        // ⑴ 夹具自检 —— 拿引擎**自己那把尺子**量,不许另写一份。
        let spec = OctaveSpec::new();
        let mag = spec.frame_mag(&audio400, last, HOP).expect("末帧的窗越界了");
        let gap = spec.prominence_db(&mag, 400.0) - spec.prominence_db(&mag, 200.0);
        assert!(
            (12.0..24.0).contains(&gap),
            "夹具没复现真实条件:证据差 {gap:.2} dB 不在 (12, 24) 里 —— \
             低于 12 连补刀遍都不该修,高于 24 则 Viterbi 自己就修了,\
             两种情况下这条判据都证明不了 `fix_octave_outliers_inplace` 存在的必要"
        );

        // ⑵ 岛末单帧:后面必须真的接无声 —— 「岛末」正是它比岛内更难修的原因。
        let mut edge = vec![400.0f32; n_frames];
        edge[last] = 200.0;
        edge.extend(std::iter::repeat_n(0.0f32, 20));
        let mut both = edge.clone();
        let moved = fix_octave_outliers_inplace(&mut both, &audio400, HOP);
        assert_eq!(moved, 1, "岛末那一帧没被补刀遍抬起来(改动 {moved} 帧)");
        assert!((both[last] - 400.0).abs() < 1.0, "抬完没落在 400 Hz 上:{}", both[last]);
        assert!(
            both[..last].iter().all(|v| (*v - 400.0).abs() < 1.0),
            "补刀遍动了它不该动的帧"
        );
        assert!(both[n_frames..].iter().all(|v| *v == 0.0), "补刀遍动了无声帧");

        // ⑶ ⛔ 承重:同一份输入,**只跑 Viterbi 那一遍**抬不动它。
        let mut viterbi_only = edge.clone();
        fix_octave_inplace(&mut viterbi_only, &audio400, HOP);
        assert!(
            (viterbi_only[last] - 200.0).abs() < 1.0,
            "Viterbi 居然抬动了岛末单帧({} Hz)—— 那这一整遍补刀就是多余的,\
             说明换档罚或观测代价被改过,这条判据与 `fix_octave_outliers_inplace` 的\
             存在理由都要重新算",
            viterbi_only[last]
        );

        // ⑷ 阴性:那一帧的音频**真的是** 200 Hz ⇒ 谱证据反向 ⇒ 一帧都不许动。
        //    窗是 40 ms(±320 样本),真低音那一段要足够宽才不会被邻帧的 400 盖住。
        let c = last * HOP;
        let genuine_audio = tone(400.0, Some((c - 400, c + 400, 200.0)));
        let mut genuine = edge.clone();
        let moved_g = fix_octave_outliers_inplace(&mut genuine, &genuine_audio, HOP);
        assert_eq!(
            moved_g, 0,
            "音频那一帧本来就是 200 Hz,补刀遍却动了 {moved_g} 帧 —— 谱证据的门失效了"
        );

        // ⑸ 成段的错(5 帧)不归它管 —— 那是 Viterbi 的活,这里一帧都不许碰。
        let mut run5 = vec![400.0f32; n_frames];
        for v in run5.iter_mut().take(30).skip(25) {
            *v = 200.0;
        }
        let before = run5.clone();
        let moved_r = fix_octave_outliers_inplace(&mut run5, &audio400, HOP);
        assert_eq!(
            moved_r, 0,
            "5 帧的成段错超过 OCTAVE_OUTLIER_MAX_RUN,补刀遍却动了 {moved_r} 帧"
        );
        assert_eq!(run5, before);
    }
    /// ⛔⛔ S165 —— chunk tier 钉死旋钮([`chunk_max_s`])。
    ///
    /// 它存在的唯一理由是**让 A/B 成立**:tier 按渲染那一刻的可用 commit 选,
    /// 实测 19 s 臂坏率 13.0-13.6 % vs 32 s 臂 6.2-6.4 % ⇒ tier 不同的两条臂没有可比性。
    /// ⇒ ⑴ 不设 = `None`(逐位回到自动选择);⑵ 设了要落在**不大于它**的第一档;
    ///    ⑶ 比最小档还小 ⇒ 落到最小档(而不是 panic 或回到默认)。
    #[test]
    fn the_chunk_tier_can_be_pinned_so_two_arms_are_comparable() {
        // ⑴ 解析:只认正数
        assert_eq!(super::chunk_max_s_from(None), None);
        assert_eq!(super::chunk_max_s_from(Some("")), None);
        assert_eq!(super::chunk_max_s_from(Some("0")), None);
        assert_eq!(super::chunk_max_s_from(Some("-5")), None);
        assert_eq!(super::chunk_max_s_from(Some(" 32 ")), Some(32.0));
        // ⑵/⑶ 选档:`find(x_max <= pin)`,兜底最小档
        let pick = |pin: f32| -> usize {
            super::CHUNK_TIERS
                .iter()
                .find(|t| (t.x_max as f32) <= pin)
                .unwrap_or(super::CHUNK_TIERS.last().expect("tiers non-empty"))
                .x_max
        };
        assert_eq!(pick(41.0), 41, "钉在最大档应当就是最大档");
        assert_eq!(pick(32.0), 32);
        assert_eq!(pick(40.0), 32, "钉 40 拿不到 41,应当降到 32");
        assert_eq!(pick(1.0), super::CHUNK_TIERS.last().unwrap().x_max, "钉得比最小档还小 ⇒ 最小档");
        // ⛔ 承重:至少要有两档,否则「钉死」这件事本身没有意义
        assert!(super::CHUNK_TIERS.len() >= 2);
    }
    /// ⛔ S165 —— 补刀遍的门槛是**照实测的空档**定的,不许被悄悄调松。
    ///
    /// 实测四个候选的 `prom(2f) − prom(f)`:真的错 +21.8 / +15.4,真的低音起音 +0.7 / −6.0。
    /// ⇒ 门槛必须落在 `(0.65, 15.41)` 里面,否则要么放过真错、要么误伤真低音。
    /// ⚠ 断言写**字面量**,不许拿常量自己跟自己比(那样改常量判据照绿)。
    #[test]
    fn the_outlier_threshold_stays_inside_the_measured_gap() {
        assert!(
            OCTAVE_OUTLIER_MIN_DB > 0.65 && OCTAVE_OUTLIER_MIN_DB < 15.41,
            "门槛 {OCTAVE_OUTLIER_MIN_DB} dB 掉出了实测空档 (0.65, 15.41) —— \
             要么放过 4:28.640 那一族,要么误伤 4:26.910 那一族的低音起音"
        );
        assert_eq!(OCTAVE_OUTLIER_MAX_RUN, 2, "孤立游程上限改了就不再是「孤立」");
    }
    /// ⭐⭐⭐⭐⭐ S165 —— **八度修复:两种情形必须分开**。
    ///
    /// 这条判据的全部价值在于它的**两个对照**:
    /// ⑴ **阳性**:音频真基频 400 Hz,而 RMVPE 报 **200** ⇒ 必须抬回 400
    ///    (200 处**没有谱峰** —— 那是谐波之间);
    /// ⑵ ⛔ **阴性**:音频真基频**就是** 200 Hz(谐波 200/400/600…),RMVPE 报 200
    ///    ⇒ **一帧都不许动**。
    ///    ⭐ 机理上这两种情形是这样分开的:对 `fc=400` 的候选,背景取 `0.5·fc=200` 与 `1.5·fc=600`,
    ///    而在真 200 Hz 的信号里**那两处都有谐波** ⇒ 背景高 ⇒ `prom(2fc)` 反而低。
    /// ⑶ **无声帧**(`f0 == 0`)不许被抬。
    /// ⑷ **时间连续性**:整段一致的判断,不许出现逐帧来回翻
    ///    (S160 登记过「逐帧独立版判负」)。
    #[test]
    fn octave_repair_lifts_a_halved_track_and_leaves_a_genuine_low_one_alone() {
        const SR: f32 = 16_000.0;
        const HOP: usize = 160; // 100 fps
        let n_frames = 120usize;
        let n = n_frames * HOP + 4 * HOP;
        // 造一个真基频 `f` 的多谐波音
        let tone = |f: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR;
                    let mut v = 0.0f32;
                    for k in 1..=6 {
                        let kf = f * k as f32;
                        if kf < SR / 2.0 * 0.9 {
                            v += (2.0 * std::f32::consts::PI * kf * t).sin() / k as f32;
                        }
                    }
                    v * 0.3
                })
                .collect()
        };
        // ⑴ 阳性:音频是 400,轨报成 200
        let audio400 = tone(400.0);
        let mut halved = vec![200.0f32; n_frames];
        let lifted = fix_octave_inplace(&mut halved, &audio400, HOP);
        assert!(
            lifted >= n_frames * 8 / 10,
            "真基频 400 而轨报 200:只抬了 {lifted}/{n_frames} 帧 —— 八度修复没生效"
        );
        assert!(
            halved.iter().filter(|v| (**v - 400.0).abs() < 1.0).count() >= n_frames * 8 / 10,
            "抬完之后没落在 400 Hz 上"
        );
        // ⑴b ⭐ **三倍那一支**(实测逼出来的):用户点名的「ぴゃ」上 RMVPE 报的是真 f0 的 **1/3**
        //     (源实测 1422.6 Hz,轨报 471)。只有 `{fc,2fc}` 时那一段会被抬到 942 ——
        //     **还是错的,只是换了个错法**。⇒ 真基频 600、轨报 200 ⇒ 必须抬到 600。
        let audio600 = tone(600.0);
        let mut third = vec![200.0f32; n_frames];
        let lifted3 = fix_octave_inplace(&mut third, &audio600, HOP);
        assert!(
            lifted3 >= n_frames * 8 / 10,
            "真基频 600 而轨报 200:只抬了 {lifted3}/{n_frames} 帧"
        );
        assert!(
            third.iter().filter(|v| (**v - 600.0).abs() < 1.0).count() >= n_frames * 8 / 10,
            "三倍那一支没落在 600 Hz 上(可能只抬到了 400)"
        );

        // ⑴c ⛔⛔ **`×3` 的先验代价必须真的在拦** —— 它是 `×3` 那一支唯一的防误伤手段。
        //     实测:先验 = 0 时**阴性对照段被抬 8.7-23.9 %**(几乎全部来自 `×3`),
        //     而只有 `{fc,2fc}` 时是 0.0-0.8 %。⇒ 这里构造一个「`×3` 略占优但不该被选」的局面:
        //     音频真基频 **200**(谐波 200/400/600…),轨也报 200 ——
        //     `600` 处**确实有谐波**,所以 `prom(3·fc)` 不算低;只有先验能把它压住。
        //     ⭐ 这一条与 ⑵ 是**同一个夹具的两个断言**:⑵ 说「一帧都没动」,
        //     这一条说明**为什么**那不是白捡的(把先验设成 0,`×3` 就会开始抢)。
        //     ⚠ 变异测试:把 `OCTAVE_TRIPLE_PRIOR_DB` 设成 0,下面这条会红。
        {
            // ⛔ 夹具必须让 `×3` **真的想抢** —— 用 `tone(200)` 时 600 分量只有 1/3 振幅,
            //    `prom(3·fc)` 本来就不够高,先验设成 0 也不会动 ⇒ 那样的断言是空的(试过,变异照绿)。
            //    ⇒ 这里把 **600 Hz 分量做成与基频等强**(真人共振峰落在三次谐波上就是这个形状)。
            let audio_h3: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f32 / SR;
                    let two_pi = 2.0 * std::f32::consts::PI;
                    ((two_pi * 200.0 * t).sin()
                        + 0.5 * (two_pi * 400.0 * t).sin()
                        + 1.0 * (two_pi * 600.0 * t).sin()   // ← 与基频等强
                        + 0.2 * (two_pi * 800.0 * t).sin())
                        * 0.3
                })
                .collect();
            let mut t3 = vec![200.0f32; n_frames];
            let moved3 = fix_octave_inplace(&mut t3, &audio_h3, HOP);
            // ⚠⚠ **如实登记:这一条也抓不到把 `OCTAVE_TRIPLE_PRIOR_DB` 变异成 0。**
            //    根因与上面 ⑷ 那条**是同一个**:`OCTAVE_SWITCH_PENALTY_DB`(12 dB 的转移代价)
            //    让 Viterbi 从 `×1` 起步后**根本不肯换档**,先验加不加都一样 ⇒ 合成音上
            //    这两个常量的作用**互相掩盖**,端到端夹具分不开它们。
            //    ⇒ ▶ 两条断言共用一个解法:**把 Viterbi 从 `fix_octave_inplace` 里拆出来、
            //      直接喂观测代价序列**,那时 `pen` 与 `prior` 各自的作用才是可分的。
            //      **这是可测性欠账,不是算法欠账**(先验的值是在真实素材上扫出来的,见
            //      [`OCTAVE_TRIPLE_PRIOR_DB`] 的那张表:阴性对照 23.9 % → 1.5 %)。
            //    ⛔ 别把它读成「已经钉住了」。
            assert_eq!(
                moved3, 0,
                "真基频 200(三次谐波与基频等强)被抬了 {moved3} 帧
                 (注意:见上面那段,这条断言【不】承担 OCTAVE_TRIPLE_PRIOR_DB 那个自由度)"
            );
        }

        // ⑵ ⛔ 阴性对照:音频真的就是 200
        let audio200 = tone(200.0);
        let mut genuine = vec![200.0f32; n_frames];
        let moved = fix_octave_inplace(&mut genuine, &audio200, HOP);
        assert_eq!(
            moved, 0,
            "音频真基频就是 200 Hz,却被抬了 {moved} 帧 —— 这个修复会毁掉正常的低音
             (三个候选 ×1/×2/×3 都不许被选中)"
        );
        // ⑶ 无声帧
        let mut with_rest = vec![200.0f32; n_frames];
        for v in with_rest.iter_mut().take(20) {
            *v = 0.0;
        }
        fix_octave_inplace(&mut with_rest, &audio400, HOP);
        assert!(
            with_rest[..20].iter().all(|v| *v == 0.0),
            "无声帧被抬了 —— uv=0 的语义被破坏"
        );
        // ⑷ ⛔⛔ 时间连续性 —— **夹具必须造「真值不变、但观测不确定」的局面**,前三版都错了:
        //    ① 干净的 400 Hz:每帧观测一边倒,penalty 不参与决策 ⇒ 变异照绿;
        //    ② 弱帧整体降幅度:`prom` 是**比值**,等比缩放不改变它 ⇒ 变异照绿;
        //    ③ 某些帧真的换成 200 Hz:那是**真值变了**,跟随它是**对的行为** ⇒ 正常配置也红。
        //    ⇒ ④ 正确的做法:真基频**始终是 400**,但在部分帧**只把基频分量削掉、保留高次谐波** ——
        //      `prom(400)` 掉下来、真值却没变,逐帧最优会误判,只有转移代价能把它救回来。
        let mut veiled: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / SR;
            let fr = i / HOP;
            let f1 = if fr % 12 < 4 { 0.04f32 } else { 1.0 }; // 基频被遮住的帧
            let mut v = (2.0 * std::f32::consts::PI * 400.0 * t).sin() * f1;
            for k in 2..=6 {
                let kf = 400.0 * k as f32;
                if kf < SR / 2.0 * 0.9 {
                    v += (2.0 * std::f32::consts::PI * kf * t).sin() / k as f32;
                }
            }
            v * 0.3
        }).collect();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for v in veiled.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v += ((seed >> 40) as f32 / 8_388_608.0 - 1.0) * 0.004;
        }
        let mut veil_track = vec![200.0f32; n_frames];
        fix_octave_inplace(&mut veil_track, &veiled, HOP);
        let flips = veil_track.windows(2).filter(|w| (w[0] - w[1]).abs() > 1.0).count();
        // ⚠⚠ **如实登记:这一条【目前抓不到】把 `OCTAVE_SWITCH_PENALTY_DB` 变异成 0。**
        //    四版夹具都试过(见上面那段),合成音上两个候选的观测代价始终一边倒,
        //    转移代价不参与决策 ⇒ 这条断言现在只是**接线闸**(证明这一路跑得通、不乱跳),
        //    **不承担「转移代价是否承重」那个自由度**。
        //    ⇒ 要真正钉住它,得把 Viterbi 从 `fix_octave_inplace` 里拆出来、直接喂观测代价序列
        //      (那时可以手工构造「逐帧最优 ≠ 全局最优」的局面)。**这是可测性欠账,不是算法欠账。**
        //    ⛔ 别把它读成「已经钉住了」—— 那正是 S165 §77 那条血训(「实验证明了方向」被记成「已经修好了」)。
        assert!(
            flips <= 6,
            "基频被间歇遮住的素材上翻转了 {flips} 次 —— 这一路的行为不对(注意:见上面那段,
             这条断言【不】承担 OCTAVE_SWITCH_PENALTY_DB 那个自由度)"
        );
        // ⛔ 而且它仍然要把大部分帧抬上去 —— 否则「不翻转」可以由「什么都不做」满足。
        let lifted_w = veil_track.iter().filter(|v| (**v - 400.0).abs() < 1.0).count();
        assert!(
            lifted_w >= n_frames / 2,
            "被遮住的素材上只抬了 {lifted_w}/{n_frames} 帧 —— 「不翻转」是靠什么都不做换来的"
        );
    }

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

#[cfg(test)]
mod s160d_tests {
    use super::fill_isolated_uv_inplace;

    /// S160d —— 只填**长度恰好 1** 的洞,而且要**先收集再写**。
    #[test]
    fn fills_only_isolated_single_frame_holes() {
        // 单帧洞 ⇒ 填成两侧均值
        let mut a = [100.0f32, 0.0, 200.0];
        assert_eq!(fill_isolated_uv_inplace(&mut a), 1);
        assert_eq!(a[1], 150.0);

        // ⛔ 两帧一个不碰(⚠ 这条是「先收集再写」的判据:边扫边写会把 [1] 补成浊音,
        //    于是 [2] 就变成「两侧都是浊音的单帧洞」而被一起吃掉)。
        let mut b = [100.0f32, 0.0, 0.0, 200.0];
        assert_eq!(fill_isolated_uv_inplace(&mut b), 0);
        assert_eq!(b, [100.0, 0.0, 0.0, 200.0]);

        // 端点上的 0 不算(没有「两侧」)
        let mut c = [0.0f32, 100.0, 0.0];
        assert_eq!(fill_isolated_uv_inplace(&mut c), 0);

        // 全浊音 ⇒ 不动;太短 ⇒ 不动
        let mut d = [100.0f32, 110.0, 120.0];
        assert_eq!(fill_isolated_uv_inplace(&mut d), 0);
        let mut e = [0.0f32, 100.0];
        assert_eq!(fill_isolated_uv_inplace(&mut e), 0);

        // 多个孤立洞一起填
        let mut f = [100.0f32, 0.0, 200.0, 0.0, 300.0];
        assert_eq!(fill_isolated_uv_inplace(&mut f), 2);
        assert_eq!(f, [100.0, 150.0, 200.0, 250.0, 300.0]);
    }
}
