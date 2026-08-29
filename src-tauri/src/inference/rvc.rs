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
/// with a running-RMS window before the model sees it. **On by default at 100 ms**;
/// `UTAI_COVER_FLATTEN_MS=0` turns it off.
///
/// Flipped on after three rulers -- each of which had to pass its own negative controls --
/// agreed on the direction (see the call site for why the shape is one-sided):
///   spectral flatness (noise-ness):        2 better, 0 worse, 7 unchanged
///   spectral-structure similarity to src:  8 better, 1 worse, 0 unchanged, p50 +0.099
///   whole-song bad-window rate:            5.8% -> 3.0%
/// and the source envelope is tracked BETTER afterwards, not worse (r 0.734 -> 0.861).
/// ⚠ The one segment that all three call worse is 4:05.4, inside the region the user
/// flagged as its own kind of damage -- tracked separately, not a reason to hold this back.
fn cover_flatten_ms() -> Option<f32> {
    parse_flatten_ms(std::env::var("UTAI_COVER_FLATTEN_MS").ok().as_deref())
}

/// The parsing half of [`cover_flatten_ms`], split out so it can be pinned by a test
/// without touching process-wide env (which would race the rest of the suite).
fn parse_flatten_ms(raw: Option<&str>) -> Option<f32> {
    /// ⛔ Do not change without re-running the three-ruler acceptance in S165 §99.
    const DEFAULT_MS: f32 = 100.0;
    match raw {
        Some(v) => {
            let t = v.trim();
            // An explicit 0 (or anything out of range) is a deliberate OFF, not a typo we
            // should paper over with the default -- the arm has to stay reachable, or the
            // knob stops being falsifiable.
            if t == "0" {
                return None;
            }
            t.parse::<f32>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 10.0 && *v <= 1000.0)
        }
        None => Some(DEFAULT_MS),
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
    const SR: f64 = 16_000.0;
    let win = (0.040 * SR) as usize; // 40 ms
    let nfft = win.next_power_of_two();
    // 汉宁窗(一次算好)
    let w: Vec<f64> = (0..win)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / win as f64).cos())
        .collect();
    // 每帧三个状态(×1 / ×2 / ×3)的观测代价
    let mut obs: Vec<[f64; 3]> = Vec::with_capacity(pitchf.len());
    for (i, &fc) in pitchf.iter().enumerate() {
        if fc <= 20.0 || 2.0 * f64::from(fc) >= SR / 2.0 * 0.9 {
            obs.push([0.0, 1e9, 1e9]); // 无声/太高 ⇒ 只能保持
            continue;
        }
        let c = i * hop;
        let a = c.saturating_sub(win / 2);
        if a + win > audio16k.len() {
            obs.push([0.0, 1e9, 1e9]);
            continue;
        }
        let mut re: Vec<f64> = (0..nfft)
            .map(|k| if k < win { f64::from(audio16k[a + k]) * w[k] } else { 0.0 })
            .collect();
        let mut im = vec![0.0f64; nfft];
        octave_fft(&mut re, &mut im);
        let mag: Vec<f64> =
            (0..=nfft / 2).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()).collect();
        let bin = |f: f64| -> f64 { f * nfft as f64 / SR };
        let peak = |f: f64| -> f64 {
            let (lo, hi) = (bin(f * 0.88).floor().max(0.0) as usize, bin(f * 1.12).ceil() as usize);
            mag[lo.min(nfft / 2)..=hi.min(nfft / 2)].iter().copied().fold(0.0f64, f64::max)
        };
        let bg = |f: f64| -> f64 {
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
        let f = f64::from(fc);
        let p1 = 20.0 * (peak(f).max(1e-12) / bg(f).max(1e-12)).log10();
        let p2 = 20.0 * (peak(2.0 * f).max(1e-12) / bg(2.0 * f).max(1e-12)).log10();
        // ⭐ 3 倍那一支:超出 Nyquist 就直接不可用(而不是读到一个折叠回来的假峰)。
        let p3 = if 3.0 * f < SR / 2.0 * 0.9 {
            20.0 * (peak(3.0 * f).max(1e-12) / bg(3.0 * f).max(1e-12)).log10()
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
        super::vocal_range::apply_dead_only_windows(
            &mut audio_opt,
            final_sr,
            out_frames as i64,
            &range_jobs,
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


    /// S165 —— 钉住输入包络提升的**出厂值**。
    ///
    /// 这一刀是翻了默认的(不再是探针),所以它值多少必须有判据看着 —— 三把尺子各自过了
    /// 自己的阴性对照之后才翻的:谱平坦度 2 好 0 坏、谱结构相似度 8 好 1 坏(p50 +0.099)、
    /// 全曲坏窗率 5.8% → 3.0%,而且源包络反而**更**还原得住(r 0.734 → 0.861)。
    ///
    /// ⚠ 第二条断言(显式 `0` 必须真的关掉)不是形式主义:阴性臂要是被默认值悄悄吃掉,
    /// 以后就再也没法证伪这把刀了 —— S129「一条从没被执行过的错误分支就是一条空判据」。
    #[test]
    fn the_input_envelope_lift_ships_on_at_100ms_and_stays_switchable() {
        // ⑴ 没设环境变量 = 出厂 ⇒ 开着,100 ms
        assert_eq!(
            super::parse_flatten_ms(None),
            Some(100.0),
            "出厂应当是开着的 100 ms —— 改这个值前先重跑 S165 §99 的三尺子验收"
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
