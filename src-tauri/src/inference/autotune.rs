//! ② 自动音高调教(旋钮线 Phase A 立室,S73)——note 特征 → autotune_a1.onnx → per-note θ。
//!
//! 单一真源纪律:特征构造 = SVC2SVS `pitch/dataset.py` {build_note_arrays, note_features} 的
//! Rust 孪生(本文件是【孪生】不是【定义】);θ 组装(assemble_theta)已进 ONNX 图
//! (branch-free 重表达,SVC2SVS `pitch/export_onnx.py`)。凡改 dataset.py 特征口径必须
//! 重跑 export_onnx(重导模型 + parity 夹具 `tests/fixtures/autotune_parity.json`)并重跑
//! `tests/autotune_parity.rs`。
//!
//! θ 单位 = 绝对 ms/cents/Hz(SynthV 约定,与 tempo 解耦)。TS 侧负责 tick→ms 换算、
//! θ→Note.transition/vibrato 写入、以及调教所有权规则(用户调教 vs 机器调教)——
//! 见 `src/lib/vocal/autoTune.ts`。retake 的随机 vibrato phase 也是 TS 侧图外后处理
//! (SVC2SVS export_karm 同款语义)。
//!
//! 分段(chunk):训练分布是乐句级窗口(E1 verse/chorus 13-23s 同口径),整轨一次喂会把
//! 序列长度/位置特征(f10/f11)推出分布;按 ≥CHUNK_GAP_MS 的休止切乐句(长休止两侧
//! 无 transition 耦合,切口零损),超长乐句(>MAX_CHUNK_NOTES)在最大内部间隙硬切兜底
//! (硬切若落在贴合缝上,切口首音符失去 abut_prev 上下文——病理输入下的已记录偏差)。

use super::engine::{InputTensor, OnnxEngine};
use crate::{Result, UtaiError};

/// 微缝吸附阈值(ms)——dataset.py ABUT_SNAP_MS 同值。
pub const ABUT_SNAP_MS: f64 = 2.0;
/// 每 note 特征维数——dataset.py N_FEATS 同值。
pub const N_FEATS: usize = 12;
/// 乐句切分:休止 ≥ 此值(ms)另起一段。
const CHUNK_GAP_MS: f64 = 2000.0;
/// 单段音符数上限(O(N²) attention 与训练分布双重考量)。
const MAX_CHUNK_NOTES: usize = 1000;

/// 命令入参的单音符(TS 已按 tick→ms 换算;pitch = 书写音高 + detune/100 的 float MIDI,
/// 与训练侧 GAME 含-cents 口径同构)。
#[derive(Clone, Copy, Debug)]
pub struct NoteIn {
    pub start_ms: f64,
    pub dur_ms: f64,
    pub pitch: f64,
}

/// 吸附后的音符数组(dataset.py build_note_arrays 的输出形态)。
pub struct NoteArrays {
    pub tick: Vec<f64>,
    pub dur: Vec<f64>,
    pub pitch: Vec<f64>,
    pub abut_prev: Vec<bool>,
    pub abut_next: Vec<bool>,
}

/// per-note θ(f64;列序 = ONNX 图输出 = f0_twin θ 布局)。
/// transition: offsetMs, durLeftMs, durRightMs, depthLeftCents, depthRightCents, openEdgeCents
/// vibrato:    depthCents, freqHz, phase(恒0), startMs, easeInMs, easeOutMs
pub struct Theta {
    pub transition: [f64; 6],
    pub vibrato: [f64; 6],
}

/// dataset.py `build_note_arrays` 孪生:微缝(0≤gap<2ms)延长前音闭缝、微重叠(-2<gap<0)
/// 收回,然后精确判等出贴合旗。abut_prev[0] 恒 false(ONNX 图 offset 行 0 的契约)。
pub fn build_note_arrays(notes: &[NoteIn]) -> NoteArrays {
    let n = notes.len();
    let tick: Vec<f64> = notes.iter().map(|x| x.start_ms).collect();
    let mut dur: Vec<f64> = notes.iter().map(|x| x.dur_ms).collect();
    let pitch: Vec<f64> = notes.iter().map(|x| x.pitch).collect();
    for i in 0..n.saturating_sub(1) {
        let gap = tick[i + 1] - (tick[i] + dur[i]);
        if (0.0..ABUT_SNAP_MS).contains(&gap) {
            dur[i] += gap;
        } else if gap > -ABUT_SNAP_MS && gap < 0.0 {
            dur[i] += gap;
        }
    }
    let mut abut_prev = vec![false; n];
    for i in 1..n {
        abut_prev[i] = tick[i] == tick[i - 1] + dur[i - 1];
    }
    let mut abut_next = vec![false; n];
    for i in 0..n.saturating_sub(1) {
        abut_next[i] = abut_prev[i + 1];
    }
    NoteArrays { tick, dur, pitch, abut_prev, abut_next }
}

/// dataset.py `note_features` 孪生:[N, N_FEATS] 行主序 f32(表达式 f64 计算、存储时截断
/// f32 = numpy float32 数组赋值同语义)。
pub fn note_features(a: &NoteArrays) -> Vec<f32> {
    let n = a.tick.len();
    let mut f = vec![0.0f32; n * N_FEATS];
    let pos_div = n.saturating_sub(1).max(1) as f64;
    for i in 0..n {
        let gap_prev = if i == 0 { 0.0 } else { a.tick[i] - (a.tick[i - 1] + a.dur[i - 1]) };
        let gap_next = if i + 1 < n { a.tick[i + 1] - (a.tick[i] + a.dur[i]) } else { 0.0 };
        let int_prev = if i == 0 { 0.0 } else { a.pitch[i] - a.pitch[i - 1] };
        let int_next = if i + 1 < n { a.pitch[i + 1] - a.pitch[i] } else { 0.0 };
        let r = &mut f[i * N_FEATS..(i + 1) * N_FEATS];
        r[0] = ((a.pitch[i] - 60.0) / 24.0) as f32;
        r[1] = (a.dur[i].ln_1p() / 7.0) as f32;
        r[2] = (gap_prev.max(0.0).ln_1p() / 7.0) as f32;
        r[3] = (gap_next.max(0.0).ln_1p() / 7.0) as f32;
        r[4] = if a.abut_prev[i] { 1.0 } else { 0.0 };
        r[5] = if a.abut_next[i] { 1.0 } else { 0.0 };
        r[6] = (int_prev / 12.0).clamp(-2.0, 2.0) as f32;
        r[7] = (int_next / 12.0).clamp(-2.0, 2.0) as f32;
        r[8] = if i == 0 { 1.0 } else { 0.0 };
        r[9] = if i + 1 == n { 1.0 } else { 0.0 };
        r[10] = (i as f64 / pos_div) as f32;
        r[11] = ((n as f64).ln_1p() / 4.0) as f32;
    }
    f
}

/// 乐句切分(半开区间 [lo,hi));在全局吸附后的数组上做。
/// ★迭代 worklist 非递归(S73 审查:病理 100k 全贴合输入曾把递归深度推到 O(N),
/// spawn_blocking 2MiB 栈直接溢出=进程 abort,不可捕获)。
pub fn chunk_ranges(a: &NoteArrays) -> Vec<(usize, usize)> {
    let n = a.tick.len();
    if n == 0 {
        return Vec::new();
    }
    let mut cuts = vec![0usize];
    for i in 1..n {
        let gap = a.tick[i] - (a.tick[i - 1] + a.dur[i - 1]);
        if gap >= CHUNK_GAP_MS {
            cuts.push(i);
        }
    }
    cuts.push(n);
    let mut out = Vec::new();
    let mut work: Vec<(usize, usize)> = cuts.windows(2).rev().map(|w| (w[0], w[1])).collect();
    while let Some((lo, hi)) = work.pop() {
        if hi - lo <= MAX_CHUNK_NOTES {
            out.push((lo, hi));
        } else {
            let cut = oversize_cut(a, lo, hi);
            // 左半先弹出 → out 保持升序
            work.push((cut, hi));
            work.push((lo, cut));
        }
    }
    out
}

/// 超长乐句的切点:只在【中半区】[lo+span/4, hi-span/4) 里找最大正间隙(每刀至少砍掉
/// 25% → 段数/均衡有界,不会退化出 size-1 chunk);中半区无正间隙(全贴合)→ 中点硬切。
/// ★S73 审查:旧版全区间 argmax + 严格 `>` 平局取首,全贴合链退化成逐音符剥离
/// (500 个 size-1 chunk = 零上下文出分布 θ),中点兜底还是死代码。
fn oversize_cut(a: &NoteArrays, lo: usize, hi: usize) -> usize {
    let span = hi - lo;
    let mut best = lo + span / 2;
    let mut best_gap = 0.0f64;
    let w_lo = (lo + span / 4).max(lo + 1);
    let w_hi = hi - span / 4;
    for i in w_lo..w_hi {
        let gap = a.tick[i] - (a.tick[i - 1] + a.dur[i - 1]);
        if gap > best_gap {
            best_gap = gap;
            best = i;
        }
    }
    best
}

/// 取一个 chunk 的子数组;贴合旗按 chunk 内重算(chunk = 一个 clip:首音符 abut_prev=false,
/// 与训练侧「窗口内 build_note_arrays」同语义)。
fn slice_arrays(a: &NoteArrays, lo: usize, hi: usize) -> NoteArrays {
    let tick = a.tick[lo..hi].to_vec();
    let dur = a.dur[lo..hi].to_vec();
    let pitch = a.pitch[lo..hi].to_vec();
    let n = hi - lo;
    let mut abut_prev = vec![false; n];
    for i in 1..n {
        abut_prev[i] = tick[i] == tick[i - 1] + dur[i - 1];
    }
    let mut abut_next = vec![false; n];
    for i in 0..n.saturating_sub(1) {
        abut_next[i] = abut_prev[i + 1];
    }
    NoteArrays { tick, dur, pitch, abut_prev, abut_next }
}

/// 全链:吸附 → 乐句切分 → 逐段 特征+ONNX → θ(与输入 notes 同序同长)。
pub fn run_autotune_model(
    engine: &OnnxEngine,
    session_id: &str,
    notes: &[NoteIn],
) -> Result<Vec<Theta>> {
    let arrays = build_note_arrays(notes);
    let mut out = Vec::with_capacity(notes.len());
    for (lo, hi) in chunk_ranges(&arrays) {
        let sub = slice_arrays(&arrays, lo, hi);
        let n = hi - lo;
        let ni = n as i64;
        let feats = note_features(&sub);
        let inputs = vec![
            ("feats", InputTensor::F32 { data: feats, shape: vec![1, ni, N_FEATS as i64] }),
            (
                "dur_ms",
                InputTensor::F32 {
                    data: sub.dur.iter().map(|&d| d as f32).collect(),
                    shape: vec![1, ni],
                },
            ),
            ("abut_prev", InputTensor::Bool { data: sub.abut_prev.clone(), shape: vec![1, ni] }),
        ];
        let outputs = engine.run(session_id, inputs)?;
        if outputs.len() < 2 {
            return Err(UtaiError::Inference("AUTOTUNE_NO_OUTPUT".into()));
        }
        let (tt, tv) = (&outputs[0], &outputs[1]);
        if tt.len() != n * 6 || tv.len() != n * 6 {
            return Err(UtaiError::Inference(format!(
                "AUTOTUNE_SHAPE: expected {}x6, got theta_t {} / theta_v {}",
                n,
                tt.len(),
                tv.len()
            )));
        }
        for i in 0..n {
            out.push(Theta {
                transition: core::array::from_fn(|j| tt[i * 6 + j] as f64),
                vibrato: core::array::from_fn(|j| tv[i * 6 + j] as f64),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notes(v: &[(f64, f64, f64)]) -> Vec<NoteIn> {
        v.iter().map(|&(s, d, p)| NoteIn { start_ms: s, dur_ms: d, pitch: p }).collect()
    }

    #[test]
    fn snap_closes_micro_gap_and_overlap() {
        // 0≤gap<2 延长闭缝;-2<gap<0 收回;≥2 保持
        let a = build_note_arrays(&notes(&[
            (0.0, 100.0, 60.0),
            (101.5, 100.0, 62.0), // gap 1.5 → 前音延长为 101.5
            (203.0, 100.0, 64.0), // gap 1.5 → 闭缝
            (302.0, 100.0, 65.0), // gap -1.0 → 前音收回
            (405.0, 100.0, 67.0), // gap 3.0 → 保持开缝
        ]));
        assert_eq!(a.dur[0], 101.5);
        assert!(a.abut_prev[1] && a.abut_prev[2] && a.abut_prev[3]);
        assert!(!a.abut_prev[4]);
        assert_eq!(a.abut_next, vec![true, true, true, false, false]);
    }

    #[test]
    fn chunker_splits_on_long_rest_only() {
        let a = build_note_arrays(&notes(&[
            (0.0, 500.0, 60.0),
            (500.0, 500.0, 62.0),
            (1500.0, 500.0, 64.0),   // gap 500 → 不切
            (4500.0, 500.0, 65.0),   // gap 2500 → 切
            (5000.0, 500.0, 67.0),
        ]));
        assert_eq!(chunk_ranges(&a), vec![(0, 3), (3, 5)]);
    }

    #[test]
    fn chunk_slice_resets_leading_abut() {
        // 硬切兜底:>MAX_CHUNK_NOTES 的全贴合长链走中点硬切,切口首音符 abut 重算为 false
        let seq: Vec<(f64, f64, f64)> =
            (0..(MAX_CHUNK_NOTES + 10)).map(|i| (i as f64 * 100.0, 100.0, 60.0)).collect();
        let a = build_note_arrays(&notes(&seq));
        let ranges = chunk_ranges(&a);
        assert!(ranges.len() >= 2);
        assert_eq!(ranges.iter().map(|(l, h)| h - l).sum::<usize>(), MAX_CHUNK_NOTES + 10);
        for &(lo, hi) in &ranges {
            let sub = slice_arrays(&a, lo, hi);
            assert!(!sub.abut_prev[0]);
            assert!(hi - lo <= MAX_CHUNK_NOTES);
        }
    }

    #[test]
    fn chunker_pathological_shapes_stay_balanced() {
        // S73 审查回归:①100k 全贴合(旧版=逐音符剥离→递归溢栈+size-1 chunks)
        // ②100k 间隙递增但全 <CHUNK_GAP(旧版=argmax 恒在链尾→同样剥离)。
        // 新版恒等式:覆盖完整、无超限、无退化小段(中半区窗口 ⇒ 每段 ≥ MAX/4)。
        let n = 100_000usize;
        let abutted: Vec<(f64, f64, f64)> =
            (0..n).map(|i| (i as f64 * 100.0, 100.0, 60.0)).collect();
        let mut t = 0.0f64;
        let growing: Vec<(f64, f64, f64)> = (0..n)
            .map(|i| {
                let s = t;
                t += 100.0 + (i as f64 / n as f64) * 1900.0; // 间隙 0→1900ms,恒 <2000
                (s, 100.0, 60.0)
            })
            .collect();
        for seq in [abutted, growing] {
            let a = build_note_arrays(&notes(&seq));
            let ranges = chunk_ranges(&a);
            assert_eq!(ranges.iter().map(|(l, h)| h - l).sum::<usize>(), n);
            let mut prev_end = 0usize;
            for &(lo, hi) in &ranges {
                assert_eq!(lo, prev_end, "chunk 必须首尾相接升序");
                prev_end = hi;
                let sz = hi - lo;
                assert!(sz <= MAX_CHUNK_NOTES);
                assert!(
                    sz >= MAX_CHUNK_NOTES / 4,
                    "退化小段:size={sz}(旧版病理=size-1 剥离)"
                );
            }
            assert_eq!(prev_end, n);
        }
    }

    #[test]
    fn features_single_note_edge() {
        let a = build_note_arrays(&notes(&[(10.0, 250.0, 65.5)]));
        let f = note_features(&a);
        assert_eq!(f.len(), N_FEATS);
        assert_eq!(f[8], 1.0); // is_first
        assert_eq!(f[9], 1.0); // is_last
        assert_eq!(f[10], 0.0); // pos 0/max(0,1)
        assert!((f[0] - ((65.5 - 60.0) / 24.0) as f32).abs() < 1e-7);
    }
}
