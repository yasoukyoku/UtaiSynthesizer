//! S60-1 GAME port-parity gate: the Rust 5-graph pipeline vs the python-ORT oracle
//! AND the UST ground truth.
//!
//! ★ Per-note BITWISE parity is NOT a valid target here — the OFFICIAL pipeline is
//! itself nondeterministic: ORT CPU multithreaded reductions give run-to-run fp32
//! noise, and the D3PM segmenter feeds boundaries back through 8 steps, amplifying a
//! flipped threshold frame into ±2% note-count variance (the python oracle produced
//! 500 / 504 / 509 voiced notes across identical runs). Do NOT chase exact counts.
//! The valid gates are:
//!   1. slicer2 port: chunk boundaries SAMPLE-EXACT (deterministic, pure DSP);
//!   2. statistical equivalence vs the stored oracle snapshot (overlap-matched);
//!   3. QUALITY vs the UST ground truth that synthesized the test vocal — must equal
//!      the oracle's own scores (oracle: 500/500 matched, onset median 5 ms,
//!      pitch exact-semitone 97.6%, 0 octave errors, 19 spurious).
//!
//! The oracle (TESTING\utai-v2-testing\game_oracle\game_oracle.py, converter venv) runs
//! the OFFICIAL deployment flow: ONNX.md + Game.cs semantics + callbacks.py assembly,
//! slicer2.py imported VERBATIM, infer.py extract defaults.
//!
//! Fixtures (skip-if-absent, audition_render.rs precedent):
//!   D:\MyDev\TESTING\utai-v2-testing\game_oracle\{vocal_solo_44k.wav, oracle_notes.json}
//!   D:\MyDev\TESTING\虽然歌声无形ust\main.ust   (the score the vocal was rendered from)
//!   data\models\aux\game\ (the 1.0.3 medium ONNX package)
//!
//! Run:  cargo test --test game_parity -- --ignored --nocapture   (CPU, ~1 min)

use std::path::PathBuf;

use utai_lib::inference::engine::OnnxEngine;
use utai_lib::inference::midi_extract;

fn app_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn init_ort() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("utai_lib=info")),
        )
        .try_init();
    utai_lib::suppress_windows_dll_error_dialogs();
    utai_lib::init_ort_runtime(&app_root());
}

fn oracle_dir() -> PathBuf {
    PathBuf::from(r"D:\MyDev\TESTING\utai-v2-testing\game_oracle")
}

#[derive(serde::Deserialize)]
struct OracleChunk {
    offset: f64,
    length: f64,
}

#[derive(serde::Deserialize)]
struct OracleNote {
    onset: f64,
    offset: f64,
    pitch: f64,
}

#[derive(serde::Deserialize)]
struct OracleFile {
    chunks: Vec<OracleChunk>,
    notes: Vec<OracleNote>,
}

#[test]
#[ignore]
fn game_parity_vs_oracle() {
    let wav_path = oracle_dir().join("vocal_solo_44k.wav");
    let oracle_path = oracle_dir().join("oracle_notes.json");
    let models_dir = app_root().join("data").join("models");
    if !wav_path.exists() || !oracle_path.exists() || !midi_extract::game_installed(&models_dir) {
        eprintln!("SKIP: fixtures missing (run game_oracle/prep_input.py + game_oracle.py, install GAME)");
        return;
    }
    init_ort();

    let oracle: OracleFile =
        serde_json::from_str(&std::fs::read_to_string(&oracle_path).unwrap()).unwrap();
    let buf = utai_lib::audio::load_audio(&wav_path).unwrap();
    assert_eq!(buf.channels, 1, "prep input must be mono");
    let sr = buf.sample_rate;
    assert_eq!(sr, 44100);

    // ── 1. slicer parity: chunk boundaries sample-exact ──
    let chunks = midi_extract::slice_silence(&buf.samples, sr);
    assert_eq!(
        chunks.len(),
        oracle.chunks.len(),
        "chunk count mismatch: rust {} vs oracle {}",
        chunks.len(),
        oracle.chunks.len()
    );
    for (i, (c, oc)) in chunks.iter().zip(&oracle.chunks).enumerate() {
        let off = c.offset_samples as f64 / sr as f64;
        let len = c.samples.len() as f64 / sr as f64;
        assert!(
            (off - oc.offset).abs() < 1e-9 && (len - oc.length).abs() < 1e-9,
            "chunk {i}: rust off={off:.6} len={len:.6} vs oracle off={:.6} len={:.6}",
            oc.offset,
            oc.length
        );
    }
    eprintln!("slicer parity: {} chunks sample-exact ✓", chunks.len());

    // ── 2. full pipeline parity ──
    let engine = OnnxEngine::new();
    let t0 = std::time::Instant::now();
    let notes = midi_extract::extract_notes(
        &engine,
        &models_dir,
        &buf.samples,
        0,
        &|| false,
        &mut |_| {},
    )
    .unwrap();
    eprintln!("rust pipeline: {} notes in {:.1}s", notes.len(), t0.elapsed().as_secs_f64());

    // dump for offline diffing (diagnose divergences without re-running the pipeline)
    let dump: Vec<serde_json::Value> = notes
        .iter()
        .map(|n| serde_json::json!({"onset": n.onset_sec, "offset": n.offset_sec, "pitch": n.pitch}))
        .collect();
    std::fs::write(
        oracle_dir().join("rust_notes.json"),
        serde_json::to_string_pretty(&serde_json::json!({ "notes": dump })).unwrap(),
    )
    .unwrap();

    // ── 2a. statistical equivalence vs the oracle snapshot (D3PM nondeterminism → overlap
    //        matching, not index pairing; see header) ──
    let count_dev = (notes.len() as f64 - oracle.notes.len() as f64).abs() / oracle.notes.len() as f64;
    assert!(
        count_dev <= 0.04,
        "note count deviates {:.1}% (rust {} vs oracle {}) — beyond the D3PM variance band",
        count_dev * 100.0,
        notes.len(),
        oracle.notes.len()
    );
    let mut matched = 0usize;
    for on in &oracle.notes {
        let best = notes
            .iter()
            .map(|n| {
                let ov = n.offset_sec.min(on.offset) - n.onset_sec.max(on.onset);
                (ov, n)
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .filter(|(ov, _)| *ov > 0.3 * (on.offset - on.onset));
        if let Some((_, n)) = best {
            if (n.onset_sec - on.onset).abs() <= 0.025 && (n.pitch as f64 - on.pitch).abs() <= 0.5 {
                matched += 1;
            }
        }
    }
    let match_rate = matched as f64 / oracle.notes.len() as f64;
    eprintln!(
        "oracle equivalence: {}/{} matched ({:.1}%), count dev {:.1}%",
        matched,
        oracle.notes.len(),
        match_rate * 100.0,
        count_dev * 100.0
    );
    assert!(match_rate >= 0.95, "only {:.1}% of oracle notes reproduced", match_rate * 100.0);

    // ── 2b. QUALITY vs the UST ground truth — must equal the oracle's own scores ──
    let gt = parse_ust_ground_truth(&PathBuf::from(r"D:\MyDev\TESTING\虽然歌声无形ust\main.ust"));
    assert!(gt.len() >= 400, "UST parse looks broken: {} voiced notes", gt.len());
    let mut onset_errs: Vec<f64> = Vec::new();
    let mut pitch_exact = 0usize;
    let mut octave_errs = 0usize;
    let mut unmatched_gt = 0usize;
    let mut used = std::collections::HashSet::new();
    for (g_on, g_off, g_midi) in &gt {
        let mut best: Option<(f64, usize)> = None;
        for (j, n) in notes.iter().enumerate() {
            let ov = n.offset_sec.min(*g_off) - n.onset_sec.max(*g_on);
            if best.map(|(b, _)| ov > b).unwrap_or(ov > 0.0) {
                best = Some((ov, j));
            }
        }
        match best.filter(|(ov, _)| *ov >= 0.3 * (g_off - g_on)) {
            None => unmatched_gt += 1,
            Some((_, j)) => {
                used.insert(j);
                let n = &notes[j];
                onset_errs.push((n.onset_sec - g_on).abs());
                let perr = n.pitch as f64 - *g_midi as f64;
                if (n.pitch as f64).round() as i32 == *g_midi {
                    pitch_exact += 1;
                }
                if (perr.abs() - 12.0).abs() < 1.0 {
                    octave_errs += 1;
                }
            }
        }
    }
    onset_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let onset_median = onset_errs.get(onset_errs.len() / 2).copied().unwrap_or(f64::NAN);
    let matched_gt = gt.len() - unmatched_gt;
    let spurious = notes.len() - used.len();
    let exact_rate = pitch_exact as f64 / matched_gt.max(1) as f64;
    eprintln!(
        "UST ground truth: matched {}/{} (spurious {}), onset median {:.0}ms, pitch exact {:.1}%, octave errs {}",
        matched_gt,
        gt.len(),
        spurious,
        onset_median * 1000.0,
        exact_rate * 100.0,
        octave_errs
    );
    assert_eq!(unmatched_gt, 0, "GT notes missed: {unmatched_gt}");
    assert!(onset_median <= 0.010, "onset median {:.1}ms > 10ms", onset_median * 1000.0);
    assert!(exact_rate >= 0.96, "pitch exact-semitone rate {:.1}% < 96%", exact_rate * 100.0);
    assert_eq!(octave_errs, 0, "octave-class errors: {octave_errs}");
    assert!(
        spurious as f64 <= 0.06 * gt.len() as f64,
        "spurious notes {} > 6% of GT",
        spurious
    );
}

/// Parse the UST that synthesized the test vocal into voiced (onset_sec, offset_sec, midi).
/// UST: 480 ticks/quarter, Tempo in [#SETTING], Lyric=R marks rests.
fn parse_ust_ground_truth(path: &PathBuf) -> Vec<(f64, f64, i32)> {
    let bytes = std::fs::read(path).expect("read main.ust");
    let text = String::from_utf8_lossy(&bytes);
    let mut tempo = 120.0f64;
    let mut notes = Vec::new();
    let mut cur_tick = 0.0f64;
    let mut lyric = String::new();
    let mut num = 60i32;
    let mut length = 0.0f64;
    let mut in_note = false;
    let flush = |lyric: &str, num: i32, length: f64, cur_tick: &mut f64, tempo: f64, notes: &mut Vec<(f64, f64, i32)>| {
        let sec_per_tick = 60.0 / tempo / 480.0;
        let on = *cur_tick * sec_per_tick;
        let off = (*cur_tick + length) * sec_per_tick;
        if !lyric.is_empty() && lyric != "R" && lyric != "r" {
            notes.push((on, off, num));
        }
        *cur_tick += length;
    };
    for line in text.lines() {
        let line = line.trim_start_matches('\u{feff}').trim();
        if line.starts_with("[#") {
            if in_note {
                flush(&lyric, num, length, &mut cur_tick, tempo, &mut notes);
            }
            in_note = line[2..line.len() - 1].chars().all(|c| c.is_ascii_digit());
            lyric = String::new();
            num = 60;
            length = 0.0;
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            match k {
                "Tempo" => tempo = v.trim().parse().unwrap_or(tempo),
                "Lyric" if in_note => lyric = v.trim().to_string(),
                "NoteNum" if in_note => num = v.trim().parse().unwrap_or(60),
                "Length" if in_note => length = v.trim().parse().unwrap_or(0.0),
                _ => {}
            }
        }
    }
    if in_note {
        flush(&lyric, num, length, &mut cur_tick, tempo, &mut notes);
    }
    notes
}
