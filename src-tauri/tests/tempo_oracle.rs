//! tempo_oracle.rs — DEV-ONLY oracle harness for the S59 tempo/beat-grid detector.
//!
//! Runs `utai_dsp::tempo::analyze_tempo` over local real-music files and dumps JSON for the
//! librosa comparison script (D:\MyDev\TESTING\utai-v2-testing\tempo_oracle\tempo_oracle.py) —
//! librosa (ISC) is the reference oracle, per the S59 validation plan. Ignored by default: it
//! depends on machine-local test audio under D:\MyDev\TESTING. Playbook:
//!   1. cargo test --test tempo_oracle -- --ignored --nocapture
//!   2. training\.venv\Scripts\python.exe D:\MyDev\TESTING\utai-v2-testing\tempo_oracle\tempo_oracle.py
//! Acceptance: every file within 2% of librosa's octave FAMILY; exact-level disagreements are
//! the known ×1.5/÷1.5 ambiguity class (fix via the clip's ×2/÷2 menu, not the detector).
//!
//! No ORT involvement (pure DSP + wav decode), so no init_ort needed.

use std::path::PathBuf;

fn mono_of(buf: &utai_lib::audio::AudioBuffer) -> Vec<f32> {
    let ch = buf.channels.max(1) as usize;
    if ch == 1 {
        return buf.samples.clone();
    }
    buf.samples
        .chunks_exact(ch)
        .map(|fr| fr.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn test_files() -> Vec<PathBuf> {
    let mut files = vec![
        PathBuf::from(r"D:\MyDev\TESTING\MSST\perf_mix_120s.wav"),
        PathBuf::from(r"D:\MyDev\TESTING\MSST\mono_20s.wav"),
        PathBuf::from(r"D:\MyDev\TESTING\ikanaiteyo\instr.wav"),
        PathBuf::from(r"D:\MyDev\TESTING\ikanaiteyo\vocal.wav"),
        PathBuf::from(r"D:\MyDev\TESTING\ikanaiteyo\请不要带我走 _1.wav"),
    ];
    // the ytdlp full song has fullwidth punctuation in its name — locate it by substring
    let ytdir = PathBuf::from(r"D:\MyDev\TESTING\ARCHIVE\Utai_backup_0425\app\temp\ytdlp_downloads");
    if let Ok(rd) = std::fs::read_dir(&ytdir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.contains("月を追う") && name.ends_with(".wav") {
                files.push(e.path());
            }
        }
    }
    files
}

#[test]
#[ignore]
fn dump_tempo_analysis_for_oracle() {
    let out_path = PathBuf::from(r"D:\MyDev\TESTING\utai-v2-testing\tempo_oracle\rust_tempo.json");
    let mut results = Vec::new();
    for path in test_files() {
        if !path.exists() {
            eprintln!("SKIP (missing): {}", path.display());
            continue;
        }
        let buf = utai_lib::audio::load_audio(&path).expect("decode");
        let mono = mono_of(&buf);
        let started = std::time::Instant::now();
        let entry = match utai_dsp::tempo::analyze_tempo(&mono, buf.sample_rate, 4) {
            Ok(a) => serde_json::json!({
                "file": path.to_string_lossy(),
                "sr": buf.sample_rate,
                "secs": mono.len() as f64 / buf.sample_rate as f64,
                "elapsed_ms": started.elapsed().as_millis() as u64,
                "bpm": a.bpm,
                "anchor_ms": a.grid_anchor_ms,
                "confidence": a.confidence,
                "not_constant": a.not_constant,
                "downbeat_index": a.downbeat_index,
                "downbeat_margin": a.downbeat_margin,
                "candidates": a.candidates,
                "beats_ms": a.beats_ms,
            }),
            Err(e) => serde_json::json!({
                "file": path.to_string_lossy(),
                "sr": buf.sample_rate,
                "error": format!("{e:?}"),
            }),
        };
        eprintln!(
            "{} → {}",
            path.file_name().unwrap().to_string_lossy(),
            serde_json::to_string(&entry).unwrap().chars().take(220).collect::<String>()
        );
        results.push(entry);
    }
    std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();
    std::fs::write(&out_path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    eprintln!("wrote {}", out_path.display());
}
