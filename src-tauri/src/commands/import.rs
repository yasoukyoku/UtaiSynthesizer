// S56 — Import ustx / ust / midi → 人声轨(vocal tracks). Rust owns PARSING (binary MIDI + YAML are
// cleaner here, matching the open_project_archive precedent); the TS side only maps the returned data to
// store actions. ADDITIVE: this command reads a file and returns a plain data structure — it touches no
// project state, so it can never regress the editor.
//
// SCOPE (user-confirmed): NOTES (tick/duration/pitch/lyric) + BPM + first time-signature + the part's
// start OFFSET + (ustx only) tuning→detune, and PITCH (S73 线2, user-confirmed「手绘调教性质」):
//   - a part with ANY real tuning (nonzero pitch-point y / vibrato / pitd) is BAKED: the full OpenUTAU
//     pitch curve (platform + vibrato-overwrite + point deltas + pitd; commands/ou_pitch.rs) minus our
//     written-pitch step is imported as the segment's additive pitchDev (hand-drawn layer, exact, no θ
//     fitting), notes get ZERO transitions (or our defaults would double the glides) and NO VibratoSpec
//     (or the mapped vibrato would double the baked one). Baked notes count as USER-tuned → auto-tune 绕行.
//   - a COMPLETELY untuned part (all point y==0 default glides, no vibrato, no pitd) keeps our own
//     SynthV-style defaults instead (智能跳过, user-confirmed) — vibrato mapping stays for parity with
//     the old behavior (vacuously: untuned parts have none).
// One file at a time; a multi-track file yields ONE ImportedTrack per part that has notes.
//
// RESOLUTION: our app is 480 ticks/quarter (TICKS_PER_BEAT). ust and ustx are ALSO 480 → 1:1, no scaling.
// MIDI PPQ comes from the header division and is scaled: our = round(midi_tick * 480 / ppq). Note pitch IS
// the MIDI note number end-to-end (ustx `tone`, ust `NoteNum`, midi key).
//
// OFFSET: each track's notes are placed at their natural absolute position, then REBASED so the created
// vocal SEGMENT starts at the first note: `start_tick` = absolute tick of the FIRST note; each
// note.tick = absolute − start_tick (first note → tick 0). Rests become gaps (the render treats gaps as
// rests). A track with no notes is skipped.

use serde::Serialize;

/// One vocal track's worth of imported score. `start_tick` is where the created SEGMENT (part) begins in
/// our project ticks (= the absolute tick of the first note); each note's `tick` is relative to it.
#[derive(Serialize, Debug, Clone)]
pub struct ImportedTrack {
    pub name: String,
    pub start_tick: i64,
    pub notes: Vec<ImportedNote>,
    /// S73: OU 音高曲线烤入(segment 相对 tick / 整数 cents 折线);None = 未调教 part(智能跳过)
    /// 或非 ustx 来源。Some 时前端必须:①setSegmentPitchDev ②给音符置零 transition ③不映射 vibrato。
    pub pitch_dev: Option<ImportedCurve>,
    /// S73: part 有调教但超出可烤上限被放弃(前端须 toast 提示;绝不与「未调教」混同)。
    pub pitch_dev_dropped: bool,
}

/// 稀疏折线(与 TS PitchCurve 同构;xs 严格递增)。
#[derive(Serialize, Debug, Clone)]
pub struct ImportedCurve {
    pub xs: Vec<i64>,
    pub ys: Vec<i64>,
}

/// One imported note. Ticks/durations are ALREADY in our 480-ppq space. `pitch` is the MIDI note number.
#[derive(Serialize, Debug, Clone)]
pub struct ImportedNote {
    pub tick: i64,
    pub duration: i64,
    pub pitch: i32,
    pub lyric: String,
    /// ustx `tuning`(cents)→ 我们的 detune(无损旋钮映射;烤制基线同样含它,不双重)。
    pub detune: Option<f32>,
}
// S73 注:S56 的 ImportedVibrato/map_vibrato(ustx vibrato→VibratoSpec 有损映射)已删——
// 新语义下带 vibrato 的 part 必然「已调教」→ 整体烤入 pitchDev(含 drift/渐变全语义,
// 严格更保真),映射路径永不可达=死代码。

/// The whole parse result. `bpm` / `time_sig` are the file's FIRST tempo / meter (our app has no tempo
/// map — a single global scalar). `None` = the file carried no bpm/meter → the frontend leaves the editor
/// as-is. Field names reach TS as snake_case (`start_tick`, `time_sig`) — the TS mapper reads them so.
#[derive(Serialize, Debug, Clone)]
pub struct ImportedScore {
    pub tracks: Vec<ImportedTrack>,
    pub bpm: Option<f64>,
    pub time_sig: Option<[u32; 2]>,
}

/// Our project resolution — ust/ustx share it (1:1); midi is scaled to it.
const TICKS_PER_BEAT: i64 = 480;

#[tauri::command]
pub async fn import_score_file(path: String) -> Result<ImportedScore, String> {
    let ext = std::path::Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let bytes = std::fs::read(&path).map_err(|e| format!("IMPORT_READ_FAIL: {e}"))?;
    let score = match ext.as_str() {
        "ust" => parse_ust(&bytes)?,
        "ustx" => parse_ustx(&bytes)?,
        "mid" | "midi" => parse_midi(&bytes)?,
        other => return Err(format!("IMPORT_UNSUPPORTED: {other}")),
    };
    if score.tracks.is_empty() {
        return Err("IMPORT_EMPTY".to_string());
    }
    Ok(score)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// ust (INI-like, UTF-8 or legacy Shift-JIS)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// Decode ust bytes: if a `Charset=Shift-JIS` line is declared (any case/spelling) OR the bytes are not
/// valid UTF-8, decode via Shift-JIS (legacy UTAU); else UTF-8. The scan for the Charset line is done on
/// the ASCII-safe head, so it works before we know the encoding.
fn decode_ust_bytes(bytes: &[u8]) -> String {
    let head = &bytes[..bytes.len().min(512)];
    let head_str = String::from_utf8_lossy(head).to_ascii_lowercase();
    let declares_sjis = head_str.contains("charset=shift-jis")
        || head_str.contains("charset=shift_jis")
        || head_str.contains("charset=sjis");
    if declares_sjis || std::str::from_utf8(bytes).is_err() {
        encoding_rs::SHIFT_JIS.decode(bytes).0.into_owned()
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// One parsed ust section: the header line + the few keys we care about.
struct UstSection {
    header: String,
    lyric: Option<String>,
    notenum: Option<i32>,
    length: Option<i64>,
    tempo: Option<f64>,
}

fn parse_ust(bytes: &[u8]) -> Result<ImportedScore, String> {
    let text = decode_ust_bytes(bytes);

    // Pass 1: collect sections in file order. A section starts at a `[#...]` header; key=value lines below
    // it fill its fields. Keys are matched case-insensitively (some ust dialects vary casing).
    let mut secs: Vec<UstSection> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            secs.push(UstSection { header: line.to_string(), lyric: None, notenum: None, length: None, tempo: None });
            continue;
        }
        let Some(sec) = secs.last_mut() else { continue }; // stray line before any header — ignore
        let Some((k, v)) = line.split_once('=') else { continue };
        let key = k.trim();
        let val = v.trim();
        if key.eq_ignore_ascii_case("Lyric") {
            sec.lyric = Some(val.to_string());
        } else if key.eq_ignore_ascii_case("NoteNum") {
            sec.notenum = val.parse().ok();
        } else if key.eq_ignore_ascii_case("Length") {
            sec.length = val.parse().ok();
        } else if key.eq_ignore_ascii_case("Tempo") {
            sec.tempo = val.parse().ok();
        }
    }

    // Pass 2: walk sections. Position is CUMULATIVE (each note/rest's absolute tick = sum of prior
    // Lengths). `Lyric=R`/`r` (or empty) is a REST → advance position, emit no note. Only `[#<number>]`
    // sections are notes; `[#VERSION]`/`[#SETTING]`/`[#TRACKEND]`/`[#PREV]`/`[#NEXT]`/… are skipped
    // (envelope-helper sections have Length too, so filter on a NUMERIC header, not on Length presence).
    let mut tempo: Option<f64> = None;
    let mut cursor: i64 = 0; // absolute tick
    let mut start_tick: Option<i64> = None; // abs tick of the first real note
    let mut notes: Vec<ImportedNote> = Vec::new();

    for sec in &secs {
        // Take the FIRST tempo seen (SETTING precedes the notes; a per-note Tempo override is ignored once set).
        if tempo.is_none() {
            if let Some(t) = sec.tempo {
                if t.is_finite() && t > 0.0 {
                    tempo = Some(t);
                }
            }
        }
        let inner = sec.header.trim_start_matches("[#").trim_end_matches(']');
        let is_numbered = !inner.is_empty() && inner.bytes().all(|c| c.is_ascii_digit());
        if !is_numbered {
            continue;
        }
        let Some(len) = sec.length else { continue }; // a numbered section without Length isn't a note
        let len = len.max(0);
        let lyric = sec.lyric.clone().unwrap_or_default();
        let is_rest = {
            let l = lyric.trim();
            l.eq_ignore_ascii_case("R") || l.is_empty()
        };
        if is_rest {
            cursor += len; // rest advances the cursor; no note emitted (becomes a gap = rest at render)
            continue;
        }
        if start_tick.is_none() {
            start_tick = Some(cursor);
        }
        let base = start_tick.unwrap();
        notes.push(ImportedNote {
            tick: cursor - base,
            duration: len.max(1),
            pitch: sec.notenum.unwrap_or(60).clamp(0, 127),
            lyric,
            detune: None,
        });
        cursor += len;
    }

    let tracks = if notes.is_empty() {
        Vec::new()
    } else {
        // ust carries no track name → empty; the frontend names it by the file (per-file, one track).
        vec![ImportedTrack { name: String::new(), start_tick: start_tick.unwrap_or(0), notes, pitch_dev: None, pitch_dev_dropped: false }]
    };
    Ok(ImportedScore { tracks, bpm: tempo, time_sig: None }) // ust has NO time signature
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// ustx (YAML, OpenUTAU 0.6)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

// serde ignores unknown fields by default (we do NOT set deny_unknown_fields) — ustx has dozens of keys
// we don't care about (pitch curves, expressions, phoneme overrides, wave_parts…). Only the fields below
// are read.
#[derive(serde::Deserialize)]
struct UstxRoot {
    #[serde(default)]
    bpm: Option<f64>, // legacy top-level fallback
    #[serde(default)]
    beat_per_bar: Option<u32>, // legacy fallback
    #[serde(default)]
    beat_unit: Option<u32>, // legacy fallback
    #[serde(default)]
    tempos: Vec<UstxTempo>, // 0.6 authoritative — prefer tempos[0]
    #[serde(default)]
    time_signatures: Vec<UstxTimeSig>, // prefer time_signatures[0]
    #[serde(default)]
    tracks: Vec<UstxTrack>,
    #[serde(default)]
    voice_parts: Vec<UstxVoicePart>,
}

#[derive(serde::Deserialize)]
struct UstxTempo {
    #[serde(default)]
    bpm: Option<f64>,
}

#[derive(serde::Deserialize)]
struct UstxTimeSig {
    #[serde(default)]
    beat_per_bar: Option<u32>,
    #[serde(default)]
    beat_unit: Option<u32>,
}

#[derive(serde::Deserialize)]
struct UstxTrack {
    #[serde(default)]
    track_name: Option<String>,
}

#[derive(serde::Deserialize)]
struct UstxVoicePart {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    track_no: Option<usize>,
    #[serde(default)]
    position: i64, // part offset in project ticks
    #[serde(default)]
    notes: Vec<UstxNote>,
    #[serde(default)]
    curves: Vec<UstxCurve>, // S73: part 级表达曲线;只消费 abbr=="pitd"
}

/// part 级曲线(UCurve 序列化形态:平行数组,xs=part 相对 tick,ys=整数;pitd 的 ys=cents)。
#[derive(serde::Deserialize)]
struct UstxCurve {
    #[serde(default)]
    xs: Vec<i64>,
    #[serde(default)]
    ys: Vec<i64>,
    #[serde(default)]
    abbr: String,
}

#[derive(serde::Deserialize)]
struct UstxNote {
    #[serde(default)]
    position: i64, // PART-RELATIVE tick
    #[serde(default)]
    duration: i64,
    #[serde(default)]
    tone: i32, // MIDI note number
    #[serde(default)]
    lyric: String,
    #[serde(default)]
    vibrato: Option<UstxVibrato>,
    #[serde(default)]
    pitch: Option<UstxPitch>, // S73: note 级音高控制点
    #[serde(default)]
    tuning: Option<f32>, // cents(UNote.AdjustedTone = tone + tuning/100)
}

/// note 级 pitch 块(UPitch:data 控制点 + snap_first)。
#[derive(serde::Deserialize)]
struct UstxPitch {
    #[serde(default)]
    data: Vec<UstxPitchPoint>,
    #[serde(default = "default_true")]
    snap_first: bool,
}

fn default_true() -> bool {
    true
}

/// 音高控制点:x=相对音符起点 ms(可负,伸进前音符);y=0.1 半音;shape∈{io,l,i,o,sp}。
#[derive(serde::Deserialize)]
struct UstxPitchPoint {
    #[serde(default)]
    x: f32,
    #[serde(default)]
    y: f32,
    #[serde(default)]
    shape: Option<String>,
}

/// OpenUTAU vibrato. `length` = % of the note covered (from the note END); `period` = ms/cycle; `depth` =
/// cents; `in`/`out` = % of the vibrato length that fades in/out; `shift` = phase %. `drift`/`vol_link`
/// are ignored (no equivalent in our model).
#[derive(serde::Deserialize)]
struct UstxVibrato {
    #[serde(default)]
    length: f32,
    #[serde(default)]
    period: f32,
    #[serde(default)]
    depth: f32,
    #[serde(rename = "in", default)]
    fade_in: f32,
    #[serde(default)]
    out: f32,
    #[serde(default)]
    shift: f32,
    #[serde(default)]
    drift: f32, // S73: 烤制路径消费(ou_pitch::OuVibrato.drift;VibratoSpec 映射已删)
}

/// Strip a leading UTF-8 BOM (OpenUTAU writes one) so the YAML parser sees a clean document.
fn strip_bom(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text)
}

fn parse_ustx(bytes: &[u8]) -> Result<ImportedScore, String> {
    let raw = String::from_utf8_lossy(bytes);
    let text = strip_bom(&raw);
    let root: UstxRoot = serde_yaml_ng::from_str(text).map_err(|e| format!("IMPORT_PARSE_USTX: {e}"))?;

    let bpm = root.tempos.first().and_then(|t| t.bpm).or(root.bpm);
    let time_sig = root
        .time_signatures
        .first()
        .and_then(|ts| Some([ts.beat_per_bar?, ts.beat_unit?]))
        .or(match (root.beat_per_bar, root.beat_unit) {
            (Some(n), Some(d)) => Some([n, d]),
            _ => None,
        });
    let file_bpm = bpm.unwrap_or(120.0);

    let mut tracks: Vec<ImportedTrack> = Vec::new();
    for vp in &root.voice_parts {
        if vp.notes.is_empty() {
            continue;
        }
        // Absolute note tick = voice_part.position + note.position. Rebase to the first note.
        let mut abs: Vec<(i64, &UstxNote)> = vp.notes.iter().map(|n| (vp.position + n.position, n)).collect();
        abs.sort_by_key(|(t, _)| *t);
        let base = abs[0].0;
        // Track name = tracks[track_no].track_name, fallback voice_part.name, fallback generic.
        let name = vp
            .track_no
            .and_then(|no| root.tracks.get(no))
            .and_then(|t| t.track_name.clone())
            .filter(|s| !s.trim().is_empty())
            .or_else(|| vp.name.clone().filter(|s| !s.trim().is_empty()))
            .unwrap_or_default(); // no track/part name → empty; the frontend names it by the file (+ index).
        // ── S73 全量音高线导入:OU 音符视图 → 未调教检测 → 烤制 pitchDev ──
        // 重叠归一(审查):同 part 重叠音符裁到后继起点(镜像编辑器 resolveOverlaps 的
        // 一位一音符语义,min 1 防零)——烤制平台基线与前端 findNoteAt 归属才一致。
        // tuning 钳位单点(±1200 = TS DETUNE_CAP 镜像):烤制基线与 detune 必须同一个值,
        // 各自钳会在病理大 tuning 下基线错位。
        let clipped: Vec<(i64, i64, &UstxNote)> = abs
            .iter()
            .enumerate()
            .map(|(i, (t, n))| {
                let mut dur = n.duration.max(1);
                if let Some((next_t, _)) = abs.get(i + 1) {
                    if *next_t > *t {
                        dur = dur.min(next_t - t).max(1);
                    } else {
                        dur = 1; // 同 tick 重叠:保 1 tick(resolveOverlaps minTicks=1 同款)
                    }
                }
                (*t, dur, *n)
            })
            .collect();
        let tuning_of = |n: &UstxNote| -> f64 {
            n.tuning
                .map(|v| v as f64)
                .filter(|v| v.is_finite())
                .map(|v| v.clamp(-1200.0, 1200.0))
                .unwrap_or(0.0)
        };
        let ou_notes: Vec<super::ou_pitch::OuNote> = clipped
            .iter()
            .map(|(t, dur, n)| super::ou_pitch::OuNote {
                abs_tick: *t,
                duration: *dur,
                tone: n.tone.clamp(0, 127),
                tuning_cents: tuning_of(n),
                pitch_points: n
                    .pitch
                    .as_ref()
                    .map(|p| {
                        p.data
                            .iter()
                            .filter(|pp| pp.x.is_finite() && pp.y.is_finite())
                            .map(|pp| super::ou_pitch::OuPitchPoint {
                                x_ms: pp.x as f64,
                                y_tenths: pp.y as f64,
                                shape: super::ou_pitch::OuShape::parse(pp.shape.as_deref()),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                snap_first: n.pitch.as_ref().map(|p| p.snap_first).unwrap_or(true),
                vibrato: n.vibrato.as_ref().map(|v| super::ou_pitch::OuVibrato {
                    length: v.length as f64,
                    period: v.period as f64,
                    depth: v.depth as f64,
                    fade_in: v.fade_in as f64,
                    fade_out: v.out as f64,
                    shift: v.shift as f64,
                    drift: v.drift as f64,
                }),
            })
            .collect();
        let pitd = vp
            .curves
            .iter()
            .find(|c| c.abbr == "pitd" && !c.xs.is_empty() && c.xs.len() == c.ys.len())
            .map(|c| {
                // 防御:binary_search 前提=xs 升序;OU 恒有序,乱序文件排一遍不冤枉
                let mut pairs: Vec<(i64, i64)> =
                    c.xs.iter().copied().zip(c.ys.iter().copied()).collect();
                pairs.sort_by_key(|&(x, _)| x);
                pairs.dedup_by_key(|p| p.0);
                super::ou_pitch::OuPitd {
                    xs: pairs.iter().map(|&(x, _)| x).collect(),
                    ys: pairs.iter().map(|&(_, y)| y).collect(),
                }
            });
        let (pitch_dev, pitch_dev_dropped) =
            if super::ou_pitch::part_is_untuned(&ou_notes, pitd.as_ref()) {
                (None, false) // 完全未调教(默认零点滑音)→ 智能跳过,走我们的 SynthV 默认
            } else {
                match super::ou_pitch::bake_pitch_dev(&ou_notes, pitd.as_ref(), vp.position, base, file_bpm) {
                    super::ou_pitch::BakeOutcome::Baked { xs, ys } => (Some(ImportedCurve { xs, ys }), false),
                    super::ou_pitch::BakeOutcome::NoDiff => (None, false),
                    // 超上限:调教未导入,必须让前端提示(静默丢=与「未调教」不可区分,审查)
                    super::ou_pitch::BakeOutcome::Overflow => (None, true),
                }
            };
        let notes: Vec<ImportedNote> = clipped
            .iter()
            .map(|(t, dur, n)| {
                let lyric = n.lyric.trim();
                let lyric = if lyric.is_empty() { "あ".to_string() } else { lyric.to_string() };
                ImportedNote {
                    tick: t - base,
                    duration: *dur,
                    pitch: n.tone.clamp(0, 127),
                    lyric,
                    detune: {
                        let tn = tuning_of(n);
                        if tn != 0.0 { Some(tn as f32) } else { None }
                    },
                }
            })
            .collect();
        tracks.push(ImportedTrack { name, start_tick: base, notes, pitch_dev, pitch_dev_dropped });
    }

    Ok(ImportedScore { tracks, bpm, time_sig })
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// midi (binary SMF)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// Decode a MIDI text-meta payload: try UTF-8, fall back to Shift-JIS (common for Japanese MIDI lyrics).
fn decode_midi_text(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.trim().to_string(),
        Err(_) => encoding_rs::SHIFT_JIS.decode(bytes).0.trim().to_string(),
    }
}

/// Close the oldest open note-on for `key` (FIFO — first-on/first-off, the MIDI convention), pushing a
/// finished (start, end, pitch, lyric) tuple. Zero-length notes are dropped. A missing stashed lyric
/// (or an empty one) becomes the placeholder "あ" (lyric-less midi requirement).
fn close_note(
    open: &mut std::collections::HashMap<u8, Vec<(i64, Option<String>)>>,
    raw: &mut Vec<(i64, i64, i32, String)>,
    key: u8,
    end: i64,
) {
    if let Some(stack) = open.get_mut(&key) {
        if !stack.is_empty() {
            let (start, lyric_opt) = stack.remove(0);
            if end > start {
                let lyric = match lyric_opt {
                    Some(l) if !l.trim().is_empty() => l,
                    _ => "あ".to_string(),
                };
                raw.push((start, end, key as i32, lyric));
            }
        }
    }
}

fn parse_midi(bytes: &[u8]) -> Result<ImportedScore, String> {
    use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
    use std::collections::HashMap;

    let smf = Smf::parse(bytes).map_err(|e| format!("IMPORT_PARSE_MIDI: {e}"))?;
    let ppq: i64 = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as i64,
        Timing::Timecode(_, _) => {
            return Err("IMPORT_SMPTE".to_string())
        }
    };
    if ppq <= 0 {
        return Err("IMPORT_PPQ".to_string());
    }
    // our = round(midi_tick * 480 / ppq). midi ticks are non-negative (absolute, accumulated from 0).
    let to_our = |mt: i64| -> i64 { (mt * TICKS_PER_BEAT + ppq / 2) / ppq };

    // Tempo/time-sig: FIRST across all tracks (they usually live in a master/tempo track).
    let mut file_bpm: Option<f64> = None;
    let mut file_ts: Option<[u32; 2]> = None;
    let mut tracks: Vec<ImportedTrack> = Vec::new();

    for track in &smf.tracks {
        let mut cur: i64 = 0; // absolute midi ticks (accumulated deltas)
        let mut open: HashMap<u8, Vec<(i64, Option<String>)>> = HashMap::new();
        let mut stash_lyric: Option<String> = None;
        let mut track_name: Option<String> = None;
        let mut raw: Vec<(i64, i64, i32, String)> = Vec::new(); // (start, end, pitch, lyric) in midi ticks

        for ev in track.iter() {
            cur += ev.delta.as_int() as i64;
            match ev.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(us)) => {
                    if file_bpm.is_none() {
                        let us = us.as_int() as f64;
                        if us > 0.0 {
                            file_bpm = Some(60_000_000.0 / us);
                        }
                    }
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(num, denom_pow, _, _)) => {
                    if file_ts.is_none() {
                        let den = 1u32 << (denom_pow as u32).min(30);
                        file_ts = Some([num as u32, den]);
                    }
                }
                TrackEventKind::Meta(MetaMessage::TrackName(name)) => {
                    if track_name.is_none() {
                        track_name = Some(decode_midi_text(name));
                    }
                }
                TrackEventKind::Meta(MetaMessage::Lyric(text)) => {
                    // A Lyric meta precedes its note-on — stash it for the next note-on.
                    stash_lyric = Some(decode_midi_text(text));
                }
                TrackEventKind::Midi { message, .. } => match message {
                    MidiMessage::NoteOn { key, vel } => {
                        if vel.as_int() > 0 {
                            open.entry(key.as_int()).or_default().push((cur, stash_lyric.take()));
                        } else {
                            close_note(&mut open, &mut raw, key.as_int(), cur); // vel 0 = note-off
                        }
                    }
                    MidiMessage::NoteOff { key, .. } => {
                        close_note(&mut open, &mut raw, key.as_int(), cur);
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        if raw.is_empty() {
            continue; // tempo-only / empty track → not a vocal track
        }
        raw.sort_by_key(|(s, _, _, _)| *s);
        let base_our = to_our(raw[0].0);
        let notes: Vec<ImportedNote> = raw
            .iter()
            .map(|(s, e, pitch, lyric)| {
                let os = to_our(*s);
                let oe = to_our(*e);
                ImportedNote {
                    tick: os - base_our,
                    duration: (oe - os).max(1),
                    pitch: (*pitch).clamp(0, 127),
                    lyric: lyric.clone(),
                    detune: None,
                }
            })
            .collect();
        // no MIDI TrackName → empty; the frontend names nameless tracks by the file (+ index if several).
        let name = track_name.filter(|s| !s.trim().is_empty()).unwrap_or_default();
        tracks.push(ImportedTrack { name, start_tick: base_our, notes, pitch_dev: None, pitch_dev_dropped: false });
    }

    Ok(ImportedScore { tracks, bpm: file_bpm, time_sig: file_ts })
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ust_cumulative_position_and_leading_rest_offset() {
        // Leading R (Length 960) = the offset; then three notes; a mid rest (R, 240) becomes a gap.
        let ust = "\
[#VERSION]
UST Version1.2
Charset=UTF-8
[#SETTING]
Tempo=120
Tracks=1
[#0000]
Lyric=R
NoteNum=60
Length=960
[#0001]
Lyric=あ
NoteNum=60
Length=240
[#0002]
Lyric=い
NoteNum=62
Length=240
[#0003]
Lyric=r
NoteNum=60
Length=240
[#0004]
Lyric=う
NoteNum=64
Length=480
[#TRACKEND]
";
        let s = parse_ust(ust.as_bytes()).unwrap();
        assert_eq!(s.bpm, Some(120.0));
        assert_eq!(s.time_sig, None); // ust has no time signature
        assert_eq!(s.tracks.len(), 1);
        let t = &s.tracks[0];
        // First real note is at absolute tick 960 (after the leading rest) → segment start_tick = 960.
        assert_eq!(t.start_tick, 960);
        assert_eq!(t.notes.len(), 3); // the mid rest emits no note
        // Rebased ticks: 0, 240, then a 240 gap (the r rest), then 720.
        assert_eq!(t.notes[0].tick, 0);
        assert_eq!(t.notes[0].duration, 240);
        assert_eq!(t.notes[0].pitch, 60);
        assert_eq!(t.notes[0].lyric, "あ");
        assert_eq!(t.notes[1].tick, 240);
        assert_eq!(t.notes[1].pitch, 62);
        // note[2] follows the skipped rest: 240 (あ) + 240 (い) + 240 (rest) = 720.
        assert_eq!(t.notes[2].tick, 720);
        assert_eq!(t.notes[2].duration, 480);
        assert_eq!(t.notes[2].pitch, 64);
        assert!(t.pitch_dev.is_none()); // ust 无音高曲线通道
    }

    #[test]
    fn ust_shift_jis_decodes() {
        // "あ" in Shift-JIS is 0x82 0xA0. Build the bytes with a declared Shift-JIS charset.
        let mut bytes: Vec<u8> = b"[#SETTING]\nTempo=100\n[#0000]\nLyric=".to_vec();
        bytes.extend_from_slice(&[0x82, 0xA0]); // あ (Shift-JIS)
        bytes.extend_from_slice(b"\nNoteNum=60\nLength=240\n[#TRACKEND]\n");
        // Prepend a charset declaration so decode_ust_bytes picks Shift-JIS.
        let mut full = b"[#VERSION]\nCharset=Shift-JIS\n".to_vec();
        full.extend_from_slice(&bytes);
        let s = parse_ust(&full).unwrap();
        assert_eq!(s.tracks.len(), 1);
        assert_eq!(s.tracks[0].notes[0].lyric, "あ");
    }

    #[test]
    fn ustx_position_and_tempo_preference_and_vibrato_bakes() {
        // tempos[0] (156) must win over the legacy top-level bpm (120). time_signatures[0] wins too.
        // S73:带 vibrato 的 part = 已调教 → 整体烤入 pitchDev(不再映射 VibratoSpec)。
        let ustx = "\
ustx_version: \"0.6\"
resolution: 480
bpm: 120
beat_per_bar: 4
beat_unit: 4
tempos:
- position: 0
  bpm: 156
time_signatures:
- bar_position: 0
  beat_per_bar: 3
  beat_unit: 8
tracks:
- track_name: LEAD
voice_parts:
- name: PARTNAME
  track_no: 0
  position: 1000
  notes:
  - position: 200
    duration: 480
    tone: 65
    lyric: よ
    vibrato: {length: 50, period: 200, depth: 50, in: 10, out: 20, shift: 0, drift: 0, vol_link: 0}
  - position: 680
    duration: 240
    tone: 67
    lyric: い
    vibrato: {length: 0, period: 175, depth: 25, in: 10, out: 10, shift: 0, drift: 0, vol_link: 0}
";
        let s = parse_ustx(ustx.as_bytes()).unwrap();
        assert_eq!(s.bpm, Some(156.0)); // tempos[0] preferred over top-level 120
        assert_eq!(s.time_sig, Some([3, 8])); // time_signatures[0]
        assert_eq!(s.tracks.len(), 1);
        let t = &s.tracks[0];
        assert_eq!(t.name, "LEAD"); // tracks[track_no].track_name, not the part name
        // Absolute first note = position(1000) + note.position(200) = 1200 → start_tick, first note tick 0.
        assert_eq!(t.start_tick, 1200);
        assert_eq!(t.notes.len(), 2);
        assert_eq!(t.notes[0].tick, 0);
        assert_eq!(t.notes[0].pitch, 65);
        assert_eq!(t.notes[1].tick, 480); // 680 - 200
        // vibrato(length 50, depth 50)= 调教痕迹 → 烤入曲线:后半段应出现 ±≈50¢ 摆动
        let dev = t.pitch_dev.as_ref().expect("vibrato part must bake");
        assert!(dev.xs.windows(2).all(|w| w[0] < w[1]), "xs 严格递增");
        let max_abs = dev
            .xs
            .iter()
            .zip(dev.ys.iter())
            .filter(|(&x, _)| (240..480).contains(&x)) // note0 后半 = 振音区
            .map(|(_, &y)| y.abs())
            .max()
            .unwrap_or(0);
        assert!((40..=60).contains(&max_abs), "烤入的颤音深度≈50¢,得 {max_abs}");
    }

    #[test]
    fn ustx_untuned_part_skips_bake_and_tuning_maps_to_detune() {
        // 完全未调教(默认零 y 点 + 无可闻 vibrato + 无 pitd)→ 智能跳过;tuning → detune 无损映射
        let ustx = "\
bpm: 120
voice_parts:
- name: P
  position: 0
  notes:
  - position: 0
    duration: 480
    tone: 60
    lyric: あ
    tuning: 25
    pitch:
      data:
      - {x: -40, y: 0, shape: io}
      - {x: 40, y: 0, shape: io}
      snap_first: true
  - position: 480
    duration: 480
    tone: 64
    lyric: い
    pitch:
      data:
      - {x: -40, y: 0, shape: io}
      - {x: 40, y: 0, shape: io}
      snap_first: true
";
        let s = parse_ustx(ustx.as_bytes()).unwrap();
        let t = &s.tracks[0];
        assert!(t.pitch_dev.is_none(), "默认滑音形状不算调教");
        assert_eq!(t.notes[0].detune, Some(25.0));
        assert_eq!(t.notes[1].detune, None);
    }

    #[test]
    fn ustx_hand_tuned_points_and_pitd_bake() {
        // 手拖过的点(y≠0)+ pitd 手绘 → 烤入;曲线在 pitd 峰值处 ≈ 60¢(+点的贡献远处为 0)
        let ustx = "\
bpm: 120
voice_parts:
- name: P
  position: 0
  notes:
  - position: 0
    duration: 960
    tone: 60
    lyric: あ
    pitch:
      data:
      - {x: 0, y: 0, shape: io}
      - {x: 50, y: -8, shape: io}
      - {x: 120, y: 0, shape: io}
      snap_first: false
  curves:
  - xs: [600, 700, 800]
    ys: [0, 60, 0]
    abbr: pitd
";
        let s = parse_ustx(ustx.as_bytes()).unwrap();
        let t = &s.tracks[0];
        let dev = t.pitch_dev.as_ref().expect("nonzero point y / pitd must bake");
        // pitd 峰值(t=700,远离音高点影响区 ≤120ms≈96t)
        let near_peak = dev
            .xs
            .iter()
            .zip(dev.ys.iter())
            .filter(|(&x, _)| (680..=720).contains(&x))
            .map(|(_, &y)| y)
            .max()
            .unwrap_or(0);
        assert!((55..=62).contains(&near_peak), "pitd 峰值≈60¢,得 {near_peak}");
        // 音高点区:y=-8 tenths = −80¢ 谷(x=50ms@120bpm=40t 附近)
        let near_dip = dev
            .xs
            .iter()
            .zip(dev.ys.iter())
            .filter(|(&x, _)| (20..=70).contains(&x))
            .map(|(_, &y)| y)
            .min()
            .unwrap_or(0);
        assert!((-85..=-60).contains(&near_dip), "点谷≈−80¢,得 {near_dip}");
    }

    #[test]
    fn ustx_multi_part_splits_and_skips_empty() {
        let ustx = "\
bpm: 100
tracks:
- track_name: A
- track_name: B
voice_parts:
- name: partA
  track_no: 0
  position: 0
  notes:
  - {position: 0, duration: 240, tone: 60, lyric: a}
- name: emptyPart
  track_no: 1
  position: 0
  notes: []
- name: partB
  track_no: 1
  position: 480
  notes:
  - {position: 0, duration: 240, tone: 62, lyric: b}
";
        let s = parse_ustx(ustx.as_bytes()).unwrap();
        assert_eq!(s.bpm, Some(100.0));
        assert_eq!(s.time_sig, None); // no time_signatures / legacy meter present
        assert_eq!(s.tracks.len(), 2); // the empty part is skipped
        assert_eq!(s.tracks[0].name, "A");
        assert_eq!(s.tracks[0].start_tick, 0);
        assert_eq!(s.tracks[1].name, "B");
        assert_eq!(s.tracks[1].start_tick, 480);
    }

    #[test]
    fn midi_ppq_scaling_lyrics_placeholder_and_multitrack() {
        use midly::num::{u15, u24, u28, u4, u7};
        use midly::{Header, Format, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind};

        // PPQ = 240 → our tick = midi * 2. Build 3 tracks: a tempo/meter master, a note track with a
        // lyric, and a note track WITHOUT a lyric (→ placeholder あ).
        let mut smf = Smf::new(Header::new(Format::Parallel, Timing::Metrical(u15::new(240))));

        // Track 0: tempo (120 bpm = 500000 us/qn) + time-sig 3/4, no notes → skipped.
        let mut master: Track = Vec::new();
        master.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(500_000))) });
        master.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::TimeSignature(3, 2, 24, 8)) });
        master.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) });
        smf.tracks.push(master);

        // Track 1: name "melody"; a lyric then a note-on at midi-tick 240 for 240 ticks.
        let mut mel: Track = Vec::new();
        mel.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::TrackName(b"melody")) });
        mel.push(TrackEvent { delta: u28::new(240), kind: TrackEventKind::Meta(MetaMessage::Lyric("か".as_bytes())) });
        mel.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Midi { channel: u4::new(0), message: MidiMessage::NoteOn { key: u7::new(60), vel: u7::new(100) } } });
        mel.push(TrackEvent { delta: u28::new(240), kind: TrackEventKind::Midi { channel: u4::new(0), message: MidiMessage::NoteOff { key: u7::new(60), vel: u7::new(0) } } });
        mel.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) });
        smf.tracks.push(mel);

        // Track 2: no name, a note WITHOUT a lyric at midi-tick 0 for 480 ticks (→ placeholder あ).
        let mut plain: Track = Vec::new();
        plain.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Midi { channel: u4::new(0), message: MidiMessage::NoteOn { key: u7::new(72), vel: u7::new(90) } } });
        plain.push(TrackEvent { delta: u28::new(480), kind: TrackEventKind::Midi { channel: u4::new(0), message: MidiMessage::NoteOn { key: u7::new(72), vel: u7::new(0) } } });
        plain.push(TrackEvent { delta: u28::new(0), kind: TrackEventKind::Meta(MetaMessage::EndOfTrack) });
        smf.tracks.push(plain);

        let mut buf: Vec<u8> = Vec::new();
        smf.write(&mut buf).unwrap();

        let s = parse_midi(&buf).unwrap();
        assert_eq!(s.bpm, Some(120.0));
        assert_eq!(s.time_sig, Some([3, 4]));
        assert_eq!(s.tracks.len(), 2); // master (no notes) skipped
        // Track "melody": note at midi-tick 240 → our 480 → rebased to tick 0 (it's the first note).
        let mel = &s.tracks[0];
        assert_eq!(mel.name, "melody");
        assert_eq!(mel.start_tick, 480); // 240 midi * 480/240
        assert_eq!(mel.notes.len(), 1);
        assert_eq!(mel.notes[0].tick, 0);
        assert_eq!(mel.notes[0].duration, 480); // 240 midi ticks * 2
        assert_eq!(mel.notes[0].pitch, 60);
        assert_eq!(mel.notes[0].lyric, "か");
        // Track 2: lyric-less note → placeholder あ; PPQ scaling 480 midi → 960 our.
        let plain = &s.tracks[1];
        assert_eq!(plain.name, ""); // no TrackName → empty; the frontend names it by the file (+ index)
        assert_eq!(plain.start_tick, 0);
        assert_eq!(plain.notes[0].lyric, "あ");
        assert_eq!(plain.notes[0].duration, 960);
        assert_eq!(plain.notes[0].pitch, 72);
    }

    #[test]
    fn unsupported_extension_errors() {
        // (async command wrapper is thin; the dispatch is covered by exercising the parsers directly.)
        // A non-note ustx (no voice_parts) yields empty tracks — the command turns that into an error.
        let s = parse_ustx(b"bpm: 120\n").unwrap();
        assert!(s.tracks.is_empty());
    }

    // ── Integration: parse the REAL test files. Run with `cargo test -- --ignored import`. ──
    #[test]
    #[ignore]
    fn real_files_stats() {
        let cases: &[(&str, &str)] = &[
            ("main.ust", r"D:\MyDev\TESTING\虽然歌声无形ust\main.ust"),
            ("harm.ust", r"D:\MyDev\TESTING\虽然歌声无形ust\harm.ust"),
            ("fuel.ustx", r"D:\MyDev\TESTING\fuel_ustx\fuel.ustx"),
            ("虽然歌声无形.mid", r"D:\MyDev\TESTING\虽然歌声无形.mid"),
        ];
        for (label, path) in cases {
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[import] SKIP {label}: {e}");
                    continue;
                }
            };
            let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("");
            let score = match ext.to_ascii_lowercase().as_str() {
                "ust" => parse_ust(&bytes),
                "ustx" => parse_ustx(&bytes),
                "mid" | "midi" => parse_midi(&bytes),
                _ => panic!("unknown ext"),
            }
            .unwrap_or_else(|e| panic!("[import] parse {label} failed: {e}"));

            eprintln!("[import] {label}: bpm={:?} time_sig={:?} tracks={}", score.bpm, score.time_sig, score.tracks.len());
            assert!(!score.tracks.is_empty(), "{label}: no tracks");
            for (i, t) in score.tracks.iter().enumerate() {
                let last_end = t.notes.iter().map(|n| n.tick + n.duration).max().unwrap_or(0);
                let dev_pts = t.pitch_dev.as_ref().map(|c| c.xs.len()).unwrap_or(0);
                eprintln!(
                    "[import]   track[{i}] name={:?} start_tick={} notes={} first_tick={} last_end={} baked_dev_pts={}",
                    t.name, t.start_tick, t.notes.len(), t.notes.first().map(|n| n.tick).unwrap_or(-1), last_end, dev_pts,
                );
                assert!(!t.notes.is_empty(), "{label} track[{i}]: no notes");
                // Rebase invariant: the first note (tick order) is at tick 0.
                let min_tick = t.notes.iter().map(|n| n.tick).min().unwrap();
                assert_eq!(min_tick, 0, "{label} track[{i}]: first note not rebased to 0");
                assert!(t.notes.iter().all(|n| (0..=127).contains(&n.pitch)), "{label}: pitch out of range");
                assert!(t.notes.iter().all(|n| n.duration >= 1), "{label}: non-positive duration");
            }
            assert!(score.bpm.is_some(), "{label}: bpm not parsed");
        }
    }
}
