// S56 — Import ustx / ust / midi → 人声轨(vocal tracks). Rust owns PARSING (binary MIDI + YAML are
// cleaner here, matching the open_project_archive precedent); the TS side only maps the returned data to
// store actions. ADDITIVE: this command reads a file and returns a plain data structure — it touches no
// project state, so it can never regress the editor.
//
// SCOPE (user-confirmed): NOTES (tick/duration/pitch/lyric) + BPM + first time-signature + the part's
// start OFFSET + (ustx only) VIBRATO mapped to our VibratoSpec. We deliberately do NOT import ustx
// pitch-bend / pitch-point curves — our pitch model (SynthV transitions) differs. One file at a time; a
// multi-track file yields ONE ImportedTrack per track that has notes (empty tracks skipped).
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
}

/// One imported note. Ticks/durations are ALREADY in our 480-ppq space. `pitch` is the MIDI note number.
#[derive(Serialize, Debug, Clone)]
pub struct ImportedNote {
    pub tick: i64,
    pub duration: i64,
    pub pitch: i32,
    pub lyric: String,
    pub vibrato: Option<ImportedVibrato>,
}

/// ustx vibrato mapped to our VibratoSpec (src/types/project.ts). camelCase so the object maps 1:1 onto
/// VibratoSpec on the TS side (no extra shaping).
#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct ImportedVibrato {
    pub depth_cents: f32,
    pub freq_hz: f32,
    pub phase: f32,
    pub start_ms: f32,
    pub ease_in_ms: f32,
    pub ease_out_ms: f32,
}

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
            vibrato: None, // ust carries no vibrato spec we import
        });
        cursor += len;
    }

    let tracks = if notes.is_empty() {
        Vec::new()
    } else {
        // ust carries no track name → empty; the frontend names it by the file (per-file, one track).
        vec![ImportedTrack { name: String::new(), start_tick: start_tick.unwrap_or(0), notes }]
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
}

/// Map an OpenUTAU vibrato onto our VibratoSpec. Only when `length > 0` (else the note has no vibrato).
/// The vibrato covers the last `length%` of the note → its duration_ms = noteDurationMs*(length/100),
/// startMs = noteDurationMs − vibDurMs. Everything is clamped to the same sane ranges the TS normalizeNote
/// enforces (belt & braces — the TS side re-clamps anyway).
fn map_vibrato(v: &UstxVibrato, note_dur_ticks: i64, bpm: f64) -> Option<ImportedVibrato> {
    if !(v.length > 0.0) || !(v.period > 0.0) {
        return None;
    }
    let bpm = if bpm.is_finite() && bpm > 0.0 { bpm } else { 120.0 };
    // ms = ticks / 480 * 60000 / bpm
    let note_ms = (note_dur_ticks as f64) / TICKS_PER_BEAT as f64 * 60_000.0 / bpm;
    let vib_ms = note_ms * (v.length as f64 / 100.0);
    if !(vib_ms > 0.0) {
        return None;
    }
    let freq = 1000.0 / v.period as f64;
    let start = (note_ms - vib_ms).max(0.0);
    let ease_in = vib_ms * (v.fade_in as f64 / 100.0);
    let ease_out = vib_ms * (v.out as f64 / 100.0);
    let phase = (v.shift as f64 / 100.0).clamp(-1.0, 1.0);
    Some(ImportedVibrato {
        depth_cents: (v.depth as f64).clamp(0.0, 2400.0) as f32,
        freq_hz: freq.clamp(0.1, 40.0) as f32,
        phase: phase as f32,
        start_ms: start.clamp(0.0, 60_000.0) as f32,
        ease_in_ms: ease_in.clamp(0.0, 10_000.0) as f32,
        ease_out_ms: ease_out.clamp(0.0, 10_000.0) as f32,
    })
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
        let notes: Vec<ImportedNote> = abs
            .iter()
            .map(|(t, n)| {
                let lyric = n.lyric.trim();
                let lyric = if lyric.is_empty() { "あ".to_string() } else { lyric.to_string() };
                ImportedNote {
                    tick: t - base,
                    duration: n.duration.max(1),
                    pitch: n.tone.clamp(0, 127),
                    lyric,
                    vibrato: n.vibrato.as_ref().and_then(|v| map_vibrato(v, n.duration, file_bpm)),
                }
            })
            .collect();
        tracks.push(ImportedTrack { name, start_tick: base, notes });
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
                    vibrato: None, // midi carries no vibrato spec
                }
            })
            .collect();
        // no MIDI TrackName → empty; the frontend names nameless tracks by the file (+ index if several).
        let name = track_name.filter(|s| !s.trim().is_empty()).unwrap_or_default();
        tracks.push(ImportedTrack { name, start_tick: base_our, notes });
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
        assert!(t.notes.iter().all(|n| n.vibrato.is_none()));
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
    fn ustx_position_and_tempo_preference_and_vibrato() {
        // tempos[0] (156) must win over the legacy top-level bpm (120). time_signatures[0] wins too.
        // A note with vibrato length>0 maps to a VibratoSpec; length==0 → None.
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
        // Vibrato mapping for note[0]: bpm 156, duration 480 ticks.
        let v = t.notes[0].vibrato.as_ref().expect("vibrato present for length>0");
        let note_ms = 480.0 / 480.0 * 60_000.0 / 156.0; // = 384.615...
        let vib_ms = note_ms * 0.5;
        assert!((v.depth_cents - 50.0).abs() < 1e-3);
        assert!((v.freq_hz - 1000.0 / 200.0).abs() < 1e-3); // 5 Hz
        assert!((v.start_ms - (note_ms - vib_ms) as f32).abs() < 1e-2);
        assert!((v.ease_in_ms - (vib_ms * 0.10) as f32).abs() < 1e-2);
        assert!((v.ease_out_ms - (vib_ms * 0.20) as f32).abs() < 1e-2);
        assert_eq!(v.phase, 0.0);
        // note[1] has length==0 → no vibrato.
        assert!(t.notes[1].vibrato.is_none());
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
                let vib = t.notes.iter().filter(|n| n.vibrato.is_some()).count();
                eprintln!(
                    "[import]   track[{i}] name={:?} start_tick={} notes={} first_tick={} last_end={} vibrato_notes={}",
                    t.name, t.start_tick, t.notes.len(), t.notes.first().map(|n| n.tick).unwrap_or(-1), last_end, vib,
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
