/**
 * AMT (Audio→MIDI Transcription) 相关 TS 类型.
 * 与 Rust 后端 amt.rs / amt_export.rs / amt_lyrics.rs / amt_models.rs 里的 #[derive(Serialize)] 严格对齐.
 * 所有 invoke<any> 都应该用这些类型代替.
 */

export interface AmtMidiResult {
  midi_path: string;
  processing_time_secs: number;
  total_notes?: number;
  vocal_midi_path?: string;
  accompaniment_midi_path?: string;
  merged_midi_path?: string;
  stem_midi_paths?: Record<string, string>;
  /** Stems 输出路径 (Rust 里 HashMap<String, StemStems>).
   *  key 是 stem 名 ("vocal"/"drums"/"bass"/"piano"/"guitar"/"other"), value 是音频绝对路径. */
  stems?: Record<string, string>;
  /** 原始音频路径 */
  source_audio_path?: string;
}

export interface AmtPlaybackResult {
  duration: number;
  transcription_wav: string;
  original_wav: string;
  midi_gain_db: number;
  instrument_wavs: Record<string, string>;
  /** 换声后的干声路径 */
  vocal_audio_path?: string;
}

export interface AmtMidiTrackInfo {
  name: string;
  note_count: number;
  program?: number;
  channel?: number;
}

export interface AmtMidiMetadata {
  track_count: number;
  total_notes: number;
  ppq: number;
  bpm?: number;
  time_signature?: [number, number];
  tracks: AmtMidiTrackInfo[];
  size_bytes: number;
}

export interface AmtNoteResult {
  tick: number;
  duration: number;
  pitch: number;
  velocity: number;
  lyric?: string;
}

export interface AmtTrackNotes {
  name: string;
  program?: number;
  notes: AmtNoteResult[];
  /** stem 输出音频路径 (从 MIDI metadata 里读 name 匹配 stems) */
  audio?: string;
  /** 音频轨道的类型标记 */
  stem_kind?: "vocal" | "drums" | "bass" | "piano" | "guitar" | "other";
}

export interface AmtStemAudioExportResult {
  zip_path: string;
  members: string[];
}

export interface AmtSheetMusicExportResult {
  zip_path: string;
  members: string[];
}

export interface AmtLyricSegment {
  start: number;
  end: number;
  text: string;
  language?: string;
  language_probability: number;
  duration: number;
  device: string;
  model: string;
}

export interface AmtLyricsResult {
  language?: string;
  language_probability: number;
  duration: number;
  device: string;
  model: string;
  segments: AmtLyricSegment[];
}

export interface AmtConvertProgress {
  phase: string;
  phase_index: number;
  total_phases: number;
  progress_pct: number;
  detail?: string;
}

/** AmtConversionDialog 里的 hardware_info */
export interface HardwareInfo {
  backend: string;          // "cuda" | "dml" | "cpu"
  device_name: string;
  memory_mb?: number;
  cuda_version?: string;
  ort_version: string;
}
