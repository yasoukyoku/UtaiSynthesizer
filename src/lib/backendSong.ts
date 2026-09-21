//! Frontend IPC layer for Rust commands in `commands/song.rs`.
//!
//! Thin typed wrappers — params & return types mirror the Rust `Serialize` structs.
//! Every command name here must stay byte-identical to `lib.rs` `generate_handler!`.

import { invoke } from "@tauri-apps/api/core";

// ── Types ──────────────────────────────────────────────────────────────────

export interface SongModelFile {
  filename: string;
  size: number;
}

export interface SongServiceProbe {
  ok: boolean;
  status: number;
  message: string;
}

export interface SongOutput {
  label: string;
  audio_path?: string;
  midi_path?: string;
  lrc_path?: string;
  abc_path?: string;
  stems?: Record<string, string>;
  processing_time_secs?: number;
}

export interface SongGenRequest {
  // Common parameters
  model: string;
  song_name: string;
  lyrics: string;
  prompt: string;
  audio_duration: number;
  seed?: number;
  cfg_scale?: number;
  format?: string;
  output_dir?: string;
  ref_audio_input?: string;
  audio2audio_enable?: boolean;
  
  // YuE-2 specific parameters
  cot?: string;
  abc?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  penalty_window?: number;
  min_tokens?: number;
  max_tokens?: number;
  ode_steps?: number;
  ode_method?: string;
  
  // ACE-Step specific parameters
  vram_mode?: string;
  inference_steps?: number;
  guidance_scale?: number;
  audio_format?: string;
  mp3_bitrate?: string;
  mp3_sample_rate?: number;
  task?: string;
  repaint_start?: number;
  repaint_end?: number;
  src_audio_path?: string;
  /** ACE-Step v1.5 多轨任务（lego/extract/complete）扩展字段 */
  track_name?: string;
  complete_track_classes?: string[];
  audio_cover_strength?: number;
  edit_target_prompt?: string;
  edit_target_lyrics?: string;
  edit_n_min?: number;
  edit_n_max?: number;
  edit_n_avg?: number;
  batch_size?: number;
  
  // HeartMuLa specific parameters
  topk?: number;
  
  // Output products (applies to all models)
  want_midi?: boolean;
  want_stems?: boolean;
  want_lrc?: boolean;
}

// ── Model directory / list / download / delete ──────────────────────────────

export function getSongModelsDir(): Promise<string> {
  return invoke<string>("get_song_models_dir");
}

export function listSongModels(): Promise<SongModelFile[]> {
  return invoke<SongModelFile[]>("list_song_models");
}

export function downloadSongModel(
  urls: string[],
  id: string,
  filename: string,
  sha256?: string,
): Promise<string> {
  return invoke<string>("download_song_model", { urls, id, filename, sha256 });
}

export function deleteSongModel(filename: string): Promise<void> {
  return invoke("delete_song_model", { filename });
}

// ── History persistence ────────────────────────────────────────────────────

export function loadSongHistory(): Promise<string> {
  return invoke<string>("load_song_history");
}

export function saveSongHistory(entries: unknown): Promise<void> {
  return invoke("save_song_history", { entries });
}

// ── External inference service ──────────────────────────────────────────────

export function songServiceProbe(url: string): Promise<SongServiceProbe> {
  return invoke<SongServiceProbe>("song_service_probe", { url });
}

export function songGenerate(req: SongGenRequest): Promise<SongOutput[]> {
  return invoke<SongOutput[]>("song_generate", { req });
}

// ── ABC → MIDI ─────────────────────────────────────────────────────────────

export function abcToMidi(abc: string, outPath: string): Promise<string> {
  return invoke<string>("abc_to_midi", { abc, outPath });
}

export function autoStartService(model: string): Promise<string> {
  return invoke<string>("auto_start_service", { model });
}
