//! Frontend IPC layer for Rust commands in `commands/song.rs`.
//!
//! Thin typed wrappers — params & return types mirror the Rust `Serialize` structs.
//! Every command name here must stay byte-identical to `lib.rs` `generate_handler!`.
import { invoke } from "@tauri-apps/api/core";
// ── Model directory / list / download / delete ──────────────────────────────
export function getSongModelsDir() {
    return invoke("get_song_models_dir");
}
export function listSongModels() {
    return invoke("list_song_models");
}
export function downloadSongModel(urls, id, filename, sha256) {
    return invoke("download_song_model", { urls, id, filename, sha256 });
}
export function deleteSongModel(filename) {
    return invoke("delete_song_model", { filename });
}
// ── History persistence ────────────────────────────────────────────────────
export function loadSongHistory() {
    return invoke("load_song_history");
}
export function saveSongHistory(entries) {
    return invoke("save_song_history", { entries });
}
// ── External inference service ──────────────────────────────────────────────
export function songServiceProbe(url) {
    return invoke("song_service_probe", { url });
}
export function songGenerate(req) {
    return invoke("song_generate", { req });
}
// ── ABC → MIDI ─────────────────────────────────────────────────────────────
export function abcToMidi(abc, outPath) {
    return invoke("abc_to_midi", { abc, outPath });
}
export function autoStartService(model) {
    return invoke("auto_start_service", { model });
}
