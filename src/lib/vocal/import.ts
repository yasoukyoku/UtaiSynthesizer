// S56 — Import ustx / ust / midi → 人声轨. The TS side is a thin MAPPER: Rust parses the file (see
// src-tauri/src/commands/import.rs) and returns plain data; here we turn that into store actions. ADDITIVE
// (creates NEW vocal tracks — never replaces the document, so no discard prompt / teardown like open).
//
// The whole import is coalesced into ONE undo step (beginTransaction/commitTransaction) so a single Ctrl+Z
// removes it. Notes funnel through the store's applyNoteEdits (→ normalizeNotesArray), which clamps
// tick≥0 / duration≥1 / pitch[0,127] / velocity — so we DON'T pre-clamp here.

import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import i18n from "../../i18n";
import type { Note, VibratoSpec } from "../../types/project";
import { blankTrack } from "../trackFactory";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { flushAutosaveNow } from "../project/autosave";
import { TICKS_PER_BEAT, SCORE_EXTENSIONS } from "../constants";

const t = (k: string) => i18n.t(k);

// ── Rust contract (see ImportedScore in import.rs). Outer fields are snake_case (serde default);
//    ImportedVibrato is camelCase → maps 1:1 onto VibratoSpec. ──
interface ImportedScore {
  tracks: ImportedTrack[];
  bpm: number | null;
  time_sig: [number, number] | null;
}
interface ImportedTrack {
  name: string;
  start_tick: number;
  notes: ImportedNote[];
}
interface ImportedNote {
  tick: number;
  duration: number;
  pitch: number;
  lyric: string;
  vibrato: VibratoSpec | null; // camelCase from Rust == VibratoSpec shape exactly
}

// Single in-flight guard: the import opens a native dialog + mutates the document; a second invocation
// (double-click the menu item) while one is pending is ignored.
let busy = false;

/** Map a backend import error CODE → a localized toast. Rust returns stable CODEs (never hardcoded Chinese —
 *  the S56 i18n rule); some carry a "CODE: detail" suffix (the parse/IO error text) which we append. */
function mapImportError(msg: string): string {
  const codes: Record<string, string> = {
    IMPORT_UNSUPPORTED: "unsupported",
    IMPORT_READ_FAIL: "readFail",
    IMPORT_PARSE_USTX: "parseUstx",
    IMPORT_PARSE_MIDI: "parseMidi",
    IMPORT_SMPTE: "smpte",
    IMPORT_PPQ: "ppq",
    IMPORT_EMPTY: "empty",
  };
  for (const code of Object.keys(codes)) {
    const at = msg.indexOf(code);
    if (at < 0) continue;
    const detail = msg.slice(at + code.length).replace(/^\s*[:：]\s*/, "").trim();
    const base = t(`import.error.${codes[code]}`);
    return detail ? `${base}: ${detail}` : base;
  }
  return `${t("import.error.generic")}: ${msg}`;
}

/**
 * File-menu "导入 / Import": pick a .ustx/.ust/.mid/.midi, parse it in Rust, and build one NEW vocal track
 * per track that has notes. BPM / time-sig FOLLOW the file and OVERRIDE the project globally (only when the
 * file carries them). Each track's notes are rebased so its created part starts at the first note.
 */
export async function importScoreFile(): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    const sel = await open({
      title: t("menu.import"),
      directory: false,
      multiple: false,
      filters: [{ name: "Score", extensions: SCORE_EXTENSIONS }],
    });
    if (!sel || typeof sel !== "string") return;

    const score = await invoke<ImportedScore>("import_score_file", { path: sel });
    if (!score.tracks.length) {
      useAppStore.getState().showToast(t("import.error.empty"), "error");
      return;
    }

    const store = useProjectStore.getState();
    const history = useHistoryStore.getState();

    // ONE undo step for the whole import (tempo/meter override + every track + notes).
    history.beginTransaction();
    try {
      // BPM / time-sig follow the file and override globally — but ONLY when the file carries them (a file
      // with no tempo/meter leaves the editor as-is). setTempo/setTimeSignature no-op on an unchanged value.
      if (score.bpm != null && Number.isFinite(score.bpm) && score.bpm > 0) store.setTempo(score.bpm);
      if (score.time_sig) store.setTimeSignature(score.time_sig[0], score.time_sig[1]);

      // Track naming: prefer the file's OWN track name (ustx track_name, midi TrackName); a nameless track
      // (every ust; a nameless midi/ustx track) is named by the FILE basename — with a 1-based index when
      // several nameless tracks live in one file (so importing a multi-track file doesn't yield "song song").
      const fileBase = sel.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "").trim() || "Vocal";
      const namelessTotal = score.tracks.filter((tk) => tk.notes.length && !tk.name.trim()).length;
      let namelessIdx = 0;

      for (const it of score.tracks) {
        if (!it.notes.length) continue; // defensive: Rust already skips empty tracks
        const trackId = crypto.randomUUID();
        const named = it.name.trim();
        const name = named || (namelessTotal > 1 ? `${fileBase} ${++namelessIdx}` : fileBase);
        store.addTrack(blankTrack(trackId, name, "vocal"));

        // Part spans from the first note (tick 0, rebased in Rust) to the last note's end.
        const lastEnd = it.notes.reduce((m, n) => Math.max(m, n.tick + n.duration), 0);
        const durationTicks = lastEnd > 0 ? lastEnd : TICKS_PER_BEAT;
        const segId = store.createVocalPart(trackId, it.start_tick, durationTicks);

        const notes: Note[] = it.notes.map((n) => ({
          id: crypto.randomUUID(),
          tick: n.tick,
          duration: n.duration,
          pitch: n.pitch,
          lyric: n.lyric,
          velocity: 100,
          ...(n.vibrato ? { vibrato: n.vibrato } : {}),
        }));
        store.applyNoteEdits(trackId, segId, { add: notes });
      }
    } finally {
      history.commitTransaction();
    }

    // Import is a milestone — snapshot to disk NOW so a fast reload doesn't lose it to the autosave debounce.
    flushAutosaveNow();
    useAppStore.getState().showBanner(`${t("import.done")} · ${score.tracks.length}`, "load");
  } catch (e) {
    useAppStore.getState().showToast(mapImportError(e instanceof Error ? e.message : String(e)), "error");
  } finally {
    busy = false;
  }
}
