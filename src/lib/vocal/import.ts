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
import type { Note } from "../../types/project";
import { ZERO_TRANSITION } from "../vocalNotes";
import { blankTrack } from "../trackFactory";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { flushAutosaveNow } from "../project/autosave";
import { TICKS_PER_BEAT, SCORE_EXTENSIONS } from "../constants";
import { loadSetting, saveSetting } from "../settings";
import { quantizeSpans, offGridCount, QUANTIZE_IMPORT_KEY, type QuantSpan } from "./quantize";

const t = (k: string, vars?: Record<string, unknown>) => i18n.t(k, vars ?? {}) as string;

// ── Rust contract (see ImportedScore in import.rs). Fields are snake_case (serde default). ──
interface ImportedScore {
  tracks: ImportedTrack[];
  bpm: number | null;
  time_sig: [number, number] | null;
}
interface ImportedTrack {
  name: string;
  start_tick: number;
  notes: ImportedNote[];
  /** S73 烤入的 OU 音高曲线(segment 相对 tick / 整数 cents)。非空 = BAKED part:音符要置零
   *  transition(否则我们的默认滑音与曲线里的 OU 滑音双重叠加)。null = 未调教(智能跳过)/非 ustx。 */
  pitch_dev: { xs: number[]; ys: number[] } | null;
  /** part 有调教但超出可烤上限被放弃 → 须提示用户(绝不静默当未调教)。 */
  pitch_dev_dropped: boolean;
}
interface ImportedNote {
  tick: number;
  duration: number;
  pitch: number;
  lyric: string;
  /** ustx tuning(cents)→ Note.detune(无损旋钮映射)。 */
  detune: number | null;
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

    // S87 — grid rounding is the USER's call, per import. It CURES a score whose notes are shorter than one
    // 20 ms render frame (a 30t note @ tempo 222 = 16.9 ms rounds to ZERO frames and never sounds) and it
    // RUINS a UTAU CVVC score, whose off-grid starts ARE the alias author's preutterance compensation. The
    // dialog shows how much of THIS file is off-grid, so the choice is informed rather than a coin flip.
    const absSpans = (tk: ImportedTrack): QuantSpan[] =>
      tk.notes.map((n) => ({ start: tk.start_tick + n.tick, end: tk.start_tick + n.tick + n.duration }));
    const allSpans = score.tracks.filter((tk) => tk.notes.length).flatMap(absSpans);
    const off = offGridCount(allSpans);
    let quantize = loadSetting(QUANTIZE_IMPORT_KEY, true);
    const choice = await useAppStore.getState().showConfirm({
      title: t("import.options.title"),
      body: t("import.options.body", {
        total: allSpans.length,
        off,
        pct: allSpans.length ? Math.round((off / allSpans.length) * 100) : 0,
      }),
      check: {
        label: t("import.options.quantize"),
        initial: quantize,
        onChange: (v) => {
          quantize = v;
        },
      },
      buttons: [
        { id: "cancel", label: t("common.cancel") },
        { id: "ok", label: t("import.options.start"), kind: "primary" },
      ],
    });
    if (choice !== "ok") return; // Cancel / Esc / backdrop — nothing was touched yet
    saveSetting(QUANTIZE_IMPORT_KEY, quantize);

    const store = useProjectStore.getState();
    const history = useHistoryStore.getState();
    let movedTotal = 0;
    let widenedTotal = 0;
    let skippedBaked = 0;

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

        // S73:BAKED part(带烤入曲线)的音符显式置零 transition = 纯阶梯——OU 的滑音/颤音/手绘
        // 全在 pitchDev 曲线里,我们的默认滑音再叠加=双重;零 transition 也让这些音符被
        // 自动调教的所有权谓词识别为「用户调教过」(导入的调教=用户资产,autoTune 绕行)。
        const baked = !!it.pitch_dev && it.pitch_dev.xs.length > 0;
        // S87 rounding, in ABSOLUTE tick space (the grid lines are absolute; a part start is wherever its
        // first note is, so rounding relative ticks would land notes on lines that are not the drawn ones).
        // ⚠ A BAKED part is EXEMPT: its hand-drawn pitch curve is keyed to the ORIGINAL note timing, so
        // moving the notes out from under it would desynchronize the two. Reported, never silently skipped.
        const doQuant = quantize && !baked;
        if (quantize && baked) skippedBaked++;
        let partStart = it.start_tick;
        let placed = it.notes.map((n) => ({ tick: n.tick, duration: n.duration }));
        if (doQuant) {
          // "widen" — an imported note carries a LYRIC, so a collapsed one is lengthened, never dropped
          // (see CollapsePolicy); that policy also guarantees every index survives.
          const { spans, moved, widened } = quantizeSpans(absSpans(it), "widen");
          partStart = spans.reduce((m, s) => Math.min(m, s!.start), Infinity);
          placed = spans.map((s) => ({ tick: s!.start - partStart, duration: s!.end - s!.start }));
          movedTotal += moved;
          widenedTotal += widened;
        }

        // Part spans from the first note (rebased to tick 0) to the last note's end.
        const lastEnd = placed.reduce((m, p) => Math.max(m, p.tick + p.duration), 0);
        const durationTicks = lastEnd > 0 ? lastEnd : TICKS_PER_BEAT;
        const segId = store.createVocalPart(trackId, partStart, durationTicks);

        const notes: Note[] = it.notes.map((n, i) => ({
          id: crypto.randomUUID(),
          tick: placed[i]!.tick,
          duration: placed[i]!.duration,
          pitch: n.pitch,
          lyric: n.lyric,
          velocity: 100,
          ...(n.detune != null && n.detune !== 0 ? { detune: n.detune } : {}),
          ...(baked ? { transition: { ...ZERO_TRANSITION } } : {}),
        }));
        store.applyNoteEdits(trackId, segId, { add: notes });
        if (baked && it.pitch_dev) {
          // 走 normalizeCurve 漏斗(整数 cents/排序/去重);同一事务 = 同一步 undo
          store.setSegmentPitchDev(trackId, segId, { xs: it.pitch_dev.xs, ys: it.pitch_dev.ys });
        }
      }
    } finally {
      history.commitTransaction();
    }

    // Import is a milestone — snapshot to disk NOW so a fast reload doesn't lose it to the autosave debounce.
    flushAutosaveNow();
    useAppStore.getState().showBanner(`${t("import.done")} · ${score.tracks.length}`, "load");
    // S87: say OUT LOUD what rounding did. A moved note, a note widened off a collapse, or a part exempted
    // for its baked pitch curve must never be a silent surprise (the S84/S85 silent-drop lesson).
    const report: string[] = [];
    if (movedTotal > 0) report.push(t("import.quantMoved", { count: movedTotal }));
    if (widenedTotal > 0) report.push(t("import.quantWidened", { count: widenedTotal }));
    if (skippedBaked > 0) report.push(t("import.quantSkippedBaked", { count: skippedBaked }));
    if (report.length) useAppStore.getState().showToast(report.join(" · "), "info");
    // S73:某 part 的调教超出可烤上限被放弃 → 响亮告知(音符已导入,音高线没有)。
    if (score.tracks.some((tk) => tk.pitch_dev_dropped)) {
      useAppStore.getState().showToast(t("import.pitchDropped"), "error");
    }
  } catch (e) {
    useAppStore.getState().showToast(mapImportError(e instanceof Error ? e.message : String(e)), "error");
  } finally {
    busy = false;
  }
}
