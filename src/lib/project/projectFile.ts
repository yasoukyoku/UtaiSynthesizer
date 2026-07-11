import { save as saveDialog, open as openDialog } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import i18n from "../../i18n";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { useHistoryStore } from "../../store/history";
import { clearWaveformCache } from "../waveformCache";
import { stopPlayback, clearBufferCache } from "../audio/playback";
import { useWorkflowStore } from "../../store/workflow";
import { clearNodeHistories } from "../workflow/nodeHistory";
import { buildSaveBundle, parseLoadedBundle, type LoadedProject } from "./bundle";
import { hasUnsavedWork, isRecoveryPending, markAutosaveBaseline } from "./autosave";

const t = (k: string) => i18n.t(k);

/** Surface an error as a toast (one place for the repeated `e instanceof Error ? …` idiom). */
function reportError(e: unknown) {
  useAppStore.getState().showToast(e instanceof Error ? e.message : String(e), "error");
}

// Single in-flight guard: project I/O opens native dialogs and replaces the whole document; a second
// invocation (double shortcut, or menu + shortcut) while one is pending is ignored.
let busy = false;

/** Load waveform peaks for each ORIGINAL audio source so opened clips render (rendered lanes already
 *  carry their own saved peaks inline). Fire-and-forget; a missing/moved file just stays blank. */
function loadOriginalPeaks(tracks: LoadedProject["tracks"]) {
  const seen = new Set<string>();
  for (const tr of tracks) {
    for (const sg of tr.segments) {
      if (sg.content.type === "audioClip" && !seen.has(sg.content.sourcePath)) {
        seen.add(sg.content.sourcePath);
        void useAudioStore.getState().loadAudioFile(sg.content.sourcePath).catch(() => {});
      }
    }
  }
}

/** Prompt before discarding unsaved changes. Returns true if it's OK to proceed. Gates on the same
 *  DOCUMENT-level compare as the window-close flow (hasUnsavedWork), not just `dirty`: the dirty flag
 *  is recomputed from meaningfulSig on undo, which EXCLUDES baked renders — undoing back to the saved
 *  sig after a render would otherwise skip the prompt and silently discard the unsaved render (and
 *  markAutosaveBaseline would delete its recovery file too). */
async function confirmDiscardIfDirty(): Promise<boolean> {
  if (!useProjectStore.getState().dirty && !hasUnsavedWork()) return true;
  const choice = await useAppStore.getState().showConfirm({
    title: t("project.discardTitle"),
    body: t("project.discardBody"),
    buttons: [
      { id: "cancel", label: t("common.cancel") },
      { id: "discard", label: t("project.discard"), kind: "danger" },
    ],
  });
  return choice === "discard";
}

/** Folder name (minus the .usp suffix) → project display name. */
function deriveName(dir: string): string {
  const bn = dir.replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop() ?? "Untitled";
  return bn.replace(/\.usp$/i, "") || "Untitled";
}

/** S59: monotonic DOCUMENT-LOAD epoch. Async analysis/stretch flows capture it before their
 *  awaits and drop their store write if a load replaced the document mid-flight — matching ids +
 *  values alone can't tell "same project reopened" apart from "nothing changed" (audit). */
let loadEpoch = 0;
export function getLoadEpoch(): number {
  return loadEpoch;
}

/** Stop the transport and drop the previous project's audio caches before loading a different
 *  document — otherwise the old project keeps playing and its decoded buffers/peaks are served stale
 *  (and leak). Called only once we're committed to replacing the project. */
function teardownForLoad() {
  loadEpoch++;
  stopPlayback();
  useAudioStore.getState().setPlaying(false);
  clearWaveformCache();
  clearBufferCache(); // decoded AudioBuffer cache (playback.ts)
  useAudioStore.setState({ audioFiles: {}, loadingPaths: [] }); // decoded peaks/duration + in-flight markers
  // Cancel any in-flight render before discarding the document: its segment is about to vanish, so the
  // editor's Stop button becomes unreachable and the detached engine loop + global Rust separation would
  // keep running and phantom-list in the quit/busy warning. Flipping running→error un-sticks that warning;
  // fire the global separation cancel best-effort (the whole project + its single render is being replaced).
  const wf = useWorkflowStore.getState();
  let hadRunning = false;
  for (const [id, e] of Object.entries(wf.executions)) {
    if (e.status === "running") { wf.cancelExecution(id); hadRunning = true; }
  }
  if (hadRunning) {
    void invoke("cancel_separation").catch(() => {});
    void invoke("cancel_voice").catch(() => {}); // voice runs are direct awaits — flag is the only abort
  }
  // Close the docked node editor before the document is replaced: its segment is about to vanish (a
  // phantom panel would otherwise stay mounted), and a stale activePane:'workflow' would suppress
  // timeline Delete/Ctrl+K and misroute Ctrl+Z to the dead node stack. closeWorkflow() clears
  // workflowSegmentId (unmounts it) + resets activePane:'timeline'. Covers new/open/recover (all 3
  // route through here).
  useAppStore.getState().closeWorkflow();
  // ② Same for the docked vocal (piano-roll) editor: its notes segment is about to vanish; a stale
  // activePane:'vocal' would misroute Ctrl+Z + suppress timeline Delete, and dangling selectedNotes ids
  // would highlight ghosts. closeVocalEditor resets vocalSegmentId + activePane:'timeline' (§9.6).
  useAppStore.getState().closeVocalEditor();
  useProjectStore.getState().selectNotes([]);
  // Per-segment node-graph undo stacks reference the OLD document's segments — no undo across a load.
  clearNodeHistories();
}

export async function newProjectFile(): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    if (!(await confirmDiscardIfDirty())) return;
    teardownForLoad();
    useProjectStore.setState({
      name: "Untitled", filePath: null, tracks: [],
      tempo: 120, timeSignature: [4, 4], dirty: false, playheadTick: 0,
    });
    useAppStore.getState().clearSelection();
    useHistoryStore.getState().reset(); // a fresh project = clean history (no undo until a new edit)
    useHistoryStore.getState().markSaved(); // baseline = the empty project
    void markAutosaveBaseline(); // a fresh project — any previous recovery file is now obsolete
  } catch (e) {
    reportError(e);
  } finally {
    busy = false;
  }
}

export async function openProjectFile(): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    if (!(await confirmDiscardIfDirty())) return;
    const sel = await openDialog({
      title: t("project.openTitle"),
      directory: false,
      multiple: false,
      filters: [{ name: "UTAI Project", extensions: ["usp"] }],
    });
    if (!sel || typeof sel !== "string") return;
    // Extract the archive (to a work dir) BEFORE tearing down the current project, so a bad/missing
    // archive leaves the open project intact.
    const opened = await invoke<{ work_dir: string; project_json: string }>("open_project_archive", { uspPath: sel });
    const loaded = parseLoadedBundle(opened.project_json, opened.work_dir);
    teardownForLoad();
    useProjectStore.setState({
      name: loaded.name, filePath: sel, tracks: loaded.tracks,
      tempo: loaded.tempo, timeSignature: loaded.timeSignature, dirty: false, playheadTick: 0,
    });
    useAppStore.getState().clearSelection();
    // Opening a project = clean history; you can't undo back across the load (matches every DAW).
    useHistoryStore.getState().reset();
    useHistoryStore.getState().markSaved(); // the loaded state is the clean baseline
    void markAutosaveBaseline(); // opened a project — any previous recovery file is now obsolete
    loadOriginalPeaks(loaded.tracks);
    // The load is COMMITTED — only now is it safe to reclaim older extractions. Rust deliberately
    // defers this cleanup to us: a failed open must never delete the previously-open project's
    // extracted media (see open_project_archive). Two guards: skip while a crash-recovery prompt is
    // still open (its media lives in usp_work — Ctrl+O works during the prompt), and AWAIT inside the
    // busy section so the prune can never race a subsequent open's staging extraction.
    if (!isRecoveryPending()) {
      await invoke("prune_usp_work", { keepDir: opened.work_dir }).catch(() => {});
    }
    useAppStore.getState().showBanner(`${t("project.loaded")} · ${loaded.name}`, "load");
  } catch (e) {
    reportError(e);
  } finally {
    busy = false;
  }
}

/** True when there's nothing worth saving (an empty project) — saving it is meaningless. */
function isEmptyProject(): boolean {
  return useProjectStore.getState().tracks.length === 0;
}

/** Save to the current archive, or fall through to Save As if the project has never been saved. */
export async function saveProjectFile(): Promise<boolean> {
  if (busy) return false;
  if (isEmptyProject()) {
    useAppStore.getState().showBanner(t("project.emptyNoSave"), "info");
    return false;
  }
  const fp = useProjectStore.getState().filePath;
  if (!fp) return saveProjectFileAs();
  // If the saved archive no longer exists (the user deleted/moved it), don't silently re-create it at
  // the old path — prompt for a new location via Save As.
  if (!(await invoke<boolean>("path_exists", { path: fp }))) return saveProjectFileAs();
  busy = true;
  try {
    return await writeArchive(fp, false);
  } finally {
    busy = false;
  }
}

export async function saveProjectFileAs(): Promise<boolean> {
  if (busy) return false;
  if (isEmptyProject()) {
    useAppStore.getState().showBanner(t("project.emptyNoSave"), "info");
    return false;
  }
  busy = true;
  try {
    const name = useProjectStore.getState().name || "Untitled";
    const uspPath = await saveDialog({
      title: t("project.saveAsTitle"),
      defaultPath: `${name}.usp`,
      filters: [{ name: "UTAI Project", extensions: ["usp"] }],
    });
    if (!uspPath) return false;
    return await writeArchive(uspPath, true);
  } catch (e) {
    reportError(e);
    return false;
  } finally {
    busy = false;
  }
}

async function writeArchive(uspPath: string, rename: boolean): Promise<boolean> {
  const s = useProjectStore.getState();
  try {
    const { projectJson, copies } = buildSaveBundle(s.name || "Untitled", s.tracks, s.tempo, s.timeSignature);
    const missing = await invoke<string[]>("save_project_archive", { uspPath, projectJson, copies });
    // The single-file archive is self-contained on disk; the live session keeps its current media paths
    // (the open project's work dir, or external imports for a never-opened project) — no rebind needed.
    const name = rename ? deriveName(uspPath) : s.name;
    useProjectStore.setState({ filePath: uspPath, dirty: false, name });
    useHistoryStore.getState().markSaved(); // undoing back to here reads as "no unsaved changes"
    void markAutosaveBaseline(); // saved → nothing unsaved to recover
    if (missing.length > 0) {
      // The archive was written, but some referenced audio no longer existed on disk (cache sweep /
      // deleted source) and was skipped — a clean "saved" banner would read as everything intact.
      useAppStore.getState().showToast(`${t("project.saveMissing")} × ${missing.length}`, "error");
    } else {
      useAppStore.getState().showBanner(`${t("project.saved")} · ${name}`, "save");
    }
    return true;
  } catch (e) {
    reportError(e);
    return false;
  }
}

/** Restore a project from an autosave envelope left by an unclean exit. The recovered document is marked
 *  DIRTY (it was never saved to a real `.usp`) so the user is nudged to save it properly; the autosave
 *  file is left in place (it's still the snapshot of this as-yet-unsaved work). Media paths in the
 *  envelope are absolute, so `parseLoadedBundle` passes them through untouched. */
export function restoreAutosave(env: { filePath: string | null; name: string; projectJson: string }): void {
  try {
    const loaded = parseLoadedBundle(env.projectJson, ""); // absolute media paths pass through untouched
    teardownForLoad();
    useProjectStore.setState({
      name: loaded.name,
      filePath: env.filePath,
      tracks: loaded.tracks,
      tempo: loaded.tempo,
      timeSignature: loaded.timeSignature,
      dirty: true, // recovered work was never saved → keep it dirty until the user saves for real
      playheadTick: 0,
    });
    useAppStore.getState().clearSelection();
    useHistoryStore.getState().reset(); // fresh history; do NOT markSaved (savedSig stays null → stays dirty)
    loadOriginalPeaks(loaded.tracks);
    useAppStore.getState().showBanner(`${t("project.recovered")} · ${loaded.name}`, "load");
  } catch (e) {
    reportError(e);
  }
}
