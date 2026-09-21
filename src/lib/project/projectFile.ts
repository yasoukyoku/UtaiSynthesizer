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
// NOTE deliberate runtime-only import cycle: midiExtract.ts imports getLoadEpoch from this
// module. Neither side calls the other during module EVALUATION (both are plain function
// references used at runtime), which ESM resolves safely.
import { cancelExtractionsForTeardown } from "../vocal/midiExtract";
import { clearClipboard } from "../clipboard";
import { fitTimelineToContent } from "../timeline/fitView";
import { buildSaveBundle, parseLoadedBundle, type LoadedProject } from "./bundle";
import { hasUnsavedWork, isRecoveryPending, markAutosaveBaseline } from "./autosave";
import { healLoadedTrackAvatars } from "../workflow/modelPathHeal";
import { useSoundfontStore, defaultSoundfontFor, firstPresetOf } from "../../store/soundfont";
import { buildDemoTracks } from "./demoContent";
import { rememberRecentProject } from "./recentProjects";

const t = (k: string) => i18n.t(k);

/** 启动欢迎页在工程被加载/保存时自动关闭(File 菜单、快捷键等旁路入口)。事件在 DOM 提交
 *  之后派发;欢迎页自身发起的动作已先行 onClose,重复关闭是无害幂等。 */
function announceDocumentLoaded(): void {
  try {
    window.dispatchEvent(new Event("utai:project-loaded"));
  } catch {
    /* 非 DOM 环境(测试) — 无监听者 */
  }
}

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
  // keep running and phantom-list in the quit/busy warning. cancelExecution only FLAGS the run (S62b:
  // the entry stays "running" until the engine loop obeys the cancel and settles it — typically within
  // seconds; the quit/busy warning stays honest for exactly that window). Fire the global separation
  // cancel best-effort (the whole project + its single render is being replaced).
  const wf = useWorkflowStore.getState();
  let hadRunning = false;
  for (const [id, e] of Object.entries(wf.executions)) {
    if (e.status === "running") { wf.cancelExecution(id); hadRunning = true; }
  }
  if (hadRunning) {
    void invoke("cancel_separation").catch(() => {});
    void invoke("cancel_voice").catch(() => {}); // voice runs are direct awaits — flag is the only abort
  }
  // S60: same for in-flight MIDI extractions — the old document's jobs would keep burning CPU for
  // minutes (results dropped by the epoch guard anyway) AND leave the undo interceptor armed to eat
  // the NEW document's first Ctrl+Z. Cancels + clears job state + unregisters, no toast.
  cancelExtractionsForTeardown();
  // The workflow store's per-segment SESSION render state (executions / node cache / badges /
  // renderLinks) belongs to the OLD document. Keeping it was inert while the settle watcher only keyed
  // on loading lanes (saves strip those), but hasUndepositedCache (S62) compares the surviving cache
  // against freshly LOADED deposits — same segment ids (re-open the same .usp) would silently
  // re-deposit a render the user just DISCARDED. Keep only the entries cancelled above: the detached
  // engine loops poll isCancelled() to abort, and their cache is wiped here so the empty-cache
  // pre-gate keeps them inert.
  useWorkflowStore.setState((s) => ({
    executions: Object.fromEntries(Object.entries(s.executions).filter(([, e]) => e.cancelled === true)),
    nodeOutputs: {},
    nodeStatuses: {},
    nodeProgress: {},
    nodeErrors: {},
    renderLinks: {},
  }));
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
  // S61: the arrangement clipboard holds the OLD document's render paths / mixer entries / sigs —
  // pasting them into a different project would resurrect half-broken caches. Cleared like node undo.
  clearClipboard();
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
    fitTimelineToContent(); // empty → reset zoom to 1×
    announceDocumentLoaded(); // 欢迎页若还开着 → 让位关闭
  } catch (e) {
    reportError(e);
  } finally {
    busy = false;
  }
}

/** 一键加载示例工程（File 菜单 / 新手入口）：8 小节四轨模板，新建语义 ——
 *  复用 new/open 的同一套 busy guard + 丢弃确认 + teardown（音频/编辑器/剪贴板状态
 *  全部按"换文档"清理），历史重置（不可 undo 回加载前，与每个 DAW 一致）。 */
export async function loadDemoProject(): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    if (!(await confirmDiscardIfDirty())) return;
    // 音色分配在 teardown 之前刷新（不依赖当前文档，失败 → 轨道不挂音色，不阻塞）。
    const kindByTrack: Array<"piano" | "chords" | "bass" | "drums"> = ["piano", "chords", "bass", "drums"];
    const assigns: Array<{ fontId: string; presetId: string; presetName: string } | undefined> = [];
    try {
      await useSoundfontStore.getState().refresh();
      const fonts = useSoundfontStore.getState().fonts;
      for (const kind of kindByTrack) {
        const font = defaultSoundfontFor(kind, fonts);
        const preset = font ? firstPresetOf(font) : null;
        assigns.push(font && preset ? { fontId: font.id, presetId: preset.id, presetName: preset.name } : undefined);
      }
    } catch {
      // 列不出音源 → 四轨全部不挂音色（轨道头占位提示），流程继续。
    }
    teardownForLoad();
    const tracks = buildDemoTracks();
    tracks.forEach((tr, i) => { const sf = assigns[i]; if (sf) tr.soundfont = sf; });
    useProjectStore.setState({
      name: "Demo · 儿歌示例", filePath: null, tracks,
      tempo: 120, timeSignature: [4, 4], dirty: false, playheadTick: 0,
    });
    useAppStore.getState().clearSelection();
    useHistoryStore.getState().reset();
    useHistoryStore.getState().markSaved();
    void markAutosaveBaseline();
    fitTimelineToContent();
    announceDocumentLoaded(); // 欢迎页若还开着 → 让位关闭
    useAppStore.getState().showBanner(t("demo.loaded"), "load");
  } catch (e) {
    reportError(e);
  } finally {
    busy = false;
  }
}

export async function openProjectFile(): Promise<void> {
  if (busy) return;
  const sel = await openDialog({
    title: t("project.openTitle"),
    directory: false,
    multiple: false,
    filters: [{ name: "MunoAI Project", extensions: ["usp"] }],
  });
  if (!sel || typeof sel !== "string") return;
  await openProjectFromPath(sel);
}

/** 直接打开指定 .usp(启动欢迎页"最近打开"卡片,无文件对话框)。与 openProjectFile 共用同一
 *  busy guard + 丢弃确认 + teardown 纪律。返回 false = 用户取消/打开失败(欢迎页保持打开)。 */
export async function openProjectFromPath(uspPath: string): Promise<boolean> {
  if (busy) return false;
  busy = true;
  try {
    if (!(await confirmDiscardIfDirty())) return false;
    // Extract the archive (to a work dir) BEFORE tearing down the current project, so a bad/missing
    // archive leaves the open project intact.
    const opened = await invoke<{ work_dir: string; project_json: string }>("open_project_archive", { uspPath });
    const loaded = parseLoadedBundle(opened.project_json, opened.work_dir);
    // S64 portability: avatar paths persist absolute — re-resolve from the singer registry so a
    // project from a moved install / another machine shows its avatars (history resets below, so
    // this can't create an undo step; dirty is explicitly set right after).
    await healLoadedTrackAvatars(loaded.tracks);
    teardownForLoad();
    useProjectStore.setState({
      name: loaded.name, filePath: uspPath, tracks: loaded.tracks,
      tempo: loaded.tempo, timeSignature: loaded.timeSignature, dirty: false, playheadTick: 0,
    });
    useAppStore.getState().clearSelection();
    // Opening a project = clean history; you can't undo back across the load (matches every DAW).
    useHistoryStore.getState().reset();
    useHistoryStore.getState().markSaved(); // the loaded state is the clean baseline
    void markAutosaveBaseline(); // opened a project — any previous recovery file is now obsolete
    fitTimelineToContent(); // default: see the whole song at a glance
    loadOriginalPeaks(loaded.tracks);
    // The load is COMMITTED — only now is it safe to reclaim older extractions. Rust deliberately
    // defers this cleanup to us: a failed open must never delete the previously-open project's
    // extracted media (see open_project_archive). Two guards: skip while a crash-recovery prompt is
    // still open (its media lives in usp_work — Ctrl+O works during the prompt), and AWAIT inside the
    // busy section so the prune can never race a subsequent open's staging extraction.
    if (!isRecoveryPending()) {
      await invoke("prune_usp_work", { keepDir: opened.work_dir }).catch(() => {});
    }
    rememberRecentProject(uspPath, loaded.name); // 欢迎页"最近打开"列表
    announceDocumentLoaded();
    useAppStore.getState().showBanner(`${t("project.loaded")} · ${loaded.name}`, "load");
    return true;
  } catch (e) {
    reportError(e);
    return false;
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
      filters: [{ name: "MunoAI Project", extensions: ["usp"] }],
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
    rememberRecentProject(uspPath, name); // 欢迎页"最近打开"列表
    announceDocumentLoaded();
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
export async function restoreAutosave(env: { filePath: string | null; name: string; projectJson: string }): Promise<void> {
  try {
    const loaded = parseLoadedBundle(env.projectJson, ""); // absolute media paths pass through untouched
    await healLoadedTrackAvatars(loaded.tracks); // same avatar re-resolve as openProjectFile
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
    fitTimelineToContent(); // default: see the whole song at a glance
    loadOriginalPeaks(loaded.tracks);
    announceDocumentLoaded(); // 恢复的工程接管界面 → 欢迎页让位关闭
    useAppStore.getState().showBanner(`${t("project.recovered")} · ${loaded.name}`, "load");
  } catch (e) {
    reportError(e);
  }
}

