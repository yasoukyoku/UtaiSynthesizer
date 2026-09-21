import { create } from "zustand";
import { loadSetting, saveSetting } from "../lib/settings";
import type { SongTaskId } from "../lib/models/song-tasks";
import type { SongSource } from "../lib/song/types";

/** 规划 13.2：全局"待启动歌曲任务"——DAW 右键发起，歌曲制作弹窗挂载时消费（带参打开）。 */
export interface PendingSongTask {
  task: SongTaskId;
  source: SongSource;
  sourceLabel?: string;
  /** repaint 选区（源音频秒区间） */
  range?: [number, number];
  /** lego/extract 单轨种 */
  trackClass?: string;
  /** complete/extract 多选轨种 */
  trackClasses?: string[];
  presetPrompt?: string;
  presetLyrics?: string;
  /** 回流目标：任务完成后结果可"发送回原轨道位置" */
  returnTarget?: { kind: "track"; trackId: string; segmentId?: string; align: boolean };
  tool?: "multiTrack" | "creative" | "cover" | "midi";
}

interface SegmentSelection {
  trackId: string;
  segmentId: string;
}

/** A selected sub-lane GROUP within a segment (P3). Output nodes are many-to-one: `outputNodeId` selects
 *  ALL lanes fanned into that Output node. `clipIndex` = the piece under the cursor at click time — the
 *  target for Ctrl+K slice / Delete. Distinct from segment selection; drives lane trim/slice/delete. */
interface LaneSelection {
  trackId: string;
  segmentId: string;
  outputNodeId: string;
  clipIndex: number;
}

interface ToastState {
  message: string;
  type: "error" | "info" | "success" | "warning";
  id: number;
}

export interface ConfirmButton {
  id: string;
  label: string;
  /** Visual emphasis — "primary" = accent, "danger" = destructive; omit for a neutral button. */
  kind?: "primary" | "danger";
}

/** Optional text-input mode for the confirm dialog (e.g. "new group" name prompt). When present, the
 *  PRIMARY button (and Enter) resolves with the TRIMMED input value instead of the button id; a
 *  non-empty value is always required, `invalid` adds extra validation (returns an error message to
 *  show, or null when ok). Cancel/Esc/backdrop still resolve "". */
export interface ConfirmInput {
  placeholder?: string;
  initial?: string;
  invalid?: (value: string) => string | null;
}

/** S87 — optional CHECKBOX row (the import / MIDI-extract "round note boundaries to the 1/12 grid" option).
 *  The dialog owns the checkbox state and reports every change through `onChange`, so the promise still
 *  resolves the BUTTON id and the existing `Promise<string>` contract (30+ call sites, and `input`'s
 *  value-overloading) is untouched — the caller keeps the flag in its own variable. Not combinable with
 *  `input` (no caller needs both; the dialog renders whichever is set). */
export interface ConfirmCheck {
  label: string;
  initial: boolean;
  onChange: (checked: boolean) => void;
}

interface ConfirmRequest {
  title: string;
  body: string;
  buttons: ConfirmButton[];
  input?: ConfirmInput;
  check?: ConfirmCheck;
  /** S74: error modals (maybeShowErrorModal) — long body scrolls, text is selectable, and a
   *  Copy button appears. Ordinary confirms/prompts leave it unset (compact, non-scrolling). */
  scrollable?: boolean;
  resolve: (id: string) => void;
  seq: number;
}

/** S66 pre-run model check: one actionable problem found by the workflow/vocal preflight.
 *  - msstConvert: installed MSST ckpt with no ONNX — one-click conversion (filename/precision/architecture)
 *  - msstMissing: node references a model file that is not installed — guide to the model manager
 *  - auxPack:     the core inference asset pack (aux-inference) has missing files — one-click download */
export interface MissingModelItem {
  kind: "msstConvert" | "msstMissing" | "auxPack";
  label: string;
  filename?: string;
  precision?: "fp32" | "fp16";
  architecture?: string;
}

export interface AmtResult {
  trackId: string;
  outputDir: string;
  midiPaths: string[];
  audioPreview?: string;
  originalAudio?: string;
  /** The raw source audio the conversion ran on — lets the result panel lazily
   *  re-run FluidSynth playback prep when audioPreview/originalAudio are empty. */
  sourceAudioPath?: string;
  /** 人声分离出的干音 WAV（vocal_split / six_stem 模式）—— 自动歌词提取最准的源。 */
  vocalAudioPath?: string;
  stems?: Record<string, { midi: string; audio?: string }>;
  /** True for separation-only results (vocal/six-stem): audio stems, no MIDI yet. */
  separationOnly?: boolean;
}

interface AppState {
  /** "dark" | "light" — 全局主题, 持久化到 localStorage, 首帧前注入防 FOUC */
  theme: "dark" | "light";
  setTheme: (t: "dark" | "light") => void;
  trainingPageOpen: boolean;
  modelManagerOpen: boolean;
  /** Muno 音源管理器(阶段1)是否打开——SoundfontManager 弹窗的宿主开关。 */
  soundfontManagerOpen: boolean;
  /** Song Studio (歌曲制作)主面板开关。 */
  songStudioOpen: boolean;
  /** 规划 13.2：待启动歌曲任务（DAW 右键 → 歌曲弹窗带参打开），弹窗消费即清。 */
  pendingSongTask: PendingSongTask | null;
  setPendingSongTask: (p: PendingSongTask) => void;
  clearPendingSongTask: () => void;
  logViewerOpen: boolean;
  settingsOpen: boolean;
  toggleSettings: () => void;
  activeTrackId: string | null;
  /** Primary/anchor selection (drives Toolbar split/delete). */
  selectedSegment: SegmentSelection | null;
  /** Full multi-selection set (Ctrl+click adds/removes; drives highlight + move-together). */
  selectedSegments: SegmentSelection[];
  /** Selected sub-lane group (P3), or null. Non-null makes Ctrl+K/Delete act on the lane, not the segment. */
  selectedLane: LaneSelection | null;
  workflowSegmentId: string | null;
  /** ② The notes segment whose VOCAL (piano-roll) editor is docked at the bottom, or null. Mutually
   *  exclusive with workflowSegmentId — the bottom dock shows one editor at a time (a segment is either a
   *  notes part or an audioClip). Mirrors workflowSegmentId (§9.6). */
  vocalSegmentId: string | null;
  /** AMT conversion result to show in the bottom panel. */
  amtResult: AmtResult | null;
  /** A requested Output-group DETACH ("ungroup") waiting for that segment's workflow editor to perform
   *  it — the editor is the ONE code path (its graph state + local undo own the op); a timeline
   *  right-click first opens the editor, then this hands the request over. Consumed on mount/change. */
  pendingLaneDetach: { segmentId: string; outputNodeId: string } | null;
  /** 超级原创向导：新导入片段的「挂载即自动执行」请求。一键向导建轨→写模板工作流→openWorkflow
   *  后置此标志；WorkflowEditor 挂载时消费并触发 handleExecute（复用全部 preflight/进度/取消
   *  逻辑），避免向导旁路执行引擎。一次性，消费即清。 */
  workflowAutoRun: string | null;
  clearWorkflowAutoRun: () => void;
  /** Which pane owns Ctrl+Z and the Delete/Ctrl+K edit keys: the track timeline, the bottom-docked
   *  workflow editor, the bottom-docked vocal (piano-roll) editor, or the AMT result panel. Set on
   *  pointer/focus into each pane. */
  activePane: "timeline" | "workflow" | "vocal" | "amt";
  /** Right-side track-inspector drawer: which track's properties are shown (null = closed).
   *  Studio Pro-style "click the icon → side panel pops out" UX. */
  inspectorTrackId: string | null;
  openInspector: (trackId: string) => void;
  closeInspector: () => void;
  toggleInspector: (trackId: string) => void;
  /** Height (px) of the bottom workflow panel when open; persisted across sessions. */
  workflowPanelHeight: number;
  /** ② Height (px) of the bottom vocal-editor panel when open; persisted (own value, §9.0). */
  vocalPanelHeight: number;
  /** Height (px) of the AMT result panel. */
  amtPanelHeight: number;
  zoom: number;
  /** Vertical zoom — scales track display height (header + lanes). */
  vZoom: number;
  scrollX: number;
  scrollY: number;
  canvasWidth: number;
  canvasHeight: number;
  /** While dragging a NEW track in between existing ones, the placeholder gap to open (so both the
   *  canvas and the track-header column show an empty slot at `index`, `count` rows tall). */
  ghostInsert: { index: number; count: number } | null;
  /** Snap dragged/resized clips to other clips' edges + the playhead. */
  snapSegments: boolean;
  /** Snap the playhead (drag on ruler / arrangement) to clip edges. */
  snapPlayhead: boolean;
  /** S87: ② vocal piano-roll GRID snapping. ON (default) = the old behavior — a drawn note lands on the
   *  selected grid cell, a move/resize steps by 1/12 (40t). OFF = fully CONTINUOUS placement (per-tick),
   *  for scores whose timing is AUTHORED off-grid (a UTAU CVVC .ust hand-shifts note starts as
   *  preutterance compensation: 82.6% of Main.ust's notes sit off the 1/12 grid ON PURPOSE). The 1/12
   *  grid LINES stay visible either way — with snapping off they are a reference, not a magnet. */
  snapNotes: boolean;
  /** ② A vocal (score→singing) render is in flight — GLOBAL single-flight: only one at a time, since the
   *  shared ORT engine + release_gpu_sessions_except would make concurrent renders evict each other's
   *  session mid-inference. Gates the Render button everywhere + backs the throw-guard in vocalRender.ts. */
  vocalRenderActive: boolean;
  /** ② The TRACK id whose vocal segment is CURRENTLY (re-)rendering — drives a spinner on that track's header
   *  so a re-render (which keeps the old bake, no loading placeholder) doesn't LOOK frozen (§user). null = idle. */
  renderingVocalTrackId: string | null;
  /** ② S58 OOV verdicts (runtime-only, not undoable/persisted): segmentId → the note ids whose lyric can't
   *  be sung in its effective language, from the debounced `validate_lyrics` watcher (oovWatch.ts — the
   *  §9.5 single Rust classifier, so this ALWAYS equals what the render would reject). Drives the red
   *  note marking (VocalEditor), the segment badge (Arrangement) and the track header warning (TrackList). */
  vocalOov: Record<string, string[]>;
  /** S109 (§C15): the SUBSET of `vocalOov` whose failure was a mistyped PHONEME (a bracket hint or a
   *  `phoneme_input` override carrying a symbol outside the 210-token inventory) rather than an
   *  unrecognized LYRIC. Same red mark, different sentence — the third instance of the split S85b
   *  made for `vocalDropped` and S87 for `vocalShort`, and for the identical reason recorded there:
   *  **a merged channel forces the wording to lie about at least one of its members.** Here it was
   *  lying about the phoneme case, telling the user to "check the lyric or the note/track language"
   *  about a lyric that is fine (the S90 debt; S99 fixed the render's wording, this fixes the
   *  editor's).
   *
   *  ⚠ UNLIKE its two siblings this map is a SUBSET, not a disjoint channel: every id in it is ALSO
   *  in `vocalOov`. That is deliberate — it leaves the red marking, the segment badge and the
   *  blocking counts reading exactly one map, so this change cannot make a note stop being red.
   *  ⇒ NEVER union it into a red/blocking set: you would double-count. Its only consumer is the
   *  wording, plus `vocalOov.length - vocalUnknownPhone.length` = "are there ALSO plain OOV notes". */
  vocalUnknownPhone: Record<string, string[]>;
  /** S85b: notes whose span rounds to ZERO 50fps frames (S84 D 刀 dropped notes) — same red
   *  marking as OOV (both = "will not sound") but a SEPARATE map so the track-header text can
   *  tell the truth: these are too-short notes, not unrecognized lyrics(用户实机反馈:合并
   *  通道让「音符过短」穿了「歌词 OOV」的文案)。 */
  vocalDropped: Record<string, string[]>;
  /** S87 #3: notes that rounded to zero frames and were RESCUED by borrowing a frame from a neighbour.
   *  A NON-BLOCKING condition — the note sounds, it was merely nudged — so it must NOT wear the same red
   *  as vocalOov / vocalDropped, which both mean "this note will not sound at all" (§user: 现在红色一律
   *  是阻塞性警告). Third map for the same reason S85b split dropped out of oov: a merged channel forces
   *  the wording to lie about at least one of its members. */
  vocalShort: Record<string, string[]>;
  /** S113 (§C14): the NON-ERROR hint channel — notes that resolved perfectly well but whose ALIAS
   *  produced a shape its convention cannot make (today: two or more nuclei, i.e. an English word
   *  typed into an alias track — `love` under X-SAMPA sings [l oʊ v ɛ]). The note SOUNDS, and it
   *  sounds exactly as it did before this channel existed; nothing here is blocking.
   *
   *  ⚠ Fourth map for the reason the first three were split (S85b `vocalDropped`, S87 `vocalShort`,
   *  S109 `vocalUnknownPhone`): **a merged channel forces the wording to lie about at least one of
   *  its members.** Never union it into a red/blocking set — red means "this note will not sound"
   *  (§user), and these notes do.
   *
   *  ⛔ ADMISSION RULE for anything that ever joins this channel (the user's instruction,
   *  2026-08-06): it must be something the user can make go away by editing the score. A condition
   *  they cannot clear — an upstream dictionary row that is wrong, a limit of our own mapping —
   *  must NOT be surfaced here; it belongs in the retraining bucket (queue §C22/§G10). A warning
   *  nobody can silence is not observability, it is noise. The Rust side states this per variant
   *  (`AliasHint::user_can_clear`) and a gate enforces it. */
  vocalAliasHint: Record<string, string[]>;
  /** S60 GAME MIDI extraction in flight — key = `${segmentId}:${group}` (lane group), value =
   *  the job context. Drives the lane-row "extracting" indicator (Arrangement per-frame overlay),
   *  the menu-item double-trigger guard, and the undo-cancels-extraction interceptor. Runtime-only. */
  midiExtracting: Record<string, { trackId: string; segId: string; group: string; jobIds: string[] }>;
  toasts: ToastState[];
  /** Transient corner banner (undo/redo info, save/load confirmation, …). `seq` bumps each time so a
   *  rapid retrigger updates the same single banner in place (no stacking, no viewport jump). */
  banner: { message: string; kind: BannerKind; seq: number } | null;
  /** A pending styled confirm dialog (replaces the native `ask` popup). null = nothing shown. */
  confirm: ConfirmRequest | null;
  /** S64 update flow: a new version found by update_check, shown by UpdateDialog. null = closed.
   *  Opened from the startup auto-check AND Settings' manual check — hence store-level, not local. */
  updateDialog: { version: string; currentVersion: string; notes: string | null } | null;
  /** True while UpdateDialog is downloading/installing — the quit flows consult it (a busy update
   *  must not be silently abandoned by tray-quit / window-X; audit S64). */
  updateBusy: boolean;
  /** S66 pre-run model check: unconverted/missing models found by the workflow/vocal preflight,
   *  shown by MissingModelsDialog with per-item one-click actions. null = closed. */
  missingModels: MissingModelItem[] | null;
  /** Track ID for the AMT (Audio-to-MIDI) conversion dialog. null = closed. */
  amtConversionTrackId: string | null;
  /** Segment ID for the AMT (Audio-to-MIDI) conversion dialog. null = not specified. */
  amtConversionSegmentId: string | null;
  /** Forced source audio path + track/segment name captured AT right-click time (as a hard
   *  guarantee the "音乐转MIDI" dialog always shows the source + playable waveform). */
  amtSourceAudioPath: string | null;
  amtSourceTrackName: string | null;

  toggleTrainingPage: () => void;
  toggleModelManager: () => void;
  /** Muno 音源管理器(阶段1):打开/关闭音源管理弹窗。 */
  toggleSoundfontManager: () => void;
  /** Song Studio (歌曲制作)主面板:打开/关闭。 */
  toggleSongStudio: () => void;
  toggleLogViewer: () => void;
  setActiveTrack: (id: string | null) => void;
  selectSegment: (trackId: string, segmentId: string) => void;
  /** Replace the whole selection with `items` (first = primary/anchor) — S61 paste selects its output. */
  selectSegments: (items: SegmentSelection[]) => void;
  toggleSegment: (trackId: string, segmentId: string) => void;
  /** Select a sub-lane group (all lanes of `outputNodeId`) in a segment, with the clicked piece index. */
  selectLane: (trackId: string, segmentId: string, outputNodeId: string, clipIndex: number) => void;
  clearSelection: () => void;
  openWorkflow: (segmentId: string) => void;
  closeWorkflow: () => void;
  /** ② Open the vocal (piano-roll) editor on a notes segment; closes any open workflow editor (§9.6). */
  openVocalEditor: (segmentId: string) => void;
  closeVocalEditor: () => void;
  openAmtResult: (result: AmtResult) => void;
  closeAmtResult: () => void;
  requestLaneDetach: (segmentId: string, outputNodeId: string) => void;
  clearLaneDetach: () => void;
  setActivePane: (pane: "timeline" | "workflow" | "vocal" | "amt") => void;
  setWorkflowPanelHeight: (h: number) => void;
  setVocalPanelHeight: (h: number) => void;
  setAmtPanelHeight: (h: number) => void;
  setZoom: (zoom: number) => void;
  setVZoom: (vZoom: number) => void;
  /** 轨头列宽(px) — DawView 里轨头列与排列画布之间的竖向分隔条可拖拽调整,持久化。 */
  trackHeaderWidth: number;
  setTrackHeaderWidth: (w: number) => void;
  setScroll: (x: number, y: number) => void;
  setCanvasWidth: (w: number) => void;
  setCanvasHeight: (h: number) => void;
  setGhostInsert: (g: { index: number; count: number } | null) => void;
  toggleSnapSegments: () => void;
  toggleSnapPlayhead: () => void;
  toggleSnapNotes: () => void;
  /** ③ B8: 吸附步长（栅格量化步进，单位 ticks）。0 = 关闭栅格吸附（仅保留片段边缘/播放头磁吸）。 */
  snapGridStep: number;
  setSnapGridStep: (v: number) => void;
  setVocalRenderActive: (v: boolean) => void;
  setRenderingVocalTrackId: (id: string | null) => void;
  /** 内置皮肤 id（对应 theme.css 的 [data-skin="…"]）。"junzi" = 君子·紫（品牌默认）。持久化于 utai.skin。 */
  skin: string;
  setSkin: (skin: string) => void;
  /** ② S58: publish one segment's OOV verdict (null = clear the entry). No-op-guarded (identical
   *  verdicts don't re-render subscribers). Written ONLY by the oovWatch validation watcher. */
  setVocalOov: (segmentId: string, noteIds: string[] | null) => void;
  /** S109 (§C15): publish the phoneme-typo subset of one segment's OOV verdict. Same writer
   *  (oovWatch), same merge/no-op semantics; see the field's doc for why it is a SUBSET. */
  setVocalUnknownPhone: (segmentId: string, noteIds: string[] | null) => void;
  /** S85b: publish one segment's dropped-note verdict(语义同 setVocalOov,写者同为 oovWatch)。 */
  setVocalDropped: (segmentId: string, noteIds: string[] | null) => void;
  /** S87: publish one segment's frame-borrow verdict(语义同上,写者同为 oovWatch)。 */
  setVocalShort: (segmentId: string, noteIds: string[] | null) => void;
  /** S113 (§C14): publish one segment's alias-hint notes(语义同上,写者同为 oovWatch)。 */
  setVocalAliasHint: (segmentId: string, noteIds: string[] | null) => void;
  /** S60: publish/clear one lane group's MIDI-extraction job (null = done/cancelled). */
  setMidiExtracting: (key: string, v: { trackId: string; segId: string; group: string; jobIds: string[] } | null) => void;
  openUpdateDialog: (info: { version: string; currentVersion: string; notes: string | null }) => void;
  closeUpdateDialog: () => void;
  openMissingModels: (items: MissingModelItem[]) => void;
  closeMissingModels: () => void;
  openAmtConversion: (trackId: string, segmentId?: string) => void;
  closeAmtConversion: () => void;
  /** Cache the resolved source (name + audio path) so the dialog shows it reliably. */
  setAmtSource: (audioPath: string | null, trackName: string | null) => void;
  setUpdateBusy: (v: boolean) => void;
  showToast: (message: string, type?: "error" | "info" | "success" | "warning") => void;
  dismissToast: (id: number) => void;
  showBanner: (message: string, kind: BannerKind) => void;
  /** Show a styled confirm dialog; resolves with the chosen button id, or "" if dismissed (Esc/backdrop).
   *  With `input` set, the primary button/Enter resolves the trimmed input VALUE instead (see ConfirmInput). */
  showConfirm: (opts: { title: string; body: string; buttons: ConfirmButton[]; input?: ConfirmInput; check?: ConfirmCheck; scrollable?: boolean }) => Promise<string>;
}

export type BannerKind = "undo" | "redo" | "save" | "load" | "info";

/** Monotonic toast id (Date.now() collides within one millisecond — see showToast). */
let toastSeq = 0;

/** S85b: shared merge/no-op semantics for the per-segment verdict maps (vocalOov / vocalDropped).
 *  Returns the next map, or null when the verdict is identical (identical verdicts must not
 *  re-render every canvas subscriber on each revalidation — the S58 no-op guard, single source). */
function verdictMapUpdate(
  map: Record<string, string[]>,
  segmentId: string,
  noteIds: string[] | null,
): Record<string, string[]> | null {
  const cur = map[segmentId];
  if (noteIds === null && cur === undefined) return null;
  if (noteIds !== null && cur !== undefined && cur.length === noteIds.length && cur.every((v, i) => v === noteIds[i])) return null;
  const next = { ...map };
  if (noteIds === null) delete next[segmentId];
  else next[segmentId] = noteIds;
  return next;
}

export const useAppStore = create<AppState>((set, get) => ({
  theme: (typeof localStorage !== "undefined" && localStorage.getItem("utai.theme") === "light") ? "light" : "dark",
  setTheme: (t) => {
    if (typeof document !== "undefined") document.documentElement.setAttribute("data-theme", t);
    if (typeof localStorage !== "undefined") localStorage.setItem("utai.theme", t);
    set({ theme: t });
  },
  trainingPageOpen: false,
  modelManagerOpen: false,
  soundfontManagerOpen: false,
  songStudioOpen: false,
  logViewerOpen: false,
  settingsOpen: false,
  activeTrackId: null,
  selectedSegment: null,
  selectedSegments: [],
  selectedLane: null,
  workflowSegmentId: null,
  vocalSegmentId: null,
  amtResult: null,
  pendingLaneDetach: null,
  workflowAutoRun: null,
  activePane: "timeline",
  inspectorTrackId: null,
  workflowPanelHeight: loadSetting("utai.workflowPanelHeight", 460),
  vocalPanelHeight: loadSetting("utai.vocalPanelHeight", 460),
  amtPanelHeight: loadSetting("utai.amtPanelHeight", 460),
  zoom: 1.0,
  vZoom: 1.0,
  trackHeaderWidth: loadSetting("utai.trackHeaderWidth", 220),
  scrollX: 0,
  scrollY: 0,
  canvasWidth: 800,
  canvasHeight: 600,
  ghostInsert: null,
  snapSegments: loadSetting("utai.snapSegments", true),
  snapPlayhead: loadSetting("utai.snapPlayhead", true),
  snapNotes: loadSetting("utai.snapNotes", true),
  snapGridStep: loadSetting("utai.snapGridStep", 120),
  vocalRenderActive: false,
  renderingVocalTrackId: null,
  skin: loadSetting("utai.skin", "junzi"),
  vocalOov: {},
  vocalUnknownPhone: {},
  vocalDropped: {},
  vocalShort: {},
  vocalAliasHint: {},
  midiExtracting: {},
  toasts: [],
  banner: null,
  confirm: null,
  updateDialog: null,
  missingModels: null,
  amtConversionTrackId: null,
  amtConversionSegmentId: null,
  amtSourceAudioPath: null,
  amtSourceTrackName: null,
  updateBusy: false,

  toggleTrainingPage: () =>
    set((s) => ({ trainingPageOpen: !s.trainingPageOpen })),
  toggleModelManager: () =>
    set((s) => ({ modelManagerOpen: !s.modelManagerOpen })),
  toggleSoundfontManager: () =>
    set((s) => ({ soundfontManagerOpen: !s.soundfontManagerOpen })),
  toggleSongStudio: () =>
    set((s) => ({ songStudioOpen: !s.songStudioOpen })),
  pendingSongTask: null,
  setPendingSongTask: (p) => set({ pendingSongTask: p }),
  clearPendingSongTask: () => set({ pendingSongTask: null }),
  toggleLogViewer: () =>
    set((s) => ({ logViewerOpen: !s.logViewerOpen })),
  toggleSettings: () =>
    set((s) => ({ settingsOpen: !s.settingsOpen })),
  setActiveTrack: (id) => set({ activeTrackId: id }),
  selectSegment: (trackId, segmentId) =>
    set({
      selectedSegment: { trackId, segmentId },
      selectedSegments: [{ trackId, segmentId }],
      selectedLane: null,
      activeTrackId: trackId,
    }),
  selectSegments: (items) =>
    set({
      selectedSegment: items[0] ?? null,
      selectedSegments: items,
      selectedLane: null,
      activeTrackId: items[0]?.trackId ?? null,
    }),
  toggleSegment: (trackId, segmentId) =>
    set((s) => {
      const exists = s.selectedSegments.some((x) => x.trackId === trackId && x.segmentId === segmentId);
      const next = exists
        ? s.selectedSegments.filter((x) => !(x.trackId === trackId && x.segmentId === segmentId))
        : [...s.selectedSegments, { trackId, segmentId }];
      return {
        selectedSegments: next,
        selectedSegment: exists ? (next[next.length - 1] ?? null) : { trackId, segmentId },
        selectedLane: null,
        activeTrackId: trackId,
      };
    }),
  // Selecting a lane anchors the parent as `selectedSegment` (so Ctrl+K/Delete + the Split button have a
  // coherent fallback target) but does NOT add it to `selectedSegments` — otherwise the parent segment
  // ALSO lit up gold, competing with the sub-lane's own gold highlight and reading as confusing. Only the
  // sub-lane group is cued (via selectedLane); the non-null selectedLane routes the edit to the lane.
  selectLane: (trackId, segmentId, outputNodeId, clipIndex) =>
    set({
      selectedLane: { trackId, segmentId, outputNodeId, clipIndex },
      selectedSegment: { trackId, segmentId },
      selectedSegments: [],
      activeTrackId: trackId,
    }),
  clearSelection: () => set({ selectedSegment: null, selectedSegments: [], selectedLane: null }),
  // Opening either bottom-dock editor closes the OTHER (the dock shows one at a time) so activePane, the
  // divider cue, and undo routing can never point at a hidden editor (§9.6 exclusivity).
  openWorkflow: (segmentId) => set({ workflowSegmentId: segmentId, vocalSegmentId: null, activePane: "workflow" }),
  closeWorkflow: () => set({ workflowSegmentId: null, activePane: "timeline" }),
  openVocalEditor: (segmentId) => set({ vocalSegmentId: segmentId, workflowSegmentId: null, activePane: "vocal" }),
  closeVocalEditor: () => set({ vocalSegmentId: null, activePane: "timeline" }),
  openAmtResult: (result) => set({ amtResult: result, workflowSegmentId: null, vocalSegmentId: null, activePane: "amt" }),
  closeAmtResult: () => set({ amtResult: null, activePane: "timeline" }),
  requestLaneDetach: (segmentId, outputNodeId) => set({ pendingLaneDetach: { segmentId, outputNodeId } }),
  clearLaneDetach: () => set({ pendingLaneDetach: null }),
  clearWorkflowAutoRun: () => set({ workflowAutoRun: null }),
  setActivePane: (pane) => set((s) => (s.activePane === pane ? s : { activePane: pane })),
  openInspector: (trackId) => set({ inspectorTrackId: trackId }),
  closeInspector: () => set({ inspectorTrackId: null }),
  toggleInspector: (trackId) =>
    set((s) => ({ inspectorTrackId: s.inspectorTrackId === trackId ? null : trackId })),
  setWorkflowPanelHeight: (h) => set({ workflowPanelHeight: h }),
  setVocalPanelHeight: (h) => set({ vocalPanelHeight: h }),
  setAmtPanelHeight: (h) => set({ amtPanelHeight: h }),
  setZoom: (zoom) => set({ zoom: Math.max(0.1, Math.min(10, zoom)) }),
  setVZoom: (vZoom) => set({ vZoom: Math.max(0.6, Math.min(3, vZoom)) }),
  // 拖拽结束时才 saveSetting(拖动过程中每帧 set,不写 localStorage —— 与面板高度 resize 的纪律一致)
  setTrackHeaderWidth: (w) => set({ trackHeaderWidth: Math.max(140, Math.min(420, Math.round(w))) }),
  setScroll: (x, y) => set({ scrollX: x, scrollY: y }),
  setCanvasWidth: (w) => set({ canvasWidth: w }),
  setCanvasHeight: (h) => set({ canvasHeight: h }),
  setGhostInsert: (g) => set({ ghostInsert: g }),
  toggleSnapSegments: () =>
    set((s) => {
      const v = !s.snapSegments;
      saveSetting("utai.snapSegments", v);
      return { snapSegments: v };
    }),
  toggleSnapPlayhead: () =>
    set((s) => {
      const v = !s.snapPlayhead;
      saveSetting("utai.snapPlayhead", v);
      return { snapPlayhead: v };
    }),
  toggleSnapNotes: () =>
    set((s) => {
      const v = !s.snapNotes;
      saveSetting("utai.snapNotes", v);
      return { snapNotes: v };
    }),
  setSnapGridStep: (v) => {
    saveSetting("utai.snapGridStep", v);
    set({ snapGridStep: v });
  },
  setVocalRenderActive: (v) => set({ vocalRenderActive: v }),
  setRenderingVocalTrackId: (id) => set({ renderingVocalTrackId: id }),
  setSkin: (skin) => {
    saveSetting("utai.skin", skin);
    // 同步写 DOM:画布类组件(排列/钢琴等)在 useEffect 里同步重绘并读取 --bg-base 等 CSS 变量,
    // 而 React 的 effect 是子组件先于父组件执行 —— 若仅靠 App.tsx 的 useEffect 改 data-skin,
    // 画布烘焙时读到的仍是旧皮肤变量,且 staticKey 已写入新 skin id,之后永不再重烘焙
    // (画布背景滞后一个皮肤的根因)。在 store 动作里同步更新属性,保证任何 effect 执行前
    // CSS 变量已是新值。App.tsx 的 effect 保留,负责首挂载时应用持久化的皮肤。
    document.documentElement.dataset.skin = skin;
    set({ skin });
  },
  setVocalOov: (segmentId, noteIds) =>
    set((s) => {
      const next = verdictMapUpdate(s.vocalOov, segmentId, noteIds);
      return next ? { vocalOov: next } : {};
    }),
  setVocalUnknownPhone: (segmentId, noteIds) =>
    set((s) => {
      const next = verdictMapUpdate(s.vocalUnknownPhone, segmentId, noteIds);
      return next ? { vocalUnknownPhone: next } : {};
    }),
  setVocalDropped: (segmentId, noteIds) =>
    set((s) => {
      const next = verdictMapUpdate(s.vocalDropped, segmentId, noteIds);
      return next ? { vocalDropped: next } : {};
    }),
  setVocalShort: (segmentId, noteIds) =>
    set((s) => {
      const next = verdictMapUpdate(s.vocalShort, segmentId, noteIds);
      return next ? { vocalShort: next } : {};
    }),
  setVocalAliasHint: (segmentId, noteIds) =>
    set((s) => {
      const next = verdictMapUpdate(s.vocalAliasHint, segmentId, noteIds);
      return next ? { vocalAliasHint: next } : {};
    }),
  setMidiExtracting: (key, v) =>
    set((s) => {
      const next = { ...s.midiExtracting };
      if (v === null) {
        if (!(key in next)) return {};
        delete next[key];
      } else {
        next[key] = v;
      }
      return { midiExtracting: next };
    }),
  openUpdateDialog: (info) => set({ updateDialog: info }),
  closeUpdateDialog: () => set({ updateDialog: null }),
  openMissingModels: (items) => set({ missingModels: items }),
  closeMissingModels: () => set({ missingModels: null }),
  openAmtConversion: (trackId, segmentId) => set({ amtConversionTrackId: trackId, amtConversionSegmentId: segmentId ?? null }),
  closeAmtConversion: () => set({ amtConversionTrackId: null, amtConversionSegmentId: null, amtSourceAudioPath: null, amtSourceTrackName: null }),
  setAmtSource: (audioPath, trackName) => set({ amtSourceAudioPath: audioPath, amtSourceTrackName: trackName }),
  setUpdateBusy: (v) => set({ updateBusy: v }),
  showToast: (message, type = "error") => {
    // monotonic id — Date.now() collides when two toasts fire in the same millisecond (e.g. the
    // auto-render batch reporting several failures back-to-back), making the first timeout dismiss both.
    const id = ++toastSeq;
    set((s) => ({ toasts: [...s.toasts, { message, type, id }] }));
    // 按类型差异化时长: error 8s (看清错误), warning 6s, info/success 4s
    const duration = type === "error" ? 8000 : type === "warning" ? 6000 : 4000;
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, duration);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  showBanner: (message, kind) =>
    set((s) => ({ banner: { message, kind, seq: (s.banner?.seq ?? 0) + 1 } })),
  showConfirm: (opts) =>
    new Promise<string>((resolve) => {
      // Capture the prior seq BEFORE settling the previous dialog: its resolve() nulls `confirm`, so
      // reading `s.confirm?.seq` afterwards always yielded 1 — a stacked dialog then reused the prior
      // dialog's keyed input state (ConfirmDialog remounts on `key={seq}`).
      const prevSeq = get().confirm?.seq ?? 0;
      // Settle any already-open dialog as dismissed first, so its awaiter never hangs (e.g. the native
      // window-close button firing onCloseRequested while a New/Open discard dialog is still up).
      get().confirm?.resolve("");
      set({
        confirm: {
          title: opts.title,
          body: opts.body,
          buttons: opts.buttons,
          input: opts.input,
          check: opts.check,
          scrollable: opts.scrollable,
          seq: prevSeq + 1,
          resolve: (id: string) => {
            set({ confirm: null });
            resolve(id);
          },
        },
      });
    }),
}));
