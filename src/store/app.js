import { create } from "zustand";
import { loadSetting, saveSetting } from "../lib/settings";
/** Monotonic toast id (Date.now() collides within one millisecond — see showToast). */
let toastSeq = 0;
/** S85b: shared merge/no-op semantics for the per-segment verdict maps (vocalOov / vocalDropped).
 *  Returns the next map, or null when the verdict is identical (identical verdicts must not
 *  re-render every canvas subscriber on each revalidation — the S58 no-op guard, single source). */
function verdictMapUpdate(map, segmentId, noteIds) {
    const cur = map[segmentId];
    if (noteIds === null && cur === undefined)
        return null;
    if (noteIds !== null && cur !== undefined && cur.length === noteIds.length && cur.every((v, i) => v === noteIds[i]))
        return null;
    const next = { ...map };
    if (noteIds === null)
        delete next[segmentId];
    else
        next[segmentId] = noteIds;
    return next;
}
export const useAppStore = create((set, get) => ({
    theme: (typeof localStorage !== "undefined" && localStorage.getItem("utai.theme") === "light") ? "light" : "dark",
    setTheme: (t) => {
        if (typeof document !== "undefined")
            document.documentElement.setAttribute("data-theme", t);
        if (typeof localStorage !== "undefined")
            localStorage.setItem("utai.theme", t);
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
    toggleTrainingPage: () => set((s) => ({ trainingPageOpen: !s.trainingPageOpen })),
    toggleModelManager: () => set((s) => ({ modelManagerOpen: !s.modelManagerOpen })),
    toggleSoundfontManager: () => set((s) => ({ soundfontManagerOpen: !s.soundfontManagerOpen })),
    toggleSongStudio: () => set((s) => ({ songStudioOpen: !s.songStudioOpen })),
    pendingSongTask: null,
    setPendingSongTask: (p) => set({ pendingSongTask: p }),
    clearPendingSongTask: () => set({ pendingSongTask: null }),
    toggleLogViewer: () => set((s) => ({ logViewerOpen: !s.logViewerOpen })),
    toggleSettings: () => set((s) => ({ settingsOpen: !s.settingsOpen })),
    setActiveTrack: (id) => set({ activeTrackId: id }),
    selectSegment: (trackId, segmentId) => set({
        selectedSegment: { trackId, segmentId },
        selectedSegments: [{ trackId, segmentId }],
        selectedLane: null,
        activeTrackId: trackId,
    }),
    selectSegments: (items) => set({
        selectedSegment: items[0] ?? null,
        selectedSegments: items,
        selectedLane: null,
        activeTrackId: items[0]?.trackId ?? null,
    }),
    toggleSegment: (trackId, segmentId) => set((s) => {
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
    selectLane: (trackId, segmentId, outputNodeId, clipIndex) => set({
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
    toggleInspector: (trackId) => set((s) => ({ inspectorTrackId: s.inspectorTrackId === trackId ? null : trackId })),
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
    toggleSnapSegments: () => set((s) => {
        const v = !s.snapSegments;
        saveSetting("utai.snapSegments", v);
        return { snapSegments: v };
    }),
    toggleSnapPlayhead: () => set((s) => {
        const v = !s.snapPlayhead;
        saveSetting("utai.snapPlayhead", v);
        return { snapPlayhead: v };
    }),
    toggleSnapNotes: () => set((s) => {
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
    setVocalOov: (segmentId, noteIds) => set((s) => {
        const next = verdictMapUpdate(s.vocalOov, segmentId, noteIds);
        return next ? { vocalOov: next } : {};
    }),
    setVocalUnknownPhone: (segmentId, noteIds) => set((s) => {
        const next = verdictMapUpdate(s.vocalUnknownPhone, segmentId, noteIds);
        return next ? { vocalUnknownPhone: next } : {};
    }),
    setVocalDropped: (segmentId, noteIds) => set((s) => {
        const next = verdictMapUpdate(s.vocalDropped, segmentId, noteIds);
        return next ? { vocalDropped: next } : {};
    }),
    setVocalShort: (segmentId, noteIds) => set((s) => {
        const next = verdictMapUpdate(s.vocalShort, segmentId, noteIds);
        return next ? { vocalShort: next } : {};
    }),
    setVocalAliasHint: (segmentId, noteIds) => set((s) => {
        const next = verdictMapUpdate(s.vocalAliasHint, segmentId, noteIds);
        return next ? { vocalAliasHint: next } : {};
    }),
    setMidiExtracting: (key, v) => set((s) => {
        const next = { ...s.midiExtracting };
        if (v === null) {
            if (!(key in next))
                return {};
            delete next[key];
        }
        else {
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
    showBanner: (message, kind) => set((s) => ({ banner: { message, kind, seq: (s.banner?.seq ?? 0) + 1 } })),
    showConfirm: (opts) => new Promise((resolve) => {
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
                resolve: (id) => {
                    set({ confirm: null });
                    resolve(id);
                },
            },
        });
    }),
}));
