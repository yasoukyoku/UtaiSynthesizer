import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { flushAutosaveNow } from "../../lib/project/autosave";
import { ReactFlow, Background, Controls, MiniMap, addEdge, useNodesState, useEdgesState, BackgroundVariant, } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { AudioInputNode } from "./nodes/AudioInputNode";
import { AudioOutputNode } from "./nodes/AudioOutputNode";
import { RvcNode } from "./nodes/RvcNode";
import { SoVitsNode } from "./nodes/SoVitsNode";
import { SeparationNode } from "./nodes/SeparationNode";
import { TransposeNode } from "./nodes/TransposeNode";
import { AmtMidiNode } from "./nodes/AmtMidiNode";
import { SpeedShiftNode } from "./nodes/SpeedShiftNode";
import { ChordDetectNode } from "./nodes/ChordDetectNode";
import { AutoArrangeNode } from "./nodes/AutoArrangeNode";
import { DeepOriginalNode } from "./nodes/DeepOriginalNode";
import { MidiFileInNode } from "./nodes/MidiFileInNode";
import { ChordBlockInNode } from "./nodes/ChordBlockInNode";
import { SoundfontRenderNode } from "./nodes/SoundfontRenderNode";
import { MelodySimilarityNode } from "./nodes/MelodySimilarityNode";
import { SplitNode } from "./nodes/SplitNode";
import { MergeNode } from "./nodes/MergeNode";
import { ComplianceCheckNode } from "./nodes/ComplianceCheckNode";
import { LufsNormalizeNode } from "./nodes/LufsNormalizeNode";
import { DitherNode } from "./nodes/DitherNode";
import { BusEqNode } from "./nodes/BusEqNode";
import { StereoWidthNode } from "./nodes/StereoWidthNode";
import { SaturateNode } from "./nodes/SaturateNode";
import { PhaseRotateNode } from "./nodes/PhaseRotateNode";
import { DcRemoveNode } from "./nodes/DcRemoveNode";
// Phase 5 分析可视化节点族
import { SpectrogramNode } from "./nodes/SpectrogramNode";
import { F0CurveNode } from "./nodes/F0CurveNode";
import { TimbreMetricsNode } from "./nodes/TimbreMetricsNode";
import { HarmonicityCheckNode } from "./nodes/HarmonicityCheckNode";
import { SpectralCompareNode } from "./nodes/SpectralCompareNode";
import { DtwAlignNode } from "./nodes/DtwAlignNode";
import { AbCompareNode } from "./nodes/AbCompareNode";
import { LufsAnalyzeNode } from "./nodes/LufsAnalyzeNode";
import { HarmonizerNode } from "./nodes/HarmonizerNode";
import { MelodyGenNode } from "./nodes/MelodyGenNode";
import { SongLyricsNode } from "./nodes/SongLyricsNode";
import { SongPromptNode } from "./nodes/SongPromptNode";
import { SongGenNode } from "./nodes/SongGenNode";
import { SongCoverNode } from "./nodes/SongCoverNode";
import { SongRepaintNode } from "./nodes/SongRepaintNode";
import { SongCompleteNode } from "./nodes/SongCompleteNode";
import { SongExtractNode } from "./nodes/SongExtractNode";
import { SongLegoNode } from "./nodes/SongLegoNode";
import { SongStemsNode } from "./nodes/SongStemsNode";
import { SongSheetNode } from "./nodes/SongSheetNode";
import { MidiHumanizeNode } from "./nodes/MidiHumanizeNode";
import { VelocityCurveNode } from "./nodes/VelocityCurveNode";
import { SwingQuantizeNode } from "./nodes/SwingQuantizeNode";
import { MelodyReharmNode } from "./nodes/MelodyReharmNode";
import { RhythmRestructureNode } from "./nodes/RhythmRestructureNode";
import { ContourMorphNode } from "./nodes/ContourMorphNode";
import { MotifDevelopNode } from "./nodes/MotifDevelopNode";
import { ReharmonizeNode } from "./nodes/ReharmonizeNode";
import { RhythmVariationNode } from "./nodes/RhythmVariationNode";
import { StructureEditNode } from "./nodes/StructureEditNode";
import { BreathPlannerNode } from "./nodes/BreathPlannerNode";
import { NodePalette, getPaletteDefs } from "./NodePalette";
import { ExampleWorkflowLibrary } from "./ExampleWorkflowLibrary";
import { TutorialOverlay, shouldShowTutorial } from "./TutorialOverlay";
import { NodeAnnotationEditor } from "./NodeAnnotation";
import { SmartConnectionHelper } from "./SmartConnectionHelper";
import { EnhancedErrorDisplay } from "./EnhancedErrorDisplay";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useWorkflowStore } from "../../store/workflow";
import { useMsstModelStore } from "../../store/msst-models";
import { useAppStore } from "../../store/app";
import { setUndoScope, useHistoryStore } from "../../store/history";
import { DEFAULT_OUTPUT_GROUP } from "../../lib/constants";
import i18n from "../../i18n";
import { executeWorkflow, executeSingleNode, preflightRun, collectCachedPaths, loadCachedOutput, outputLanes, rehydrateRenderState, planDetachGroup } from "../../lib/workflow/engine";
import { nodeHistoryFor } from "../../lib/workflow/nodeHistory";
import { listNodeParamPresets, saveNodeParamPreset, deleteNodeParamPreset } from "../../lib/workflow/nodeParamPresets";
import { computeAutoLayout } from "../../lib/workflow/autoLayout";
import { alignNodes, distributeNodes } from "../../lib/workflow/alignNodes";
import { logToBackend } from "../../lib/log";
import { clearBufferCache, loadAudioBuffer } from "../../lib/audio/playback";
import { ContextMenu } from "../common/ContextMenu";
import { useTranslation } from "react-i18next";
import { save, open } from "@tauri-apps/plugin-dialog";
import { rfTypeToWfType, wfTypeToRfType } from "../../lib/workflow/rfTypes";
import { NODE_DAMAGE, cumulativeSnrDb } from "../../lib/workflow/damage";
import { canConnect } from "../../lib/workflow/ports";
import "./WorkflowEditor.css";
// Phase 4-4: 母带三档预设 (规划 8.6 表 → 一键预连线链). complianceCheck 恒在链尾 —
// 体检必须看最终成品; 链头留空由用户接音频源, 链尾 out-0 接 Output.
const MASTER_CHAIN_LABELS = {
    dcRemove: "workflow.nodeDcRemove",
    busEq: "workflow.nodeBusEq",
    stereoWidth: "workflow.nodeStereoWidth",
    saturate: "workflow.nodeSaturate",
    phaseRotate: "workflow.nodePhaseRotate",
    lufsNormalize: "workflow.nodeLufsNormalize",
    dither: "workflow.nodeDither",
    complianceCheck: "workflow.nodeComplianceCheck",
};
const MASTER_CHAINS = {
    // 🟢 透明: ①dc ⑥lufs ⑦dither ⑧compliance — 混音已好, 只规整响度 (>130dB SNR).
    transparent: [
        { type: "dcRemove" },
        { type: "lufsNormalize", params: { targetLufs: -16 } },
        { type: "dither", params: { ditherType: 2 } },
        { type: "complianceCheck" },
    ],
    // 🟡 标准 (默认): + ②busEq ③stereoWidth — 多数情况 (~112dB SNR).
    standard: [
        { type: "dcRemove" },
        { type: "busEq" },
        { type: "stereoWidth" },
        { type: "lufsNormalize", params: { targetLufs: -16 } },
        { type: "dither", params: { ditherType: 2 } },
        { type: "complianceCheck" },
    ],
    // 🟠 响度: 全开 (含 ④saturate + 相位旋转给限幅余量) — 竞争性响度的流行/电子 (~68dB SNR).
    loud: [
        { type: "dcRemove" },
        { type: "busEq" },
        { type: "stereoWidth" },
        { type: "saturate", params: { drive: 0.05 } },
        { type: "phaseRotate", params: { strength: 0.25 } },
        { type: "lufsNormalize", params: { targetLufs: -16 } },
        { type: "dither", params: { ditherType: 2 } },
        { type: "complianceCheck" },
    ],
};
const nodeTypes = {
    audioInput: AudioInputNode,
    audioOutput: AudioOutputNode,
    rvc: RvcNode,
    sovits: SoVitsNode,
    separation: SeparationNode,
    transpose: TransposeNode,
    amtMidi: AmtMidiNode,
    // 新增 4 个纯前端/半纯前端节点:
    speedShift: SpeedShiftNode,
    chordDetect: ChordDetectNode,
    autoArrange: AutoArrangeNode,
    deepOriginal: DeepOriginalNode,
    midiFileIn: MidiFileInNode,
    chordBlockIn: ChordBlockInNode,
    soundfontRender: SoundfontRenderNode,
    melodySimilarity: MelodySimilarityNode,
    split: SplitNode,
    // Phase 1 母带合规节点族
    merge: MergeNode,
    complianceCheck: ComplianceCheckNode,
    lufsNormalize: LufsNormalizeNode,
    dither: DitherNode,
    // Phase 4 母带补全节点族
    busEq: BusEqNode,
    stereoWidth: StereoWidthNode,
    saturate: SaturateNode,
    phaseRotate: PhaseRotateNode,
    dcRemove: DcRemoveNode,
    // Phase 5 分析可视化节点族
    spectrogram: SpectrogramNode,
    f0Curve: F0CurveNode,
    timbreMetrics: TimbreMetricsNode,
    harmonicityCheck: HarmonicityCheckNode,
    spectralCompare: SpectralCompareNode,
    dtwAlign: DtwAlignNode,
    abCompare: AbCompareNode,
    lufsAnalyze: LufsAnalyzeNode,
    harmonizer: HarmonizerNode,
    melodyGen: MelodyGenNode,
    // P2-14 歌曲节点族
    songLyrics: SongLyricsNode,
    songPrompt: SongPromptNode,
    songGen: SongGenNode,
    songCover: SongCoverNode,
    songRepaint: SongRepaintNode,
    songComplete: SongCompleteNode,
    songExtract: SongExtractNode,
    songLego: SongLegoNode,
    songStems: SongStemsNode,
    songSheet: SongSheetNode,
    // P2 符号域原创化节点族
    midiHumanize: MidiHumanizeNode,
    velocityCurve: VelocityCurveNode,
    swingQuantize: SwingQuantizeNode,
    melodyReharm: MelodyReharmNode,
    rhythmRestructure: RhythmRestructureNode,
    contourMorph: ContourMorphNode,
    motifDevelop: MotifDevelopNode,
    reharmonize: ReharmonizeNode,
    rhythmVariation: RhythmVariationNode,
    structureEdit: StructureEditNode,
    breathPlanner: BreathPlannerNode,
    // Legacy — kept for loading old workflows ("msst" was the pre-catalog separation type).
    // The dead Effects node types (pitchShift/formantShift/audioEnhance) are migrated to
    // "transpose" at LOAD (parseLoadedBundle), so they never reach ReactFlow.
    msst: SeparationNode,
};
let nodeCounter = 0;
function workflowToReactFlow(wf) {
    const nodes = wf.nodes.map((n, i) => ({
        id: n.id,
        type: wfTypeToRfType[n.nodeType] ?? n.nodeType,
        position: { x: n.position.x, y: n.position.y },
        data: {
            label: n.nodeType,
            params: n.params,
            bypass: n.bypass === true,
            annotation: n.annotation,
        },
        deletable: n.nodeType !== "input", // Output nodes are deletable now (live deposit — deleting one drops its lanes)
        zIndex: i,
    }));
    const edges = wf.connections.map((c, i) => ({
        id: `e-${i}`,
        source: c.fromNode,
        sourceHandle: `out-${c.fromPort}`,
        target: c.toNode,
        targetHandle: `in-${c.toPort}`,
        animated: true,
    }));
    return { nodes, edges };
}
function reactFlowToWorkflow(nodes, edges) {
    const wfNodes = nodes.map((n) => {
        const base = {
            id: n.id,
            nodeType: rfTypeToWfType[n.type ?? ""] ?? n.type,
            position: { x: n.position.x, y: n.position.y },
            params: n.data?.params ?? {},
        };
        if (n.data?.bypass === true) {
            base.bypass = true;
        }
        if (n.data?.annotation) {
            base.annotation = n.data.annotation;
        }
        return base;
    });
    const wfConns = edges.map((e) => ({
        fromNode: e.source,
        fromPort: parseInt(e.sourceHandle?.replace("out-", "") ?? "0", 10),
        toNode: e.target,
        toPort: parseInt(e.targetHandle?.replace("in-", "") ?? "0", 10),
    }));
    return { nodes: wfNodes, connections: wfConns };
}
/** Structural signature of the node graph for undo diffing — EXCLUDES selection/dimensions (which
 *  must not create undo steps), includes node type/bypass/params and edge wiring. */
function sigOfGraph(nodes, edges) {
    const ns = nodes
        // Node POSITION is intentionally EXCLUDED — moving a node around the canvas is not a meaningful edit and
        // must not create an undo step (position is still persisted via the debounced save, just not undoable).
        .map((n) => `${n.id}:${n.type}:${n.data?.bypass === true ? "B" : "b"}:${JSON.stringify(n.data?.params ?? {})}`)
        .sort()
        .join("|");
    const es = edges.map((e) => `${e.source}.${e.sourceHandle}>${e.target}.${e.targetHandle}`).sort().join("|");
    return `${ns}||${es}`;
}
/** Describe (i18n key under "history.") the node-graph op that transforms from→to, for the banner. */
function describeNodeDelta(from, to) {
    if (to.nodes.length > from.nodes.length)
        return "nodeAdd";
    if (to.nodes.length < from.nodes.length)
        return "nodeRemove";
    if (to.edges.length > from.edges.length)
        return "nodeConnect";
    if (to.edges.length < from.edges.length)
        return "nodeDisconnect";
    const fById = new Map(from.nodes.map((n) => [n.id, n]));
    for (const tn of to.nodes) {
        const fn = fById.get(tn.id);
        if (!fn)
            return "nodeEdit";
        if (Math.round(fn.position.x) !== Math.round(tn.position.x) || Math.round(fn.position.y) !== Math.round(tn.position.y))
            return "nodeMove";
        if (fn.data?.bypass !== tn.data?.bypass)
            return "nodeBypass";
        if (JSON.stringify(fn.data?.params ?? {}) !== JSON.stringify(tn.data?.params ?? {}))
            return "nodeParam";
    }
    return "nodeEdit";
}
function announceNode(from, to, kind) {
    const verb = i18n.t(kind === "undo" ? "history.undone" : "history.redone");
    useAppStore.getState().showBanner(`${verb} · ${i18n.t(`history.${describeNodeDelta(from, to)}`)}`, kind);
}
const defaultNodes = [
    {
        id: "input-1",
        type: "audioInput",
        position: { x: 50, y: 200 },
        data: { label: "Audio In" },
        deletable: false,
        zIndex: 0,
    },
    {
        id: "output-1",
        type: "audioOutput",
        position: { x: 600, y: 200 },
        data: { label: "Output", params: { laneLabel: DEFAULT_OUTPUT_GROUP } },
        deletable: true,
        zIndex: 1,
    },
];
export function WorkflowEditor({ segmentId, onClose, style }) {
    const { t } = useTranslation();
    // Narrow selectors (NOT the whole store): otherwise WorkflowEditor re-renders on every project-store
    // change incl. the playhead during playback → the reconcile effect churns at frame rate.
    const tracks = useProjectStore((s) => s.tracks);
    const mergeProcessedOutputs = useProjectStore((s) => s.mergeProcessedOutputs);
    const updateTrackRef = useRef(useProjectStore.getState().updateTrack);
    updateTrackRef.current = useProjectStore.getState().updateTrack;
    const executionState = useWorkflowStore((s) => s.executions[segmentId]);
    // A pending split-mid-render LINK target mirrors its render SOURCE for display (the source's single render
    // feeds both halves via RenderLinkWatcher) and is LOCKED — it must not run its own copy of the nodes
    // (that would start a rejected/duplicate render). renderLocked drives the read-through + the run lock.
    const linkedSource = useWorkflowStore((s) => s.renderLinks[segmentId]);
    const linkedExec = useWorkflowStore((s) => (linkedSource ? s.executions[linkedSource] : undefined));
    const effExec = linkedExec ?? executionState;
    const renderLocked = linkedSource !== undefined;
    // The reconciler (below) re-runs when this segment's render cache changes (a node just finished).
    const nodeOutputsForSegment = useWorkflowStore((s) => s.nodeOutputs[segmentId]);
    // Node status map (a handful of writes per run, no per-frame churn): re-fires the edge render-lock
    // classes (displayEdges below) as nodes enter/leave the queued/running set.
    const segNodeStatuses = useWorkflowStore((s) => s.nodeStatuses[segmentId]);
    // Focus-based Ctrl+Z / edit-key ownership (the panel is co-visible with the tracks now): clicking or
    // focusing anywhere in the editor claims the "workflow" pane; the track area reclaims "timeline".
    const setActivePane = useAppStore((s) => s.setActivePane);
    const activePane = useAppStore((s) => s.activePane);
    const focusEditor = useCallback(() => setActivePane("workflow"), [setActivePane]);
    useEffect(() => {
        useMsstModelStore.getState().fetchModelsDir();
        useMsstModelStore.getState().fetchInstalled();
    }, []);
    // On open, if this segment was rendered in a PRIOR session (persisted processedOutputs) but the runtime
    // render cache is cold (project just loaded/autoloaded), rebuild the cache + node badges from the saved
    // deposits — so the separation/render nodes show completed and deleting an Output edge then reconnecting
    // it re-deposits from cache instead of forcing a full re-separation of audio that already exists. Runs
    // once per open (the panel is keyed by segmentId → it remounts on every segment switch). A pending
    // split-mid-render LINK target is skipped (RenderLinkWatcher owns its lanes until the source settles).
    useEffect(() => {
        if (useWorkflowStore.getState().renderLinks[segmentId])
            return;
        const seg = useProjectStore.getState().tracks.flatMap((t) => t.segments).find((s) => s.id === segmentId);
        if (seg)
            rehydrateRenderState(segmentId, seg);
    }, [segmentId]);
    const segment = tracks
        .flatMap((tr) => tr.segments.map((s) => ({ trackId: tr.id, seg: s })))
        .find((x) => x.seg.id === segmentId);
    const initialData = segment?.seg.workflow
        ? workflowToReactFlow(segment.seg.workflow)
        : { nodes: defaultNodes, edges: [] };
    const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);
    const saveTimer = useRef(undefined);
    const segTrackIdRef = useRef(segment?.trackId);
    segTrackIdRef.current = segment?.trackId;
    const [rfInstance, setRfInstance] = useState(null);
    // 可折叠左侧节点面板(默认展开;折叠后释放画布空间)。
    const [paletteOpen, setPaletteOpen] = useState(true);
    const paletteCollapseLabel = i18n.language.startsWith("ja")
        ? "ノードパネルを折りたたむ"
        : i18n.language.startsWith("en")
            ? "Collapse node palette"
            : "收起节点面板";
    const paletteExpandLabel = i18n.language.startsWith("ja")
        ? "ノードパネルを展開"
        : i18n.language.startsWith("en")
            ? "Expand node palette"
            : "展开节点面板";
    const [nodeCtx, setNodeCtx] = useState(null);
    const [edgeCtx, setEdgeCtx] = useState(null);
    // Canvas-background right-click → "add a node" menu, populated from the SAME palette source.
    const [paneCtx, setPaneCtx] = useState(null);
    // ─── 新增功能状态管理 ─────────────────────────────────────────────
    // 示例工作流库
    const [showExampleLibrary, setShowExampleLibrary] = useState(false);
    // 新手引导
    const [showTutorial] = useState(shouldShowTutorial());
    // 节点注释系统
    const [editingAnnotation, setEditingAnnotation] = useState(null);
    // 智能连线建议
    const [connectingFrom, setConnectingFrom] = useState(null);
    // 增强错误提示
    const [errorDisplay, setErrorDisplay] = useState(null);
    // 批量节点操作
    const [selectedNodeIds, setSelectedNodeIds] = useState([]);
    const [batchCtx, setBatchCtx] = useState(null);
    const [showShortcuts, setShowShortcuts] = useState(false);
    const [snapToGrid, setSnapToGrid] = useState(() => {
        const saved = localStorage.getItem("workflow.snapToGrid");
        return saved ? JSON.parse(saved) : false;
    });
    // ─── 工作流预设 · 全局保存/加载 ─────────────────────────────────────────────
    // 保存:把当前节点图(类型/位置/参数 + 连接)序列化后按自定义名字存入应用数据目录;
    // 加载:在下拉里选一个已保存的预设,替换整张图(随后的 debounced save 会写回当前片段)。
    const [presetNames, setPresetNames] = useState([]);
    const loadPresetList = useCallback(async () => {
        try {
            const raw = await invoke("load_workflow_presets");
            const arr = JSON.parse(raw);
            setPresetNames(arr.map((p) => p.name));
            return arr;
        }
        catch {
            setPresetNames([]);
            return [];
        }
    }, []);
    const savePreset = async () => {
        const name = await useAppStore.getState().showConfirm({
            title: t("workflow.savePresetTitle"),
            body: "",
            buttons: [
                { id: "ok", label: t("common.confirm"), kind: "primary" },
                { id: "cancel", label: t("common.cancel") },
            ],
            input: { placeholder: t("workflow.savePresetPlaceholder") },
        });
        if (!name)
            return;
        const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
        try {
            await invoke("save_workflow_preset", { name, workflow: wf });
            await loadPresetList();
            useAppStore.getState().showToast(`${t("workflow.presetSaved")} ${name}`, "success");
        }
        catch (e) {
            useAppStore.getState().showToast(String(e), "error");
        }
    };
    const loadPreset = async (name) => {
        const arr = await loadPresetList();
        const entry = arr.find((p) => p.name === name);
        if (!entry)
            return;
        const loaded = workflowToReactFlow(entry.workflow);
        setNodes(loaded.nodes);
        setEdges(loaded.edges);
        useAppStore.getState().showToast(`${t("workflow.presetLoaded")} ${name}`, "success");
    };
    // 规划 6-7 预设导出/导入:导出把应用数据目录里的全部工作流预设写到用户选定的 JSON 文件;
    // 导入从 JSON 文件读取并按名字合并(同名覆盖),随后刷新下拉列表。数量走后端返回值。
    const exportPresets = async () => {
        const dest = await save({
            title: t("workflow.exportPresets"),
            filters: [{ name: "JSON", extensions: ["json"] }],
        });
        if (!dest || typeof dest !== "string")
            return;
        try {
            const n = await invoke("export_workflow_presets", { dest });
            useAppStore.getState().showToast(t("workflow.presetsExported", { n }), "success");
        }
        catch (e) {
            useAppStore.getState().showToast(String(e), "error");
        }
    };
    const importPresets = async () => {
        const src = await open({
            title: t("workflow.importPresets"),
            multiple: false,
            directory: false,
            filters: [{ name: "JSON", extensions: ["json"] }],
        });
        if (!src || typeof src !== "string")
            return;
        try {
            const n = await invoke("import_workflow_presets", { src });
            await loadPresetList();
            useAppStore.getState().showToast(t("workflow.presetsImported", { n }), "success");
        }
        catch (e) {
            useAppStore.getState().showToast(String(e), "error");
        }
    };
    useEffect(() => { void loadPresetList(); }, [loadPresetList]);
    // Live refs (used by single-node run + the modal-local undo capture / drag coalescing).
    const nodesRef = useRef(nodes);
    nodesRef.current = nodes;
    const edgesRef = useRef(edges);
    edgesRef.current = edges;
    // --- Modal-local undo/redo for the node graph -----------------------------
    // The workflow editor OWNS Ctrl+Z while open (registered via setUndoScope below) — a self-contained
    // snapshot stack of {nodes, edges} that is independent of the timeline history. Selection/dimension
    // changes are excluded; a node DRAG is coalesced into one step (dragStart→dragStop); add/delete/
    // connect/param edits are one step each. A successful RUN is a COMMIT BARRIER that CLEARS the stack
    // — so pressing undo right after a render does nothing, and you can never undo across a render. This
    // is the agreed "a render is a commit" rule (audio-track node workflows only; vocal-track rendering
    // will get its own, more complex rule later). The persisted segment.workflow is deliberately kept
    // OUT of the timeline undo (excluded from its meaningful diff), so node editing never makes a
    // timeline step.
    // The stack lives PER SEGMENT in the module-level nodeHistory map, NOT in refs — the keyed remount
    // on a segment switch used to destroy the previous segment's history (detach piece 1 → detach piece
    // 2 → back to piece 1: its 解组 could never be undone). The map entry is mutated in place; a
    // remount resumes it (safe — only this segment's own open editor ever mutates its graph).
    const hist = useMemo(() => nodeHistoryFor(segmentId, () => ({
        past: [],
        future: [],
        commit: { nodes: initialData.nodes, edges: initialData.edges },
        sig: sigOfGraph(initialData.nodes, initialData.edges),
    })), 
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [segmentId]);
    const draggingNodeRef = useRef(false);
    const applyingLocalRef = useRef(false);
    const captureLocal = useCallback(() => {
        if (draggingNodeRef.current)
            return; // mid-drag frames coalesce; recorded on dragStop
        const sig = sigOfGraph(nodesRef.current, edgesRef.current);
        if (sig === hist.sig) {
            // No STRUCTURAL change (e.g. a pure node MOVE — position is out of the signature, so dragging a node
            // is not an undo step, or a selection-only change). Still refresh the baseline graph so a LATER real
            // edit's undo doesn't snap nodes back to a stale position.
            hist.commit = { nodes: nodesRef.current, edges: edgesRef.current };
            return;
        }
        hist.past.push(hist.commit);
        if (hist.past.length > 100)
            hist.past.shift();
        hist.future = [];
        hist.commit = { nodes: nodesRef.current, edges: edgesRef.current };
        hist.sig = sig;
    }, [hist]);
    const applyLocal = useCallback((snap) => {
        applyingLocalRef.current = true; // consumed by the capture effect so undo/redo doesn't re-record
        // Node POSITION is not undoable: overlay each SURVIVING node's CURRENT position onto the restored
        // structure, so undoing/redoing a graph edit never yanks nodes back to where they sat at capture time
        // (the reported "Ctrl+Z reverts my node move together with the real op"). A node RE-ADDED by the
        // undo/redo has no current position, so it keeps the snapshot's.
        const curPos = new Map(nodesRef.current.map((n) => [n.id, n.position]));
        const nodes = snap.nodes.map((n) => {
            const pos = curPos.get(n.id);
            return pos ? { ...n, position: pos } : n;
        });
        setNodes(nodes);
        setEdges(snap.edges);
        hist.commit = { nodes, edges: snap.edges };
        hist.sig = sigOfGraph(nodes, snap.edges);
    }, [setNodes, setEdges, hist]);
    const localUndo = useCallback(() => {
        if (draggingNodeRef.current || applyingLocalRef.current)
            return; // not mid-drag / re-entrant
        if (hist.past.length === 0)
            return;
        const cur = hist.commit;
        hist.future.push(cur);
        const before = hist.past.pop();
        applyLocal(before);
        announceNode(before, cur, "undo"); // the undone node op transformed before→cur
    }, [applyLocal, hist]);
    const localRedo = useCallback(() => {
        if (draggingNodeRef.current || applyingLocalRef.current)
            return;
        if (hist.future.length === 0)
            return;
        const cur = hist.commit;
        hist.past.push(cur);
        const after = hist.future.pop();
        applyLocal(after);
        announceNode(cur, after, "redo");
    }, [applyLocal, hist]);
    const onNodeDragStart = useCallback(() => { draggingNodeRef.current = true; }, []);
    const onNodeDragStop = useCallback(() => { draggingNodeRef.current = false; captureLocal(); }, [captureLocal]);
    // Auto-capture node/edge edits (add / delete / connect / param) as undo steps. Mid-drag frames and
    // undo/redo applies are skipped (drag is captured on dragStop instead).
    useEffect(() => {
        if (applyingLocalRef.current) {
            applyingLocalRef.current = false;
            return;
        }
        captureLocal();
    }, [nodes, edges, captureLocal]);
    // Claim Ctrl+Z / Ctrl+Y while the editor is open; release on close so the timeline regains it.
    // canUndo/canRedo read the local stacks so the Edit menu's enablement matches this scope.
    useEffect(() => {
        setUndoScope({
            undo: localUndo,
            redo: localRedo,
            canUndo: () => hist.past.length > 0,
            canRedo: () => hist.future.length > 0,
        });
        return () => setUndoScope(null);
    }, [localUndo, localRedo, hist]);
    useEffect(() => {
        const trackId = segTrackIdRef.current;
        if (!trackId)
            return;
        clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(() => {
            const wf = reactFlowToWorkflow(nodes, edges);
            const currentTracks = useProjectStore.getState().tracks;
            const track = currentTracks.find((tr) => tr.id === trackId);
            if (!track)
                return;
            const updatedSegments = track.segments.map((s) => s.id === segmentId ? { ...s, workflow: wf } : s);
            updateTrackRef.current(trackId, { segments: updatedSegments });
        }, 300);
        return () => clearTimeout(saveTimer.current);
    }, [nodes, edges, segmentId]);
    // A node is "busy" while it is running or queued in the active run. Deleting it mid-render would
    // only drop the UI while the backend job keeps going — so we block deletion of busy nodes.
    const isNodeBusy = useCallback((nodeId) => {
        // Only guard during a LIVE run. Once it settles (completed / error / cancelled) the node frees
        // up — otherwise a stale "waiting" left behind by a cancel or partial failure would make the
        // downstream nodes permanently un-deletable until a full re-run.
        if (useWorkflowStore.getState().executions[segmentId]?.status !== "running")
            return false;
        // Output nodes are NEVER busy: deleting one mid-run only drops its deposit intent (the backend job
        // runs on the upstream nodes + the dispatch-time graph snapshot), so they stay freely deletable —
        // even while showing the blue "depositing" pulse. Only the upstream RENDER path locks.
        if (nodesRef.current.find((n) => n.id === nodeId)?.type === "audioOutput")
            return false;
        const st = useWorkflowStore.getState().nodeStatuses[segmentId]?.[nodeId];
        return st === "running" || st === "waiting";
    }, [segmentId]);
    const onConnect = useCallback((connection) => {
        // Symmetric with the delete lock: while a node is queued/running, its INPUTS are frozen — wiring a
        // new edge into it mid-run would show a graph the in-flight job (dispatch snapshot) isn't using.
        if (connection.target && isNodeBusy(connection.target))
            return;
        setEdges((eds) => addEdge({ ...connection, animated: true }, eds));
        // 连线完成后清除智能连线建议状态
        setConnectingFrom(null);
    }, [setEdges, isNodeBusy]);
    // 智能连线建议:拖线开始时记录源端口 → 面板高亮所有类型兼容的目标端口;
    // 松手(无论落线成功与否)都必须清除,否则提示面板会永久挂在画布上。
    const onConnectStart = useCallback((_e, params) => {
        if (params.nodeId && params.handleType === "source") {
            setConnectingFrom({ node: params.nodeId, handle: params.handleId ?? "out-0" });
        }
    }, []);
    const onConnectEnd = useCallback(() => setConnectingFrom(null), []);
    // 规划 6-2 连线校验:拖新线瞬间按端口类型表放行/拒绝(同型或任一端 any 才允许落线,自连拒绝),
    // 从源头杜绝「报告 JSON 接进音频链」这类哑线。已存在于旧图里的类型不匹配连线不受影响(只拦新拖)。
    const isValidConnection = useCallback((conn) => {
        if (!conn.source || !conn.target || conn.source === conn.target)
            return false;
        const fromNode = nodesRef.current.find((n) => n.id === conn.source);
        const toNode = nodesRef.current.find((n) => n.id === conn.target);
        if (!fromNode || !toNode)
            return false;
        const fromWf = rfTypeToWfType[fromNode.type ?? ""];
        const toWf = rfTypeToWfType[toNode.type ?? ""];
        if (!fromWf || !toWf)
            return true; // 未知类型不拦(向后兼容)
        const fromPort = Number.parseInt((conn.sourceHandle ?? "").replace(/^out-/, ""), 10);
        const toPort = Number.parseInt((conn.targetHandle ?? "").replace(/^in-/, ""), 10);
        if (!Number.isFinite(fromPort) || !Number.isFinite(toPort))
            return true;
        return canConnect(fromWf, fromPort, toWf, toPort);
    }, []);
    // Phase 4-4: 在 flowPos 处垂直铺开一条母带预设链并顺序连线 (Handle 命名见 NodeShell: in-0 / out-0).
    // 声明必须先于 onAddNode/onDropNode/addNodeAtPane — 三者都把 masterPreset:* 路由到这.
    const insertMasterChain = useCallback((presetKey, flowPos) => {
        const chain = MASTER_CHAINS[presetKey];
        if (!chain)
            return;
        const ids = [];
        const newNodes = chain.map((step, i) => {
            nodeCounter++;
            const id = `${step.type}-${crypto.randomUUID().slice(0, 8)}`;
            ids.push(id);
            return {
                id,
                type: step.type,
                position: { x: flowPos.x, y: flowPos.y + i * 250 },
                data: { label: i18n.t(MASTER_CHAIN_LABELS[step.type] ?? step.type), params: { ...(step.params ?? {}) } },
                zIndex: nodeCounter + 100,
            };
        });
        const newEdges = [];
        for (let i = 0; i + 1 < ids.length; i++) {
            const source = ids[i];
            const target = ids[i + 1];
            if (!source || !target)
                continue;
            newEdges.push({
                id: `e-${source}-${target}`,
                source,
                target,
                sourceHandle: "out-0",
                targetHandle: "in-0",
                animated: true,
            });
        }
        setNodes((nds) => [...nds, ...newNodes]);
        setEdges((eds) => [...eds, ...newEdges]);
    }, [setNodes, setEdges]);
    const onAddNode = useCallback((type, label, extraParams) => {
        // masterPreset:* 不是单节点 — 是一条预连线母带链 (调色板点击与拖放都汇聚到这).
        if (type.startsWith("masterPreset:")) {
            insertMasterChain(type.slice("masterPreset:".length), { x: 300 + Math.random() * 100, y: 150 + Math.random() * 100 });
            return;
        }
        nodeCounter++;
        const id = `${type}-${crypto.randomUUID().slice(0, 8)}`;
        const defaultParams = { ...(extraParams ?? {}) };
        if (type === "audioOutput")
            defaultParams.laneLabel = DEFAULT_OUTPUT_GROUP;
        // S62c: range extension is OPT-IN per node (user decision — the whole-render recolor
        // tradeoff must never be on by default). Absent key = OFF via DEFAULTS/serde false; a
        // pre-S62 node's explicitly-written `range_extend: true` keeps rendering extended.
        const newNode = {
            id,
            type,
            position: { x: 300 + Math.random() * 100, y: 150 + Math.random() * 100 },
            data: { label, params: defaultParams },
            zIndex: nodeCounter + 100,
        };
        setNodes((nds) => [...nds, newNode]);
    }, [setNodes, insertMasterChain]);
    const onDropNode = useCallback((type, label, clientX, clientY, extraParams) => {
        if (!rfInstance)
            return;
        // Check if drop is within the canvas area
        const canvasEl = document.querySelector(".workflow-canvas");
        if (!canvasEl)
            return;
        const rect = canvasEl.getBoundingClientRect();
        if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom)
            return;
        const position = rfInstance.screenToFlowPosition({ x: clientX, y: clientY });
        if (type.startsWith("masterPreset:")) {
            insertMasterChain(type.slice("masterPreset:".length), position);
            return;
        }
        nodeCounter++;
        const id = `${type}-${crypto.randomUUID().slice(0, 8)}`;
        const defaultParams = { ...(extraParams ?? {}) };
        if (type === "audioOutput")
            defaultParams.laneLabel = DEFAULT_OUTPUT_GROUP;
        // S62c: range extension is OPT-IN per node (user decision — the whole-render recolor
        // tradeoff must never be on by default). Absent key = OFF via DEFAULTS/serde false; a
        // pre-S62 node's explicitly-written `range_extend: true` keeps rendering extended.
        const newNode = {
            id,
            type,
            position,
            data: { label, params: defaultParams },
            zIndex: nodeCounter + 100,
        };
        setNodes((nds) => [...nds, newNode]);
    }, [setNodes, rfInstance, insertMasterChain]);
    const closeMenus = useCallback(() => { setNodeCtx(null); setEdgeCtx(null); setPaneCtx(null); setBatchCtx(null); }, []);
    const onSelectionChange = useCallback(({ nodes: selectedNodes }) => {
        setSelectedNodeIds(selectedNodes.map((n) => n.id));
    }, []);
    // 批量操作一律吃「菜单打开那一刻的 id 快照」(batchCtx.ids)或快捷键当场读到的选区,不再读可能已
    // 被后续点击改写的 selectedNodeIds —— 菜单停留期间误改选区不会让操作打到别的节点上。
    const handleBatchDelete = useCallback((ids) => {
        // 与单节点删除同一口径:IO 节点 deletable:false,执行中/排队中的节点锁定。
        const victims = new Set(ids.filter((id) => {
            const node = nodesRef.current.find((n) => n.id === id);
            return !!node && node.deletable !== false && !isNodeBusy(id);
        }));
        if (victims.size === 0)
            return;
        setNodes((nds) => nds.filter((n) => !victims.has(n.id)));
        // 任一端被删 → 连线必须一起删(留下悬空边会让 parseWorkflowGraph 报「成环」并弄坏整张图)。
        setEdges((eds) => eds.filter((e) => !victims.has(e.source) && !victims.has(e.target)));
        setBatchCtx(null);
    }, [setNodes, setEdges, isNodeBusy]);
    const handleBatchBypass = useCallback((ids, bypass) => {
        const targets = new Set(ids.filter((id) => {
            const node = nodesRef.current.find((n) => n.id === id);
            return !!node && node.type !== "audioInput" && node.type !== "audioOutput";
        }));
        if (targets.size === 0)
            return;
        setNodes((nds) => nds.map((n) => (targets.has(n.id) ? { ...n, data: { ...n.data, bypass } } : n)));
        setBatchCtx(null);
    }, [setNodes]);
    // 复制排除 audioInput:一张图只能有一个输入节点(多一个 → errGraphMultiInput),粘贴出第二个
    // 只会让图不可运行。只保留「两端都在选区内」的连线,避免粘贴出指向原图节点的跨图边。
    const handleBatchCopy = useCallback((ids) => {
        const picked = new Set(ids);
        const copyNodes = nodesRef.current
            .filter((n) => picked.has(n.id) && n.type !== "audioInput")
            .map(({ selected, dragging, ...rest }) => rest);
        if (copyNodes.length === 0)
            return;
        const copyIds = new Set(copyNodes.map((n) => n.id));
        const copyEdges = edgesRef.current.filter((e) => copyIds.has(e.source) && copyIds.has(e.target));
        try {
            localStorage.setItem("workflow-clipboard", JSON.stringify({ nodes: copyNodes, edges: copyEdges }));
            useAppStore.getState().showToast(t("workflow.batchCopied", { count: copyNodes.length }), "success");
        }
        catch {
            useAppStore.getState().showToast(t("workflow.batchCopyError"), "error");
        }
        setBatchCtx(null);
    }, [t]);
    const handleBatchPaste = useCallback(() => {
        const raw = localStorage.getItem("workflow-clipboard");
        if (!raw)
            return;
        try {
            const parsed = JSON.parse(raw);
            const clipNodes = Array.isArray(parsed.nodes) ? parsed.nodes : [];
            const clipEdges = Array.isArray(parsed.edges) ? parsed.edges : [];
            if (clipNodes.length === 0)
                return;
            // 全新 id + 偏移 40px 落点,并重写连线端点到新 id —— 粘贴出的是一份独立子图。
            const idMap = new Map();
            const newNodes = clipNodes.map((n) => {
                nodeCounter++;
                const newId = `${n.type}-${crypto.randomUUID().slice(0, 8)}`;
                idMap.set(n.id, newId);
                return {
                    ...n,
                    id: newId,
                    position: { x: (n.position?.x ?? 0) + 40, y: (n.position?.y ?? 0) + 40 },
                    selected: true,
                    zIndex: nodeCounter + 100,
                };
            });
            const newEdges = clipEdges.flatMap((e) => {
                const source = idMap.get(e.source);
                const target = idMap.get(e.target);
                if (!source || !target)
                    return []; // 端点不在本次粘贴集合内 → 丢弃,绝不留悬空边
                return [{ ...e, id: `e-${crypto.randomUUID().slice(0, 8)}`, source, target, selected: false }];
            });
            // 粘贴后选区切到新节点(旧选区取消),可以直接拖走或继续批量操作。
            setNodes((nds) => [...nds.map((n) => (n.selected ? { ...n, selected: false } : n)), ...newNodes]);
            setEdges((eds) => [...eds, ...newEdges]);
            useAppStore.getState().showToast(t("workflow.batchPasted", { count: newNodes.length }), "success");
        }
        catch {
            useAppStore.getState().showToast(t("workflow.batchPasteError"), "error");
        }
    }, [setNodes, setEdges, t]);
    const handleSelectAll = useCallback(() => {
        setNodes((nds) => nds.map((n) => (n.selected ? n : { ...n, selected: true })));
    }, [setNodes]);
    // ─── Opt 9 · 一键整理画布 ───────────────────────────────────────────────────
    // 注意:节点位置**刻意不进**撤销栈(见 sigOfGraph 上方那段注释 —— 位置不是有意义的编辑,
    // 拖节点不该产生一步撤销)。这意味着整理之后 Ctrl+Z 是**救不回**原布局的,所以这里自己留一份
    // 整理前的坐标快照,并在工具栏上放一个「还原布局」按钮。不去改 sigOfGraph 把位置纳入签名:
    // 那会让每次拖动节点都变成一步撤销,破坏既有手感(那正是它被排除的原因)。
    const preLayoutRef = useRef(null);
    const [canRestoreLayout, setCanRestoreLayout] = useState(false);
    const handleAutoLayout = useCallback(() => {
        const cur = nodesRef.current;
        if (cur.length === 0)
            return;
        const { positions, moved } = computeAutoLayout(cur.map((n) => ({
            id: n.id,
            type: n.type,
            position: n.position,
            // ReactFlow v12 渲染后把实测尺寸回填到 node.measured;首帧或隐藏节点可能还没有,
            // computeAutoLayout 内部会退回默认尺寸。
            ...(n.measured ? { measured: n.measured } : {}),
        })), edgesRef.current.map((e) => ({ source: e.source, target: e.target })));
        const app = useAppStore.getState();
        if (moved === 0) {
            // 已经是整理后的样子:什么都不写,免得白占一次「还原」快照。
            app.showToast(t("workflow.autoLayoutNoop"), "info");
            return;
        }
        preLayoutRef.current = new Map(cur.map((n) => [n.id, { ...n.position }]));
        setCanRestoreLayout(true);
        setNodes((nds) => nds.map((n) => {
            const p = positions.get(n.id);
            return p && (p.x !== n.position.x || p.y !== n.position.y) ? { ...n, position: p } : n;
        }));
        // 整理后把视野拉回到整张图上 —— 不然节点被挪到视口外会看着像「消失了」。
        // 位置写入是同步的,但 ReactFlow 要等下一帧才知道新坐标,所以 fitView 放到下一帧。
        requestAnimationFrame(() => rfInstance?.fitView({ padding: 0.15, duration: 300 }));
        app.showToast(t("workflow.autoLayoutDone", { count: moved }), "success");
    }, [setNodes, rfInstance, t]);
    const handleRestoreLayout = useCallback(() => {
        const snap = preLayoutRef.current;
        if (!snap)
            return;
        // 只还原**仍然存在**的节点;整理之后新加的节点没有快照,保持原地不动。
        setNodes((nds) => nds.map((n) => {
            const p = snap.get(n.id);
            return p && (p.x !== n.position.x || p.y !== n.position.y) ? { ...n, position: { ...p } } : n;
        }));
        preLayoutRef.current = null;
        setCanRestoreLayout(false);
        requestAnimationFrame(() => rfInstance?.fitView({ padding: 0.15, duration: 300 }));
        useAppStore.getState().showToast(t("workflow.autoLayoutRestored"), "info");
    }, [setNodes, rfInstance, t]);
    const handleAlign = useCallback((mode) => {
        const cur = nodesRef.current.filter((n) => n.selected);
        if (cur.length === 0)
            return;
        const { positions, moved } = alignNodes(cur.map((n) => ({
            id: n.id,
            type: n.type,
            position: n.position,
            ...(n.measured ? { measured: n.measured } : {}),
        })), mode);
        if (moved === 0) {
            useAppStore.getState().showToast(t("workflow.alignNoop"), "info");
            return;
        }
        setNodes((nds) => nds.map((n) => {
            const p = positions.get(n.id);
            return p ? { ...n, position: p } : n;
        }));
        useAppStore.getState().showToast(t("workflow.alignDone", { count: moved }), "success");
    }, [setNodes, t]);
    const handleDistribute = useCallback((mode) => {
        const cur = nodesRef.current.filter((n) => n.selected);
        if (cur.length < 3) {
            useAppStore.getState().showToast(t("workflow.distributeNeedThree"), "info");
            return;
        }
        const { positions, moved } = distributeNodes(cur.map((n) => ({
            id: n.id,
            type: n.type,
            position: n.position,
            ...(n.measured ? { measured: n.measured } : {}),
        })), mode);
        if (moved === 0) {
            useAppStore.getState().showToast(t("workflow.alignNoop"), "info");
            return;
        }
        setNodes((nds) => nds.map((n) => {
            const p = positions.get(n.id);
            return p ? { ...n, position: p } : n;
        }));
        useAppStore.getState().showToast(t("workflow.alignDone", { count: moved }), "success");
    }, [setNodes, t]);
    const toggleSnapToGrid = useCallback(() => {
        const newValue = !snapToGrid;
        setSnapToGrid(newValue);
        localStorage.setItem("workflow.snapToGrid", JSON.stringify(newValue));
        useAppStore.getState().showToast(t(newValue ? "workflow.snapToGridOn" : "workflow.snapToGridOff"), "info");
    }, [snapToGrid, t]);
    // Canvas right-click → add a node AT the clicked flow position (mirrors onDropNode, which places
    // a dragged palette node at the drop point; the pane menu is the click-only equivalent).
    const addNodeAtPane = useCallback((def, pos) => {
        if (!rfInstance)
            return;
        const position = rfInstance.screenToFlowPosition({ x: pos.x, y: pos.y });
        if (def.type.startsWith("masterPreset:")) {
            insertMasterChain(def.type.slice("masterPreset:".length), position);
            setPaneCtx(null);
            return;
        }
        nodeCounter++;
        const id = `${def.type}-${crypto.randomUUID().slice(0, 8)}`;
        const defaultParams = { ...(def.extraParams ?? {}) };
        if (def.type === "audioOutput")
            defaultParams.laneLabel = DEFAULT_OUTPUT_GROUP;
        const newNode = {
            id,
            type: def.type,
            position,
            data: { label: def.label, params: defaultParams },
            zIndex: nodeCounter + 100,
        };
        setNodes((nds) => [...nds, newNode]);
        setPaneCtx(null);
    }, [rfInstance, setNodes, insertMasterChain]);
    // An edge is render-locked ONLY when its CONSUMER (target) is queued/running in the live run — that
    // input is what the run is (about to be) computing with, so cutting it would misrepresent the run.
    // An edge OUT of a busy node into an idle/new node affects nothing that is running (the backend job
    // works on the dispatch-time snapshot) and stays freely deletable — the old predicate locked on
    // EITHER endpoint, which froze exactly the "rendering node → my freshly added node" wire the user
    // wants to cut. Edges INTO an Output node are never locked (isNodeBusy is false for audioOutput by
    // construction: deposit intent is not a render path).
    const isEdgeRenderLocked = useCallback((edge) => isNodeBusy(edge.target), [isNodeBusy]);
    // Edge affordance classes: a deletable wire lights up on hover (CSS .wf-edge-free:hover), a
    // render-locked one gets a not-allowed cursor + dim. className is display-only (reactFlowToWorkflow
    // never reads it). The lock VALUE is read through isEdgeRenderLocked — the same predicate the delete
    // veto uses — so the visual and the behavior can never drift; segNodeStatuses/executionState are
    // subscribed purely to re-fire this memo when the busy set changes.
    const displayEdges = useMemo(() => edges.map((e) => {
        const isLocked = isEdgeRenderLocked(e);
        const sourceStatus = segNodeStatuses?.[e.source];
        const isSourceRunning = sourceStatus === "running";
        return {
            ...e,
            className: isLocked ? "wf-edge-locked" : "wf-edge-free",
            animated: isSourceRunning,
        };
    }), [edges, segNodeStatuses, executionState?.status, isEdgeRenderLocked]);
    const onNodeContextMenu = useCallback((event, node) => {
        event.preventDefault();
        if (node.deletable === false)
            return;
        if (isNodeBusy(node.id))
            return; // 正在执行/排队的节点不弹删除菜单
        // 如果右键的节点在多选集合中,显示批量操作菜单
        if (selectedNodeIds.length > 1 && selectedNodeIds.includes(node.id)) {
            setNodeCtx(null);
            setEdgeCtx(null);
            setBatchCtx({ x: event.clientX, y: event.clientY, ids: selectedNodeIds });
            return;
        }
        setEdgeCtx(null);
        setBatchCtx(null);
        setNodeCtx({ x: event.clientX, y: event.clientY, nodeId: node.id });
    }, [isNodeBusy, selectedNodeIds]);
    const onEdgeContextMenu = useCallback((event, edge) => {
        event.preventDefault();
        if (isEdgeRenderLocked(edge))
            return; // only edges feeding a queued/running node are locked
        setNodeCtx(null);
        setEdgeCtx({ x: event.clientX, y: event.clientY, edgeId: edge.id });
    }, [isEdgeRenderLocked]);
    const handleDeleteNode = useCallback((nodeId) => {
        setNodes((nds) => nds.filter((n) => n.id !== nodeId));
        setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
        setNodeCtx(null);
    }, [setNodes, setEdges]);
    const handleDeleteEdge = useCallback((edgeId) => {
        setEdges((eds) => eds.filter((e) => e.id !== edgeId));
        setEdgeCtx(null);
    }, [setEdges]);
    // Veto deletion (Delete key OR programmatic) of busy nodes. IO nodes are already non-deletable
    // (deletable:false) so ReactFlow filters them out before this runs; we only guard busy ones.
    const onBeforeDelete = useCallback(async ({ nodes: delNodes, edges: delEdges }) => {
        // A node is locked while running/queued. An EDGE is locked only when its TARGET is (see
        // isEdgeRenderLocked) — source-side edges into idle nodes stay deletable even mid-run.
        const allowedNodes = delNodes.filter((n) => !isNodeBusy(n.id));
        const deletedIds = new Set(allowedNodes.map((n) => n.id));
        // An edge MUST be removed if EITHER endpoint is being removed — never leave a dangling edge
        // (a deleted-source edge poisons parseWorkflowGraph → "contains a cycle" and bricks the graph).
        // Otherwise it's the user deleting just the edge: allowed unless it feeds a queued/running node.
        const allowedEdges = delEdges.filter((e) => deletedIds.has(e.source) || deletedIds.has(e.target) || !isEdgeRenderLocked(e));
        if (allowedNodes.length === delNodes.length && allowedEdges.length === delEdges.length) {
            return { nodes: delNodes, edges: delEdges };
        }
        if (allowedNodes.length === 0 && allowedEdges.length === 0)
            return false;
        return { nodes: allowedNodes, edges: allowedEdges };
    }, [isNodeBusy, isEdgeRenderLocked]);
    // 规划 6-3 旁通切换:仅翻转 bypass 标志(sigOfGraph 含旁通位 → 撤销自动捕获为一步)。
    const toggleBypass = useCallback((nodeId) => {
        setNodes((nds) => nds.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, bypass: n.data?.bypass !== true } } : n)));
        setNodeCtx(null);
    }, [setNodes]);
    // ── Opt 7 节点参数预设 ────────────────────────────────────────────────────
    // localStorage 不是响应式的:存/删之后用这个计数器让菜单项重算,否则右键菜单里
    // 还是上一次的列表。
    const [paramPresetTick, setParamPresetTick] = useState(0);
    const handleSaveNodeParamPreset = useCallback(async (nodeId) => {
        const node = nodesRef.current.find((n) => n.id === nodeId);
        setNodeCtx(null);
        if (!node?.type)
            return;
        const params = node.data?.params ?? {};
        const app = useAppStore.getState();
        if (Object.keys(params).length === 0) {
            app.showToast(t("workflow.paramPreset.noParams"), "info");
            return;
        }
        // 带 input 的 showConfirm 里,primary 键 resolve 的是输入框文本,取消/Esc/点遮罩 resolve ""
        // (见 ConfirmDialog),所以空串就是取消 —— 和工作流预设那边同一个口径。
        const name = await app.showConfirm({
            title: t("workflow.paramPreset.saveTitle"),
            body: "",
            buttons: [
                { id: "ok", label: t("common.confirm"), kind: "primary" },
                { id: "cancel", label: t("common.cancel") },
            ],
            input: { placeholder: t("workflow.paramPreset.savePlaceholder") },
        });
        if (!name)
            return;
        if (!saveNodeParamPreset(node.type, name, params)) {
            app.showToast(t("workflow.paramPreset.limit"), "warning");
            return;
        }
        setParamPresetTick((v) => v + 1);
        app.showToast(`${t("workflow.paramPreset.saved")}${name}`, "success");
    }, [t]);
    const handleApplyNodeParamPreset = useCallback((nodeId, preset) => {
        // 合并而不是整体替换:预设可能是旧版本存的,缺的键保留节点当前值,免得把必填参数清空。
        // 走 setNodes → sigOfGraph 含 params → 自动被捕获成一步撤销,Ctrl+Z 可以退回。
        setNodes((nds) => nds.map((n) => (n.id === nodeId
            ? { ...n, data: { ...n.data, params: { ...(n.data?.params ?? {}), ...preset.params } } }
            : n)));
        setNodeCtx(null);
        useAppStore.getState().showToast(`${t("workflow.paramPreset.applied")}${preset.name}`, "success");
    }, [setNodes, t]);
    const handleDeleteNodeParamPreset = useCallback((nodeType, name) => {
        deleteNodeParamPreset(nodeType, name);
        setParamPresetTick((v) => v + 1);
        setNodeCtx(null);
        useAppStore.getState().showToast(`${t("workflow.paramPreset.deleted")}${name}`, "info");
    }, [t]);
    // 示例工作流库:整图替换当前画布(与 loadPreset 同一口径 —— 走 workflowToReactFlow,
    // 由 300ms 防抖的保存 effect 落盘;撤销栈照常捕获,误载可以直接 Ctrl+Z 退回)。
    const handleLoadExampleWorkflow = useCallback((example) => {
        const loaded = workflowToReactFlow(example.workflow);
        setNodes(loaded.nodes);
        setEdges(loaded.edges);
        useAppStore.getState().showToast(`${t("workflow.exampleLibrary.loaded")} ${i18n.language === "en" ? example.nameEn : example.name}`, "success");
    }, [setNodes, setEdges, t]);
    const handleSaveAnnotation = useCallback((nodeId, content, color) => {
        setNodes(prev => prev.map(n => {
            if (n.id !== nodeId)
                return n;
            const existing = n.data?.annotation;
            const now = Date.now();
            return {
                ...n,
                data: {
                    ...n.data,
                    annotation: {
                        content,
                        color: color ?? existing?.color ?? "yellow",
                        createdAt: existing?.createdAt ?? now,
                        updatedAt: now,
                    },
                },
            };
        }));
        setEditingAnnotation(null);
    }, [setNodes]);
    const handleDeleteAnnotation = useCallback((nodeId) => {
        setNodes(prev => prev.map(n => {
            if (n.id !== nodeId)
                return n;
            const { annotation, ...rest } = n.data ?? {};
            return { ...n, data: rest };
        }));
        setEditingAnnotation(null);
    }, [setNodes]);
    const nodeCtxNode = nodeCtx ? nodesRef.current.find((n) => n.id === nodeCtx.nodeId) : undefined;
    // Opt 7:参数预设只对处理节点有意义 —— IO 节点排除(和旁通同一口径)。
    const presetNodeType = nodeCtxNode?.type && nodeCtxNode.type !== "audioInput" && nodeCtxNode.type !== "audioOutput"
        ? nodeCtxNode.type
        : null;
    const paramPresetList = useMemo(() => (presetNodeType ? listNodeParamPresets(presetNodeType) : []), 
    // eslint-disable-next-line react-hooks/exhaustive-deps -- paramPresetTick 是存/删后的重算触发器(localStorage 非响应式)
    [presetNodeType, paramPresetTick]);
    const nodeCtxItems = nodeCtx ? [
        // "Detach": only for a MULTI-input Output node — splits it into one Output (group) per inbound edge.
        ...(nodeCtxNode?.type === "audioOutput" &&
            edgesRef.current.filter((e) => e.target === nodeCtx.nodeId).length >= 2
            ? [{
                    label: t("workflow.detachGroup"),
                    onClick: () => {
                        useAppStore.getState().requestLaneDetach(segmentId, nodeCtx.nodeId);
                        setNodeCtx(null);
                    },
                }]
            : []),
        // 规划 6-3 旁通/取消旁通(IO 节点无旁通意义,菜单里排除)。
        ...(nodeCtxNode && nodeCtxNode.type !== "audioInput" && nodeCtxNode.type !== "audioOutput"
            ? [{
                    label: nodeCtxNode.data?.bypass === true ? t("workflow.unbypassNode") : t("workflow.bypassNode"),
                    onClick: () => toggleBypass(nodeCtx.nodeId),
                }]
            : []),
        // 节点注释功能
        {
            label: nodeCtxNode?.data?.annotation ? t("workflow.annotation.edit") : t("workflow.annotation.add"),
            onClick: () => {
                setEditingAnnotation(nodeCtx.nodeId);
                setNodeCtx(null);
            },
        },
        // Opt 7 参数预设:存当前参数 / 套用 / 删除。套用与删除在没有预设时给一个禁用占位项,
        // 而不是把子菜单整块藏掉 —— 空子菜单点开是空白面板,反而像坏了。
        ...(presetNodeType
            ? [{
                    type: "submenu",
                    label: t("workflow.paramPreset.menu"),
                    items: [
                        {
                            label: t("workflow.paramPreset.save"),
                            onClick: () => { void handleSaveNodeParamPreset(nodeCtx.nodeId); },
                        },
                        { type: "divider" },
                        ...(paramPresetList.length === 0
                            ? [{ label: t("workflow.paramPreset.empty"), disabled: true, onClick: () => { } }]
                            : [
                                {
                                    type: "submenu",
                                    label: t("workflow.paramPreset.apply"),
                                    items: paramPresetList.map((p) => ({
                                        label: p.name,
                                        onClick: () => handleApplyNodeParamPreset(nodeCtx.nodeId, p),
                                    })),
                                },
                                {
                                    type: "submenu",
                                    label: t("workflow.paramPreset.delete"),
                                    danger: true,
                                    items: paramPresetList.map((p) => ({
                                        label: p.name,
                                        danger: true,
                                        onClick: () => handleDeleteNodeParamPreset(presetNodeType, p.name),
                                    })),
                                },
                            ]),
                    ],
                }]
            : []),
        { label: t("toolbar.delete"), shortcut: "Del", danger: true, onClick: () => handleDeleteNode(nodeCtx.nodeId) },
    ] : [];
    const edgeCtxItems = edgeCtx ? [
        { label: t("workflow.deleteConnection"), shortcut: "Del", danger: true, onClick: () => handleDeleteEdge(edgeCtx.edgeId) },
    ] : [];
    const batchCtxItems = batchCtx ? [
        {
            label: t("workflow.batchBypass"),
            onClick: () => handleBatchBypass(batchCtx.ids, true)
        },
        {
            label: t("workflow.batchUnbypass"),
            onClick: () => handleBatchBypass(batchCtx.ids, false)
        },
        {
            label: t("workflow.batchCopy"),
            shortcut: "Ctrl+C",
            onClick: () => handleBatchCopy(batchCtx.ids)
        },
        {
            type: "submenu",
            label: t("workflow.align"),
            items: [
                { label: t("workflow.alignLeft"), onClick: () => handleAlign("left") },
                { label: t("workflow.alignHCenter"), onClick: () => handleAlign("hcenter") },
                { label: t("workflow.alignRight"), onClick: () => handleAlign("right") },
                { label: t("workflow.alignTop"), onClick: () => handleAlign("top") },
                { label: t("workflow.alignVCenter"), onClick: () => handleAlign("vcenter") },
                { label: t("workflow.alignBottom"), onClick: () => handleAlign("bottom") },
            ],
        },
        {
            type: "submenu",
            label: t("workflow.distribute"),
            items: [
                { label: t("workflow.distributeH"), onClick: () => handleDistribute("horizontal") },
                { label: t("workflow.distributeV"), onClick: () => handleDistribute("vertical") },
            ],
        },
        {
            label: t("workflow.batchDelete"),
            shortcut: "Del",
            danger: true,
            onClick: () => handleBatchDelete(batchCtx.ids)
        },
    ] : [];
    // Pane menu = EVERY palette node (same defs as the sidebar → never drifts), flat in sidebar order.
    const installedModels = useMsstModelStore((s) => s.installed);
    const paletteGroups = useMemo(() => getPaletteDefs(i18n.language, new Set(installedModels.map((m) => m.filename))), [installedModels, i18n.language]);
    const paneCtxItems = paneCtx
        ? paletteGroups.flatMap((g) => g.nodes.map((n) => ({
            label: n.label,
            icon: `[${n.icon}]`,
            onClick: () => addNodeAtPane(n, paneCtx),
        })))
        : [];
    const handleExecute = useCallback(async () => {
        const trackId = segTrackIdRef.current;
        if (!segment || !trackId)
            return;
        if (useWorkflowStore.getState().renderLinks[segmentId])
            return; // linked half: its render is the source's
        const wf = reactFlowToWorkflow(nodes, edges);
        try {
            // Busy gate FIRST — before the deposit invalidation below. When the run is rejected (voice drain
            // drop, separation busy, double-run) NOTHING may have been touched yet: gating inside
            // executeWorkflow (the old shape) ran AFTER the lanes were stripped, so a rejected run cost the
            // track its deposits for nothing.
            if (!(await preflightRun(segmentId, wf, null)))
                return;
            // Barrier floor captured at DISPATCH so edits made DURING the async render stay undoable down to —
            // but not past — this point (read after the gate, so drain-wait edits count as pre-dispatch).
            const dispatchPastLen = hist.past.length;
            // The RECONCILER (running live during + after the run) owns the track lanes: at run start it places
            // loading placeholders for the connected Output lanes, then deposits each lane the moment its branch
            // finishes (executeWorkflow caches per node → the reconcile effect fires), and cleans the
            // placeholders of branches that never ran. So handleExecute no longer touches processedOutputs — it
            // only drives the render + the node-graph undo barrier.
            // Invalidate this segment's prior deposit + decoded buffers UP FRONT so every connected lane
            // shows a loading placeholder from the first moment of the run and each branch re-decodes fresh
            // as it finishes. (Output paths are run-unique now, so this is pure UX immediacy — staleness
            // itself is already impossible.)
            const segBefore = useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
            const stale = new Set();
            for (const o of segBefore?.processedOutputs ?? []) {
                if (o.loading)
                    continue;
                clearBufferCache(o.audioPath);
                if (o.outputNodeId)
                    stale.add(o.outputNodeId);
            }
            for (const outId of stale)
                useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outId);
            const laneCount = await executeWorkflow(segmentId, segment.seg, wf);
            if (laneCount === 0) {
                // Nothing reached an Output node (none connected, or no stems produced) — say so, don't sit silent.
                useAppStore.getState().showToast(i18n.t("workflow.noOutputs"), "error");
                return;
            }
            flushAutosaveNow(); // a render is a commit — snapshot to disk NOW (don't wait for the 1.5s debounce)
            // A render is a COMMIT BARRIER for the node-graph undo: drop the pre-render history (and any redo
            // branch) but KEEP edits made while the render was in flight undoable.
            hist.past = hist.past.slice(Math.min(dispatchPastLen, hist.past.length));
            hist.future = [];
        }
        catch (err) {
            // Live-deposit model: branches that FINISHED stay on the track (what rendered, rendered); the
            // reconciler removes the loading placeholders of branches that didn't. Just drop stuck badges here.
            useWorkflowStore.getState().clearPendingStatuses(segmentId);
            if (err instanceof Error && err.message === "Cancelled")
                return;
            console.error("Workflow execution failed:", err);
            const errorMsg = err instanceof Error ? err.message : String(err);
            setErrorDisplay({ error: errorMsg, nodeType: "workflow" });
        }
    }, [segmentId, segment, nodes, edges, hist]);
    // 键盘快捷键系统。handleExecute 随 nodes/edges/hist 变化 —— 若放进依赖数组,window 监听器
    // 会在每次改图时解绑重绑。改用 ref 承载最新实现(与本文件 nodesRef/segmentRef 同一套做法),
    // 监听器只注册一次。
    //
    // 这里**刻意不接** Ctrl+Z / Ctrl+Y / Ctrl+S,它们已经由 App.tsx 的全局 document 监听器处理:
    //  - Ctrl+Z/Y → routeUndo()/routeRedo(),而 store/history.ts 的 workflowUndoActive() 在本面板
    //    聚焦时会把它们转给上面 setUndoScope 注册的 localUndo/localRedo,所以撤销**已经**是工作流
    //    级的;在这里再接一次会让一次按键撤销两步。
    //  - Ctrl+S 是全局「保存项目」(App.tsx 明确写了 fire regardless of focus),和工作流预设
    //    另存不是一回事;两边都接会一次按键同时存项目 + 弹预设命名框。预设另存走工具栏按钮。
    const selectedIdsRef = useRef(selectedNodeIds);
    selectedIdsRef.current = selectedNodeIds;
    const shortcutsRef = useRef({
        handleSelectAll, handleBatchCopy, handleBatchPaste, handleBatchBypass,
        handleExecute, closeMenus, handleAutoLayout, handleAlign, handleDistribute,
        toggleSnapToGrid,
    });
    shortcutsRef.current = {
        handleSelectAll, handleBatchCopy, handleBatchPaste, handleBatchBypass,
        handleExecute, closeMenus, handleAutoLayout, handleAlign, handleDistribute,
        toggleSnapToGrid,
    };
    useEffect(() => {
        const onKeyDown = (e) => {
            const h = shortcutsRef.current;
            if (useAppStore.getState().activePane !== "workflow")
                return;
            const el = e.target;
            const inInput = el && (el.isContentEditable || el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT");
            // Escape: 关闭菜单/取消选择
            if (e.key === "Escape") {
                e.preventDefault();
                h.closeMenus();
                if (selectedIdsRef.current.length > 0) {
                    setNodes((nds) => nds.map((n) => (n.selected ? { ...n, selected: false } : n)));
                }
                return;
            }
            // ?键: 显示/隐藏快捷键面板
            if (e.key === "?" && !inInput) {
                e.preventDefault();
                setShowShortcuts((s) => !s);
                return;
            }
            if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey)
                return;
            if (inInput)
                return;
            const ids = selectedIdsRef.current;
            const key = e.key.toLowerCase();
            // Ctrl+E: 执行工作流
            if (key === "e") {
                e.preventDefault();
                void h.handleExecute();
            }
            // Ctrl+A: 全选
            else if (key === "a") {
                e.preventDefault();
                h.handleSelectAll();
            }
            // Ctrl+C: 复制
            else if (key === "c") {
                if (ids.length === 0)
                    return;
                e.preventDefault();
                h.handleBatchCopy(ids);
            }
            // Ctrl+V: 粘贴
            else if (key === "v") {
                e.preventDefault();
                h.handleBatchPaste();
            }
            // Ctrl+L: 一键整理画布。App.tsx 的全局 keydown 只占了 z/y/s/o/n,并且只拦
            // f/g/p/u/j/r 的浏览器默认行为 —— l 两边都没人用,不会打架。
            else if (key === "l") {
                e.preventDefault();
                h.handleAutoLayout();
            }
            // Ctrl+B: 批量旁通
            else if (key === "b") {
                if (ids.length === 0)
                    return;
                e.preventDefault();
                const picked = new Set(ids);
                const anyActive = nodesRef.current.some((n) => picked.has(n.id) && n.data?.bypass !== true);
                h.handleBatchBypass(ids, anyActive);
            }
            // Ctrl+G: 切换网格吸附
            else if (key === "g") {
                e.preventDefault();
                h.toggleSnapToGrid();
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [setNodes]);
    const handleCancel = useCallback(() => {
        const wf = useWorkflowStore.getState();
        if (wf.executions[segmentId]?.cancelled)
            return; // already cancelling — the backend stop is in flight
        // Flag only — the execution SETTLES (and the badges clear) when the backend actually stops
        // (engine catch → failExecution → clearPendingStatuses). Settling here made the UI claim
        // "stopped" seconds early and the still-working node looked frozen (S62b); the button shows
        // "Cancelling…" for the interim instead.
        wf.cancelExecution(segmentId);
        invoke("cancel_separation").catch(() => { });
        // Force-kill any live AMT (audio→MIDI) sidecar too — its Python worker is a
        // separate process that "voice"/"separation" cancels never reach.
        invoke("cancel_amt_all").catch(() => { });
        // Voice invokes (run_rvc/run_sovits) are direct awaits — no polling loop ever re-checks
        // isCancelled mid-run, so the Rust-side flag is the only way to abort a long
        // diffusion/synthesis. Global like the separation cancel.
        invoke("cancel_voice").catch(() => { });
    }, [segmentId]);
    const segmentRef = useRef(segment);
    segmentRef.current = segment;
    // Flush a pending debounced save on unmount so closing fast doesn't drop the last <300ms of edits
    // (the save effect's cleanup only clears the timer — this writes the final graph through).
    useEffect(() => {
        return () => {
            clearTimeout(saveTimer.current);
            const trackId = segTrackIdRef.current;
            if (!trackId)
                return;
            const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
            const track = useProjectStore.getState().tracks.find((tr) => tr.id === trackId);
            if (!track)
                return;
            updateTrackRef.current(trackId, {
                segments: track.segments.map((s) => (s.id === segmentId ? { ...s, workflow: wf } : s)),
            });
        };
    }, [segmentId]);
    // 超级原创向导的「挂载即自动执行」：向导建轨→写模板工作流→openWorkflow + 置 workflowAutoRun
    // 标志；本编辑器挂载时消费标志并直接触发 handleExecute —— 完整复用 preflight（缺失组件弹
    // MissingModelsDialog）、进度、取消、轨道沉积，向导绝不旁路执行引擎。挂载渲染的
    // handleExecute 闭包里的 nodes/edges 正是 segment.workflow 的初始图（同步 useState 初始化），
    // 因此无需等待任何 settle。一次性：消费即清（失败/取消也清——用户手动点 Run 即可重试）。
    useEffect(() => {
        const app = useAppStore.getState();
        if (app.workflowAutoRun !== segmentId)
            return;
        app.clearWorkflowAutoRun();
        void handleExecute();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [segmentId]);
    const handleRunSingleNode = useCallback((nodeId) => {
        const seg = segmentRef.current;
        if (!seg)
            return;
        if (useWorkflowStore.getState().renderLinks[segmentId])
            return; // linked half: its render is the source's
        const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
        // Invalidation is PATH-driven: every run writes into a fresh run dir (engine ensureRunDir), so each
        // node this run actually executes — the clicked node AND any upstream re-run as an uncached/sparse
        // dependency — emits NEW output paths, and the reconciler's path-inequality check re-decodes every
        // deposited lane they feed (including lanes of OTHER Output nodes fed by a re-run upstream, which a
        // clicked-node-only invalidation used to leave stale). Nodes reused from a dense cache keep their
        // old paths → their lanes are untouched.
        void (async () => {
            // Same busy gate as the main Run (double-run guard matters most here: node Run buttons stay
            // clickable while a full run is live — dispatching would clobber the live execution entry).
            if (!(await preflightRun(segmentId, wf, nodeId)))
                return;
            await executeSingleNode(segmentId, seg.seg, wf, nodeId).catch((err) => {
                console.error("Single node execution failed:", err);
            });
        })();
    }, [segmentId]);
    // --- Output AUTO-DEPOSIT reconciler ----------------------------------------
    // Output nodes are LIVE: connecting a rendered edge into one auto-deposits its audio onto the track,
    // disconnecting the edge or deleting the node removes it — there is no manual button. ONE reconciler
    // makes each Output node's track lanes MATCH its current inbound edges, so it stays correct under
    // connect / disconnect / node-delete / a node finishing a render / even Ctrl+Z on an edge (all change
    // the inputs the effect below watches). It only writes processedOutputs (a non-undoable overlay) so it
    // NEVER clears the redo stack. Removal is driven by edge STRUCTURE, not render-cache freshness, so
    // reopening a saved segment (cold cache) never wipes its persisted lanes.
    // DEBOUNCED playback reschedule: rapid lane changes (a multi-branch run depositing one-by-one, or
    // several removals) coalesce into ONE stop+replay instead of restarting playback per change —
    // restarting repeatedly mid-playback is what caused the audible "ghosting"/stutter.
    const rescheduleTimerRef = useRef(undefined);
    const scheduleReschedule = useCallback(() => {
        clearTimeout(rescheduleTimerRef.current);
        rescheduleTimerRef.current = setTimeout(() => {
            if (useAudioStore.getState().isPlaying)
                useAudioStore.getState().bumpSchedule();
        }, 140);
    }, []);
    const reconcilingRef = useRef(new Set());
    const pendingReconcileRef = useRef(new Set());
    const reconcileSigRef = useRef(""); // last (structure+cache+status) sig — skip no-op effect re-fires
    const reconcileOutputNodeRef = useRef(async () => { });
    const reconcileOutputNode = useCallback(async (outputNodeId) => {
        const trackId = segTrackIdRef.current;
        if (!trackId || !segmentRef.current)
            return;
        // One reconcile per node at a time; coalesce a change that lands mid-flight into one re-run after.
        if (reconcilingRef.current.has(outputNodeId)) {
            pendingReconcileRef.current.add(outputNodeId);
            return;
        }
        reconcilingRef.current.add(outputNodeId);
        const setStatus = (s) => useWorkflowStore.getState().setNodeStatus(segmentId, outputNodeId, s);
        const segOutputs = () => useProjectStore.getState().tracks.find((t) => t.id === trackId)
            ?.segments.find((s) => s.id === segmentId)?.processedOutputs ?? [];
        const reschedule = scheduleReschedule; // debounced — coalesces rapid lane changes into one replay
        try {
            const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
            const lanes = outputLanes(wf, outputNodeId); // structural lanes [{laneId, laneLabel, group}]
            const cached = new Map(collectCachedPaths(segmentId, outputNodeId, wf).paths.map((p) => [p.laneId, p]));
            const mine = segOutputs().filter((o) => o.outputNodeId === outputNodeId);
            const deposited = new Map(mine.filter((o) => !o.loading).map((o) => [o.laneId, o]));
            // "Is a render in flight for THIS segment's lanes?" — for a split-mid-render LINKED half the render
            // runs on its SOURCE (it has no execution of its own), so read the source's status via renderLinks.
            // Without this a linked half was always runActive=false → connected/new lanes hit "idle → no lane"
            // (loading never showed / carried placeholders got wiped), which is why the reconciler used to skip
            // linked halves entirely. Reading the source's status lets the reconciler run NORMALLY on a linked
            // half: add placeholders for newly-connected lanes, keep the connected ones, prune the deleted ones.
            const wfNow = useWorkflowStore.getState();
            const activeExec = wfNow.executions[wfNow.renderLinks[segmentId] ?? segmentId];
            const runActive = activeExec?.status === "running";
            // The run's dispatch-time node roster (split halves share node ids with their source, so the
            // source's roster answers for a linked half too). Placeholders key on MEMBERSHIP, not on the bare
            // "some run is active": a lane whose feeder is NOT part of the running job will never be fed by it
            // — showing "loading" for it (the old behavior) promised audio that the settle-prune then silently
            // deleted, leaving a confusing empty lane.
            const participants = new Set(activeExec?.participants ?? []);
            // Build the DESIRED on-track set for this node + the list of lanes needing a (re)decode.
            const target = [];
            const toDecode = [];
            for (const { laneId, laneLabel, group, fromNode } of lanes) {
                const c = cached.get(laneId);
                const dep = deposited.get(laneId);
                if (c && (!dep || dep.audioPath !== c.audioPath)) {
                    toDecode.push(c);
                    target.push({ laneId, laneLabel: c.laneLabel, group: c.group, audioPath: c.audioPath, totalDurationMs: 0, loading: true, outputNodeId });
                }
                else if (dep) {
                    // unchanged, or cold-cache persisted → KEEP (refresh the display label/group if changed — use
                    // the STRUCTURAL label so a rename propagates even with a cold cache where c is undefined)
                    target.push(laneLabel !== dep.laneLabel || group !== dep.group ? { ...dep, laneLabel, group } : dep);
                }
                else if (runActive && participants.has(fromNode)) {
                    // connected and its feeder IS part of the live run but hasn't rendered YET → loading
                    // placeholder so the user sees the lane "coming" (covers connecting a stem mid-run).
                    target.push({ laneId, laneLabel, group, audioPath: `__pending_${laneId}`, totalDurationMs: 0, loading: true, outputNodeId });
                }
                // else: uncached + idle (or fed by a node outside the active run) → no lane
            }
            const anyLoading = target.some((o) => o.loading);
            const sig = (arr) => arr.map((o) => `${o.laneId}|${o.audioPath}|${o.laneLabel}|${o.group ?? ""}|${o.loading ? 1 : 0}`).sort().join(",");
            const changed = sig(target) !== sig(mine);
            if (!changed) {
                const want = anyLoading ? "running" : target.length > 0 ? "completed" : "idle";
                if (useWorkflowStore.getState().nodeStatuses[segmentId]?.[outputNodeId] !== want)
                    setStatus(want);
                return;
            }
            if (target.length === 0) {
                useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outputNodeId);
                setStatus("idle");
                reschedule();
                return;
            }
            // Show the placeholders/keeps IMMEDIATELY (instant feedback), then decode + replace.
            mergeProcessedOutputs(trackId, segmentId, target);
            if (!useProjectStore.getState().tracks.find((t) => t.id === trackId)?.expanded) {
                useProjectStore.getState().toggleTrackExpanded(trackId);
            }
            setStatus(anyLoading ? "running" : "completed");
            reschedule();
            if (toDecode.length > 0) {
                // S59 deposit-perf O2: decode lanes CONCURRENTLY (the old per-lane sequential awaits
                // serialized 4-5 multi-second stem decodes). Promise.all keeps the throw-on-any-failure
                // semantics the catch below depends on (drop placeholders, keep finished lanes).
                const decoded = new Map();
                const results = await Promise.all(toDecode.map((p) => {
                    clearBufferCache(p.audioPath);
                    void loadAudioBuffer(p.audioPath); // warm the playback decode in parallel (promise-deduped)
                    return loadCachedOutput(p);
                }));
                for (let i = 0; i < toDecode.length; i++)
                    decoded.set(toDecode[i].laneId, results[i]);
                // The node may have been deleted while we were decoding — drop its lanes, don't deposit a phantom.
                if (!nodesRef.current.some((n) => n.id === outputNodeId)) {
                    useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outputNodeId);
                    reschedule();
                    return;
                }
                const finalTarget = target.map((o) => decoded.get(o.laneId) ?? o); // decoded replace placeholders; pending stay
                mergeProcessedOutputs(trackId, segmentId, finalTarget);
                setStatus(finalTarget.some((o) => o.loading) ? "running" : "completed");
                reschedule();
            }
        }
        catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            logToBackend("error", `Output auto-deposit failed: ${msg}`);
            setStatus("error");
            useAppStore.getState().showToast(i18n.t("workflow.depositFailed"), "error");
            // Drop this node's still-loading placeholder(s): the change-sig would otherwise rebuild an
            // IDENTICAL placeholder next reconcile and early-return → a lane stuck "loading" forever. Keep its
            // finished (non-loading) lanes; a later trigger can retry the decode.
            const keep = segOutputs().filter((o) => o.outputNodeId === outputNodeId && !o.loading);
            useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, outputNodeId);
            if (keep.length > 0)
                mergeProcessedOutputs(trackId, segmentId, keep);
            reschedule();
        }
        finally {
            reconcilingRef.current.delete(outputNodeId);
            if (pendingReconcileRef.current.has(outputNodeId)) {
                pendingReconcileRef.current.delete(outputNodeId);
                void reconcileOutputNodeRef.current(outputNodeId);
            }
        }
    }, [segmentId, mergeProcessedOutputs, scheduleReschedule]);
    reconcileOutputNodeRef.current = reconcileOutputNode;
    // Reconcile every Output node whenever the graph (nodes/edges), this segment's render cache, or the run
    // state changes — INCLUDING DURING a live run, so each Output lane deposits the moment its branch
    // finishes (and shows a loading placeholder before then). Mutating the project store inside does NOT
    // re-trigger this effect (its deps are graph + workflow-store state), so there is no feedback loop.
    useEffect(() => {
        const trackId = segTrackIdRef.current;
        if (!trackId || !segmentRef.current)
            return;
        const outputIds = nodesRef.current.filter((n) => n.type === "audioOutput").map((n) => n.id);
        const outSet = new Set(outputIds);
        // Idempotency guard: only act when something that affects deposits changed — the Output-node set, the
        // edges INTO them, this segment's render cache, and the run status. Without it the effect re-runs on
        // unrelated re-renders (progress ticks, playback) and the per-frame reconcile floods the log + CPU.
        const wf = useWorkflowStore.getState();
        // A split-mid-render LINKED half reconciles NORMALLY (no early-return): its per-node reconcile reads
        // `runActive` from the SOURCE's execution (see reconcileOutputNode), so during the render it correctly
        // shows loading placeholders for connected/newly-added lanes, keeps them, and prunes deleted ones — all
        // LIVE. It won't wrongly deposit from a cold cache (that path needs cached paths this segment lacks until
        // RenderLinkWatcher clones + headless-deposits on settle). This replaced the old "skip linked entirely"
        // early-return, which made loading never update on a linked half — the regression the user hit.
        const cache = wf.nodeOutputs[segmentId] ?? {};
        const sig = JSON.stringify({
            o: [...outputIds].sort(),
            e: edgesRef.current.filter((e) => outSet.has(e.target)).map((e) => `${e.source}:${e.sourceHandle}>${e.target}:${e.targetHandle}`).sort(),
            c: Object.keys(cache).sort().map((k) => `${k}=${(cache[k] ?? []).join(",")}`),
            s: wf.executions[segmentId]?.status ?? "",
            // The Output nodes' GROUP param: a rename must re-reconcile (the KEEP branch relabels the deposited
            // lanes in place) — without this term the rename only landed on the NEXT unrelated sig change,
            // looking broken then applying as a surprise later. Output nodes only — other nodes' param edits
            // (RVC sliders etc.) must NOT churn the reconciler.
            l: nodesRef.current
                .filter((n) => n.type === "audioOutput")
                .map((n) => `${n.id}=${n.data.params?.laneLabel ?? ""}`)
                .sort(),
            // Upstream stem names feeding the outputs: the structural laneLabel is `group · stem`, and the
            // stem comes from the FEEDER's stemLabels param (changes with a model/preset switch, no run) —
            // without this the relabel is just as late as a group rename used to be.
            t: edgesRef.current
                .filter((e) => outSet.has(e.target))
                .map((e) => {
                const port = parseInt(e.sourceHandle?.replace("out-", "") ?? "0", 10);
                const stems = nodesRef.current.find((n) => n.id === e.source)?.data
                    ?.params?.stemLabels;
                return `${e.source}:${port}=${stems?.[port] ?? ""}`;
            })
                .sort(),
        });
        if (sig === reconcileSigRef.current)
            return;
        reconcileSigRef.current = sig;
        // Orphan cleanup: a deleted Output node is gone from the graph, so the per-node loop won't see it —
        // drop any track lanes whose producing node no longer exists (+ refresh live playback).
        const seg = useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
        for (const o of seg?.processedOutputs ?? []) {
            if (o.outputNodeId && !outSet.has(o.outputNodeId)) {
                useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, o.outputNodeId);
                scheduleReschedule();
            }
        }
        for (const id of outputIds)
            void reconcileOutputNodeRef.current(id);
        // `linkedSource` (renderLinks[segmentId]) IS a dependency: the per-node reconcile derives runActive
        // from the SOURCE while linked, so the effect must re-fire when the link RESOLVES (unlinkRender →
        // linkedSource becomes undefined) — the sig gate alone wouldn't re-run it, and the orphan cleanup /
        // final reconcile for edits made while linked would only land on the next unrelated change.
    }, [nodes, edges, nodeOutputsForSegment, executionState?.status, linkedSource, segmentId, reconcileOutputNode, scheduleReschedule]);
    // --- Output-group DETACH ("ungroup") -----------------------------------------
    // Requested via app-store `pendingLaneDetach` (from THIS editor's node menu, or from a timeline lane
    // right-click that opened this editor first) — ONE code path: split the multi-input Output node into
    // one single-edge Output per inbound edge (stem-named groups), rewriting the deposited lanes IN PLACE
    // (same audio, no re-decode). The graph edit lands in the NODE-GRAPH undo stack (auto-capture); undoing
    // it restores the old node and the reconciler converges the lanes back (laneOps' old key is kept).
    // The store half runs under history.runSilent — laneOps/laneControls are in the timeline meaningfulSig,
    // and a machine bookkeeping write must not push a phantom timeline step / wash the redo stack.
    const pendingDetach = useAppStore((s) => s.pendingLaneDetach);
    useEffect(() => {
        // LIVE read (not the subscribed closure value): under React.StrictMode the mount effect runs twice
        // with the SAME closure — the synchronous clear below makes the second invocation read null and
        // no-op, where a closure check would double-apply the detach (duplicate nodes/edges/lanes,
        // review-caught HIGH). The subscription (`pendingDetach`) exists only to re-fire this effect.
        const pending = useAppStore.getState().pendingLaneDetach;
        if (!pending || pending.segmentId !== segmentId)
            return;
        useAppStore.getState().clearLaneDetach();
        const trackId = segTrackIdRef.current;
        if (!trackId || !segmentRef.current)
            return;
        const wf = reactFlowToWorkflow(nodesRef.current, edgesRef.current);
        const plan = planDetachGroup(wf, pending.outputNodeId);
        if (!plan)
            return;
        nodeCounter++;
        const zBase = nodeCounter + 100;
        const newRfNodes = [
            ...nodesRef.current.filter((n) => n.id !== plan.oldNodeId),
            ...plan.newNodes.map((nn, i) => ({
                id: nn.id,
                type: "audioOutput",
                position: nn.position,
                // `detached: true` = the PERSISTENT independence marker (rides the graph → .usp / split copies;
                // graph-undo of the detach removes it with the node). getLanes exempts detached 组s from the
                // same-name row merging — without it a detached 组 is structurally identical to a re-created
                // Output node and would merge right back onto the sibling pieces' original rows/bar, making
                // 解组's independent volume/pan impossible.
                data: { label: "output", params: { laneLabel: nn.group, detached: true } },
                deletable: true,
                zIndex: zBase + i,
            })),
        ];
        const newRfEdges = [
            ...edgesRef.current.filter((e) => e.target !== plan.oldNodeId),
            ...plan.newNodes.map((nn) => ({
                id: `e-detach-${nn.id}`,
                source: nn.edge.fromNode,
                sourceHandle: `out-${nn.edge.fromPort}`,
                target: nn.id,
                targetHandle: "in-0",
                animated: true,
            })),
        ];
        setNodes(newRfNodes);
        setEdges(newRfEdges);
        // Persist the post-detach graph to segment.workflow SYNCHRONOUSLY (same write the debounced save
        // does, which alone would lag ~300ms): applyLaneDetach rewrites the deposited lanes to the new node
        // ids IMMEDIATELY, and a save/autosave/split landing in the debounce window would otherwise capture
        // lanes referencing nodes that exist in no persisted graph — on reload the orphan cleanup would
        // silently delete every detached (already-rendered) lane. workflow is history-excluded → no undo step.
        {
            const wfAfter = reactFlowToWorkflow(newRfNodes, newRfEdges);
            const track = useProjectStore.getState().tracks.find((tr) => tr.id === trackId);
            if (track) {
                updateTrackRef.current(trackId, {
                    segments: track.segments.map((s) => (s.id === segmentId ? { ...s, workflow: wfAfter } : s)),
                });
            }
        }
        useHistoryStore.getState().runSilent(() => useProjectStore.getState().applyLaneDetach(trackId, segmentId, plan.oldNodeId, plan.mapping));
        // The lane selection (set by the lane right-click that requested this) points at the REMOVED node —
        // remap it onto the first detached group so the gold cue + Ctrl+K/Delete keep targeting a real lane
        // (a stale selection would lazily clear and mis-route Ctrl+K to a whole-segment split).
        const sel = useAppStore.getState().selectedLane;
        if (sel && sel.segmentId === segmentId && sel.outputNodeId === plan.oldNodeId && plan.mapping.length > 0) {
            useAppStore.getState().selectLane(trackId, segmentId, plan.mapping[0].newNodeId, sel.clipIndex);
        }
        scheduleReschedule();
    }, [pendingDetach, segmentId, setNodes, setEdges, scheduleReschedule]);
    // Clicking an Output node selects its GROUP's sub-lanes on the track (the many-to-one bridge): all
    // rows of that group in the open segment light up gold — same selection the lane click sets. Only
    // when the group actually has deposited lanes; expands the track so the cue is visible. activePane
    // stays "workflow" (selectLane doesn't touch it), so Delete keeps deleting NODES, not lane pieces.
    const onNodeClick = useCallback((_e, node) => {
        closeMenus();
        if (node.type !== "audioOutput")
            return;
        const trackId = segTrackIdRef.current;
        if (!trackId)
            return;
        const ps = useProjectStore.getState();
        const track = ps.tracks.find((tr) => tr.id === trackId);
        const seg = track?.segments.find((sg) => sg.id === segmentId);
        if (!track || !seg?.processedOutputs?.some((o) => o.outputNodeId === node.id && !o.loading))
            return;
        if (!track.expanded)
            ps.toggleTrackExpanded(trackId);
        useAppStore.getState().selectLane(trackId, segmentId, node.id, 0);
    }, [closeMenus, segmentId]);
    useEffect(() => {
        useWorkflowStore.getState().registerSingleNodeRunner(handleRunSingleNode);
        useWorkflowStore.getState().registerAnnotationEditor(setEditingAnnotation);
        return () => {
            useWorkflowStore.getState().registerSingleNodeRunner(null);
            useWorkflowStore.getState().registerAnnotationEditor(null);
        };
    }, [handleRunSingleNode]);
    const isRunning = effExec?.status === "running";
    // 规划 6-5 链路 SNR(实时):非 IO、未旁通的节点按损伤表累加 → 整条链保真度估计(旁通切换经
    // setNodes 立即重算)。全透传图(无损耗节点)→ null,不显示。
    const chainSnrDb = useMemo(() => {
        const snrs = [];
        for (const n of nodes) {
            if (n.type === "audioInput" || n.type === "audioOutput" || n.data?.bypass === true)
                continue;
            const wf = rfTypeToWfType[n.type ?? ""];
            if (wf)
                snrs.push(NODE_DAMAGE[wf].snrDb);
        }
        return cumulativeSnrDb(snrs);
    }, [nodes]);
    return (_jsxs("div", { className: "workflow-editor", style: style, onPointerDownCapture: focusEditor, onFocusCapture: focusEditor, children: [_jsxs("div", { className: "workflow-header", children: [_jsxs("div", { className: "workflow-header-left", children: [_jsx("button", { className: "workflow-close", onClick: onClose, children: "x" }), _jsxs("span", { className: "workflow-title", children: [t("workflow.title"), " \u2014 ", segmentId.slice(0, 8)] })] }), _jsxs("div", { className: "workflow-header-actions", children: [_jsxs("button", { className: "wf-run-btn", onClick: () => setShowExampleLibrary(true), title: t("workflow.exampleLibrary.title"), children: ["\uD83D\uDCDA ", t("workflow.exampleLibrary.button")] }), _jsxs("button", { className: "wf-run-btn", onClick: handleAutoLayout, title: t("workflow.autoLayoutTip"), children: ["\u2317 ", t("workflow.autoLayout")] }), canRestoreLayout && (_jsxs("button", { className: "wf-run-btn", onClick: handleRestoreLayout, title: t("workflow.autoLayoutRestoreTip"), children: ["\u21A9 ", t("workflow.autoLayoutRestore")] })), _jsxs("select", { className: "wf-preset-select", value: "", onChange: (e) => { const v = e.target.value; if (v)
                                    void loadPreset(v); }, title: presetNames.length ? t("workflow.loadPreset") : t("workflow.noPresets"), children: [_jsx("option", { value: "", disabled: true, children: presetNames.length ? t("workflow.loadPreset") : t("workflow.noPresets") }), presetNames.map((n) => (_jsx("option", { value: n, children: n }, n)))] }), _jsx("button", { className: "wf-run-btn", onClick: () => void savePreset(), children: t("workflow.savePreset") }), _jsx("button", { className: "wf-run-btn", onClick: () => void exportPresets(), title: t("workflow.exportPresets"), children: t("workflow.exportPresetsShort") }), _jsx("button", { className: "wf-run-btn", onClick: () => void importPresets(), title: t("workflow.importPresets"), children: t("workflow.importPresetsShort") }), chainSnrDb != null && (_jsx("span", { className: "wf-chain-snr", title: t("workflow.chainSnrTip"), children: t("workflow.chainSnr", { snr: Math.round(chainSnrDb) }) })), effExec?.status === "error" && (_jsx("span", { className: "wf-error", children: effExec.error })), renderLocked ? (
                            // Linked half: the source half owns the single shared render — show its state, don't offer a
                            // Run/Stop here (stop it from the source half). Prevents a rejected/duplicate render. Literal
                            // text to match the adjacent Run/Stop buttons (also literal).
                            _jsx("span", { className: "wf-run-btn running", title: "Rendering \u2014 the source half owns this render; stop it there", children: "Rendering\u2026" })) : isRunning && effExec?.cancelled ? (
                            // Cancel acknowledged, backend still winding down (the run settles when its in-flight
                            // invoke returns) — an explicit interim so the node doesn't read as frozen. Literal text
                            // to match the adjacent Run/Stop buttons (also literal).
                            _jsx("span", { className: "wf-run-btn running", title: "Stopping \u2014 waiting for the backend to finish its current step", children: "Cancelling\u2026" })) : isRunning ? (_jsx("button", { className: "wf-run-btn running", onClick: handleCancel, children: "Stop" })) : (_jsx("button", { className: "wf-run-btn", onClick: handleExecute, children: "Run" }))] })] }), _jsxs("div", { className: "workflow-body", children: [paletteOpen && _jsx(NodePalette, { onAddNode: onAddNode, onDropNode: onDropNode }), _jsx("button", { className: `palette-collapse ${paletteOpen ? "open" : ""}`, title: paletteOpen ? paletteCollapseLabel : paletteExpandLabel, onClick: () => setPaletteOpen((v) => !v), children: paletteOpen ? "◀" : "▶" }), _jsxs("div", { className: "workflow-canvas", children: [_jsxs(ReactFlow, { nodes: nodes, edges: displayEdges, onNodesChange: onNodesChange, onEdgesChange: onEdgesChange, onConnect: onConnect, onConnectStart: onConnectStart, onConnectEnd: onConnectEnd, isValidConnection: isValidConnection, onNodeDragStart: onNodeDragStart, onNodeDragStop: onNodeDragStop, onNodeContextMenu: onNodeContextMenu, onEdgeContextMenu: onEdgeContextMenu, onPaneContextMenu: (e) => {
                                    // Right-click the canvas BACKGROUND → add any palette node at the cursor.
                                    e.preventDefault();
                                    setNodeCtx(null);
                                    setEdgeCtx(null);
                                    setPaneCtx({ x: e.clientX, y: e.clientY });
                                }, onBeforeDelete: onBeforeDelete, onPaneClick: closeMenus, onNodeClick: onNodeClick, onMoveStart: closeMenus, onInit: setRfInstance, onSelectionChange: onSelectionChange, nodeTypes: nodeTypes, fitView: true, 
                                // 批量操作: Ctrl/Cmd+点击 累加选择, Shift+拖拽 框选。
                                multiSelectionKeyCode: ["Control", "Meta"], selectionKeyCode: "Shift", 
                                // S66 手感: a dragged wire END snaps to the nearest port within this radius —
                                // the pickup end is covered by the handles' 24px ::after hit zone (CSS).
                                connectionRadius: 24, deleteKeyCode: activePane === "workflow" ? "Delete" : null, defaultEdgeOptions: { animated: true }, snapToGrid: snapToGrid, snapGrid: [20, 20], proOptions: { hideAttribution: true }, children: [_jsx(Background, { variant: BackgroundVariant.Dots, gap: 20, size: 1, color: "rgba(57, 197, 187, 0.1)" }), _jsx(Controls, { showInteractive: false, style: { background: "var(--bg-panel)", borderColor: "var(--border-default)" } }), _jsx(MiniMap, { style: { background: "var(--bg-base)" }, nodeColor: "var(--accent-primary-dim)", maskColor: "rgba(13, 18, 32, 0.7)" })] }), nodeCtx && _jsx(ContextMenu, { x: nodeCtx.x, y: nodeCtx.y, items: nodeCtxItems, onClose: () => setNodeCtx(null) }), edgeCtx && _jsx(ContextMenu, { x: edgeCtx.x, y: edgeCtx.y, items: edgeCtxItems, onClose: () => setEdgeCtx(null) }), paneCtx && _jsx(ContextMenu, { x: paneCtx.x, y: paneCtx.y, items: paneCtxItems, onClose: () => setPaneCtx(null) }), batchCtx && _jsx(ContextMenu, { x: batchCtx.x, y: batchCtx.y, items: batchCtxItems, onClose: () => setBatchCtx(null) }), selectedNodeIds.length > 1 && (_jsx("div", { className: "wf-batch-badge", children: t("workflow.batchSelected", { count: selectedNodeIds.length }) })), connectingFrom && (_jsx(SmartConnectionHelper, { nodes: nodes, sourceNode: connectingFrom.node, sourceHandle: connectingFrom.handle })), errorDisplay && (_jsx(EnhancedErrorDisplay, { error: errorDisplay.error, nodeType: errorDisplay.nodeType, onDismiss: () => setErrorDisplay(null) }))] })] }), showExampleLibrary && (_jsx(ExampleWorkflowLibrary, { onLoad: handleLoadExampleWorkflow, onClose: () => setShowExampleLibrary(false) })), editingAnnotation && (_jsx(NodeAnnotationEditor, { annotation: nodes.find((n) => n.id === editingAnnotation)?.data?.annotation ?? {
                    nodeId: editingAnnotation,
                    content: "",
                    color: "yellow",
                    createdAt: Date.now(),
                    updatedAt: Date.now(),
                }, onSave: (content, color) => handleSaveAnnotation(editingAnnotation, content, color), onDelete: () => handleDeleteAnnotation(editingAnnotation), onClose: () => setEditingAnnotation(null) })), showTutorial && (_jsx(TutorialOverlay, {})), showShortcuts && (_jsx("div", { className: "wf-shortcuts-overlay", onClick: () => setShowShortcuts(false), children: _jsxs("div", { className: "wf-shortcuts-panel", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "wf-shortcuts-header", children: [_jsx("h3", { children: t("workflow.shortcuts") }), _jsx("button", { className: "wf-shortcuts-close", onClick: () => setShowShortcuts(false), children: "\u00D7" })] }), _jsxs("div", { className: "wf-shortcuts-content", children: [_jsxs("div", { className: "wf-shortcuts-section", children: [_jsx("h4", { children: t("workflow.shortcutsGeneral") }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "Z" }), _jsx("span", { children: t("workflow.shortcutUndo") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "Y" }), _jsx("span", { children: t("workflow.shortcutRedo") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "S" }), _jsx("span", { children: t("workflow.shortcutSave") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "E" }), _jsx("span", { children: t("workflow.shortcutExecute") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "L" }), _jsx("span", { children: t("workflow.shortcutAutoLayout") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Esc" }), _jsx("span", { children: t("workflow.shortcutCancel") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "?" }), _jsx("span", { children: t("workflow.shortcutHelp") })] })] }), _jsxs("div", { className: "wf-shortcuts-section", children: [_jsx("h4", { children: t("workflow.shortcutsSelection") }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "A" }), _jsx("span", { children: t("workflow.shortcutSelectAll") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "Click" }), _jsx("span", { children: t("workflow.shortcutMultiSelect") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Shift" }), "+", _jsx("kbd", { children: "Drag" }), _jsx("span", { children: t("workflow.shortcutBoxSelect") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Del" }), _jsx("span", { children: t("workflow.shortcutDelete") })] })] }), _jsxs("div", { className: "wf-shortcuts-section", children: [_jsx("h4", { children: t("workflow.shortcutsBatch") }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "C" }), _jsx("span", { children: t("workflow.shortcutCopy") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "V" }), _jsx("span", { children: t("workflow.shortcutPaste") })] }), _jsxs("div", { className: "wf-shortcut-item", children: [_jsx("kbd", { children: "Ctrl" }), "+", _jsx("kbd", { children: "B" }), _jsx("span", { children: t("workflow.shortcutBypass") })] })] })] })] }) }))] }));
}
