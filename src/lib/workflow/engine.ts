import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
// ── 纯前端音频管线辅助函数 (Web Audio 侧) ──
import type { Workflow, WorkflowNodeType } from "../../types/project";
import { parseWorkflowGraph } from "./graph";
import { NODE_PORTS } from "./ports";
import { useProjectStore } from "../../store/project";
import { useWorkflowStore } from "../../store/workflow";
import { useAmtStore } from "../../store/amt";
import { useAppStore, type MissingModelItem } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { useMsstModelStore } from "../../store/msst-models";
import { useAudioStore } from "../../store/audio";
import { logToBackend } from "../log";
import { backendErrorMessage, isCancelError } from "../backendError";
import { maybeShowErrorModal } from "../errorDisplay";
import { DEFAULT_OUTPUT_GROUP } from "../constants";
import { MSST_CATALOG, MSST_DEFAULT_PRECISION, type MsstArchitecture } from "../models/msst-catalog";
import { RVC_DEFAULTS, SOVITS_DEFAULTS, buildVoiceOptions } from "./voiceDefaults";
import { healVoiceModelPath, healMsstModelPath } from "./modelPathHeal";
import { matchInstrumentWav } from "../amtSource";
import { analyzeChords, type ChordAnalysisNote, type ChordSegment } from "../analysis/chordAnalysis";
import { arrange } from "../arrangement/arranger";
import { generateChordMidi } from "../arrangement/chordMidi";
import { TICKS_PER_BEAT } from "../constants";
import type { ArrangeMood, ArrangeStyle } from "../arrangement/styles";
// P2-14 歌曲制作节点族（规划 11.4）：统一走 runSongTask
import { runSongTask, type SongTaskPayload, type SongTaskResult } from "../song/runSongTask";
import { ACE_TRACK_CLASSES, DEMUCS_STEM_KINDS, type SongTaskId } from "../models/song-tasks";
import { abcToMidi } from "../backendSong";
// P2 符号域原创化节点族（规划 7.1）：确定性 MIDI 变换 / 旋律重构 / 和声重配 / 曲式编辑 / 换气规划。
import {
  humanizeNotes,
  applyVelocityCurve,
  swingQuantizeNotes,
  varyRhythm,
  type VelocityCurveKind,
  type RhythmVariationMode,
} from "../symbol/midiTransforms";
import {
  degreeSwap,
  rhythmRestructure,
  contourMorph,
  motifDevelop,
  melodySimilarity,
  type MotifTechnique,
} from "../symbol/melodyRestructure";
import { reharmonizeSegments, type ReharmStrategy } from "../symbol/reharmonize";
import { editStructure, type StructureOp } from "../symbol/structure";
import { planBreathPoints } from "../symbol/breathPlan";
import { parseChordLabel, isMinorishQuality, PITCH_NAMES, segmentsToChordBlock } from "../symbol/chords";
import { DEFAULT_SYMBOL_SEED } from "../symbol/rng";

export async function audioBufferToWav(buf: AudioBuffer): Promise<ArrayBuffer> {
  const numCh = buf.numberOfChannels;
  const sr = buf.sampleRate;
  const samples = buf.length;
  const bytesPerSample = 2;
  const blockAlign = numCh * bytesPerSample;
  const byteRate = sr * blockAlign;
  const dataSize = samples * blockAlign;
  const bufSize = 44 + dataSize;
  const out = new ArrayBuffer(bufSize);
  const view = new DataView(out);
  let off = 0;
  const writeStr = (s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(off++, s.charCodeAt(i)); };
  writeStr("RIFF"); view.setUint32(off, 36 + dataSize, true); off += 4;
  writeStr("WAVE"); writeStr("fmt ");
  view.setUint32(off, 16, true); off += 4;
  view.setUint16(off, 1, true); off += 2;          // PCM
  view.setUint16(off, numCh, true); off += 2;
  view.setUint32(off, sr, true); off += 4;
  view.setUint32(off, byteRate, true); off += 4;
  view.setUint16(off, blockAlign, true); off += 2;
  view.setUint16(off, bytesPerSample * 8, true); off += 2;
  writeStr("data"); view.setUint32(off, dataSize, true); off += 4;
  const chans: Float32Array[] = [];
  for (let c = 0; c < numCh; c++) chans.push(buf.getChannelData(c));
  for (let s = 0; s < samples; s++) {
    for (let c = 0; c < numCh; c++) {
      const v = Math.max(-1, Math.min(1, (chans[c] ?? new Float32Array(buf.length))[s] ?? 0));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      off += 2;
    }
  }
  return out;
}
export function arrayBufToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i] ?? 0);
  return btoa(bin);
}
// Tauri writeFile (dynamic import, avoid static import breaking browser preview)
export async function writeFile(path: string, data: ArrayBuffer): Promise<void> {
  const fs = await import("@tauri-apps/plugin-fs");
  await fs.writeFile(path, new Uint8Array(data));
}
import type { Segment, ProcessedOutput, Track } from "../../types/project";

import i18n from "../../i18n";

interface AudioFileInfo {
  duration_ms: number;
  peaks: number[];
}

let runSeq = 0;

/** Live voice invokes per segment. A cancelled run_rvc/run_sovits invoke keeps DRAINING
 *  until the Rust pipeline hits its next cancel poll (the cancel flag LATCHES — one click
 *  always takes effect at the next poll — but that poll can sit behind a multi-second
 *  ONNX Run) — starting a new run for the same segment during that window produced two
 *  live runs emitting `voice-progress` for the SAME node (the "possessed" jumping bar) and
 *  a late「已取消」rejection that looked like the NEW run failing. Both run entry points
 *  AWAIT the drain and then start automatically (no manual retry); a second click while
 *  one is already queued is dropped. Keyed per segment so other segments are unaffected. */
const voiceInvokesInFlight = new Map<string, number>();
const voiceDrainWaiters = new Set<string>();

/** Wait for the segment's draining voice invoke(s) to settle, then proceed. Returns false
 *  when this attempt should be dropped (a run is already queued, or the drain timed out). */
async function waitVoiceDrain(segmentId: string): Promise<boolean> {
  if ((voiceInvokesInFlight.get(segmentId) ?? 0) === 0) return true;
  const toast = useAppStore.getState().showToast;
  if (voiceDrainWaiters.has(segmentId)) {
    toast(i18n.t("workflow.drainQueued"), "info");
    return false;
  }
  voiceDrainWaiters.add(segmentId);
  toast(i18n.t("workflow.drainWaiting"), "info");
  try {
    // Generous cap: a CPU-mode extractor pass over a 30 s piece is the longest single
    // uninterruptible step. A hang past this is a real bug, not a slow drain.
    const deadline = Date.now() + 120_000;
    while ((voiceInvokesInFlight.get(segmentId) ?? 0) > 0) {
      if (Date.now() > deadline) {
        toast(i18n.t("workflow.drainTimeout"), "error");
        return false;
      }
      await new Promise((r) => setTimeout(r, 200));
    }
    return true;
  } finally {
    voiceDrainWaiters.delete(segmentId);
  }
}

/** Cancel sentinel — delegated to THE single check in backendError.ts (shared with every toast
 *  funnel). Runs BEFORE any error-code localization, or a user cancel would surface as a red error. */
function isCancelMessage(msg: string): boolean {
  return isCancelError(msg);
}

/** PRE-FLIGHT separation-busy gate. The Rust SeparationManager is a GLOBAL single slot — dispatching a
 *  run whose separation node would hit its "already in progress" guard used to START the run anyway and
 *  fail it seconds later with a red error, flipping this segment's button back to Run while the OTHER
 *  (earlier) backend job kept going — the UI read as "backend stopped" when it hadn't. So: if the run
 *  would actually EXECUTE a separation node (for a single-node run, dense-cached upstreams are reused
 *  and never invoke the backend) and a live separation is in flight, REJECT before startExecution with
 *  a toast — no execution state is ever created, nothing to un-wind. The Rust guard stays as the
 *  authoritative backstop for the query→dispatch race (its SEPARATION_BUSY code maps to the same text
 *  in executeNode). */
async function rejectIfSeparationBusy(
  segmentId: string,
  workflow: Workflow,
  targetNodeId: string | null,
): Promise<boolean> {
  let needsSeparation = false;
  try {
    const graph = parseWorkflowGraph(workflow);
    const cache = useWorkflowStore.getState().nodeOutputs[segmentId] ?? {};
    // Single-node runs only ever execute the target's ANCESTOR chain (see ancestorSetOf).
    const scope = targetNodeId !== null ? ancestorSetOf(graph, targetNodeId) : null;
    for (const nodeId of graph.sorted) {
      if (scope && !scope.has(nodeId)) continue;
      const gn = graph.nodes.get(nodeId)!;
      if (gn.node.nodeType === "msstSeparation") {
        // Mirrors executeSingleNode's reuse rule: a dense-cached non-target node is skipped, never run.
        const cached = cache[nodeId];
        const reused = targetNodeId !== null && nodeId !== targetNodeId
          && !!cached && cached.length > 0 && isDenseCache(cached);
        if (!reused) { needsSeparation = true; break; }
      }
      if (targetNodeId !== null && nodeId === targetNodeId) break;
    }
  } catch {
    return false; // broken graph — let the normal run path surface its own error
  }
  if (!needsSeparation) return false;
  const status = await invoke<{ state: string | Record<string, string> }>("get_separation_status")
    .catch(() => null);
  const busy = status !== null && typeof status.state === "string"
    && (status.state === "Separating" || status.state === "LoadingModel");
  if (busy) useAppStore.getState().showToast(i18n.t("workflow.separationBusy"), "error");
  return busy;
}

/** S66 — pre-run model availability scan (the "don't make users guess" rule): every model a run
 *  参与节点 references is checked BEFORE dispatch, and problems surface as ONE dialog with per-item
 *  one-click actions instead of a mid-run MSST_MODEL_NOT_CONVERTED / AUX_FILE_MISSING error toast.
 *  Scope follows the run's real execution domain (Run All = whole graph, single node = its
 *  ancestor set — the S62b rule). Best-effort: an IPC failure never blocks the run (the Rust
 *  pipeline still errors loudly). */
export async function collectMissingModels(
  workflow: Workflow,
  targetNodeId: string | null,
): Promise<MissingModelItem[]> {
  let scope: Set<string> | null = null;
  if (targetNodeId !== null) {
    try {
      scope = ancestorSetOf(parseWorkflowGraph(workflow), targetNodeId);
    } catch {
      scope = null; // unparseable graph → scan everything; the run itself will report the parse error
    }
  }
  const nodes = workflow.nodes.filter((n) => scope === null || scope.has(n.id));
  const items: MissingModelItem[] = [];
  const seen = new Set<string>();

  const msstNodes = nodes.filter((n) => n.nodeType === "msstSeparation");
  if (msstNodes.length > 0) {
    let installed: Array<{ filename: string; architecture: string; has_onnx: boolean; has_fp16: boolean }> = [];
    try {
      // straight from Rust — the store copy may never have been fetched this session
      installed = await invoke("list_msst_models");
    } catch {
      return items; // can't scan → don't block
    }
    for (const n of msstNodes) {
      const modelFile = (n.params.modelFile as string) ?? "";
      if (!modelFile || seen.has(modelFile)) continue;
      seen.add(modelFile);
      const entry = installed.find((m) => m.filename === modelFile);
      if (!entry) {
        items.push({ kind: "msstMissing", label: modelFile });
        continue;
      }
      if (!entry.has_onnx && !entry.has_fp16) {
        // mirror the executeNode effective-precision derivation (catalog arch wins over detection;
        // Rust's "unknown" detection verdict is not a usable hint)
        const detected =
          entry.architecture !== "unknown" ? (entry.architecture as MsstArchitecture) : undefined;
        const arch = MSST_CATALOG.find((e) => e.filename === modelFile)?.architecture ?? detected;
        const precision =
          (n.params.precision as "fp32" | "fp16" | undefined) ??
          (arch !== undefined ? MSST_DEFAULT_PRECISION[arch] : undefined);
        items.push({
          kind: "msstConvert",
          label: modelFile,
          filename: modelFile,
          precision,
          architecture: arch,
        });
      }
    }
  }

  if (nodes.some((n) => n.nodeType === "rvc" || n.nodeType === "sovits")) {
    try {
      const packs = await invoke<Array<{ id: string; missing: number; downloading: boolean }>>(
        "asset_pack_status",
      );
      const aux = packs.find((p) => p.id === "aux-inference");
      if ((aux?.missing ?? 0) > 0 && !(aux?.downloading ?? false)) {
        items.push({ kind: "auxPack", label: "aux-inference" });
      }
    } catch {
      /* best-effort */
    }
  }
  return items;
}

/** Localize a `parseWorkflowGraph` throw. These are FRONTEND sentinels (English, thrown by graph.ts), so
 *  they never appear in `backendErrorMessage`'s Rust CODE map and used to reach the user as raw English —
 *  including the "contains a cycle" text a dangling edge could trigger on a graph with no visible cycle. */
function graphErrorMessage(msg: string): string {
  if (msg.includes("more than one input node")) return i18n.t("workflow.errGraphMultiInput");
  if (msg.includes("no input node")) return i18n.t("workflow.errGraphNoInput");
  if (msg.includes("no output nodes")) return i18n.t("workflow.errGraphNoOutput");
  if (msg.includes("cycle")) return i18n.t("workflow.errGraphCycle");
  return i18n.t("workflow.errGraphInvalid");
}

/** Gate a run BEFORE the caller mutates anything (deposit invalidation, store state). The two run entry
 *  points (WorkflowEditor handleExecute / handleRunSingleNode) MUST await this FIRST — previously the
 *  busy/drain checks lived inside executeWorkflow, i.e. AFTER handleExecute had already stripped the
 *  segment's deposited lanes, so a rejected run cost the track its lanes for nothing (and the drain-drop
 *  path returned 0, stacking a misleading "no outputs" error toast on top). Returns false (after
 *  toasting) when the run must not start; nothing has been touched. */
export async function preflightRun(
  segmentId: string,
  workflow: Workflow,
  targetNodeId: string | null,
): Promise<boolean> {
  const running = () => useWorkflowStore.getState().executions[segmentId]?.status === "running";
  // Same-segment double-run guard: the per-node Run button stays reachable during a live run (the main
  // Run button flips to Stop, but node buttons don't) — dispatching would clobber the live execution
  // entry and orphan its UI state.
  if (running()) {
    useAppStore.getState().showToast(i18n.t("workflow.runBusy"), "info");
    return false;
  }
  // Graph legality BEFORE anything is touched. executeWorkflow parses twice: the first parse (participant
  // roster) swallows the throw, then startExecution + clearNodeOutputs run, and only the second parse
  // reports the failure — so an unrunnable graph (cycle / dangling edge / missing IO node) cost the
  // segment every deposited lane before erroring out, exactly the "rejected run cost the track its lanes
  // for nothing" hazard this gate exists to prevent. Rejecting here keeps the lanes intact.
  try {
    parseWorkflowGraph(workflow);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    useAppStore.getState().showToast(graphErrorMessage(msg), "error");
    logToBackend("warn", `Workflow run rejected — invalid graph: ${msg}`);
    return false;
  }
  // S66: unconverted/missing models → the one-click dialog instead of a mid-run error. Read-only
  // scan, so it rides before the drain; the running() rechecks below still cover its awaits.
  const missing = await collectMissingModels(workflow, targetNodeId);
  if (missing.length > 0) {
    useAppStore.getState().openMissingModels(missing);
    return false;
  }
  if (running()) { // a run may have started while the scan's IPC was in flight
    useAppStore.getState().showToast(i18n.t("workflow.runBusy"), "info");
    return false;
  }
  if (!(await waitVoiceDrain(segmentId))) return false;
  if (running()) { // a run may have started while we drained
    useAppStore.getState().showToast(i18n.t("workflow.runBusy"), "info");
    return false;
  }
  if (await rejectIfSeparationBusy(segmentId, workflow, targetNodeId)) return false;
  if (running()) { // …or while we queried the separation status (the path to startExecution is sync from here)
    useAppStore.getState().showToast(i18n.t("workflow.runBusy"), "info");
    return false;
  }
  return true;
}

/** Per-RUN output directory under the segment's cache dir. Node output paths were previously
 *  deterministic (`${cacheDir}/${nodeId}_rvc.wav`, MSST stems by label), which ALIASED across a split:
 *  both halves' deposited lanes reference the ORIGINAL segment's files, so re-running one half silently
 *  overwrote the other half's audio (and waveform) in place — and a re-run at the SAME path could never
 *  be told apart from the old run, so the reconciler's KEEP branch retained stale deposits after a
 *  dependency re-run. A fresh dir per run makes every output path unique: existing deposits keep playing
 *  their own files untouched, and a path CHANGE is itself the re-render signal (placeholder → fresh
 *  decode). Old run dirs are pruned by the startup cache sweep (age/byte budget). */
async function ensureRunDir(segmentId: string): Promise<string> {
  const raw = await invoke<string>("ensure_cache_dir", {
    segmentId: `${segmentId}/r${Date.now().toString(36)}${(runSeq++).toString(36)}`,
  });
  return raw.replace(/\\/g, "/");
}

/** 规划 6-3 旁通透传：bypass 节点不执行，把上游输入原样落到输出端口 ——
 *  第 i 个出口拿第 min(i, 最后一个已连入口) 的值（1→1 直通、split 双路同源、merge 取首路），
 *  下游照常拿到数据、链路不断。返回的 map 同时写进 dataMap 与 nodeOutputs。 */
function bypassPassThrough(
  gn: { inEdges: Array<{ fromNode: string; fromPort: number; toPort: number }> },
  nodeType: WorkflowNodeType,
  dataMap: Map<string, Map<number, string>>,
): Map<number, string> {
  const ins: { toPort: number; path: string }[] = [];
  for (const e of gn.inEdges) {
    const v = dataMap.get(e.fromNode)?.get(e.fromPort);
    if (v) ins.push({ toPort: e.toPort, path: v });
  }
  ins.sort((a, b) => a.toPort - b.toPort);
  const pass = new Map<number, string>();
  if (ins.length > 0) {
    const outs = NODE_PORTS[nodeType]?.outputs ?? [];
    const outCount = outs.length || 1;
    for (let i = 0; i < outCount; i++) {
      // Only fill ports whose declared kind can actually carry the upstream payload. Filling
      // EVERY port meant a bypassed analysis node pushed its audio into its REPORT port (a
      // report consumer then parsed a WAV path as JSON), and a bypassed msstSeparation aimed
      // all 5 stem ports at the unseparated mix — 5 identical "stems" deposited as lanes.
      // A report/chords port has no honest passthrough value, so leave it empty.
      const kind = outs[i];
      if (kind === "report" || kind === "chords") continue;
      pass.set(i, ins[Math.min(i, ins.length - 1)]!.path);
    }
  }
  return pass;
}

/** Returns the number of lanes that reached Output nodes (0 = nothing landed — the caller
 *  toasts). The actual track deposit is done by the live reconciler / RenderLinkWatcher. */
export async function executeWorkflow(
  segmentId: string,
  segment: Segment,
  workflow: Workflow,
): Promise<number> {
  const store = useWorkflowStore.getState();
  // Dispatch-time participant snapshot (every non-IO node — a full run executes them all): written in
  // the SAME store update that flips the run to "running", so the reconciler's very first pass already
  // knows which feeders belong to this run (its pending placeholders key on membership). A parse failure
  // lands [] here and throws properly inside the try below.
  let participants: string[] = [];
  try {
    const g = parseWorkflowGraph(workflow);
    participants = g.sorted.filter((id) => {
      const t = g.nodes.get(id)!.node.nodeType;
      return t !== "input" && t !== "output";
    });
  } catch { /* reported by the parse inside the try below */ }

  try {
    store.startExecution(segmentId, participants);
    store.clearNodeStatuses(segmentId);
    // A full run recomputes every node. Drop any warm/rehydrated cache first so the live reconciler shows
    // loading placeholders and deposits each lane FRESH as its node finishes — never an early decode of a
    // deterministic path this run is about to overwrite in place (the crash-recovery "keeps old stem" hazard).
    store.clearNodeOutputs(segmentId);
    logToBackend("info", `Workflow started (${workflow.nodes.length} nodes)`);
    const graph = parseWorkflowGraph(workflow);

    // Mark all non-IO nodes as waiting BEFORE the first await: the reconciler's pending placeholders
    // key on per-feeder participation (waiting/running), so the marks must land in the same tick the
    // run starts — marking them after fetchInstalled/ensureRunDir left an await-sized window in which
    // connected lanes showed no placeholder at all.
    for (const nodeId of graph.sorted) {
      const gn = graph.nodes.get(nodeId)!;
      if (gn.node.nodeType !== "input" && gn.node.nodeType !== "output") {
        // 旁通节点预标 "bypassed"（规划 6-3）：徽标直达，不留等待假象。
        store.setNodeStatus(segmentId, nodeId, gn.node.bypass ? "bypassed" : "waiting");
      }
    }

    await useMsstModelStore.getState().fetchInstalled();
    const cacheDir = await ensureRunDir(segmentId);

    const dataMap = new Map<string, Map<number, string>>();

    if (segment.content.type !== "audioClip") {
      throw new Error("Workflow execution requires an audioClip segment");
    }
    const inputData = new Map<number, string>();
    // Separate the SAME audio the original segment PLAYS — the content-addressed cache WAV, whose codec
    // pre-skip silence was TRIMMED by load_audio_file. Feeding the raw source instead produced an
    // UN-trimmed stem that played + drew shifted by ~the trim length (a full beat) vs the main track.
    // Fall back to the raw path if the clip wasn't decoded through the cache yet.
    const playbackWav = useAudioStore.getState().audioFiles[segment.content.sourcePath]?.playbackPath;
    inputData.set(0, playbackWav || segment.content.sourcePath);
    dataMap.set(graph.inputNodeId, inputData);

    const totalNodes = graph.sorted.length;

    for (let step = 0; step < totalNodes; step++) {
      const nodeId = graph.sorted[step]!;
      const gn = graph.nodes.get(nodeId)!;
      const nodeType = gn.node.nodeType;
      const params = gn.node.params as Record<string, unknown>;

      store.updateProgress(segmentId, nodeId, step / totalNodes);

      if (nodeType === "input" || nodeType === "output") continue;

      // 规划 6-3 旁通（A/B 冻结）：不执行、不耗算力，输入原样透传到输出端口后跳过。
      // 旁通节点对整链零损伤（没跑模型），SNR 统计（编辑器实时口径）也按跳过处理。
      if (gn.node.bypass) {
        const pass = bypassPassThrough(gn, nodeType, dataMap);
        dataMap.set(nodeId, pass);
        if (pass.size > 0) {
          useWorkflowStore.getState().setNodeOutputs(segmentId, nodeId, Array.from(pass.values()));
        }
        store.setNodeStatus(segmentId, nodeId, "bypassed");
        continue;
      }

      if (useWorkflowStore.getState().isCancelled(segmentId)) {
        throw new Error("Cancelled");
      }

      store.setNodeStatus(segmentId, nodeId, "running");

      const outputData = await executeNode(nodeId, nodeType, params, gn, dataMap, cacheDir, segmentId);

      dataMap.set(nodeId, outputData);
      if (outputData.size > 0) {
        useWorkflowStore.getState().setNodeOutputs(segmentId, nodeId, Array.from(outputData.values()));
      }
      // executeNode 内部的兜底分支（如 speedShift/deepOriginal 失败透传）会把节点标成 "degraded"。
      // 此时不要覆盖成 "completed"——让用户一眼看到这一步实际没生效。
      const curStatus = useWorkflowStore.getState().nodeStatuses[segmentId]?.[nodeId];
      if (curStatus !== "degraded" && curStatus !== "error") {
        store.setNodeStatus(segmentId, nodeId, "completed");
      }
    }

    const laneCount = countOutputLanes(graph, dataMap);

    store.completeExecution(segmentId);
    if (graph.outputNodeIds.length > 0 && laneCount === 0) {
      // Output nodes exist but nothing reached the track — warn loudly instead of a clean "completed".
      logToBackend("warn", "Workflow completed but produced 0 outputs — output node has no connected/rendered upstream");
    } else {
      logToBackend("info", `Workflow completed (${laneCount} outputs)`);
    }
    return laneCount;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const cancelled = isCancelMessage(msg);
    logToBackend(cancelled ? "warn" : "error", cancelled ? "Workflow cancelled" : `Workflow failed: ${msg}`);
    // THE single localization point for node/run error DISPLAY (cancel checked first — a localized
    // cancel would dodge the swallow checks downstream): known Rust CODEs (APP_BUSY, SEPARATION_BUSY,
    // TRANSPOSE_*, MSST_MODEL_NOT_CONVERTED, …) become t(...) text; unknown messages pass through raw.
    // A cancel settles as the bare frontend sentinel (not the raw "Inference error: CANCELLED" wire text).
    const display = cancelled ? "Cancelled" : (backendErrorMessage(msg) ?? msg);
    const store = useWorkflowStore.getState();
    // A real failure marks the offending node red; a user cancel marks nothing. Either way clear the
    // running/waiting badges so nodes don't stay stuck blue/yellow after the run settles.
    if (!cancelled && store.executions[segmentId]?.currentNodeId) {
      store.setNodeStatus(segmentId, store.executions[segmentId]!.currentNodeId!, "error");
      store.setNodeError(segmentId, store.executions[segmentId]!.currentNodeId!, display);
    }
    // S67c: fatal modal-class errors (INFERENCE_LOW_MEMORY …) additionally open the alert
    // dialog — the node tooltip is invisible until hovered and can't carry the guidance text.
    if (!cancelled) maybeShowErrorModal(msg, display);
    store.clearPendingStatuses(segmentId);
    store.failExecution(segmentId, display);
    throw err;
  }
}

/** True iff every index of `arr` holds a value (no holes / no null). A live run always writes a DENSE
 *  output array (Array.from(map.values())); rehydrateRenderState may write a SPARSE one (only the deposited
 *  ports), which must NOT be reused as a complete node output. `.every` can't detect holes (it skips them),
 *  so scan by index. */
function isDenseCache(arr: string[]): boolean {
  for (let i = 0; i < arr.length; i++) if (arr[i] == null) return false;
  return true;
}

/** The target node + its transitive UPSTREAM — the only nodes a single-node run may touch. A plain
 *  walk of graph.sorted "up to the target" also visits UNRELATED parallel branches that happen to sort
 *  earlier (topological order ≠ ancestry), so clicking "run this node" used to silently re-render
 *  never-rendered nodes elsewhere on the canvas (old bug, user-caught S62b). */
function ancestorSetOf(
  graph: ReturnType<typeof parseWorkflowGraph>,
  targetNodeId: string,
): Set<string> {
  const anc = new Set<string>([targetNodeId]);
  const stack = [targetNodeId];
  while (stack.length > 0) {
    const gn = graph.nodes.get(stack.pop()!);
    for (const e of gn?.inEdges ?? []) {
      if (!anc.has(e.fromNode)) {
        anc.add(e.fromNode);
        stack.push(e.fromNode);
      }
    }
  }
  return anc;
}

export async function executeSingleNode(
  segmentId: string,
  segment: Segment,
  workflow: Workflow,
  targetNodeId: string,
): Promise<void> {
  const store = useWorkflowStore.getState();
  // NOTE: we deliberately DON'T clear the target's cache here. The stale-in-place-overwrite hazard is
  // handled AFTER a successful run by handleRunSingleNode (clearBufferCache + removeProcessedOutputsForNode
  // for lanes this node feeds → the reconciler re-decodes fresh); and during the run the old deposit stays
  // present so the reconciler KEEPs it (no early decode of a to-be-overwritten file). Clearing up front
  // instead LOST the last-good cache pointer if the re-run FAILED, breaking reconnect-from-cache.
  // Participant snapshot = the target + its ANCESTOR chain, non-IO (cache-reused upstreams included —
  // harmless: their lanes resolve from the cache branch before the pending branch is consulted).
  // NOT "everything up to the target in topo order": that includes unrelated parallel branches.
  const participants: string[] = [];
  try {
    const g = parseWorkflowGraph(workflow);
    const scope = ancestorSetOf(g, targetNodeId);
    for (const id of g.sorted) {
      if (!scope.has(id)) continue;
      const t = g.nodes.get(id)!.node.nodeType;
      if (t !== "input" && t !== "output") participants.push(id);
      if (id === targetNodeId) break;
    }
  } catch { /* reported by the parse inside the try below */ }

  // Which node is actually executing — the catch below used to blame `targetNodeId` unconditionally, but
  // this loop runs the whole ancestor chain, so an upstream failure (e.g. `has no input connected`) painted
  // the target red and attached the upstream's message to it. executeWorkflow gets this right via
  // executions[].currentNodeId; the single-node path never wrote that field, hence a local cursor.
  let activeNodeId = targetNodeId;

  try {
    store.startExecution(segmentId, participants);
    // Mirror executeWorkflow: drop the previous run's badges. clearPendingStatuses (the settle path)
    // deliberately KEEPS error/degraded, so without this a stale red/amber badge from an earlier attempt
    // rode along into the new run and looked like this run had already failed.
    store.clearNodeStatuses(segmentId);
    const graph = parseWorkflowGraph(workflow);
    // Run-unique dir here too: a single-node re-run only writes the nodes it actually EXECUTES (cached
    // upstreams keep their old-run paths in dataMap), so re-executed outputs land at fresh paths and the
    // reconciler re-deposits every lane they feed — including lanes of OTHER Output nodes fed by an
    // upstream that re-ran as an uncached dependency (previously stale: same path, KEEP branch held it).
    const cacheDir = await ensureRunDir(segmentId);

    if (segment.content.type !== "audioClip") {
      throw new Error("Workflow execution requires an audioClip segment");
    }

    const dataMap = new Map<string, Map<number, string>>();
    const inputData = new Map<number, string>();
    // Separate the SAME audio the original segment PLAYS — the content-addressed cache WAV, whose codec
    // pre-skip silence was TRIMMED by load_audio_file. Feeding the raw source instead produced an
    // UN-trimmed stem that played + drew shifted by ~the trim length (a full beat) vs the main track.
    // Fall back to the raw path if the clip wasn't decoded through the cache yet.
    const playbackWav = useAudioStore.getState().audioFiles[segment.content.sourcePath]?.playbackPath;
    inputData.set(0, playbackWav || segment.content.sourcePath);
    dataMap.set(graph.inputNodeId, inputData);

    // Only the target's ANCESTOR chain may run. graph.sorted is a WHOLE-graph topological order, so
    // "walk until the target" also visits unrelated parallel branches that happen to sort earlier —
    // clicking "run this node" used to silently render never-rendered nodes elsewhere on the canvas.
    const scope = ancestorSetOf(graph, targetNodeId);

    for (const nodeId of graph.sorted) {
      if (!scope.has(nodeId)) continue;
      const gn = graph.nodes.get(nodeId)!;
      if (gn.node.nodeType === "input" || gn.node.nodeType === "output") continue;

      // 规划 6-3 旁通：单节点链路里的 bypass 节点同样不执行 —— 置于缓存复用分支之前，
      // 防止它 bypass 前遗留的旧缓存被当作旁通输出继续下沉。
      if (gn.node.bypass) {
        const pass = bypassPassThrough(gn, gn.node.nodeType, dataMap);
        dataMap.set(nodeId, pass);
        if (pass.size > 0) {
          store.setNodeOutputs(segmentId, nodeId, Array.from(pass.values()));
        }
        store.setNodeStatus(segmentId, nodeId, "bypassed");
        if (nodeId === targetNodeId) break;
        continue;
      }

      if (useWorkflowStore.getState().isCancelled(segmentId)) {
        throw new Error("Cancelled");
      }

      // Reuse a node's cached output ONLY if it's DENSE (every port present). rehydrateRenderState may warm
      // a multi-output node with just the DEPOSITED ports (a sparse array with holes); reusing that would
      // feed `undefined` to a downstream node reading a non-deposited port ("has no input connected"). A
      // hole means that port isn't cached → fall through and RE-RUN the node to regenerate all ports.
      // Read the LIVE store, not the run-start snapshot: zustand replaces the object on every set, so the
      // outputs this very loop wrote via setNodeOutputs were invisible through `store` and a node could be
      // re-executed in the same run it had just produced. (The degraded check below already re-reads for
      // the same reason.)
      const cached = useWorkflowStore.getState().nodeOutputs[segmentId]?.[nodeId];
      if (cached && cached.length > 0 && nodeId !== targetNodeId && isDenseCache(cached)) {
        const m = new Map<number, string>();
        cached.forEach((p, i) => m.set(i, p));
        dataMap.set(nodeId, m);
        continue;
      }

      store.setNodeStatus(segmentId, nodeId, "running");
      activeNodeId = nodeId;

      const outputData = await executeNode(
        nodeId, gn.node.nodeType, gn.node.params as Record<string, unknown>,
        gn, dataMap, cacheDir, segmentId,
      );

      dataMap.set(nodeId, outputData);
      if (outputData.size > 0) {
        store.setNodeOutputs(segmentId, nodeId, Array.from(outputData.values()));
      }
      // 与 executeWorkflow 同一守卫：executeNode 的兜底透传会把节点标成 "degraded"，
      // 不要覆盖成 "completed"（单节点路径此前无条件覆盖，损伤态一闪即逝）。
      const curStatus = useWorkflowStore.getState().nodeStatuses[segmentId]?.[nodeId];
      if (curStatus !== "degraded" && curStatus !== "error") {
        store.setNodeStatus(segmentId, nodeId, "completed");
      }

      if (nodeId === targetNodeId) break;
    }

    store.completeExecution(segmentId);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const cancelled = isCancelMessage(msg);
    // S67c: single-node failures now reach the backend log too (they used to be
    // tooltip-only — invisible in crash forensics), mirroring executeWorkflow's catch.
    logToBackend(cancelled ? "warn" : "error", cancelled ? "Single-node run cancelled" : `Single-node run failed: ${msg}`);
    // Same single localization point as executeWorkflow's catch (cancel checked first).
    const display = cancelled ? "Cancelled" : (backendErrorMessage(msg) ?? msg);
    if (!cancelled) {
      store.setNodeStatus(segmentId, activeNodeId, "error");
      store.setNodeError(segmentId, activeNodeId, display);
      maybeShowErrorModal(msg, display);
    }
    store.clearPendingStatuses(segmentId);
    store.failExecution(segmentId, display);
  }
}

// ── P2-14 歌曲节点辅助（规划 11.3/11.4）───────────────────────────────────

/** 11.3 约定：歌词/提示词源节点输出 `lyrics://` 前缀的虚拟文本；非前缀 = 文件路径（忽略）。 */
function songPortText(v: string | undefined): string | undefined {
  return v?.startsWith("lyrics://") ? v.slice("lyrics://".length) : undefined;
}

/** 歌曲节点统一执行壳：进度接线到节点进度条；歌曲任务无取消命令，完成后检查
 *  isCancelled 再 throw "Cancelled"，避免产物沉积到已取消的执行。 */
async function runSongNode(
  task: SongTaskId,
  payload: SongTaskPayload,
  nodeId: string,
  segmentId: string,
): Promise<SongTaskResult> {
  const result = await runSongTask(task, payload, {
    onProgress: (p) => useWorkflowStore.getState().setNodeProgress(segmentId, nodeId, p.percent / 100),
  });
  if (useWorkflowStore.getState().isCancelled(segmentId)) throw new Error("Cancelled");
  return result;
}

// ── P2 符号域节点族共享解析辅助 ──────────────────────────────────────────

/** 枚举 select 参数的安全取值：组件存 0..n-1 的索引，越界/缺省钳到 fallback。 */
function pickEnum<T extends string>(
  params: Record<string, unknown>,
  key: string,
  values: readonly T[],
  fallbackIdx: number,
): T {
  const idx = Math.max(0, Math.min(values.length - 1, Math.round((params[key] as number) ?? fallbackIdx)));
  return values[idx]!;
}

/** 符号域节点族的统一 MIDI 输入解析：三种形态 —
 *  ① JSON 音符数组（`[...]`，harmonizer/melodyGen 的输出形态）
 *  ② JSON 包裹对象（`{notes:[...], ppq?}`）
 *  ③ .mid/.midi 文件路径（走 import_score_file）
 *  返回 ChordAnalysisNote[]；解析不出音符时在源头响亮失败（不静默透传）。 */
async function resolveMidiNotes(
  input: string,
  what: string,
): Promise<{ notes: ChordAnalysisNote[]; ppq: number }> {
  const trimmed = input.trim();
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    try {
      const parsed = JSON.parse(trimmed);
      const rawNotes = Array.isArray(parsed) ? parsed : parsed?.notes;
      if (Array.isArray(rawNotes)) {
        const notes = rawNotes.filter(
          (n: any) => n && typeof n.pitch === "number" && typeof n.tick === "number",
        );
        if (notes.length > 0) return { notes, ppq: typeof parsed?.ppq === "number" ? parsed.ppq : 480 };
      }
    } catch { /* fall through → 当作文件路径 */ }
  }
  const ext = (input.split(".").pop() ?? "").toLowerCase();
  if (ext === "mid" || ext === "midi") {
    const res = await invoke<any>("import_score_file", { path: input });
    const notes = (res?.tracks ?? []).flatMap((t: any) =>
      (t.notes ?? []).map((n: any) => ({
        tick: n.tick, duration: n.duration, pitch: n.pitch, velocity: n.velocity ?? 100,
      })),
    );
    if (notes.length > 0) return { notes, ppq: res?.ppq ?? 480 };
  }
  throw new Error(`${what}: 需要 MIDI 输入（.mid/.midi 文件、音符数组 JSON 或 {notes:[...]}）`);
}

/** 符号域和弦节点族（reharmonize/structureEdit/melodyGen）的统一和弦输入解析：
 *  接受 chordBlockIn 的 `{type:"chordBlock", chords:[{label,root}], bpm}` 或
 *  chordDetect 的 `{segments:[{startTick,endTick,label}]}`，label 一律经
 *  parseChordLabel 补全 quality/bass，统一成 ChordSegment[]（缺时间戳时按每和弦一小节铺）。 */
function resolveChordSegments(input: string, what: string): { segments: ChordSegment[]; bpm?: number } {
  try {
    const parsed = JSON.parse(input);
    const raw = parsed.type === "chordBlock" ? parsed.chords : parsed.segments;
    if (!Array.isArray(raw) || raw.length === 0) throw new Error("no chords");
    const barTicks = 4 * TICKS_PER_BEAT;
    const segments = raw.flatMap((c: any, i: number): ChordSegment[] => {
      if (typeof c?.label !== "string" || !c.label.trim()) return [];
      const p = parseChordLabel(c.label);
      return [{
        startTick: typeof c.startTick === "number" ? c.startTick : i * barTicks,
        endTick: typeof c.endTick === "number" ? c.endTick : (i + 1) * barTicks,
        root: p.root,
        quality: p.quality,
        bass: p.bass,
        label: c.label,
      }];
    });
    if (segments.length === 0) throw new Error("no valid labels");
    return { segments, bpm: typeof parsed.bpm === "number" ? parsed.bpm : undefined };
  } catch {
    throw new Error(`${what}: 需要 chordBlock 输入（chordBlockIn 或 chordDetect 的 JSON）`);
  }
}

async function executeNode(
  nodeId: string,
  nodeType: string,
  params: Record<string, unknown>,
  gn: { inEdges: Array<{ fromNode: string; fromPort: number; toPort: number }> },
  dataMap: Map<string, Map<number, string>>,
  cacheDir: string,
  segmentId: string,
): Promise<Map<number, string>> {
  const inputPaths: Map<number, string> = new Map();
  for (const edge of gn.inEdges) {
    const upstream = dataMap.get(edge.fromNode);
    if (upstream) {
      const path = upstream.get(edge.fromPort);
      if (!path) continue;
      // Two edges landing on the SAME input port used to overwrite each other silently, and which one
      // survived depended on connection insertion order — the same graph could feed a different source
      // after a save/load reordered `connections`. First edge wins deterministically, and the loser is
      // logged so a mis-wired canvas is diagnosable instead of looking like the wrong node just ran.
      if (inputPaths.has(edge.toPort)) {
        logToBackend("warn", `Node "${nodeId}" (${nodeType}) has multiple edges on input port ${edge.toPort} — ignoring the one from ${edge.fromNode}:${edge.fromPort}`);
        continue;
      }
      inputPaths.set(edge.toPort, path);
    }
  }

    // Source nodes (no inputs) are self-contained — skip the primary-input guard.
  // P2-14: songLyrics/songPrompt 纯文本源；songSheet 源音频可空（11.2）。
  const SOURCE_NODE_TYPES = new Set(["midiFileIn", "chordBlockIn", "songLyrics", "songPrompt", "songSheet"]);
  const isSourceNode = SOURCE_NODE_TYPES.has(nodeType);

  const primaryInput = inputPaths.get(0);
  if (!primaryInput && !isSourceNode) {
    throw new Error(`Node "${nodeId}" (${nodeType}) has no input connected`);
  }

  const outputData = new Map<number, string>();

  switch (nodeType) {
    case "rvc":
    case "sovits": {
      const isRvc = nodeType === "rvc";
      const voiceName = params.voiceName as string | undefined;
      // S64 portability: persisted modelPath is absolute and can be stale after an install/data-dir
      // move; re-resolve by voiceName at use time (the panel pickers only heal on MOUNT).
      const modelPath = await healVoiceModelPath(nodeType, voiceName, params.modelPath as string | undefined);
      if (!voiceName || !modelPath) {
        throw new Error(`${isRvc ? "RVC" : "SoVITS"} node has no voice model selected — import one in the resource manager`);
      }
      const outputPath = `${cacheDir}/${nodeId}_${nodeType}.wav`;
      // Drive the node's (generic) progress bar off the Rust `voice-progress` events. The wire key is
      // SEGMENT-QUALIFIED, not the bare nodeId: a split copies the workflow verbatim, so both halves
      // carry the SAME node ids, and two segments may run concurrently (preflightRun only guards
      // same-segment double-dispatch). Filtering on the bare id made segment A's events drive segment
      // B's bar as well — B's listener closes over B's segmentId, so A's percentages were written to
      // B's node. Rust treats node_id as an opaque progress-routing token (progress_emitter is its
      // only consumer), so qualifying it is contract-safe. Torn down in `finally` — no leak on failure.
      const progressKey = `${segmentId}::${nodeId}`;
      const unlisten = await listen<{ node_id: string; progress: number }>(
        "voice-progress",
        (e) => {
          if (e.payload.node_id === progressKey) {
            useWorkflowStore.getState().setNodeProgress(segmentId, nodeId, e.payload.progress);
          }
        },
      );
      voiceInvokesInFlight.set(segmentId, (voiceInvokesInFlight.get(segmentId) ?? 0) + 1);
      try {
        // Options are EXACTLY the snake_case contract keys (voiceDefaults.ts, THE single source of
        // truth): node params store them verbatim, defaults fill anything unset. No other invoke
        // args — the legacy `shallowDiffusion` arg is gone (feature deferred by user decision).
        // S66/O5: Rust writes the wav to outputPath and returns just the path — the old
        // ~100MB samples JSON (response + save_temp_audio write-back) is gone.
        await invoke<{ path: string; sample_rate: number }>(
          isRvc ? "run_rvc" : "run_sovits",
          {
            voiceName,
            modelPath,
            audioPath: primaryInput!,
            // Must match the listener's filter above — Rust echoes this token back verbatim.
            nodeId: progressKey,
            outputPath,
            options: buildVoiceOptions(isRvc ? RVC_DEFAULTS : SOVITS_DEFAULTS, params),
          },
        );
      } finally {
        unlisten();
        voiceInvokesInFlight.set(segmentId, Math.max(0, (voiceInvokesInFlight.get(segmentId) ?? 1) - 1));
      }
      outputData.set(0, outputPath);
      break;
    }

    case "msstSeparation": {
      // Effective inference precision: the node's explicit choice, else the ARCH default
      // (melband = fp16 — inst_v2 fp32 saturates 12GB VRAM). Always SEND the effective value;
      // Rust degrades gracefully (missing .fp16.onnx → fp32 with a warning, and vice versa).
      // Arch comes from the catalog entry for the node's model file, falling back to the
      // installed list's detected architecture (covers locally imported models).
      const modelFile = (params.modelFile as string) ?? "";
      const arch =
        MSST_CATALOG.find((e) => e.filename === modelFile)?.architecture ??
        (useMsstModelStore.getState().installed.find((m) => m.filename === modelFile)
          ?.architecture as MsstArchitecture | undefined);
      // Hoisted out of `config` because the completion check below needs it to verify stem PROVENANCE.
      const msstOutputDir = `${cacheDir}/${nodeId}`;
      const config = {
        audioPath: primaryInput!,
        // S64 portability: recompute from the current models dir + stable modelFile (stale absolute
        // path after an install/data-dir move; the node UI only heals on mount).
        modelPath: await healMsstModelPath(
          params.modelFile as string | undefined,
          (params.modelPath as string) ?? (params.modelName as string) ?? "",
        ),
        // Per-NODE subdir: Rust names stems by LABEL only ("vocals.wav"), so two separation nodes in one
        // run emitting a same-labeled stem would overwrite each other inside the shared run dir. Rust
        // create_dir_all's the output dir before writing.
        outputDir: msstOutputDir,
        device: (params.device as string) ?? "cpu",
        normalize: (params.normalize as boolean) ?? false,
        useTta: (params.useTta as boolean) ?? false,
        shifts: (params.shifts as number) ?? 0,
        // Only override num_overlap when the user explicitly set it — otherwise OMIT it so Rust keeps
        // the model-JSON default (bs/mel=2, mdx23c/htdemucs=4). Always sending a number would force
        // every model to it and silently coarsen mdx23c/htdemucs (whose real default is 4).
        ...(params.numOverlap !== undefined ? { numOverlap: params.numOverlap as number } : {}),
        ...(params.batch !== undefined ? { batch: params.batch as number } : {}),
        // uvr_vr-only knobs: OMIT when unset so Rust keeps its own defaults (aggression 5,
        // post-process off, threshold 0.2). Other archs never set them.
        ...(params.aggression !== undefined ? { aggression: params.aggression as number } : {}),
        ...(params.postProcess !== undefined ? { postProcess: params.postProcess as boolean } : {}),
        ...(params.postProcessThreshold !== undefined ? { postProcessThreshold: params.postProcessThreshold as number } : {}),
        precision: (params.precision as string | undefined)
          ?? (arch !== undefined ? MSST_DEFAULT_PRECISION[arch] : undefined)
          ?? "fp32", // arch "unknown"/unresolvable → fp32 (Rust auto-uses fp16 if it's the only file)
      };
      // Rejection CODEs (SEPARATION_BUSY backstop / MSST_MODEL_NOT_CONVERTED) are localized once at
      // the run-catch (executeWorkflow / executeSingleNode) — the single mapping point.
      await invoke("run_msst_separation", { config });
      let status = await invoke<{ state: string | Record<string, string>; stems?: { label: string; path: string }[]; progress?: number }>("get_separation_status");
      // No-PROGRESS (stall) timeout instead of a fixed wall clock: a slow GPU / CPU fallback / TTA
      // (3+ full passes) can legitimately run very long, so we only fail when progress stops
      // advancing for STALL_TIMEOUT. A single chunk never takes this long even on CPU, so a real
      // stall (crash / OOM) is caught while a slow-but-advancing run is never killed.
      const STALL_TIMEOUT = 180 * 1000;
      let lastProgress = -1;
      let lastProgressAt = Date.now();
      while (typeof status.state === "string" && status.state !== "Completed" && status.state !== "Idle") {
        if (useWorkflowStore.getState().isCancelled(segmentId)) {
          await invoke("cancel_separation").catch(() => {});
          // Wait briefly to see if it already completed
          await new Promise((r) => setTimeout(r, 1000));
          status = await invoke("get_separation_status");
          if (status.state === "Completed") break;
          throw new Error("Cancelled");
        }
        await new Promise((r) => setTimeout(r, 500));
        status = await invoke("get_separation_status");
        if (typeof status.state === "string") {
          const p = status.progress ?? 0;
          if (p > lastProgress + 1e-4) { lastProgress = p; lastProgressAt = Date.now(); }
          useWorkflowStore.getState().setNodeProgress(segmentId, nodeId, p);
        }
        if (Date.now() - lastProgressAt > STALL_TIMEOUT) {
          // Abandon the backend job too: leaving it running permanently armed the SEPARATION_BUSY guard
          // (frontend showed "stopped" while the worker kept going — the state-desync the user hit).
          await invoke("cancel_separation").catch(() => {});
          throw new Error("MSST separation stalled: no progress for 180s (possible crash or out-of-memory)");
        }
      }
      if (typeof status.state === "object") {
        const errMsg = (status.state as Record<string, string>).Error ?? "MSST separation failed";
        throw new Error(errMsg);
      }
      if (status.state !== "Completed") {
        throw new Error(`MSST separation ended unexpectedly: ${JSON.stringify(status.state)}`);
      }
      useWorkflowStore.getState().setNodeProgress(segmentId, nodeId, 1);
      // A "Completed" status with no stems is a real failure (crash / no output written) — surface it
      // instead of marking the node green with nothing to deposit (the silent 0-output path).
      if (!status.stems || status.stems.length === 0) {
        throw new Error("MSST separation reported Completed but produced no stems");
      }
      // PROVENANCE gate. The Rust separation state is one GLOBAL slot, and `run_msst_separation` calls
      // clear_completed() then installs a fresh status — so if another segment dispatches its own
      // separation in the window between our job finishing and this read, the "Completed" we observe
      // carries THAT job's stems. We'd then deposit another segment's vocals as our own lanes: silent,
      // plausible-looking, and nearly impossible to diagnose from the UI. Every stem we accept must live
      // under the per-node outputDir we asked for, which no other node can ever be handed.
      const dirPrefix = msstOutputDir.replace(/\\/g, "/").toLowerCase();
      const ours = status.stems.filter((s) => s.path.replace(/\\/g, "/").toLowerCase().startsWith(dirPrefix));
      if (ours.length === 0) {
        throw new Error("MSST separation result belongs to another job (a separation was started elsewhere mid-run) — re-run this node");
      }
      for (let i = 0; i < ours.length; i++) {
        outputData.set(i, ours[i]!.path);
      }
      break;
    }

    case "transpose": {
      // The Signalsmith node (spectral transpose + formant controls) — built for
      // instrumentals. All-neutral = exact passthrough: forward the input path untouched so
      // an inert node costs nothing and downstream lanes keep byte-identical audio. A
      // non-default follow alone (0 st, 0 offset) is also inert — with no transpose there is
      // nothing for formants to follow or resist.
      const semitones = typeof params.semitones === "number" ? params.semitones : 0;
      const formantOffset = typeof params.formantOffset === "number" ? params.formantOffset : 0;
      // formantFollow: 1 = classic full-spectrum shift (pre-S82 default); a same-session
      // preserveFormants=true save reads as follow 0 (the checkbox this slider replaced).
      const formantFollow = typeof params.formantFollow === "number"
        ? params.formantFollow
        : params.preserveFormants === true ? 0 : 1;
      if (semitones === 0 && formantOffset === 0) {
        outputData.set(0, primaryInput!);
        break;
      }
      const outputPath = `${cacheDir}/${nodeId}_transpose.wav`;
      // TRANSPOSE_* CODEs are localized once at the run-catch (the single mapping point).
      await invoke("transpose_audio", {
        path: primaryInput,
        semitones,
        formantFollow,
        formantOffset,
        outputPath,
      });
      outputData.set(0, outputPath);
      break;
    }

    case "split": {
      const numOutputs = (params.outputs as number) ?? 2;
      for (let i = 0; i < numOutputs; i++) {
        outputData.set(i, primaryInput!);
      }
      break;
    }

    // ── Phase 1 母带合规节点族: merge / complianceCheck / lufsNormalize / dither ──
    case "merge": {
      const inputs = [...inputPaths.values()];
      if (inputs.length <= 1) {
        // 单路输入直接透传 (全局守卫已保证端口 0 必有连接).
        outputData.set(0, primaryInput!);
        break;
      }
      const outputPath = `${cacheDir}/${nodeId}_merge.wav`;
      // mix_audio_files: 单声道自动升为立体声, 短输入补静音, 未钳位 Float 求和.
      await invoke("mix_audio_files", { paths: inputs, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    case "complianceCheck": {
      // 音频原样透传 (端口 0), 合规报告 JSON 走端口 1 — 不打断下游音频链.
      const report = await invoke<Record<string, unknown>>("compliance_check", { path: primaryInput });
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify(report));
      break;
    }

    case "lufsNormalize": {
      const targetLufs = (params.targetLufs as number) ?? -16;
      const outputPath = `${cacheDir}/${nodeId}_lufs.wav`;
      await invoke("normalize_to_lufs", { input: primaryInput, targetLufs, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    case "dither": {
      // 0=无 / 1=TPDF / 2=TPDF+二阶噪声整形 (默认带整形, 固定种子结果可复现).
      const ditherType = Math.max(0, Math.min(2, Math.round((params.ditherType as number) ?? 2)));
      const outputPath = `${cacheDir}/${nodeId}_dither.wav`;
      await invoke("apply_dither", { input: primaryInput, ditherType, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    // ── Phase 4 母带补全节点族: busEq / stereoWidth / saturate / phaseRotate / dcRemove ──
    case "busEq": {
      // 5 段母线 EQ (hpf30 / lowShelf120 / peak500 / peak2.5k / highShelf10k).
      // BandDto 的 serde 字段名保持 snake_case gain_db — Tauri 只转换命令参数名, 不转换结构体字段.
      const g = (i: number) => Math.max(-12, Math.min(12, (params[`eqGain${i}`] as number) ?? 0));
      const bands = [
        { type: "hpf", freq: 30, gain_db: 0, q: 0.707, enabled: ((params.eqHpf as number) ?? 1) > 0 },
        { type: "lowShelf", freq: 120, gain_db: g(0), q: 0.707, enabled: true },
        { type: "peaking", freq: 500, gain_db: g(1), q: 1.0, enabled: true },
        { type: "peaking", freq: 2500, gain_db: g(2), q: 1.0, enabled: true },
        { type: "highShelf", freq: 10000, gain_db: g(3), q: 0.707, enabled: true },
      ];
      if (!bands.some((b) => b.enabled && (b.type === "hpf" || Math.abs(b.gain_db) > 1e-6))) {
        // 全部归零 → 恒等直通, 免一次无意义的重采样/重写.
        outputData.set(0, primaryInput!);
        break;
      }
      const outputPath = `${cacheDir}/${nodeId}_buseq.wav`;
      await invoke("apply_bus_eq", { input: primaryInput, bands, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    case "stereoWidth": {
      // M/S 宽度: 0=单声道, 1=原样, 上限 1.5 (Rust 侧二次钳位兜底).
      const width = Math.max(0, Math.min(1.5, (params.width as number) ?? 1));
      if (Math.abs(width - 1) < 1e-6) {
        outputData.set(0, primaryInput!);
        break;
      }
      const outputPath = `${cacheDir}/${nodeId}_width.wav`;
      await invoke("apply_stereo_width", { input: primaryInput, width, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    case "saturate": {
      // tanh 归一化饱和: drive 0=旁通, 计划默认 5% 轻激励; 输出永不超 0dBFS.
      const drive = Math.max(0, Math.min(1, (params.drive as number) ?? 0.05));
      if (drive < 1e-4) {
        outputData.set(0, primaryInput!);
        break;
      }
      const outputPath = `${cacheDir}/${nodeId}_saturate.wav`;
      await invoke("apply_saturate", { input: primaryInput, drive, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    case "phaseRotate": {
      // 8 级全通级联相位旋转: strength 0=旁通; 仅做峰值余量工具, 不做检测规避.
      const strength = Math.max(0, Math.min(1, (params.strength as number) ?? 0));
      if (strength < 1e-6) {
        outputData.set(0, primaryInput!);
        break;
      }
      const outputPath = `${cacheDir}/${nodeId}_phase.wav`;
      await invoke("apply_phase_rotate", { input: primaryInput, strength, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    case "dcRemove": {
      // 测量并移除直流偏移 (配方 F ①, 母带链第一环).
      const outputPath = `${cacheDir}/${nodeId}_dc.wav`;
      await invoke("remove_dc", { input: primaryInput, output: outputPath });
      outputData.set(0, outputPath);
      break;
    }

    // ── Phase 5 分析可视化节点族: 零损伤 — 端口 0 音频原样透传, 报告 JSON 走旁路端口 ──
    case "spectrogram": {
      // 5-1: Inferno PNG 落盘为 artifact (端口 1), 几何报告 JSON 走端口 2.
      const pngPath = `${cacheDir}/${nodeId}_spectrogram.png`;
      const report = await invoke<Record<string, unknown>>("analyze_spectrogram", {
        input: primaryInput,
        output: pngPath,
      });
      outputData.set(0, primaryInput!);
      outputData.set(1, pngPath);
      outputData.set(2, JSON.stringify(report));
      break;
    }

    case "f0Curve": {
      // 5-2: 全分辨率 NCCF 音高轨迹 (~86fps) — 报告 JSON 走端口 1.
      const report = await invoke<Record<string, unknown>>("track_f0", { input: primaryInput });
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify(report));
      break;
    }

    case "timbreMetrics": {
      // 5-3: 谱质心 / 滚降 / 过零率 / MFCC — 报告 JSON 走端口 1.
      const report = await invoke<Record<string, unknown>>("analyze_timbre", { input: primaryInput });
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify(report));
      break;
    }

    case "harmonicityCheck": {
      // 5-4: 谐波健康 8 指标 (S163 内核的工作流节点面); f0Hz=0 → 自动检测.
      const f0Hz = Math.max(0, (params.f0Hz as number) ?? 0);
      const report = await invoke<Record<string, unknown>>("analyze_harmonicity", {
        input: primaryInput,
        f0Hz: f0Hz > 0 ? f0Hz : null,
      });
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify(report));
      break;
    }

    case "spectralCompare": {
      // 5-5: B 口接对比音频 — 倍频程频段差 + 对数梅尔余弦/L2 (B 自动重采样); A 原样透传.
      const bInput = inputPaths.get(1);
      if (!bInput) throw new Error(`Node "${nodeId}" (spectralCompare) has no B input connected`);
      const report = await invoke<Record<string, unknown>>("compare_spectra", {
        inputA: primaryInput,
        inputB: bInput,
      });
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify(report));
      break;
    }

    case "dtwAlign": {
      // 5-6: 对数梅尔序列 DTW; band=Sakoe-Chiba 带宽 (0=不限); A 原样透传.
      const bInput = inputPaths.get(1);
      if (!bInput) throw new Error(`Node "${nodeId}" (dtwAlign) has no B input connected`);
      const band = Math.max(0, Math.round((params.band as number) ?? 0));
      const report = await invoke<Record<string, unknown>>("dtw_compare", {
        inputA: primaryInput,
        inputB: bInput,
        band: band > 0 ? band : null,
      });
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify(report));
      break;
    }

    case "abCompare": {
      // 5-7: 双输入零处理直通 — A→端口 0, B→端口 1 (纯前端, 无 Rust 命令).
      const bInput = inputPaths.get(1);
      if (!bInput) throw new Error(`Node "${nodeId}" (abCompare) has no B input connected`);
      outputData.set(0, primaryInput!);
      outputData.set(1, bInput);
      break;
    }

    case "lufsAnalyze": {
      // 5-8: BS.1770 整合响度 + 真峰值 (后端 measure_loudness); targetLufs 仅作报告对照基准, 音频原样透传.
      const targetLufs = (params.targetLufs as number) ?? -16;
      const report = await invoke<Record<string, unknown>>("measure_loudness", {
        path: primaryInput,
      });
      const integrated = report.integrated_lufs as number;
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify({
        ...report,
        target_lufs: targetLufs,
        delta_lufs: Number.isFinite(integrated)
          ? Math.round((integrated - targetLufs) * 10) / 10
          : null,
      }));
      break;
    }

    case "amtMidi": {
      const midiMode = (params.midiMode as string) ?? "smart";
      const useGpu = (params.useGpu as boolean) ?? true;
      const backend = (params.backend as string) ?? "yourmt3";
      const quantizeGrid = (params.quantizeGrid as string) ?? "off";
      const midiTrackMode = (params.midiTrackMode as string) ?? "multi_track";
      const muscriptorInstruments = (params.muscriptorInstruments as string[]) ?? [];
      // Python contract: only "official" | "telknet" (never the legacy "sustain_connect").
      // Sanitize legacy persisted values so old projects keep working.
      const rawChain = (params.muscriptorChain as string) ?? "official";
      const muscriptorChain = rawChain === "telknet" ? "telknet" : "official";
      const outDir = `${cacheDir}/amt_${nodeId}`;
      const isMulti = ["smart", "vocal_split", "six_stem_split"].includes(midiMode);

      // Stream sidecar progress into the node's progress bar. The sidecar
      // reports overall progress in [0,1] on the `progress` field.
      // Segment-qualified wire key, same reason as the rvc/sovits bar: split halves share node ids, so
      // the bare id let one segment's sidecar drive the other's bar. Here it ALSO fixes a real
      // process-registry collision — Rust keys `active_amt` (pid map behind cancel_amt_midi) by this
      // token, so two concurrent segments running the same node id overwrote each other's pid entry and
      // a cancel killed one sidecar while orphaning the other. The frontend-side ids below
      // (materializeAmtMidiTracks / useAmtStore.setRun) stay BARE — they key UI state, not the wire.
      const progressKey = `${segmentId}::${nodeId}`;
      // Declared BEFORE the listener that writes it: the subscription is live the moment `listen`
      // resolves, so a progress line arriving in that gap would hit the binding in its TDZ.
      let lastProgressAt = Date.now();
      const unlisten = await listen<{
        node_id: string | null;
        progress: number;
        total: number;
        message: string | null;
      }>("amt-progress", (e) => {
        if (e.payload.node_id && e.payload.node_id !== progressKey) return;
        const p = e.payload.total > 0 ? e.payload.progress / e.payload.total : e.payload.progress;
        lastProgressAt = Date.now();
        useWorkflowStore.getState().setNodeProgress(segmentId, nodeId, Math.min(1, Math.max(0, p)));
      });

      // Stall watchdog. `run_amt_midi` awaits the sidecar's stdout to EOF, so a wedged sidecar (CUDA
      // hang, OOM-thrash, a model download stuck on a dead socket) never resolves and never rejects —
      // the node sat at its last percentage forever and, because the execution entry stays "running",
      // the segment could not be re-run even after the user gave up. Same no-PROGRESS policy as the MSST
      // loop rather than a wall clock: a legitimately slow CPU transcription keeps emitting progress and
      // is never killed, while a true stall is force-terminated via the pid registry.
      const AMT_STALL_TIMEOUT = 300 * 1000;
      let stalled = false;
      const watchdog = setInterval(() => {
        if (Date.now() - lastProgressAt <= AMT_STALL_TIMEOUT) return;
        stalled = true;
        // Killing the sidecar is what makes the pending invoke reject — without it the await
        // below would keep hanging and this flag would never be read.
        void invoke("cancel_amt_midi", { nodeId: progressKey }).catch(() => {});
      }, 5000);

      try {
        let res;
        try {
          res = await invoke<{
          midi_path: string;
          total_notes: number | null;
          processing_time_secs: number;
          stem_midi_paths: Record<string, string> | null;
          vocal_midi_path: string | null;
          accompaniment_midi_path: string | null;
          merged_midi_path: string | null;
          separated_audio: Record<string, string> | null;
        }>("run_amt_midi", {
          audioPath: primaryInput!,
          midiMode,
          transcriptionBackend: isMulti ? backend : null,
          yourmt3Model: null,
          muscriptorModel: null,
          midiTrackMode: isMulti ? midiTrackMode : null,
          tempoMode: null,
          customBpm: null,
          quantizeNotes: quantizeGrid !== "off",
          quantizeGrid: quantizeGrid === "off" ? null : quantizeGrid,
          useGpu,
          gpuDevice: 0,
          outputDir: outDir,
          nodeId: progressKey,
          muscriptorInstruments: backend === "muscriptor" ? muscriptorInstruments : null,
          muscriptorProcessingChain: backend === "muscriptor" ? muscriptorChain : null,
        });
        } catch (e) {
          // The Stop button force-kills the AMT sidecar (cancel_amt_all). A kill makes the
          // in-flight run_amt_midi reject with a non-cancel error; if the run was actually
          // flagged cancelled, settle as a clean "Cancelled" so no red node / error modal.
          if (useWorkflowStore.getState().isCancelled(segmentId)) {
            await invoke("cancel_amt_all").catch(() => {});
            throw new Error("Cancelled");
          }
          // Our own watchdog killed it — report the stall, not the kill's generic sidecar error
          // (which reads like a crash and sends users hunting for a nonexistent model problem).
          if (stalled) {
            throw new Error("AMT transcription stalled: no progress for 300s (possible crash or out-of-memory)");
          }
          throw e;
        }
        // Separation-only modes (vocal_split / six_stem_split) emit WAV stems
        // in separated_audio rather than a MIDI — fall back to the first stem
        // so the node has a real, playable output.
        const primaryOut = res.midi_path || Object.values(res.separated_audio ?? {})[0] || "";
        outputData.set(0, primaryOut);

        // Expose additional MIDI artifacts (per-stem, vocal/accomp, merged) as
        // extra output slots so the node preview can show download buttons for
        // each. Slot 0 = primary; 1+ = supplementary.
        let slot = 1;
        if (res.merged_midi_path) outputData.set(slot++, res.merged_midi_path);
        if (res.vocal_midi_path) outputData.set(slot++, res.vocal_midi_path);
        if (res.accompaniment_midi_path) outputData.set(slot++, res.accompaniment_midi_path);
        if (res.stem_midi_paths) {
          for (const [, p] of Object.entries(res.stem_midi_paths)) {
            if (p) outputData.set(slot++, p);
          }
        }
        // Also expose separated WAV stems as output slots (for split modes).
        if (res.separated_audio) {
          for (const [, p] of Object.entries(res.separated_audio)) {
            if (p) outputData.set(slot++, p);
          }
        }

        useWorkflowStore.getState().setNodeProgress(segmentId, nodeId, 1);
        // Capture run context so the MIDI workbench can re-quantize / re-tempo.
        useAmtStore.getState().setRun({
          nodeId,
          audioPath: primaryInput!,
          midiPath: res.midi_path,
          mode: midiMode,
          backend,
          totalNotes: res.total_notes,
          processingTimeSecs: res.processing_time_secs,
        });
        // 节点转换完成后，把输出的 MIDI 自动落到主时间线：一个乐器音符轨道，
        // 替换本节点之前生成的轨道（去重），轨道名用 MIDI 里的乐器名。
        if (res.midi_path) {
          await materializeAmtMidiTracks(nodeId, res.midi_path, midiTrackMode, primaryInput, outDir);
        }

        // 🎵 Beat-This: 音频源自动测速 → 填工程 BPM (fire-and-forget,不阻塞节点执行).
        const audioExts = new Set(["wav", "mp3", "flac", "ogg", "m4a", "aiff", "aac", "opus"]);
        const isAudio = primaryInput && audioExts.has((primaryInput.split(".").pop() ?? "").toLowerCase());
        if (isAudio) {
          void (async () => {
            try {
              const tempo = await invoke<{ bpm: number; confidence: number; not_constant?: boolean }>(
                "analyze_segment_tempo", { path: primaryInput, windowStartMs: 0, windowEndMs: 0, beats_per_bar: 4 }
              );
              // Nothing awaits this IIFE, so it outlives the node — and used to land AFTER a Stop,
              // rewriting project BPM and toasting for a run the user had already cancelled. Re-check
              // once the invoke resolves and drop the result if the run is gone.
              if (useWorkflowStore.getState().isCancelled(segmentId)) return;
              if (tempo && tempo.bpm > 30 && tempo.bpm < 300) {
                const prev = useProjectStore.getState().tempo;
                if (Math.abs(prev - tempo.bpm) > 0.5) {
                  useProjectStore.getState().setTempo(Math.round(tempo.bpm * 10) / 10);
                  useAppStore.getState().showToast(
                    `🎵 Beat-This: 检测 BPM ${Math.round(tempo.bpm * 10) / 10} (置信度 ${Math.round(tempo.confidence * 100)}%) — 工程已自动填入`,
                    "info",
                  );
                }
              }
            } catch { /* tempo 检测失败不影响转谱 */ }
          })();
        }
      } finally {
        clearInterval(watchdog);
        unlisten();
      }
      break;
    }

    // ── 纯前端/AI 自动编曲节点 (不需要 Tauri backend, 浏览器里也能跑) ──

    case "speedShift": {
      // Signalsmith spectral time-stretch (保音高变速). Rust backend handles the DSP.
      // time_factor = 1/rate  (rate=2.0x -> factor=0.5  half duration)
      // rate=0.5x -> factor=2.0  double duration)
      const rate = (params.rate as number) ?? 1.0;
      if (Math.abs(rate - 1.0) < 0.001) {
        // Near-identity: passthrough (no DSP cost, cache-friendly)
        outputData.set(0, primaryInput!);
        break;
      }
      if (isSourceNode) {
        throw new Error("speedShift requires an input connection (audio or MIDI)");
      }
      const timeFactor = 1 / rate;
      try {
        const result = await invoke<{ output_path: string; duration_ms: number; sample_rate: number; channels: number }>(
          "stretch_segment_audio", { path: primaryInput, time_factor: timeFactor }
        );
        outputData.set(0, result.output_path);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        // STRETCH_RATIO_RANGE: user set an extreme rate we can't handle
        if (msg.includes("STRETCH_RATIO_RANGE")) {
          throw new Error("Speed rate must be between 0.25x and 4.0x");
        }
        // Other failure (missing file etc.) — fallback passthrough, don't break the graph.
        // 节点随后会被外层标成 "completed"，静默兜底会让用户误以为变速生效了 —— 标 degraded + toast 双保险。
        outputData.set(0, primaryInput!);
        const detail = backendErrorMessage(msg) ?? msg;
        console.warn("[speedShift] stretch failed, passthrough fallback:", msg);
        useWorkflowStore.getState().setNodeStatus(segmentId, nodeId, "degraded");
        useWorkflowStore.getState().setNodeError(segmentId, nodeId, detail);
        useAppStore.getState().showToast(
          i18n.t("workflow.warnSpeedShiftFallback", { detail }),
          "warning",
        );
      }
      break;
    }

    case "chordDetect": {
      if (isSourceNode) throw new Error("chordDetect needs audio/MIDI input");
      if (!primaryInput) throw new Error("chordDetect: no input connected");
      const ext = (primaryInput.split(".").pop() ?? "").toLowerCase();
      let notes: Array<{ tick: number; duration: number; pitch: number; velocity?: number }> = [];
      let ppq = 480;
      if (ext === "mid" || ext === "midi") {
        const res = await invoke<any>("import_score_file", { path: primaryInput });
        notes = (res?.tracks ?? []).flatMap((t: any) =>
          (t.notes ?? []).map((n: any) => ({ tick: n.tick, duration: n.duration, pitch: n.pitch, velocity: n.velocity }))
        );
        ppq = res?.ppq ?? 480;
      } else {
        throw new Error("chordDetect needs MIDI input. Connect an AMT node upstream (audio → AMT → chordDetect)");
      }
      if (notes.length === 0) throw new Error("chordDetect: input has no notes");
      const result = analyzeChords(notes, ppq, 4);
      outputData.set(0, JSON.stringify({
        segments: result.segments.map((s) => ({ startTick: s.startTick, endTick: s.endTick, label: s.label })),
        key: result.key.label, confidence: result.key.confidence,
      }));
      outputData.set(1, result.segments.map((s) => s.label).join(" | "));
      break;
    }

    case "autoArrange": {
      if (isSourceNode) throw new Error("autoArrange needs MIDI or chord input");
      if (!primaryInput) throw new Error("autoArrange: no input connected");

      // 1. Parse input to notes (MIDI file) or chord block (JSON)
      let notes: ChordAnalysisNote[] = [];
      
      let chordBlock: { chords: Array<{ label: string; root: number; quality?: string }>; bpm?: number } | null = null;

      if (primaryInput.trim().startsWith("{")) {
        // JSON input: could be chordBlock or notes array
        try {
          const parsed = JSON.parse(primaryInput);
          if (parsed.type === "chordBlock" && parsed.chords) {
            chordBlock = parsed;
          } else if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].pitch != null) {
            notes = parsed as ChordAnalysisNote[];
          }
        } catch { /* ignore */ }
      } else {
        // File path — try import_score_file
        const ext = (primaryInput.split(".").pop() ?? "").toLowerCase();
        if (ext === "mid" || ext === "midi") {
          const res = await invoke<any>("import_score_file", { path: primaryInput });
          notes = (res?.tracks ?? []).flatMap((t: any) =>
            (t.notes ?? []).map((n: any) => ({ tick: n.tick, duration: n.duration, pitch: n.pitch, velocity: n.velocity ?? 100 }))
          );
          void (res?.ppq);
        } else {
          throw new Error("autoArrange needs MIDI input (.mid/.midi), chord block, or AMT node upstream");
        }
      }

      // 2. If we got notes → run arrange() (real engine). If only chord block → build fake "block chord" notes.
      let chordAnalysisNotes: ChordAnalysisNote[] = notes;
      if (chordBlock && notes.length === 0) {
        // Synthesize block chord notes from chord labels → arrange engine treats them as "stub melody".
        // Each chord gets 2 block octaves on beats 1+3 (C4..C5), then piano takes the voicing below.
        const beatsPerBar = 4;
        const chordNotes: ChordAnalysisNote[] = [];
        chordBlock.chords.forEach((c, i) => {
          const barStart = i * beatsPerBar * TICKS_PER_BEAT;
          const rootPc = c.root;
          chordNotes.push({ tick: barStart, duration: beatsPerBar * TICKS_PER_BEAT, pitch: rootPc + 60, velocity: 90 });
          chordNotes.push({ tick: barStart + TICKS_PER_BEAT * 2, duration: TICKS_PER_BEAT * 2, pitch: rootPc + 72, velocity: 80 });
        });
        chordAnalysisNotes = chordNotes;
      }
      if (chordAnalysisNotes.length === 0) throw new Error("autoArrange: no notes or chord blocks to arrange");

      // 3. Call real arrange() engine — same as DAW menu layer
      const proj = useProjectStore.getState();
      const style = (params.style as ArrangeStyle) ?? "pop";
      const mood = (params.mood as ArrangeMood) ?? "neutral";
      const res = arrange({
        notes: chordAnalysisNotes,
        tempo: proj.tempo,
        timeSignature: proj.timeSignature,
        style,
        mood,
      });
      if (!res || res.bars === 0) throw new Error("autoArrange: arrange engine produced no output");

      // 4. Write the per-track JSON files, outputData holds paths.
      // The dir must be created FIRST: nothing else makes it (MSST/AMT get theirs from Rust/sidecar), so
      // writeTextFile into `<cacheDir>/arrange_<nodeId>/…` failed on a non-existent parent and the node
      // threw for what looks like an arrange failure. ensure_cache_dir is create_dir_all on the Rust side.
      const outDir = (
        await invoke<string>("ensure_cache_dir", { segmentId: `${segmentId}/arrange_${nodeId}` })
      ).replace(/\\/g, "/");
      const fs = await import("@tauri-apps/plugin-fs").catch(() => null);
      /** Returns the PATH on success, or null when there's no fs to write with — the caller decides what
       *  to publish. Previously the no-fs branch returned the JSON text while the caller stored `path`
       *  regardless, so every port pointed at a file that had never been written. */
      const writeJson = async (filePath: string, data: unknown): Promise<string | null> => {
        if (!fs || typeof (fs as { writeTextFile?: unknown }).writeTextFile !== "function") return null;
        await fs.writeTextFile(filePath, JSON.stringify(data));
        return filePath;
      };

      // Collects: outputData maps slot → result
      // 🎵 全部 11 轨输出 (OPT1+OPT2 扩充乐器现在也走 workflow 节点)
      const tracks = [
        { key: "drums",       label: "🥁 Drums",       notes: res.drums },
        { key: "bass",        label: "🎸 Bass",        notes: res.bass },
        { key: "piano",       label: "🎹 Piano",       notes: res.piano },
        { key: "guitarArp",   label: "🪕 GuitarArp",  notes: res.guitarArp },
        { key: "guitarStrum", label: "🎶 Strum",       notes: res.guitarStrum },
        { key: "epiano",      label: "🎼 E.Piano",     notes: res.epiano },
        { key: "strings",     label: "🎻 Strings",     notes: res.strings },
        { key: "pad",         label: "🪟 Pad",         notes: res.pad },
        { key: "synthPad",    label: "🎛 SynthPad",    notes: res.synthPad },
        { key: "pluck",       label: "💠 Pluck",       notes: res.pluck },
        { key: "melody",      label: "✨ Lead",        notes: res.melody },
      ];
      const manifest: any = { style, mood, key: res.key.label, confidence: res.key.confidence, bars: res.bars, startTick: res.startTick, endTick: res.endTick };
      for (let i = 0; i < tracks.length; i++) {
        const t = tracks[i]!;
        const payload = { key: res.key.label, chords: res.chords, notes: t.notes };
        const written = await writeJson(`${outDir}/${t.key}.json`, payload);
        manifest[t.key + "Notes"] = t.notes.length;
        // Publish the file path when it exists, else the JSON inline. Both readings are accepted by the
        // symbolic consumers (harmonizer/autoArrange sniff a leading `{` and read `.notes`), so the
        // fallback still carries real data instead of naming a file that was never written.
        outputData.set(i, written ?? JSON.stringify(payload));
      }
      // 和弦摘要挂在末端口，供下游 harmonizer/melodyGen 直接当 chordBlock 读（内联 JSON，与
      // chordDetect/chordBlockIn 的出口口径一致 —— chords 类端口一律传内容而非路径）。
      outputData.set(tracks.length, JSON.stringify({ chords: res.chords, key: res.key.label, bars: res.bars }));
      break;
    }

    case "deepOriginal": {
      const rate = (params.rate as number) ?? 1.0;
      const semitones = (params.semitones as number) ?? 0;
      let outPath = primaryInput ?? "";
      if (!isSourceNode && primaryInput && Math.abs(rate - 1.0) > 0.001) {
        try {
          const s = await invoke<{ output_path: string }>("stretch_segment_audio", { path: primaryInput, time_factor: 1 / rate });
          outPath = s.output_path;
        } catch (err) {
          // 兜底透传保持图能跑完，但节点会被外层标 "completed" —— 标 degraded + toast 双保险。
          const msg = err instanceof Error ? err.message : String(err);
          const detail = backendErrorMessage(msg) ?? msg;
          console.warn("[deepOriginal] stretch failed, passthrough fallback:", msg);
          useWorkflowStore.getState().setNodeStatus(segmentId, nodeId, "degraded");
          useWorkflowStore.getState().setNodeError(segmentId, nodeId, detail);
          useAppStore.getState().showToast(
            i18n.t("workflow.warnDeepOrigStretchFallback", { detail }),
            "warning",
          );
        }
      }
      if (!isSourceNode && primaryInput && semitones !== 0) {
        try {
          const tpath = cacheDir + "/deepOrig_" + nodeId + ".wav";
          await invoke("transpose_audio", { path: outPath, semitones, formantFollow: 1, formantOffset: 0, outputPath: tpath });
          outPath = tpath;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          const detail = backendErrorMessage(msg) ?? msg;
          console.warn("[deepOriginal] transpose failed, passthrough fallback:", msg);
          useWorkflowStore.getState().setNodeStatus(segmentId, nodeId, "degraded");
          useWorkflowStore.getState().setNodeError(segmentId, nodeId, detail);
          useAppStore.getState().showToast(
            i18n.t("workflow.warnDeepOrigTransposeFallback", { detail }),
            "warning",
          );
        }
      }
      for (let p = 0; p < 5; p++) outputData.set(p, outPath);
      break;
    }

    case "midiFileIn": {
      const filePath = (params.filePath as string) ?? "";
      if (!filePath) throw new Error("MIDI File In: pick a .mid file first (click the button)");
      outputData.set(0, filePath);
      break;
    }

    case "chordBlockIn": {
      const chordStr = (params.chords as string) ?? "C | Am | F | G";
      const bpm = (params.bpm as number) ?? 120;
      const rootMap2: Record<string, number> = { C: 0, "C#": 1, Db: 1, D: 2, "D#": 3, Eb: 3, E: 4, F: 5, "F#": 6, Gb: 6, G: 7, "G#": 8, Ab: 8, A: 9, "A#": 10, Bb: 10, B: 11 };
      const parts = chordStr.split(/[|,;]/).map((s) => s.trim()).filter(Boolean);
      const chords = parts.map((label) => {
        const m = label.match(/^([A-G][#b]?)/);
        return { label, root: (rootMap2[m?.[1] ?? "C"] ?? 0) };
      });
      outputData.set(0, JSON.stringify({ type: "chordBlock", chords, bpm, ppq: 480 }));
      break;
    }

    case "soundfontRender": {
      // 符号域 → 声音域：把上游的 MIDI 文件路径 / 音符 JSON 用 SoundFont 离线渲染成 WAV。
      // 入口两种形态都要吃：midiFileIn/amtMidi 给的是 .mid 路径，harmonizer/melodyGen 等
      // 符号域节点给的是 JSON.stringify(notes)。与 harmonizer 的解析口径保持一致。
      if (!primaryInput) throw new Error("soundfontRender: no input connected");
      const fontId = (params.fontId as string) ?? "";
      const presetId = (params.presetId as string) ?? "";
      if (!fontId || !presetId) {
        throw new Error("soundfontRender: pick a soundfont and preset on the node first");
      }
      let sfNotes: ChordAnalysisNote[] = [];
      // MIDI 文件自带 bpm，优先用它；JSON 音符流没有 tempo 信息，退回工程 BPM。
      let sfBpm = useProjectStore.getState().tempo;
      if (primaryInput.trim().startsWith("{") || primaryInput.trim().startsWith("[")) {
        try {
          const parsed = JSON.parse(primaryInput);
          if (Array.isArray(parsed)) sfNotes = parsed;
          else if (Array.isArray(parsed.notes)) sfNotes = parsed.notes;
          if (!Array.isArray(parsed) && typeof parsed.bpm === "number") sfBpm = parsed.bpm;
        } catch {
          throw new Error("soundfontRender: input is not valid note JSON");
        }
      } else {
        const sfExt = (primaryInput.split(".").pop() ?? "").toLowerCase();
        if (sfExt !== "mid" && sfExt !== "midi") {
          throw new Error("soundfontRender needs MIDI input (a .mid path or a note-JSON stream)");
        }
        const score = await invoke<any>("import_score_file", { path: primaryInput });
        sfNotes = (score?.tracks ?? []).flatMap((tr: any) =>
          (tr.notes ?? []).map((n: any) => ({
            // 轨道自身的 start_tick 是段落偏移，音符 tick 是轨内相对量 —— 漏加会把
            // 所有轨道压到 0 起点，多轨渲染直接串味。
            tick: (tr.start_tick ?? 0) + n.tick,
            duration: n.duration,
            pitch: n.pitch,
            velocity: n.velocity,
          })),
        );
        if (typeof score?.bpm === "number" && score.bpm > 0) sfBpm = score.bpm;
      }
      if (sfNotes.length === 0) throw new Error("soundfontRender: input has no notes");
      const sfTempo = sfBpm > 0 ? sfBpm : 120;
      const secPerTick = 60 / sfTempo / TICKS_PER_BEAT;
      const renderNotes = sfNotes.map((n) => ({
        start: n.tick * secPerTick,
        dur: n.duration * secPerTick,
        key: n.pitch,
        vel: n.velocity ?? 100,
      }));
      const sfWavPath = await invoke<string>("render_soundfont_notes", {
        fontId,
        presetId,
        notes: renderNotes,
        sampleRate: (params.sampleRate as number) ?? 44100,
        backend: (params.backend as string) ?? "builtin",
      });
      outputData.set(0, sfWavPath);
      break;
    }

    case "melodySimilarity": {
      // 原创性闸门：A(端口 0)是待查旋律，B(端口 1)是参考旋律(通常是扒下来的原曲)。
      // 相似度算法直接复用 symbol/melodyRestructure 那套(音高间隔 3-gram Jaccard +
      // 相对首音音高 3-gram + Parsons 轮廓 Pearson)，这里只负责取音符、判阈值、出报告。
      if (!primaryInput) throw new Error(`Node "${nodeId}" (melodySimilarity) has no A input connected`);
      const msBInput = inputPaths.get(1);
      if (!msBInput) throw new Error(`Node "${nodeId}" (melodySimilarity) has no B input connected`);
      const msA = await resolveMidiNotes(primaryInput, "melodySimilarity A");
      const msB = await resolveMidiNotes(msBInput, "melodySimilarity B");
      const msScore = melodySimilarity(msA.notes, msB.notes);
      // 节点上的阈值是百分数(0-100)，算法给的是 0-1 —— 比之前先归一化，别拿 35 去比 0.42。
      const msThreshold = Math.max(0, Math.min(100, (params.threshold as number) ?? 35)) / 100;
      // 低于阈值才算过闸：相似度越高越可疑，这个判断方向反了整个节点就废了。
      const msPass = msScore < msThreshold;
      outputData.set(0, primaryInput);
      outputData.set(1, JSON.stringify({
        similarity: Number(msScore.toFixed(4)),
        similarityPercent: Number((msScore * 100).toFixed(2)),
        threshold: Number(msThreshold.toFixed(4)),
        thresholdPercent: Number((msThreshold * 100).toFixed(2)),
        pass: msPass,
        verdict: msPass ? "original" : "too_similar",
        notesA: msA.notes.length,
        notesB: msB.notes.length,
      }));
      break;
    }

    case "harmonizer": {
      if (isSourceNode) throw new Error("harmonizer needs MIDI or chord input");
      if (!primaryInput) throw new Error("harmonizer: no input connected");
      let notes: ChordAnalysisNote[] = [];
      if (primaryInput.trim().startsWith("{")) {
        try {
          const parsed = JSON.parse(primaryInput);
          if (Array.isArray(parsed)) notes = parsed;
          else if (parsed.notes) notes = parsed.notes;
        } catch { /* ignore */ }
      } else {
        const ext = (primaryInput.split(".").pop() ?? "").toLowerCase();
        if (ext === "mid" || ext === "midi") {
          const res = await invoke<any>("import_score_file", { path: primaryInput });
          notes = (res?.tracks ?? []).flatMap((t: any) =>
            (t.notes ?? []).map((n: any) => ({ tick: n.tick, duration: n.duration, pitch: n.pitch, velocity: n.velocity ?? 100 }))
          );
        } else {
          throw new Error("harmonizer needs MIDI input or chordDetect/autoArrange output");
        }
      }
      if (notes.length === 0) throw new Error("harmonizer: no notes");
      const proj = useProjectStore.getState();
      const style = (params.chordStyle as string) ?? "POP_STANDARD";
      const chordsPerBar = (params.chordsPerBar as number) ?? 1;
      const keyName = (params.key as string) ?? "auto";
      const res = generateChordMidi({
        notes,
        timeSignature: proj.timeSignature,
        style: style as any,
        chordsPerBar: chordsPerBar === 2 ? 2 : 1,
        key: keyName,
      });
      if (!res) throw new Error("harmonizer: no chord output");
      outputData.set(0, JSON.stringify(res.notes));
      outputData.set(1, JSON.stringify({ segments: res.segments, key: res.key.label, bars: res.bars }));
      break;
    }

    case "melodyGen": {
      if (isSourceNode) throw new Error("melodyGen needs chord input");
      if (!primaryInput) throw new Error("melodyGen: no chord input connected");
      // P2-8 重做（规划 4.2）：chordBlock → externalChords/externalKey 直喂 arrange，
      // 旋律由和弦内音+调式音阶按小节生成。stub 骨架只负责撑起与和弦段对齐的时值跨度
      // （arrange 对空音符直接返回 null），起音 tick 跟随和弦段，保证旋律小节落在正确的和弦上。
      const { segments: chordSegs, bpm } = resolveChordSegments(primaryInput, "melodyGen");
      const proj = useProjectStore.getState();
      const style = (params.style as ArrangeStyle) ?? "pop";
      const mood = (params.mood as ArrangeMood) ?? "neutral";
      const stubNotes: ChordAnalysisNote[] = chordSegs.map((c) => ({
        tick: c.startTick,
        duration: Math.max(TICKS_PER_BEAT, c.endTick - c.startTick),
        pitch: c.root + 60,
        velocity: 90,
      }));
      const first = chordSegs[0]!;
      const minor = isMinorishQuality(first.quality);
      const res = arrange({
        notes: stubNotes,
        tempo: bpm ?? proj.tempo,
        timeSignature: proj.timeSignature,
        style,
        mood,
        externalChords: chordSegs,
        externalKey: {
          tonic: first.root,
          major: !minor,
          confidence: 0.5,
          label: PITCH_NAMES[first.root] + (minor ? "m" : ""),
        },
      });
      if (!res) throw new Error("melodyGen: arrange failed");
      outputData.set(0, JSON.stringify(res.melody));
      outputData.set(1, JSON.stringify({ style, mood, key: res.key.label, bars: res.bars, chordCount: chordSegs.length }));
      break;
    }

    // ── P2-14 歌曲制作节点族（规划 11.2/11.4，统一走 runSongTask）──────────

    case "songLyrics": {
      const text = ((params.text as string) ?? "").trim();
      if (!text) throw new Error(i18n.t("songNode.errEmptyLyrics"));
      outputData.set(0, `lyrics://${text}`);
      break;
    }

    case "songPrompt": {
      const text = ((params.text as string) ?? "").trim();
      if (!text) throw new Error(i18n.t("songNode.errEmptyPrompt"));
      // 同用 lyrics:// 前缀标记"虚拟文本"（11.3）；songGen 按端口序区分歌词/提示词
      outputData.set(0, `lyrics://${text}`);
      break;
    }

    case "songGen": {
      const task: SongTaskId = (params.songTask as string) === "instrumental" ? "instrumental" : "generate";
      const lyrics = songPortText(inputPaths.get(0)) ?? (params.lyrics as string) ?? "";
      const prompt = songPortText(inputPaths.get(1)) ?? (params.prompt as string) ?? "";
      const refAudio = inputPaths.get(2);
      const payload: SongTaskPayload = {
        model: (params.model as string) ?? "acestep-v1.5",
        lyrics,
        prompt,
        durationSec: (params.durationSec as number) ?? 0,
        seed: (params.seed as number) ?? undefined,
        wantStems: (params.wantStems as boolean) ?? false,
        wantMidi: (params.wantMidi as boolean) ?? false,
        wantLrc: (params.wantLrc as boolean) ?? false,
        extra: {
          ...(params.cot ? { cot: params.cot as string } : {}),
          ...((params.guidanceScale as number) ? { guidance_scale: params.guidanceScale as number } : {}),
          ...((params.numInferenceSteps as number) ? { inference_steps: params.numInferenceSteps as number } : {}),
          // 参考音频（可空端口 2）：非前缀 = 真实文件路径 → audio2audio
          ...(refAudio && !refAudio.startsWith("lyrics://")
            ? { ref_audio_input: refAudio, audio2audio_enable: true }
            : {}),
        },
      };
      const result = await runSongNode(task, payload, nodeId, segmentId);
      const outs = result.outputs;
      const audio = outs.find((o) => o.audio_path)?.audio_path;
      const midi = outs.find((o) => o.midi_path)?.midi_path;
      const lrc = outs.find((o) => o.lrc_path)?.lrc_path;
      if (audio) outputData.set(0, audio);
      if (midi) outputData.set(1, midi);
      if (lrc) outputData.set(2, lrc);
      const stems = outs.find((o) => o.stems)?.stems;
      if (stems) {
        // 端口 3+i 按 ACE_TRACK_CLASSES 固定顺序展开（与 UI outputLabels 一致）
        ACE_TRACK_CLASSES.forEach((cls, i) => {
          const p = stems[cls];
          if (p) outputData.set(3 + i, p);
        });
      }
      break;
    }

    case "songCover": {
      const prompt = songPortText(inputPaths.get(2)) ?? "";
      const refAudio = inputPaths.get(1);
      const payload: SongTaskPayload = {
        songName: (params.songName as string) || undefined,
        srcAudioPath: primaryInput!,
        prompt,
        coverStrength: (params.coverStrength as number) ?? 0.5,
        extra: refAudio && !refAudio.startsWith("lyrics://")
          ? { ref_audio_input: refAudio, audio2audio_enable: true }
          : undefined,
      };
      const result = await runSongNode("cover", payload, nodeId, segmentId);
      const audio = result.outputs.find((o) => o.audio_path)?.audio_path;
      if (audio) outputData.set(0, audio);
      break;
    }

    case "songRepaint": {
      const prompt = songPortText(inputPaths.get(1)) ?? "";
      const payload: SongTaskPayload = {
        songName: (params.songName as string) || undefined,
        srcAudioPath: primaryInput!,
        prompt,
        repaintStart: (params.repaintStart as number) ?? 0,
        repaintEnd: (params.repaintEnd as number) ?? 0,
      };
      const result = await runSongNode("repaint", payload, nodeId, segmentId);
      const audio = result.outputs.find((o) => o.audio_path)?.audio_path;
      if (audio) outputData.set(0, audio);
      break;
    }

    case "songComplete": {
      const classes = (params.trackClasses as string[]) ?? ["drums", "bass", "guitar"];
      const payload: SongTaskPayload = {
        srcAudioPath: primaryInput!,
        prompt: songPortText(inputPaths.get(1)) ?? "",
        trackClasses: classes,
      };
      const result = await runSongNode("complete", payload, nodeId, segmentId);
      const outs = result.outputs;
      const audio = outs.find((o) => o.audio_path)?.audio_path;
      if (audio) outputData.set(0, audio);
      const stems = outs.find((o) => o.stems)?.stems;
      if (stems) {
        // 端口 1+i 按勾选顺序（与 UI outputLabels 一致）
        classes.forEach((cls, i) => {
          const p = stems[cls];
          if (p) outputData.set(1 + i, p);
        });
      }
      break;
    }

    case "songExtract": {
      const classes = (params.trackClasses as string[]) ?? [...ACE_TRACK_CLASSES];
      const payload: SongTaskPayload = {
        srcAudioPath: primaryInput!,
        trackClasses: classes,
      };
      const result = await runSongNode("extract", payload, nodeId, segmentId);
      const stems = result.outputs.find((o) => o.stems)?.stems;
      if (stems) {
        classes.forEach((cls, i) => {
          const p = stems[cls];
          if (p) outputData.set(i, p);
        });
      }
      break;
    }

    case "songLego": {
      const payload: SongTaskPayload = {
        srcAudioPath: primaryInput!,
        prompt: songPortText(inputPaths.get(1)) ?? "",
        trackName: (params.trackName as string) ?? "guitar",
      };
      const result = await runSongNode("lego", payload, nodeId, segmentId);
      const audio = result.outputs.find((o) => o.audio_path)?.audio_path;
      if (audio) outputData.set(0, audio);
      break;
    }

    case "songStems": {
      const payload: SongTaskPayload = {
        songName: (params.songName as string) || undefined,
        srcAudioPath: primaryInput!,
      };
      const result = await runSongNode("stems", payload, nodeId, segmentId);
      const stems = result.outputs.find((o) => o.stems)?.stems;
      if (stems) {
        // demucs 四轨固定顺序（与 UI outputLabels 一致）
        DEMUCS_STEM_KINDS.forEach((cls, i) => {
          const p = stems[cls];
          if (p) outputData.set(i, p);
        });
      }
      break;
    }

    case "songSheet": {
      const payload: SongTaskPayload = {
        prompt: (params.prompt as string) || undefined,
      };
      const result = await runSongNode("sheet", payload, nodeId, segmentId);
      const outs = result.outputs;
      const abcPath = outs.find((o) => o.abc_path)?.abc_path;
      const midiPath = outs.find((o) => o.midi_path)?.midi_path;
      if (abcPath) outputData.set(0, abcPath);
      if (midiPath) {
        outputData.set(1, midiPath);
      } else if (abcPath) {
        // 后端只回了 ABC 时，前端补一步 abc_to_midi（注意：入参是 ABC 文本内容而非路径）
        const fs = await import("@tauri-apps/plugin-fs");
        const abc = await fs.readTextFile(abcPath);
        // Write beside the RUN's other artifacts, not next to the backend's ABC. The ABC lives in the
        // song-task output dir, which is NOT run-unique — a re-run produced the same `.mid` path, so the
        // reconciler's KEEP branch held the previous deposit and the new sheet never reached the lane.
        const outPath = `${cacheDir}/sheet_${nodeId}.mid`;
        const midi = await abcToMidi(abc, outPath);
        outputData.set(1, midi || outPath);
      }
      break;
    }

    // ── P2 符号域原创化节点族（规划 7.1/4.2）：确定性变换、种子可复现。
    // 音符族输出统一为音符数组 JSON（`[...]`），可被同族节点继续链式消费
    // （resolveMidiNotes 同时接受数组与 {notes:[...]} 包裹两种形态）。

    case "midiHumanize": {
      const { notes } = await resolveMidiNotes(primaryInput!, "midiHumanize");
      const out = humanizeNotes(notes, {
        timingMs: (params.timingMs as number) ?? 12,
        velocityJitter: (params.velocityJitter as number) ?? 8,
        tempo: useProjectStore.getState().tempo,
        seed: (params.seed as number) ?? DEFAULT_SYMBOL_SEED,
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "velocityCurve": {
      const { notes } = await resolveMidiNotes(primaryInput!, "velocityCurve");
      const curve = pickEnum<VelocityCurveKind>(params, "curve", ["crescendo", "decrescendo", "arch", "custom"], 2);
      // custom 曲线：0-100 采样值序列（逗号/空白分隔），归一到 0..1（applyVelocityCurve 的 shape 约定）
      const shape = ((params.customShape as string) ?? "")
        .split(/[\s,;]+/)
        .filter((s) => s !== "")
        .map((s) => Number(s))
        .filter((n) => Number.isFinite(n))
        .map((n) => Math.min(1, Math.max(0, n / 100)));
      const out = applyVelocityCurve(notes, {
        curve,
        intensity: (params.intensity as number) ?? 60,
        ...(curve === "custom" && shape.length >= 2 ? { shape } : {}),
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "swingQuantize": {
      const { notes } = await resolveMidiNotes(primaryInput!, "swingQuantize");
      const out = swingQuantizeNotes(notes, {
        grid: (params.grid as number) === 1 ? 16 : 8,
        swing: (params.swing as number) ?? 55,
        quantize: (params.quantize as number) ?? 0,
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "melodyReharm": {
      const { notes } = await resolveMidiNotes(primaryInput!, "melodyReharm");
      const out = degreeSwap(notes, {
        density: (params.density as number) ?? 40,
        seed: (params.seed as number) ?? DEFAULT_SYMBOL_SEED,
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "rhythmRestructure": {
      const { notes } = await resolveMidiNotes(primaryInput!, "rhythmRestructure");
      const proj = useProjectStore.getState();
      const out = rhythmRestructure(notes, {
        strength: (params.strength as number) ?? 60,
        beatsPerBar: (params.beatsPerBar as number) ?? proj.timeSignature[0] ?? 4,
        seed: (params.seed as number) ?? DEFAULT_SYMBOL_SEED,
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "contourMorph": {
      const { notes } = await resolveMidiNotes(primaryInput!, "contourMorph");
      const out = contourMorph(notes, {
        strength: (params.strength as number) ?? 60,
        keepClimax: (params.keepClimax as number) !== 0,
        seed: (params.seed as number) ?? DEFAULT_SYMBOL_SEED,
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "motifDevelop": {
      const { notes } = await resolveMidiNotes(primaryInput!, "motifDevelop");
      const out = motifDevelop(notes, {
        technique: pickEnum<MotifTechnique>(params, "technique", ["sequence", "invert", "retrograde", "augment", "diminish", "mixed"], 5),
        motifBars: Math.min(4, Math.max(2, (params.motifBars as number) ?? 2)),
        beatsPerBar: useProjectStore.getState().timeSignature[0] ?? 4,
        seed: (params.seed as number) ?? DEFAULT_SYMBOL_SEED,
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "reharmonize": {
      const { segments: chordSegs, bpm } = resolveChordSegments(primaryInput!, "reharmonize");
      const out = reharmonizeSegments(chordSegs, {
        strategies: [pickEnum<ReharmStrategy>(params, "strategy", ["diatonic", "borrowed", "tritone", "extension"], 0)],
        density: (params.density as number) ?? 50,
        seed: (params.seed as number) ?? DEFAULT_SYMBOL_SEED,
      });
      // 端口 0：chordBlock 形态（与 chordBlockIn 输出对齐，可回喂 melodyGen/harmonizer）；
      // 端口 1：chordDetect 兼容的 segments + bars 概览
      outputData.set(0, JSON.stringify(segmentsToChordBlock(out, bpm)));
      const barTicks = (useProjectStore.getState().timeSignature[0] ?? 4) * TICKS_PER_BEAT;
      outputData.set(1, JSON.stringify({
        segments: out.map((s) => ({ startTick: s.startTick, endTick: s.endTick, label: s.label })),
        bars: Math.ceil((out[out.length - 1]?.endTick ?? 0) / barTicks),
      }));
      break;
    }

    case "rhythmVariation": {
      const { notes } = await resolveMidiNotes(primaryInput!, "rhythmVariation");
      const out = varyRhythm(notes, {
        mode: pickEnum<RhythmVariationMode>(params, "mode", ["push", "layBack", "syncopate", "sparse"], 0),
        amount: (params.amount as number) ?? 50,
        beatsPerBar: useProjectStore.getState().timeSignature[0] ?? 4,
        seed: (params.seed as number) ?? DEFAULT_SYMBOL_SEED,
      });
      outputData.set(0, JSON.stringify(out));
      break;
    }

    case "structureEdit": {
      const { segments: chordSegs, bpm } = resolveChordSegments(primaryInput!, "structureEdit");
      const beatsPerBar = useProjectStore.getState().timeSignature[0] ?? 4;
      const out = editStructure(chordSegs, {
        op: pickEnum<StructureOp>(params, "op", ["repeatTail", "dropTail", "transposeTail", "lengthenTail"], 0),
        sectionBars: Math.min(8, Math.max(1, (params.sectionBars as number) ?? 4)),
        beatsPerBar,
        semitones: Math.min(12, Math.max(-12, (params.semitones as number) ?? 2)),
        ppq: TICKS_PER_BEAT,
      });
      outputData.set(0, JSON.stringify(segmentsToChordBlock(out, bpm)));
      outputData.set(1, JSON.stringify({
        segments: out.map((s) => ({ startTick: s.startTick, endTick: s.endTick, label: s.label })),
        bars: Math.ceil((out[out.length - 1]?.endTick ?? 0) / (beatsPerBar * TICKS_PER_BEAT)),
      }));
      break;
    }

    case "breathPlanner": {
      const { notes, ppq } = await resolveMidiNotes(primaryInput!, "breathPlanner");
      const lyrics = ((params.lyrics as string) ?? "").trim();
      const points = planBreathPoints(notes, {
        minGapBeats: (params.minGapBeats as number) ?? 1,
        ppq,
        ...(lyrics ? { lyrics } : {}),
      });
      // 端口 0：原样透传（换气点是「标注层」，不改变音符流）；端口 1：换气点 JSON
      outputData.set(0, primaryInput!);
      outputData.set(1, JSON.stringify(points));
      break;
    }

    default:
      // 良构的 WorkflowNodeType 到不了这里（input/output 在 :379/:520 已被上游跳过），
      // 但旧存档或「union 加了新成员却没写 engine case」以前会静默落到这里——
      // 返回空 outputData 后节点照样 "completed"，下游只会报出莫名其妙的 "no input connected"。
      // 在源头响亮地失败，别让错误漂到下游。
      throw new Error(`executeNode: unhandled node type "${nodeType}"`);
  }

  return outputData;
}

/** Single merged MIDI file → per-instrument note tracks in the main timeline.
 *  The AMT sidecar writes ONE merged .mid whose TRACKS are the instruments (this is the
 *  reliable source of per-stem data — the sidecar's `stem_midi_paths` map is unpopulated at
 *  runtime). Repeating the same node REPLACES its prior auto-generated tracks (tagged by
 *  `amtNodeId`) instead of piling up duplicates. Track names come from the MIDI TrackName
 *  (fallback: <midi file base>_<index>), so "出来多少个就显示多少个，名字是乐器名"。 */
async function materializeAmtMidiTracks(
  nodeId: string,
  midiPath: string,
  trackMode = "multi_track",
  sourceAudioPath?: string,
  playbackDir?: string,
): Promise<void> {
  const showToast = useAppStore.getState().showToast;
  // 撤销全覆盖（§user）：节点的"去旧建新"落轨合并成一步撤销——没有事务时，
  // removeTrack + N×addTrack 会碎成 N+1 步，撤销只能一条条退，体验极差。
  useHistoryStore.getState().beginTransaction();
  try {
    const project = useProjectStore.getState();
    // Replace any earlier auto-generated tracks from THIS node (dedup on re-run).
    for (const t of [...project.tracks]) {
      if (t.amtNodeId === nodeId) useProjectStore.getState().removeTrack(t.id);
    }

    let score: ImportedAmtScore;
    try {
      score = await invoke<ImportedAmtScore>("import_score_file", { path: midiPath });
    } catch (e) {
      showToast(i18n.t("amt.midiParseFailed", "MIDI 解析失败：无法读取转换结果。"), "error");
      return;
    }
    const tracks = score.tracks?.filter((t) => t.notes && t.notes.length > 0) ?? [];
    if (tracks.length === 0) return;

    const base =
      midiPath.split(/[/\\]/).pop()?.replace(/\.[^.]+$/, "") || nodeId;
    const addTrack = useProjectStore.getState().addTrack;

    // Synthesize per-instrument playback WAVs so every materialized track is AUDIBLE —
    // notes-only tracks play nothing in the DAW transport. Best-effort: on failure we
    // still import the notes (visible, editable), they'd just be silent.
    let wavs: Record<string, string> = {};
    let fullMixWav: string | undefined;
    if (sourceAudioPath && playbackDir) {
      try {
        const pb = await invoke<any>("amt_prepare_playback", {
          midiPath,
          audioPath: sourceAudioPath,
          outputDir: playbackDir,
        });
        wavs = (pb?.instrument_wavs ?? {}) as Record<string, string>;
        fullMixWav = pb?.transcription_wav || undefined;
        await invoke("allow_asset_dir", { dir: playbackDir }).catch(() => {});
      } catch (e) {
        console.warn(`[amtMidi] playback synthesis failed for ${nodeId}:`, e);
      }
    }

    // Per-track GM metadata (program+channel) maps track names onto the sidecar's
    // "gm:NNN"/"drums" WAV keys — the human names alone never match.
    let metaTracks: any[] = [];
    try {
      const meta = await invoke<any>("amt_midi_metadata", { midiPath });
      metaTracks = meta?.tracks ?? [];
    } catch { /* best-effort */ }
    const metaFor = (name: string) =>
      metaTracks.find((m: any) => m.name === name) || undefined;

    /** Build a playable audio lane for one track's WAV (peaks + duration). */
    const buildLane = async (
      wav: string,
      segmentId: string,
      label: string,
      maxTick: number,
    ): Promise<ProcessedOutput | undefined> => {
      let totalDurationMs = Math.max(1, (maxTick / (480 * (120 / 60))) * 1000);
      let waveformPeaks: number[] | undefined;
      try {
        const data = await useAudioStore.getState().loadAudioFile(wav);
        if (data.durationMs > 0) totalDurationMs = data.durationMs;
        if (data.peaks && data.peaks.length > 0) waveformPeaks = data.peaks;
      } catch { /* best-effort waveform */ }
      return {
        laneId: segmentId,
        laneLabel: label,
        group: label,
        audioPath: wav,
        totalDurationMs,
        waveformPeaks,
      } as ProcessedOutput;
    };

    // 单轨道模式：把所有乐器合并进一条轨道（轨道名 = MIDI 文件名），整条挂
    // 全乐器混音 WAV 保证可播放。
    if (trackMode === "single_track") {
      const allNotes = tracks.flatMap((it) =>
        it.notes.map((n) => ({
          id: crypto.randomUUID(),
          tick: n.tick,
          duration: n.duration,
          pitch: n.pitch,
          lyric: n.lyric || "La",
          velocity: n.velocity ?? 100,
        })),
      );
      let maxTick = 0;
      for (const n of allNotes) maxTick = Math.max(maxTick, n.tick + n.duration);
      const segmentId = crypto.randomUUID();
      const lane = fullMixWav ? await buildLane(fullMixWav, segmentId, base, maxTick) : undefined;
      addTrack({
        id: crypto.randomUUID(),
        name: localizeTrackName(base, 0),
        trackType: "instrument",
        volumeDb: 0,
        pan: 0,
        muted: false,
        solo: false,
        expanded: false,
        laneControls: {},
        amtNodeId: nodeId,
        segments: [
          {
            id: segmentId,
            startTick: 0,
            durationTicks: Math.max(1, maxTick),
            content: { type: "notes", notes: allNotes },
            processedOutputs: lane ? [lane] : undefined,
          },
        ],
      });
      showToast(i18n.t("amt.nodeTracksGenerated", { count: 1 }), "success");
      return;
    }

    // 全轨道（multi_track）模式：一个乐器一条轨道，轨道名 = 乐器名，每条挂
    // 对应乐器的合成 WAV 保证可播放。
    for (let i = 0; i < tracks.length; i++) {
      const it = tracks[i];
      if (!it?.notes || it.notes.length === 0) continue;
      const name = localizeTrackName(it.name?.trim() || `${base}_${i + 1}`, i);
      const maxTick = it.notes.reduce(
        (m, n) => Math.max(m, n.tick + n.duration),
        it.start_tick ?? 0,
      );
      const mt = metaFor(it.name || "");
      const wav = matchInstrumentWav(wavs, it.name || "", mt?.program ?? null, mt?.channel ?? null);
      const segmentId = crypto.randomUUID();
      const lane = wav ? await buildLane(wav, segmentId, name, maxTick) : undefined;
      addTrack({
        id: crypto.randomUUID(),
        name,
        trackType: "instrument",
        volumeDb: 0,
        pan: 0,
        muted: false,
        solo: false,
        expanded: false,
        laneControls: {},
        amtNodeId: nodeId,
        segments: [
          {
            id: segmentId,
            startTick: 0,
            durationTicks: Math.max(1, maxTick),
            content: {
              type: "notes",
              notes: it.notes.map((n) => ({
                id: crypto.randomUUID(),
                tick: n.tick,
                duration: n.duration,
                pitch: n.pitch,
                lyric: n.lyric || "La",
                velocity: n.velocity ?? 100,
              })),
            },
            processedOutputs: lane ? [lane] : undefined,
          },
        ],
      });
    }
    showToast(i18n.t("amt.nodeTracksGenerated", { count: tracks.length }), "success");
  } catch (e) {
    // 从不阻断主流程：MIDI 落轨失败只提示，不影响工作流运行结果本身。
    showToast(i18n.t("amt.nodeTracksFailed", "MIDI 音轨生成失败。"), "error");
  } finally {
    useHistoryStore.getState().commitTransaction();
  }
}

/** Shape of `import_score_file`'s result consumed by materializeAmtMidiTracks. */
interface ImportedAmtScore {
  tracks: {
    name: string;
    start_tick: number;
    notes: { tick: number; duration: number; pitch: number; lyric?: string; velocity?: number }[];
  }[];
}

/** MuScriptor 乐器名 / 六轨分离 stem 名 → 中文轨道名。
 *  匹配时先精确匹配乐器 ID，再大小写不敏感做子串兜底，
 *  最后 fallback 到原始名（MIDI TrackName 可能是任意字符串）。 */
const STEM_ZH_MAP: Record<string, string> = {
  // 六轨分离 stem 名
  vocals: "人声",
  voice: "人声",
  drums: "鼓组",
  drum: "鼓组",
  bass: "贝斯",
  guitar: "吉他",
  piano: "钢琴",
  other: "其他",
  // MuScriptor 乐器名
  acoustic_piano: "原声钢琴",
  electric_piano: "电钢琴",
  chromatic_percussion: "半音阶打击乐",
  organ: "风琴",
  acoustic_guitar: "原声吉他",
  clean_electric_guitar: "干净电吉他",
  distorted_electric_guitar: "失真电吉他",
  acoustic_bass: "原声贝斯",
  electric_bass: "电贝斯",
  violin: "小提琴",
  viola: "中提琴",
  cello: "大提琴",
  contrabass: "低音提琴",
  orchestral_harp: "管弦乐竖琴",
  timpani: "定音鼓",
  string_ensemble: "弦乐合奏",
  synth_strings: "合成弦乐",
  orchestra_hit: "管弦乐击奏",
  trumpet: "小号",
  trombone: "长号",
  tuba: "大号",
  french_horn: "圆号",
  brass_section: "铜管乐组",
  soprano_and_alto_sax: "高音/中音萨克斯",
  tenor_sax: "次中音萨克斯",
  baritone_sax: "上低音萨克斯",
  oboe: "双簧管",
  english_horn: "英国管",
  bassoon: "巴松管",
  clarinet: "单簧管",
  flutes: "长笛组",
  synth_lead: "合成主音",
  synth_pad: "合成铺底",
};

function localizeTrackName(raw: string | undefined | null, index: number): string {
  if (!raw) return `轨道 ${index + 1}`;
  const trimmed = raw.trim();
  // 先精确匹配
  if (STEM_ZH_MAP[trimmed]) return STEM_ZH_MAP[trimmed]!;
  // 再做大小写不敏感子串匹配（处理 "Drums-1" / "vocals_clean" 这类）
  const lower = trimmed.toLowerCase();
  for (const [key, zh] of Object.entries(STEM_ZH_MAP)) {
    if (lower.includes(key)) return zh;
  }
  // 最后 fallback：如果全是英文就原样返回（可能是 unknown instrument），否则原样
  return trimmed;
}

/**
 * Display label + stem suffix for edges into an Output node ("轨道组 · stem"). Lane IDENTITY/dedup is
 * handled separately by `laneId` (see laneIdFor + getLanes in trackLayout.ts), so same-named lanes
 * never collapse — the suffix is purely cosmetic.
 */
/** The stem suffix for one edge into an Output node. When the upstream node NAMES its ports
 *  (`stemLabels`, e.g. a separation node's vocals/instrumental) the stem is used EVEN FOR A
 *  SINGLE-EDGE output — a lone "Main" that is actually the instrumental stem was the root of the
 *  same-name collision confusion (two bare same-group lanes are indistinguishable; see getLanes'
 *  display numbering for what remains). Unnamed ports keep the bare group label when single. */
function laneStem(
  graph: ReturnType<typeof parseWorkflowGraph>,
  inEdgeCount: number,
  edge: { fromNode: string; fromPort: number },
): string | null {
  const stems = (graph.nodes.get(edge.fromNode)?.node.params as Record<string, unknown> | undefined)
    ?.stemLabels as string[] | undefined;
  const stem = stems?.[edge.fromPort];
  if (stem) return stem;
  return inEdgeCount > 1 ? `out${edge.fromPort}` : null;
}

function laneLabelFor(
  graph: ReturnType<typeof parseWorkflowGraph>,
  base: string,
  inEdgeCount: number,
  edge: { fromNode: string; fromPort: number },
): string {
  const stem = laneStem(graph, inEdgeCount, edge);
  // A group named exactly like its stem (e.g. a DETACHED lane whose new group IS the stem name)
  // would read "vocals · vocals" — collapse to the bare name.
  return stem && stem !== base ? `${base} · ${stem}` : base;
}

/** Stable lane IDENTITY for one edge into an Output node = `${outputNodeId}::${fromNode}:${fromPort}`.
 *  Keyed on the PHYSICAL EDGE — NOT the inbound-edge count, NOT the display stem — so adding/removing a
 *  SIBLING edge never re-keys an existing lane (a count-dependent id would wipe a persisted lane when the
 *  count crosses 1<->2), and two DIFFERENT upstream nodes feeding one Output stay distinct (e.g. blending
 *  two voices). Canvas / header / laneControls all key on THIS, not the label; stable across re-runs +
 *  save/load since node ids + ports persist in the graph. */
function laneIdFor(
  outputNodeId: string,
  edge: { fromNode: string; fromPort: number },
): string {
  return `${outputNodeId}::${edge.fromNode}:${edge.fromPort}`;
}

/** Count the lanes that reached Output nodes — NO decode (S59 deposit-perf O3). The old
 *  collectOutputs invoked load_audio_file per lane just to build a return value the sole caller
 *  read as `.length`, double-decoding every freshly-rendered stem in parallel with the live
 *  reconciler's own deposit (S32's "deposit slower than inference" bottleneck #1). The deposit
 *  itself is the reconciler's / RenderLinkWatcher's job via loadCachedOutput. The missing-feeder
 *  warn is preserved verbatim. */
function countOutputLanes(
  graph: ReturnType<typeof parseWorkflowGraph>,
  dataMap: Map<string, Map<number, string>>,
): number {
  let count = 0;
  for (const outId of graph.outputNodeIds) {
    const gn = graph.nodes.get(outId)!;
    const base = (gn.node.params as Record<string, unknown>).laneLabel as string ?? DEFAULT_OUTPUT_GROUP;
    for (const edge of gn.inEdges) {
      const audioPath = dataMap.get(edge.fromNode)?.get(edge.fromPort);
      if (!audioPath) {
        // Don't silently swallow a missing feeder — a dropped lane with no trace reads as "it worked".
        logToBackend("warn", `Output "${base}": upstream ${edge.fromNode} port ${edge.fromPort} produced no audio — lane skipped`);
        continue;
      }
      count++;
    }
  }
  return count;
}

/** Decode-failure memo for the SETTLE deposit path: a cached path that repeatedly fails to decode
 *  (file deleted/corrupt — e.g. swept externally) must stop re-arming hasUndepositedCache, or one dead
 *  file turns every watcher tick into a failing multi-second load_audio_file invoke forever
 *  (review-caught). Keyed segment|path; paths are RUN-UNIQUE so entries never need invalidation — a
 *  re-render mints new paths. A couple of retries are kept for transient Windows file locks. */
const cacheDecodeFailures = new Map<string, number>();
const DECODE_GIVE_UP = 3;
function noteDecodeFailure(segmentId: string, audioPath: string): void {
  const k = `${segmentId}|${audioPath}`;
  cacheDecodeFailures.set(k, (cacheDecodeFailures.get(k) ?? 0) + 1);
}
function decodeGivenUp(segmentId: string, audioPath: string): boolean {
  return (cacheDecodeFailures.get(`${segmentId}|${audioPath}`) ?? 0) >= DECODE_GIVE_UP;
}

export interface CachedPath {
  laneId: string;
  laneLabel: string;
  /** The Output node's group name (laneLabel's base) — carried onto the deposited lane. */
  group: string;
  audioPath: string;
  outputNodeId: string;
}

/**
 * Collect a single Output node's cached upstream PATHS (no audio decode) — the fast first half of a
 * deposit, so the caller can show per-lane loading placeholders immediately, then decode + load each
 * one. `missing` = at least one feeder had no cached audio (caller warns rather than silently dropping).
 */
export function collectCachedPaths(
  segmentId: string,
  outputNodeId: string,
  workflow: Workflow,
): { paths: CachedPath[]; missing: boolean } {
  const graph = parseWorkflowGraph(workflow);
  const gn = graph.nodes.get(outputNodeId);
  if (!gn) return { paths: [], missing: false };
  const base = ((gn.node.params as Record<string, unknown>).laneLabel as string) ?? DEFAULT_OUTPUT_GROUP;
  const cache = useWorkflowStore.getState().nodeOutputs[segmentId] ?? {};

  const paths: CachedPath[] = [];
  let missing = false;
  for (const edge of gn.inEdges) {
    const audioPath = cache[edge.fromNode]?.[edge.fromPort];
    if (!audioPath) {
      // Upstream not rendered yet — normal mid-run; the live reconciler waits + retries on cache change.
      // No log here: collectCachedPaths runs on every reconcile, so a warn would flood the panel at frame
      // rate. (A genuinely-never-rendered lane just never deposits — visible as no lane on the track.)
      missing = true;
      continue;
    }
    // MIDI is NOT an audio lane: deposits always decode via load_audio_file, which a .mid can never
    // satisfy (ffmpeg "Invalid data"). AMT's MIDI result is surfaced as NOTE tracks by
    // materializeAmtMidiTracks instead — skip it here so the Output node never errors/litters a lane.
    if (!isAudioFile(audioPath)) continue;
    paths.push({ laneId: laneIdFor(outputNodeId, edge), laneLabel: laneLabelFor(graph, base, gn.inEdges.length, edge), group: base, audioPath, outputNodeId });
  }
  return { paths, missing };
}

/** Is this a decodable AUDIO file (not a MIDI/score file)? Deposit-as-lane and matching all go
 *  through audio decode, so non-audio outputs (MIDI) must be excluded from audio lanes. */
export function isAudioFile(p: string): boolean {
  const ext = p.split(/[/\\]/).pop()?.split(".").pop()?.toLowerCase() ?? "";
  return !["mid", "midi", "kar", "smf"].includes(ext);
}

/**
 * HEADLESS deposit — resolve a segment's Output-node lanes from the render cache using the segment's OWN
 * persisted `workflow`, with NO open editor / ReactFlow refs. The normal LIVE deposit is done by the
 * WorkflowEditor reconciler, which only runs while THAT segment's editor is open; if you navigate away from a
 * rendering segment before it finishes, its loading placeholders never resolve to real lanes (their branch
 * finished in the cache, but nothing deposited it). This lets an always-mounted watcher settle them — e.g. so
 * a split-mid-render SOURCE whose editor was closed becomes "ready" and its linked halves can inherit.
 * Respects the CURRENT graph: orphan-cleans lanes whose Output node was deleted. Returns true if it changed
 * anything. CONTRACT: call at RENDER SETTLE only — leftover `loading` placeholders are PRUNED as dead
 * (the run that would have finished them is over); real (non-loading) lanes are never touched.
 */
export async function depositFromCache(trackId: string, segmentId: string, workflow: Workflow): Promise<boolean> {
  // The settle check happens at DISPATCH, but the decodes below await for seconds — a NEW run can start
  // for this segment mid-deposit (reopen editor + Run). Depositing then would clobber the new run's
  // placeholders with old-run audio, and the settle-prune would eat its live placeholders — so re-check
  // liveness around every store write and bail the moment a run owns the segment again.
  const runningNow = () => useWorkflowStore.getState().executions[segmentId]?.status === "running";
  if (runningNow()) return false;
  let graph: ReturnType<typeof parseWorkflowGraph> | null = null;
  try { graph = parseWorkflowGraph(workflow); } catch { /* broken/incomplete graph — still prune below */ }
  let changed = false;
  if (graph) {
    const outSet = new Set(graph.outputNodeIds);
    // Lanes already deposited at the SAME path need no re-decode (paths are run-unique — same path ⇒
    // same content); skipping them keeps a settle deposit that refreshes ONE re-rendered lane from
    // re-decoding every sibling stem. Loading placeholders are NOT "deposited" (they must resolve).
    const segBefore = useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
    const alreadyAt = new Map((segBefore?.processedOutputs ?? []).filter((o) => !o.loading).map((o) => [o.laneId, o.audioPath] as const));
    for (const outId of graph.outputNodeIds) {
      const { paths } = collectCachedPaths(segmentId, outId, workflow);
      const fresh = paths.filter((p) => alreadyAt.get(p.laneId) !== p.audioPath && !decodeGivenUp(segmentId, p.audioPath));
      if (fresh.length === 0) continue;
      // S59 deposit-perf O2: decode the lanes CONCURRENTLY (each is an independent load_audio_file
      // → hound decode + peaks); the old sequential awaits serialized 4-5 multi-second decodes.
      const decoded = (await Promise.all(fresh.map((p) => loadCachedOutput(p).catch(() => {
        noteDecodeFailure(segmentId, p.audioPath); // dead/corrupt cache file — stop re-arming the watcher after a few tries
        return null;
      })))).filter((o): o is ProcessedOutput => o !== null);
      if (runningNow()) return changed;
      if (decoded.length > 0) {
        // The store merge REPLACES by outputNodeId — it must receive the node's COMPLETE lane set:
        // re-attach the lanes the fresh-filter skipped (already deposited at the same cached path), or
        // the merge would silently delete this node's healthy sibling lanes — and the settle watcher
        // would then oscillate forever re-depositing the alternating halves (review-caught HIGH).
        const kept = (segBefore?.processedOutputs ?? []).filter(
          (o) => o.outputNodeId === outId && !o.loading
            && paths.some((p) => p.laneId === o.laneId && p.audioPath === o.audioPath),
        );
        useProjectStore.getState().mergeProcessedOutputs(trackId, segmentId, [...kept, ...decoded]);
        changed = true;
      }
    }
    // Orphan cleanup: drop lanes whose producing Output node no longer exists in the current graph.
    const seg = useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
    for (const o of seg?.processedOutputs ?? []) {
      if (o.outputNodeId && !outSet.has(o.outputNodeId)) {
        useProjectStore.getState().removeProcessedOutputsForNode(trackId, segmentId, o.outputNodeId);
        changed = true;
      }
    }
  }
  // SETTLE-TIME PRUNE: this runs when the render has SETTLED (RenderLinkWatcher), so any lane STILL
  // `loading` after the merges above was never finished by the run (cancelled / failed mid-branch —
  // its feeder has no cache) and nothing will ever finish it now. The open editor's reconciler prunes
  // these for the segment it shows ("uncached + idle → no lane"); this is the headless twin — without
  // it, split-mid-render + force-stop left the LINKED half's placeholder spinning forever (the watcher's
  // source-GONE path stripped loading lanes, the settle path didn't — this closes that asymmetry).
  // Non-loading lanes are NEVER touched here (cold cache ≠ remove).
  if (runningNow()) return changed; // a new run owns the placeholders now — never prune them
  const segNow = useProjectStore.getState().tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
  const outs = segNow?.processedOutputs ?? [];
  if (outs.some((o) => o.loading)) {
    useProjectStore.getState().replaceProcessedOutputs(trackId, segmentId, outs.filter((o) => !o.loading));
    changed = true;
  }
  return changed;
}

/** Decode one cached path into a finished ProcessedOutput (duration + waveform peaks). */
export async function loadCachedOutput(p: CachedPath): Promise<ProcessedOutput> {
  const info = await invoke<AudioFileInfo>("load_audio_file", { path: p.audioPath });
  return {
    laneId: p.laneId,
    laneLabel: p.laneLabel,
    group: p.group,
    audioPath: p.audioPath,
    totalDurationMs: info.duration_ms,
    waveformPeaks: info.peaks,
    outputNodeId: p.outputNodeId,
  };
}

/** All inbound-edge lane IDs for ONE Output node — STRUCTURE only, no cache/audio. Lets the auto-deposit
 *  reconciler know which lanes the node SHOULD carry, so it removes a lane only when its producing edge
 *  is gone — NOT merely because this session's render cache is cold (which would wipe persisted lanes on
 *  reopening a saved segment). `fromNode` = the lane's direct feeder, so the reconciler can ask whether
 *  that branch actually participates in the active run (per-feeder pending placeholders). */
export function outputLanes(workflow: Workflow, outputNodeId: string): { laneId: string; laneLabel: string; group: string; fromNode: string }[] {
  const graph = parseWorkflowGraph(workflow);
  const gn = graph.nodes.get(outputNodeId);
  if (!gn) return [];
  const base = ((gn.node.params as Record<string, unknown>).laneLabel as string) ?? DEFAULT_OUTPUT_GROUP;
  return gn.inEdges.map((edge) => ({
    laneId: laneIdFor(outputNodeId, edge),
    laneLabel: laneLabelFor(graph, base, gn.inEdges.length, edge),
    group: base,
    fromNode: edge.fromNode,
  }));
}

/** True iff this segment's render CACHE holds, for some structural Output lane, an audio path that the
 *  track does not carry yet (lane missing, or deposited at a DIFFERENT path — paths are run-unique, so a
 *  path difference IS a newer render). This is the settle watcher's "something landed that never
 *  deposited" signal: a re-render of an already-deposited lane keeps the OLD lane in place (the
 *  reconciler's KEEP branch — non-loading), so the watcher cannot rely on loading placeholders alone;
 *  without this check, closing the editor mid-re-render silently stranded the finished render in the
 *  cache (the track kept playing the previous version until the editor was reopened). Pure + cheap:
 *  reads graph structure and the cache map, decodes nothing. */
export function hasUndepositedCache(
  segmentId: string,
  workflow: Workflow | undefined,
  outs: ProcessedOutput[] | undefined,
): boolean {
  if (!workflow) return false;
  // Cheap pre-gate: no session render cache ⇒ nothing can be undeposited (skips the graph parse for
  // the many cold segments the settle watcher iterates over).
  const cache = useWorkflowStore.getState().nodeOutputs[segmentId];
  if (!cache || Object.keys(cache).length === 0) return false;
  let graph: ReturnType<typeof parseWorkflowGraph>;
  try { graph = parseWorkflowGraph(workflow); } catch { return false; }
  const deposited = new Map((outs ?? []).filter((o) => !o.loading).map((o) => [o.laneId, o.audioPath] as const));
  for (const outId of graph.outputNodeIds) {
    const { paths } = collectCachedPaths(segmentId, outId, workflow);
    for (const p of paths) {
      // A path whose decode has permanently failed (dead cache file) counts as deposited — otherwise
      // the settle watcher re-arms forever on a file that will never load (see cacheDecodeFailures).
      if (deposited.get(p.laneId) !== p.audioPath && !decodeGivenUp(segmentId, p.audioPath)) return true;
    }
  }
  return false;
}

/** All Output-group names in use across the project (every segment's persisted Output nodes), plus any
 *  `extra` (e.g. the calling node's not-yet-saved current value). The dropdown's option list — the group
 *  "registry" IS this union (per the project decision 先并集): a group exists by being assigned; there is
 *  no separate persisted list to migrate or drift. */
export function collectGroupNames(tracks: Track[], extra: string[] = []): string[] {
  const names = new Set<string>([DEFAULT_OUTPUT_GROUP, ...extra.filter(Boolean)]);
  for (const t of tracks) {
    for (const seg of t.segments) {
      for (const n of seg.workflow?.nodes ?? []) {
        if (n.nodeType !== "output") continue;
        const g = n.params?.laneLabel;
        if (typeof g === "string" && g) names.add(g);
      }
    }
  }
  return [...names].sort((a, b) => a.localeCompare(b));
}

export interface DetachPlan {
  oldNodeId: string;
  /** One new single-edge Output node per inbound edge of the old node. */
  newNodes: { id: string; group: string; position: { x: number; y: number }; edge: { fromNode: string; fromPort: number } }[];
  /** Deposited-lane rewrite: old laneId (under the old node) → the new node's identity. */
  mapping: { oldLaneId: string; newLaneId: string; newNodeId: string; group: string; laneLabel: string }[];
}

/**
 * Plan an "ungroup" (解组): split a multi-input Output node into one single-edge Output node per inbound
 * edge. What splits is the 组 — the CO-OPERATION unit (lanes sharing one Output node: co-selected,
 * co-sliced, shared settings). The 轨道组 NAME is deliberately KEPT: every new node inherits the old
 * node's group name, so the lanes stay in "Main" with their exact display labels ("Main · vocals");
 * only the shared-node linkage is broken. PURE — computes the graph delta + the deposited-lane rewrite;
 * the caller applies it to the editor graph (so it lands in the node-graph undo stack) and to the project
 * store (laneOps/laneControls inheritance rides in `applyLaneDetach`). Null when < 2 inbound edges.
 */
export function planDetachGroup(workflow: Workflow, outputNodeId: string): DetachPlan | null {
  let graph: ReturnType<typeof parseWorkflowGraph>;
  try { graph = parseWorkflowGraph(workflow); } catch { return null; }
  const gn = graph.nodes.get(outputNodeId);
  if (!gn || gn.inEdges.length < 2) return null;
  const base = ((gn.node.params as Record<string, unknown>).laneLabel as string) ?? DEFAULT_OUTPUT_GROUP;
  const pos = gn.node.position;
  const newNodes: DetachPlan["newNodes"] = [];
  const mapping: DetachPlan["mapping"] = [];
  gn.inEdges.forEach((edge, i) => {
    const id = `audioOutput-${crypto.randomUUID().slice(0, 8)}`;
    newNodes.push({ id, group: base, position: { x: pos.x + i * 40, y: pos.y + i * 96 }, edge });
    mapping.push({
      oldLaneId: laneIdFor(outputNodeId, edge),
      newLaneId: laneIdFor(id, edge),
      newNodeId: id,
      group: base,
      // Single-edge label via the SAME formula deposits use (a stem-labeled feeder keeps its suffix →
      // the display is IDENTICAL to before the ungroup), so the reconciler's KEEP branch matches without
      // a re-deposit. Two no-stem lanes both labeled bare "Main" de-collide at display time (getLanes).
      laneLabel: laneLabelFor(graph, base, 1, edge),
    });
  });
  return { oldNodeId: outputNodeId, newNodes, mapping };
}

/**
 * Rebuild the RUNTIME render cache + node badges for a segment from its PERSISTED processedOutputs. The
 * workflow store (nodeOutputs / nodeStatuses) is runtime-only and cold after a project load/autoload, but
 * the rendered audio is KEPT (each deposited lane carries its audioPath). Without this, on reopening a
 * loaded project the render nodes show idle and — worse — deleting an Output edge and reconnecting it
 * finds a cold cache and re-runs a full separation of audio that already exists. This reconstructs, per
 * deposited lane, the DIRECT feeder node's output path at its port (so collectCachedPaths re-finds it →
 * reconnect re-deposits from cache, no re-run) and marks every node UPSTREAM of a deposited lane
 * "completed" (the deposit proves they all ran). Idempotent + non-destructive: no-op if the cache is
 * already warm, and it only writes runtime overlays (never processedOutputs / never the undo doc).
 */
export function rehydrateRenderState(
  segmentId: string,
  segment: { workflow?: Workflow; processedOutputs?: ProcessedOutput[] },
): void {
  const wf = segment.workflow;
  const outs = (segment.processedOutputs ?? []).filter(
    (o) => !o.loading && o.outputNodeId && o.audioPath && !o.audioPath.startsWith("__pending"),
  );
  if (!wf || outs.length === 0) return;
  const store = useWorkflowStore.getState();
  const warm = store.nodeOutputs[segmentId];
  if (warm && Object.keys(warm).length > 0) return; // already warm (live / just-run) — don't clobber

  let graph: ReturnType<typeof parseWorkflowGraph>;
  try {
    graph = parseWorkflowGraph(wf);
  } catch {
    return; // incomplete/invalid graph (no input/output/cycle) — nothing to safely rehydrate
  }

  const byLaneId = new Map(outs.map((o) => [o.laneId, o] as const));
  const nodeOutputs: Record<string, string[]> = {};
  const outputIds = new Set<string>();

  for (const outId of new Set(outs.map((o) => o.outputNodeId as string))) {
    const gn = graph.nodes.get(outId);
    if (!gn) continue;
    for (const edge of gn.inEdges) {
      const po = byLaneId.get(laneIdFor(outId, edge)); // laneId is ALWAYS `${out}::${fromNode}:${fromPort}`
      if (!po) continue;
      (nodeOutputs[edge.fromNode] ??= [])[edge.fromPort] = po.audioPath; // index = port, matches collectCachedPaths
      outputIds.add(outId);
    }
  }
  if (Object.keys(nodeOutputs).length === 0) return;
  // Mark "completed" ONLY the nodes whose output we ACTUALLY warmed (the deposited lanes' DIRECT feeders)
  // plus the Output nodes — so a green badge always means "cache-backed / reusable". A deeper ancestor in a
  // chain (separation → transpose → output) is NOT warmable (only the deposited lane's direct-feeder audio is
  // persisted), so greening it would be a badge no cache backs — and single-running a downstream node would
  // still re-run it. This also matches the user's rationale ("we kept the separation RESULT" = the deposited
  // lane = the direct feeder). The common input→separation→output graph still greens the separation node,
  // since it IS the direct feeder. (Excludes the input node too — a real run never sets its status.)
  store.hydrateRenderState(segmentId, nodeOutputs, [...Object.keys(nodeOutputs), ...outputIds]);
}
