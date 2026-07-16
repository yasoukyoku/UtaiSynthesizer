import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { Segment, ProcessedOutput, Track } from "../../types/project";
import type { Workflow } from "../../types/project";
import { parseWorkflowGraph } from "./graph";
import { useProjectStore } from "../../store/project";
import { useWorkflowStore } from "../../store/workflow";
import { useAppStore, type MissingModelItem } from "../../store/app";
import { useMsstModelStore } from "../../store/msst-models";
import { useAudioStore } from "../../store/audio";
import { logToBackend } from "../log";
import { backendErrorMessage, isCancelError } from "../backendError";
import { maybeShowErrorModal } from "../errorDisplay";
import { DEFAULT_OUTPUT_GROUP } from "../constants";
import { MSST_CATALOG, MSST_DEFAULT_PRECISION, type MsstArchitecture } from "../models/msst-catalog";
import { RVC_DEFAULTS, SOVITS_DEFAULTS, buildVoiceOptions } from "./voiceDefaults";
import { healVoiceModelPath, healMsstModelPath } from "./modelPathHeal";
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
  store.startExecution(segmentId, participants);
  store.clearNodeStatuses(segmentId);
  // A full run recomputes every node. Drop any warm/rehydrated cache first so the live reconciler shows
  // loading placeholders and deposits each lane FRESH as its node finishes — never an early decode of a
  // deterministic path this run is about to overwrite in place (the crash-recovery "keeps old stem" hazard).
  store.clearNodeOutputs(segmentId);

  try {
    logToBackend("info", `Workflow started (${workflow.nodes.length} nodes)`);
    const graph = parseWorkflowGraph(workflow);

    // Mark all non-IO nodes as waiting BEFORE the first await: the reconciler's pending placeholders
    // key on per-feeder participation (waiting/running), so the marks must land in the same tick the
    // run starts — marking them after fetchInstalled/ensureRunDir left an await-sized window in which
    // connected lanes showed no placeholder at all.
    for (const nodeId of graph.sorted) {
      const gn = graph.nodes.get(nodeId)!;
      if (gn.node.nodeType !== "input" && gn.node.nodeType !== "output") {
        store.setNodeStatus(segmentId, nodeId, "waiting");
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

      if (useWorkflowStore.getState().isCancelled(segmentId)) {
        throw new Error("Cancelled");
      }

      store.setNodeStatus(segmentId, nodeId, "running");

      const outputData = await executeNode(nodeId, nodeType, params, gn, dataMap, cacheDir, segmentId);

      dataMap.set(nodeId, outputData);
      if (outputData.size > 0) {
        useWorkflowStore.getState().setNodeOutputs(segmentId, nodeId, Array.from(outputData.values()));
      }
      store.setNodeStatus(segmentId, nodeId, "completed");
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
  store.startExecution(segmentId, participants);

  try {
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

      if (useWorkflowStore.getState().isCancelled(segmentId)) {
        throw new Error("Cancelled");
      }

      // Reuse a node's cached output ONLY if it's DENSE (every port present). rehydrateRenderState may warm
      // a multi-output node with just the DEPOSITED ports (a sparse array with holes); reusing that would
      // feed `undefined` to a downstream node reading a non-deposited port ("has no input connected"). A
      // hole means that port isn't cached → fall through and RE-RUN the node to regenerate all ports.
      const cached = store.nodeOutputs[segmentId]?.[nodeId];
      if (cached && cached.length > 0 && nodeId !== targetNodeId && isDenseCache(cached)) {
        const m = new Map<number, string>();
        cached.forEach((p, i) => m.set(i, p));
        dataMap.set(nodeId, m);
        continue;
      }

      store.setNodeStatus(segmentId, nodeId, "running");

      const outputData = await executeNode(
        nodeId, gn.node.nodeType, gn.node.params as Record<string, unknown>,
        gn, dataMap, cacheDir, segmentId,
      );

      dataMap.set(nodeId, outputData);
      if (outputData.size > 0) {
        store.setNodeOutputs(segmentId, nodeId, Array.from(outputData.values()));
      }
      store.setNodeStatus(segmentId, nodeId, "completed");

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
      store.setNodeStatus(segmentId, targetNodeId, "error");
      store.setNodeError(segmentId, targetNodeId, display);
      maybeShowErrorModal(msg, display);
    }
    store.clearPendingStatuses(segmentId);
    store.failExecution(segmentId, display);
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
      if (path) inputPaths.set(edge.toPort, path);
    }
  }

  const primaryInput = inputPaths.get(0);
  if (!primaryInput) {
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
      // Drive the node's (generic) progress bar off the Rust `voice-progress` events, filtered
      // by nodeId. The listener is torn down in `finally` so a failed/cancelled run can't leak it.
      const unlisten = await listen<{ node_id: string; progress: number }>(
        "voice-progress",
        (e) => {
          if (e.payload.node_id === nodeId) {
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
            audioPath: primaryInput,
            nodeId,
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
      const config = {
        audioPath: primaryInput,
        // S64 portability: recompute from the current models dir + stable modelFile (stale absolute
        // path after an install/data-dir move; the node UI only heals on mount).
        modelPath: await healMsstModelPath(
          params.modelFile as string | undefined,
          (params.modelPath as string) ?? (params.modelName as string) ?? "",
        ),
        // Per-NODE subdir: Rust names stems by LABEL only ("vocals.wav"), so two separation nodes in one
        // run emitting a same-labeled stem would overwrite each other inside the shared run dir. Rust
        // create_dir_all's the output dir before writing.
        outputDir: `${cacheDir}/${nodeId}`,
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
      for (let i = 0; i < status.stems.length; i++) {
        outputData.set(i, status.stems[i]!.path);
      }
      break;
    }

    case "transpose": {
      // Fidelity transpose (spectral, Signalsmith) — built for instrumentals. 0 = exact
      // passthrough: forward the input path untouched so an inert node costs nothing and
      // downstream lanes keep byte-identical audio.
      const semitones = typeof params.semitones === "number" ? params.semitones : 0;
      if (semitones === 0) {
        outputData.set(0, primaryInput);
        break;
      }
      const outputPath = `${cacheDir}/${nodeId}_transpose.wav`;
      // TRANSPOSE_* CODEs are localized once at the run-catch (the single mapping point).
      await invoke("transpose_audio", {
        path: primaryInput,
        semitones,
        outputPath,
      });
      outputData.set(0, outputPath);
      break;
    }

    case "split": {
      const numOutputs = (params.outputs as number) ?? 2;
      for (let i = 0; i < numOutputs; i++) {
        outputData.set(i, primaryInput);
      }
      break;
    }
  }

  return outputData;
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
    paths.push({ laneId: laneIdFor(outputNodeId, edge), laneLabel: laneLabelFor(graph, base, gn.inEdges.length, edge), group: base, audioPath, outputNodeId });
  }
  return { paths, missing };
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
