import { invoke } from "@tauri-apps/api/core";
import type { Segment, ProcessedOutput } from "../../types/project";
import type { Workflow } from "../../types/project";
import { parseWorkflowGraph } from "./graph";
import { useWorkflowStore } from "../../store/workflow";
import { useMsstModelStore } from "../../store/msst-models";

interface AudioFileInfo {
  duration_ms: number;
  peaks: number[];
}

export async function executeWorkflow(
  segmentId: string,
  segment: Segment,
  workflow: Workflow,
): Promise<ProcessedOutput[]> {
  const store = useWorkflowStore.getState();
  store.startExecution(segmentId);
  store.clearNodeStatuses(segmentId);

  try {
    await useMsstModelStore.getState().fetchInstalled();

    const graph = parseWorkflowGraph(workflow);
    const rawCacheDir = await invoke<string>("ensure_cache_dir", { segmentId });
    const cacheDir = rawCacheDir.replace(/\\/g, "/");

    const dataMap = new Map<string, Map<number, string>>();

    if (segment.content.type !== "audioClip") {
      throw new Error("Workflow execution requires an audioClip segment");
    }
    const inputData = new Map<number, string>();
    inputData.set(0, segment.content.sourcePath);
    dataMap.set(graph.inputNodeId, inputData);

    // Mark all non-IO nodes as waiting
    for (const nodeId of graph.sorted) {
      const gn = graph.nodes.get(nodeId)!;
      if (gn.node.nodeType !== "input" && gn.node.nodeType !== "output") {
        store.setNodeStatus(segmentId, nodeId, "waiting");
      }
    }

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

    const results = await collectOutputs(graph, dataMap);

    store.completeExecution(segmentId);
    return results;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const store = useWorkflowStore.getState();
    if (store.executions[segmentId]?.currentNodeId) {
      store.setNodeStatus(segmentId, store.executions[segmentId]!.currentNodeId!, "error");
      store.setNodeError(segmentId, store.executions[segmentId]!.currentNodeId!, msg);
    }
    store.failExecution(segmentId, msg);
    throw err;
  }
}

export async function executeSingleNode(
  segmentId: string,
  segment: Segment,
  workflow: Workflow,
  targetNodeId: string,
): Promise<void> {
  const store = useWorkflowStore.getState();
  store.startExecution(segmentId);

  try {
    const graph = parseWorkflowGraph(workflow);
    const rawCacheDir = await invoke<string>("ensure_cache_dir", { segmentId });
    const cacheDir = rawCacheDir.replace(/\\/g, "/");

    if (segment.content.type !== "audioClip") {
      throw new Error("Workflow execution requires an audioClip segment");
    }

    const dataMap = new Map<string, Map<number, string>>();
    const inputData = new Map<number, string>();
    inputData.set(0, segment.content.sourcePath);
    dataMap.set(graph.inputNodeId, inputData);

    for (const nodeId of graph.sorted) {
      const gn = graph.nodes.get(nodeId)!;
      if (gn.node.nodeType === "input" || gn.node.nodeType === "output") continue;

      if (useWorkflowStore.getState().isCancelled(segmentId)) {
        throw new Error("Cancelled");
      }

      const cached = store.nodeOutputs[segmentId]?.[nodeId];
      if (cached && cached.length > 0 && nodeId !== targetNodeId) {
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
    if (msg !== "Cancelled") {
      store.setNodeStatus(segmentId, targetNodeId, "error");
      store.setNodeError(segmentId, targetNodeId, msg);
    }
    store.failExecution(segmentId, msg);
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
    case "rvc": {
      const outputPath = `${cacheDir}/${nodeId}_rvc.wav`;
      const result = await invoke<{ audio: number[]; sample_rate: number }>("run_rvc", {
        voiceName: params.voiceName ?? "default",
        modelPath: params.modelPath ?? "",
        audioPath: primaryInput,
        options: {
          f0_shift: params.pitchShift ?? 0,
          speaker_id: null,
          index_ratio: params.indexRatio ?? 0.5,
          protect_voiceless: params.protectVoiceless ?? 0.33,
          l2_normalize: false,
        },
      });
      await invoke("save_temp_audio", {
        samples: result.audio,
        sampleRate: result.sample_rate,
        outputPath,
      });
      outputData.set(0, outputPath);
      break;
    }

    case "sovits": {
      const outputPath = `${cacheDir}/${nodeId}_sovits.wav`;
      const result = await invoke<{ audio: number[]; sample_rate: number }>("run_sovits", {
        voiceName: params.voiceName ?? "default",
        modelPath: params.modelPath ?? "",
        audioPath: primaryInput,
        options: {
          f0_shift: params.pitchShift ?? 0,
          speaker_id: null,
          index_ratio: 0,
          protect_voiceless: 0.33,
          l2_normalize: false,
        },
        shallowDiffusion: params.shallowDiffusion ?? false,
      });
      await invoke("save_temp_audio", {
        samples: result.audio,
        sampleRate: result.sample_rate,
        outputPath,
      });
      outputData.set(0, outputPath);
      break;
    }

    case "msstSeparation": {
      const config = {
        audioPath: primaryInput,
        modelPath: (params.modelPath as string) ?? (params.modelName as string) ?? "",
        outputDir: cacheDir,
        device: (params.device as string) ?? "cpu",
      };
      await invoke("run_msst_separation", { config });
      let status = await invoke<{ state: string | Record<string, string>; stems?: { label: string; path: string }[] }>("get_separation_status");
      const pollStart = Date.now();
      const POLL_TIMEOUT = 10 * 60 * 1000;
      while (typeof status.state === "string" && status.state !== "Completed" && status.state !== "Idle") {
        if (useWorkflowStore.getState().isCancelled(segmentId)) {
          await invoke("cancel_separation").catch(() => {});
          // Wait briefly to see if it already completed
          await new Promise((r) => setTimeout(r, 1000));
          status = await invoke("get_separation_status");
          if (status.state === "Completed") break;
          throw new Error("Cancelled");
        }
        if (Date.now() - pollStart > POLL_TIMEOUT) {
          throw new Error("MSST separation timed out");
        }
        await new Promise((r) => setTimeout(r, 500));
        status = await invoke("get_separation_status");
      }
      if (typeof status.state === "object") {
        const errMsg = (status.state as Record<string, string>).Error ?? "MSST separation failed";
        throw new Error(errMsg);
      }
      if (status.state !== "Completed") {
        throw new Error(`MSST separation ended unexpectedly: ${JSON.stringify(status.state)}`);
      }
      if (status.stems) {
        for (let i = 0; i < status.stems.length; i++) {
          outputData.set(i, status.stems[i]!.path);
        }
      }
      break;
    }

    case "pitchShift":
    case "formantShift":
    case "audioEnhance": {
      const effectsList = Array.isArray(params.effects)
        ? (params.effects as Array<{ type: string; params: Record<string, unknown> }>).map(
            (fx) => buildEffect(fx.type === "enhance" ? "audioEnhance" : fx.type, fx.params),
          )
        : [buildEffect(nodeType, params)];

      if (effectsList.length === 0) {
        outputData.set(0, primaryInput);
        break;
      }

      const outputPath = `${cacheDir}/${nodeId}_fx.wav`;
      await invoke("process_effects", {
        request: {
          audio_path: primaryInput,
          effects: effectsList,
          output_path: outputPath,
        },
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

async function collectOutputs(
  graph: ReturnType<typeof parseWorkflowGraph>,
  dataMap: Map<string, Map<number, string>>,
): Promise<ProcessedOutput[]> {
  const results: ProcessedOutput[] = [];
  for (const outId of graph.outputNodeIds) {
    const gn = graph.nodes.get(outId)!;
    const laneLabel = (gn.node.params as Record<string, unknown>).laneLabel as string ?? "Main";

    for (const edge of gn.inEdges) {
      const upstream = dataMap.get(edge.fromNode);
      const audioPath = upstream?.get(edge.fromPort);
      if (!audioPath) continue;

      const info = await invoke<AudioFileInfo>("load_audio_file", { path: audioPath });
      results.push({
        laneLabel,
        audioPath,
        totalDurationMs: info.duration_ms,
        waveformPeaks: info.peaks,
      });
    }
  }
  return results;
}

function buildEffect(
  nodeType: string,
  params: Record<string, unknown>,
): Record<string, unknown> {
  switch (nodeType) {
    case "pitchShift":
      return { PitchShift: { semitones: params.semitones ?? 0, vocoder: params.vocoder ?? "World" } };
    case "formantShift":
      return { FormantShift: { ratio: params.ratio ?? 1.0, vocoder: params.vocoder ?? "World" } };
    case "audioEnhance":
      return { AudioEnhance: {} };
    default:
      return {};
  }
}
