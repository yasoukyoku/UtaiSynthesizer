import { useEffect, useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../../store/app";
import { useMsstModelStore } from "../../store/msst-models";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { useWorkflowStore } from "../../store/workflow";
import { useTrainingStore } from "../../store/training";
import { useVoiceModelStore } from "../../store/voice-models";
import { applyMirror, applyGhMirror, hfBaseForMirror } from "../../lib/models/msst-catalog";
import { stretchedArtifactPaths, stretchInFlight } from "../../lib/audio/stretchCache";
import { clipboardReferencedPaths } from "../../lib/clipboard";
import { historyReferencedAudioPaths } from "../../store/history";
import { backendErrorMessage, isCancelError } from "../../lib/backendError";
import { autoUpdateCheckEnabled, setAutoUpdateCheckEnabled, checkForUpdate, type UpdateInfo } from "../../lib/update";
import { startupComponentCheckEnabled, setStartupComponentCheckEnabled } from "../../lib/startupCheck";
import { useFloatingPanel } from "../../lib/useFloatingPanel";
import { PanelResizeHandles } from "./PanelResizeHandles";
import "./Settings.css";

interface GpuAdapter {
  name: string;
  vendor: string; // "nvidia" | "amd" | "intel" | "other"
}

interface HardwareInfo {
  gpu_name: string;
  cuda_available: boolean;
  directml_available: boolean;
  current_device: string;
  gpus: GpuAdapter[];
  recommended_variant: string;
}

interface CudaProgress {
  stage: string;
  progress: number;
  message: string;
  /** S64c structured i18n: stable code + proper-noun label; message stays the raw-English fallback. */
  code?: string | null;
  label?: string | null;
}

/** Mirror of Rust `pyenv::PackStatus` (meta flattened) — list_runtime_packs entry. */
interface RuntimePack {
  id: string;
  variant: string;
  label: string;
  python: string;
  torch: string;
  disk_bytes: number;
  path: string;
  envtest?: { overall?: string } | null;
}

interface RuntimeCatalogItem {
  id: string;
  variant: string;
  label: string;
  download_bytes: number;
  disk_bytes: number;
  experimental: boolean;
  downloadable: boolean;
  installed: boolean;
  /** Whether THIS machine's hardware can run the variant (backend variant_supported):
   *  CPU always; NVIDIA needs an sm_75+ card; AMD/Intel need the matching-vendor GPU.
   *  Unsupported variants are hidden from the download list (local-file install still works). */
  supported: boolean;
}

interface RuntimeEnvInfo {
  root: string;
  root_ascii_ok: boolean;
  packs: RuntimePack[];
  catalog: RuntimeCatalogItem[];
  /** Backend busy flags — the component rebuilds its state from these on (re)mount,
   *  so closing/reopening the panel mid-install keeps the progress + cancel UI alive. */
  installing: boolean;
  envtest_running: boolean;
}

interface PyenvProgress {
  id: string;
  phase: string;
  progress: number;
  /** English log/fallback text — render from `code`+`params` when present. */
  message: string;
  /** Stable stage/outcome/error CODE (STAGE_* / INSTALL_DONE / …) for localization. */
  code?: string | null;
  /** Positional payload for the localized template (e.g. [name, doneMB, totalMB]). */
  params?: string[];
}

/** Localize a backend rejection via the app-wide CODE map, falling back to the raw text. */
const backendErrText = (e: unknown): string => backendErrorMessage(e) ?? String(e);

/** Mirror of Rust `download::ProbeResult` — download-source throughput probe. */
interface ProbeResult {
  reachable: boolean;
  verdict: string; // ok | slow | throttled | http_error | unreachable
  mbps: number;
  ttfb_ms: number;
  bytes: number;
  http_status?: number | null;
  error?: string | null;
}

// A real published asset used only as the probe target (its host = the source being
// tested). ~236 MB file → Range-GET of the first few MB measures real throughput.
const PROBE_ASSET = "https://huggingface.co/datasets/yasoukyoku/utai-runtimes/resolve/main/runtime-cpu-v1.tar.zst";

// GH-mirror probe target — the GAME zip's GitHub release URL, same asset as Rust
// GAME_SOURCES[0] (src-tauri/src/commands/midi_extract.rs). 179 MB real asset; the
// probe only Range-GETs the first ~4 MB.
const GH_PROBE_ASSET = "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-medium-onnx.zip";

/** Run the backend throughput probe against `url`, funneling invoke failures into a
 *  ProbeResult row — ONE funnel shared by the HF and GH source tests. */
async function runSrcProbe(url: string): Promise<ProbeResult> {
  try {
    return await invoke<ProbeResult>("test_download_source", { url });
  } catch (e) {
    return { reachable: false, verdict: "unreachable", mbps: 0, ttfb_ms: 0, bytes: 0, error: String(e) };
  }
}

const fmtGB = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
/** Sub-MB friendly size (the report has KB-scale rows like logs/dictionaries). */
const fmtSize = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(0, Math.round(b / 1e3))} KB`);

/** Mirror of Rust `commands::assets::{AssetPackStatus, AssetPackProgress}` (S64). */
interface AssetPackStatus {
  id: string;
  fileCount: number;
  missing: number;
  totalBytes: number;
  missingBytes: number;
  downloading: boolean;
}
interface AssetPackProgress {
  pack: string;
  stage: string; // "download" | "done" | "failed" | "cancelled"
  file: string;
  fileIndex: number;
  fileCount: number;
  downloaded: number;
  total: number;
  error: string | null;
}

/** Mirror of Rust `commands::storage::{StorageReport, WorkspaceUsage}` (S61). */
interface WorkspaceUsage {
  slug: string;
  name: string;
  family: string;
  bytes: number;
  has_pool: boolean;
}
interface StorageReport {
  data_dir: string;
  cache_bytes: number;
  models_bytes: number;
  msst_bytes: number;
  runtimes_bytes: number;
  dictionaries_bytes: number;
  logs_bytes: number;
  audition_bytes: number;
  training_bytes: number;
  workspaces: WorkspaceUsage[];
}

/** Everything the OPEN project still references inside the cache tree — passed to
 *  cleanup_render_cache so the sweep can't break the current session: clip sources
 *  (audio_cache copies), deposited lane audio (run dirs), decode playback paths, and the
 *  runtime node-output cache (single-node re-runs read these paths). */
function collectProtectedPaths(): string[] {
  const prot = new Set<string>();
  for (const t of useProjectStore.getState().tracks) {
    for (const s of t.segments) {
      if (s.content.type === "audioClip") prot.add(s.content.sourcePath);
      for (const o of s.processedOutputs ?? []) prot.add(o.audioPath);
    }
  }
  for (const [p, info] of Object.entries(useAudioStore.getState().audioFiles)) {
    prot.add(p);
    if (info.playbackPath) prot.add(info.playbackPath);
  }
  for (const perSeg of Object.values(useWorkflowStore.getState().nodeOutputs)) {
    for (const paths of Object.values(perSeg)) {
      for (const pp of paths) if (pp) prot.add(pp);
    }
  }
  // Stretched artifacts the session has already resolved: the stretchCache memo would keep
  // serving a deleted path (no existence re-check) → stretched clips dead until restart.
  for (const p of stretchedArtifactPaths()) prot.add(p);
  // UNDO/REDO snapshots + the arrangement clipboard also hold live references (audit S61 MAJOR):
  // deleting a cut/deleted segment's stem lets a later paste/undo stamp a bake "valid" over a
  // missing file — permanently silent false-clean. Protect everything they can resurrect.
  for (const p of historyReferencedAudioPaths()) prot.add(p);
  for (const p of clipboardReferencedPaths()) prot.add(p);
  return [...prot];
}

export function Settings({ onClose }: { onClose: () => void }) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const { style: panelStyle, startDrag, startResize } = useFloatingPanel({
    storageKey: "utai.settingsRect",
    initial: () => ({ x: 72, y: 84, w: 340, h: Math.round(window.innerHeight * 0.7) }),
    minW: 300,
    minH: 280,
  });
  // Download-source preference — the store is shared with the resource manager
  // (which consumes `mirror` for model downloads); the CONFIG UI now lives here.
  const mirror = useMsstModelStore((s) => s.mirror);
  const setMirror = useMsstModelStore((s) => s.setMirror);
  const [srcTest, setSrcTest] = useState<ProbeResult | null>(null);
  const [srcTesting, setSrcTesting] = useState(false);
  const handleSrcTest = useCallback(async () => {
    setSrcTesting(true);
    setSrcTest(null);
    // test the SELECTED source's host by pulling a few real MB through it — a
    // ping/HEAD would pass the GFW's small-packet allowance and false-positive.
    setSrcTest(await runSrcProbe(applyMirror(PROBE_ASSET, mirror)));
    setSrcTesting(false);
  }, [mirror]);
  // a stale verdict must not linger next to a different, untested source
  useEffect(() => {
    setSrcTest(null);
  }, [mirror.type, mirror.customUrl]);
  // GitHub mirror — its own selection + probe state, fully independent of the HF
  // test above so the two verdicts can never cross-talk. S66: presets are data
  // (remote-refreshable); the selected id falls back to the list head when the
  // remote list dropped it (mirrors ghProxyPrefix's resolution).
  const ghMirror = useMsstModelStore((s) => s.ghMirror);
  const setGhMirror = useMsstModelStore((s) => s.setGhMirror);
  const ghPresets = useMsstModelStore((s) => s.ghPresets);
  const refreshGhPresets = useMsstModelStore((s) => s.refreshGhPresets);
  useEffect(() => { void refreshGhPresets(); }, [refreshGhPresets]);
  const ghEffectivePresetId = ghPresets.find((p) => p.id === ghMirror.presetId)?.id ?? ghPresets[0]?.id;
  const [ghSrcTest, setGhSrcTest] = useState<ProbeResult | null>(null);
  const [ghSrcTesting, setGhSrcTesting] = useState(false);
  const handleGhSrcTest = useCallback(async () => {
    setGhSrcTesting(true);
    setGhSrcTest(null);
    setGhSrcTest(await runSrcProbe(applyGhMirror(GH_PROBE_ASSET, ghMirror, ghPresets)));
    setGhSrcTesting(false);
  }, [ghMirror, ghPresets]);
  useEffect(() => {
    setGhSrcTest(null);
    // ghEffectivePresetId included (review S66): a remote preset-list refresh can change which
    // proxy the SAME persisted choice resolves to — the old verdict would describe a different host.
  }, [ghMirror.type, ghMirror.presetId, ghMirror.customUrl, ghEffectivePresetId]);
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [device, setDevice] = useState("auto");
  const [saving, setSaving] = useState(false);
  const [cudaReady, setCudaReady] = useState(false);
  const [cudaDownloading, setCudaDownloading] = useState(false);
  const [cudaProgress, setCudaProgress] = useState<CudaProgress | null>(null);
  const [cudaError, setCudaError] = useState<string | null>(null);
  const [cudaJustInstalled, setCudaJustInstalled] = useState(false);
  // S66: CUDA arena cap (MB; 0/blank = unlimited). Text-state so typing stays free; the
  // commit round-trips through Rust (persist + GPU-session eviction) and re-reads the truth.
  const [cudaMemLimitText, setCudaMemLimitText] = useState("");
  useEffect(() => {
    invoke<number>("get_cuda_mem_limit")
      .then((mb) => setCudaMemLimitText(mb > 0 ? String(mb) : ""))
      .catch(() => {});
  }, []);
  const commitCudaMemLimit = useCallback(async () => {
    const mb = Math.max(0, parseInt(cudaMemLimitText || "0", 10) || 0);
    try {
      await invoke("set_cuda_mem_limit", { mb });
      setCudaMemLimitText(mb > 0 ? String(mb) : "");
    } catch {
      /* leave the text; the next successful commit heals it */
    }
  }, [cudaMemLimitText]);

  // S66: on-disk layout for the panel (copyable paths for inspection/support) + per-lane
  // presence + the exact filenames the local-install picker expects.
  const [cudaPaths, setCudaPaths] = useState<{ ortDir: string; dllDir: string; missing: string[]; expectedFiles: string[] } | null>(null);
  const refreshCudaPaths = useCallback(() => {
    invoke<{ ortDir: string; dllDir: string; missing: string[]; expectedFiles: string[] }>("cuda_runtime_paths")
      .then(setCudaPaths)
      .catch(() => {});
  }, []);
  const [dataDir, setDataDir] = useState("");
  // S64: configured-but-missing data dir recovered at startup (recreated empty / fell back) — the
  // persistent surface for the startup toast (App.tsx), so the cause stays findable after 5s.
  const [dataDirIssue, setDataDirIssue] = useState<{ configured: string; effective: string; fell_back: boolean } | null>(null);
  const [relocating, setRelocating] = useState(false);
  const [relocateMsg, setRelocateMsg] = useState<string | null>(null);
  const showConfirm = useAppStore((s) => s.showConfirm);
  const [rt, setRt] = useState<RuntimeEnvInfo | null>(null);
  const [rtBusy, setRtBusy] = useState(false);
  const [rtProgress, setRtProgress] = useState<PyenvProgress | null>(null);
  const [rtError, setRtError] = useState<string | null>(null);
  // Terminal "done" payload — kept whole so the render maps code+params to L() text
  // and the success (green) state keys on the stable INSTALL_DONE code.
  const [rtNotice, setRtNotice] = useState<PyenvProgress | null>(null);
  const [envtesting, setEnvtesting] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const refreshRuntime = useCallback(() => {
    invoke<RuntimeEnvInfo>("get_runtime_env_info")
      .then((info) => {
        setRt(info);
        // Rebuild busy state from the backend (panel may have been closed and
        // reopened mid-install — component state alone would strand the cancel
        // button and mislabel every other button as available).
        setRtBusy(info.installing);
        setEnvtesting((prev) => {
          if (info.envtest_running) return prev ?? "__backend__";
          return prev === "__backend__" ? null : prev;
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    invoke<HardwareInfo>("get_hardware_info").then(setHw).catch(() => {});
    invoke<string>("get_device_preference").then(setDevice).catch(() => {});
    invoke<boolean>("is_cuda_runtime_ready").then(setCudaReady).catch(() => {});
    // Re-latch onto an in-flight CUDA download after a panel remount (S64c audit: `cudaDownloading`
    // is component-local; the backend refcount is the truth, and the busy interlock rejects a
    // second start anyway — this keeps the button/progress honest).
    invoke<string[]>("running_tasks")
      .then((ts) => { if (ts.includes("cuda_download")) setCudaDownloading(true); })
      .catch(() => {});
    invoke<string>("get_data_dir").then(setDataDir).catch(() => {});
    invoke<{ configured: string; effective: string; fell_back: boolean } | null>("get_data_dir_issue")
      .then(setDataDirIssue)
      .catch(() => {});
    refreshRuntime();
    refreshCudaPaths();
  }, [refreshRuntime, refreshCudaPaths]);

  useEffect(() => {
    const unlisten = listen<PyenvProgress>("pyenv-progress", (e) => {
      setRtProgress(e.payload);
      if (e.payload.phase === "done" || e.payload.phase === "error") {
        setRtBusy(false);
        if (e.payload.phase === "error") setRtError(backendErrText(e.payload.message));
        // The done payload can carry a REAL verdict (INSTALLED_ENVTEST_FAILED: …) —
        // it must survive the progress bar disappearing, not vanish with it.
        if (e.payload.phase === "done") setRtNotice(e.payload);
        refreshRuntime();
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, [refreshRuntime]);

  useEffect(() => {
    // Envtest lifecycle channel: the final {type:"done"} is the ONLY signal a
    // backend-started (or another-instance-started) self-test has ended — without
    // this, a panel that entered the "__backend__" sentinel could never leave it.
    const unlisten = listen<{ id: string; event: { type?: string } }>("pyenv-envtest", (e) => {
      if (e.payload?.event?.type === "done") refreshRuntime();
    });
    return () => { unlisten.then((f) => f()); };
  }, [refreshRuntime]);

  const handleRtDownload = useCallback(async (id: string) => {
    setRtBusy(true);
    setRtError(null);
    setRtNotice(null);
    setRtProgress(null);
    try {
      await invoke("download_runtime_pack", { id });
    } catch (e) {
      setRtError(backendErrText(e));
    } finally {
      setRtBusy(false);
      refreshRuntime();
    }
  }, [refreshRuntime]);

  const handleRtLocalInstall = useCallback(async () => {
    const file = await open({
      multiple: false,
      title: "runtime pack (.tar.zst / .part01)",
      filters: [{ name: "Runtime pack", extensions: ["zst", "part01"] }],
    });
    if (!file || typeof file !== "string") return;
    setRtBusy(true);
    setRtError(null);
    setRtNotice(null);
    setRtProgress(null);
    try {
      await invoke("install_runtime_pack_local", { path: file });
    } catch (e) {
      setRtError(backendErrText(e));
    } finally {
      setRtBusy(false);
      refreshRuntime();
    }
  }, [refreshRuntime]);

  const handleRtEnvtest = useCallback(async (id: string) => {
    setEnvtesting(id);
    setRtError(null);
    setRtNotice(null);
    try {
      await invoke("run_pack_envtest", { id });
    } catch (e) {
      setRtError(backendErrText(e));
    } finally {
      setEnvtesting(null);
      refreshRuntime();
    }
  }, [refreshRuntime]);

  const handleRtDelete = useCallback(async (id: string) => {
    const choice = await showConfirm({
      title: L("rtDeleteTitle"),
      body: `${id}\n${L("rtDeleteBody")}`,
      buttons: [
        { id: "cancel", label: L("rtCancelBtn") },
        { id: "del", label: L("rtDelete"), kind: "danger" },
      ],
    });
    if (choice !== "del") return;
    // Visible busy state for the whole removal (a 1 GB tree takes seconds — with
    // no feedback users assume a hang and click again).
    setDeleting(id);
    setRtError(null);
    setRtNotice(null);
    try {
      await invoke("delete_runtime_pack", { id });
    } catch (e) {
      setRtError(backendErrText(e));
    } finally {
      setDeleting(null);
      refreshRuntime();
    }
  }, [refreshRuntime, showConfirm, lang]);

  const handleRelocate = useCallback(async () => {
    const dir = await open({ directory: true, multiple: false, title: "Choose data directory" });
    if (!dir || typeof dir !== "string") return;
    setRelocating(true);
    setRelocateMsg(null);
    try {
      await invoke("migrate_data_dir", { newDir: dir });
      setDataDir(dir);
      setRelocateMsg("migrated");
    } catch (e) {
      setRelocateMsg(String(e).includes("TRAINING_ACTIVE") ? "error: training" : `error: ${e}`);
    } finally {
      setRelocating(false);
    }
  }, []);

  // ── S61 storage usage + cleanup ──
  const [storage, setStorage] = useState<StorageReport | null>(null);
  const [storageScanning, setStorageScanning] = useState(false);
  const [cleanBusy, setCleanBusy] = useState<string | null>(null); // "cache"|"audition"|"logs"|<slug>
  const [cleanMsg, setCleanMsg] = useState<string | null>(null);
  // Live gates: never sweep the render cache while anything might be mid-write into it.
  const isPlaying = useAudioStore((s) => s.isPlaying);
  const vocalRenderActive = useAppStore((s) => s.vocalRenderActive);
  const anyWorkflowRunning = useWorkflowStore((s) => Object.values(s.executions).some((e) => e.status === "running"));
  const trainingBusy = useTrainingStore((s) => s.snapshot.state === "running" || s.snapshot.state === "starting");
  const midiExtracting = useAppStore((s) => Object.keys(s.midiExtracting).length > 0);
  const rangeTesting = useVoiceModelStore((s) => Object.keys(s.rangeTesting).length > 0);
  const decoding = useAudioStore((s) => s.loadingPaths.length > 0); // in-flight decode writes audio_cache
  const cacheCleanBlocked = isPlaying || vocalRenderActive || anyWorkflowRunning || midiExtracting || rangeTesting || decoding;

  const refreshStorage = useCallback(async () => {
    setStorageScanning(true);
    try {
      setStorage(await invoke<StorageReport>("get_storage_report"));
    } catch (e) {
      setCleanMsg(String(e));
    } finally {
      setStorageScanning(false);
    }
  }, []);
  useEffect(() => { void refreshStorage(); }, [refreshStorage]);

  const cleanupErrText = (e: unknown): string => {
    const msg = String(e);
    if (msg.includes("TRAINING_ACTIVE")) return L("stErrTraining");
    if (msg.includes("CLEANUP_BUSY")) return L("stErrBusy");
    if (msg.includes("WORKSPACE_MISSING")) return L("stErrWsMissing");
    // Any other storage code (WORKSPACE_DELETE_FAILED / STORAGE_JOIN / …) → the app-wide map,
    // raw-string fallback.
    return backendErrorMessage(msg) ?? msg;
  };

  const runCleanup = useCallback(async (key: string, fn: () => Promise<number>, confirm?: { title: string; body: string }) => {
    if (confirm) {
      const choice = await showConfirm({
        title: confirm.title,
        body: confirm.body,
        buttons: [
          { id: "cancel", label: L("rtCancelBtn") },
          { id: "clean", label: L("stCleanBtn"), kind: "danger" },
        ],
      });
      if (choice !== "clean") return;
    }
    setCleanBusy(key);
    setCleanMsg(null);
    try {
      const freed = await fn();
      setCleanMsg(`${L("stFreed")} ${fmtSize(freed)}`);
      await refreshStorage();
    } catch (e) {
      setCleanMsg(cleanupErrText(e));
    } finally {
      setCleanBusy(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshStorage, showConfirm, lang]);

  const handleCleanCache = useCallback(() => {
    // Non-reactive last-moment gate: an in-flight stretch's output path is minted Rust-side and
    // can't be protected — refuse instead of racing it (audit S61).
    if (stretchInFlight()) {
      setCleanMsg(L("stErrBusy"));
      return;
    }
    void runCleanup(
      "cache",
      () => invoke<number>("cleanup_render_cache", { protected: collectProtectedPaths() }),
      { title: L("stCacheTitle"), body: L("stCacheBody") },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runCleanup, lang]);

  const handleCleanAudition = useCallback(() => {
    void runCleanup("audition", () => invoke<number>("cleanup_audition_caches"), {
      title: L("stAuditionTitle"),
      body: L("stAuditionBody"),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runCleanup, lang]);

  const handleCleanLogs = useCallback(() => {
    void runCleanup("logs", () => invoke<number>("cleanup_logs"));
  }, [runCleanup]);

  const handleDeleteWorkspace = useCallback((ws: WorkspaceUsage) => {
    void runCleanup(
      ws.slug,
      async () => {
        const freed = await invoke<number>("delete_training_workspace", { slug: ws.slug });
        // Keep the training page coherent: the diff card's cached workspace facts (免导入直训 /
        // 续训 hints) must reflect the deletion immediately — re-probe the CURRENT name.
        const ts = useTrainingStore.getState();
        const curName = ts.config.modelName.trim();
        if (curName) {
          invoke("get_training_workspace_info", { name: curName })
            .then((info) => ts.setDiffWsInfo(info as never))
            .catch(() => {});
        }
        return freed;
      },
      {
        title: L("stWsTitle"),
        body: `${ws.name}${ws.family ? ` · ${ws.family}` : ""} · ${fmtSize(ws.bytes)}\n${L("stWsBody")}${ws.has_pool ? `\n${L("stWsPoolNote")}` : ""}`,
      },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runCleanup, lang]);

  useEffect(() => {
    const unlisten = listen<CudaProgress>("cuda-download-progress", (e) => {
      setCudaProgress(e.payload);
      if (e.payload.stage === "done") {
        setCudaDownloading(false);
        // Re-query the REAL readiness instead of optimistically flipping true — is_cuda_runtime_ready
        // also verifies the full provider dependency set, and an optimistic true that reverts on the
        // next restart reads as "CUDA disappeared" (S64b beta report). The hardware badge re-queries
        // too: the in-session setup_cuda_dll_paths re-run can flip cuda_available without a restart.
        // The green "restart to activate" note keys on the SAME real verdict (review S66: a partial
        // local install must not read as ready).
        invoke<boolean>("is_cuda_runtime_ready")
          .then((r) => {
            setCudaReady(r);
            setCudaJustInstalled(r);
          })
          .catch(() => {});
        invoke<HardwareInfo>("get_hardware_info").then(setHw).catch(() => {});
        invoke<{ ortDir: string; dllDir: string; missing: string[]; expectedFiles: string[] }>("cuda_runtime_paths")
          .then(setCudaPaths)
          .catch(() => {});
      } else if (e.payload.stage === "error") {
        // Terminal failure/cancel (review S66): without this, a late buffered progress event
        // re-latched cudaDownloading=true forever after a cancel.
        setCudaDownloading(false);
        invoke<{ ortDir: string; dllDir: string; missing: string[]; expectedFiles: string[] }>("cuda_runtime_paths")
          .then(setCudaPaths)
          .catch(() => {});
      } else {
        // Any non-terminal event re-latches a remounted panel onto the running download.
        setCudaDownloading(true);
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  const handleLangChange = (value: string) => {
    i18n.changeLanguage(value);
    try { localStorage.setItem("lang", value); } catch { /* ignore */ }
  };

  // S64 — version & update-check section state. The dialog itself (progress/install) is the app-level
  // UpdateDialog; this section only runs the CHECK and reports its outcome inline.
  const [appVersion, setAppVersion] = useState("");
  useEffect(() => { void getVersion().then(setAppVersion).catch(() => {}); }, []);
  const [autoCheck, setAutoCheck] = useState(autoUpdateCheckEnabled);
  const [updChecking, setUpdChecking] = useState(false);
  const [updResult, setUpdResult] = useState<{ kind: "latest" } | { kind: "found"; info: UpdateInfo } | { kind: "error"; msg: string } | null>(null);

  const handleAutoCheckToggle = (v: boolean) => {
    setAutoCheck(v);
    setAutoUpdateCheckEnabled(v);
  };

  // S66: startup missing-component check toggle (the dialog's "不再提醒" writes the same key).
  const [startupCompCheck, setStartupCompCheck] = useState(startupComponentCheckEnabled);
  const handleStartupCompToggle = (v: boolean) => {
    setStartupCompCheck(v);
    setStartupComponentCheckEnabled(v);
  };

  const handleUpdateCheck = useCallback(async () => {
    setUpdChecking(true);
    setUpdResult(null);
    try {
      const info = await checkForUpdate();
      if (info) {
        setUpdResult({ kind: "found", info });
        useAppStore.getState().openUpdateDialog(info);
      } else {
        setUpdResult({ kind: "latest" });
      }
    } catch (e) {
      setUpdResult({ kind: "error", msg: backendErrorMessage(e) ?? String(e) });
    } finally {
      setUpdChecking(false);
    }
  }, []);

  // S64 — model-asset packs (assets.rs): aux inference models + training bases, HF-hosted.
  const [assetPacks, setAssetPacks] = useState<AssetPackStatus[]>([]);
  const [assetActive, setAssetActive] = useState<string | null>(null);
  const [assetProgress, setAssetProgress] = useState<AssetPackProgress | null>(null);
  const [assetMsg, setAssetMsg] = useState<string | null>(null);
  const refreshAssets = useCallback(() => {
    invoke<AssetPackStatus[]>("asset_pack_status").then(setAssetPacks).catch(() => {});
  }, []);
  useEffect(() => { refreshAssets(); }, [refreshAssets]);
  useEffect(() => {
    // Backend events are the progress truth — a remounted panel re-attaches seamlessly (the
    // pyenv/GAME pattern; per-pack `downloading` in asset_pack_status covers the between-events
    // gap). ANY non-"download" stage is terminal: done, cancelled (silent, the app-wide cancel
    // convention) or failed (localized error shown) — the backend always emits one (audit S64).
    const un = listen<AssetPackProgress>("asset-pack-progress", (e) => {
      if (e.payload.stage !== "download") {
        setAssetProgress(null);
        setAssetActive(null);
        if (e.payload.stage === "failed" && e.payload.error) {
          setAssetMsg(backendErrorMessage(e.payload.error) ?? e.payload.error);
        }
        refreshAssets();
      } else {
        setAssetProgress(e.payload);
        setAssetActive(e.payload.pack);
      }
    });
    return () => { un.then((f) => f()); };
  }, [refreshAssets]);

  const handleAssetDownload = useCallback(async (id: string) => {
    setAssetMsg(null);
    setAssetActive(id);
    try {
      // The chosen HF source rides along as a host-replacement base and is tried FIRST (Rust
      // dedupes it out of the fixed huggingface.co → hf-mirror.com rotation) — the hf-mirror
      // preset must reorder the rotation, not be silently ignored (audit S64).
      await invoke("download_asset_pack", { id, hfBase: hfBaseForMirror(mirror) });
    } catch (e) {
      if (!isCancelError(e)) setAssetMsg(backendErrorMessage(e) ?? String(e));
    } finally {
      setAssetActive(null);
      setAssetProgress(null);
      refreshAssets();
    }
  }, [mirror, refreshAssets]);

  const handleDeviceChange = useCallback(async (value: string) => {
    setDevice(value);
    setSaving(true);
    try {
      await invoke("set_device_preference", { device: value });
    } catch (e) {
      console.error("Failed to save device preference:", e);
    }
    setSaving(false);
  }, []);

  // S64c structured progress → localized line (code + proper-noun label; raw message = fallback,
  // the pyenv pgDownloading/pgExtracting pattern).
  const cudaProgressText = (p: CudaProgress): string => {
    const map: Record<string, string> = {
      CUDA_DL_DOWNLOADING: L("cudaPgDownloading"),
      CUDA_DL_EXTRACTING: L("cudaPgExtracting"),
      CUDA_DL_SKIP: L("cudaPgSkip"),
      CUDA_DL_FINALIZING: L("cudaPgFinalizing"),
      CUDA_DL_DONE: L("cudaPgDone"),
      CUDA_DL_LOCAL_PARTIAL: L("cudaPgLocalPartial"),
      CUDA_DL_CANCELLED: L("cudaPgCancelled"),
      CUDA_DL_FAILED: L("cudaPgFailed"),
    };
    const base = p.code ? map[p.code] : undefined;
    if (!base) return p.message;
    return p.label ? `${base} · ${p.label}` : base;
  };

  const handleCudaDownload = useCallback(async () => {
    setCudaDownloading(true);
    setCudaError(null);
    setCudaProgress({ stage: "start", progress: 0, message: "Starting..." });
    try {
      // preferCnMirrors: the HF-source choice doubles as the "I'm in mainland China" signal —
      // it reorders the CUDA rotation (Chinese PyPI mirrors / hf-mirror first). S66.
      await invoke("download_cuda_runtime", { preferCnMirrors: mirror.type === "hf-mirror" });
    } catch (e) {
      if (isCancelError(e)) {
        // user cancelled — resumable, not an error (every .part is kept)
        setCudaDownloading(false);
        setCudaProgress(null);
        return;
      }
      // CUDA_GPU_REQUIRED etc. → localized via the shared mapper (raw fallback for oddballs).
      setCudaError(backendErrorMessage(e) ?? String(e));
      setCudaDownloading(false);
    }
  }, [mirror.type]);

  // S66 install-from-local-file: the user picks the wheels/nupkg listed in the note below
  // (exact filenames from cuda_runtime_paths.expectedFiles) — the offline escape hatch.
  const handleCudaLocalInstall = useCallback(async () => {
    const picked = await open({
      multiple: true,
      filters: [{ name: "CUDA runtime files", extensions: ["whl", "nupkg", "zip"] }],
    });
    if (!picked) return;
    const paths = Array.isArray(picked) ? picked : [picked];
    setCudaError(null);
    setCudaDownloading(true);
    setCudaProgress({ stage: "local", progress: 0, message: "Installing from local files..." });
    try {
      await invoke("install_cuda_runtime_local", { paths });
    } catch (e) {
      setCudaError(backendErrorMessage(e) ?? String(e));
    } finally {
      setCudaDownloading(false);
      invoke<boolean>("is_cuda_runtime_ready").then(setCudaReady).catch(() => {});
      invoke<HardwareInfo>("get_hardware_info").then(setHw).catch(() => {});
      refreshCudaPaths();
    }
  }, [refreshCudaPaths]);

  const L = (key: string) => {
    const map: Record<string, Record<string, string>> = {
      title: { zh: "设置", en: "Settings", ja: "設定" },
      language: { zh: "界面语言", en: "Language", ja: "表示言語" },
      verTitle: { zh: "版本与更新", en: "Version & Updates", ja: "バージョンと更新" },
      verCurrent: { zh: "当前版本", en: "Current version", ja: "現在のバージョン" },
      verCheckBtn: { zh: "检查更新", en: "Check for Updates", ja: "更新を確認" },
      verChecking: { zh: "检查中…", en: "Checking…", ja: "確認中…" },
      verLatest: { zh: "已是最新版本", en: "You're on the latest version", ja: "最新バージョンです" },
      verFound: { zh: "发现新版本", en: "New version available", ja: "新しいバージョンがあります" },
      verView: { zh: "查看", en: "View", ja: "表示" },
      verAutoCheck: { zh: "启动时自动检查更新", en: "Check for updates on startup", ja: "起動時に更新を確認" },
      assetTitle: { zh: "模型资产", en: "Model Assets", ja: "モデルアセット" },
      assetAux: { zh: "推理核心模型包", en: "Core inference models", ja: "推論コアモデル" },
      assetRvc: { zh: "RVC 训练底模", en: "RVC training base models", ja: "RVC 学習ベースモデル" },
      assetSovits: { zh: "SoVITS 训练底模", en: "SoVITS training base models", ja: "SoVITS 学習ベースモデル" },
      assetSovitsV2: { zh: "SoVITS 4.0-v2 训练底模", en: "SoVITS 4.0-v2 training base models", ja: "SoVITS 4.0-v2 学習ベースモデル" },
      assetInstalled: { zh: "已安装", en: "Installed", ja: "インストール済み" },
      assetMissing: { zh: "缺失", en: "Missing", ja: "不足" },
      assetDownload: { zh: "下载", en: "Download", ja: "ダウンロード" },
      assetNote: { zh: "推理核心包 = 人声轨合成 / 翻唱 / 音高提取的必备模型（约 1.4GB）；训练底模按需下载。已在下载中断处自动续传。", en: "The core pack holds the models required for vocal-track synthesis / covers / pitch extraction (~1.4GB); training bases are optional. Interrupted downloads resume automatically.", ja: "推論コアパックはボーカルトラック合成／カバー／ピッチ抽出に必須のモデルです（約 1.4GB）。学習ベースは必要に応じて。中断したダウンロードは自動で再開されます。" },
      assetStartupCheck: { zh: "启动时检查必要组件（缺失时弹窗提示）", en: "Check required components on startup (dialog when missing)", ja: "起動時に必須コンポーネントを確認（不足時にダイアログ）" },
      hardware: { zh: "硬件", en: "Hardware", ja: "ハードウェア" },
      gpu: { zh: "显卡", en: "GPU", ja: "GPU" },
      cuda: { zh: "CUDA 可用", en: "CUDA Available", ja: "CUDA 利用可能" },
      directml: { zh: "DirectML 可用", en: "DirectML Available", ja: "DirectML 利用可能" },
      epLabel: { zh: "推理设备", en: "Inference Device", ja: "推論デバイス" },
      auto: { zh: "自动", en: "Auto", ja: "自動" },
      cudaOpt: { zh: "CUDA (NVIDIA GPU)", en: "CUDA (NVIDIA GPU)", ja: "CUDA (NVIDIA GPU)" },
      dmlOpt: { zh: "DirectML (通用 GPU)", en: "DirectML (Any GPU)", ja: "DirectML (汎用 GPU)" },
      cpuOpt: { zh: "CPU", en: "CPU", ja: "CPU" },
      saved: { zh: "已保存", en: "Saved", ja: "保存済み" },
      note: { zh: "切换设备后需要重启应用才能生效。", en: "Restart the app after changing device.", ja: "デバイス変更後、アプリを再起動してください。" },
      yes: { zh: "是", en: "Yes", ja: "はい" },
      no: { zh: "否", en: "No", ja: "いいえ" },
      cudaRuntime: { zh: "CUDA 运行时", en: "CUDA Runtime", ja: "CUDAランタイム" },
      cudaInstalled: { zh: "已就绪", en: "Ready", ja: "準備完了" },
      cudaNotInstalled: { zh: "未安装", en: "Not Installed", ja: "未インストール" },
      cudaDownload: { zh: "下载 CUDA 运行时", en: "Download CUDA Runtime", ja: "CUDAランタイムをダウンロード" },
      cudaDownloading: { zh: "下载中...", en: "Downloading...", ja: "ダウンロード中..." },
      cudaNote: { zh: "无需安装 CUDA Toolkit——自动下载全部运行库（ORT CUDA + cudart/cuBLAS/cuFFT/cuDNN，共约 1.6GB）。需要 NVIDIA 显卡和较新的驱动。", en: "No CUDA Toolkit needed — downloads the full runtime (ORT CUDA + cudart/cuBLAS/cuFFT/cuDNN, ~1.6GB total). Requires an NVIDIA GPU with a recent driver.", ja: "CUDA Toolkit のインストールは不要 — ランタイム一式（ORT CUDA + cudart/cuBLAS/cuFFT/cuDNN、合計約 1.6GB）を自動ダウンロードします。NVIDIA GPU と新しめのドライバーが必要です。" },
      cudaRestart: { zh: "下载完成，重启应用后生效。", en: "Download complete. Restart to activate.", ja: "ダウンロード完了。再起動で有効になります。" },
      cudaPgDownloading: { zh: "下载中", en: "Downloading", ja: "ダウンロード中" },
      cudaPgExtracting: { zh: "解压中", en: "Extracting", ja: "展開中" },
      cudaPgSkip: { zh: "已存在，跳过", en: "Already present — skipped", ja: "既に存在するためスキップ" },
      cudaPgFinalizing: { zh: "收尾中…", en: "Finalizing…", ja: "仕上げ中…" },
      cudaPgDone: { zh: "CUDA 运行时就绪，重启应用后生效", en: "CUDA runtime ready — restart to activate", ja: "CUDA ランタイム準備完了 — 再起動で有効になります" },
      cudaCancelDl: { zh: "取消下载", en: "Cancel download", ja: "ダウンロード中止" },
      cudaPgLocalPartial: { zh: "已安装所选文件，但仍缺组件", en: "Selected files installed — parts still missing", ja: "選択ファイルを導入しましたが、まだ不足があります" },
      cudaMemLimit: { zh: "显存上限（CUDA）", en: "VRAM limit (CUDA)", ja: "VRAM 上限（CUDA）" },
      cudaMemLimitNote: { zh: "0 或留空 = 不限制（默认）。限制的是 CUDA 显存池上限：设得过低会让大任务直接报「分配失败」而不是变慢——低显存爆显存时再按需设置（例如 6144）。仅对 CUDA 生效。", en: "0 / blank = unlimited (default). Caps the CUDA memory arena: set too low, big jobs fail with an allocation error instead of slowing down — only set it (e.g. 6144) if you actually hit VRAM exhaustion. CUDA only.", ja: "0 または空欄 = 無制限（既定）。CUDA メモリアリーナの上限です。低すぎると大きなジョブは遅くなる代わりに「割り当て失敗」になります — VRAM 不足が実際に起きる場合のみ設定してください（例: 6144）。CUDA のみ有効。" },
      cudaPgCancelled: { zh: "已取消（进度已保留，可续传）", en: "Cancelled (progress kept — resumable)", ja: "キャンセルしました（進捗は保持、再開可能）" },
      cudaPgFailed: { zh: "下载失败", en: "Download failed", ja: "ダウンロードに失敗しました" },
      cudaMissing: { zh: "缺失组件", en: "Missing parts", ja: "不足コンポーネント" },
      cudaLocalNote: { zh: "「从本地文件安装」接受以下官方文件：", en: "“Install from local file” accepts these official files:", ja: "「ローカルからインストール」は以下の公式ファイルを受け付けます：" },
      cudaDllDir: { zh: "CUDA 运行库目录", en: "CUDA DLL folder", ja: "CUDA DLL フォルダ" },
      cudaOrtDir: { zh: "ORT CUDA 目录", en: "ORT CUDA folder", ja: "ORT CUDA フォルダ" },
      storage: { zh: "存储位置", en: "Storage", ja: "保存場所" },
      stDirRecreated: { zh: "警告：配置的数据目录 {configured} 此前不存在，已重新创建（内容为空）。若数据在别处，请检查该盘/目录。", en: "Warning: the configured data folder {configured} was missing and has been recreated (empty). If your data lives elsewhere, check that drive/folder.", ja: "警告：設定されたデータフォルダ {configured} が存在しなかったため、再作成しました（空です）。データが別の場所にある場合は、そのドライブ/フォルダを確認してください。" },
      stDirFellBack: { zh: "警告：配置的数据目录 {configured} 不可用（盘符不存在？），本次改用程序旁的默认目录。恢复该盘后重启即可回到原目录。", en: "Warning: the configured data folder {configured} is unavailable (drive missing?). Using the default folder next to the program for this session; restore the drive and restart to return to it.", ja: "警告：設定されたデータフォルダ {configured} が利用できません（ドライブ未接続？）。今回はプログラム横の既定フォルダを使用します。ドライブを戻して再起動すると元に戻ります。" },
      dataDir: { zh: "数据目录（模型 + 缓存）", en: "Data folder (models + cache)", ja: "データフォルダ（モデル + キャッシュ）" },
      relocate: { zh: "更改并迁移…", en: "Change & migrate…", ja: "変更して移行…" },
      relocating: { zh: "迁移中…", en: "Migrating…", ja: "移行中…" },
      relocated: { zh: "已迁移，重启后生效（旧数据保留，确认无误后可手动删除）", en: "Migrated — restart to apply (old data kept; delete it manually once confirmed)", ja: "移行完了 — 再起動で有効（旧データは保持）" },
      dataDirNote: { zh: "默认在程序目录旁，避免占用 C 盘。模型/缓存会很大，可换到其他盘。", en: "Defaults next to the program (off C:). Models/cache grow large — point this at another drive.", ja: "既定はプログラム横（Cドライブ外）。" },
      stTitle: { zh: "存储占用与清理", en: "Storage Usage & Cleanup", ja: "ストレージ使用量とクリーンアップ" },
      stScanning: { zh: "统计中…", en: "Scanning…", ja: "集計中…" },
      stRefresh: { zh: "重新统计", en: "Rescan", ja: "再集計" },
      stClean: { zh: "清理", en: "Clean", ja: "クリーン" },
      stCleanBtn: { zh: "清理", en: "Clean", ja: "クリーン" },
      stCleaning: { zh: "清理中…", en: "Cleaning…", ja: "クリーン中…" },
      stFreed: { zh: "已释放", en: "Freed", ja: "解放済み" },
      stCache: { zh: "渲染/解码缓存", en: "Render/decode caches", ja: "レンダー/デコードキャッシュ" },
      stCacheTitle: { zh: "清理渲染/解码缓存", en: "Clean render/decode caches", ja: "レンダーキャッシュをクリーン" },
      stCacheBody: { zh: "删除可再生的缓存（解码副本、变速产物、旧渲染输出）。当前打开工程引用的文件会保留；已保存的 .usp 工程自带渲染副本，不受影响。未保存工程的旧渲染需要重新渲染。", en: "Deletes regenerable caches (decode copies, stretch products, old render outputs). Files referenced by the open project are kept; saved .usp projects carry their own render copies. Unsaved projects' old renders will need re-rendering.", ja: "再生成可能なキャッシュ（デコードコピー、テンポ産物、古いレンダー出力）を削除します。開いているプロジェクトが参照するファイルは保持されます。保存済み .usp はレンダーコピーを内蔵しているため影響ありません。未保存プロジェクトの古いレンダーは再レンダリングが必要になります。" },
      stCacheBlocked: { zh: "播放/渲染进行中，暂不可清理", en: "Unavailable while playing/rendering", ja: "再生/レンダリング中は使用不可" },
      stAudition: { zh: "试听缓存", en: "Audition caches", ja: "試聴キャッシュ" },
      stAuditionTitle: { zh: "清理试听缓存", en: "Clean audition caches", ja: "試聴キャッシュをクリーン" },
      stAuditionBody: { zh: "删除模型试听音频与训练候选试听目录（重新试听会自动重建）。", en: "Deletes model audition wavs + training candidate audition dirs (re-auditioning rebuilds them).", ja: "モデル試聴音声とトレーニング候補の試聴フォルダを削除します（再試聴で再生成されます）。" },
      stLogs: { zh: "日志", en: "Logs", ja: "ログ" },
      stTraining: { zh: "训练工作区", en: "Training workspaces", ja: "トレーニングワークスペース" },
      stWsDelete: { zh: "删除", en: "Delete", ja: "削除" },
      stWsTitle: { zh: "删除训练工作区", en: "Delete training workspace", ja: "ワークスペースを削除" },
      stWsBody: { zh: "将删除该模型的数据集副本、预处理特征与全部训练 checkpoint——不可恢复，续训将不再可用（已导入到资源管理器的成品模型不受影响）。", en: "Deletes this model's dataset copies, preprocessed features and ALL training checkpoints — irreversible; resume-training becomes unavailable (models already imported into the resource manager are unaffected).", ja: "このモデルのデータセットコピー・前処理特徴・全チェックポイントを削除します。元に戻せず、続きからのトレーニングは不可になります（リソースマネージャに取り込んだモデルは影響ありません）。" },
      stWsPoolNote: { zh: "注意：该工作区带有可复用数据池——删除后，浅扩散训练将需要重新导入数据。", en: "Note: this workspace holds a reusable dataset pool — after deletion, shallow-diffusion training will require importing data again.", ja: "注意：このワークスペースには再利用可能なデータプールがあります。削除後、浅い拡散トレーニングはデータの再インポートが必要になります。" },
      stWsPool: { zh: "共享池", en: "pool", ja: "プール" },
      stWsNone: { zh: "（无训练工作区）", en: "(no workspaces)", ja: "（ワークスペースなし）" },
      stModels: { zh: "模型资源", en: "Model assets", ja: "モデルアセット" },
      stModelsNote: { zh: "在「资源管理器」与 MSST 模型管理中管理", en: "Managed in the resource manager & MSST manager", ja: "リソースマネージャと MSST 管理で管理" },
      stMsst: { zh: "其中分离模型", en: "incl. separation models", ja: "うち分離モデル" },
      stRuntimes: { zh: "训练环境包", en: "Training runtime packs", ja: "トレーニングランタイム" },
      stRuntimesNote: { zh: "在下方「训练环境」面板管理", en: "Managed in the Training Runtime panel below", ja: "下の「トレーニング環境」パネルで管理" },
      stDicts: { zh: "发音词典（必需）", en: "G2P dictionaries (required)", ja: "発音辞書（必須）" },
      stTotal: { zh: "合计", en: "Total", ja: "合計" },
      stErrTraining: { zh: "训练进行中，无法清理", en: "Training is running — cleanup unavailable", ja: "トレーニング中はクリーンアップできません" },
      stErrBusy: { zh: "渲染/试听进行中，稍后再试", en: "Rendering/audition in flight — try again later", ja: "レンダリング/試聴中です。後でもう一度お試しください" },
      stErrWsMissing: { zh: "工作区不存在（可能已被删除）", en: "Workspace not found (already deleted?)", ja: "ワークスペースが見つかりません" },
      rtTitle: { zh: "训练环境（内嵌 Python 运行时）", en: "Training Runtime (embedded Python)", ja: "トレーニング環境（内蔵 Python）" },
      rtRoot: { zh: "运行时目录", en: "Runtime folder", ja: "ランタイムフォルダ" },
      rtAsciiWarn: { zh: "数据目录路径含非英文字符——内嵌 Python/torch 在此类路径下会加载失败。请先在「存储位置」迁移到纯英文路径（如 D:\\UtaiData）。", en: "Data folder path contains non-ASCII characters — the embedded Python/torch will fail to load there. Migrate the data folder to an ASCII-only path first.", ja: "データフォルダに非 ASCII 文字が含まれています。内蔵 Python/torch が読み込めないため、英数字のみのパスへ移行してください。" },
      rtTestPass: { zh: "自检通过", en: "Self-test passed", ja: "セルフテスト合格" },
      rtTestFail: { zh: "自检未通过", en: "Self-test failed", ja: "セルフテスト不合格" },
      rtTestNone: { zh: "未自检", en: "Not tested", ja: "未テスト" },
      rtTest: { zh: "自检", en: "Self-test", ja: "セルフテスト" },
      rtTesting: { zh: "自检中…", en: "Testing…", ja: "テスト中…" },
      rtDelete: { zh: "删除", en: "Delete", ja: "削除" },
      rtDeleting: { zh: "删除中…", en: "Deleting…", ja: "削除中…" },
      rtDeleteTitle: { zh: "删除运行时包", en: "Delete runtime pack", ja: "ランタイムパックを削除" },
      rtDeleteBody: { zh: "该运行时包将从磁盘删除（之后可重新下载或从本地文件安装）。", en: "The pack will be removed from disk (it can be re-downloaded or re-installed later).", ja: "ディスクから削除されます（後で再ダウンロード/再インストール可能）。" },
      rtCancelBtn: { zh: "取消", en: "Cancel", ja: "キャンセル" },
      rtDownload: { zh: "下载", en: "Download", ja: "ダウンロード" },
      rtNotPublished: { zh: "在线包尚未发布——可用「从本地文件安装」。", en: "Online pack not published yet — use “Install from local file”.", ja: "オンライン版は未公開——「ローカルから」をご利用ください。" },
      rtLocalInstall: { zh: "从本地文件安装…", en: "Install from local file…", ja: "ローカルからインストール…" },
      rtCancel: { zh: "取消安装", en: "Cancel install", ja: "インストール中止" },
      rtExperimental: { zh: "实验性", en: "Experimental", ja: "実験的" },
      rtNote: { zh: "模型转换与训练使用此内嵌运行时（无需系统 Python）。GPU 训练包（NVIDIA 20-50 系 / AMD / Intel）按阶段加入。", en: "Model conversion and training run on this embedded runtime (no system Python needed). GPU packs (NVIDIA 20-50 / AMD / Intel) arrive in stages.", ja: "モデル変換とトレーニングはこの内蔵ランタイムで実行されます。GPU 版は段階的に追加されます。" },
      rtRecommend: { zh: "本机推荐变体", en: "Recommended variant", ja: "推奨バリアント" },
      rtPackLabel_cpu: { zh: "CPU 运行时（模型转换基座 + CPU 训练）", en: "CPU runtime (model conversion base + CPU training)", ja: "CPU ランタイム（モデル変換基盤 + CPU トレーニング）" },
      rtPackLabel_nv_cu130: { zh: "NVIDIA 运行时（cu130；RTX 20-50 训练 + 模型转换）", en: "NVIDIA runtime (cu130; RTX 20-50 training + conversion)", ja: "NVIDIA ランタイム（cu130；RTX 20-50 トレーニング + 変換）" },
      rtPackLabel_amd: { zh: "AMD 运行时（ROCm；RDNA3/4 训练 + 模型转换）", en: "AMD runtime (ROCm; RDNA3/4 training + conversion)", ja: "AMD ランタイム（ROCm；RDNA3/4 トレーニング + 変換）" },
      rtPackLabel_xpu: { zh: "Intel 运行时（XPU；Arc 训练 + 模型转换）", en: "Intel runtime (XPU; Arc training + conversion)", ja: "Intel ランタイム（XPU；Arc トレーニング + 変換）" },
      // pyenv-progress channel (code+params → text; zh reproduces the pre-i18n wording)
      pgFetchManifest: { zh: "获取包清单...", en: "Fetching pack manifest...", ja: "パックマニフェストを取得中..." },
      pgDownloading: { zh: "下载", en: "Downloading", ja: "ダウンロード中" },
      pgExtracting: { zh: "解压运行时包...", en: "Extracting runtime pack...", ja: "ランタイムパックを展開中..." },
      pgFiles: { zh: "个文件", en: "files", ja: "ファイル" },
      pgEnvtest: { zh: "运行环境自检...", en: "Running environment self-test...", ja: "環境セルフテストを実行中..." },
      pgVerify: { zh: "校验分卷 sha256...", en: "Verifying part sha256...", ja: "分割ファイルの sha256 を検証中..." },
      pgVerifySkipped: { zh: "未找到 manifest——跳过校验（仅建议用于本地构建的包）", en: "No manifest found — verification skipped (only recommended for locally built packs)", ja: "manifest が見つかりません——検証をスキップします（ローカルビルドのパックのみ推奨）" },
      pgInstallDone: { zh: "安装完成，自检通过。", en: "Install complete; self-test passed.", ja: "インストール完了、セルフテスト合格。" },
      pgInstalledSkipped: { zh: "已安装（取消跳过了自检——可在列表中手动自检）。", en: "Installed (cancel skipped the self-test — run it manually from the pack list).", ja: "インストール済み（キャンセルによりセルフテストをスキップ——リストから手動で実行できます）。" },
      pgInstalledFailed: { zh: "已安装，但自检未通过：", en: "Installed, but the self-test failed: ", ja: "インストール済みですが、セルフテスト不合格：" },
      srcTitle: { zh: "下载源 / 网络", en: "Download Source / Network", ja: "ダウンロードソース / ネットワーク" },
      srcHF: { zh: "HuggingFace（默认）", en: "HuggingFace (default)", ja: "HuggingFace（既定）" },
      srcMirror: { zh: "HF Mirror (hf-mirror.com) — 中国大陆加速", en: "HF Mirror (hf-mirror.com) — China mainland", ja: "HF Mirror (hf-mirror.com) — 中国本土" },
      srcCustom: { zh: "自定义", en: "Custom", ja: "カスタム" },
      srcNote: { zh: "声音 / 分离模型的下载来源，中国大陆建议选 HF Mirror。训练运行时包已自动在 HuggingFace 与镜像间回退，无需在此设置。", en: "Where voice/separation models download from — HF Mirror is recommended in mainland China. Runtime packs already auto-fail-over between HuggingFace and the mirror.", ja: "モデルのダウンロード元。中国本土では HF Mirror 推奨。トレーニングランタイムは自動でフェイルオーバーします。" },
      srcTest: { zh: "测试连接", en: "Test connection", ja: "接続テスト" },
      srcTesting: { zh: "测试中…", en: "Testing…", ja: "テスト中…" },
      srcOk: { zh: "通畅", en: "Good", ja: "良好" },
      srcSlow: { zh: "偏慢", en: "Slow", ja: "やや遅い" },
      srcThrottled: { zh: "疑似被限速 / 干扰（大文件可能失败）", en: "Throttled / interfered (large downloads may fail)", ja: "スロットリング / 妨害の疑い（大容量は失敗する可能性）" },
      srcUnreachable: { zh: "不通", en: "Unreachable", ja: "接続不可" },
      srcHttpErr: { zh: "源拒绝 / 无测试文件", en: "Rejected / no test file", ja: "拒否 / テストファイルなし" },
      // GitHub-mirror sub-block (the two preset options show their domain literally —
      // not translated; the custom option reuses srcCustom, same label ONE source).
      ghSrcTitle: { zh: "GitHub 镜像", en: "GitHub Mirror", ja: "GitHub ミラー" },
      ghDirect: { zh: "官方直连", en: "Direct", ja: "直接接続" },
      ghNote: { zh: "作用于 GitHub 直链下载（分离模型、MIDI 引擎等）。加速代理为社区公共服务，随时可能失效；不可用时请填自定义前缀。", en: "Applies to direct GitHub downloads (separation models, MIDI engine, …). The preset proxies are community-run public services and may vanish; enter a custom prefix if they stop working.", ja: "GitHub 直リンクのダウンロード（分離モデル、MIDI エンジンなど）に適用されます。プリセットのプロキシはコミュニティ運営の公共サービスで、突然使えなくなることがあります。その場合はカスタムプレフィックスを入力してください。" },
    };
    return map[key]?.[lang] ?? map[key]?.en ?? key;
  };

  /** Catalog labels live in Rust (single catalog source) but as data, not copy —
   *  translate per variant here, falling back to the backend label for variants
   *  this build doesn't know yet. */
  const packLabel = (c: RuntimeCatalogItem) => {
    const key = `rtPackLabel_${c.variant.replace(/-/g, "_")}`;
    const v = L(key);
    return v === key ? c.label : v;
  };

  const srcTestLabel = (r: ProbeResult) => {
    const speed = r.bytes > 0 ? ` · ${r.mbps.toFixed(2)} MB/s` : "";
    if (r.verdict === "ok") return L("srcOk") + speed;
    if (r.verdict === "slow") return L("srcSlow") + speed;
    if (r.verdict === "throttled") return L("srcThrottled") + speed;
    if (r.verdict === "http_error") return L("srcHttpErr") + (r.http_status ? ` (${r.http_status})` : "");
    // PROBE_* codes localize via the app-wide backend-error map (raw fallback).
    return L("srcUnreachable") + (r.error ? ` — ${backendErrText(r.error)}` : "");
  };

  /** pyenv-progress line: localized from the stable code+params; raw message fallback
   *  (older/legacy emits carry no code). */
  const progressText = (p: PyenvProgress): string => {
    const P = p.params ?? [];
    switch (p.code) {
      case "STAGE_FETCH_MANIFEST": return L("pgFetchManifest");
      case "STAGE_DOWNLOADING": return `${L("pgDownloading")} ${P[0] ?? ""}  ${P[1] ?? "?"} / ${P[2] ?? "?"} MB`;
      case "STAGE_EXTRACTING": return P[0] ? `${L("pgExtracting")} ${P[0]} ${L("pgFiles")}` : L("pgExtracting");
      case "STAGE_ENVTEST": return L("pgEnvtest");
      case "STAGE_VERIFY": return L("pgVerify");
      case "STAGE_VERIFY_SKIPPED": return L("pgVerifySkipped");
      case "INSTALL_DONE": return L("pgInstallDone");
      case "INSTALLED_ENVTEST_SKIPPED": return L("pgInstalledSkipped");
      // params[0] = the inner envtest error (itself CODE-bearing) — localize it too.
      case "INSTALLED_ENVTEST_FAILED": return L("pgInstalledFailed") + (P[0] ? backendErrText(P[0]) : "");
      default: return p.message;
    }
  };

  return (
    <aside className="settings-panel" style={panelStyle}>
      <div className="panel-header" onMouseDown={startDrag}>
        <span className="panel-title">{L("title")}</span>
        <button className="panel-close" onClick={onClose}>X</button>
      </div>
      <PanelResizeHandles start={startResize} />

      <div className="settings-content">
        <section className="settings-section">
          <h3 className="settings-section-title">{L("language")}</h3>
          <div className="settings-field">
            <select value={lang} onChange={(e) => handleLangChange(e.target.value)}>
              <option value="zh">简体中文</option>
              <option value="en">English</option>
              <option value="ja">日本語</option>
            </select>
          </div>
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("verTitle")}</h3>
          <div className="settings-row">
            <span className="settings-label">{L("verCurrent")}</span>
            <span className="settings-value">{appVersion ? `v${appVersion}` : "…"}</span>
            <button className="settings-mini-btn" disabled={updChecking} onClick={() => void handleUpdateCheck()}>
              {updChecking ? L("verChecking") : L("verCheckBtn")}
            </button>
          </div>
          {updResult?.kind === "latest" && <div className="settings-note">{L("verLatest")}</div>}
          {updResult?.kind === "found" && (
            <div className="settings-note">
              {L("verFound")}: v{updResult.info.version}{" "}
              <button className="settings-mini-btn" onClick={() => useAppStore.getState().openUpdateDialog(updResult.info)}>
                {L("verView")}
              </button>
            </div>
          )}
          {updResult?.kind === "error" && <div className="settings-error">{updResult.msg}</div>}
          <label className="training-check-row" style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
            <input type="checkbox" checked={autoCheck} onChange={(e) => handleAutoCheckToggle(e.target.checked)} />
            <span>{L("verAutoCheck")}</span>
          </label>
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("hardware")}</h3>

          {hw && (
            <div className="settings-hw-info">
              <div className="settings-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                <span className="settings-label">{L("gpu")}</span>
                {(hw.gpus?.length
                  ? hw.gpus
                  : hw.gpu_name.split(", ").map((name) => ({ name, vendor: "" }))
                ).map((g, i) => (
                  <span key={i} className="settings-value" style={{ maxWidth: "100%", display: "flex", gap: 6, alignItems: "center" }}>
                    {/* The NAME is the shrink absorber (min-width:0 + ellipsis); the badge is pinned
                        (flexShrink:0). A bare text node was an unshrinkable nowrap flex item, so a long
                        GPU name pushed the badge past the container's overflow:hidden — its right border
                        was clipped invisible (S59c rightmost-child-clipping class of bug). */}
                    <span style={{ minWidth: 0, flex: "0 1 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{g.name}</span>
                    {g.vendor && g.vendor !== "other" && (
                      <span className="settings-badge ok" style={{ textTransform: "uppercase", flexShrink: 0 }}>{g.vendor}</span>
                    )}
                  </span>
                ))}
              </div>
              <div className="settings-row">
                <span className="settings-label">{L("cuda")}</span>
                <span className={`settings-badge ${hw.cuda_available ? "ok" : "no"}`}>
                  {hw.cuda_available ? L("yes") : L("no")}
                </span>
              </div>
              <div className="settings-row">
                <span className="settings-label">{L("directml")}</span>
                <span className={`settings-badge ${hw.directml_available ? "ok" : "no"}`}>
                  {hw.directml_available ? L("yes") : L("no")}
                </span>
              </div>
            </div>
          )}

          <div className="settings-field">
            <label>{L("epLabel")}</label>
            <select value={device} onChange={(e) => handleDeviceChange(e.target.value)}>
              <option value="auto">{L("auto")}</option>
              <option value="cuda" disabled={!hw?.cuda_available}>{L("cudaOpt")}</option>
              <option value="directml" disabled={!hw?.directml_available}>{L("dmlOpt")}</option>
              <option value="cpu">{L("cpuOpt")}</option>
            </select>
            {saving && <span className="settings-saving">...</span>}
          </div>

          <p className="settings-note">{L("note")}</p>
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("storage")}</h3>
          <div className="settings-field" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
            <label>{L("dataDir")}</label>
            <span className="settings-value" style={{ maxWidth: "100%", wordBreak: "break-all", fontSize: 11 }}>{dataDir || "…"}</span>
          </div>
          {dataDirIssue && (
            <p className="settings-error" style={{ wordBreak: "break-all" }}>
              {(dataDirIssue.fell_back ? L("stDirFellBack") : L("stDirRecreated")).replace("{configured}", dataDirIssue.configured)}
            </p>
          )}
          <div className="settings-field">
            <button className="settings-btn" onClick={handleRelocate} disabled={relocating} style={{ padding: "5px 12px", cursor: "pointer" }}>
              {relocating ? L("relocating") : L("relocate")}
            </button>
          </div>
          {relocateMsg === "migrated" && <p className="settings-note">{L("relocated")}</p>}
          {relocateMsg === "error: training" && <p className="settings-note" style={{ color: "#f87171" }}>{L("stErrTraining")}</p>}
          {relocateMsg?.startsWith("error:") && relocateMsg !== "error: training" && <p className="settings-note" style={{ color: "#f87171" }}>{relocateMsg}</p>}
          <p className="settings-note">{L("dataDirNote")}</p>
        </section>

        {/* ── S61 存储占用与清理 — everything file/data-related lives together with the data-dir config ── */}
        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("stTitle")}</h3>
          {!storage && <p className="settings-note">{storageScanning ? L("stScanning") : "…"}</p>}
          {storage && (
            <div className="settings-hw-info settings-storage">
              <div className="settings-row">
                <span className="settings-label">{L("stCache")}</span>
                <span className="settings-value">{fmtSize(storage.cache_bytes)}</span>
                <button
                  className="settings-mini-btn"
                  disabled={cleanBusy !== null || cacheCleanBlocked}
                  title={cacheCleanBlocked ? L("stCacheBlocked") : undefined}
                  onClick={handleCleanCache}
                >
                  {cleanBusy === "cache" ? L("stCleaning") : L("stClean")}
                </button>
              </div>
              <div className="settings-row">
                <span className="settings-label">{L("stAudition")}</span>
                <span className="settings-value">{fmtSize(storage.audition_bytes)}</span>
                <button
                  className="settings-mini-btn"
                  disabled={cleanBusy !== null || trainingBusy}
                  onClick={handleCleanAudition}
                >
                  {cleanBusy === "audition" ? L("stCleaning") : L("stClean")}
                </button>
              </div>
              <div className="settings-row">
                <span className="settings-label">{L("stLogs")}</span>
                <span className="settings-value">{fmtSize(storage.logs_bytes)}</span>
                <button className="settings-mini-btn" disabled={cleanBusy !== null} onClick={handleCleanLogs}>
                  {cleanBusy === "logs" ? L("stCleaning") : L("stClean")}
                </button>
              </div>
              <div className="settings-row">
                <span className="settings-label">{L("stTraining")}</span>
                <span className="settings-value">{fmtSize(storage.training_bytes)}</span>
              </div>
              {storage.workspaces.length === 0 && (
                <p className="settings-note" style={{ margin: "2px 0 0" }}>{L("stWsNone")}</p>
              )}
              {storage.workspaces.map((ws) => (
                <div className="settings-row settings-ws-row" key={ws.slug}>
                  <span className="settings-value settings-ws-name" title={`${ws.name} (${ws.slug})`}>
                    {ws.name}
                    {ws.family ? ` · ${ws.family}` : ""}
                    {ws.has_pool ? ` · ${L("stWsPool")}` : ""}
                  </span>
                  <span className="settings-value">{fmtSize(ws.bytes)}</span>
                  <button
                    className="settings-mini-btn danger"
                    disabled={cleanBusy !== null || trainingBusy}
                    onClick={() => handleDeleteWorkspace(ws)}
                  >
                    {cleanBusy === ws.slug ? L("stCleaning") : L("stWsDelete")}
                  </button>
                </div>
              ))}
              <div className="settings-row">
                <span className="settings-label">{L("stModels")}</span>
                <span className="settings-value">
                  {fmtSize(storage.models_bytes)}
                  {storage.msst_bytes > 0 ? `（${L("stMsst")} ${fmtSize(storage.msst_bytes)}）` : ""}
                </span>
              </div>
              <p className="settings-note" style={{ margin: "0 0 4px" }}>{L("stModelsNote")}</p>
              <div className="settings-row">
                <span className="settings-label">{L("stRuntimes")}</span>
                <span className="settings-value">{fmtSize(storage.runtimes_bytes)}</span>
              </div>
              <p className="settings-note" style={{ margin: "0 0 4px" }}>{L("stRuntimesNote")}</p>
              <div className="settings-row">
                <span className="settings-label">{L("stDicts")}</span>
                <span className="settings-value">{fmtSize(storage.dictionaries_bytes)}</span>
              </div>
              <div className="settings-row" style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 4, marginTop: 2 }}>
                <span className="settings-label">{L("stTotal")}</span>
                <span className="settings-value">
                  {fmtSize(storage.cache_bytes + storage.models_bytes + storage.runtimes_bytes + storage.dictionaries_bytes + storage.training_bytes + storage.logs_bytes)}
                </span>
              </div>
            </div>
          )}
          <div className="settings-field" style={{ marginTop: 6 }}>
            <button className="settings-mini-btn" disabled={storageScanning || cleanBusy !== null} onClick={() => void refreshStorage()}>
              {storageScanning ? L("stScanning") : L("stRefresh")}
            </button>
            {cleanMsg && <span className="settings-value" style={{ fontSize: 11 }}>{cleanMsg}</span>}
          </div>
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("srcTitle")}</h3>
          <div className="settings-source">
            {(["huggingface", "hf-mirror", "custom"] as const).map((t) => (
              <label key={t} className={`settings-source-opt ${mirror.type === t ? "active" : ""}`}>
                <input
                  type="radio"
                  name="dlsource"
                  checked={mirror.type === t}
                  onChange={() => setMirror({ type: t, customUrl: mirror.customUrl })}
                />
                <span>{t === "huggingface" ? L("srcHF") : t === "hf-mirror" ? L("srcMirror") : L("srcCustom")}</span>
              </label>
            ))}
            {mirror.type === "custom" && (
              <input
                type="text"
                className="settings-source-url"
                placeholder="https://your-mirror.com"
                value={mirror.customUrl}
                onChange={(e) => setMirror({ type: "custom", customUrl: e.target.value })}
              />
            )}
          </div>
          <div className="settings-source-test">
            <button className="settings-mini-btn" disabled={srcTesting} onClick={handleSrcTest}>
              {srcTesting ? L("srcTesting") : L("srcTest")}
            </button>
            {srcTest && (
              <span className={`settings-source-result ${srcTest.verdict}`}>{srcTestLabel(srcTest)}</span>
            )}
          </div>
          <p className="settings-note">{L("srcNote")}</p>

          {/* GitHub direct-link mirror — a separate axis from the HF mirror above
              (the HF rewrite touches only huggingface.co, the GH proxy prefixes only
              github.com-family hosts; download URLs chain both, see msst-models.ts). */}
          <div className="settings-field" style={{ marginTop: 4 }}>
            <label>{L("ghSrcTitle")}</label>
            <div className="settings-source">
              {/* S66: presets are DATA (remote-refreshable mirrors.json > builtin fallback) —
                  public proxies rot in 6-18 months, so the list must never be an enum again. */}
              <label className={`settings-source-opt ${ghMirror.type === "direct" ? "active" : ""}`}>
                <input
                  type="radio"
                  name="ghsource"
                  checked={ghMirror.type === "direct"}
                  onChange={() => setGhMirror({ type: "direct", customUrl: ghMirror.customUrl })}
                />
                <span>{L("ghDirect")}</span>
              </label>
              {ghPresets.map((p) => {
                const active = ghMirror.type === "preset" && ghEffectivePresetId === p.id;
                return (
                  <label key={p.id} className={`settings-source-opt ${active ? "active" : ""}`}>
                    <input
                      type="radio"
                      name="ghsource"
                      checked={active}
                      onChange={() => setGhMirror({ type: "preset", presetId: p.id, customUrl: ghMirror.customUrl })}
                    />
                    <span>{p.id}</span>
                  </label>
                );
              })}
              <label className={`settings-source-opt ${ghMirror.type === "custom" ? "active" : ""}`}>
                <input
                  type="radio"
                  name="ghsource"
                  checked={ghMirror.type === "custom"}
                  onChange={() => setGhMirror({ type: "custom", customUrl: ghMirror.customUrl })}
                />
                <span>{L("srcCustom")}</span>
              </label>
              {ghMirror.type === "custom" && (
                <input
                  type="text"
                  className="settings-source-url"
                  placeholder="https://your-gh-proxy.com"
                  value={ghMirror.customUrl}
                  onChange={(e) => setGhMirror({ type: "custom", customUrl: e.target.value })}
                />
              )}
            </div>
          </div>
          <div className="settings-source-test">
            <button className="settings-mini-btn" disabled={ghSrcTesting} onClick={handleGhSrcTest}>
              {ghSrcTesting ? L("srcTesting") : L("srcTest")}
            </button>
            {ghSrcTest && (
              <span className={`settings-source-result ${ghSrcTest.verdict}`}>{srcTestLabel(ghSrcTest)}</span>
            )}
          </div>
          <p className="settings-note">{L("ghNote")}</p>
        </section>

        {/* ── S64 model-asset packs — right under the download-source config they ride on ── */}
        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("assetTitle")}</h3>
          {assetPacks.map((p) => {
            const label =
              p.id === "aux-inference"
                ? L("assetAux")
                : p.id === "training-rvc"
                  ? L("assetRvc")
                  : p.id === "training-sovits-v2"
                    ? L("assetSovitsV2")
                    : L("assetSovits");
            // p.downloading = backend truth (survives a panel remount before the next chunk event).
            const isDl = assetActive === p.id || assetProgress?.pack === p.id || p.downloading;
            const anyDl = assetActive !== null || assetPacks.some((x) => x.downloading);
            return (
              <div key={p.id} className="settings-row" style={{ alignItems: "center", gap: 8 }}>
                <span className="settings-label">{label}</span>
                <span className={`settings-badge ${p.missing === 0 ? "ok" : "no"}`}>
                  {p.missing === 0
                    ? L("assetInstalled")
                    : `${L("assetMissing")} ${p.missing}/${p.fileCount} · ${fmtSize(p.missingBytes)}`}
                </span>
                {p.missing > 0 && !isDl && (
                  <button className="settings-mini-btn" disabled={anyDl} onClick={() => void handleAssetDownload(p.id)}>
                    {L("assetDownload")}
                  </button>
                )}
                {isDl && (
                  <button
                    className="settings-mini-btn danger"
                    onClick={() => void invoke("cancel_asset_pack_download").catch(() => {})}
                  >
                    {L("rtCancelBtn")}
                  </button>
                )}
              </div>
            );
          })}
          {assetProgress && (
            <div className="settings-progress">
              <div className="settings-progress-bar">
                <div
                  className="settings-progress-fill"
                  style={{ width: `${assetProgress.total > 0 ? Math.round((assetProgress.downloaded / assetProgress.total) * 100) : 0}%` }}
                />
              </div>
              <span className="settings-progress-text">
                {`${assetProgress.fileIndex + 1}/${assetProgress.fileCount} · ${fmtSize(assetProgress.downloaded)} / ${fmtSize(assetProgress.total)}`}
              </span>
            </div>
          )}
          {assetMsg && <p className="settings-error">{assetMsg}</p>}
          <p className="settings-note">{L("assetNote")}</p>
          <label className="training-check-row" style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}>
            <input
              type="checkbox"
              checked={startupCompCheck}
              onChange={(e) => handleStartupCompToggle(e.target.checked)}
            />
            <span>{L("assetStartupCheck")}</span>
          </label>
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("cudaRuntime")}</h3>

          <div className="settings-hw-info">
            <div className="settings-row">
              <span className="settings-label">{L("cudaRuntime")}</span>
              <span className={`settings-badge ${cudaReady ? "ok" : "no"}`}>
                {cudaReady ? L("cudaInstalled") : L("cudaNotInstalled")}
              </span>
            </div>
          </div>

          {!cudaReady && !cudaDownloading && (
            <>
              <button className="settings-download-btn" onClick={handleCudaDownload}>
                {L("cudaDownload")}
              </button>
              <p className="settings-note">{L("cudaNote")}</p>
            </>
          )}

          {cudaDownloading && cudaProgress && (
            <div className="settings-progress">
              <div className="settings-progress-bar">
                <div
                  className="settings-progress-fill"
                  style={{ width: `${Math.round(cudaProgress.progress * 100)}%` }}
                />
              </div>
              <span className="settings-progress-text">{cudaProgressText(cudaProgress)}</span>
              {/* local-file installs are a fast synchronous extraction — cancel_cuda_download
                  only reaches the NETWORK flow's flag (review S66: a no-op button lies). */}
              {cudaProgress.stage !== "local" && (
                <button
                  className="settings-mini-btn"
                  onClick={() => { void invoke("cancel_cuda_download"); }}
                >
                  {L("cudaCancelDl")}
                </button>
              )}
            </div>
          )}

          {cudaError && (
            <p className="settings-error">{cudaError}</p>
          )}

          {cudaJustInstalled && (
            <p className="settings-note" style={{ color: "#4ade80" }}>{L("cudaRestart")}</p>
          )}

          {/* S66: CUDA arena cap — visible ⟺ effective (user decision): only when the CUDA
              runtime is actually installed AND an NVIDIA GPU is present. 0 = unlimited. */}
          {cudaReady && hw?.gpus?.some((g) => g.vendor === "nvidia") && (
            <div className="settings-field" style={{ marginTop: 6 }}>
              <label>{L("cudaMemLimit")}</label>
              {/* .settings-field is a COLUMN — the input and its unit need their own row,
                  or "MB" wraps below the box with a stray gap (user-reported). */}
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  type="text"
                  className="settings-source-url"
                  style={{ width: 90, flex: "none" }}
                  inputMode="numeric"
                  value={cudaMemLimitText}
                  placeholder="0"
                  onChange={(e) => setCudaMemLimitText(e.target.value.replace(/[^0-9]/g, ""))}
                  onBlur={() => void commitCudaMemLimit()}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                  }}
                />
                <span className="settings-value" style={{ maxWidth: "none" }}>MB</span>
              </div>
            </div>
          )}
          {cudaReady && hw?.gpus?.some((g) => g.vendor === "nvidia") && (
            <p className="settings-note">{L("cudaMemLimitNote")}</p>
          )}

          {/* S66: local-file install + on-disk layout (copyable paths for inspection/support). */}
          {!cudaDownloading && (
            <button className="settings-mini-btn" style={{ marginTop: 6 }} onClick={handleCudaLocalInstall}>
              {L("rtLocalInstall")}
            </button>
          )}
          {cudaPaths && (
            <>
              {cudaPaths.missing.length > 0 && !cudaDownloading && (
                <p className="settings-note">
                  {L("cudaMissing")}: {cudaPaths.missing.join(" · ")}
                </p>
              )}
              <p className="settings-note" style={{ userSelect: "text" }}>
                {L("cudaLocalNote")} {cudaPaths.expectedFiles.join(" · ")}
              </p>
              <div className="settings-field" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                <label>{L("cudaDllDir")}</label>
                <span className="settings-value selectable" style={{ maxWidth: "100%", wordBreak: "break-all", whiteSpace: "normal", fontSize: 11 }}>{cudaPaths.dllDir}</span>
                <label style={{ marginTop: 2 }}>{L("cudaOrtDir")}</label>
                <span className="settings-value selectable" style={{ maxWidth: "100%", wordBreak: "break-all", whiteSpace: "normal", fontSize: 11 }}>{cudaPaths.ortDir}</span>
              </div>
            </>
          )}
        </section>

        <section className="settings-section" style={{ marginTop: 16 }}>
          <h3 className="settings-section-title">{L("rtTitle")}</h3>
          {rt && (
            <>
              <div className="settings-field" style={{ flexDirection: "column", alignItems: "flex-start", gap: 2 }}>
                <label>{L("rtRoot")}</label>
                <span className="settings-value" style={{ maxWidth: "100%", wordBreak: "break-all", whiteSpace: "normal", fontSize: 11 }}>{rt.root || "…"}</span>
              </div>
              {!rt.root_ascii_ok && <p className="settings-error">{L("rtAsciiWarn")}</p>}

              {rt.packs.length > 0 && (
                <div className="settings-hw-info">
                  {rt.packs.map((p) => (
                    <div key={p.id} className="settings-pack">
                      <div className="settings-row">
                        <span className="settings-label">{p.variant}</span>
                        <span className={`settings-badge ${p.envtest?.overall === "pass" ? "ok" : "no"}`}>
                          {p.envtest ? (p.envtest.overall === "pass" ? L("rtTestPass") : L("rtTestFail")) : L("rtTestNone")}
                        </span>
                      </div>
                      <span className="settings-value" style={{ textAlign: "left", maxWidth: "100%" }}>
                        {p.id} · torch {p.torch || "?"} · py {p.python || "?"} · {fmtGB(p.disk_bytes)}
                      </span>
                      <div className="settings-pack-actions">
                        <button
                          className="settings-mini-btn"
                          disabled={rtBusy || envtesting !== null || deleting !== null}
                          onClick={() => handleRtEnvtest(p.id)}
                        >
                          {envtesting === p.id ? L("rtTesting") : L("rtTest")}
                        </button>
                        <button
                          className="settings-mini-btn danger"
                          disabled={rtBusy || envtesting !== null || deleting !== null}
                          onClick={() => handleRtDelete(p.id)}
                        >
                          {deleting === p.id ? L("rtDeleting") : L("rtDelete")}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Only offer packs this machine's hardware can run (backend-gated:
                  CPU always; NVIDIA needs sm_75+; AMD/Intel need the matching GPU).
                  Unsupported variants are hidden here — local-file install stays open. */}
              {rt.catalog.filter((c) => !c.installed && c.supported).map((c) => (
                <div key={c.id} className="settings-field">
                  <label>{packLabel(c)}{c.experimental ? `（${L("rtExperimental")}）` : ""}</label>
                  {c.downloadable ? (
                    <button
                      className="settings-download-btn"
                      disabled={rtBusy || envtesting !== null || deleting !== null}
                      onClick={() => handleRtDownload(c.id)}
                    >
                      {L("rtDownload")}（~{fmtGB(c.download_bytes)} / {L("rtRoot")} {fmtGB(c.disk_bytes)}）
                    </button>
                  ) : (
                    <p className="settings-note">{L("rtNotPublished")}</p>
                  )}
                </div>
              ))}

              <div className="settings-pack-actions">
                <button className="settings-mini-btn" disabled={rtBusy || envtesting !== null || deleting !== null} onClick={handleRtLocalInstall}>
                  {L("rtLocalInstall")}
                </button>
                {rtBusy && (
                  <button className="settings-mini-btn danger" onClick={() => { invoke("cancel_runtime_install").catch(() => {}); }}>
                    {L("rtCancel")}
                  </button>
                )}
              </div>

              {rtBusy && rtProgress && (
                <div className="settings-progress">
                  <div className="settings-progress-bar">
                    <div
                      className="settings-progress-fill"
                      style={{ width: `${Math.round(Math.min(1, Math.max(0, rtProgress.progress)) * 100)}%` }}
                    />
                  </div>
                  <span className="settings-progress-text">{progressText(rtProgress)}</span>
                </div>
              )}
              {rtNotice && !rtBusy && (
                <p
                  className="settings-note"
                  style={rtNotice.code === "INSTALL_DONE" ? { color: "#4ade80" } : undefined}
                >
                  {progressText(rtNotice)}
                </p>
              )}
              {rtError && <p className="settings-error">{rtError}</p>}
              <p className="settings-note">
                {L("rtNote")}
                {hw?.recommended_variant ? ` ${L("rtRecommend")}: ${hw.recommended_variant}` : ""}
              </p>
            </>
          )}
        </section>
      </div>
    </aside>
  );
}
