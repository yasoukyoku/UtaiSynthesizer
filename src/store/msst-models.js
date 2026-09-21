import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { DEFAULT_MIRROR, applyMirror, DEFAULT_GH_MIRROR, BUILTIN_GH_PRESETS, migrateGhMirror, ghMirrorCandidates, } from "../lib/models/msst-catalog";
import { loadSetting, saveSetting } from "../lib/settings";
import { backendErrorMessage } from "../lib/backendError";
import { maybeShowErrorModal } from "../lib/errorDisplay";
const GH_PRESET_CACHE_KEY = "utai.ghPresetsRemote";
const GH_PRESET_TTL_MS = 24 * 3600 * 1000;
function validPresets(v) {
    if (!Array.isArray(v))
        return null;
    const out = [];
    for (const e of v) {
        const id = e?.id;
        const prefix = e?.prefix;
        if (typeof id === "string" && typeof prefix === "string" && /^https:\/\//.test(prefix)) {
            out.push({ id, prefix: prefix.replace(/\/+$/, "") });
        }
    }
    return out.length > 0 ? out : null;
}
export const useMsstModelStore = create((set, get) => ({
    installed: [],
    modelsDir: "",
    downloading: {},
    error: null,
    mirror: loadSetting("utai.mirror", DEFAULT_MIRROR),
    // migrate persisted pre-S66 enum values ({type:"ghfast"|"ghproxy"}) — localStorage outlives updates.
    ghMirror: migrateGhMirror(loadSetting("utai.ghMirror", DEFAULT_GH_MIRROR)),
    ghPresets: loadSetting(GH_PRESET_CACHE_KEY, null)?.presets?.length
        ? (validPresets(loadSetting(GH_PRESET_CACHE_KEY, null)?.presets) ?? BUILTIN_GH_PRESETS)
        : BUILTIN_GH_PRESETS,
    refreshGhPresets: async (force) => {
        const cache = loadSetting(GH_PRESET_CACHE_KEY, null);
        if (!force && cache && Date.now() - cache.ts < GH_PRESET_TTL_MS)
            return; // fresh enough
        try {
            const remote = await invoke("fetch_mirror_list");
            const presets = validPresets(remote?.gh_prefixes);
            if (presets) {
                saveSetting(GH_PRESET_CACHE_KEY, { ts: Date.now(), presets });
                set({ ghPresets: presets });
            }
        }
        catch {
            // network-less / HF unreachable — builtin (or cached) list stays in effect
        }
    },
    fetchInstalled: async () => {
        try {
            const models = await invoke("list_msst_models");
            set({ installed: models, error: null });
        }
        catch (e) {
            set({ error: String(e) });
        }
    },
    fetchModelsDir: async () => {
        try {
            const dir = await invoke("get_msst_models_dir");
            set({ modelsDir: dir });
        }
        catch (e) {
            set({ error: String(e) });
        }
    },
    downloadEntry: async (entry, precision) => {
        const { mirror, ghMirror, ghPresets } = get();
        // HF rewrite first (host swap), then the GH axis expands into failover candidates
        // (chosen proxy → direct ONLY — MSST files carry no sha256, so the preset-proxy tail
        // stays OFF per the review-S66 poisoning rule; see ghMirrorCandidates) — each axis
        // only touches its own host family, so they never interfere with each other or with
        // non-mirrored hosts (e.g. dl.fbaipublicfiles.com).
        // The original yaml must land BEFORE the ckpt and be named <ckpt stem>.yaml:
        // the ckpt download auto-converts on completion, and the converter reads the SIBLING
        // yaml (chunk/overlap, stem labels). The URL basename can differ from the ckpt name
        // (e.g. config_melbandroformer_inst_v2.yaml), so always rename to the ckpt stem.
        if (entry.configUrl) {
            try {
                const cfgUrls = ghMirrorCandidates(applyMirror(entry.configUrl, mirror), ghMirror, ghPresets);
                const stem = entry.filename.replace(/\.[^.]+$/, "");
                await get().downloadUrl(cfgUrls, `${stem}.yaml`);
            }
            catch {
                // config download failure is non-fatal
            }
        }
        const urls = ghMirrorCandidates(applyMirror(entry.downloadUrl, mirror), ghMirror, ghPresets);
        // precision/architecture only apply to the MODEL download (auto-convert target), never the
        // yaml sidecar. Passing the catalog architecture matters for hash-named official weights
        // (e.g. demucs 5c90dfd2-34c22ccb.th) that Rust's name detection cannot classify.
        await get().downloadUrl(urls, entry.filename, precision, entry.architecture);
    },
    downloadUrl: async (urls, filename, precision, architecture) => {
        set((s) => ({
            downloading: {
                ...s.downloading,
                [filename]: { filename, downloaded: 0, total: 0, stage: "download" },
            },
            error: null,
        }));
        try {
            await invoke("download_msst_model", { urls, filename, precision, architecture });
            set((s) => {
                const { [filename]: _, ...rest } = s.downloading;
                return { downloading: rest };
            });
            await get().fetchInstalled();
        }
        catch (e) {
            set((s) => {
                const { [filename]: _, ...rest } = s.downloading;
                return { downloading: rest, error: String(e) };
            });
        }
    },
    // 补转/convert an INSTALLED model to the given precision (undefined = full fp32 export from
    // ckpt). fp16 with the fp32 onnx on disk is the fast post-hoc path (~1-2 min); fp32 (or fp16
    // without the fp32 file) is a full re-export. Reuses the download record's "converting" stage
    // so the existing DownloadBar / indicators track it.
    convertPrecision: async (filename, precision, architecture) => {
        set((s) => ({
            downloading: {
                ...s.downloading,
                [filename]: { filename, downloaded: 0, total: 0, stage: "converting" },
            },
            error: null,
        }));
        try {
            await invoke("convert_msst_model", { filename, precision, architecture });
            await get().fetchInstalled();
        }
        catch (e) {
            // S74: converter failures are blocking → the scrollable/copyable modal funnel (the raw
            // stderr is already logged backend-side by run_converter). CONVERT_LOW_MEMORY /
            // MSST_CONVERT_FAILED are modal:true; fall back to the inline banner only if a dialog
            // is already open (maybeShowErrorModal declines rather than clobber it).
            const display = backendErrorMessage(e) ?? String(e);
            if (!maybeShowErrorModal(e, display))
                set({ error: String(e) });
        }
        get().removeDownload(filename);
    },
    deleteModel: async (filename) => {
        try {
            await invoke("delete_msst_model", { filename });
            await get().fetchInstalled();
        }
        catch (e) {
            set({ error: String(e) });
        }
    },
    importLocal: async (path) => {
        try {
            await invoke("import_local_msst_model", { sourcePath: path });
            await get().fetchInstalled();
        }
        catch (e) {
            set({ error: String(e) });
        }
    },
    clearError: () => set({ error: null }),
    setMirror: (mirror) => { saveSetting("utai.mirror", mirror); set({ mirror }); },
    setGhMirror: (ghMirror) => { saveSetting("utai.ghMirror", ghMirror); set({ ghMirror }); },
    updateDownloadProgress: (filename, downloaded, total, stage) => set((s) => ({
        downloading: {
            ...s.downloading,
            [filename]: {
                filename,
                downloaded,
                total,
                stage: (stage === "converting" ? "converting" : "download"),
            },
        },
    })),
    removeDownload: (filename) => set((s) => {
        const { [filename]: _, ...rest } = s.downloading;
        return { downloading: rest };
    }),
}));
let progressUnlisten = null;
export async function setupDownloadListener() {
    if (progressUnlisten)
        return;
    progressUnlisten = await listen("msst-download-progress", (event) => {
        useMsstModelStore
            .getState()
            .updateDownloadProgress(event.payload.filename, event.payload.downloaded, event.payload.total, event.payload.stage);
    });
}
