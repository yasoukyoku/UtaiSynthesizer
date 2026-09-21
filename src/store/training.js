/**
 * Training store — mirrors the Rust TrainingManager (protocol v2, S37 rewrite).
 *
 * Event-driven (training-stage / training-step / training-ckpt / training-done),
 * NOT polled: install the module-level listeners once via setupTrainingListeners()
 * (msst-models.ts pattern — global, so progress survives the page being closed).
 * The Rust side keeps the authoritative loss history (get_training_history) so a
 * re-mounted page reconstructs the curve; live points append via training-step.
 */
import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import i18n from "../i18n";
import { backendErrorMessage, isBusyError } from "../lib/backendError";
import { hfBaseForMirror } from "../lib/models/msst-catalog";
import { useAppStore } from "./app";
import { useMsstModelStore } from "./msst-models";
import { isTauri } from "../lib/tauri";
/** Generation counter for the checkpoint scan — see refreshProjectCkpts. */
let ckptScanSeq = 0;
/** Bumped by `enterProject`. Anything that stages data ASYNCHRONOUSLY has to check it after
 *  its await: probing a few hundred files takes long enough for the user to have moved to
 *  another project, and the result would land in THAT project's form. */
let projectEpoch = 0;
/** Nonce for the「文件落到这位歌手」pulse — the same singer must be able to re-flash. */
let flashSeq = 0;
/** Same file, seen from the two sources. Paths come from Rust on one side and from the
 *  sidecar on the other, so compare them case- and separator-insensitively (Windows). */
function ckptKey(p) {
    return p.replace(/\\/g, "/").toLowerCase();
}
/** THE single reconciliation of「运行中刚落的存档」with「磁盘上真实存在的存档」.
 *
 *  The disk scan wins wherever both know a file (it carries size / kind / imported), and a
 *  ckpt the sidecar just announced but the scan has not seen yet is still shown — otherwise a
 *  checkpoint would blink out of the list for the seconds between the event and the next
 *  scan. Pure + tested so the two sources can never drift into duplicate or flickering rows. */
export function mergeCkptSources(eventCkpts, scanned, family) {
    const seen = new Set(scanned.map((r) => ckptKey(r.path)));
    const extra = eventCkpts
        .filter((c) => !seen.has(ckptKey(c.path)))
        .map((c) => ({
        rel: c.path.replace(/\\/g, "/").split("/").slice(-2).join("/"),
        path: c.path,
        family,
        // ★§F2⒝ 批 2 ④ —— **不猜**(见 `CkptRecord.runId` 的注释:`""` 是「槽根就是 run」这个
        // 肯定事实,不是「不知道」)。
        runId: null,
        // Do NOT infer a kind here. The event only says periodic/best/final/stop, and the same
        // file can be a release snapshot (rvc/sovits weights) or a resume point (diffusion
        // model_<step>.pt) depending on family — guessing made a diffusion ckpt read「快照」for
        // the few seconds before the scan caught up, then flip to「可续训」.
        kind: "pending",
        step: c.step,
        bytes: 0,
        mtimeMs: Number.MAX_SAFE_INTEGER, // just written ⇒ newest
        imported: false,
        companions: [],
    }));
    return [...extra, ...scanned].sort((a, b) => b.mtimeMs - a.mtimeMs || a.rel.localeCompare(b.rel));
}
/** S41 共享池模式 — THE single predicate for "a diff run may start without
 *  importing data" (root tab gating, DataStep next button, RunStep start
 *  guard all share it; Rust start_training re-verifies authoritatively). */
export function diffPoolReady(backend, info) {
    return (backend === "sovits_diff" &&
        !!info?.exists &&
        info.family === "sovits" &&
        info.has_dataset &&
        // S76: shallow diffusion refuses multi-speaker workspaces (Rust side), and since the
        // dataset became a PROJECT-level shared layer a multi-singer project's dataset is stored
        // per singer — which this run, carrying no singer groups, cannot consume. Advertising
        // 免导入直训 there would let the user skip the data page straight into a refusal.
        info.n_speakers <= 1);
}
/** THE predicate for「这个项目盘上已经有可复用的数据集,本次运行不必再导入」.
 *
 *  For shallow diffusion it stays the sovits-host-specific `diffPoolReady`. For every other
 *  backend it is `poolFlat` — the project has an on-disk dataset AND that dataset is flat
 *  (single-speaker). The asymmetry mirrors the backend exactly: `try_start` reuses an empty
 *  `dataset_files` for a flat pool, but refuses a multi-speaker one for a run carrying no
 *  speaker groups (`PROJECT_DATASET_SHAPE`) — reconstructing the singer groups from disk so a
 *  multi-speaker project can resume without re-importing is batch 5's job. */
export function poolReusable(backend, poolFlat, diffInfo) {
    return backend === "sovits_diff" ? diffPoolReady(backend, diffInfo) : poolFlat;
}
/** Which of the four step tabs a segment belongs to. The landing and a project's detail are
 *  both「1 · 项目」— picking a project is a move WITHIN that step, not a step of its own. */
export function segTab(seg) {
    switch (seg) {
        case "projects":
        case "detail":
            return 1;
        case "data":
            return 2;
        case "params":
            return 3;
        case "run":
        case "archive":
            // the archive is reached from a project's model card, not from the step rail; it rides
            // under 4 (运行/产物) so the tab highlight has somewhere sensible to sit.
            return 4;
    }
}
const TRAINING_BACKENDS = [
    "rvc",
    "sovits",
    "sovits_v2",
    "sovits_diff",
    "vocoder",
];
/** `TrainingSnapshot.backend` is a plain string from Rust. Widening it into the form config
 *  without checking would let an unknown value drive every backend-keyed branch on the page. */
export function asTrainingBackend(s) {
    return TRAINING_BACKENDS.includes(s) ? s : null;
}
/** Which architecture SLOT a backend trains into — mirrors Rust `training::backend_family`.
 *  Shallow diffusion is not a family of its own: it lives in the sovits slot because it shares
 *  the main model's preprocessing caches, which is the entire reason it exists. THE single
 *  frontend source for this mapping; do not re-derive it per call site. */
export function backendFamily(backend) {
    return backend === "sovits_diff" ? "sovits" : backend;
}
/** ①c: which backends take a SINGER LIST (multi-speaker co-train) — SoVITS (α) + RVC (α′)
 *  + SoVITS 4.0-v2 (S68, natively multi-speaker upstream).
 *  THE single source for the DataStep singer-list gating so the store + page never drift.
 *  Shallow-diffusion / vocoder stay flat-dataset (their loaders assume one speaker). */
export function backendSupportsMultiSpeaker(backend) {
    return backend === "sovits" || backend === "rvc" || backend === "sovits_v2";
}
/** THE single predicate for "step 2 (data) is satisfied" — shared by the root
 *  wizard gating (step3Ok) AND the DataStep next button so they never drift.
 *  ①c: SoVITS/RVC data is a SINGER LIST (default 1 singer = single-speaker, the
 *  degenerate case of N). Every singer needs files; with ≥2 singers each also
 *  needs a (unique) name. Other backends keep the flat-dataset / shared-pool rule. */
/** THE single predicate for「数据这一步满足了吗」 — the wizard gating (`step3Ok`), the data
 *  page's Next button, and「点某个架构该跳到哪一段」all ask this one, so they cannot disagree.
 *
 *  S78: it asks the DISK, not a staging form. Importing became its own act (files land in
 *  `<project>/dataset/` immediately), so「有没有数据」is a property of the project — which is
 *  also what makes an existing project trainable without re-importing anything.
 */
export function trainingDataOk(backend, ds, 
/** Shallow diffusion asks a different question: it needs the SoVITS slot's cached slices, not
 *  merely raw audio (`diffPoolReady`). */
diffInfo) {
    // ★★S144(用户实机报的)—— `diffPoolReady` 是一条**捷径**(宿主槽已经有缓存切片 ⇒ 这一步
    // 不必导入),**不是浅扩散的唯一入口**。此前这里写的是 `return diffPoolReady(...)`,而那让
    // **「没有主模型先训扩散」(diff-first)在界面上结构性地做不到**:
    //   `diffPoolReady` 要求 `info.family === "sovits"`,那个字段来自 **run manifest** ⇒ 一个
    //   从没训练过的槽恒 `""` ⇒ 谓词恒假;而**路由、向导、数据页的下一步、开始按钮四处用的是
    //   同一个谓词** ⇒ 点浅扩散的「开始训练」被扔进数据段,**再导入多少音频也出不来**。
    // ⛔ 后端一侧 diff-first 是**明写受支持的形状**(`training/mod.rs` 的 `run_has_main_model`
    //   那段 doc:「diff-first, which is a supported shape」;`gate_pool_table` 的 check(6) 也
    //   逐条钉着它)⇒ 这是前端比后端**更严**,而不是一条后端约束的转述。
    // ⇒ 捷径优先,不成立时**回落到通用规则**:项目有音频,且(浅扩散不支持多歌手 ⇒)数据集是平铺的。
    if (diffPoolReady(backend, diffInfo))
        return true;
    if (!ds || ds.files === 0)
        return false;
    // A flat dataset feeds any backend. A per-singer one only feeds the co-training backends —
    // python's fingerprint hard-fails on a subdirectory for the flat ones, so offering to start
    // would just move the refusal later and further from the cause.
    return ds.speakers.length === 0 || backendSupportsMultiSpeaker(backend);
}
/** Exported so the run segment can substitute it when the one live snapshot belongs to a
 *  DIFFERENT project than the page is currently pointed at (S76 batch 4). */
export const IDLE_SNAPSHOT = {
    state: "idle",
    backend: "",
    model_name: "",
    model_slug: "",
    project_id: "",
    workspace: "",
    run_id: "",
    total_epochs: 0,
    ckpts: [],
    stop_requested: false,
    elapsed_secs: 0,
    stderr_tail: [],
};
const DEFAULT_CONFIG = {
    modelName: "",
    runId: "",
    backend: "rvc",
    version: "v2",
    sampleRate: "48k",
    totalEpoch: 200,
    batchSize: 6,
    saveEveryEpoch: 25,
    saveEveryWeights: true,
    keepOnlyLatest: true,
    cacheGpu: false,
    fp16: true,
    gpu: "",
    forceCpu: false,
    augCopies: 0,
    sovitsVersion: "4.1",
    sovitsTotalEpoch: 1000,
    sovitsBatchSize: 6,
    sovitsSaveEverySteps: 800,
    sovitsKeepCkpts: 3,
    sovitsFp16: false,
    sovitsVolEmbedding: true,
    sovitsLoudnorm: false,
    sovitsKmeans: false,
    sovitsAllInMem: false,
    sovitsAugCopies: 0,
    diffVersion: "4.1",
    diffTotalSteps: 100000,
    diffBatchSize: 48,
    diffSaveEverySteps: 2000,
    diffForceSaveSteps: 10000,
    diffKStepMax: 0,
    diffFp16: false,
    diffAugCopies: 0,
    diffCacheAllData: true,
    vocTotalSteps: 2000,
    vocSaveEverySteps: 500,
    vocBatchSize: 8,
    vocKeepCkpts: 5,
    vocCropMelFrames: 32,
    vocFreezeMpd: false,
    vocAugCopies: 0,
};
/** Client-side mirror of the Rust `training::HISTORY_CAP` (15_000): thin to half when
 *  exceeded. Must stay in lockstep with the backend cap, otherwise `refresh()` re-merges a
 *  local tail far longer than the authoritative history and the loss curve never flattens
 *  back down after a long run. */
const HISTORY_CAP = 15_000;
export const useTrainingStore = create((set, get) => ({
    snapshot: IDLE_SNAPSHOT,
    snapshotAt: Date.now(),
    history: [],
    dragOverSpeakerId: null,
    flashSpeaker: null,
    config: { ...DEFAULT_CONFIG },
    route: { seg: "projects", projectId: "" },
    starting: false,
    diffWsInfo: null,
    slotInfo: null,
    retrainIntent: false,
    poolFlat: false,
    poolCount: 0,
    projectDataset: null,
    projectHasProgress: false,
    setRoute: (r) => set({ route: r }),
    setProjectInfo: (d) => set({
        projectDataset: d?.dataset ?? null,
        poolCount: d?.dataset.files ?? 0,
        poolFlat: !!d && d.dataset.files > 0 && d.dataset.speakers.length === 0,
        // ★§F2⒝ 批 2 ④ —— 对每个 run 求或(与 ProjectDetail 的 `dataHasDependents` 同一条规则)。
        projectHasProgress: d?.slots.some((s) => s.ckptCount > 0 || s.runs.some((r) => r.hasResumePoint || r.info.has_main_progress)) ?? false,
    }),
    refreshProjectDataset: async () => {
        const pid = get().route.projectId;
        if (!pid) {
            get().setProjectInfo(null);
            return;
        }
        const epoch = projectEpoch;
        try {
            const d = await invoke("get_training_project", { projectId: pid });
            // the user may have left for another project while this was in flight — landing now would
            // describe project A while the page is on B (same rule as `addFiles`)
            if (epoch !== projectEpoch)
                return;
            get().setProjectInfo(d);
        }
        catch {
            if (epoch === projectEpoch)
                get().setProjectInfo(null);
        }
    },
    importIntoProject: async (files, speaker = null) => {
        const pid = get().route.projectId;
        if (!pid || files.length === 0)
            return;
        await invoke("import_project_dataset", { projectId: pid, files, speaker });
        await get().refreshProjectDataset();
        // one-shot「落到这位歌手了」pulse, resolved AFTER the refresh so a brand-new singer (whose
        // card did not exist when the import started) flashes too
        if (speaker) {
            const g = get().projectDataset?.groups.find((x) => (x.name || x.slug) === speaker);
            if (g)
                set({ flashSpeaker: { id: g.slug, nonce: ++flashSeq } });
        }
    },
    deleteFromProject: async (rels) => {
        const pid = get().route.projectId;
        if (!pid || rels.length === 0)
            return;
        await invoke("delete_project_dataset_files", { projectId: pid, rels });
        await get().refreshProjectDataset();
    },
    enterProject: (projectId, seg = "detail") => {
        // Invalidate everything that is mid-flight FOR THE OLD PROJECT: an in-progress file probe
        // (`addFiles`) and an in-progress archive scan (`refreshProjectCkpts`) would otherwise land
        // after this clear and repopulate the new project's form with the old project's contents.
        projectEpoch++;
        ckptScanSeq++;
        set((s) => {
            // Re-entering the project the LIVE run belongs to is not a fresh start: that run already
            // fixed the identity, and blanking it would leave its own run segment describing「—」.
            const live = s.snapshot;
            const isLiveRun = !!projectId && live.project_id === projectId && live.state !== "idle";
            const backend = (isLiveRun && asTrainingBackend(live.backend)) || DEFAULT_CONFIG.backend;
            return {
                route: { seg: projectId ? seg : "projects", projectId },
                // Everything below belongs to the project we are LEAVING. Keeping any of it is how a run
                // ends up describing one project while writing into another.
                // `backend` too: the data segment renders a singer list or a flat list depending on it,
                // so leaving the previous project's pick behind means「导入数据」opens the wrong shape.
                config: { ...s.config, modelName: isLiveRun ? live.model_name : "", backend },
                diffWsInfo: null,
                slotInfo: null,
                retrainIntent: false,
                // Unknown until the new project's detail loads and re-derives it — leaving the old
                // project's value would let a run skip the data page on a project that has none.
                poolFlat: false,
                poolCount: 0,
                projectDataset: null,
                projectHasProgress: false,
                projectCkpts: [],
            };
        });
    },
    goSeg: (seg) => set((s) => seg === "projects" || s.route.projectId
        ? { route: { seg, projectId: seg === "projects" ? "" : s.route.projectId } }
        : {}),
    setDiffWsInfo: (info) => set({ diffWsInfo: info }),
    setSlotInfo: (info) => set({ slotInfo: info }),
    setRetrainIntent: (v) => set({ retrainIntent: v }),
    updateConfig: (u) => set((s) => ({ config: { ...s.config, ...u } })),
    setDragOverSpeakerId: (id) => set({ dragOverSpeakerId: id }),
    clearFlashSpeaker: (nonce) => set((s) => (s.flashSpeaker?.nonce === nonce ? { flashSpeaker: null } : {})),
    projectCkpts: [],
    refreshProjectCkpts: async (projectId, backend) => {
        const seq = ++ckptScanSeq;
        if (!projectId) {
            if (seq === ckptScanSeq)
                set({ projectCkpts: [] });
            return;
        }
        try {
            const rows = await invoke("list_project_ckpts", {
                projectId,
                family: backendFamily(backend),
            });
            // Drop a response a newer scan has already superseded — switching architecture cards
            // re-fires this, so out-of-order replies are the normal case, not the exotic one.
            if (seq === ckptScanSeq)
                set({ projectCkpts: rows });
        }
        catch (e) {
            console.error("project ckpt scan failed", e);
            if (seq === ckptScanSeq)
                set({ projectCkpts: [] });
        }
    },
    refresh: async () => {
        // No Tauri → no backend → nothing to refresh. Bail silently so dev-preview
        // doesn't spam "Cannot read properties of undefined (reading 'invoke')" errors.
        if (!isTauri())
            return;
        try {
            const snapshot = await invoke("get_training_status");
            const history = await invoke("get_training_history");
            set((s) => {
                // during a live run, step events may have appended points while we
                // awaited — keep the local tail newer than the fetched copy instead of
                // clobbering it (it re-syncs fully at done anyway)
                const lastFetched = history.length ? history[history.length - 1].step : -1;
                const localTail = s.history.filter((p) => p.step > lastFetched);
                return {
                    snapshot,
                    history: localTail.length ? [...history, ...localTail] : history,
                    snapshotAt: Date.now(),
                };
            });
        }
        catch (e) {
            console.error("training refresh failed", e);
        }
    },
    start: async (fresh, wipeConfirmed, resumeFrom) => {
        // S64 release gating (the S43 decision): no REAL training interpreter (dev venv / runtime pack /
        // manual slot) → the spawn is doomed on an end-user machine; offer the runtime download instead.
        // Dev machines always resolve training/.venv, so this only ever fires on packaged installs.
        try {
            if (!(await invoke("training_env_ready"))) {
                const c = await useAppStore.getState().showConfirm({
                    title: i18n.t("training.envMissingTitle"),
                    body: i18n.t("training.envMissingBody"),
                    buttons: [
                        { id: "cancel", label: i18n.t("common.cancel") },
                        { id: "goto", label: i18n.t("training.envMissingGoto"), kind: "primary" },
                    ],
                });
                if (c === "goto" && !useAppStore.getState().settingsOpen)
                    useAppStore.getState().toggleSettings();
                return;
            }
        }
        catch {
            /* ready-check unavailable → fall through; start_training's own error still surfaces loudly */
        }
        // S66 base-model gating: resolve the run's required assets through the SAME Rust source
        // try_start verifies against (training_required_assets ↔ resolve_training_assets) and turn
        // "missing base model" from a toast into a dialog with a one-click pack download. The param
        // derivation mirrors the request builder below (backend → version/sample_rate/aug fields).
        try {
            const cfg = get().config;
            const assetParams = cfg.backend === "rvc"
                ? { backend: "rvc", version: cfg.version, sampleRate: cfg.sampleRate, augCopies: cfg.augCopies }
                : cfg.backend === "vocoder"
                    ? { backend: "vocoder", version: "nsf_hifigan", sampleRate: "44k", augCopies: cfg.vocAugCopies }
                    : cfg.backend === "sovits_diff"
                        ? { backend: "sovits_diff", version: cfg.diffVersion, sampleRate: "44k", augCopies: 0 }
                        : cfg.backend === "sovits_v2"
                            ? { backend: "sovits_v2", version: "4.0-v2", sampleRate: "44k", augCopies: cfg.sovitsAugCopies }
                            : { backend: "sovits", version: cfg.sovitsVersion, sampleRate: "44k", augCopies: cfg.sovitsAugCopies };
            const assets = await invoke("training_required_assets", assetParams);
            const missing = assets.filter((a) => !a.exists);
            if (missing.length > 0) {
                // S75: ONE dialog. The S66 split existed because the license-bound vocoder ckpt was the one
                // asset the download button could NOT fetch — now it is mirrored like everything else, so
                // the split has nothing left to separate. What survives from that review is the real
                // obligation: state the terms BEFORE fetching (license line + upstream link below).
                const packItems = missing.filter((a) => a.pack);
                if (packItems.length > 0) {
                    const packs = [...new Set(packItems.map((a) => a.pack))];
                    const files = packItems.map((a) => `· ${a.path}`).join("\n");
                    // A licensed pack also carries its NOTICE files (attribution travels WITH the weights),
                    // so the download may write MORE files than this list — never fewer.
                    const licenses = [
                        ...new Set(packItems.map((a) => a.license).filter((l) => !!l)),
                    ];
                    const upstream = packItems.find((a) => a.selfUrl)?.selfUrl ?? null;
                    const buttons = [{ id: "cancel", label: i18n.t("common.cancel") }];
                    if (upstream)
                        buttons.push({ id: "open", label: i18n.t("training.assetsUpstream") });
                    buttons.push({ id: "dl", label: i18n.t("training.assetsMissingDl"), kind: "primary" });
                    const c = await useAppStore.getState().showConfirm({
                        title: i18n.t("training.assetsMissingTitle"),
                        body: `${i18n.t("training.assetsMissingBody")}\n${files}` +
                            (licenses.length > 0
                                ? `\n\n${i18n.t("training.assetsLicenseNote", { license: licenses.join(" / ") })}`
                                : ""),
                        buttons,
                    });
                    if (c === "open" && upstream)
                        void openUrl(upstream).catch(() => { });
                    if (c === "dl") {
                        if (!useAppStore.getState().settingsOpen)
                            useAppStore.getState().toggleSettings();
                        const hfBase = hfBaseForMirror(useMsstModelStore.getState().mirror);
                        // pack downloads are single-flight Rust-side — chain them; progress/errors surface
                        // in the Settings "Model Assets" section the dialog just opened.
                        void (async () => {
                            for (const p of packs) {
                                try {
                                    await invoke("download_asset_pack", { id: p, hfBase });
                                }
                                catch {
                                    /* shown by the Settings asset section (busy/cancel/fail all covered there) */
                                }
                            }
                        })();
                    }
                    return;
                }
                // missing but not pack-distributed (unexpected) → fall through; try_start errors loudly.
            }
        }
        catch {
            /* pre-flight unavailable → fall through; start_training's own gate still rejects loudly */
        }
        const { config, route } = get();
        set({ starting: true });
        try {
            // ①c: SoVITS (α) + RVC (α′) data is the singer list. 1 singer = single-speaker: send its
            // files as the flat dataset_files, NO `speakers` key -> byte-identical to pre-①c. ≥2
            // singers: send `speakers` (matching Rust StartTrainingRequest.speakers:
            // Vec<SpeakerGroup{name,files}>) + empty dataset_files. diff/vocoder keep the flat `dataset`.
            const isMulti = backendSupportsMultiSpeaker(config.backend);
            // S78: the data lives on disk BEFORE a run starts — importing is its own act now, so a
            // start never carries files. A per-singer project instead DECLARES its structure (every
            // group's `files` empty), which the backend validates against the directories that are
            // actually there and then consumes without copying anything. Names come from the disk
            // listing so the slug they derive to is the slug the data is already under.
            const diskGroups = get().projectDataset?.groups ?? [];
            const multi = isMulti && diskGroups.length > 1;
            const datasetFiles = [];
            const base = {
                model_name: config.modelName.trim(),
                // S76 batch 4: WHICH project, explicitly. `model_name` is now「本次训练名」— editable
                // and only the ARTIFACT identity — so letting the backend re-derive the directory from
                // it would fork a second project the first time somebody renames a run.
                project_id: route.projectId,
                backend: config.backend,
                dataset_files: datasetFiles,
                ...(multi
                    ? {
                        speakers: diskGroups.map((g) => ({
                            name: g.name || g.slug,
                            files: [],
                        })),
                    }
                    : {}),
                gpu: config.gpu,
                force_cpu: config.forceCpu,
                spk_id: 0,
                fresh,
                wipe_confirmed: wipeConfirmed,
                // ⚠ empty string, not omitted: the Rust field is `#[serde(default)] String` and the
                // request builder normalizes "" -> "latest", so an old payload and a fresh run behave
                // exactly as they did before this existed.
                resume_from: fresh ? "" : (resumeFrom ?? "latest"),
                // ★§F2⒝ 批 2 ④ —— 同样是「空串而不是省略」:Rust 那边是 `#[serde(default)] String`,
                // 而 `run_id_of` 把空串归成 `None`,于是旧载荷与今天的每槽一 run 行为逐字节相同。
                run_id: config.runId,
            };
            const request = config.backend === "rvc"
                ? {
                    ...base,
                    version: config.version,
                    sample_rate: config.sampleRate,
                    total_epoch: config.totalEpoch,
                    batch_size: config.batchSize,
                    save_every_epoch: config.saveEveryEpoch,
                    save_every_weights: config.saveEveryWeights,
                    keep_only_latest: config.keepOnlyLatest,
                    cache_gpu: config.cacheGpu,
                    fp16: config.fp16,
                    aug_copies: config.augCopies,
                }
                : config.backend === "vocoder"
                    ? {
                        ...base,
                        // fixed markers, not user choices (一期单格式类); total_epoch 0
                        // = the step-based sentinel (the UI hides epoch displays)
                        version: "nsf_hifigan",
                        sample_rate: "44k",
                        total_epoch: 0,
                        batch_size: config.vocBatchSize,
                        total_steps: config.vocTotalSteps,
                        save_every_steps: config.vocSaveEverySteps,
                        keep_ckpts: config.vocKeepCkpts,
                        crop_mel_frames: config.vocCropMelFrames,
                        freeze_mpd: config.vocFreezeMpd,
                        aug_copies: config.vocAugCopies,
                    }
                    : config.backend === "sovits_diff"
                        ? {
                            ...base,
                            version: config.diffVersion,
                            sample_rate: "44k",
                            // sentinel: diffusion progress is step-based; the UI hides
                            // epoch displays when total_epochs is 0
                            total_epoch: 0,
                            batch_size: config.diffBatchSize,
                            save_every_steps: config.diffSaveEverySteps,
                            total_steps: config.diffTotalSteps,
                            k_step_max: config.diffKStepMax,
                            interval_force_save: config.diffForceSaveSteps,
                            cache_all_data: config.diffCacheAllData,
                            fp16: config.diffFp16,
                            // honoured only when the sovits slot holds no main model (see `eff_aug_copies`)
                            aug_copies: config.diffAugCopies,
                        }
                        : config.backend === "sovits_v2"
                            ? {
                                ...base,
                                // S68 VISinger2: fixed version marker; shares the sovits* form
                                // fields. No vol_embedding/all_in_mem (v2 has neither); fp16
                                // structurally off (pure fp32 upstream — Rust normalizes too)
                                version: "4.0-v2",
                                sample_rate: "44k",
                                total_epoch: config.sovitsTotalEpoch,
                                batch_size: config.sovitsBatchSize,
                                save_every_steps: config.sovitsSaveEverySteps,
                                keep_ckpts: config.sovitsKeepCkpts,
                                fp16: false,
                                loudnorm: config.sovitsLoudnorm,
                                kmeans: config.sovitsKmeans,
                                aug_copies: config.sovitsAugCopies,
                            }
                            : {
                                ...base,
                                version: config.sovitsVersion,
                                sample_rate: "44k",
                                total_epoch: config.sovitsTotalEpoch,
                                batch_size: config.sovitsBatchSize,
                                save_every_steps: config.sovitsSaveEverySteps,
                                keep_ckpts: config.sovitsKeepCkpts,
                                fp16: config.sovitsFp16,
                                // 响度嵌入 is a 4.1 feature — the 4.0 card trains ecosystem-
                                // compatible checkpoints, so it stays structurally off there
                                vol_embedding: config.sovitsVersion === "4.1" ? config.sovitsVolEmbedding : false,
                                loudnorm: config.sovitsLoudnorm,
                                kmeans: config.sovitsKmeans,
                                all_in_mem: config.sovitsAllInMem,
                                aug_copies: config.sovitsAugCopies,
                            };
            await invoke("start_training", { request });
            // Only on success, and before the refresh — a rejected start must leave the user on the
            // page they pressed the button from, with their form intact.
            set({ route: { seg: "run", projectId: route.projectId }, history: [] });
            useAppStore.getState().showToast(i18n.t("training.started"), "info");
            await get().refresh();
        }
        catch (e) {
            // Localize known backend CODEs (APP_BUSY: an audition/render holds the FlightGuard); the raw
            // error still rethrows unchanged for any caller that inspects it.
            useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
            throw e;
        }
        finally {
            set({ starting: false });
        }
    },
    stop: async () => {
        try {
            await invoke("stop_training");
            set((s) => ({ snapshot: { ...s.snapshot, stop_requested: true } }));
        }
        catch (e) {
            useAppStore.getState().showToast(String(e), "error");
        }
    },
    forceStop: async () => {
        try {
            await invoke("force_stop_training");
        }
        catch (e) {
            useAppStore.getState().showToast(String(e), "error");
        }
    },
    resetRun: async () => {
        try {
            await invoke("reset_training_display");
            // only clear locally once the backend agreed (it refuses while running)
            set({ snapshot: IDLE_SNAPSHOT, history: [], snapshotAt: Date.now() });
            return true;
        }
        catch (e) {
            // APP_BUSY: an audition/render holds the FlightGuard → info, real failures stay errors.
            useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
            return false;
        }
    },
}));
let unlistens = null;
let installing = false;
/** Idempotent global listener install (App mount) — keeps the titlebar indicator
 *  and the loss history live even while the training page is closed. The sync
 *  sentinel closes the await window (StrictMode double-mount would double-install
 *  and duplicate every history point + toast). */
export async function setupTrainingListeners() {
    if (unlistens || installing)
        return;
    // No Tauri WebView → no Rust event bridge → nothing to listen for.
    // Bail early so dev-preview / E2E / accidental-browser-open doesn't spam the console
    // with "Cannot read properties of undefined (reading 'invoke')" every reload.
    if (!isTauri())
        return;
    installing = true;
    try {
        unlistens = await Promise.all([
            listen("training-stage", (e) => {
                useTrainingStore.setState((s) => ({
                    snapshot: { ...s.snapshot, stage: e.payload },
                }));
            }),
            listen("training-step", (e) => {
                useTrainingStore.setState((s) => {
                    let history = s.history;
                    if (history.length >= HISTORY_CAP) {
                        history = history.filter((_, i) => i % 2 === 0);
                    }
                    // NB: snapshotAt is NOT touched here — it anchors the elapsed
                    // extrapolation to the last full refresh (elapsed_secs base); resetting
                    // it per step would freeze the displayed elapsed at the base value
                    return {
                        snapshot: { ...s.snapshot, state: "running", step: e.payload },
                        history: [
                            ...history,
                            { step: e.payload.step, lr: e.payload.lr, losses: e.payload.losses },
                        ],
                    };
                });
            }),
            listen("training-ckpt", (e) => {
                useTrainingStore.setState((s) => {
                    const kept = e.payload.kind === "best" || e.payload.kind === "final"
                        ? s.snapshot.ckpts.filter((c) => c.kind !== e.payload.kind)
                        : s.snapshot.ckpts;
                    return { snapshot: { ...s.snapshot, ckpts: [...kept, e.payload] } };
                });
            }),
            listen("training-done", (e) => {
                useTrainingStore.setState({ snapshot: e.payload, snapshotAt: Date.now() });
                const t = i18n.t.bind(i18n);
                const app = useAppStore.getState();
                if (e.payload.state === "completed") {
                    app.showToast(t("training.doneCompleted"), "success");
                }
                else if (e.payload.state === "stopped") {
                    app.showToast(t("training.doneStopped"), "info");
                }
                else if (e.payload.state === "error") {
                    // snapshot.error carries the run_worker's stable CODE strings — localize known ones.
                    const err = e.payload.error ?? "";
                    app.showToast(`${t("training.doneError")}: ${backendErrorMessage(err) ?? err}`, "error");
                }
                // the final force-emitted step may have landed Rust-side only — resync once
                void useTrainingStore.getState().refresh();
            }),
            listen("training-state", (e) => {
                if (e.payload === "running") {
                    useTrainingStore.setState((s) => ({
                        snapshot: { ...s.snapshot, state: "running" },
                    }));
                }
            }),
        ]);
    }
    catch {
        /* Not in a Tauri WebView — no event listeners available. reset sentinel so
           a real-Tauri restart has a clean shot (setupTrainingListeners is only
           called once at App mount, but in dev HMR can re-run it). */
        installing = false;
        unlistens = null;
    }
}
