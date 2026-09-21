import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { useMsstModelStore, setupDownloadListener } from "../../store/msst-models";
import { useAmtModelStore, setupAmtDownloadListener } from "../../store/amt-models";
import { useAppStore } from "../../store/app";
import { MSST_CATALOG, ALL_CATEGORIES, CATEGORY_LABELS, ARCHITECTURE_LABELS, MSST_DEFAULT_PRECISION, MSST_FP16_ARCHS, MSST_FP16_TIP, ghRouteOrder, hfBaseForMirror, urlNeedsVpn, t18, } from "../../lib/models/msst-catalog";
import { AMT_CATALOG } from "../../lib/models/amt-catalog";
import { useFloatingPanel } from "../../lib/useFloatingPanel";
import { PanelResizeHandles } from "../common/PanelResizeHandles";
import { backendErrorMessage, isBusyError, isCancelError } from "../../lib/backendError";
import { maybeShowErrorModal } from "../../lib/errorDisplay";
import { logToBackend } from "../../lib/log";
import { VOICE_STRINGS } from "../workflow/nodes/VoiceModelPicker";
import { downloadSongModel, listSongModels, deleteSongModel } from "../../lib/backendSong";
import { SONG_MODEL_CATALOG, SONG_FAMILY_LABELS, getSongInstallStatus, songModelTotalSize, } from "../../lib/models/song-catalog";
import { useVoiceModelStore, voiceVersionBadge, voiceSpeakerOptions, formatSampleRateKhz, vocoderFormatMatches, vocoderFormatLabel, } from "../../store/voice-models";
import { runRangeTest, runRangeTestBatch, collectRangeTestTargets, midiName, deriveCautionZones, SCAN_VERSION } from "../../lib/vocal/rangeTest";
import { targetRange } from "../../lib/vocal/rangeBounds";
import { preview } from "../common/previewPlayer";
import { RangeBoundsEditor } from "../vocal/RangeBoundsEditor";
import { readFile } from "@tauri-apps/plugin-fs";
import { exportOneAudioFileToFolder, laneExportErrorMessage } from "../../lib/audio/exportLaneAudio";
import "./MsstModelManager.css";
function StatusIcon({ type, title, onClick }) {
    if (type === "installed") {
        return (_jsx("span", { className: "model-status-icon installed", title: title, children: _jsx("svg", { viewBox: "0 0 24 24", width: "16", height: "16", children: _jsx("path", { fill: "none", stroke: "currentColor", strokeWidth: "3", strokeLinecap: "round", strokeLinejoin: "round", d: "M5 13l4 4L19 7" }) }) }));
    }
    if (type === "downloading") {
        return (_jsx("span", { className: "model-status-icon downloading", title: title, children: _jsx("svg", { viewBox: "0 0 24 24", width: "16", height: "16", children: _jsx("path", { fill: "none", stroke: "currentColor", strokeWidth: "3", strokeLinecap: "round", d: "M12 3a9 9 0 1 0 9 9" }) }) }));
    }
    return (_jsx("button", { className: "model-status-icon primary", onClick: onClick, title: title, children: _jsx("svg", { viewBox: "0 0 24 24", width: "14", height: "14", children: _jsx("path", { fill: "none", stroke: "currentColor", strokeWidth: "3", strokeLinecap: "round", strokeLinejoin: "round", d: "M12 5v14M5 12l7 7 7-7" }) }) }));
}
function VpnBadge({ lang }) {
    const label = lang === "zh" ? "需魔法" : lang === "ja" ? "要VPN" : "VPN";
    const tip = lang === "zh"
        ? "该资源无镜像线路，需可访问外网（魔法）才能下载"
        : lang === "ja"
            ? "このリソースにはミラー経路がなく、外部ネットワークへのアクセスが必要です"
            : "No mirror route — requires external network access to download";
    return _jsx("span", { className: "model-vpn-badge", title: tip, children: label });
}
/** Reusable AMT model card (转谱模型 / 歌词识别 / 运行环境 tabs share this — one look, one
 *  download/delete flow). Renders the status icon, name + VPN/必备 badges, description, newbie
 *  tip, size/rating/features, download progress (download / extract), and the installed delete row. */
function AmtModelCard({ entry, lang, isInstalled, dl, onDownload, confirming, onRequestDelete, onConfirmDelete, onCancelDelete, }) {
    const isDownloading = !!dl;
    return (_jsxs("div", { className: `msst-model-card-wrap ${isInstalled ? "installed" : ""}`, children: [_jsx("div", { className: "msst-model-card-status", children: _jsx(StatusIcon, { type: isInstalled ? "installed" : isDownloading ? "downloading" : "pending", title: isInstalled ? (lang === "zh" ? "已安装" : "Installed") : isDownloading ? (lang === "zh" ? "正在下载" : "Downloading") : (lang === "zh" ? "下载" : "Download"), onClick: isInstalled || isDownloading ? undefined : onDownload }) }), _jsxs("div", { className: "msst-model-card", children: [_jsxs("div", { className: "model-card-header", children: [_jsxs("span", { className: "model-card-name", children: [t18(entry.name, lang), urlNeedsVpn(entry.downloadUrl) && _jsx(VpnBadge, { lang: lang }), entry.isEssential && (_jsx("span", { className: "essential-badge", children: lang === "zh" ? "必备" : "Required" }))] }), _jsx("span", { className: "model-card-arch", children: entry.architecture.toUpperCase() })] }), _jsx("p", { className: "model-card-desc", children: t18(entry.description, lang) }), entry.newbie_tip && (_jsxs("div", { className: "model-newbie-tip", children: [_jsx("span", { className: "tip-icon", children: "\uD83D\uDCA1" }), t18(entry.newbie_tip, lang)] })), _jsxs("div", { className: "model-card-meta", children: [_jsx("span", { className: "model-card-size", children: formatSize(entry.fileSize) }), entry.rating && (_jsxs("span", { className: "model-rating", children: ["★".repeat(entry.rating), "☆".repeat(5 - entry.rating)] })), entry.ui_features && (_jsx("span", { className: "model-card-stems", children: entry.ui_features.map((f) => t18(f, lang)).join(" / ") }))] }), isDownloading && (_jsx("div", { className: "model-download-progress", children: dl.stage === "extract" ? (_jsxs(_Fragment, { children: [_jsx("div", { className: "model-download-bar model-convert-bar", style: { width: "100%" } }), _jsx("span", { className: "model-download-text", children: lang === "zh" ? "正在解压安装..." : "Extracting..." })] })) : (_jsxs(_Fragment, { children: [_jsx("div", { className: "model-download-bar", style: { width: dl.total > 0 ? `${(dl.downloaded / dl.total) * 100}%` : "0%" } }), _jsxs("span", { className: "model-download-text", children: [formatSize(dl.downloaded), " / ", dl.total > 0 ? formatSize(dl.total) : "..."] })] })) })), isInstalled && (_jsxs("div", { className: "model-card-actions", children: [_jsx("span", { className: "model-status-installed", children: lang === "zh" ? "已安装" : "Installed" }), confirming ? (_jsxs("div", { className: "model-confirm-delete", children: [_jsx("button", { className: "danger", onClick: onConfirmDelete, children: lang === "zh" ? "确认" : "OK" }), _jsx("button", { onClick: onCancelDelete, children: lang === "zh" ? "取消" : "Cancel" })] })) : (_jsx("button", { className: "model-delete-btn", onClick: onRequestDelete, children: lang === "zh" ? "删除" : "Delete" }))] }))] })] }));
}
export function MsstModelManager({ onClose }) {
    const { t, i18n } = useTranslation();
    const lang = i18n.language;
    const showToast = useAppStore((s) => s.showToast);
    const { installed, downloading, error, fetchInstalled, fetchModelsDir, modelsDir, clearError, deleteModel, downloadEntry, convertPrecision, } = useMsstModelStore();
    const [pyenvProgress, setPyenvProgress] = useState(null);
    const [isInstallingPyenv, setIsInstallingPyenv] = useState(false);
    const [sidecarInstalled, setSidecarInstalled] = useState(null);
    const [sidecarPath, setSidecarPath] = useState("");
    const { style: panelStyle, startDrag, startResize } = useFloatingPanel({
        storageKey: "utai.msstManagerRect",
        initial: () => ({ x: 100, y: 96, w: 440, h: Math.round(window.innerHeight * 0.72) }),
        minW: 380,
        minH: 320,
    });
    const [topTab, setTopTab] = useState("separation");
    const [category, setCategory] = useState("vocals");
    const [confirmDelete, setConfirmDelete] = useState(null);
    // Download-time precision choice per catalog entry (roformers only); absent = arch default.
    const [dlPrecision, setDlPrecision] = useState({});
    const { installed: amtInstalled, downloading: amtDownloading, fetchInstalled: fetchAmtInstalled, fetchModelsDir: fetchAmtModelsDir, deleteModel: deleteAmtModel, downloadEntry: downloadAmtEntry, } = useAmtModelStore();
    useEffect(() => {
        fetchModelsDir();
        fetchInstalled();
        setupDownloadListener();
        fetchAmtModelsDir();
        fetchAmtInstalled();
        setupAmtDownloadListener();
        invoke("amt_sidecar_installed").then(setSidecarInstalled).catch(() => setSidecarInstalled(false));
        invoke("amt_sidecar_path").then(setSidecarPath).catch(() => { });
    }, [fetchModelsDir, fetchInstalled, fetchAmtModelsDir, fetchAmtInstalled]);
    useEffect(() => {
        const unlisten = listen("pyenv-progress", (event) => {
            setPyenvProgress(event.payload);
            if (event.payload.phase === "done") {
                setSidecarInstalled(true);
                setIsInstallingPyenv(false);
                showToast(t("amt.envInstalled") || "AMT 运行环境安装成功", "success");
                setTimeout(() => setPyenvProgress(null), 3000);
            }
            else if (event.payload.phase === "error") {
                setIsInstallingPyenv(false);
                showToast(event.payload.message, "error");
            }
        });
        return () => { void unlisten.then(fn => fn()); };
    }, [t, showToast]);
    const handleDownloadRuntime = useCallback(async () => {
        if (isInstallingPyenv)
            return;
        setIsInstallingPyenv(true);
        setPyenvProgress({ id: "init", phase: "init", progress: 0, message: "正在初始化...", params: [] });
        try {
            const { mirror, ghMirror, ghPresets } = useMsstModelStore.getState();
            const hf_base = hfBaseForMirror(mirror) || undefined;
            const gh_routes = ghRouteOrder(ghMirror, ghPresets);
            await invoke("download_runtime_pack", {
                id: "runtime-cpu-v1",
                hf_base,
                gh_routes
            });
        }
        catch (e) {
            showToast(String(e), "error");
            setIsInstallingPyenv(false);
        }
    }, [isInstallingPyenv, showToast]);
    // ── §user 一键下载全部必备组件 ─────────────────────────────────────────────
    // 必备 = AMT_CATALOG 里 isEssential 的条目（FFmpeg/FluidSynth/默认音源/默认转谱
    // 模型/节拍模型/Whisper 歌词模型/MuseScore 乐谱运行时）+ AMT Python 推理引擎。
    // 逐项串行下载（复用单条下载链路：镜像轮询、断点续传、SHA256、进度卡），
    // 单项失败只弹 toast 不中断后续项。
    const [bulkEssential, setBulkEssential] = useState({ active: false, current: 0, total: 0, label: "" });
    const missingEssentials = useMemo(() => AMT_CATALOG.filter((e) => e.isEssential && !amtInstalled.find((m) => m.id === e.id)?.is_available), [amtInstalled]);
    const handleOneClickEssentials = useCallback(async () => {
        if (bulkEssential.active)
            return;
        // 实时读 store（列表可能滞后于最近一次下载完成）
        const missing = AMT_CATALOG.filter((e) => e.isEssential && !useAmtModelStore.getState().installed.find((m) => m.id === e.id)?.is_available);
        const needRuntime = sidecarInstalled === false;
        if (missing.length === 0 && !needRuntime) {
            showToast(t18({ zh: "所有必备组件都已就绪，无需下载", en: "All essential components are ready", ja: "必須コンポーネントはすべて揃っています" }, lang), "info");
            return;
        }
        const total = missing.length + (needRuntime ? 1 : 0);
        setBulkEssential({ active: true, current: 0, total, label: "" });
        // §user「一键下载并行 2-3 线程」：小文件（<100MB，如 FluidSynth/FFmpeg/小模型）
        // 最多 3 个并行；大文件（≥100MB，如音源/大模型/推理引擎）串行，大小队列同时
        // 推进——总耗时约省 40%。单项失败只弹 toast（downloadAmtEntry 内部处理），
        // 不中断其它项。
        let done = 0;
        const setLabel = (label) => setBulkEssential((b) => (b.active ? { ...b, current: done, label } : b));
        const bump = () => {
            done++;
            setBulkEssential((b) => (b.active ? { ...b, current: done } : b));
        };
        try {
            const SMALL_THRESHOLD = 100_000_000;
            const small = missing.filter((e) => e.fileSize < SMALL_THRESHOLD);
            const large = missing.filter((e) => e.fileSize >= SMALL_THRESHOLD);
            const runtimeLabel = t18({ zh: "AMT 推理引擎（约 1.8 GB）", en: "AMT inference engine (~1.8 GB)", ja: "AMT 推論エンジン（約1.8GB）" }, lang);
            // 小文件并行池（≤3 线程）。
            const smallPool = (async () => {
                let idx = 0;
                const workers = [];
                for (let w = 0; w < Math.min(3, small.length); w++) {
                    workers.push((async () => {
                        while (idx < small.length) {
                            const entry = small[idx++];
                            setLabel(t18(entry.name, lang));
                            await downloadAmtEntry(entry);
                            bump();
                        }
                    })());
                }
                await Promise.all(workers);
            })();
            // 大文件 + 推理引擎串行队列（与小文件池并行推进）。
            const largeQueue = (async () => {
                if (needRuntime) {
                    setLabel(runtimeLabel);
                    await handleDownloadRuntime();
                    bump();
                }
                for (const entry of large) {
                    setLabel(t18(entry.name, lang));
                    await downloadAmtEntry(entry);
                    bump();
                }
            })();
            await Promise.all([smallPool, largeQueue]);
            // 完成统计按刷新后的实际可用性数（失败项不虚报）。
            await useAmtModelStore.getState().fetchInstalled().catch(() => { });
            const runtimeOk = await invoke("amt_sidecar_installed").catch(() => false);
            const okCount = AMT_CATALOG.filter((e) => e.isEssential && useAmtModelStore.getState().installed.find((m) => m.id === e.id)?.is_available).length + (runtimeOk ? 1 : 0);
            showToast(t18({
                zh: "必备组件下载完成（" + okCount + "/" + total + "）。失败项可稍后单独重试或更换镜像线路。",
                en: "Essential components done (" + okCount + "/" + total + "). Failed items can be retried individually or via another mirror.",
                ja: "必須コンポーネント完了（" + okCount + "/" + total + "）。失敗項目は個別に再試行できます。",
            }, lang), okCount === total ? "success" : "info");
            // §user「下载完成后自动弹『下一步做什么』提示」：新手零摸索。
            void useAppStore.getState().showConfirm({
                title: t18({ zh: "必备组件已就绪，下一步做什么？", en: "Essentials ready — what's next?", ja: "必須コンポーネント準備完了。次は？" }, lang),
                body: t18({
                    zh: "① 转谱：关闭本面板 → 选择歌曲音频 → 选模型开始转换\n② 歌词：结果工作台会自动提取人声歌词（按音符铺词），双击音符即可改词\n③ 音源：工作台右上角音源下拉切换，切换后自动 2 秒预览试听",
                    en: "1) Transcribe: close this panel → pick a song → choose a model → start.\n2) Lyrics: the workbench auto-extracts per-note lyrics; double-click a note to edit.\n3) Soundfont: switch in the workbench header — a 2s preview plays automatically.",
                    ja: "① 採譜：このパネルを閉じ → 楽曲を選択 → モデル選択で開始\n② 歌詞：ワークベンチが自動で音符ごとの歌詞を抽出、音符ダブルクリックで編集\n③ 音源：ワークベンチ右上の音源セレクターで切替、2秒プレビューが自動再生",
                }, lang),
                buttons: [{ id: "ok", label: t18({ zh: "知道了", en: "Got it", ja: "了解" }, lang), kind: "primary" }],
            });
        }
        finally {
            setBulkEssential({ active: false, current: 0, total: 0, label: "" });
        }
    }, [bulkEssential.active, sidecarInstalled, downloadAmtEntry, handleDownloadRuntime, lang, showToast]);
    const installedFilenames = new Set(installed.map((m) => m.filename));
    const filtered = MSST_CATALOG.filter((m) => m.category === category);
    const handleDownload = useCallback(async (entry) => {
        // Only the fp16-verified roformers get a precision choice; other archs download as before.
        const precision = MSST_FP16_ARCHS.has(entry.architecture)
            ? (dlPrecision[entry.id] ?? MSST_DEFAULT_PRECISION[entry.architecture])
            : undefined;
        await downloadEntry(entry, precision);
    }, [downloadEntry, dlPrecision]);
    const handleMsstImport = useCallback(async () => {
        const path = await open({
            title: lang === "zh" ? "选择 MSST 模型文件" : "Select MSST Model File",
            filters: [{ name: "Model", extensions: ["ckpt", "th", "pth", "onnx"] }],
        });
        if (path)
            await useMsstModelStore.getState().importLocal(path);
    }, [lang]);
    const handleDelete = useCallback(async (filename) => { await deleteModel(filename); setConfirmDelete(null); }, [deleteModel]);
    return (_jsxs("aside", { className: "msst-model-manager", style: panelStyle, children: [_jsxs("div", { className: "panel-header", onMouseDown: startDrag, children: [_jsx("span", { className: "panel-title", children: lang === "zh" ? "资源管理" : lang === "ja" ? "リソース管理" : "Resource Manager" }), _jsx("button", { className: "panel-close", onClick: onClose, children: "X" })] }), _jsx(PanelResizeHandles, { start: startResize }), error && _jsx("div", { className: "msst-error", onClick: clearError, children: backendErrorMessage(error) ?? error }), _jsxs("div", { className: "rm-top-tabs", children: [_jsx("button", { className: topTab === "separation" ? "active" : "", onClick: () => setTopTab("separation"), children: lang === "zh" ? "音频分离" : lang === "ja" ? "音声分離" : "Separation" }), _jsx("button", { className: topTab === "amt" ? "active" : "", onClick: () => setTopTab("amt"), children: lang === "zh" ? "MIDI 模型" : lang === "ja" ? "MIDIモデル" : "MIDI Models" }), _jsx("button", { className: topTab === "lyrics" ? "active" : "", onClick: () => setTopTab("lyrics"), children: lang === "zh" ? "歌词识别" : lang === "ja" ? "歌詞認識" : "Lyrics" }), _jsx("button", { className: topTab === "runtime" ? "active" : "", onClick: () => setTopTab("runtime"), children: lang === "zh" ? "运行环境" : lang === "ja" ? "実行環境" : "Runtime" }), _jsx("button", { className: topTab === "voice" ? "active" : "", onClick: () => setTopTab("voice"), children: lang === "zh" ? "声音模型" : lang === "ja" ? "ボイスモデル" : "Voice Models" })] }), _jsx("div", { className: "rm-download-note", children: lang === "zh"
                    ? "下载说明：HuggingFace 与 GitHub 资源可在设置中切换镜像加速；标注「需魔法」的资源无镜像线路，需可访问外网。"
                    : lang === "ja"
                        ? "ダウンロード：HuggingFace / GitHub は設定でミラー加速に切替可能。「要VPN」はミラー経路がなく、外部アクセスが必要です。"
                        : "Download note: HuggingFace & GitHub assets can use mirror acceleration in Settings; assets marked “VPN” have no mirror route and need external network access." }), topTab === "amt" && (_jsxs("div", { className: "msst-model-list", children: [_jsxs("div", { className: "amt-oneclick-bar", children: [_jsxs("div", { className: "amt-oneclick-info", children: [_jsx("span", { className: "amt-oneclick-title", children: t18({ zh: "⚡ 一键下载必备组件", en: "⚡ One-click Essentials", ja: "⚡ 必須コンポーネント一括DL" }, lang) }), _jsx("span", { className: "amt-oneclick-sub", children: bulkEssential.active
                                            ? t18({ zh: "已完成 " + bulkEssential.current + "/" + bulkEssential.total + " · 正在下载：" + bulkEssential.label + "（小文件并行）", en: "Done " + bulkEssential.current + "/" + bulkEssential.total + " · downloading: " + bulkEssential.label + " (small files in parallel)", ja: "完了 " + bulkEssential.current + "/" + bulkEssential.total + " ・ DL中：" + bulkEssential.label + "（小ファイル並列）" }, lang)
                                            : missingEssentials.length === 0 && sidecarInstalled === true
                                                ? t18({ zh: "全部必备组件已就绪 ✓", en: "All essentials ready ✓", ja: "すべて準備完了 ✓" }, lang)
                                                : t18({
                                                    zh: "待下载 " + missingEssentials.length + " 项（约 " + formatSize(missingEssentials.reduce((a, e) => a + e.fileSize, 0)) + (sidecarInstalled === false ? " + 推理引擎约 1.8 GB" : "") + "），含引擎/音源/默认模型/歌词模型",
                                                    en: missingEssentials.length + " pending (~" + formatSize(missingEssentials.reduce((a, e) => a + e.fileSize, 0)) + (sidecarInstalled === false ? " + ~1.8 GB engine" : "") + ") — engine, soundfont, models, lyrics",
                                                    ja: "未入手 " + missingEssentials.length + " 件（約 " + formatSize(missingEssentials.reduce((a, e) => a + e.fileSize, 0)) + "）",
                                                }, lang) })] }), _jsx("button", { className: "amt-oneclick-btn", disabled: bulkEssential.active || (missingEssentials.length === 0 && sidecarInstalled === true), title: t18({
                                    zh: "依次下载所有必备组件：AMT 推理引擎、FFmpeg、FluidSynth、默认音源、默认转谱/节拍模型、Whisper 歌词模型、MuseScore 乐谱运行时。下载完即可使用全部核心功能。",
                                    en: "Download every essential component in order: AMT engine, FFmpeg, FluidSynth, default soundfont, default transcription/beat models, Whisper lyrics model, MuseScore runtime.",
                                    ja: "必須コンポーネントを順に一括ダウンロードします。",
                                }, lang), onClick: () => void handleOneClickEssentials(), children: bulkEssential.active
                                    ? t18({ zh: "下载中…", en: "Downloading…", ja: "ダウンロード中…" }, lang)
                                    : t18({ zh: "一键下载", en: "Download All", ja: "一括DL" }, lang) })] }), _jsx("div", { className: "amt-section-header", style: { margin: "16px 8px 4px", color: "var(--text-secondary)", fontSize: "14px", fontWeight: "bold" }, children: lang === "zh" ? "转谱引擎（核心运行时）" : lang === "ja" ? "転写エンジン（コア実行環境）" : "Transcription Engine (Core Runtime)" }), _jsxs("div", { className: `msst-model-card-wrap ${sidecarInstalled ? "installed" : ""}`, children: [_jsx("div", { className: "msst-model-card-status", children: _jsx(StatusIcon, { type: sidecarInstalled ? "installed" : isInstallingPyenv ? "downloading" : "pending", title: sidecarInstalled ? (lang === "zh" ? "已安装" : "Installed") : isInstallingPyenv ? (lang === "zh" ? "正在安装" : "Installing") : (lang === "zh" ? "下载" : "Download"), onClick: handleDownloadRuntime }) }), _jsxs("div", { className: "msst-model-card", children: [_jsxs("div", { className: "model-card-header", children: [_jsxs("span", { className: "model-card-name", children: ["AMT \u00B7 ", t18({ zh: "全轨音频转 MIDI 推理引擎", en: "Full-track Audio-to-MIDI Engine", ja: "全トラック音声→MIDI推論エンジン" }, lang), _jsx("span", { className: "essential-badge", children: lang === "zh" ? "必备" : "Required" })] }), _jsx("span", { className: "model-card-arch", children: "Python \u00B7 Sidecar" })] }), _jsx("p", { className: "model-card-desc", children: t18({
                                            zh: "独立 Python 推理引擎，包含 PyTorch 与模型推理逻辑。支持 FP16/FP32 精度切换，是所有音乐转 MIDI 功能的核心依赖。",
                                            en: "Independent Python inference engine, including PyTorch and model logic. Core dependency for all music-to-midi features.",
                                            ja: "独立した Python 推論エンジン。PyTorch とモデル推論ロジックを含みます。すべての音声→MIDI機能のコア依存関係です。",
                                        }, lang) }), _jsxs("div", { className: "model-card-meta", children: [_jsx("span", { className: "model-card-size", children: "~1.8 GB" }), _jsx("span", { className: "model-card-stems", style: { fontSize: 10, opacity: 0.7, wordBreak: "break-all" }, children: sidecarPath || "..." })] }), isInstallingPyenv && pyenvProgress && (_jsxs("div", { className: "model-download-progress", children: [_jsx("div", { className: `model-download-bar ${pyenvProgress.phase !== "download" ? "model-convert-bar" : ""}`, style: { width: `${pyenvProgress.progress}%` } }), _jsxs("span", { className: "model-download-text", children: [pyenvProgress.message, " (", pyenvProgress.progress, "%)"] })] })), sidecarInstalled && (_jsx("div", { className: "model-card-actions", children: _jsx("span", { className: "model-status-installed", children: lang === "zh" ? "环境已就绪" : "Environment Ready" }) }))] })] }), _jsx("div", { className: "amt-section-header", style: { margin: "16px 8px 4px", color: "var(--text-secondary)", fontSize: "14px", fontWeight: "bold" }, children: lang === "zh" ? "转谱模型库" : lang === "ja" ? "転写モデル" : "Transcription Models" }), AMT_CATALOG.filter(e => !["fluidsynth", "soundfont", "musescore", "ffmpeg", "whisper", "bs_roformer", "polarformer"].includes(e.architecture)).map((entry) => {
                        const isInstalled = !!amtInstalled.find((m) => m.id === entry.id)?.is_available;
                        return (_jsx(AmtModelCard, { entry: entry, lang: lang, isInstalled: isInstalled, dl: amtDownloading[entry.id], onDownload: () => downloadAmtEntry(entry), confirming: confirmDelete === entry.id, onRequestDelete: () => setConfirmDelete(entry.id), onConfirmDelete: () => { deleteAmtModel(entry.filename); setConfirmDelete(null); }, onCancelDelete: () => setConfirmDelete(null) }, entry.id));
                    })] })), topTab === "lyrics" && (_jsxs("div", { className: "msst-model-list", children: [_jsx("div", { className: "amt-section-header", style: { margin: "16px 8px 4px", color: "var(--text-secondary)", fontSize: "14px", fontWeight: "bold" }, children: lang === "zh" ? "歌词识别模型" : lang === "ja" ? "歌詞認識モデル" : "Lyrics Recognition Models" }), AMT_CATALOG.filter(e => e.architecture === "whisper").map((entry) => {
                        const isInstalled = !!amtInstalled.find((m) => m.id === entry.id)?.is_available;
                        return (_jsx(AmtModelCard, { entry: entry, lang: lang, isInstalled: isInstalled, dl: amtDownloading[entry.id], onDownload: () => downloadAmtEntry(entry), confirming: confirmDelete === entry.id, onRequestDelete: () => setConfirmDelete(entry.id), onConfirmDelete: () => { deleteAmtModel(entry.filename); setConfirmDelete(null); }, onCancelDelete: () => setConfirmDelete(null) }, entry.id));
                    })] })), topTab === "runtime" && (_jsxs("div", { className: "msst-model-list", children: [_jsx(GameEngineTab, { lang: lang }), _jsx("div", { className: "amt-section-header", style: { margin: "16px 8px 4px", color: "var(--text-secondary)", fontSize: "14px", fontWeight: "bold" }, children: lang === "zh" ? "运行环境组件" : lang === "ja" ? "実行環境コンポーネント" : "Runtime Components" }), AMT_CATALOG.filter(e => ["fluidsynth", "ffmpeg", "musescore"].includes(e.architecture)).map((entry) => {
                        const isInstalled = !!amtInstalled.find((m) => m.id === entry.id)?.is_available;
                        return (_jsx(AmtModelCard, { entry: entry, lang: lang, isInstalled: isInstalled, dl: amtDownloading[entry.id], onDownload: () => downloadAmtEntry(entry), confirming: confirmDelete === entry.id, onRequestDelete: () => setConfirmDelete(entry.id), onConfirmDelete: () => { deleteAmtModel(entry.filename); setConfirmDelete(null); }, onCancelDelete: () => setConfirmDelete(null) }, entry.id));
                    })] })), topTab === "voice" && _jsx(VoiceModelsTab, { lang: lang }), topTab === "separation" && (_jsxs(_Fragment, { children: [_jsx("div", { className: "msst-filter", children: ALL_CATEGORIES.map((cat) => (_jsx("button", { className: category === cat ? "active" : "", onClick: () => setCategory(cat), children: t18(CATEGORY_LABELS[cat], lang) }, cat))) }), _jsxs("div", { className: "msst-model-list", children: [filtered.map((entry) => {
                                const isInstalled = installedFilenames.has(entry.filename);
                                const dl = downloading[entry.filename];
                                const isDownloading = !!dl;
                                const fp16Capable = MSST_FP16_ARCHS.has(entry.architecture);
                                const chosenPrecision = dlPrecision[entry.id] ?? MSST_DEFAULT_PRECISION[entry.architecture];
                                return (_jsxs("div", { className: `msst-model-card-wrap ${isInstalled ? "installed" : ""}`, children: [_jsx("div", { className: "msst-model-card-status", children: _jsx(StatusIcon, { type: isInstalled ? "installed" : isDownloading ? "downloading" : "pending", title: isInstalled ? (lang === "zh" ? "已安装" : "Installed") : isDownloading ? (lang === "zh" ? "正在下载" : "Downloading") : (lang === "zh" ? "下载" : "Download"), onClick: () => handleDownload(entry) }) }), _jsxs("div", { className: "msst-model-card", children: [_jsxs("div", { className: "model-card-header", children: [_jsxs("span", { className: "model-card-name", children: [t18(entry.name, lang), urlNeedsVpn(entry.downloadUrl) && _jsx(VpnBadge, { lang: lang })] }), _jsxs("span", { className: "model-card-arch", children: [ARCHITECTURE_LABELS[entry.architecture], entry.source === "community" && _jsx("span", { className: "model-card-community", children: " *" })] })] }), _jsx("p", { className: "model-card-desc", children: t18(entry.description, lang) }), _jsxs("div", { className: "model-card-meta", children: [_jsx("span", { className: "model-card-stems", children: entry.stems.join(" / ") }), entry.sdrScore && _jsxs("span", { className: "model-card-sdr", children: ["SDR ", entry.sdrScore] }), _jsx("span", { className: "model-card-size", children: formatSize(entry.fileSize) })] }), !isInstalled && !isDownloading && fp16Capable && (_jsxs("div", { className: "model-card-precision", children: [_jsx("span", { className: "model-precision-label", children: t18({ zh: "下载精度", en: "Precision", ja: "精度" }, lang) }), _jsx("div", { className: "model-precision-seg", title: t18(MSST_FP16_TIP, lang), children: ["fp32", "fp16"].map((p) => (_jsx("button", { className: chosenPrecision === p ? "active" : "", onClick: () => setDlPrecision((s) => ({ ...s, [entry.id]: p })), children: p }, p))) })] })), isDownloading && _jsx(DownloadBar, { dl: dl, lang: lang }), isInstalled && (_jsxs("div", { className: "model-card-actions", children: [_jsx("span", { className: "model-status-installed", children: lang === "zh" ? "已安装" : "Installed" }), confirmDelete === entry.filename ? (_jsxs("div", { className: "model-confirm-delete", children: [_jsx("button", { className: "danger", onClick: () => handleDelete(entry.filename), children: lang === "zh" ? "确认" : "OK" }), _jsx("button", { onClick: () => setConfirmDelete(null), children: lang === "zh" ? "取消" : "Cancel" })] })) : (_jsx("button", { className: "model-delete-btn", onClick: () => setConfirmDelete(entry.filename), children: lang === "zh" ? "删除" : "Delete" }))] }))] })] }, entry.id));
                            }), filtered.length === 0 && _jsx("p", { className: "msst-empty", children: lang === "zh" ? "此分类暂无模型" : "No models in this category" })] }), _jsxs("div", { className: "msst-installed-section", children: [_jsxs("div", { className: "msst-installed-header", children: [_jsxs("span", { children: [lang === "zh" ? "已安装文件" : "Installed Files", " ", _jsx("span", { className: "mono", children: modelsDir })] }), _jsx("button", { className: "msst-import-btn", onClick: handleMsstImport, children: lang === "zh" ? "导入" : "Import" })] }), installed.length === 0 ? (_jsx("p", { className: "msst-empty", children: lang === "zh" ? "暂无模型" : "No models installed" })) : (_jsx("div", { className: "msst-installed-list", children: installed.map((m) => {
                                    const isConverting = downloading[m.filename]?.stage === "converting";
                                    // S66: conversions are single-flight app-wide (Rust convert slot is the
                                    // authority) — gray every other convert button while one runs.
                                    const anyConverting = Object.values(downloading).some((d) => d.stage === "converting");
                                    // Catalog arch wins: hash-named official weights (demucs .th) defeat Rust's
                                    // filename detection, which reports "unknown" for them.
                                    const arch = (MSST_CATALOG.find((e) => e.filename === m.filename)?.architecture
                                        ?? m.architecture);
                                    const archHint = arch === "unknown" ? undefined : arch;
                                    const fp16Capable = MSST_FP16_ARCHS.has(arch);
                                    return (_jsxs("div", { className: "msst-installed-item", children: [_jsx("span", { className: "msst-installed-name", title: m.filename, children: m.filename }), _jsxs("span", { className: "msst-installed-meta", children: [m.has_onnx && _jsx("span", { className: "msst-onnx-ok", children: "fp32" }), m.has_fp16 && _jsx("span", { className: "msst-onnx-ok", children: "fp16" }), isConverting ? (_jsx("span", { className: "msst-converting", children: "..." })) : !m.has_onnx && !m.has_fp16 ? (_jsx("button", { className: "msst-convert-btn", disabled: anyConverting, onClick: () => convertPrecision(m.filename, undefined, archHint), children: "Convert" })) : fp16Capable && !m.has_fp16 ? (_jsx("button", { className: "msst-convert-btn", disabled: anyConverting, title: t18(MSST_FP16_TIP, lang), onClick: () => convertPrecision(m.filename, "fp16", archHint), children: t18({ zh: "补转 fp16", en: "Convert to fp16", ja: "fp16に変換" }, lang) })) : fp16Capable && !m.has_onnx ? (_jsxs(_Fragment, { children: [m.has_fp16 && !m.fp16_recipe_ok && (_jsx("button", { className: "msst-convert-btn", disabled: anyConverting, title: t18({ zh: "用当前转换配方从 ckpt 重新生成 fp16（较慢；旧版转换的 fp16 在部分显卡上有数值问题）", en: "Regenerate fp16 from the ckpt with the current recipe (slower; older fp16 conversions can misbehave numerically on some GPUs)", ja: "現在のレシピで ckpt から fp16 を再生成（時間がかかります。旧版の fp16 は一部の GPU で数値問題があります）" }, lang), onClick: () => convertPrecision(m.filename, "fp16", archHint), children: t18({ zh: "重转 fp16", en: "Redo fp16", ja: "fp16再変換" }, lang) })), _jsx("button", { className: "msst-convert-btn", disabled: anyConverting, title: t18({ zh: "从 ckpt 完整导出 fp32（较慢）", en: "Full fp32 export from the ckpt (slower)", ja: "ckpt から fp32 を完全エクスポート（時間がかかります）" }, lang), onClick: () => convertPrecision(m.filename, "fp32", archHint), children: t18({ zh: "补转 fp32", en: "Convert to fp32", ja: "fp32に変換" }, lang) })] })) : fp16Capable && !m.fp16_recipe_ok ? (
                                                    // S68c: both variants installed but the fp16 lacks the current-recipe stamp
                                                    // (`<stem>.fp16.recipe`) → offer a REFRESH (cheap, from the fp32 on disk).
                                                    // Older builds converted roformer fp16 without the fp32 norm-stats
                                                    // protection — those files can NaN on true-fp16 GPU kernels. A successful
                                                    // reconvert stamps the recipe and this button disappears (§user).
                                                    _jsx("button", { className: "msst-convert-btn", disabled: anyConverting, title: t18({ zh: "用当前转换配方重新生成 fp16（旧版转换的 fp16 在部分显卡上有数值问题）", en: "Regenerate fp16 with the current recipe (older fp16 conversions can misbehave numerically on some GPUs)", ja: "現在のレシピで fp16 を再生成（旧版で変換した fp16 は一部の GPU で数値問題があります）" }, lang), onClick: () => convertPrecision(m.filename, "fp16", archHint), children: t18({ zh: "重转 fp16", en: "Redo fp16", ja: "fp16再変換" }, lang) })) : null, " ", formatSize(m.size)] })] }, m.filename));
                                }) }))] })] }))] }));
}
function ImportDialog({ lang, voiceType, onClose, onDone }) {
    const [modelPath, setModelPath] = useState("");
    const [indexPath, setIndexPath] = useState("");
    const [diffusionPath, setDiffusionPath] = useState("");
    const [diffusionConfigPath, setDiffusionConfigPath] = useState("");
    const [avatarPath, setAvatarPath] = useState("");
    const [vocoderConfigPath, setVocoderConfigPath] = useState("");
    const [modelName, setModelName] = useState("");
    const [importing, setImporting] = useState(false);
    const [err, setErr] = useState("");
    const isVocoder = voiceType === "vocoder";
    const browse = useCallback(async (title, exts) => {
        // "*" filter: community vocoder checkpoints are often extensionless
        // (so-vits pretrain names the file just "model")
        const filters = exts.includes("*")
            ? [{ name: "File", extensions: exts.filter((e) => e !== "*") }, { name: "All", extensions: ["*"] }]
            : [{ name: "File", extensions: exts }];
        const path = await open({ title, filters });
        return path ? path : "";
    }, []);
    const handleBrowseModel = useCallback(async () => {
        const p = isVocoder
            ? await browse(lang === "zh" ? "选择声码器权重 (.ckpt / .pt / .onnx)" : "Select vocoder checkpoint (.ckpt / .pt / .onnx)", ["ckpt", "pt", "onnx", "*"])
            : await browse(lang === "zh" ? "选择模型文件 (.pth)" : "Select model file (.pth)", ["pth", "onnx"]);
        if (p) {
            setModelPath(p);
            const filename = p.split(/[/\\]/).pop() ?? "";
            setModelName(filename.replace(/\.(pth|onnx|ckpt|pt)$/i, ""));
        }
    }, [browse, lang, isVocoder]);
    const handleBrowseVocoderConfig = useCallback(async () => {
        const p = await browse(lang === "zh" ? "选择声码器配置 (config.json)" : "Select vocoder config (config.json)", ["json"]);
        if (p)
            setVocoderConfigPath(p);
    }, [browse, lang]);
    const handleBrowseIndex = useCallback(async () => {
        // RVC: FAISS .index / pre-extracted .npy. SoVITS: cluster kmeans .pt / feature-retrieval
        // .pkl / pre-converted .npy — the backend routes by model type + file extension.
        const isRvcPick = voiceType === "rvc";
        const title = isRvcPick
            ? (lang === "zh" ? "选择索引文件 (.index)" : "Select index file (.index)")
            : (lang === "zh" ? "选择聚类/检索模型 (.pt / .pkl)" : "Select cluster/retrieval model (.pt / .pkl)");
        const exts = isRvcPick ? ["index", "npy"] : ["pt", "pkl", "pickle", "npy"];
        const p = await browse(title, exts);
        if (p)
            setIndexPath(p);
    }, [browse, lang, voiceType]);
    // SoVITS only: the separate shallow-diffusion model pair (.pt + config .yaml). The .yaml is
    // optional here — export_diffusion.py auto-resolves it next to the .pt (same stem → unique
    // .yaml in dir → config.yaml) and errors in Chinese when ambiguous.
    const handleBrowseDiffusion = useCallback(async () => {
        const p = await browse(lang === "zh" ? "选择扩散模型 (.pt)" : "Select diffusion model (.pt)", ["pt"]);
        if (p)
            setDiffusionPath(p);
    }, [browse, lang]);
    const handleBrowseDiffusionConfig = useCallback(async () => {
        const p = await browse(lang === "zh" ? "选择扩散配置 (.yaml)" : "Select diffusion config (.yaml)", ["yaml", "yml"]);
        if (p)
            setDiffusionConfigPath(p);
    }, [browse, lang]);
    const handleBrowseAvatar = useCallback(async () => {
        const p = await browse(lang === "zh" ? "选择角色头图" : "Select character avatar", ["png", "jpg", "jpeg", "bmp", "webp"]);
        if (p)
            setAvatarPath(p);
    }, [browse, lang]);
    const handleImport = useCallback(async () => {
        if (!modelPath || !modelName)
            return;
        // S60 audit: a running range test stamps THIS name's sidecar at its TAIL (after the render
        // guard released) — a REPLACE import racing that window would get the OLD model's record
        // stamped onto the NEW files. Block while the test runs.
        if (useVoiceModelStore.getState().rangeTesting[modelName] !== undefined) {
            setErr(t18({ zh: "该模型正在音域测试中，请稍后再导入", en: "This model's range test is running — import later", ja: "このモデルは音域テスト中です。後で取り込んでください" }, lang));
            return;
        }
        setImporting(true);
        setErr("");
        try {
            const outcome = await invoke("import_model", {
                name: modelName,
                path: modelPath,
                modelType: voiceType,
                indexPath: indexPath || null,
                diffusionPath: diffusionPath || null,
                diffusionConfigPath: diffusionConfigPath || null,
                avatarPath: avatarPath || null,
                vocoderConfigPath: vocoderConfigPath || null,
            });
            for (const w of outcome?.warnings ?? []) {
                // Import warnings arrive as "WARN_X: detail" CODE strings — localize known ones.
                useAppStore.getState().showToast(backendErrorMessage(w) ?? w, "info");
            }
            // S60-2: fresh import → background range test (default speaker; the record died with any
            // REPLACEd sidecar). Fire-and-forget — failures/busy toast from rangeTest itself.
            if ((voiceType === "rvc" || voiceType === "sovits") && outcome?.entry) {
                void runRangeTest(outcome.entry.name, voiceType, outcome.entry.path);
            }
            onDone();
        }
        catch (e) {
            const msg = String(e);
            setErr(backendErrorMessage(msg) ?? msg);
        }
        setImporting(false);
    }, [modelPath, modelName, voiceType, indexPath, diffusionPath, diffusionConfigPath, avatarPath, vocoderConfigPath, onDone, lang]);
    const isRvc = voiceType === "rvc";
    const Z = (key) => {
        const map = {
            title: isVocoder
                ? { zh: "导入声码器", en: "Import Vocoder", ja: "ボコーダー取り込み" }
                : { zh: `导入 ${voiceType.toUpperCase()} 模型`, en: `Import ${voiceType.toUpperCase()} Model`, ja: `${voiceType.toUpperCase()} モデル取り込み` },
            model: isVocoder
                ? { zh: "声码器权重 (.ckpt / .pt / .onnx，社区包内常为无后缀的 model 文件)", en: "Vocoder checkpoint (.ckpt / .pt / .onnx; community zips often name it just \"model\")", ja: "ボコーダー重み (.ckpt / .pt / .onnx。コミュニティ配布では拡張子なしの model の場合あり)" }
                : { zh: "模型文件 (.pth)", en: "Model file (.pth)", ja: "モデルファイル (.pth)" },
            vocoderCfg: { zh: "声码器配置 (config.json)  — 可留空自动查找", en: "Vocoder config (config.json) — blank = auto-detect", ja: "ボコーダー設定 (config.json) — 空欄で自動検出" },
            vocoderNote: {
                zh: "支持经典 NSF-HiFiGAN（如 openvpi 2022.12/2024.02 社区声码器及其微调产物）；PC-NSF（mini_nsf）暂不支持。导入后在 SoVITS 推理节点的「声码器」下拉中选用。",
                en: "Classic NSF-HiFiGAN only (openvpi 2022.12/2024.02 community vocoders and their fine-tunes); PC-NSF (mini_nsf) is not supported yet. After import, pick it in the SoVITS node's Vocoder dropdown.",
                ja: "クラシック NSF-HiFiGAN のみ対応（openvpi 2022.12/2024.02 コミュニティボコーダーとその微調整版）。PC-NSF（mini_nsf）は未対応。取り込み後、SoVITS ノードの「ボコーダー」で選択できます。",
            },
            index: { zh: "索引文件 (.index)  — 可选", en: "Index file (.index) — optional", ja: "インデックス (.index) — 任意" },
            cluster: { zh: "聚类/检索模型 (.pt / .pkl)  — 可选", en: "Cluster/retrieval model (.pt / .pkl) — optional", ja: "クラスタ/検索モデル (.pt / .pkl) — 任意" },
            diffusion: { zh: "扩散模型 (.pt)  — 可选，启用浅扩散", en: "Diffusion model (.pt) — optional, enables shallow diffusion", ja: "拡散モデル (.pt) — 任意、浅い拡散を有効化" },
            diffusionCfg: { zh: "扩散配置 (.yaml)  — 可留空自动查找", en: "Diffusion config (.yaml) — blank = auto-detect", ja: "拡散設定 (.yaml) — 空欄で自動検出" },
            avatar: { zh: "角色头图 — 可选", en: "Character avatar — optional", ja: "キャラクター画像 — 任意" },
            name: { zh: "模型名称", en: "Model name", ja: "モデル名" },
            import: { zh: "导入", en: "Import", ja: "取り込み" },
            cancel: { zh: "取消", en: "Cancel", ja: "キャンセル" },
            importing: { zh: "导入并转换中...", en: "Importing & converting...", ja: "取り込み・変換中..." },
            browseBtn: { zh: "浏览", en: "Browse", ja: "参照" },
            required: { zh: "必填", en: "Required", ja: "必須" },
        };
        return map[key]?.[lang] ?? map[key]?.en ?? key;
    };
    return (_jsx("div", { className: "rm-import-overlay", onClick: onClose, children: _jsxs("div", { className: "rm-import-dialog", onClick: (e) => e.stopPropagation(), children: [_jsx("div", { className: "rm-import-title", children: Z("title") }), err && _jsx("div", { className: "rm-import-error", children: err }), _jsxs("div", { className: "rm-import-field", children: [_jsxs("label", { children: [Z("model"), " ", _jsx("span", { className: "rm-required", children: Z("required") })] }), _jsxs("div", { className: "rm-import-row", children: [_jsx("input", { type: "text", readOnly: true, value: modelPath, placeholder: "...", className: "rm-import-path" }), _jsx("button", { onClick: handleBrowseModel, children: Z("browseBtn") })] })] }), isVocoder && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "rm-import-field", children: [_jsx("label", { children: Z("vocoderCfg") }), _jsxs("div", { className: "rm-import-row", children: [_jsx("input", { type: "text", readOnly: true, value: vocoderConfigPath, placeholder: "...", className: "rm-import-path" }), _jsx("button", { onClick: handleBrowseVocoderConfig, children: Z("browseBtn") })] })] }), _jsx("p", { className: "rm-voice-hint", children: Z("vocoderNote") })] })), !isVocoder && (_jsxs("div", { className: "rm-import-field", children: [_jsx("label", { children: isRvc ? Z("index") : Z("cluster") }), _jsxs("div", { className: "rm-import-row", children: [_jsx("input", { type: "text", readOnly: true, value: indexPath, placeholder: "...", className: "rm-import-path" }), _jsx("button", { onClick: handleBrowseIndex, children: Z("browseBtn") })] })] })), !isRvc && !isVocoder && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "rm-import-field", children: [_jsx("label", { children: Z("diffusion") }), _jsxs("div", { className: "rm-import-row", children: [_jsx("input", { type: "text", readOnly: true, value: diffusionPath, placeholder: "...", className: "rm-import-path" }), _jsx("button", { onClick: handleBrowseDiffusion, children: Z("browseBtn") })] })] }), diffusionPath && (_jsxs("div", { className: "rm-import-field", children: [_jsx("label", { children: Z("diffusionCfg") }), _jsxs("div", { className: "rm-import-row", children: [_jsx("input", { type: "text", readOnly: true, value: diffusionConfigPath, placeholder: "...", className: "rm-import-path" }), _jsx("button", { onClick: handleBrowseDiffusionConfig, children: Z("browseBtn") })] })] }))] })), !isVocoder && (_jsxs("div", { className: "rm-import-field", children: [_jsx("label", { children: Z("avatar") }), _jsxs("div", { className: "rm-import-row", children: [_jsx("input", { type: "text", readOnly: true, value: avatarPath, placeholder: "...", className: "rm-import-path" }), _jsx("button", { onClick: handleBrowseAvatar, children: Z("browseBtn") })] })] })), _jsxs("div", { className: "rm-import-field", children: [_jsx("label", { children: Z("name") }), _jsx("input", { type: "text", value: modelName, onChange: (e) => setModelName(e.target.value), className: "rm-import-name" })] }), _jsxs("div", { className: "rm-import-actions", children: [_jsx("button", { onClick: onClose, disabled: importing, children: Z("cancel") }), _jsx("button", { className: "primary", onClick: handleImport, disabled: importing || !modelPath || !modelName, children: importing ? Z("importing") : Z("import") })] })] }) }));
}
// ─── Sub-components ─────────────────────────────────────────
function DownloadBar({ dl, lang }) {
    return (_jsx("div", { className: "model-download-progress", children: dl.stage === "converting" ? (_jsxs(_Fragment, { children: [_jsx("div", { className: "model-download-bar model-convert-bar", style: { width: "100%" } }), _jsx("span", { className: "model-download-text", children: lang === "zh" ? "转换为 ONNX..." : "Converting to ONNX..." })] })) : (_jsxs(_Fragment, { children: [_jsx("div", { className: "model-download-bar", style: { width: dl.total > 0 ? `${(dl.downloaded / dl.total) * 100}%` : "0%" } }), _jsxs("span", { className: "model-download-text", children: [formatSize(dl.downloaded), " / ", dl.total > 0 ? formatSize(dl.total) : "..."] })] })) }));
}
function GameEngineTab({ lang }) {
    const [installed, setInstalled] = useState(null);
    const [dl, setDl] = useState(null);
    const [busy, setBusy] = useState(false);
    const [confirmDelete, setConfirmDelete] = useState(false);
    const showToast = useAppStore((s) => s.showToast);
    const unlistenRef = useRef(null);
    const refresh = useCallback(async () => {
        try {
            const st = await invoke("midi_extract_status");
            setInstalled(st.installed);
            // a download started before an unmount is still running (Rust single-flight) —
            // restore the busy view instead of offering a second download (audit S60)
            if (st.downloading)
                setBusy(true);
        }
        catch {
            setInstalled(false);
        }
    }, []);
    useEffect(() => {
        void refresh();
        let disposed = false;
        void listen("game-download-progress", (e) => {
            setDl(e.payload);
            // remounted mid-download: no pending invoke here, so the terminal event drives the state
            if (e.payload.stage === "done") {
                setBusy(false);
                setDl(null);
                void refresh();
            }
            else {
                setBusy(true);
            }
        }).then((un) => {
            if (disposed)
                un();
            else
                unlistenRef.current = un;
        });
        return () => {
            disposed = true;
            unlistenRef.current?.();
            unlistenRef.current = null;
        };
    }, [refresh]);
    const handleDownload = useCallback(async () => {
        if (busy)
            return;
        setBusy(true);
        setDl({ stage: "download", downloaded: 0, total: 0 });
        try {
            // ghRoutes → Rust gh_routes: the full ordered GH failover chain (chosen proxy →
            // direct → other presets, S66); the backend interleaves it with its static rotation.
            const { ghMirror, ghPresets } = useMsstModelStore.getState();
            const st = await invoke("download_game_package", {
                ghRoutes: ghRouteOrder(ghMirror, ghPresets),
            });
            setInstalled(st.installed);
            showToast(t18({ zh: "GAME 引擎已安装", en: "GAME engine installed", ja: "GAME エンジンをインストールしました" }, lang), "success");
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            if (msg.includes("GAME_DL_BUSY"))
                return; // another flight is running — its events drive the UI
            if (isCancelError(msg))
                return; // user cancelled the download — silent settle
            const base = msg.includes("GAME_DL_EXTRACT")
                ? t18({ zh: "解压安装失败", en: "Extraction failed", ja: "展開に失敗しました" }, lang)
                : t18({ zh: "下载失败", en: "Download failed", ja: "ダウンロードに失敗しました" }, lang);
            showToast(`${base}: ${backendErrorMessage(msg) ?? msg}`, "error");
        }
        finally {
            setBusy(false);
            setDl(null);
        }
    }, [busy, lang, showToast]);
    const handleDelete = useCallback(async () => {
        setConfirmDelete(false);
        try {
            const st = await invoke("delete_game_package");
            setInstalled(st.installed);
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            const base = t18({ zh: "删除失败", en: "Delete failed", ja: "削除に失敗しました" }, lang);
            showToast(msg.includes("GAME_DELETE_FAILED") ? `${base}: ${msg}` : msg, "error");
        }
    }, [lang, showToast]);
    const stageText = (p) => {
        if (p.stage === "extract")
            return t18({ zh: "解压安装中...", en: "Extracting...", ja: "展開中..." }, lang);
        return `${formatSize(p.downloaded)} / ${p.total > 0 ? formatSize(p.total) : "..."}`;
    };
    return (_jsxs("div", { className: `msst-model-card-wrap ${installed ? "installed" : ""}`, children: [_jsx("div", { className: "msst-model-card-status", children: _jsx(StatusIcon, { type: installed ? "installed" : busy ? "downloading" : "pending", title: installed ? (lang === "zh" ? "已安装" : "Installed") : busy ? (lang === "zh" ? "正在安装" : "Installing") : (lang === "zh" ? "下载" : "Download"), onClick: handleDownload }) }), _jsxs("div", { className: "msst-model-card", children: [_jsxs("div", { className: "model-card-header", children: [_jsxs("span", { className: "model-card-name", children: ["GAME \u00B7 ", t18({ zh: "人声转 MIDI", en: "Vocal-to-MIDI", ja: "歌声→MIDI" }, lang), _jsx("span", { className: "essential-badge", children: lang === "zh" ? "必备" : "Required" })] }), _jsx("span", { className: "model-card-arch", children: "openvpi \u00B7 1.0.3" })] }), _jsx("p", { className: "model-card-desc", children: t18({
                            zh: "从人声干声提取音符。模型权重由 openvpi 发布，需在此下载。",
                            en: "Extracts notes from vocal stems. Weights are released by openvpi and must be downloaded here.",
                            ja: "ボーカルステムからノートを抽出します。モデル重みは openvpi が公開しており、ここでダウンロードが必要です。",
                        }, lang) }), _jsxs("div", { className: "model-card-meta", children: [_jsx("span", { className: "model-card-stems", children: "en / ja / yue / zh" }), _jsx("span", { className: "model-card-size", children: formatSize(179775226) })] }), busy && dl && (_jsxs("div", { className: "model-download-progress", children: [_jsx("div", { className: `model-download-bar ${dl.stage !== "download" ? "model-convert-bar" : ""}`, style: { width: dl.stage !== "download" ? "100%" : dl.total > 0 ? `${(dl.downloaded / dl.total) * 100}%` : "0%" } }), _jsx("span", { className: "model-download-text", children: stageText(dl) })] })), installed && (_jsxs("div", { className: "model-card-actions", children: [_jsx("span", { className: "model-status-installed", children: lang === "zh" ? "已安装" : lang === "ja" ? "インストール済み" : "Installed" }), confirmDelete ? (_jsxs("div", { className: "model-confirm-delete", children: [_jsx("button", { className: "danger", onClick: handleDelete, children: lang === "zh" ? "确认" : "OK" }), _jsx("button", { onClick: () => setConfirmDelete(false), children: lang === "zh" ? "取消" : lang === "ja" ? "キャンセル" : "Cancel" })] })) : (_jsx("button", { className: "model-delete-btn", onClick: () => setConfirmDelete(true), children: lang === "zh" ? "删除" : lang === "ja" ? "削除" : "Delete" }))] }))] })] }));
}
// ─── AMT (music-to-midi) runtime card — multi-instrument audio→MIDI sidecar.
// The Python engine + PyTorch + model weights are GB-sized and partly
// CC BY-NC, so they are not bundled; the card reports whether the sidecar
// was found at the expected data/amt/python path. ───
function VoiceAvatar({ path, name, onSet }) {
    if (path) {
        return (_jsx("div", { className: "rm-voice-avatar", onClick: onSet, title: name, children: _jsx("img", { src: convertFileSrc(path), alt: name }) }));
    }
    return (_jsx("div", { className: "rm-voice-avatar rm-voice-avatar-empty", onClick: onSet, title: "Set avatar", children: _jsx("span", { children: name.charAt(0).toUpperCase() }) }));
}
function VoiceModelsTab({ lang }) {
    // S82d: ONE active speaker per model row, shared by the audition button AND the range row.
    // The two rows used to carry their own selects (each with its own state, able to point at
    // DIFFERENT singers = semantic drift + the crowding the user reported); the single selector
    // lives in the model meta line, replacing the static "N speakers" badge.
    const [voiceSpk, setVoiceSpk] = useState({});
    const [voiceType, setVoiceType] = useState("rvc");
    const [showImport, setShowImport] = useState(false);
    const [deleteConfirm, setDeleteConfirm] = useState(null);
    // S146f: 音域边界编辑器展开时,四条滑条会占满整行 —— 试听按钮与它挤在同一行里视觉重叠
    // (用户实机报的)。编辑态提到这一层,让同排的动作按钮能让位。
    const [rangeEditing, setRangeEditing] = useState(null);
    // S167c: which model's EXPORT format chooser is open — the chooser owns its row (audition /
    // range row / delete yield), same tab-level pattern as rangeEditing (S146f: 同排按钮要跟着让位).
    const [exportPick, setExportPick] = useState(null);
    // Shared store — the SAME list the RVC/SoVITS workflow nodes read (one source of truth).
    const models = useVoiceModelStore((s) => s.models[voiceType === "song" ? "rvc" : voiceType]);
    const voiceError = useVoiceModelStore((s) => s.error);
    const { fetchModels, deleteModel, setAvatar, clearError } = useVoiceModelStore();
    // Song models state
    const [songModels, setSongModels] = useState([]);
    const [songDownloading, setSongDownloading] = useState(null);
    const [songDeleteConfirm, setSongDeleteConfirm] = useState(null);
    // built-in default vocoder facts — refetched on tab entry (cheap disk stat)
    const [defaultVoc, setDefaultVoc] = useState(null);
    useEffect(() => {
        if (voiceType === "song") {
            void listSongModels().then(setSongModels);
        }
    }, [voiceType]);
    useEffect(() => {
        if (voiceType !== "vocoder")
            return;
        void invoke("get_default_vocoder_info")
            .then(setDefaultVoc)
            .catch(() => setDefaultVoc(null));
    }, [voiceType]);
    useEffect(() => { void fetchModels(); }, [fetchModels]);
    // S60-4: tab unmount = the audition UI is gone — stop OUR playback (ownership proven
    // against preview.path; a foreign consumer's playback is untouched) and clear the state.
    useEffect(() => () => {
        const a = useVoiceModelStore.getState().auditionState;
        if (a) {
            if (a.phase === "playing" && preview.path === a.path) {
                preview.onEnd = null;
                preview.stop();
            }
            useVoiceModelStore.getState().setAuditionState(null);
        }
    }, []);
    const handleDelete = useCallback(async (name) => {
        // S60 audit: a running range test writes this model's sidecar at its tail (and an
        // audition writes a wav beside it — Rust also guards that one); block the delete.
        if (voiceType === "song")
            return;
        const vm = useVoiceModelStore.getState();
        if (vm.rangeTesting[name] !== undefined || vm.auditionState?.name === name) {
            useAppStore.getState().showToast(t18({ zh: "该模型正在测试/试听中，稍后再删除", en: "This model is being tested/auditioned — delete later", ja: "このモデルはテスト/試聴中です。後で削除してください" }, lang), "info");
            setDeleteConfirm(null);
            return;
        }
        // type-scoped: same-name entries across types are standard (rvc+sovits pair
        // + a vocoder named after the singer) — an untyped delete hits the first
        // scan match, i.e. potentially the WRONG model's files (S40 红队 A5)
        await deleteModel(name, voiceType); // errors land in voiceError
        setDeleteConfirm(null);
    }, [deleteModel, voiceType, lang]);
    // S78 batch 7: import a `.zip` model package (the Export counterpart). Rust figures the registry
    // type from the manifest, so afterwards we jump to that tab so the imported model is visible.
    const handleImportPackage = useCallback(async () => {
        const file = await open({
            title: t18({ zh: "导入模型包 (.zip)", en: "Import model package (.zip)", ja: "モデルパッケージを取り込み (.zip)" }, lang),
            filters: [{ name: "Muno Model / Zip", extensions: ["zip"] }],
        });
        if (!file || typeof file !== "string")
            return;
        try {
            const outcome = await invoke("import_model_package", { packagePath: file });
            await fetchModels();
            const vt = MODEL_TYPE_TO_VOICE[outcome.entry.model_type];
            if (vt)
                setVoiceType(vt);
            useAppStore.getState().showToast(t18({ zh: `已导入 · ${outcome.entry.name}`, en: `Imported · ${outcome.entry.name}`, ja: `取り込み完了 · ${outcome.entry.name}` }, lang), "success");
            // Non-fatal import warnings ride the same mapper the ImportDialog uses.
            (outcome.warnings ?? []).forEach((w) => useAppStore.getState().showToast(backendErrorMessage(w) ?? w, "info"));
        }
        catch (e) {
            // Busy/interlock rejections (CONVERT_BUSY / MODEL_BUSY_AUDITION) are transient → info, not error
            // (the app-wide isBusyError funnel discipline; matches VoiceAuditionButton).
            useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
        }
    }, [lang, fetchModels]);
    return (_jsxs("div", { className: "rm-voice-tab", children: [voiceError && _jsx("div", { className: "msst-error", onClick: clearError, children: backendErrorMessage(voiceError) ?? voiceError }), _jsxs("div", { className: "msst-filter", children: [_jsx("button", { className: voiceType === "rvc" ? "active" : "", onClick: () => setVoiceType("rvc"), children: "RVC" }), _jsx("button", { className: voiceType === "sovits" ? "active" : "", onClick: () => setVoiceType("sovits"), children: "SoVITS" }), _jsx("button", { className: voiceType === "vocoder" ? "active" : "", onClick: () => setVoiceType("vocoder"), children: t18({ zh: "声码器", en: "Vocoder", ja: "ボコーダー" }, lang) }), _jsx("button", { className: voiceType === "song" ? "active" : "", onClick: () => setVoiceType("song"), children: t18({ zh: "生成歌曲", en: "Song Generation", ja: "楽曲生成" }, lang) }), _jsx("div", { className: "rm-filter-spacer" }), voiceType !== "song" && (_jsxs(_Fragment, { children: [_jsx("button", { className: "rm-import-top-btn", onClick: handleImportPackage, title: t18({
                                    zh: "从 .zip 模型包导入（本软件「导出」生成的包，含索引/聚类/扩散/头像）",
                                    en: "Import from a .zip model package (produced by Export — includes index / cluster / diffusion / avatar)",
                                    ja: "「書き出し」で作成した .zip モデルパッケージから取り込み（インデックス/クラスタ/拡散/アバターを含む）",
                                }, lang), children: t18({ zh: "导入模型包", en: "Import Package", ja: "パッケージ取り込み" }, lang) }), _jsxs("button", { className: "primary rm-import-top-btn", onClick: () => setShowImport(true), children: ["+ ", lang === "zh" ? "导入模型" : lang === "ja" ? "モデル取り込み" : "Import Model"] })] }))] }), voiceType !== "vocoder" && voiceType !== "song" && _jsx(RangeBatchRow, { lang: lang }), _jsxs("div", { className: "rm-voice-list", children: [models.length === 0 && voiceType !== "vocoder" && voiceType !== "song" && (_jsx("p", { className: "msst-empty", children: lang === "zh"
                            ? `暂无 ${voiceType.toUpperCase()} 模型`
                            : `No ${voiceType.toUpperCase()} models` })), voiceType === "song" && (_jsxs(_Fragment, { children: [_jsx("p", { className: "rm-voice-hint", children: t18({
                                    zh: "歌曲生成模型供「歌曲制作」功能使用。支持根据文本提示词和歌词生成完整歌曲，包括人声、伴奏、MIDI、歌词时间轴等。模型基于 Hugging Face 镜像下载，支持断点续传。",
                                    en: "Song generation models power the Song Studio feature. Generate complete songs from text prompts and lyrics, including vocals, backing tracks, MIDI, and timestamped lyrics. Models are downloaded from Hugging Face mirror with resume support.",
                                    ja: "楽曲生成モデルは「楽曲制作」機能で使用されます。テキストプロンプトと歌詞から完全な楽曲を生成し、ボーカル、伴奏、MIDI、タイムスタンプ付き歌詞を含みます。Hugging Face ミラーからダウンロード、レジューム対応。",
                                }, lang) }), Object.keys(SONG_FAMILY_LABELS).map((family) => (_jsxs("div", { children: [_jsx("div", { className: "rm-voice-hint", style: { marginTop: "12px", fontWeight: 600 }, children: t18(SONG_FAMILY_LABELS[family], lang) }), SONG_MODEL_CATALOG.filter((m) => m.family === family).map((model) => {
                                        const status = getSongInstallStatus(songModels, model);
                                        const installed = status.installed;
                                        const partial = !installed && status.ready > 0;
                                        const isDownloading = songDownloading?.id === model.id;
                                        const totalBytes = songModelTotalSize(model);
                                        return (_jsxs("div", { className: "rm-voice-item", children: [_jsxs("div", { className: "rm-voice-item-info", children: [_jsx("span", { className: "rm-voice-item-name", children: t18(model.label, lang) }), _jsxs("span", { className: "rm-voice-item-meta", children: [_jsx("span", { className: "msst-onnx-ok", title: t18({
                                                                        zh: `许可证：${model.license}`,
                                                                        en: `License: ${model.license}`,
                                                                        ja: `ライセンス：${model.license}`,
                                                                    }, lang), children: model.license }), _jsxs("span", { children: [(totalBytes / (1024 * 1024 * 1024)).toFixed(2), " GB"] }), _jsxs("span", { title: t18({ zh: "显存需求", en: "VRAM", ja: "VRAM要件" }, lang), children: [t18({ zh: "显存", en: "VRAM", ja: "VRAM" }, lang), " \u2248", model.vramGb, "GB"] }), _jsx("span", { children: model.languages }), model.supportsMultiTrack && (_jsx("span", { className: "msst-onnx-ok", children: t18({ zh: "分轨", en: "Stems", ja: "分離トラック" }, lang) })), model.supportsMidi && (_jsx("span", { className: "msst-onnx-ok", children: "MIDI" })), installed && (_jsx("span", { className: "msst-onnx-ok", children: t18({ zh: "已安装", en: "Installed", ja: "インストール済み" }, lang) })), partial && (_jsxs("span", { title: t18({ zh: `缺少：${status.missing.join("、")}`, en: `Missing: ${status.missing.join(", ")}`, ja: `未取得：${status.missing.join("、")}` }, lang), children: [t18({ zh: "未完整", en: "Incomplete", ja: "未完了" }, lang), " ", status.ready, "/", status.required] }))] }), _jsx("span", { className: "rm-voice-item-meta", style: { marginTop: "4px", fontSize: "12px", color: "var(--text-secondary)" }, children: t18(model.description, lang) }), _jsxs("span", { className: "rm-voice-item-meta", style: { marginTop: "2px", fontSize: "11px" }, children: [_jsx("span", { style: { color: "var(--text-tertiary)" }, children: t18({ zh: "来源：", en: "Source: ", ja: "ソース：" }, lang) }), model.sourceRepo] }), isDownloading && songDownloading && (_jsxs("div", { className: "msst-progress-bar", style: { marginTop: "8px" }, children: [_jsx("div", { className: "msst-progress-fill", style: {
                                                                        width: songDownloading.total
                                                                            ? `${Math.min(100, (songDownloading.downloaded / songDownloading.total) * 100)}%`
                                                                            : "0%",
                                                                    } }), _jsxs("span", { className: "msst-progress-text", children: [songDownloading.total
                                                                            ? `${((songDownloading.downloaded / songDownloading.total) * 100).toFixed(1)}%`
                                                                            : t18({ zh: "准备中...", en: "Preparing...", ja: "準備中..." }, lang), songDownloading.file ? ` · ${songDownloading.file}` : ""] })] }))] }), _jsxs("div", { className: "rm-voice-item-actions", children: [!installed && !isDownloading && (_jsx("button", { className: "primary", onClick: async () => {
                                                                setSongDownloading({ id: model.id, downloaded: 0, total: totalBytes, file: "" });
                                                                let completedBytes = 0;
                                                                let currentFile = "";
                                                                let unlisten = null;
                                                                try {
                                                                    // 已就绪的文件（路径+大小都匹配）直接跳过 → 断点续下剩余文件
                                                                    const existing = await listSongModels();
                                                                    const done = new Set(existing.filter((f) => f.filename.startsWith(`${model.id}/`)).map((f) => `${f.filename}:${f.size}`));
                                                                    unlisten = await listen("song-download-progress", (event) => {
                                                                        if (event.payload.id !== model.id)
                                                                            return;
                                                                        setSongDownloading({
                                                                            id: model.id,
                                                                            downloaded: completedBytes + event.payload.downloaded,
                                                                            total: totalBytes,
                                                                            file: currentFile,
                                                                        });
                                                                    });
                                                                    for (const f of model.files) {
                                                                        // 主路径或任一等价替代路径（如 xl-turbo/turbo 变体）已就绪则跳过。
                                                                        // 替代路径只按文件名匹配（变体权重体积不同，不能用 size 匹配）
                                                                        const donePaths = new Set(existing.filter((x) => x.filename.startsWith(`${model.id}/`)).map((x) => x.filename));
                                                                        const hasAlt = (f.alts ?? []).some((alt) => donePaths.has(`${model.id}/${alt}`));
                                                                        if (done.has(`${model.id}/${f.path}:${f.size}`) || hasAlt) {
                                                                            completedBytes += f.size;
                                                                            continue;
                                                                        }
                                                                        currentFile = f.path;
                                                                        setSongDownloading({ id: model.id, downloaded: completedBytes, total: totalBytes, file: f.path });
                                                                        await downloadSongModel(f.urls, model.id, f.path, f.sha256);
                                                                        completedBytes += f.size;
                                                                    }
                                                                    const updated = await listSongModels();
                                                                    setSongModels(updated);
                                                                    useAppStore.getState().showToast(t18({ zh: `已下载：${t18(model.label, lang)}`, en: `Downloaded: ${t18(model.label, lang)}`, ja: `ダウンロード完了：${t18(model.label, lang)}` }, lang), "success");
                                                                }
                                                                catch (e) {
                                                                    useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), "error");
                                                                }
                                                                finally {
                                                                    unlisten?.();
                                                                    setSongDownloading(null);
                                                                }
                                                            }, children: partial
                                                                ? t18({ zh: "继续下载", en: "Resume", ja: "続行" }, lang)
                                                                : t18({ zh: "下载", en: "Download", ja: "ダウンロード" }, lang) })), (installed || partial) && !isDownloading && songDeleteConfirm !== model.id && (_jsx("button", { onClick: () => setSongDeleteConfirm(model.id), children: t18({ zh: "删除", en: "Delete", ja: "削除" }, lang) })), songDeleteConfirm === model.id && (_jsxs(_Fragment, { children: [_jsx("button", { className: "danger", onClick: async () => {
                                                                        try {
                                                                            // 传模型 id = 删除整个模型目录（含子目录文件）
                                                                            await deleteSongModel(model.id);
                                                                            setSongDeleteConfirm(null);
                                                                            const updated = await listSongModels();
                                                                            setSongModels(updated);
                                                                            useAppStore.getState().showToast(t18({ zh: `已删除：${t18(model.label, lang)}`, en: `Deleted: ${t18(model.label, lang)}`, ja: `削除完了：${t18(model.label, lang)}` }, lang), "success");
                                                                        }
                                                                        catch (e) {
                                                                            useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), "error");
                                                                        }
                                                                    }, children: t18({ zh: "确认删除", en: "Confirm", ja: "確認" }, lang) }), _jsx("button", { onClick: () => setSongDeleteConfirm(null), children: t18({ zh: "取消", en: "Cancel", ja: "キャンセル" }, lang) })] }))] })] }, model.id));
                                    })] }, family)))] })), voiceType === "vocoder" && (
                    // zero-knowledge banner: THE answer to "为什么我的声码器不能用于某个模型"
                    _jsx("p", { className: "rm-voice-hint", children: t18({
                            zh: "声码器供 SoVITS 浅扩散/增强器使用（在 SoVITS 推理节点里选择）；同一歌手微调的声码器可被其所有 SoVITS 模型共享。仅频谱格式一致（44.1kHz / hop 512 / 128 mel）的声码器可被选用；RVC 模型无外部声码器接口，不适用。",
                            en: "Vocoders serve SoVITS shallow diffusion / the enhancer (picked inside the SoVITS node); one singer's fine-tuned vocoder is shared by all their SoVITS models. Only format-matching vocoders (44.1kHz / hop 512 / 128 mel) are selectable; RVC models have no external vocoder interface.",
                            ja: "ボコーダーは SoVITS の浅い拡散/エンハンサー用（SoVITS ノード内で選択）。同じ歌手のボコーダーは全 SoVITS モデルで共有可。フォーマット一致（44.1kHz / hop 512 / 128 mel）のもののみ選択可能。RVC には外部ボコーダーの接続点がありません。",
                        }, lang) })), voiceType === "vocoder" && defaultVoc && (
                    // pinned read-only row: what the node dropdown's「默认声码器」IS —
                    // the built-in aux vocoder; its facts come from disk (get_default_
                    // vocoder_info), so a missing aux install surfaces HERE as a loud
                    // chip instead of only erroring at render time
                    _jsx("div", { className: "rm-voice-item rm-voice-item-builtin", title: t18({
                            zh: "随应用分发的 OpenVPI 社区通用声码器——未选择自定义声码器时，浅扩散/增强器使用它；也是声码器格式类的基准。不可删除。",
                            en: "The OpenVPI community general vocoder shipped with the app — shallow diffusion / the enhancer use it unless a custom vocoder is picked; also the format-class reference. Not deletable.",
                            ja: "アプリ同梱の OpenVPI コミュニティ汎用ボコーダー。カスタム未選択時に浅い拡散/エンハンサーが使用。フォーマットの基準でもあります。削除不可。",
                        }, lang), children: _jsxs("div", { className: "rm-voice-item-info", children: [_jsx("span", { className: "rm-voice-item-name", children: t18({ zh: "默认声码器", en: "Default vocoder", ja: "既定ボコーダー" }, lang) }), _jsxs("span", { className: "rm-voice-item-meta", children: [_jsx("span", { className: "ver-badge", children: "NSF-HiFiGAN" }), _jsx("span", { className: "msst-onnx-ok", children: t18({ zh: "内置", en: "Built-in", ja: "内蔵" }, lang) }), defaultVoc.present ? (_jsxs(_Fragment, { children: [_jsxs("span", { children: [formatSampleRateKhz(defaultVoc.sample_rate ?? 44100), " \u00B7 hop", " ", defaultVoc.hop_size ?? "?", " \u00B7 ", defaultVoc.num_mels ?? "?", " mel"] }), _jsx("span", { className: "msst-onnx-ok", title: t18({
                                                        zh: "标准格式：可用于所有 SoVITS 模型的浅扩散/增强器",
                                                        en: "Standard format: usable by every SoVITS model's shallow diffusion / enhancer",
                                                        ja: "標準フォーマット：全 SoVITS モデルの浅い拡散/エンハンサーで使用可能",
                                                    }, lang), children: t18({ zh: "SoVITS 扩散/增强", en: "SoVITS diff/enhance", ja: "SoVITS 拡散/強化" }, lang) })] })) : (_jsx("span", { className: "rm-voice-item-warn", title: t18({
                                                zh: `缺少文件：${defaultVoc.missing.join("、")}——请到 设置→模型资产 下载推理核心包（或手动放入 data/models/auxiliary/），否则浅扩散/增强器无法运行`,
                                                en: `Missing: ${defaultVoc.missing.join(", ")} — download the core inference pack in Settings → Model Assets (or place them in data/models/auxiliary/), or shallow diffusion / the enhancer cannot run`,
                                                ja: `欠落ファイル：${defaultVoc.missing.join("、")} — 設定→モデルアセット で推論コアパックをダウンロード（または data/models/auxiliary/ に配置）してください。ないと浅い拡散/エンハンサーは動きません`,
                                            }, lang), children: t18({ zh: "缺失", en: "Missing", ja: "欠落" }, lang) }))] })] }) })), voiceType === "vocoder" && models.length === 0 && (_jsx("p", { className: "msst-empty", children: t18({
                            zh: "尚无自定义声码器——可在训练页微调后保存，或导入社区声码器（ckpt/onnx）",
                            en: "No custom vocoders yet — fine-tune one on the training page, or import a community vocoder (ckpt/onnx)",
                            ja: "カスタムボコーダーはまだありません — トレーニングページで微調整して保存するか、コミュニティボコーダー（ckpt/onnx）を取り込めます",
                        }, lang) })), models.map((m) => {
                        const isVocoder = voiceType === "vocoder";
                        const ver = isVocoder ? null : voiceVersionBadge(m);
                        const speakerOpts = isVocoder ? [] : voiceSpeakerOptions(m);
                        // a re-import can shrink the speaker list — a stale stored id falls back to 0
                        const stored = voiceSpk[m.name] ?? 0;
                        const spk = speakerOpts.some((s) => s.id === stored) ? stored : 0;
                        const vocFormatOk = isVocoder ? vocoderFormatMatches(m) : true;
                        return (_jsxs("div", { className: "rm-voice-item", children: [!isVocoder && (_jsx(VoiceAvatar, { path: m.avatar_path, name: m.name, onSet: async () => {
                                        const file = await open({ title: lang === "zh" ? "选择角色头图" : "Select avatar", filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "bmp", "webp"] }] });
                                        if (file)
                                            await setAvatar(m.name, file);
                                    } })), _jsxs("div", { className: "rm-voice-item-info", children: [_jsx("span", { className: "rm-voice-item-name", children: m.name }), isVocoder ? (_jsxs(_Fragment, { children: [_jsxs("span", { className: "rm-voice-item-meta", children: [_jsx("span", { className: "ver-badge", title: t18({ zh: "经典 NSF-HiFiGAN 架构", en: "Classic NSF-HiFiGAN architecture", ja: "クラシック NSF-HiFiGAN アーキテクチャ" }, lang), children: "NSF-HiFiGAN" }), vocFormatOk ? (_jsx("span", { className: "msst-onnx-ok", title: t18({
                                                                zh: "标准格式：可用于所有 SoVITS 模型的浅扩散/增强器",
                                                                en: "Standard format: usable by every SoVITS model's shallow diffusion / enhancer",
                                                                ja: "標準フォーマット：全 SoVITS モデルの浅い拡散/エンハンサーで使用可能",
                                                            }, lang), children: t18({ zh: "SoVITS 扩散/增强", en: "SoVITS diff/enhance", ja: "SoVITS 拡散/強化" }, lang) })) : (_jsx("span", { className: "rm-voice-item-warn", title: t18({
                                                                zh: "梅尔频谱格式与标准格式（44.1kHz / hop 512 / 128 mel / 40-16000Hz）不一致——不会出现在推理节点的声码器列表中",
                                                                en: "Mel format differs from the standard (44.1kHz / hop 512 / 128 mel / 40-16000Hz) — will not appear in the node's vocoder list",
                                                                ja: "メルフォーマットが標準（44.1kHz / hop 512 / 128 mel / 40-16000Hz）と不一致 — ノードのボコーダー一覧に表示されません",
                                                            }, lang), children: t18({ zh: "格式不匹配", en: "Format mismatch", ja: "フォーマット不一致" }, lang) }))] }), _jsx("span", { className: "rm-voice-item-meta", children: _jsx("span", { className: "rm-meta-shrink", title: vocoderFormatLabel(m), children: vocoderFormatLabel(m) }) })] })) : (_jsxs("span", { className: "rm-voice-item-meta", children: [ver && _jsx("span", { className: "ver-badge", children: ver }), m.format === "Onnx" ? _jsx("span", { className: "msst-onnx-ok", children: "ONNX" }) : _jsx("span", { children: m.format }), m.index_path && (
                                                // SoVITS carries ONE of two mutually exclusive asset kinds
                                                // (inference prefers retrieval): `*.index_vectors.npy` =
                                                // the retrieval matrix (training default), anything else
                                                // in `.cluster/` = kmeans centers — labelling both 聚类
                                                // told users their non-kmeans runs produced kmeans
                                                _jsx("span", { className: "msst-onnx-ok", title: t18(voiceType === "rvc"
                                                        ? { zh: "已附带检索索引", en: "Retrieval index present", ja: "検索インデックスあり" }
                                                        : m.index_path.endsWith(".index_vectors.npy")
                                                            ? { zh: "已附带检索特征库", en: "Retrieval feature bank present", ja: "検索特徴バンクあり" }
                                                            : { zh: "已附带聚类中心 (kmeans)", en: "Kmeans cluster centers present", ja: "クラスタ中心 (kmeans) あり" }, lang), children: voiceType === "rvc"
                                                        ? "IDX"
                                                        : m.index_path.endsWith(".index_vectors.npy")
                                                            ? t18({ zh: "检索", en: "RETR", ja: "検索" }, lang)
                                                            : t18({ zh: "聚类", en: "KMEANS", ja: "クラスタ" }, lang) })), m.diffusion_path && (_jsx("span", { className: "msst-onnx-ok", title: t18(VOICE_STRINGS.diffBadgeTip, lang), children: "DIFF" })), _jsx("span", { children: formatSampleRateKhz(m.sample_rate) }), typeof m.config?.features_dim === "number" && (_jsxs("span", { children: [m.config.features_dim, " ", t18({ zh: "维", en: "dim", ja: "次元" }, lang)] }))] })), !isVocoder && exportPick !== m.name && (_jsx(VoiceRangeRow, { m: m, voiceType: voiceType, lang: lang, spk: spk, onSpk: (id) => setVoiceSpk((s) => ({ ...s, [m.name]: id })), editing: rangeEditing === m.name, onEditing: (on) => setRangeEditing(on ? m.name : null) }))] }), !isVocoder && rangeEditing !== m.name && exportPick !== m.name && (_jsx(VoiceAuditionButton, { m: m, voiceType: voiceType, lang: lang, spk: spk })), rangeEditing !== m.name && (_jsx(VoiceExportButton, { m: m, voiceType: voiceType === "song" ? "rvc" : voiceType, lang: lang, picking: exportPick === m.name, onPicking: (on) => { setExportPick(on ? m.name : null); if (on)
                                        setDeleteConfirm(null); } })), exportPick !== m.name && (deleteConfirm === m.name ? (_jsxs("div", { className: "model-confirm-delete", children: [_jsx("button", { className: "danger", onClick: () => handleDelete(m.name), children: lang === "zh" ? "确认" : "OK" }), _jsx("button", { onClick: () => setDeleteConfirm(null), children: lang === "zh" ? "取消" : "Cancel" })] })) : (_jsx("button", { className: "model-delete-btn", onClick: () => setDeleteConfirm(m.name), children: lang === "zh" ? "删除" : "Delete" })))] }, m.name));
                    })] }), showImport && voiceType !== "song" && (_jsx(ImportDialog, { lang: lang, voiceType: voiceType, onClose: () => setShowImport(false), onDone: () => { setShowImport(false); fetchModels(); } }))] }));
}
// Rust `ModelType` enum variant name (list_models payload) → the frontend voice-tab vocabulary.
// Used to jump to the imported model's tab after a package import.
const MODEL_TYPE_TO_VOICE = {
    Rvc: "rvc",
    SoVits: "sovits",
    NsfHifigan: "vocoder",
};
/// Windows-safe default filename for the export save dialog (keeps CJK, strips illegal chars +
/// trailing dots/spaces) — mirrors Rust sanitize_file_stem's intent so the suggested name never
/// contains a character the OS save dialog rejects. Empty → "model".
function sanitizeExportName(name) {
    // eslint-disable-next-line no-control-regex
    const cleaned = name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "").replace(/[. ]+$/, "").trim();
    return cleaned || "model";
}
// S78 batch 7: export ONE installed voice model as a portable `.zip` package (re-importable via
// "Import Package"). Per-row so each has its own busy state; guards against a live test/audition
// (Rust re-checks) before opening the native save dialog.
function VoiceExportButton({ m, voiceType, lang, picking, onPicking }) {
    const [busy, setBusy] = useState(false);
    // S167: on Export the user picks the format inline: UTAI .zip (lossless re-importable package)
    // vs community-standard files into a plain folder. The community choice is enabled only when
    // the training-side source still exists — installed models are ONNX-only, so the `.pth` must
    // come from the export ledger (has_community_source).
    const [communityOk, setCommunityOk] = useState(null);
    const guardBusyModel = useCallback(() => {
        const vm = useVoiceModelStore.getState();
        if (vm.rangeTesting[m.name] !== undefined || vm.auditionState?.name === m.name) {
            useAppStore.getState().showToast(t18({ zh: "该模型正在测试/试听中，稍后再导出", en: "This model is being tested/auditioned — export later", ja: "このモデルはテスト/試聴中です。後で書き出してください" }, lang), "info");
            return false;
        }
        return true;
    }, [m.name, lang]);
    const zipFlow = useCallback(async () => {
        if (busy || !guardBusyModel())
            return;
        const dest = await save({
            title: t18({ zh: "导出模型为 .zip", en: "Export model as .zip", ja: "モデルを .zip に書き出し" }, lang),
            defaultPath: `${sanitizeExportName(m.name)}.zip`,
            filters: [{ name: "Zip", extensions: ["zip"] }],
        });
        if (!dest || typeof dest !== "string")
            return;
        setBusy(true);
        try {
            await invoke("export_model", { name: m.name, modelType: voiceType, destPath: dest });
            useAppStore.getState().showToast(t18({ zh: `已导出 · ${m.name}`, en: `Exported · ${m.name}`, ja: `書き出し完了 · ${m.name}` }, lang), "success");
        }
        catch (e) {
            useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
        }
        finally {
            setBusy(false);
        }
    }, [busy, guardBusyModel, m.name, voiceType, lang]);
    const communityFlow = useCallback(async () => {
        if (busy || !guardBusyModel())
            return;
        // community format = plain files, no zip (user 2026-08-31) ⇒ a folder picker
        const dest = await open({
            directory: true,
            title: t18({ zh: "选择社区格式的导出文件夹", en: "Pick a folder for the community-format files", ja: "コミュニティ形式の書き出し先フォルダーを選択" }, lang),
        });
        if (!dest || typeof dest !== "string")
            return;
        setBusy(true);
        try {
            const files = await invoke("export_model_community", { name: m.name, modelType: voiceType, destDir: dest });
            useAppStore.getState().showToast(t18({ zh: `已按社区格式导出 ${files.length} 个文件 · ${m.name}`, en: `Exported ${files.length} community-format file(s) · ${m.name}`, ja: `コミュニティ形式で ${files.length} 個のファイルを書き出し · ${m.name}` }, lang), "success");
        }
        catch (e) {
            useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
        }
        finally {
            setBusy(false);
        }
    }, [busy, guardBusyModel, m.name, voiceType, lang]);
    const onExport = useCallback(async () => {
        if (busy || !guardBusyModel())
            return;
        if (voiceType === "vocoder") {
            // vocoders have no community-standard format — go straight to the .zip package
            void zipFlow();
            return;
        }
        let ok = false;
        try {
            ok = await invoke("has_community_source", { name: m.name, modelType: voiceType });
        }
        catch {
            ok = false;
        }
        setCommunityOk(ok);
        onPicking(true);
    }, [busy, guardBusyModel, m.name, voiceType, zipFlow, onPicking]);
    if (picking) {
        return (_jsxs("div", { className: "model-confirm-delete", children: [_jsx("button", { className: "model-export-btn", disabled: busy, onClick: () => { onPicking(false); void zipFlow(); }, title: t18({ zh: "Muno 模型包，可在其它设备导入", en: "Muno package — import on another device", ja: "Muno パッケージ。他のデバイスで取り込めます" }, lang), children: t18({ zh: "Muno 包 (.zip)", en: "Muno package (.zip)", ja: "Muno パッケージ (.zip)" }, lang) }), _jsx("button", { className: "model-export-btn", disabled: busy || communityOk !== true, onClick: () => { onPicking(false); void communityFlow(); }, title: communityOk === true
                        ? t18({ zh: "导出社区通用格式到文件夹（不打包）", en: "Community-standard files into a plain folder (no zip)", ja: "コミュニティ標準形式でフォルダーに書き出し（zip なし）" }, lang)
                        : t18({ zh: "没有可用的社区源：v0.12 起导入模型会保留源 .pth（此模型更早导入，或当初就是 ONNX）——重新导入一次即可解锁；本机训练的模型请在训练页的存档里导出", en: "No community source: since v0.12 importing a model retains its source .pth (this one predates that, or was imported as bare ONNX) — re-import it once to unlock; locally trained models export from the training page's archive", ja: "コミュニティソースがありません：v0.12 以降のインポートは元の .pth を保持します（このモデルはそれ以前、または ONNX 直接インポート）——一度再インポートすると有効になります。ローカル学習モデルは学習ページのアーカイブから書き出してください" }, lang), children: t18({ zh: "社区格式（文件夹）", en: "Community format (folder)", ja: "コミュニティ形式（フォルダー）" }, lang) }), _jsx("button", { className: "model-export-btn", disabled: busy, onClick: () => onPicking(false), children: t18({ zh: "取消", en: "Cancel", ja: "キャンセル" }, lang) })] }));
    }
    return (_jsx("button", { className: "model-export-btn", disabled: busy, onClick: onExport, title: voiceType === "vocoder"
            ? t18({
                zh: "导出为 .zip 模型包，可在其它设备导入",
                en: "Export as a .zip package to import on another device",
                ja: "他のデバイスで取り込める .zip パッケージに書き出し",
            }, lang)
            : t18({
                zh: "导出为 .zip 模型包，可在其它设备导入（含索引/聚类/扩散/头像）",
                en: "Export as a .zip package to import on another device (includes index / cluster / diffusion / avatar)",
                ja: "他のデバイスで取り込める .zip パッケージに書き出し（インデックス/クラスタ/拡散/アバターを含む）",
            }, lang), children: busy
            ? t18({ zh: "导出中…", en: "Exporting…", ja: "書き出し中…" }, lang)
            : t18({ zh: "导出", en: "Export", ja: "書き出し" }, lang) }));
}
// ─── S60-4: per-model audition (resource manager) — the training-audition bare recipe on an
// INSTALLED model via render_model_audition (per-speaker cache in the stem family). Playback
// through the shared preview singleton (contract: stop + assign onEnd on takeover, stop +
// null onEnd on unmount — previewPlayer.ts header). ───
// The S41-era auditionBusyMessage (Chinese substring matchers for the busy guards) is GONE: the S62
// sweep converted every Rust emitter to stable CODEs, so busy classification + localization now live
// entirely in the app-wide mapper (backendErrorMessage / isBusyError — the single source).
function VoiceAuditionButton({ m, voiceType, lang, spk }) {
    // shared audition state (audit S60): the preview player is a singleton — per-row local
    // state desyncs on takeover; ownership of a stop() is proven against preview.path.
    // S82d: `spk` comes from the model row's single speaker selector (shared with the range row).
    const audition = useVoiceModelStore((s) => s.auditionState);
    const speakers = voiceSpeakerOptions(m);
    const showToast = useAppStore((s) => s.showToast);
    const phase = audition?.name === m.name ? audition.phase : "idle";
    const auditionPath = audition?.name === m.name ? (audition?.path ?? null) : null;
    // 试听完成(正在播放,path 已知)后出现的「下载」按钮：把渲染出的试听 wav 原样复制到本地自选文件夹。
    const downloadAudition = async () => {
        if (!auditionPath)
            return;
        const out = await open({ directory: true, title: "下载试听音频" });
        if (!out || typeof out !== "string")
            return;
        try {
            await exportOneAudioFileToFolder({ label: m.name, sourcePath: auditionPath }, out, `${m.name}_试听`);
            showToast("试听音频已下载", "success");
        }
        catch (e) {
            showToast(laneExportErrorMessage(e), "error");
        }
    };
    const start = useCallback(async () => {
        const st = useVoiceModelStore.getState();
        const cur = st.auditionState;
        if (cur?.name === m.name) {
            if (cur.phase === "playing") {
                if (preview.path === cur.path) {
                    // we still own the player — a foreign consumer (training page) may have taken over
                    preview.onEnd = null;
                    preview.stop();
                }
                st.setAuditionState(null);
            }
            return; // rendering → ignore (Rust FlightGuard is the real gate anyway)
        }
        if (cur)
            return; // another row is busy
        st.setAuditionState({ name: m.name, phase: "rendering" });
        try {
            const path = await invoke("render_model_audition", {
                name: m.name,
                modelType: voiceType,
                speakerId: speakers.length > 1 ? spk : null,
            });
            // the manager may have closed / the state may have been torn down mid-render
            if (useVoiceModelStore.getState().auditionState?.name !== m.name)
                return;
            const bytes = await readFile(path);
            const buf = await preview.decode(new Uint8Array(bytes));
            if (useVoiceModelStore.getState().auditionState?.name !== m.name)
                return;
            preview.stop(); // explicit user intent — supersede whatever was playing
            preview.onEnd = () => {
                preview.onEnd = null;
                const a = useVoiceModelStore.getState().auditionState;
                if (a?.name === m.name)
                    useVoiceModelStore.getState().setAuditionState(null);
            };
            await preview.play(path, buf);
            useVoiceModelStore.getState().setAuditionState({ name: m.name, phase: "playing", path });
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            const mapped = backendErrorMessage(msg);
            const busy = isBusyError(msg);
            // S74: audition runs render_model_audition (blocking inference) — leave a copyable log trace.
            logToBackend(busy ? "warn" : "error", `Model audition failed (${m.name}): ${msg}`);
            // S67c: fatal modal-class errors (INFERENCE_LOW_MEMORY) open the alert dialog instead.
            if (!(mapped && maybeShowErrorModal(msg, mapped))) {
                showToast(busy && mapped ? mapped : `${t18({ zh: "试听失败", en: "Audition failed", ja: "試聴に失敗しました" }, lang)}: ${mapped ?? msg}`, busy ? "info" : "error");
            }
            if (useVoiceModelStore.getState().auditionState?.name === m.name) {
                useVoiceModelStore.getState().setAuditionState(null);
            }
        }
    }, [m.name, voiceType, spk, speakers.length, lang, showToast]);
    return (_jsxs("span", { className: "rm-audition", children: [_jsx("button", { className: "rm-range-btn rm-audition-btn", title: t18(phase === "playing"
                    ? { zh: "停止", en: "Stop", ja: "停止" }
                    : { zh: "试听（同训练页口径：裸配方渲染打包干声片段）", en: "Audition (training-page recipe: bare render of the bundled dry clip)", ja: "試聴（トレーニングページと同条件：バンドル済みドライ音声を素の設定でレンダリング）" }, lang), onClick: () => void start(), children: phase === "rendering" ? "…" : phase === "playing" ? "■" : "▶" }), auditionPath && (_jsx("button", { className: "rm-range-btn rm-audition-dl", title: t18({ zh: "下载这段试听音频", en: "Download this audition audio", ja: "この試聴音声をダウンロード" }, lang), onClick: (e) => {
                    e.stopPropagation();
                    void downloadAudition();
                }, children: "\u2B07" }))] }));
}
// ─── S81: batch range test. A PERMANENT row, never a one-shot button (§user) — it is worth
// having whenever models are imported in bulk or the criteria change again, and a run can take
// minutes, so its state has to be visible and interruptible rather than a fire-and-forget click.
// The work list comes from the same pure `collectRangeTestTargets` the count does, so the
// number shown and the work performed can never disagree. ───
function RangeBatchRow({ lang }) {
    const models = useVoiceModelStore((s) => s.models);
    const batch = useVoiceModelStore((s) => s.rangeBatch);
    const cancelBatch = useVoiceModelStore((s) => s.cancelRangeBatch);
    const clearBatch = useVoiceModelStore((s) => s.setRangeBatch);
    const targets = useMemo(() => collectRangeTestTargets(models), [models]);
    if (batch && !batch.finished) {
        return (_jsxs("div", { className: "rm-range-batch", children: [_jsxs("span", { className: "rm-range-batch-text", children: [t18({ zh: "重测音域", en: "Re-testing ranges", ja: "音域を再測定中" }, lang), " ", batch.done, "/", batch.total, batch.cancel && ` · ${t18({ zh: "正在停止", en: "stopping", ja: "停止中" }, lang)}`] }), _jsx("div", { className: "rm-range-batch-bar", children: _jsx("div", { style: { width: `${batch.total ? (batch.done / batch.total) * 100 : 0}%` } }) }), _jsx("button", { className: "rm-range-btn", disabled: batch.cancel, onClick: cancelBatch, children: t18({ zh: "停止", en: "Stop", ja: "停止" }, lang) })] }));
    }
    if (batch?.finished) {
        // Only reachable when something failed — a clean run clears itself (store.finishRangeBatch).
        return (_jsxs("div", { className: "rm-range-batch", children: [_jsxs("span", { className: "rm-range-batch-text rm-range-missing", children: [t18({ zh: "以下模型未能测出音域", en: "These models could not be measured", ja: "以下のモデルは測定できませんでした" }, lang), `: ${batch.failed.join("、")}`] }), _jsx("button", { className: "rm-range-btn", onClick: () => clearBatch(null), children: t18({ zh: "知道了", en: "Dismiss", ja: "閉じる" }, lang) })] }));
    }
    if (!targets.length)
        return null; // nothing to offer → no row at all
    return (_jsxs("div", { className: "rm-range-batch", children: [_jsx("span", { className: "rm-range-batch-text", children: t18({
                    zh: `${targets.length} 个歌手的音域待测或建议重测`,
                    en: `${targets.length} singer(s) need a range test or a re-test`,
                    ja: `${targets.length} 名の話者が音域測定または再測定を必要としています`,
                }, lang) }), _jsx("button", { className: "rm-range-btn", title: t18({
                    zh: "逐个渲染音阶并测量，可能需要几分钟；期间可以随时停止，渲染/播放不会被长时间占用。",
                    en: "Renders and measures a scale per singer; may take minutes. Stoppable at any time, and it never holds the render lock across models.",
                    ja: "話者ごとに音階をレンダリングして測定します（数分かかる場合があります）。いつでも停止でき、レンダリングロックを跨いで保持しません。",
                }, lang), onClick: () => void runRangeTestBatch(targets), children: t18({ zh: "全部重测", en: "Test all", ja: "すべて測定" }, lang) })] }));
}
// ─── S60-2: per-model vocal-range row (v1 session20/21 UX: auto label + comfort editor
// clamped inside usable + Reset + retest; missing record → 补做 button) ───
function VoiceRangeRow({ m, voiceType, lang, spk, onSpk, editing, onEditing }) {
    const progress = useVoiceModelStore((s) => s.rangeTesting[m.name]);
    // S81: the record is keyed PER SPEAKER on every read side (Rust speaker_range, the node
    // gates, the vocal sidebar) but only speaker 0 was ever writable here, so a multi-speaker
    // model's other singers could never get a record — and their range-extend toggle stayed
    // hidden forever with no way to fix it. Co-trained speakers genuinely differ in range (that
    // is the point of co-training), and borrowing speaker 0's ceiling for another singer is
    // actively wrong, not merely imprecise.
    // S82d: ONE speaker selector per model, state lifted to the tab (the audition button
    // follows it — two per-row selects pointing at different singers were semantic drift, and
    // the row was visibly overcrowded, §user). It renders HERE at the range row's start (the
    // row reads as a sentence: singer ▾ comfort … — and this row has slack + wraps), as a
    // 16px chip matching the row rhythm (a full-height select dwarfed the xs lines = the
    // crowding, §user round 2). Switching speaker closes an open comfort edit: the lo/hi
    // sliders were seeded from the previous singer's record.
    useEffect(() => onEditing(false), [spk]); // eslint-disable-line react-hooks/exhaustive-deps
    const speakers = voiceSpeakerOptions(m);
    const speakerPicker = speakers.length > 1 && (_jsx("select", { className: "sep-model-select rm-range-spk", value: spk, title: t18({
            zh: "当前歌手——试听与音域记录都指它（多歌手模型的每位歌手各有自己的音域记录）",
            en: "Active speaker — both audition and the range record point at it (each singer of a multi-speaker model has its own range record)",
            ja: "現在の話者——試聴と音域記録の両方が対象とします（多話者モデルは話者ごとに音域記録を持ちます）",
        }, lang), onChange: (e) => onSpk(Number(e.target.value)), children: speakers.map((s) => (_jsx("option", { value: s.id, children: s.label }, s.id))) }));
    const rec = m.config.vocal_range;
    const sp = rec?.speakers?.[String(spk)];
    // what the render layer will actually target (degenerate stored comfort heals to
    // comfort_auto/usable — mirror of the Rust read side); display + slider seed use THIS
    const shown = sp ? targetRange(sp) : null;
    // model-quirk chips from the stored scan: artifact zones + in-range weak notes, so a
    // weird render at those pitches reads as the MODEL's doing (§user S60d2)
    const caution = sp ? deriveCautionZones(sp.semitones ?? {}, sp.usable, targetRange(sp)) : null;
    // S81 F1: a record measured before the timbre dimension existed still WORKS (its damage curve
    // just can't see timbre), so this is an invitation, never a rejection.
    const stale = sp !== undefined && (sp.scan_version ?? 0) < SCAN_VERSION;
    if (progress !== undefined) {
        return (_jsxs("span", { className: "rm-range-row rm-range-testing", children: [t18({ zh: "音域测试中", en: "Testing range", ja: "音域テスト中" }, lang), " ", Math.round(progress * 100), "%"] }));
    }
    if (!sp) {
        // no record (never tested / lost to a re-import / app crash) → the 补做 entry point
        return (_jsxs("span", { className: "rm-range-row", children: [speakerPicker, _jsx("span", { className: "rm-range-missing", children: t18({ zh: "无音域记录", en: "No range record", ja: "音域記録なし" }, lang) }), _jsx("button", { className: "rm-range-btn", onClick: () => void runRangeTest(m.name, voiceType, m.path, spk), children: t18({ zh: "测音域", en: "Detect range", ja: "音域を測定" }, lang) })] }));
    }
    return (_jsxs(_Fragment, { children: [_jsxs("span", { className: "rm-range-row", children: [speakerPicker, _jsxs("span", { className: "rm-range-text", title: t18({
                            zh: `目标范围 = 被救的音落在哪里。初值由扫描给出（音准 + 浊音 + 音色三项达标的区间），之后以你设的为准。可用范围 ${midiName(sp.usable[0])}–${midiName(sp.usable[1])} 决定哪些音要救，其上沿通常已经很勉强。`,
                            en: `Target is where rescued notes land (seeded by the scan — pitch + voicing + timbre all pass — and yours to override). Usable ${midiName(sp.usable[0])}–${midiName(sp.usable[1])} decides WHICH notes get rescued; its top edge is typically already strained.`,
                            ja: `目標範囲＝救済された音の着地先。初期値は測定値（音程・有声・音色の三項目を満たす範囲）で、以後はユーザー設定が優先されます。使用可能域 ${midiName(sp.usable[0])}–${midiName(sp.usable[1])} はどの音を救済するかを決めます。`,
                        }, lang), children: [t18({ zh: "目标范围", en: "Target", ja: "目標範囲" }, lang), " ", midiName(shown[0]), "\u2013", midiName(shown[1])] }), stale && (_jsx("span", { className: "rm-range-missing", title: t18({
                            zh: "这条记录是在加入音色检测之前测的，仍然可用；重测一次会让目标范围更准。",
                            en: "This record predates the timbre measurement. It still works; re-testing makes the target range more accurate.",
                            ja: "この記録は音色測定の追加前に取得されたものです。引き続き使用できますが、再測定すると目標範囲がより正確になります。",
                        }, lang), children: t18({ zh: "建议重测", en: "re-test suggested", ja: "再測定推奨" }, lang) })), editing ? (
                    // S146e: 四个滑条(可用范围 + 目标范围)现在都在共用编辑器里 —— 人声侧栏挂的是**同一个**
                    // 组件。⛔ 别在这里重写一份:两个入口的夹取规则一旦分叉,一边写出去的记录另一边
                    // 读起来就是错的,而这条 UI 线**零渲染测试**接得住。
                    _jsx("span", { className: "rm-range-edit", children: _jsx(RangeBoundsEditor, { sp: sp, modelName: m.name, backend: voiceType, speakerId: spk, speakerLabel: speakers.find((s) => s.id === spk)?.label, lang: lang, onClose: () => onEditing(false) }) })) : (_jsxs(_Fragment, { children: [_jsx("button", { className: "rm-range-btn", title: t18({ zh: "调整可用范围与目标范围（MIDI 音号）", en: "Adjust the usable and target ranges (MIDI numbers)", ja: "使用可能域と目標範囲を調整（MIDI 番号）" }, lang), onClick: () => onEditing(true), children: t18({ zh: "调整", en: "Adjust", ja: "調整" }, lang) }), _jsx("button", { className: "rm-range-btn", onClick: () => void runRangeTest(m.name, voiceType, m.path, spk), children: t18({ zh: "重测", en: "Retest", ja: "再測定" }, lang) })] }))] }), (caution.artifact.length > 0 || caution.weak.length > 0) && (_jsxs("span", { className: "rm-range-row rm-range-caution-row", children: [caution.artifact.length > 0 && (_jsxs("span", { className: "rm-range-caution", title: `${caution.artifact.map(([a, b]) => `${midiName(a)}–${midiName(b)}`).join(", ")} — ${t18({
                            zh: "模型在这些音高会发声但明显走音（中位误差≥200¢）——模型自身的伪影区，不是程序或算法问题；此区间谨慎使用",
                            en: "the model voices these pitches but lands ≥200¢ off — model-side artifact zones, not a program/algorithm issue; use with caution",
                            ja: "モデルはこの音高で発声しますが大きく音を外します（中央誤差≥200¢）——モデル自体のアーティファクト域です。プログラムの問題ではありません",
                        }, lang)}`, children: [t18({ zh: "伪影", en: "artifacts", ja: "偽影" }, lang), " ", caution.artifact.map(([a, b]) => `${midiName(a)}–${midiName(b)}`).join(", ")] })), caution.weak.length > 0 && (_jsxs("span", { className: "rm-range-caution", title: `${caution.weak.map((n) => midiName(n)).join(", ")} — ${t18({
                            zh: "可用区内部的孤立弱音（测试未达标、推导范围时被桥接跳过）——这些音上出怪声属模型自身问题，谨慎使用",
                            en: "isolated weak notes inside the usable range (failed the probe, bridged over when deriving) — oddities at these pitches are the model's own; use with caution",
                            ja: "使用可能域内の孤立した弱点（測定不合格・範囲導出時にブリッジ）——この音高での異音はモデル由来です",
                        }, lang)}`, children: [t18({ zh: "弱点", en: "weak", ja: "弱点" }, lang), " ", caution.weak.slice(0, 3).map((n) => midiName(n)).join(", "), caution.weak.length > 3 ? ` +${caution.weak.length - 3}` : ""] }))] }))] }));
}
function formatSize(bytes) {
    if (bytes >= 1_000_000_000)
        return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
    if (bytes >= 1_000_000)
        return `${(bytes / 1_000_000).toFixed(0)} MB`;
    if (bytes >= 1_000)
        return `${(bytes / 1_000).toFixed(0)} KB`;
    return `${bytes} B`;
}
