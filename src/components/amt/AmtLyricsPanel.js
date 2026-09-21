import { jsxs as _jsxs, jsx as _jsx, Fragment as _Fragment } from "react/jsx-runtime";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../../store/app";
import { isVocalTrackName } from "../../lib/gmInstruments";
import { distributeLyricsToNotes, mergeLyricLines } from "../../lib/amt/lyricDistribute";
import "./AmtLyricsPanel.css";
/** Human-readable label for a whisper language code. */
const LANGUAGE_LABELS = {
    zh: "中文",
    en: "English",
    ja: "日本語",
    ko: "한국어",
    yue: "粤语",
    fr: "Français",
    de: "Deutsch",
    es: "Español",
    ru: "Русский",
    pt: "Português",
    it: "Italiano",
};
/** whisper model size → preference weight (bigger = better quality). */
const SIZE_RANK = { tiny: 0, base: 1, small: 2, medium: 3 };
function formatTime(sec) {
    if (!Number.isFinite(sec) || sec < 0)
        sec = 0;
    const m = Math.floor(sec / 60);
    const s = sec - m * 60;
    return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}
export function AmtLyricsPanel({ result, onClose, onSavedToMidi }) {
    const { t } = useTranslation();
    const showToast = useAppStore((s) => s.showToast);
    const toggleModelManager = useAppStore((s) => s.toggleModelManager);
    // ── whisper models installed via the resource manager ──
    const [models, setModels] = useState([]);
    const [model, setModel] = useState("small");
    // ── lyrics state ──
    const [lines, setLines] = useState([]);
    const [language, setLanguage] = useState(null);
    const [extracting, setExtracting] = useState(false);
    const [progress, setProgress] = useState(0);
    const [progressMsg, setProgressMsg] = useState("");
    const [dirty, setDirty] = useState(false);
    const [saving, setSaving] = useState(false);
    const [loadedFromMidi, setLoadedFromMidi] = useState(false);
    // §user "最好也要有一个歌词按钮打开一个歌词文本框，可自由更改"：整段文本模式。
    // 一行 = 一句歌词：进入时从当前行填充；应用时按顺序写回各行的文字（时间保留，
    // 多出的行追加到末尾，未覆盖的行保持原词），与 VocalEditor 歌词总编辑一致。
    const [textMode, setTextMode] = useState(false);
    const [textValue, setTextValue] = useState("");
    /** LRC 导出对齐方式：true = 每个音符一行（时间戳 = 音符起点，改词粒度最细）；
     * false = 按句合并（相邻行间隔 ≤0.5s 视为同一句）。 */
    const [lrcNoteAligned, setLrcNoteAligned] = useState(true);
    /** 主 MIDI 的音符起点（秒，人声轨优先）：Whisper 提取后按音节铺到每个音符用。 */
    const noteOnsetsRef = useRef(null);
    const listRef = useRef(null);
    // §user "干音都要有歌词"：优先用分离出的干音 WAV 提取（无伴奏混入，识别更准），
    // 没有干音才退回整曲混音 / 原曲。
    const sourceAudio = result.vocalAudioPath || result.sourceAudioPath || result.originalAudio || "";
    const mainMidi = result.midiPaths?.[0] || "";
    // Discover installed whisper models + auto-pick the best one; also try
    // loading lyrics already embedded in the main MIDI (round-trip support).
    useEffect(() => {
        let alive = true;
        (async () => {
            try {
                const installed = await invoke("list_amt_models");
                const sizes = (installed || [])
                    .filter((m) => m.filename.startsWith("whisper/") && m.is_installed)
                    .map((m) => m.filename.split("/")[1])
                    .filter((s) => Boolean(s));
                if (!alive)
                    return;
                const unique = [...new Set(sizes)];
                setModels(unique);
                const first = unique[0];
                if (unique.length > 0 && first) {
                    setModel(unique.reduce((best, cur) => (SIZE_RANK[cur] ?? -1) > (SIZE_RANK[best] ?? -1) ? cur : best, first));
                }
            }
            catch { /* list_amt_models failure leaves models empty → guided error */ }
            if (mainMidi) {
                try {
                    const existing = await invoke("amt_read_lyrics_from_midi", { midiPath: mainMidi });
                    if (alive && existing && existing.length > 0) {
                        setLines(existing.map((l) => ({ ...l })));
                        setLoadedFromMidi(true);
                    }
                }
                catch { /* no embedded lyrics yet */ }
                // §user「Whisper 提取后按音节切分铺到每个音符」：加载主 MIDI 的音符起点
                //（人声轨优先，无则第一条有音符的轨），供提取后把句级歌词逐音节对齐。
                try {
                    const meta = await invoke("amt_midi_metadata", { midiPath: mainMidi });
                    const score = await invoke("import_score_file", { path: mainMidi });
                    const bpm = Number(meta?.bpm) > 0 ? Number(meta?.bpm) : 120;
                    const ppq = Number(meta?.ppq) > 0 ? Number(meta?.ppq) : 480;
                    const secPerTick = 1 / (ppq * (bpm / 60));
                    const tracks = score?.tracks ?? [];
                    const vocal = tracks.filter((tr) => isVocalTrackName(String(tr.name ?? "")));
                    const pool = vocal.length > 0 ? vocal : tracks;
                    const onsets = pool.flatMap((tr) => (tr.notes ?? []).map((n) => ((Number(n.tick) || 0) + (Number(tr.start_tick) || 0)) * secPerTick));
                    if (alive && onsets.length > 0)
                        noteOnsetsRef.current = onsets;
                }
                catch { /* 无音符信息 → 提取结果回退为句级歌词 */ }
            }
        })();
        return () => { alive = false; };
    }, [mainMidi]);
    // Extraction progress events.
    useEffect(() => {
        if (!extracting)
            return;
        let unlisten = null;
        let alive = true;
        listen("amt-lyrics-progress", (e) => {
            if (!alive)
                return;
            const p = Number(e.payload?.progress ?? 0);
            const total = Number(e.payload?.total ?? 1) || 1;
            setProgress(Math.min(1, Math.max(0, p / total)));
            if (e.payload?.message)
                setProgressMsg(e.payload.message);
        }).then((fn) => { unlisten = fn; }).catch(() => { });
        return () => { alive = false; unlisten?.(); };
    }, [extracting]);
    const handleExtract = useCallback(async () => {
        if (!sourceAudio) {
            showToast(t("amt.lyricsNoAudio") || "缺少源音频，无法提取歌词", "error");
            return;
        }
        if (extracting)
            return;
        setExtracting(true);
        setProgress(0);
        setProgressMsg(t("amt.lyricsLoading") || "加载模型中…");
        try {
            const res = await invoke("amt_extract_lyrics", {
                audioPath: sourceAudio,
                outputDir: result.outputDir,
                model,
                language: "",
            });
            // §user「Whisper 提取后按音节切分铺到每个音符」：有音符起点时把每句
            // 切成音节（中文一字、英文按空格、日文按 mora）逐个落到音符上，
            // 每个音符一个词；没有音符信息则保留句级。
            const onsets = noteOnsetsRef.current;
            setLines(onsets && onsets.length > 0
                ? distributeLyricsToNotes(res.segments, onsets)
                : res.segments.map((s) => ({ ...s })));
            setLanguage(res.language);
            setDirty(true);
            setLoadedFromMidi(false);
            showToast(t("amt.lyricsExtracted", {
                count: res.segments.length,
                lang: LANGUAGE_LABELS[res.language ?? ""] || res.language || "",
                device: res.device === "cuda" ? "GPU" : "CPU",
            }) || `已提取 ${res.segments.length} 句歌词（${res.device === "cuda" ? "GPU" : "CPU"}）`, "success");
        }
        catch (e) {
            const msg = String(e);
            let userMsg = msg;
            if (msg.includes("LYRICS_MODEL_NOT_INSTALLED")) {
                userMsg = t("amt.lyricsModelMissing") || "歌词模型未安装：请先在资源管理中下载 Whisper 模型（推荐 Small）";
            }
            else if (msg.includes("LYRICS_DEP_MISSING")) {
                userMsg = t("amt.lyricsDepMissing") || "歌词组件缺失：需要联网安装 faster-whisper 运行库";
            }
            else if (msg.includes("LYRICS_AUDIO_NOT_FOUND")) {
                userMsg = t("amt.lyricsNoAudio") || "缺少源音频，无法提取歌词";
            }
            else if (msg.includes("LYRICS_SPAWN_FAILED")) {
                userMsg = t("amt.lyricsSpawnFailed") || "无法启动歌词提取进程，请检查 AMT 运行环境";
            }
            showToast(userMsg, "error");
        }
        finally {
            setExtracting(false);
            setProgressMsg("");
        }
    }, [sourceAudio, result.outputDir, model, extracting, showToast, t]);
    const handleCancelExtract = useCallback(async () => {
        try {
            await invoke("cancel_amt_midi", { nodeId: "amt-lyrics" });
        }
        catch { /* nothing registered */ }
    }, []);
    const updateLine = useCallback((idx, patch) => {
        setLines((prev) => prev.map((l, i) => (i === idx ? { ...l, ...patch } : l)));
        setDirty(true);
    }, []);
    const deleteLine = useCallback((idx) => {
        setLines((prev) => prev.filter((_, i) => i !== idx));
        setDirty(true);
    }, []);
    const addLine = useCallback(() => {
        setLines((prev) => {
            const last = prev[prev.length - 1];
            const start = last ? last.end + 0.1 : 0;
            return [...prev, { start, end: start + 2, text: "" }];
        });
        setDirty(true);
        requestAnimationFrame(() => {
            listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
        });
    }, []);
    const nudgeTime = useCallback((idx, delta) => {
        setLines((prev) => prev.map((l, i) => (i === idx
            ? { ...l, start: Math.max(0, Number((l.start + delta).toFixed(2))) }
            : l)));
        setDirty(true);
    }, []);
    const enterTextMode = useCallback(() => {
        setTextValue(lines.map((l) => l.text.trim()).join("\n"));
        setTextMode(true);
    }, [lines]);
    const applyTextMode = useCallback(() => {
        const rows = textValue
            .split(/\r?\n/)
            .map((r) => r.trim())
            .filter((r) => r.length > 0);
        if (rows.length === 0) {
            setTextMode(false);
            return;
        }
        setLines((prev) => {
            const next = prev.map((l, i) => (i < rows.length ? { ...l, text: rows[i] } : l));
            if (rows.length > prev.length) {
                let cursor = prev.length > 0 ? prev[prev.length - 1].end + 0.1 : 0;
                for (let i = prev.length; i < rows.length; i++) {
                    next.push({ start: cursor, end: cursor + 2, text: rows[i] });
                    cursor += 2.1;
                }
            }
            return next;
        });
        setDirty(true);
        setTextMode(false);
    }, [textValue]);
    const handleSaveToMidi = useCallback(async () => {
        if (!mainMidi) {
            showToast(t("amt.lyricsNoMidi") || "当前结果没有 MIDI 文件，无法写回歌词", "error");
            return;
        }
        const valid = lines.filter((l) => l.text.trim().length > 0);
        if (valid.length === 0) {
            showToast(t("amt.lyricsWriteEmpty") || "没有可写回的歌词行（文本均为空）", "error");
            return;
        }
        setSaving(true);
        try {
            await invoke("amt_write_lyrics_to_midi", {
                midiPath: mainMidi,
                outPath: mainMidi,
                lyrics: valid.map((l) => ({ start: l.start, text: l.text.trim() })),
            });
            setDirty(false);
            setLoadedFromMidi(true);
            onSavedToMidi?.(mainMidi);
            showToast(t("amt.lyricsSaved") || "歌词已写回 MIDI 文件", "success");
        }
        catch (e) {
            showToast(String(e), "error");
        }
        finally {
            setSaving(false);
        }
    }, [mainMidi, lines, showToast, t, onSavedToMidi]);
    const handleExportLrc = useCallback(async () => {
        const valid = lines.filter((l) => l.text.trim().length > 0);
        if (valid.length === 0) {
            showToast(t("amt.lyricsWriteEmpty") || "没有可导出的歌词行", "error");
            return;
        }
        try {
            const { save } = await import("@tauri-apps/plugin-dialog");
            const out = await save({
                filters: [{ name: "LRC", extensions: ["lrc"] }],
                defaultPath: "lyrics.lrc",
            });
            if (!out)
                return;
            // §user「LRC 导出加按音符对齐选项」：默认每个音符一行（时间戳 = 音符起点）；
            // 关闭后按句合并导出（相邻行间隔 ≤0.5s 视为同一句）。
            const exported = lrcNoteAligned ? valid : mergeLyricLines(valid);
            const content = exported
                .map((l) => `[${formatTime(l.start)}]${l.text.trim()}`)
                .join("\r\n");
            const { writeTextFile } = await import("@tauri-apps/plugin-fs");
            await writeTextFile(out, content);
            showToast(t("amt.lyricsLrcSaved") || "LRC 歌词已导出", "success");
        }
        catch (e) {
            showToast(String(e), "error");
        }
    }, [lines, lrcNoteAligned, showToast, t]);
    const langLabel = useMemo(() => (language ? (LANGUAGE_LABELS[language] || language.toUpperCase()) : null), [language]);
    const modelOptions = useMemo(() => ["tiny", "base", "small", "medium"].filter((s) => models.includes(s)), [models]);
    return (_jsxs("div", { className: "amt-lyrics-drawer", role: "complementary", "aria-label": t("amt.lyricsTitle") || "歌词", children: [_jsxs("div", { className: "amt-lyrics-header", children: [_jsxs("div", { className: "amt-lyrics-title-group", children: [_jsxs("h3", { children: ["\uD83C\uDFA4 ", t("amt.lyricsTitle") || "歌词"] }), langLabel && (_jsx("span", { className: "amt-lyrics-lang", title: t("amt.lyricsLangDetected") || "自动检测的语言", children: langLabel })), loadedFromMidi && !dirty && (_jsx("span", { className: "amt-lyrics-badge", children: t("amt.lyricsFromMidi") || "已载入MIDI内歌词" })), dirty && (_jsx("span", { className: "amt-lyrics-badge warn", children: t("amt.lyricsDirty") || "未保存" }))] }), _jsx("button", { className: "amt-lyrics-close", onClick: onClose, title: t("amt.lyricsClose") || "关闭", children: "\u2715" })] }), _jsxs("div", { className: "amt-lyrics-toolbar", children: [_jsxs("div", { className: "amt-lyrics-model-row", children: [_jsx("label", { className: "amt-lyrics-model-label", children: t("amt.lyricsModel") || "模型" }), modelOptions.length > 0 ? (_jsx("select", { className: "amt-lyrics-model-select", value: model, onChange: (e) => setModel(e.target.value), disabled: extracting, children: modelOptions.map((s) => (_jsx("option", { value: s, children: s }, s))) })) : (_jsx("button", { className: "amt-lyrics-get-model", onClick: () => toggleModelManager(), title: t("amt.lyricsGetModelTooltip") || "打开资源管理下载 Whisper 歌词模型", children: t("amt.lyricsGetModel") || "下载歌词模型…" }))] }), _jsxs("div", { className: "amt-lyrics-actions", children: [extracting ? (_jsxs(_Fragment, { children: [_jsx("button", { className: "amt-lyrics-btn primary", disabled: true, children: t("amt.lyricsExtracting") || "提取中…" }), _jsx("button", { className: "amt-lyrics-btn danger", onClick: handleCancelExtract, children: t("amt.lyricsCancel") || "停止" })] })) : (_jsxs("button", { className: "amt-lyrics-btn primary", onClick: handleExtract, disabled: modelOptions.length === 0, title: modelOptions.length === 0
                                    ? (t("amt.lyricsModelMissing") || "请先在资源管理中下载 Whisper 模型")
                                    : (t("amt.lyricsExtractTooltip") || "使用 Whisper 从歌曲中识别带时间轴的歌词"), children: ["\u2728 ", t("amt.lyricsExtract") || "提取歌词"] })), _jsx("button", { className: "amt-lyrics-btn", onClick: () => (textMode ? setTextMode(false) : enterTextMode()), disabled: extracting, title: t("amt.lyricsTextToggleTooltip") || "整段歌词文本编辑：一行一句，按顺序写回各行", children: textMode
                                    ? (t("amt.lyricsTextExit") || "返回行模式")
                                    : (t("amt.lyricsTextMode") || "文本模式") }), _jsx("button", { className: "amt-lyrics-btn", onClick: handleSaveToMidi, disabled: extracting || saving || lines.length === 0 || textMode, title: textMode
                                    ? (t("amt.lyricsTextSaveHint") || "请先应用文本（返回行模式）再写回 MIDI")
                                    : (t("amt.lyricsSaveTooltip") || "将当前歌词（含编辑）写入 MIDI 文件"), children: saving ? (t("amt.lyricsSaving") || "保存中…") : (t("amt.lyricsSave") || "写回 MIDI") }), _jsx("button", { className: `amt-lyrics-btn${lrcNoteAligned ? " toggle-on" : ""}`, onClick: () => setLrcNoteAligned((v) => !v), disabled: extracting, title: lrcNoteAligned
                                    ? (t("amt.lyricsLrcNoteAligned") || "LRC 对齐：按音符（每个音符一行，点击切换为按句）")
                                    : (t("amt.lyricsLrcByLine") || "LRC 对齐：按句（相邻行合并为一句，点击切换为按音符）"), children: lrcNoteAligned ? "LRC·音符" : "LRC·按句" }), _jsx("button", { className: "amt-lyrics-btn", onClick: handleExportLrc, disabled: extracting || lines.length === 0, title: t("amt.lyricsLrcTooltip") || "导出为 .lrc 歌词文件", children: "\u2B07 LRC" })] })] }), extracting && (_jsxs("div", { className: "amt-lyrics-progress-wrap", children: [_jsx("div", { className: "amt-lyrics-progress-bar", children: _jsx("div", { className: "amt-lyrics-progress-fill", style: { width: `${Math.round(progress * 100)}%` } }) }), _jsxs("div", { className: "amt-lyrics-progress-text", children: [progressMsg || t("amt.lyricsExtracting") || "提取中…", " \u00B7 ", Math.round(progress * 100), "%"] })] })), textMode && (_jsxs("div", { className: "amt-lyrics-textmode", children: [_jsx("div", { className: "amt-lyrics-textmode-hint", children: t("amt.lyricsTextHint") || "一行 = 一句歌词（回车换行）。应用后按顺序写入各行，时间轴保留；多出的行自动追加到末尾。" }), _jsx("textarea", { className: "amt-lyrics-textarea", value: textValue, placeholder: t("amt.lyricsTextPlaceholder") || "把整首歌词粘贴到这里，一行一句…", onChange: (e) => setTextValue(e.target.value), spellCheck: false }), _jsxs("div", { className: "amt-lyrics-textmode-actions", children: [_jsxs("button", { className: "amt-lyrics-btn primary", onClick: applyTextMode, children: ["\u2713 ", t("amt.lyricsTextApply") || "应用"] }), _jsx("button", { className: "amt-lyrics-btn", onClick: () => setTextMode(false), children: t("amt.lyricsTextCancel") || "取消" })] })] })), !textMode && (_jsxs("div", { className: "amt-lyrics-list", ref: listRef, children: [lines.length === 0 && !extracting && (_jsxs("div", { className: "amt-lyrics-empty", children: [_jsx("div", { className: "amt-lyrics-empty-icon", children: "\uD83C\uDFA4" }), _jsx("p", { children: t("amt.lyricsEmptyHint") || "点击「提取歌词」，自动识别歌曲语言（中文 / 英文 / 日文…）并生成带时间轴的歌词。" }), _jsx("p", { className: "amt-lyrics-empty-sub", children: t("amt.lyricsEmptySub") || "识别后可自由编辑文字与时间，然后「写回 MIDI」或导出 LRC。" })] })), lines.map((line, idx) => (_jsxs("div", { className: "amt-lyrics-row", children: [_jsxs("div", { className: "amt-lyrics-time", children: [_jsx("button", { className: "amt-lyrics-nudge", onClick: () => nudgeTime(idx, -0.5), disabled: extracting, title: t("amt.lyricsNudgeBack") || "-0.5s", children: "\u25C0" }), _jsx("span", { className: "amt-lyrics-time-text", title: t("amt.lyricsStartTime") || "该行开始时间", children: formatTime(line.start) }), _jsx("button", { className: "amt-lyrics-nudge", onClick: () => nudgeTime(idx, 0.5), disabled: extracting, title: t("amt.lyricsNudgeFwd") || "+0.5s", children: "\u25B6" })] }), _jsx("input", { className: "amt-lyrics-input", value: line.text, placeholder: t("amt.lyricsLinePlaceholder") || "输入歌词…", disabled: extracting, onChange: (e) => updateLine(idx, { text: e.target.value }) }), _jsx("button", { className: "amt-lyrics-delete", onClick: () => deleteLine(idx), disabled: extracting, title: t("amt.lyricsDeleteLine") || "删除该行", children: "\u2715" })] }, idx)))] })), _jsxs("div", { className: "amt-lyrics-footer", children: [!textMode && (_jsxs("button", { className: "amt-lyrics-btn", onClick: addLine, disabled: extracting, children: ["\uFF0B ", t("amt.lyricsAddLine") || "添加一行"] })), _jsx("span", { className: "amt-lyrics-count", children: t("amt.lyricsCount", { count: lines.length }) || `${lines.length} 行` })] })] }));
}
