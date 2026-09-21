import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * MultiTrackStudio - 多轨工作室
 *
 * ACE-Step v1.5（Base 多轨 / XL-Turbo 快速任务）：
 *   1. 叠加音轨 (lego)      — 在已有音频上叠加一条新乐器轨道（12 种轨道可选）
 *   2. 分轨分离 (extract)   — 从成品音频中分离出指定轨道
 *   3. 人声转伴奏 (complete) — 上传人声干声，自动补全匹配的伴奏（Vocal2BGM）
 *   4. 风格翻唱 (cover)     — 保留原曲旋律结构，转换为目标风格
 *   5. 局部重绘 (repaint)   — 只重新生成 [start, end) 区间的内容
 *
 * YuE-2 3B（符号乐谱 ABC 方案，YuE2 无独立分轨模块）：
 *   6. YuE2 乐谱 — 新乐器轨道（按原曲 BPM/风格生成独奏轨 + ABC→MIDI）/ 乐谱续写
 */
import { useEffect, useMemo, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { songGenerate, listSongModels } from "../../lib/backendSong";
import "./MultiTrackStudio.css";
const TRACK_NAMES = [
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals",
];
const TRACK_LABELS = {
    woodwinds: "木管 Woodwinds",
    brass: "铜管 Brass",
    fx: "特效 FX",
    synth: "合成器 Synth",
    strings: "弦乐 Strings",
    percussion: "打击乐 Percussion",
    keyboard: "键盘 Keyboard",
    guitar: "吉他 Guitar",
    bass: "贝斯 Bass",
    drums: "鼓 Drums",
    backing_vocals: "和声 Backing Vocals",
    vocals: "人声 Vocals",
};
/** YuE2 独奏轨道生成用的乐器（符号乐谱方案） */
const YUE2_INSTRUMENTS = [
    { value: "piano", zh: "钢琴" },
    { value: "acoustic guitar", zh: "木吉他" },
    { value: "electric guitar", zh: "电吉他" },
    { value: "bass guitar", zh: "贝斯" },
    { value: "drums", zh: "鼓组" },
    { value: "strings ensemble", zh: "弦乐组" },
    { value: "synth pad", zh: "合成器 Pad" },
    { value: "saxophone", zh: "萨克斯" },
    { value: "erhu", zh: "二胡" },
    { value: "flute", zh: "长笛" },
];
/** 风格翻唱预设（点击追加到描述） */
const COVER_STYLE_PRESETS = [
    { label: "流行", tag: "pop" },
    { label: "摇滚", tag: "rock" },
    { label: "电子", tag: "electronic dance" },
    { label: "爵士", tag: "jazz" },
    { label: "民谣", tag: "folk acoustic" },
    { label: "说唱", tag: "hip-hop rap" },
    { label: "古风", tag: "chinese traditional" },
    { label: "轻音乐", tag: "light instrumental" },
    { label: "Lo-Fi", tag: "lo-fi chillhop" },
    { label: "管弦", tag: "orchestral cinematic" },
];
const TABS = [
    { id: "lego", title: "叠加音轨", desc: "在当前音频上叠加一条新轨道（鼓/贝斯/人声…），模型自动对齐原曲调式、BPM 与律动" },
    { id: "extract", title: "分轨分离", desc: "从成品音频中分离出指定轨道（Stem 分离，Base 模型原生提取）" },
    { id: "vocal2bgm", title: "人声转伴奏", desc: "上传人声干声，自动生成匹配的伴奏（Vocal2BGM）" },
    { id: "cover", title: "风格翻唱", desc: "保留原曲旋律与结构，把整体风格转换为目标曲风" },
    { id: "repaint", title: "局部重绘", desc: "只重新生成选定时间区间的内容，区间外保持原样" },
    { id: "yue2score", title: "YuE2 乐谱", desc: "YuE-2 无独立分轨模块，通过符号乐谱（ABC）实现：生成单乐器轨 / 续写乐谱，可转 MIDI" },
];
/** 每个 tab 对应的模型（与模型列表显示逻辑一致） */
const TAB_MODEL = {
    lego: { id: "acestep-v1.5", name: "ACE-Step V1.5 (Base)", desc: "多轨任务专用 · 50 步推理" },
    extract: { id: "acestep-v1.5", name: "ACE-Step V1.5 (Base)", desc: "原生分轨提取 · 50 步推理" },
    vocal2bgm: { id: "acestep-v1.5", name: "ACE-Step V1.5 (Base)", desc: "音轨补全 · 50 步推理" },
    cover: { id: "acestep-v1.5", name: "ACE-Step V1.5 (XL-Turbo)", desc: "快速风格转换 · 8 步推理" },
    repaint: { id: "acestep-v1.5", name: "ACE-Step V1.5 (XL-Turbo)", desc: "快速局部重绘 · 8 步推理" },
    yue2score: { id: "yue2-3b", name: "YuE-2 3B", desc: "符号乐谱 ABC → 音频 + MIDI" },
};
const BASE_FILES_PREFIX = "acestep-v1.5/checkpoints/acestep-v15-base/";
const YUE2_FILES_PREFIX = "yue2-3b/";
export function MultiTrackStudio({ songs, initialAudioPath, initialTask, onClose, onGenerated, onSendToTrack }) {
    const [tab, setTab] = useState(initialTask ?? "lego");
    const [modelInstalled, setModelInstalled] = useState(null);
    const audioOptions = useMemo(() => songs.filter((s) => s.audioPath), [songs]);
    const [srcPath, setSrcPath] = useState(initialAudioPath && initialAudioPath.length > 0 ? initialAudioPath : (audioOptions[0]?.audioPath ?? ""));
    // lego / extract
    const [track, setTrack] = useState("drums");
    const [extractTrack, setExtractTrack] = useState("vocals");
    // cover
    const [coverPrompt, setCoverPrompt] = useState("");
    const [coverStrength, setCoverStrength] = useState(1.0);
    // repaint
    const [repaintStart, setRepaintStart] = useState("0");
    const [repaintEnd, setRepaintEnd] = useState("10");
    const [repaintPrompt, setRepaintPrompt] = useState("");
    const [repaintLyrics, setRepaintLyrics] = useState("");
    // yue2
    const [yue2Mode, setYue2Mode] = useState("instrument");
    const [yue2Instrument, setYue2Instrument] = useState("piano");
    const [yue2StyleText, setYue2StyleText] = useState("");
    const [yue2Abc, setYue2Abc] = useState("");
    const [yue2SongUuid, setYue2SongUuid] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    const srcSong = audioOptions.find((s) => s.audioPath === srcPath);
    const activeTab = TABS.find((t) => t.id === tab);
    const activeModel = TAB_MODEL[tab];
    // 检查 Base / YuE2 模型是否已安装（安装清单来自后端）
    useEffect(() => {
        let cancelled = false;
        listSongModels()
            .then((files) => {
            if (cancelled)
                return;
            const base = files.some((f) => f.filename.startsWith(BASE_FILES_PREFIX));
            const yue2 = files.some((f) => f.filename.startsWith(YUE2_FILES_PREFIX));
            setModelInstalled({ base, yue2 });
        })
            .catch(() => { if (!cancelled)
            setModelInstalled({ base: true, yue2: true }); });
        return () => { cancelled = true; };
    }, []);
    function switchTab(t) {
        setTab(t);
        setError(null);
        setResult(null);
    }
    async function pickExternalFile() {
        const picked = await open({
            multiple: false,
            filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac", "ogg", "m4a"] }],
        });
        if (typeof picked === "string" && picked) {
            setSrcPath(picked);
        }
    }
    async function loadAbcFromSong() {
        const song = songs.find((s) => s.uuid === yue2SongUuid);
        if (!song?.abcPath) {
            setError("所选歌曲没有 ABC 乐谱文件（仅 YuE-2 生成的歌曲带有 .abc）");
            return;
        }
        try {
            const text = await readTextFile(song.abcPath);
            setYue2Abc(text);
            setError(null);
        }
        catch (e) {
            setError(`读取 ABC 文件失败：${String(e)}`);
        }
    }
    function validate() {
        if (tab === "yue2score") {
            if (yue2Mode === "continue" && !yue2Abc.trim())
                return "请粘贴或加载要续写的 ABC 乐谱";
            return null;
        }
        if (!srcPath)
            return "请先选择源音频（可从歌曲列表选择，或选择本地文件）";
        if (tab === "repaint") {
            const s = Number(repaintStart), e = Number(repaintEnd);
            if (Number.isNaN(s) || Number.isNaN(e))
                return "重绘区间必须是数字（秒）";
            if (s < 0 || e <= s)
                return "重绘区间无效：需要 0 ≤ 起点 < 终点";
            const dur = srcSong?.durationSec ?? 0;
            if (dur > 0 && e > dur)
                return `重绘终点 ${e}s 超出源音频时长 ${dur}s`;
        }
        return null;
    }
    async function handleGenerate() {
        const v = validate();
        if (v) {
            setError(v);
            return;
        }
        setBusy(true);
        setError(null);
        setResult(null);
        try {
            let outputs;
            let label;
            let durationSec;
            let modelId;
            let wantMidi = false;
            if (tab === "yue2score") {
                // ── YuE-2 符号乐谱方案 ──
                modelId = "yue2-3b";
                wantMidi = true;
                let prompt;
                if (yue2Mode === "instrument") {
                    // 新乐器轨道：沿用源歌曲的 BPM/风格，生成单乐器独奏（模型对齐节奏靠 BPM 提示）
                    const inst = YUE2_INSTRUMENTS.find((i) => i.value === yue2Instrument);
                    const bpmPart = srcSong?.bpm ? `${srcSong.bpm} BPM, ` : "";
                    const stylePart = (yue2StyleText.trim() || srcSong?.styleHint || "");
                    prompt = `solo ${yue2Instrument} instrumental track, ${bpmPart}${stylePart}`.trim();
                    label = `${srcSong?.label || "song"}_yue2_${inst?.zh || yue2Instrument}`;
                }
                else {
                    // 乐谱续写：把已有 ABC 作为条件传给 YuE-2，按描述延展
                    prompt = yue2StyleText.trim() || "continue the score in the same style, seamless extension";
                    label = `abc_continue_${Date.now().toString(36)}`;
                }
                const baseName = srcSong?.label?.replace(/[\\/:*?"<>|]/g, "_") || "yue2";
                outputs = await songGenerate({
                    model: modelId,
                    task: "text2music",
                    song_name: `${baseName}_yue2_${yue2Mode}`,
                    prompt,
                    lyrics: "[Instrumental]",
                    audio_duration: 0,
                    format: "wav",
                    output_dir: "",
                    want_midi: true,
                    ...(yue2Mode === "continue" ? { abc: yue2Abc } : {}),
                });
                const pa = outputs.find((o) => o.audio_path);
                if (!pa?.audio_path)
                    throw new Error("生成完成但没有返回音频文件");
                durationSec = srcSong?.durationSec && yue2Mode === "instrument" ? srcSong.durationSec : 60;
            }
            else {
                // ── ACE-Step 任务 ──
                modelId = "acestep-v1.5";
                const taskType = tab === "vocal2bgm" ? "complete" : tab;
                const trackName = tab === "lego" ? track : tab === "extract" ? extractTrack : undefined;
                const baseName = srcPath.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "src";
                label = `${baseName}_${taskType}${trackName ? `_${trackName}` : ""}`;
                const prompt = tab === "cover" ? coverPrompt.trim() : tab === "repaint" ? repaintPrompt.trim()
                    : tab === "lego" ? "" : "";
                outputs = await songGenerate({
                    model: modelId,
                    task: taskType,
                    src_audio_path: srcPath,
                    track_name: trackName,
                    prompt,
                    lyrics: tab === "repaint" ? repaintLyrics : "",
                    song_name: label,
                    audio_duration: 0, // direct-conditioning 任务自动取源音频时长
                    format: "wav",
                    output_dir: "",
                    ...(tab === "cover" ? { audio_cover_strength: coverStrength } : {}),
                    ...(tab === "repaint" ? { repaint_start: Number(repaintStart), repaint_end: Number(repaintEnd) } : {}),
                });
                const pa = outputs.find((o) => o.audio_path);
                if (!pa?.audio_path)
                    throw new Error("生成完成但没有返回音频文件");
                durationSec = srcSong?.durationSec ?? 60;
            }
            setResult({ outputs, label, durationSec, modelId, wantMidi });
        }
        catch (err) {
            const msg = typeof err === "string" ? err : err instanceof Error ? err.message : JSON.stringify(err);
            setError(msg);
        }
        finally {
            setBusy(false);
        }
    }
    const resultAudio = result?.outputs.find((o) => o.audio_path)?.audio_path;
    const resultMidi = result?.outputs.find((o) => o.midi_path)?.midi_path;
    const modelWarning = (tab === "yue2score" && modelInstalled && !modelInstalled.yue2) ? "YuE-2 3B 模型尚未下载，请先到「资源管理 → 生成歌曲」下载。" :
        (tab !== "yue2score" && modelInstalled && !modelInstalled.base) ? "ACE-Step V1.5 (Base) 模型尚未下载（约 4.8GB），请先到「资源管理 → 生成歌曲」下载。" :
            null;
    return (_jsx("div", { className: "mts-overlay", onClick: onClose, children: _jsxs("div", { className: "mts-dialog", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "mts-header", children: [_jsx("div", { className: "mts-title", children: "\uD83C\uDF9B \u591A\u8F68\u5DE5\u4F5C\u5BA4" }), _jsx("div", { className: "mts-subtitle", children: "\u6A21\u578B\u5C42\u9762\u5206\u5C42\u751F\u6210 / \u53E0\u52A0 \u00B7 \u975E\u53EF\u89C6\u5316 DAW \u7F16\u8F91\u5668" }), _jsx("button", { className: "mts-close", onClick: onClose, children: "\u2715" })] }), _jsx("div", { className: "mts-tabs", children: TABS.map((t) => (_jsx("button", { className: `mts-tab ${tab === t.id ? "active" : ""}`, onClick: () => switchTab(t.id), children: t.title }, t.id))) }), _jsx("div", { className: "mts-tab-desc", children: activeTab.desc }), _jsxs("div", { className: "mts-model-row", children: [_jsx("label", { className: "mts-label", children: "\u4F7F\u7528\u6A21\u578B" }), _jsx("select", { className: "mts-select mts-model-select", value: activeModel.id, disabled: true, children: _jsx("option", { value: activeModel.id, children: activeModel.name }) }), _jsx("span", { className: "mts-model-desc", children: activeModel.desc })] }), modelWarning && _jsxs("div", { className: "mts-warn", children: ["\u26A0 ", modelWarning] }), _jsxs("div", { className: "mts-body", children: [tab !== "yue2score" && (_jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u6E90\u97F3\u9891" }), _jsxs("div", { className: "mts-src-row", children: [_jsxs("select", { className: "mts-select", value: srcPath, onChange: (e) => setSrcPath(e.target.value), children: [audioOptions.length === 0 && _jsx("option", { value: "", children: "\uFF08\u6B4C\u66F2\u5217\u8868\u6682\u65E0\u97F3\u9891\uFF09" }), audioOptions.map((s) => (_jsxs("option", { value: s.audioPath, children: [s.label || "Untitled", "\uFF08", s.durationSec, "s\uFF09"] }, s.uuid))), srcPath && !audioOptions.some((s) => s.audioPath === srcPath) && (_jsxs("option", { value: srcPath, children: ["\u672C\u5730\u6587\u4EF6\uFF1A", srcPath.split(/[\\/]/).pop()] }))] }), _jsx("button", { className: "mts-btn mts-btn-ghost", onClick: pickExternalFile, disabled: busy, children: "\u9009\u62E9\u672C\u5730\u6587\u4EF6" })] }), _jsx("div", { className: "mts-hint", children: "\u652F\u6301 WAV / MP3 / FLAC / OGG / M4A\uFF0C\u5EFA\u8BAE 44.1kHz+\uFF1B\u5206\u8F68\u5206\u79BB / \u53E0\u52A0 / \u8865\u5168 / \u7FFB\u5531 / \u91CD\u7ED8\u90FD\u4F1A\u4FDD\u6301\u6E90\u97F3\u9891\u65F6\u957F\u3002" })] })), tab === "lego" && (_jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u8981\u53E0\u52A0\u7684\u8F68\u9053" }), _jsx("select", { className: "mts-select", value: track, onChange: (e) => setTrack(e.target.value), disabled: busy, children: TRACK_NAMES.map((t) => (_jsx("option", { value: t, children: TRACK_LABELS[t] }, t))) }), _jsx("div", { className: "mts-hint", children: "\u751F\u6210\u7ED3\u679C\u53EF\u300C\u53D1\u9001\u5230\u8F68\u9053\u300D\uFF0C\u65B0\u8F68\u9053\u4E0E\u539F\u66F2\u8C03\u5F0F/BPM \u81EA\u52A8\u5BF9\u9F50\uFF1B\u97F3\u91CF\u5E73\u8861\u5728\u4E3B\u65F6\u95F4\u7EBF\u7684\u8F68\u9053\u63A8\u5B50\u4E0A\u8C03\u8282\u3002" })] })), tab === "extract" && (_jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u8981\u5206\u79BB\u51FA\u7684\u8F68\u9053" }), _jsx("select", { className: "mts-select", value: extractTrack, onChange: (e) => setExtractTrack(e.target.value), disabled: busy, children: TRACK_NAMES.map((t) => (_jsx("option", { value: t, children: TRACK_LABELS[t] }, t))) }), _jsx("div", { className: "mts-hint", children: "\u6BCF\u6B21\u5206\u79BB\u4E00\u6761\u8F68\u9053\uFF1B\u5206\u79BB\u7ED3\u679C\u4FDD\u5B58\u5230\u6B4C\u66F2\u5217\u8868\u540E\uFF0C\u53EF\u76F4\u63A5\u4F5C\u4E3A\u300C\u98CE\u683C\u7FFB\u5531\u300D\u6216\u300C\u5C40\u90E8\u91CD\u7ED8\u300D\u7684\u6E90\u97F3\u9891\u8054\u52A8\u4F7F\u7528\u3002" })] })), tab === "vocal2bgm" && (_jsx("div", { className: "mts-hint", children: "\u9009\u62E9\u4E00\u6761\u4EBA\u58F0\u5E72\u58F0\u4F5C\u4E3A\u6E90\u97F3\u9891\uFF0C\u6A21\u578B\u4F1A\u81EA\u52A8\u8865\u5168\u4E0E\u5176\u8C03\u5F0F/BPM \u5339\u914D\u7684\u4F34\u594F\uFF08 drums / bass / \u548C\u58F0\u7B49\u5168\u4E50\u5668\uFF09\u3002" })), tab === "cover" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u98CE\u683C\u9884\u8BBE\uFF08\u70B9\u51FB\u8FFD\u52A0\uFF09" }), _jsx("div", { className: "mts-chips", children: COVER_STYLE_PRESETS.map((p) => (_jsx("button", { className: "mts-chip", onClick: () => setCoverPrompt((v) => (v.includes(p.tag) ? v : `${v ? v + ", " : ""}${p.tag}`)), disabled: busy, children: p.label }, p.tag))) })] }), _jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u76EE\u6807\u98CE\u683C\u63CF\u8FF0" }), _jsx("textarea", { className: "mts-textarea", rows: 2, placeholder: "\u4F8B\u5982\uFF1Aenergetic rock, distorted guitar, powerful drums", value: coverPrompt, onChange: (e) => setCoverPrompt(e.target.value), disabled: busy })] }), _jsxs("div", { className: "mts-field", children: [_jsxs("label", { className: "mts-label", children: ["\u98CE\u683C\u8F6C\u6362\u5F3A\u5EA6\uFF1A", coverStrength.toFixed(2)] }), _jsx("input", { type: "range", min: 0.3, max: 1, step: 0.05, value: coverStrength, onChange: (e) => setCoverStrength(Number(e.target.value)), disabled: busy }), _jsx("div", { className: "mts-hint", children: "1.0 = \u5B8C\u5168\u6309\u65B0\u98CE\u683C\u91CD\u6F14\u7ECE\uFF1B\u8C03\u4F4E\u5219\u4FDD\u7559\u66F4\u591A\u539F\u66F2\u97F3\u8272\u7EC6\u8282\u3002\u8F93\u51FA\u4E3A 48kHz WAV\u3002" })] })] })), tab === "repaint" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "mts-field mts-repaint-range", children: [_jsxs("div", { children: [_jsx("label", { className: "mts-label", children: "\u8D77\u70B9\uFF08\u79D2\uFF09" }), _jsx("input", { className: "mts-input", value: repaintStart, onChange: (e) => setRepaintStart(e.target.value.replace(/[^0-9.]/g, "")), disabled: busy })] }), _jsxs("div", { children: [_jsx("label", { className: "mts-label", children: "\u7EC8\u70B9\uFF08\u79D2\uFF09" }), _jsx("input", { className: "mts-input", value: repaintEnd, onChange: (e) => setRepaintEnd(e.target.value.replace(/[^0-9.]/g, "")), disabled: busy })] })] }), _jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u533A\u95F4\u65B0\u5185\u5BB9\u63CF\u8FF0" }), _jsx("textarea", { className: "mts-textarea", rows: 2, placeholder: "\u4F8B\u5982\uFF1Adrum fill buildup into the chorus", value: repaintPrompt, onChange: (e) => setRepaintPrompt(e.target.value), disabled: busy })] }), _jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u533A\u95F4\u6B4C\u8BCD\uFF08\u53EF\u9009\uFF0C\u542B\u4EBA\u58F0\u65F6\u586B\u5199\uFF09" }), _jsx("textarea", { className: "mts-textarea", rows: 2, value: repaintLyrics, onChange: (e) => setRepaintLyrics(e.target.value), disabled: busy })] }), _jsx("div", { className: "mts-hint", children: "\u533A\u95F4\u5916\u97F3\u9891\u4FDD\u6301\u539F\u6837\uFF1B\u6E32\u67D3\u65F6\u53EA\u5BF9 [\u8D77\u70B9, \u7EC8\u70B9) \u505A\u6269\u6563\u91CD\u7ED8\uFF0C\u901F\u5EA6\u4E0E\u533A\u95F4\u957F\u5EA6\u6210\u6B63\u6BD4\u3002" })] })), tab === "yue2score" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "mts-mode-row", children: [_jsx("button", { className: `mts-mode-btn ${yue2Mode === "instrument" ? "active" : ""}`, onClick: () => { setYue2Mode("instrument"); setError(null); setResult(null); }, disabled: busy, children: "\u65B0\u4E50\u5668\u8F68\u9053" }), _jsx("button", { className: `mts-mode-btn ${yue2Mode === "continue" ? "active" : ""}`, onClick: () => { setYue2Mode("continue"); setError(null); setResult(null); }, disabled: busy, children: "\u4E50\u8C31\u7EED\u5199" })] }), yue2Mode === "instrument" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u4E50\u5668\u7C7B\u578B" }), _jsx("select", { className: "mts-select", value: yue2Instrument, onChange: (e) => setYue2Instrument(e.target.value), disabled: busy, children: YUE2_INSTRUMENTS.map((i) => (_jsxs("option", { value: i.value, children: [i.zh, "\uFF08", i.value, "\uFF09"] }, i.value))) })] }), _jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u98CE\u683C\u8865\u5145\u63CF\u8FF0\uFF08\u53EF\u9009\uFF0C\u9ED8\u8BA4\u6CBF\u7528\u6E90\u6B4C\u66F2\u98CE\u683C\u4E0E BPM\uFF09" }), _jsx("input", { className: "mts-input", value: yue2StyleText, onChange: (e) => setYue2StyleText(e.target.value), placeholder: "\u4F8B\u5982\uFF1Awarm acoustic ballad", disabled: busy })] }), _jsx("div", { className: "mts-hint", children: "\u6280\u672F\u65B9\u6848\uFF1AYuE-2 \u65E0\u72EC\u7ACB\u5206\u8F68\u6A21\u5757\uFF0C\u6B64\u5904\u6309\u6E90\u6B4C\u66F2 BPM/\u98CE\u683C\u751F\u6210\u5355\u4E50\u5668\u72EC\u594F\uFF08\u540C\u6B65\u4F9D\u8D56 BPM \u63D0\u793A\u5BF9\u9F50\uFF09\uFF0C \u540C\u65F6\u4EA7\u51FA ABC \u4E50\u8C31\u4E0E MIDI\uFF1B\u751F\u6210\u540E\u300C\u53D1\u9001\u5230\u8F68\u9053\u300D\u5373\u53EF\u4E0E\u539F\u66F2\u53E0\u653E\uFF0C\u97F3\u91CF\u5728\u8F68\u9053\u63A8\u5B50\u4E0A\u5E73\u8861\u3002" })] })), yue2Mode === "continue" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u4ECE\u5386\u53F2\u6B4C\u66F2\u52A0\u8F7D ABC\uFF08\u4EC5 YuE-2 \u751F\u6210\u7684\u6B4C\u66F2\uFF09" }), _jsxs("div", { className: "mts-src-row", children: [_jsxs("select", { className: "mts-select", value: yue2SongUuid, onChange: (e) => setYue2SongUuid(e.target.value), disabled: busy, children: [_jsx("option", { value: "", children: "\uFF08\u9009\u62E9\u6B4C\u66F2\uFF09" }), songs.filter((s) => s.abcPath).map((s) => (_jsx("option", { value: s.uuid, children: s.label || "Untitled" }, s.uuid)))] }), _jsx("button", { className: "mts-btn mts-btn-ghost", onClick: loadAbcFromSong, disabled: busy || !yue2SongUuid, children: "\u52A0\u8F7D" })] })] }), _jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "ABC \u4E50\u8C31\uFF08\u53EF\u76F4\u63A5\u7C98\u8D34\u7F16\u8F91\uFF09" }), _jsx("textarea", { className: "mts-textarea mts-mono", rows: 6, placeholder: "X:1\nM:4/4\nL:1/8\nK:C\n|:C2D2E2G2:|", value: yue2Abc, onChange: (e) => setYue2Abc(e.target.value), disabled: busy })] }), _jsxs("div", { className: "mts-field", children: [_jsx("label", { className: "mts-label", children: "\u7EED\u5199\u65B9\u5411\u63CF\u8FF0" }), _jsx("input", { className: "mts-input", value: yue2StyleText, onChange: (e) => setYue2StyleText(e.target.value), placeholder: "\u4F8B\u5982\uFF1A\u4EE5\u540C\u6837\u7684\u98CE\u683C\u7EE7\u7EED\u53D1\u5C55\uFF0C\u52A0\u5165\u526F\u6B4C\u63A8\u8FDB", disabled: busy })] }), _jsx("div", { className: "mts-hint", children: "\u7EED\u5199\u903B\u8F91\uFF1A\u628A\u5DF2\u6709 ABC \u4F5C\u4E3A\u7ED3\u6784\u6761\u4EF6\u4F20\u7ED9 YuE-2 \u7684 CoT \u89C4\u5212\u5C42\uFF0C\u6A21\u578B\u5728\u4E50\u8C31\u9AA8\u67B6\u4E0A\u5EF6\u5C55\u5E76\u91CD\u65B0\u6E32\u67D3\u97F3\u9891\uFF0C \u540C\u65F6\u8F93\u51FA\u65B0\u7684 ABC + MIDI\uFF08\u4E0E ACE-Step \u7684\u97F3\u9891\u8865\u5168\u4E0D\u540C\uFF0C\u8FD9\u662F\u4E50\u8C31\u5C42\u7684\u7EED\u5199\uFF09\u3002" })] }))] })), _jsx("button", { className: "mts-btn mts-btn-primary", onClick: handleGenerate, disabled: busy, children: busy ? "生成中…请耐心等待" :
                                tab === "lego" ? "生成叠加轨道" :
                                    tab === "extract" ? "开始分离" :
                                        tab === "vocal2bgm" ? "生成伴奏" :
                                            tab === "cover" ? "开始翻唱" :
                                                tab === "repaint" ? "重绘区间" :
                                                    "生成乐谱轨道" }), busy && (_jsxs("div", { className: "mts-hint", children: [activeModel.desc, "\uFF1B\u9996\u6B21\u52A0\u8F7D\u6A21\u578B\u9700\u8981\u66F4\u957F\u65F6\u95F4\uFF0C\u8FDB\u5EA6\u8BE6\u60C5\u89C1\u63A7\u5236\u53F0\u65E5\u5FD7\u3002"] })), error && _jsx("div", { className: "mts-error", children: error }), result && resultAudio && (_jsxs("div", { className: "mts-result", children: [_jsxs("div", { className: "mts-result-title", children: ["\u2705 \u751F\u6210\u5B8C\u6210\uFF1A", result.label] }), _jsx("audio", { controls: true, src: convertFileSrc(resultAudio), className: "mts-audio" }), (resultMidi || result.outputs.find((o) => o.abc_path)?.abc_path) && (_jsxs("div", { className: "mts-hint", children: [resultMidi ? "已附带 MIDI 文件；" : "", result.outputs.find((o) => o.abc_path)?.abc_path ? "已附带 ABC 乐谱文件；" : "", "\u4FDD\u5B58\u5230\u6B4C\u66F2\u5217\u8868\u540E\u53EF\u5728\u6B4C\u66F2\u8BE6\u60C5\u4E2D\u4F7F\u7528\u3002"] })), _jsxs("div", { className: "mts-result-actions", children: [_jsx("button", { className: "mts-btn mts-btn-primary", onClick: () => onGenerated(result), children: "\u4FDD\u5B58\u5230\u6B4C\u66F2\u5217\u8868" }), _jsx("button", { className: "mts-btn mts-btn-ghost", onClick: () => onSendToTrack(resultAudio, result.label, result.durationSec), children: "\u53D1\u9001\u5230\u8F68\u9053" }), _jsx("button", { className: "mts-btn mts-btn-ghost", onClick: () => setResult(null), children: "\u5173\u95ED" })] })] }))] })] }) }));
}
