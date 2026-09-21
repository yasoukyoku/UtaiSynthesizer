import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * CreativeAssistant - 创作助手（简化入口）
 *
 * 三个高频创作流，统一到一个"少滚动、参数分主次"的面板里（规划 A/B/C/D 简化）：
 *   B. 单乐器叠加 — 选源歌曲 + 选乐器 → 生成并叠加（ACE-Step lego）
 *   C. 乐谱续写   — 载入历史 ABC → 续写方向 → 生成多个候选 → 试听/采用并继续（YuE-2）
 *   D. 翻唱助手   — 源歌曲 → 分离人声 → 改歌词 → 选风格 → 生成（ACE-Step cover）
 *
 * 复用后端既有任务管线（songGenerate），不新增模型/命令；
 * 生成结果统一走 onGenerated（存入歌曲列表）/ onSendToTrack（落到工程轨道）。
 */
import { useEffect, useMemo, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { songGenerate, listSongModels } from "../../lib/backendSong";
import "./CreativeAssistant.css";
const BASE_MODEL = "acestep-v1.5";
const YUE2_MODEL = "yue2-3b";
const BASE_FILES_PREFIX = "acestep-v1.5/checkpoints/acestep-v15-base/";
const YUE2_FILES_PREFIX = "yue2-3b/";
const TABS = [
    { id: "instrument", title: "🎸 叠加乐器", desc: "给现有歌曲加一条乐器轨（鼓/贝斯/吉他…），模型自动对齐原曲调式与 BPM" },
    { id: "score", title: "🎼 乐谱续写", desc: "载入历史 ABC 乐谱，按你的方向生成多个延展候选，试听后采用并继续写下去" },
    { id: "cover", title: "🎤 翻唱助手", desc: "一键流程：分离人声 → 改歌词 → 选风格 → 生成，自动沿用原曲 BPM 与曲式" },
];
/** 常用乐器（点击即选），value 为 ACE-Step 轨道名 */
const INSTRUMENTS = [
    { value: "drums", zh: "🥁 鼓组" },
    { value: "bass", zh: "🎸 贝斯" },
    { value: "guitar", zh: "🎸 吉他" },
    { value: "keyboard", zh: "🎹 键盘" },
    { value: "strings", zh: "🎻 弦乐" },
    { value: "synth", zh: "🎛 合成器" },
    { value: "brass", zh: "🎺 铜管" },
    { value: "woodwinds", zh: "🎷 木管" },
    { value: "percussion", zh: "🪘 打击乐" },
    { value: "fx", zh: "✨ 特效" },
    { value: "backing_vocals", zh: "🎤 和声" },
    { value: "vocals", zh: "🎤 人声" },
];
/** 翻唱风格预设（点击追加到描述） */
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
function baseNameOf(p) {
    return p.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "src";
}
export function CreativeAssistant({ songs, initialTab, onClose, onGenerated, onSendToTrack, onOpenMidi, }) {
    const [tab, setTab] = useState(initialTab ?? "instrument");
    const audioOptions = useMemo(() => songs.filter((s) => s.audioPath), [songs]);
    const abcOptions = useMemo(() => songs.filter((s) => s.abcPath), [songs]);
    const [srcPath, setSrcPath] = useState(audioOptions[0]?.audioPath ?? "");
    const srcSong = audioOptions.find((s) => s.audioPath === srcPath);
    // B：单乐器叠加
    const [instrument, setInstrument] = useState("drums");
    const [instStyle, setInstStyle] = useState("");
    // C：乐谱续写
    const [abcSongUuid, setAbcSongUuid] = useState("");
    const [abcText, setAbcText] = useState("");
    const [contStyle, setContStyle] = useState("");
    const [candidateCount, setCandidateCount] = useState(2);
    const [candidates, setCandidates] = useState([]);
    // D：翻唱助手
    const [vocalPath, setVocalPath] = useState(null);
    const [useVocalSource, setUseVocalSource] = useState(true);
    const [coverLyrics, setCoverLyrics] = useState("");
    const [coverPrompt, setCoverPrompt] = useState("");
    const [coverStrength, setCoverStrength] = useState(0.8);
    const [busy, setBusy] = useState(false);
    const [progressText, setProgressText] = useState(null);
    const [error, setError] = useState(null);
    const [status, setStatus] = useState(null);
    const [result, setResult] = useState(null);
    const [modelInstalled, setModelInstalled] = useState(null);
    useEffect(() => {
        let cancelled = false;
        listSongModels()
            .then((files) => {
            if (cancelled)
                return;
            setModelInstalled({
                base: files.some((f) => f.filename.startsWith(BASE_FILES_PREFIX)),
                yue2: files.some((f) => f.filename.startsWith(YUE2_FILES_PREFIX)),
            });
        })
            .catch(() => { if (!cancelled)
            setModelInstalled({ base: true, yue2: true }); });
        return () => { cancelled = true; };
    }, []);
    // 切换源歌曲时，用原曲歌词预填翻唱歌词（"自动记忆"原曲内容）
    useEffect(() => {
        if (srcSong?.lyrics && !coverLyrics.trim())
            setCoverLyrics(srcSong.lyrics);
        setVocalPath(null);
        // 仅在切换源歌曲时同步
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [srcPath]);
    function switchTab(t) {
        if (busy)
            return;
        setTab(t);
        setError(null);
        setStatus(null);
        setResult(null);
    }
    function resetOutputs() {
        setError(null);
        setStatus(null);
        setResult(null);
        setCandidates([]);
    }
    async function loadAbcFromSong(uuid) {
        const song = songs.find((s) => s.uuid === uuid);
        if (!song?.abcPath) {
            setError("所选歌曲没有 ABC 乐谱文件（仅 YuE-2 生成的歌曲带有 .abc）");
            return;
        }
        try {
            const text = await readTextFile(song.abcPath);
            setAbcText(text);
            setError(null);
            setStatus(`已载入乐谱：${song.label || "Untitled"}`);
        }
        catch (e) {
            setError(`读取 ABC 文件失败：${String(e)}`);
        }
    }
    // ── B：单乐器叠加 ────────────────────────────────────────────────────────
    async function generateInstrument() {
        if (!srcPath) {
            setError("请先选择源歌曲（或到歌曲列表生成一首）");
            return;
        }
        setBusy(true);
        resetOutputs();
        try {
            const label = `${baseNameOf(srcPath)}_lego_${instrument}`;
            const outputs = await songGenerate({
                model: BASE_MODEL,
                task: "lego",
                src_audio_path: srcPath,
                track_name: instrument,
                prompt: instStyle.trim(),
                lyrics: "",
                song_name: label,
                audio_duration: 0,
                format: "wav",
                output_dir: "",
            });
            if (!outputs.find((o) => o.audio_path)?.audio_path)
                throw new Error("生成完成但没有返回音频文件");
            setResult({ outputs, label, durationSec: srcSong?.durationSec ?? 60, modelId: BASE_MODEL, wantMidi: false });
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }
    // ── C：乐谱续写（多候选）─────────────────────────────────────────────────
    async function generateCandidates() {
        if (!abcText.trim()) {
            setError("请先加载或粘贴要续写的 ABC 乐谱");
            return;
        }
        setBusy(true);
        resetOutputs();
        const list = [];
        try {
            for (let i = 0; i < candidateCount; i++) {
                setProgressText(`候选 ${i + 1}/${candidateCount} 生成中…`);
                const label = `abc_continue_${Date.now().toString(36)}_${i + 1}`;
                const outputs = await songGenerate({
                    model: YUE2_MODEL,
                    task: "text2music",
                    song_name: label,
                    prompt: contStyle.trim() || "continue the score in the same style, seamless extension",
                    lyrics: "[Instrumental]",
                    audio_duration: 0,
                    format: "wav",
                    output_dir: "",
                    want_midi: true,
                    abc: abcText,
                });
                const audioPath = outputs.find((o) => o.audio_path)?.audio_path;
                if (!audioPath)
                    throw new Error("生成完成但没有返回音频文件");
                list.push({
                    id: crypto.randomUUID(),
                    label,
                    outputs,
                    audioPath,
                    abcPath: outputs.find((o) => o.abc_path)?.abc_path,
                    midiPath: outputs.find((o) => o.midi_path)?.midi_path,
                    durationSec: 60,
                });
                setCandidates([...list]);
            }
            setStatus(`已生成 ${list.length} 个候选，试听后「采用并继续」`);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
            setProgressText(null);
        }
    }
    /** 采用某候选的乐谱作为新的续写起点 → 形成"继续写"的循环 */
    async function adoptCandidate(c) {
        if (!c.abcPath) {
            setError("该候选没有附带 ABC 乐谱文件");
            return;
        }
        try {
            const text = await readTextFile(c.abcPath);
            setAbcText(text);
            setCandidates([]);
            setStatus("已采用该候选乐谱作为新的续写起点，可继续生成下一段");
            setError(null);
        }
        catch (e) {
            setError(`读取候选乐谱失败：${String(e)}`);
        }
    }
    // ── D：翻唱助手 ──────────────────────────────────────────────────────────
    async function extractVocals() {
        if (!srcPath) {
            setError("请先选择源歌曲");
            return;
        }
        setBusy(true);
        setError(null);
        setStatus(null);
        setVocalPath(null);
        try {
            const label = `${baseNameOf(srcPath)}_vocals`;
            const outputs = await songGenerate({
                model: BASE_MODEL,
                task: "extract",
                src_audio_path: srcPath,
                track_name: "vocals",
                prompt: "",
                lyrics: "",
                song_name: label,
                audio_duration: 0,
                format: "wav",
                output_dir: "",
            });
            const vp = outputs.find((o) => o.audio_path)?.audio_path;
            if (!vp)
                throw new Error("分离完成但没有返回人声文件");
            setVocalPath(vp);
            setStatus(`人声已分离：${vp.split(/[\\/]/).pop()}`);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }
    async function generateCover() {
        const coverSrc = useVocalSource && vocalPath ? vocalPath : srcPath;
        if (!coverSrc) {
            setError("请先选择源歌曲（或先分离人声）");
            return;
        }
        setBusy(true);
        resetOutputs();
        try {
            const label = `${baseNameOf(srcPath)}_cover`;
            const outputs = await songGenerate({
                model: BASE_MODEL,
                task: "cover",
                src_audio_path: coverSrc,
                prompt: coverPrompt.trim(),
                lyrics: coverLyrics,
                song_name: label,
                audio_duration: 0,
                format: "wav",
                output_dir: "",
                audio_cover_strength: coverStrength,
            });
            if (!outputs.find((o) => o.audio_path)?.audio_path)
                throw new Error("生成完成但没有返回音频文件");
            setResult({ outputs, label, durationSec: srcSong?.durationSec ?? 60, modelId: BASE_MODEL, wantMidi: false });
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }
    const activeTab = TABS.find((t) => t.id === tab) ?? TABS[0];
    const resultAudio = result?.outputs.find((o) => o.audio_path)?.audio_path;
    const resultMidi = result?.outputs.find((o) => o.midi_path)?.midi_path;
    const modelWarning = (tab === "score" && modelInstalled && !modelInstalled.yue2)
        ? "YuE-2 3B 模型尚未下载，请先到「资源管理 → 生成歌曲」下载。"
        : (tab !== "score" && modelInstalled && !modelInstalled.base)
            ? "ACE-Step V1.5 (Base) 模型尚未下载（约 4.8GB），请先到「资源管理 → 生成歌曲」下载。"
            : null;
    return (_jsx("div", { className: "ca-overlay", onClick: onClose, children: _jsxs("div", { className: "ca-dialog", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "ca-header", children: [_jsx("div", { className: "ca-title", children: "\uD83E\uDE84 \u521B\u4F5C\u52A9\u624B" }), _jsx("div", { className: "ca-subtitle", children: "\u628A\u5E38\u7528\u7684\u4E09\u6B65\u521B\u4F5C\uFF0C\u6536\u8FDB\u4E00\u4E2A\u9762\u677F\u91CC" }), _jsx("button", { className: "ca-close", onClick: onClose, children: "\u2715" })] }), _jsx("div", { className: "ca-tabs", children: TABS.map((t) => (_jsx("button", { className: `ca-tab ${tab === t.id ? "active" : ""}`, onClick: () => switchTab(t.id), disabled: busy, children: t.title }, t.id))) }), _jsx("div", { className: "ca-tab-desc", children: activeTab.desc }), modelWarning && _jsxs("div", { className: "ca-warn", children: ["\u26A0 ", modelWarning] }), _jsxs("div", { className: "ca-body", children: [tab === "instrument" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "ca-field", children: [_jsx("label", { className: "ca-label", children: "\u6E90\u6B4C\u66F2" }), _jsxs("select", { className: "ca-select", value: srcPath, onChange: (e) => setSrcPath(e.target.value), disabled: busy, children: [audioOptions.length === 0 && _jsx("option", { value: "", children: "\uFF08\u6B4C\u66F2\u5217\u8868\u6682\u65E0\u97F3\u9891\uFF09" }), audioOptions.map((s) => (_jsxs("option", { value: s.audioPath, children: [s.label || "Untitled", "\uFF08", s.durationSec, "s", s.bpm ? ` · ${s.bpm}BPM` : "", "\uFF09"] }, s.uuid)))] })] }), _jsxs("div", { className: "ca-field", children: [_jsx("label", { className: "ca-label", children: "\u6DFB\u52A0\u4E50\u5668" }), _jsx("div", { className: "ca-chips", children: INSTRUMENTS.map((i) => (_jsx("button", { className: `ca-chip ${instrument === i.value ? "active" : ""}`, onClick: () => setInstrument(i.value), disabled: busy, children: i.zh }, i.value))) })] }), _jsxs("details", { className: "ca-advanced", children: [_jsx("summary", { children: "\u9AD8\u7EA7\uFF1A\u98CE\u683C\u8865\u5145\uFF08\u53EF\u9009\uFF09" }), _jsx("input", { className: "ca-input", value: instStyle, onChange: (e) => setInstStyle(e.target.value), placeholder: "\u4F8B\u5982\uFF1Awarm jazz brush drums, laid back", disabled: busy })] }), _jsx("button", { className: "ca-btn ca-btn-primary", onClick: generateInstrument, disabled: busy || !srcPath, children: busy ? "生成中…请耐心等待" : "生成并叠加" })] })), tab === "score" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "ca-field", children: [_jsx("label", { className: "ca-label", children: "\u4ECE\u5386\u53F2\u6B4C\u66F2\u52A0\u8F7D\u4E50\u8C31" }), _jsxs("div", { className: "ca-row", children: [_jsxs("select", { className: "ca-select", value: abcSongUuid, onChange: (e) => setAbcSongUuid(e.target.value), disabled: busy, children: [_jsx("option", { value: "", children: "\uFF08\u9009\u62E9\u5E26 ABC \u7684\u6B4C\u66F2\uFF09" }), abcOptions.map((s) => (_jsx("option", { value: s.uuid, children: s.label || "Untitled" }, s.uuid)))] }), _jsx("button", { className: "ca-btn ca-btn-ghost", onClick: () => loadAbcFromSong(abcSongUuid), disabled: busy || !abcSongUuid, children: "\u8F7D\u5165" })] }), abcOptions.length === 0 && (_jsx("div", { className: "ca-hint", children: "\u6682\u65E0\u5E26 ABC \u7684\u6B4C\u66F2\uFF1A\u5148\u7528 YuE-2 \u751F\u6210\u4E00\u9996\u4F1A\u9644\u5E26 .abc \u4E50\u8C31\u3002" }))] }), _jsxs("div", { className: "ca-field", children: [_jsx("label", { className: "ca-label", children: "ABC \u4E50\u8C31\uFF08\u53EF\u76F4\u63A5\u7C98\u8D34\u7F16\u8F91\uFF09" }), _jsx("textarea", { className: "ca-textarea ca-mono", rows: 5, placeholder: "X:1\nM:4/4\nL:1/8\nK:C\n|:C2D2E2G2:|", value: abcText, onChange: (e) => setAbcText(e.target.value), disabled: busy })] }), _jsxs("div", { className: "ca-row", children: [_jsxs("div", { className: "ca-field ca-grow", children: [_jsx("label", { className: "ca-label", children: "\u7EED\u5199\u65B9\u5411" }), _jsx("input", { className: "ca-input", value: contStyle, onChange: (e) => setContStyle(e.target.value), placeholder: "\u4F8B\u5982\uFF1A\u4EE5\u540C\u6837\u98CE\u683C\u7EE7\u7EED\u53D1\u5C55\uFF0C\u52A0\u5165\u526F\u6B4C\u63A8\u8FDB", disabled: busy })] }), _jsxs("div", { className: "ca-field ca-field-narrow", children: [_jsx("label", { className: "ca-label", children: "\u5019\u9009\u6570\u91CF" }), _jsxs("select", { className: "ca-select", value: candidateCount, onChange: (e) => setCandidateCount(Number(e.target.value)), disabled: busy, children: [_jsx("option", { value: 1, children: "1 \u4E2A" }), _jsx("option", { value: 2, children: "2 \u4E2A" }), _jsx("option", { value: 3, children: "3 \u4E2A" })] })] })] }), _jsx("button", { className: "ca-btn ca-btn-primary", onClick: generateCandidates, disabled: busy || !abcText.trim(), children: busy ? (progressText ?? "生成中…") : "生成候选" }), candidates.length > 0 && (_jsx("div", { className: "ca-candidates", children: candidates.map((c, idx) => (_jsxs("div", { className: "ca-candidate", children: [_jsxs("div", { className: "ca-candidate-head", children: ["\u5019\u9009 ", idx + 1] }), c.audioPath && _jsx("audio", { controls: true, src: convertFileSrc(c.audioPath), className: "ca-audio" }), _jsxs("div", { className: "ca-candidate-actions", children: [_jsx("button", { className: "ca-btn ca-btn-primary ca-btn-sm", onClick: () => adoptCandidate(c), children: "\u2713 \u91C7\u7528\u5E76\u7EE7\u7EED" }), _jsx("button", { className: "ca-btn ca-btn-ghost ca-btn-sm", onClick: () => onSendToTrack(c.audioPath, c.label, c.durationSec), children: "\u2795 \u53D1\u9001\u5230\u8F68\u9053" }), c.midiPath && onOpenMidi && (_jsx("button", { className: "ca-btn ca-btn-ghost ca-btn-sm", onClick: () => onOpenMidi(c.midiPath, c.label), children: "\uD83C\uDFB9 MIDI \u7F16\u8F91" }))] })] }, c.id))) }))] })), tab === "cover" && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "ca-field", children: [_jsx("label", { className: "ca-label", children: "\u6E90\u6B4C\u66F2" }), _jsxs("select", { className: "ca-select", value: srcPath, onChange: (e) => setSrcPath(e.target.value), disabled: busy, children: [audioOptions.length === 0 && _jsx("option", { value: "", children: "\uFF08\u6B4C\u66F2\u5217\u8868\u6682\u65E0\u97F3\u9891\uFF09" }), audioOptions.map((s) => (_jsxs("option", { value: s.audioPath, children: [s.label || "Untitled", "\uFF08", s.durationSec, "s", s.bpm ? ` · ${s.bpm}BPM` : "", "\uFF09"] }, s.uuid)))] }), srcSong?.bpm && (_jsxs("div", { className: "ca-hint", children: ["\u5DF2\u81EA\u52A8\u8BB0\u5FC6\u539F\u66F2 BPM\uFF1A", srcSong.bpm, "\uFF0C\u7FFB\u5531\u5C06\u6CBF\u7528\u8BE5\u901F\u5EA6\u4E0E\u66F2\u5F0F\u7ED3\u6784\u3002"] }))] }), _jsxs("div", { className: "ca-step", children: [_jsx("span", { className: "ca-step-no", children: "\u2460" }), _jsxs("div", { className: "ca-step-body", children: [_jsx("button", { className: "ca-btn ca-btn-ghost", onClick: extractVocals, disabled: busy || !srcPath, children: vocalPath ? "重新分离人声" : "分离人声" }), _jsxs("label", { className: "ca-check", children: [_jsx("input", { type: "checkbox", checked: useVocalSource && !!vocalPath, onChange: (e) => setUseVocalSource(e.target.checked), disabled: busy || !vocalPath }), "\u7528\u5206\u79BB\u51FA\u7684\u4EBA\u58F0\u4F5C\u4E3A\u7FFB\u5531\u6E90\uFF08\u63A8\u8350\uFF09"] })] })] }), _jsxs("div", { className: "ca-step", children: [_jsx("span", { className: "ca-step-no", children: "\u2461" }), _jsxs("div", { className: "ca-step-body ca-grow", children: [_jsx("label", { className: "ca-label", children: "\u6B4C\u8BCD\uFF08\u53EF\u4FEE\u6539\uFF09" }), _jsx("textarea", { className: "ca-textarea", rows: 4, value: coverLyrics, onChange: (e) => setCoverLyrics(e.target.value), placeholder: "\u7559\u7A7A\u5219\u751F\u6210\u7EAF\u97F3\u4E50\uFF1B\u53EF\u7C98\u8D34\u6216\u6539\u5199\u539F\u66F2\u6B4C\u8BCD", disabled: busy })] })] }), _jsxs("div", { className: "ca-step", children: [_jsx("span", { className: "ca-step-no", children: "\u2462" }), _jsxs("div", { className: "ca-step-body ca-grow", children: [_jsx("label", { className: "ca-label", children: "\u76EE\u6807\u98CE\u683C\uFF08\u70B9\u51FB\u8FFD\u52A0\uFF09" }), _jsx("div", { className: "ca-chips", children: COVER_STYLE_PRESETS.map((p) => (_jsx("button", { className: "ca-chip", onClick: () => setCoverPrompt((v) => (v.includes(p.tag) ? v : `${v ? v + ", " : ""}${p.tag}`)), disabled: busy, children: p.label }, p.tag))) }), _jsx("textarea", { className: "ca-textarea", rows: 2, value: coverPrompt, onChange: (e) => setCoverPrompt(e.target.value), placeholder: "\u4F8B\u5982\uFF1Aenergetic rock, distorted guitar, powerful drums", disabled: busy })] })] }), _jsxs("details", { className: "ca-advanced", children: [_jsxs("summary", { children: ["\u9AD8\u7EA7\uFF1A\u98CE\u683C\u8F6C\u6362\u5F3A\u5EA6 ", coverStrength.toFixed(2)] }), _jsx("input", { type: "range", min: 0.3, max: 1, step: 0.05, value: coverStrength, onChange: (e) => setCoverStrength(Number(e.target.value)), disabled: busy }), _jsx("div", { className: "ca-hint", children: "1.0 = \u5B8C\u5168\u6309\u65B0\u98CE\u683C\u91CD\u6F14\u7ECE\uFF1B\u8C03\u4F4E\u4FDD\u7559\u66F4\u591A\u539F\u66F2\u97F3\u8272\u3002" })] }), _jsxs("div", { className: "ca-step", children: [_jsx("span", { className: "ca-step-no", children: "\u2463" }), _jsx("div", { className: "ca-step-body", children: _jsx("button", { className: "ca-btn ca-btn-primary", onClick: generateCover, disabled: busy || !srcPath, children: busy ? "生成中…请耐心等待" : "生成翻唱" }) })] })] })), error && _jsx("div", { className: "ca-error", children: error }), status && _jsx("div", { className: "ca-status", children: status }), result && resultAudio && (_jsxs("div", { className: "ca-result", children: [_jsxs("div", { className: "ca-result-title", children: ["\u2705 \u751F\u6210\u5B8C\u6210\uFF1A", result.label] }), _jsx("audio", { controls: true, src: convertFileSrc(resultAudio), className: "ca-audio" }), _jsxs("div", { className: "ca-candidate-actions", children: [_jsx("button", { className: "ca-btn ca-btn-primary ca-btn-sm", onClick: () => onGenerated(result), children: "\u4FDD\u5B58\u5230\u6B4C\u66F2\u5217\u8868" }), _jsx("button", { className: "ca-btn ca-btn-ghost ca-btn-sm", onClick: () => onSendToTrack(resultAudio, result.label, result.durationSec), children: "\u53D1\u9001\u5230\u8F68\u9053" }), resultMidi && onOpenMidi && (_jsx("button", { className: "ca-btn ca-btn-ghost ca-btn-sm", onClick: () => onOpenMidi(resultMidi, result.label), children: "\uD83C\uDFB9 MIDI \u7F16\u8F91" }))] })] }))] })] }) }));
}
