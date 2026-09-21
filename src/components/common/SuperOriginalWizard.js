import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { useHistoryStore } from "../../store/history";
import { useMsstModelStore } from "../../store/msst-models";
import { useVoiceModelStore } from "../../store/voice-models";
import { useAmtModelStore } from "../../store/amt-models";
import { MSST_CATALOG, t18 } from "../../lib/models/msst-catalog";
import { AUDIO_EXTENSIONS } from "../../lib/constants";
import { importAudioToNewTrack } from "../../lib/audio/import";
import { flushAutosaveNow } from "../../lib/project/autosave";
import { buildWorkflow, templateNeedsAmt, templateNeedsSeparation, templateNeedsVoiceModel, } from "../../lib/workflow/templates";
import "./SuperOriginalWizard.css";
const TEMPLATE_CARDS = [
    {
        id: "transcribe",
        icon: "🎼",
        title: { zh: "一键扒带", en: "One-Click Transcribe", ja: "ワンクリック採譜" },
        desc: {
            zh: "整首歌转成多乐器 MIDI（钢琴/吉他/贝斯/鼓…），之后可换音源、修复、任意改编。",
            en: "Turn the whole song into multi-instrument MIDI, then swap sounds, repair, and rearrange freely.",
            ja: "楽曲全体をマルチ楽器 MIDI へ。音源交換・修復・アレンジ自由。",
        },
    },
    {
        id: "clone",
        icon: "🎤",
        title: { zh: "声音复刻", en: "Voice Clone", ja: "ボイス復刻" },
        desc: {
            zh: "分离人声与伴奏，人声用你的声音模型重新演唱，伴奏原样保留。",
            en: "Separate vocals from backing, re-sing the vocals with your voice model, keep the backing as-is.",
            ja: "ボーカルと伴奏を分離し、ボーカルをあなたの声モデルで歌い直す。",
        },
    },
    {
        id: "original",
        icon: "✨",
        title: { zh: "深度原创", en: "Deep Original", ja: "完全オリジナル" },
        desc: {
            zh: "复刻 + 伴奏移调 + 同步扒带 —— 声音换了、调性变了、音源可换，改完就是你的原创。",
            en: "Clone + transposed backing + parallel transcription — new voice, new key, swappable sounds.",
            ja: "復刻＋伴奏移調＋並行採譜。声も調性も新しくオリジナルに。",
        },
    },
    // 规划 12.2：原创化两条技术路线
    {
        id: "rebuild",
        icon: "🏗️",
        title: { zh: "原创重建（推荐发布）", en: "Original Rebuild", ja: "オリジナル再構築" },
        desc: {
            zh: "MIDI 重建路线：整曲转谱、人声换声——伴奏不留原波形，换音源重演奏，原创度最高。",
            en: "MIDI rebuild: full transcription + new voice — no original waveform kept; highest originality.",
            ja: "MIDI再構築：全曲採譜＋声交換。原波形を残さずオリジナル度最高。",
        },
    },
    {
        id: "repaint",
        icon: "🖌️",
        title: { zh: "快速重绘", en: "Quick Repaint", ja: "クイック再描画" },
        desc: {
            zh: "音频重绘路线：换声 + 移调 + 变速一步到位，快速出 Demo/灵感稿。",
            en: "Audio repaint: new voice + transposed + retimed backing in one pass — fast demo drafts.",
            ja: "音声再描画：声交換＋移調＋テンポ変更で即デモ化。",
        },
    },
    // 规划 6-6:歌曲制作四模板(P2 歌曲节点族,不走 MSST 分离、不需要 AMT/声音模型)
    {
        id: "lyrics2song",
        icon: "📝",
        title: { zh: "词曲成歌", en: "Lyrics → Song", ja: "歌詞から作曲" },
        desc: {
            zh: "输入歌词和风格描述，一键生成整首歌（人声 + 伴奏），无需任何演唱录制。",
            en: "Enter lyrics and a style prompt — generate a full song (vocals + backing) in one pass.",
            ja: "歌詞とスタイルを入力し、フルソング（ボーカル+伴奏）を自動生成。",
        },
    },
    {
        id: "vocal2acc",
        icon: "🎵",
        title: { zh: "人声转伴奏", en: "Vocal → Accompaniment", ja: "ボーカルから伴奏" },
        desc: {
            zh: "上传清唱人声，自动补全伴奏与各声部，干声秒变完整编曲。",
            en: "Feed in a dry vocal — auto-complete the backing and harmonies into a full arrangement.",
            ja: "ボーカルのみの音源から伴奏・ハーモニーを自動補完。",
        },
    },
    {
        id: "stemsRebuild",
        icon: "🎚️",
        title: { zh: "分轨重建", en: "Stems Rebuild", ja: "分離トラック再構築" },
        desc: {
            zh: "把整首歌拆成人声/鼓/贝斯/其他四轨，各自落轨后自由重混、替换、再创作。",
            en: "Split a song into vocals/drums/bass/other stems — remix, replace, and rework freely.",
            ja: "楽曲をボーカル/ドラム/ベース/その他に分離し自由に再ミックス。",
        },
    },
    {
        id: "segRepaint",
        icon: "🎨",
        title: { zh: "片段重绘", en: "Segment Repaint", ja: "セグメント再描画" },
        desc: {
            zh: "框选歌曲片段 + 新风格提示词，只重绘这一段（改词、换曲风、修瑕疵）。",
            en: "Pick a segment + a new style prompt — repaint just that part (new lyrics, genre, fixes).",
            ja: "区間とスタイルを指定し、その部分だけ再生成（歌詞・曲風の差し替え）。",
        },
    },
];
/** 原创化强度（rebuild/repaint）：变调半音 + 变速系数（规划 12.2 温和/标准/激进）。 */
const INTENSITY_PRESETS = [
    { id: "mild", label: { zh: "温和", en: "Mild", ja: "穏健" }, semitones: 1, speed: 1.0 },
    { id: "standard", label: { zh: "标准", en: "Standard", ja: "標準" }, semitones: 2, speed: 1.03 },
    { id: "aggressive", label: { zh: "激进", en: "Aggressive", ja: "強め" }, semitones: 3, speed: 1.06 },
];
export function SuperOriginalWizard({ onClose }) {
    const { t, i18n } = useTranslation();
    const lang = i18n.language;
    const [audioPath, setAudioPath] = useState(null);
    const [template, setTemplate] = useState("original");
    const [voiceName, setVoiceName] = useState("");
    const [semitones, setSemitones] = useState(2);
    const [intensity, setIntensity] = useState("standard");
    const [busy, setBusy] = useState(false);
    const msstInstalled = useMsstModelStore((s) => s.installed);
    const voiceModels = useVoiceModelStore((s) => s.models.rvc);
    const amtInstalled = useAmtModelStore((s) => s.installed);
    const toggleModelManager = useAppStore((s) => s.toggleModelManager);
    // 拉取三个模型库的最新状态（可能本会话还没被任何页面触发过）。
    useEffect(() => {
        void useMsstModelStore.getState().fetchInstalled();
        void useVoiceModelStore.getState().fetchModels();
        void useAmtModelStore.getState().fetchInstalled();
    }, []);
    // Esc 关闭（忙时不关）——与 ExportAudioDialog 相同的捕获模式。
    useEffect(() => {
        const onKey = (e) => {
            e.stopPropagation();
            if (e.key === "Escape" && !busy)
                onClose();
        };
        window.addEventListener("keydown", onKey, true);
        return () => window.removeEventListener("keydown", onKey, true);
    }, [busy, onClose]);
    /** 最佳人声分离模型：已安装的 vocals 类里按 SDR 分数挑第一。 */
    const sepModel = useMemo(() => {
        const files = new Set(msstInstalled.map((m) => m.filename));
        const cands = MSST_CATALOG.filter((e) => e.category === "vocals" && files.has(e.filename));
        if (cands.length === 0)
            return null;
        cands.sort((a, b) => (b.sdrScore ?? -1) - (a.sdrScore ?? -1));
        const cat = cands[0];
        const inst = msstInstalled.find((m) => m.filename === cat.filename);
        const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
        const stems = inst?.stem_names?.length
            ? [...inst.stem_names, ...(inst.residual_name ? [inst.residual_name] : [])].map(cap)
            : cat.stems;
        return { filename: cat.filename, displayName: t18(cat.name, lang), stems };
    }, [msstInstalled, lang]);
    /** 最佳转谱后端：MuScriptor > MIROS > YourMT3+（按已安装可用性降级）。 */
    const amtBackend = useMemo(() => {
        const has = (id) => amtInstalled.some((m) => m.id === id && m.is_available);
        if (has("muscriptor_large") || has("muscriptor_small"))
            return "muscriptor";
        if (has("mc13_256_all_cross_v6"))
            return "miros";
        if (has("yptf_moe_multi_nops"))
            return "yourmt3";
        return null;
    }, [amtInstalled]);
    // 声音模型默认选第一个（列表刷新/删除后保持合法选择）。
    useEffect(() => {
        if (voiceModels.length > 0 && !voiceModels.some((m) => m.name === voiceName)) {
            setVoiceName(voiceModels[0].name);
        }
    }, [voiceModels, voiceName]);
    const needsVoice = templateNeedsVoiceModel(template);
    const needsSep = templateNeedsSeparation(template);
    const missingPieces = [];
    // 规划 6-6:只有图里真含 amtMidi 的模板才检查转谱后端 —— 歌曲模板(词曲成歌/分轨重建等)
    // 和纯换声模板(复刻/快速重绘)不再被 AMT 缺失误拦。
    if (templateNeedsAmt(template) && !amtBackend)
        missingPieces.push(t("wizard.missingAmt"));
    if (needsSep && !sepModel)
        missingPieces.push(t("wizard.missingSep"));
    if (needsVoice && voiceModels.length === 0)
        missingPieces.push(t("wizard.missingVoice"));
    const canStart = !!audioPath && missingPieces.length === 0 && !busy && (!needsVoice || !!voiceName);
    const pickAudio = async () => {
        const path = await open({
            multiple: false,
            title: t("wizard.pickSong"),
            filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
        });
        if (typeof path === "string" && path)
            setAudioPath(path);
    };
    const start = async () => {
        if (!canStart || !audioPath)
            return;
        setBusy(true);
        try {
            // 1) 建轨导入（tick 0，与「+」导入一致；解码/失败提示由 import.ts 全权负责）。
            const { trackId, segId } = await importAudioToNewTrack(audioPath, 0);
            const track = useProjectStore.getState().tracks.find((tr) => tr.id === trackId);
            if (!track || !track.segments.some((s) => s.id === segId))
                return; // 解码失败已被 toast，占位已清
            // 2) 模板工作流写进新片段（系统级设置，静默入历史 —— 与导入的 loading→loaded 同规则）。
            const vm = voiceModels.find((m) => m.name === voiceName);
            // rebuild/repaint 走强度预设（变调+变速）；其余模板沿用移调选项
            const preset = INTENSITY_PRESETS.find((p) => p.id === intensity);
            const usePreset = template === "rebuild" || template === "repaint";
            const wf = buildWorkflow(template, {
                vocalStemNames: sepModel?.stems,
                separationModelFile: sepModel?.filename,
                voiceModel: vm ? { name: vm.name, path: vm.path } : undefined,
                transposeSemitones: usePreset ? preset.semitones : semitones,
                speedFactor: template === "repaint" ? preset.speed : undefined,
                amtBackend: amtBackend ?? "muscriptor",
            });
            useHistoryStore.getState().runSilent(() => useProjectStore.getState().updateTrack(trackId, {
                segments: track.segments.map((s) => (s.id === segId ? { ...s, workflow: wf } : s)),
            }));
            flushAutosaveNow(); // 建轨+建图是一个里程碑，立即落盘
            // 3) 打开工作流编辑器并请求挂载即运行（进度/取消/缺失组件弹窗全在编辑器侧）。
            useAppStore.getState().openWorkflow(segId);
            useAppStore.setState({ workflowAutoRun: segId });
            onClose();
        }
        finally {
            setBusy(false);
        }
    };
    const fileName = audioPath ? audioPath.split(/[/\\]/).pop() : "";
    return (_jsx("div", { className: "confirm-overlay sow-overlay", onMouseDown: (e) => { if (e.target === e.currentTarget && !busy)
            onClose(); }, children: _jsxs("div", { className: "confirm-dialog sow-dialog", children: [_jsx("div", { className: "confirm-title", children: t("wizard.title") }), _jsx("div", { className: "sow-subtitle", children: t("wizard.subtitle") }), _jsxs("section", { className: "sow-step", children: [_jsxs("div", { className: "sow-step-head", children: [_jsx("span", { className: "sow-step-num", children: "1" }), _jsx("span", { children: t("wizard.stepSong") })] }), _jsxs("button", { type: "button", className: `sow-file-btn${audioPath ? " has-file" : ""}`, onClick: () => void pickAudio(), children: [_jsx("span", { className: "sow-file-icon", children: audioPath ? "🎵" : "📂" }), _jsx("span", { className: "sow-file-text", children: audioPath ? fileName : t("wizard.pickSongHint") })] })] }), _jsxs("section", { className: "sow-step", children: [_jsxs("div", { className: "sow-step-head", children: [_jsx("span", { className: "sow-step-num", children: "2" }), _jsx("span", { children: t("wizard.stepTemplate") })] }), _jsx("div", { className: "sow-cards", children: TEMPLATE_CARDS.map((c) => (_jsxs("button", { type: "button", className: `sow-card${template === c.id ? " active" : ""}`, onClick: () => setTemplate(c.id), children: [_jsx("span", { className: "sow-card-icon", children: c.icon }), _jsx("span", { className: "sow-card-title", children: t18(c.title, lang) }), _jsx("span", { className: "sow-card-desc", children: t18(c.desc, lang) })] }, c.id))) }), needsVoice && (_jsxs("div", { className: "sow-opt", children: [_jsx("label", { className: "sow-opt-label", children: t("wizard.voiceModel") }), _jsx("select", { className: "sow-select", value: voiceName, onChange: (e) => setVoiceName(e.target.value), disabled: voiceModels.length === 0, children: voiceModels.length === 0 ? (_jsx("option", { value: "", children: t("wizard.noVoiceModel") })) : (voiceModels.map((m) => (_jsx("option", { value: m.name, children: m.name }, m.name)))) })] })), template === "original" && (_jsxs("div", { className: "sow-opt", children: [_jsx("label", { className: "sow-opt-label", children: t("wizard.transpose") }), _jsx("div", { className: "sow-semi-row", children: [-3, -2, -1, 1, 2, 3].map((s) => (_jsx("button", { type: "button", className: `sow-semi${semitones === s ? " active" : ""}`, onClick: () => setSemitones(s), children: s > 0 ? `+${s}` : s }, s))) }), _jsx("div", { className: "sow-opt-hint", children: t("wizard.transposeHint") })] })), (template === "rebuild" || template === "repaint") && (_jsxs("div", { className: "sow-opt", children: [_jsx("label", { className: "sow-opt-label", children: t("wizard.intensity") }), _jsx("div", { className: "sow-semi-row", children: INTENSITY_PRESETS.map((p) => (_jsx("button", { type: "button", className: `sow-semi${intensity === p.id ? " active" : ""}`, onClick: () => setIntensity(p.id), children: t18(p.label, lang) }, p.id))) }), _jsx("div", { className: "sow-opt-hint", children: t("wizard.intensityHint") })] }))] }), missingPieces.length > 0 ? (_jsxs("div", { className: "sow-missing", children: [_jsx("div", { className: "sow-missing-title", children: t("wizard.missingTitle") }), _jsx("ul", { children: missingPieces.map((m) => _jsx("li", { children: m }, m)) }), _jsx("button", { type: "button", className: "sow-link-btn", onClick: toggleModelManager, children: t("wizard.openModelManager") })] })) : (_jsxs("div", { className: "sow-ready", children: ["\u2713 ", t("wizard.ready", { model: sepModel && needsSep ? sepModel.displayName : "" })] })), _jsxs("div", { className: "sow-actions", children: [_jsx("button", { type: "button", className: "sow-btn", onClick: onClose, disabled: busy, children: t("common.cancel") }), _jsx("button", { type: "button", className: "sow-btn primary", onClick: () => void start(), disabled: !canStart, children: busy ? t("wizard.starting") : `🚀 ${t("wizard.start")}` })] }), _jsx("div", { className: "sow-footnote", children: t("wizard.footnote") })] }) }));
}
