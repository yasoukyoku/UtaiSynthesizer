import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * 发送到轨道对话框 - 产物驱动（规划 8.1/8.2）
 * 可用性由实际产物决定：audio_path / stems / midi_path|abc_path
 * 无分轨产物时提供"⚡ 现场分轨后发送"兜底（ACE 高质量 / demucs 快速）
 * P1-12：落点选项（工程末尾/当前播放位置/指定轨道后）+ MIDI 双按钮（直接落轨 / 工作台）
 */
import { useState } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { deriveCapabilities } from "../../lib/song/types";
import { ACE_TRACK_CLASSES } from "../../lib/models/song-tasks";
import i18n from "../../i18n";
import "./SendToTrackDialog.css";
export function SendToTrackDialog({ song, onConfirm, onClose }) {
    const [selected, setSelected] = useState("single");
    const [extractEngine, setExtractEngine] = useState("demucs");
    // 规划 8.2：落点选项（默认工程末尾；"指定轨道后"默认当前激活轨）
    const [destKind, setDestKind] = useState("end");
    const activeTrackId = useAppStore((s) => s.activeTrackId);
    const [afterTrackId, setAfterTrackId] = useState(activeTrackId ?? null);
    const tracks = useProjectStore((s) => s.tracks);
    const dest = destKind === "afterTrack" && afterTrackId
        ? { kind: "afterTrack", trackId: afterTrackId }
        : destKind === "playhead"
            ? { kind: "playhead" }
            : { kind: "end" };
    const caps = song.capabilities ?? deriveCapabilities({ outputs: song.outputs });
    const hasAudio = song.outputs.some((o) => o.audio_path);
    const hasMidi = caps.hasMidi || caps.hasAbc;
    const stemKinds = Object.keys(song.outputs.find((o) => o.stems)?.stems ?? {});
    const allCards = [
        {
            id: "single",
            icon: "🎵",
            label: "单轨音频",
            description: "完整混音的单个音频文件",
            supported: hasAudio,
            badge: hasAudio ? undefined : "无音频产物",
        },
        {
            id: "multi",
            icon: "🎚️",
            label: "多轨音频",
            description: stemKinds.length
                ? `已含分轨：${stemKinds.join(" / ")}`
                : "分离的人声、伴奏、鼓等音轨",
            supported: caps.hasStems,
            badge: caps.hasStems ? undefined : "无分轨产物",
        },
        {
            id: "liveExtract",
            icon: "⚡",
            label: i18n.t("songResult.stemsOnTheFly"),
            description: "对主音频即时分轨，每个 stem 各建一条轨道",
            supported: hasAudio && !caps.hasStems,
            badge: caps.hasStems ? "已有分轨，可直接选多轨" : undefined,
        },
        {
            id: "midiTrack",
            icon: "🎹",
            label: "MIDI",
            description: caps.hasMidi
                ? "直接落为可编辑的 MIDI 音符轨"
                : caps.hasAbc
                    ? "仅有 ABC 乐谱，落轨前自动转换为 MIDI"
                    : i18n.t("songResult.noMidiHint"),
            supported: hasMidi,
            badge: hasMidi ? undefined : "不含 MIDI 产物",
        },
    ];
    // 已有分轨时不再展示"现场分轨"卡（8.1：多轨可选即不需要兜底）
    const cards = allCards.filter((c) => c.id !== "liveExtract" || !caps.hasStems);
    const current = cards.find((c) => c.id === selected) ?? cards[0];
    const fmt = (sec) => `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, "0")}`;
    // 预览清单（规划 8.2）：确认前一目了然将创建哪些轨道
    const preview = (() => {
        const label = song.label || "Untitled";
        const dur = `${fmt(song.settings.audio_duration)} · WAV`;
        switch (current.id) {
            case "single":
                return [{ name: label, type: "音频轨", note: dur }];
            case "multi":
                return stemKinds.map((k) => ({ name: `${label} - ${k}`, type: "音频轨", note: dur }));
            case "liveExtract":
                return (extractEngine === "demucs"
                    ? ["vocals", "drums", "bass", "other"]
                    : [...ACE_TRACK_CLASSES]).map((k) => ({ name: `${label} - ${k}`, type: "音频轨", note: "分轨完成后创建" }));
            case "midi":
            case "midiTrack":
                return [
                    {
                        name: `${label} (MIDI)`,
                        type: "MIDI 音符轨",
                        note: caps.hasMidi ? undefined : "ABC→MIDI 自动转换",
                    },
                ];
        }
    })();
    const destOptions = [
        { id: "end", label: "工程末尾" },
        { id: "playhead", label: "当前播放位置" },
        { id: "afterTrack", label: "指定轨道后" },
    ];
    return (_jsx("div", { className: "st-overlay", onClick: onClose, children: _jsxs("div", { className: "st-dialog", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "st-header", children: [_jsxs("div", { className: "st-title", children: [_jsx("span", { className: "st-title-icon", children: "\uD83D\uDCE4" }), i18n.t("songResult.sendToTrack")] }), _jsx("button", { className: "st-close", onClick: onClose, children: "\u2715" })] }), _jsxs("div", { className: "st-body", children: [_jsxs("div", { className: "st-song-info", children: [_jsx("div", { className: "st-song-icon", children: "\uD83C\uDFB5" }), _jsxs("div", { className: "st-song-details", children: [_jsx("div", { className: "st-song-title", children: song.label || "Untitled" }), _jsxs("div", { className: "st-song-meta", children: [song.modelFamily.toUpperCase(), " \u00B7 ", song.settings.audio_duration, "s \u00B7 ", song.settings.bpm, " BPM"] })] })] }), _jsxs("div", { className: "st-formats", children: [_jsx("div", { className: "st-formats-label", children: "\u9009\u62E9\u53D1\u9001\u5185\u5BB9" }), _jsx("div", { className: "st-formats-list", children: cards.map((card) => (_jsxs("div", { className: `st-format-card ${!card.supported ? "disabled" : ""} ${current.id === card.id ? "selected" : ""}`, title: card.supported ? undefined : card.badge, onClick: () => card.supported && setSelected(card.id), children: [_jsx("div", { className: "st-format-icon", children: card.icon }), _jsxs("div", { className: "st-format-info", children: [_jsxs("div", { className: "st-format-label", children: [card.label, !card.supported && card.badge && (_jsx("span", { className: "st-format-badge", children: card.badge }))] }), _jsx("div", { className: "st-format-desc", children: card.description })] }), card.supported && (_jsx("div", { className: "st-format-radio", children: current.id === card.id ? "●" : "○" }))] }, card.id))) })] }), current.id === "liveExtract" && (_jsxs("div", { className: "st-engine-row", children: [_jsx("button", { className: `st-engine-chip ${extractEngine === "demucs" ? "on" : ""}`, onClick: () => setExtractEngine("demucs"), children: "\u26A1 demucs \u5FEB\u901F\u56DB\u8F68" }), _jsx("button", { className: `st-engine-chip ${extractEngine === "ace" ? "on" : ""}`, onClick: () => setExtractEngine("ace"), children: "\uD83C\uDF9A ACE \u9AD8\u8D28\u91CF\u5168\u8F68" })] })), _jsxs("div", { className: "st-dest", children: [_jsx("div", { className: "st-formats-label", children: "\u843D\u70B9" }), _jsx("div", { className: "st-dest-row", children: destOptions.map((o) => (_jsx("button", { className: `st-engine-chip ${destKind === o.id ? "on" : ""}`, onClick: () => setDestKind(o.id), children: o.label }, o.id))) }), destKind === "afterTrack" && (_jsxs("select", { className: "st-dest-select", value: afterTrackId ?? "", onChange: (e) => setAfterTrackId(e.target.value), children: [tracks.length === 0 && _jsx("option", { value: "", children: "\uFF08\u5DE5\u7A0B\u4E2D\u6682\u65E0\u8F68\u9053\uFF09" }), tracks.map((t) => (_jsx("option", { value: t.id, children: t.name }, t.id)))] }))] }), _jsxs("div", { className: "st-preview", children: [_jsxs("div", { className: "st-preview-title", children: ["\u5C06\u521B\u5EFA ", preview.length, " \u6761\u8F68\u9053"] }), preview.map((p, i) => (_jsxs("div", { className: "st-preview-row", children: [_jsxs("span", { className: "st-preview-name", children: ["\uD83C\uDFB5 ", p.name] }), _jsx("span", { className: "st-preview-type", children: p.type }), p.note && _jsx("span", { className: "st-preview-note", children: p.note })] }, i)))] })] }), _jsxs("div", { className: "st-footer", children: [_jsx("button", { className: "st-btn st-btn-cancel", onClick: onClose, children: "\u53D6\u6D88" }), current.id === "midiTrack" && (_jsx("button", { className: "st-btn st-btn-secondary", onClick: () => onConfirm("midi", extractEngine, dest), children: "\u5728 MIDI \u5DE5\u4F5C\u53F0\u67E5\u770B/\u7F16\u8F91" })), _jsx("button", { className: "st-btn st-btn-confirm", onClick: () => onConfirm(current.id === "midiTrack" ? "midiTrack" : current.id, extractEngine, dest), disabled: !current.supported, children: "\u786E\u8BA4\u53D1\u9001" })] })] }) }));
}
