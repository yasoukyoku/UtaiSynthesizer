import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import { runChordMidi } from "../../lib/arrangement/autoArrange";
import { CHORD_MIDI_STYLES } from "../../lib/arrangement/chordMidi";
import "./ArrangeDialog.css";
/** 调性选项：auto + 12 大调 + 12 小调（升号记法，音乐记号无需翻译）。 */
const KEY_OPTIONS = [
    "auto",
    "C", "Cm", "C#", "C#m", "D", "Dm", "D#", "D#m", "E", "Em", "F", "Fm",
    "F#", "F#m", "G", "Gm", "G#", "G#m", "A", "Am", "A#", "A#m", "B", "Bm",
];
/**
 * Muno 阶段5「生成和弦 MIDI（配和声）」弹窗 —— 选和弦风格/每小节和弦数/调性 →
 * 生成 1 条和弦和声轨 + 顶部和弦行显示。目标 = 打开它的那条旋律轨。
 * 复用 ArrangeDialog 的 .arrange-* 样式（同一族弹窗，保持极简一致）。
 */
export function ChordMidiDialog({ trackId, onClose }) {
    const { t } = useTranslation();
    const trackName = useProjectStore((s) => s.tracks.find((tr) => tr.id === trackId)?.name) ?? "";
    const [style, setStyle] = useState("POP_STANDARD");
    const [chordsPerBar, setChordsPerBar] = useState(1);
    const [key, setKey] = useState("auto");
    const [busy, setBusy] = useState(false);
    useEffect(() => {
        const onKey = (e) => {
            if (e.key === "Escape" && !busy)
                onClose();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [busy, onClose]);
    const run = async () => {
        setBusy(true);
        try {
            await runChordMidi(trackId, { style, chordsPerBar, key });
            onClose();
        }
        finally {
            setBusy(false);
        }
    };
    return (_jsx("div", { className: "arrange-overlay", onClick: busy ? undefined : onClose, children: _jsxs("div", { className: "arrange-panel", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "arrange-head", children: [_jsxs("span", { className: "arrange-title", children: ["\uD83C\uDFB9 ", t("chordMidi.title")] }), _jsx("button", { className: "arrange-close", disabled: busy, onClick: onClose, children: "\u2715" })] }), _jsxs("div", { className: "arrange-body", children: [_jsxs("div", { className: "arrange-target", title: trackName, children: [t("chordMidi.target"), ": ", _jsx("strong", { children: trackName || "—" })] }), _jsxs("label", { className: "arrange-field", children: [_jsx("span", { children: t("chordMidi.style") }), _jsx("select", { value: style, disabled: busy, onChange: (e) => setStyle(e.target.value), children: CHORD_MIDI_STYLES.map((s) => (_jsx("option", { value: s, children: t(`chordMidi.styles.${s}`) }, s))) })] }), _jsxs("label", { className: "arrange-field", children: [_jsx("span", { children: t("chordMidi.chordsPerBar") }), _jsxs("select", { value: chordsPerBar, disabled: busy, onChange: (e) => setChordsPerBar(Number(e.target.value) === 2 ? 2 : 1), children: [_jsx("option", { value: 1, children: t("chordMidi.perBar1") }), _jsx("option", { value: 2, children: t("chordMidi.perBar2") })] })] }), _jsxs("label", { className: "arrange-field", children: [_jsx("span", { children: t("chordMidi.key") }), _jsx("select", { value: key, disabled: busy, onChange: (e) => setKey(e.target.value), children: KEY_OPTIONS.map((k) => (_jsx("option", { value: k, children: k === "auto" ? t("chordMidi.keyAuto") : k }, k))) })] }), _jsx("div", { className: "arrange-hint", children: t("chordMidi.hint") })] }), _jsxs("div", { className: "arrange-foot", children: [_jsx("button", { className: "arrange-btn", disabled: busy, onClick: onClose, children: t("chordMidi.cancel") }), _jsx("button", { className: "arrange-btn primary", disabled: busy, onClick: () => void run(), children: busy ? t("chordMidi.running") : t("chordMidi.run") })] })] }) }));
}
