import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { useVoiceModelStore } from "../../store/voice-models";
/**
 * Quick-start wizard: paste lyrics + pick voice -> create vocal track.
 * Melody notes still go in Piano Roll; G2P + auto-render fire after save.
 */
export function LyricToVocalWizard({ onClose }) {
    const showToast = useAppStore((s) => s.showToast);
    const addTrack = useProjectStore((s) => s.addTrack);
    const setTempo = useProjectStore((s) => s.setTempo);
    const setTimeSignature = useProjectStore((s) => s.setTimeSignature);
    const voiceModels = useVoiceModelStore((s) => {
        const all = [];
        for (const k of Object.keys(s.models)) {
            all.push(...s.models[k]);
        }
        return all;
    });
    const [lyrics, setLyrics] = useState("");
    const [bpm, setBpm] = useState(100);
    const [voiceModel, setVoiceModel] = useState("");
    const [busy, setBusy] = useState(false);
    const run = async () => {
        if (!lyrics.trim()) {
            showToast("Please paste some lyrics", "error");
            return;
        }
        setBusy(true);
        try {
            setTempo(bpm);
            setTimeSignature(4, 4);
            const modelName = voiceModel || voiceModels[0]?.name;
            if (!modelName) {
                showToast("No AI voices installed - go to Models menu", "error");
                onClose();
                return;
            }
            addTrack({
                type: "vocal",
                name: "AI Vocal (" + modelName + ")",
                singerId: modelName,
            });
            showToast("Track created! Open Piano Roll (P) to write melody, then RENDER.", "success");
            onClose();
        }
        catch (e) {
            showToast(String(e ?? "Unknown error"), "error");
        }
        finally {
            setBusy(false);
        }
    };
    return (_jsx("div", { className: "wizard-backdrop", onClick: onClose, children: _jsxs("div", { className: "wizard-card", onClick: (e) => e.stopPropagation(), children: [_jsx("h2", { children: "Lyrics to AI Vocal Track" }), _jsx("p", { className: "hint", children: "Quick start: paste lyrics, pick voice, we create a vocal track. Melody notes go in Piano Roll." }), _jsxs("div", { className: "field", children: [_jsx("label", { children: "Lyrics (one line per sentence)" }), _jsx("textarea", { rows: 5, value: lyrics, onChange: (e) => setLyrics(e.target.value), placeholder: "Paste song lyrics here..." })] }), _jsxs("div", { className: "field-row", children: [_jsxs("div", { className: "field", children: [_jsx("label", { children: "BPM (speed)" }), _jsx("input", { type: "number", value: bpm, min: 40, max: 200, onChange: (e) => setBpm(+e.target.value) })] }), _jsxs("div", { className: "field", style: { flex: 2 }, children: [_jsx("label", { children: "AI Voice" }), _jsxs("select", { value: voiceModel, onChange: (e) => setVoiceModel(e.target.value), children: [_jsx("option", { value: "", children: "-- Auto-pick first --" }), voiceModels.map((v) => (_jsx("option", { value: v.name, children: v.name }, v.name)))] }), voiceModels.length === 0 && (_jsx("p", { className: "hint", children: "No voices yet. Download from Models menu." }))] })] }), _jsxs("div", { className: "wizard-actions", children: [_jsx("button", { onClick: onClose, children: "Cancel" }), _jsx("button", { className: "primary", onClick: run, disabled: busy, children: busy ? "Creating..." : "Create Vocal Track" })] })] }) }));
}
