import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// 规划 12.4：原创度自检面板 —— 纯前端指标 + 现有后端检测（analyze_segment_tempo）。
// 输出：BPM 差异、原创化清单勾选、综合参考分、可导出 originality-report.json 归档。
// 免责：只提供技术指标，不构成法律意见。
import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import "./OriginalityPanel.css";
const DEFAULT_CHECKLIST = {
    voiceReplaced: false,
    lyricsRewritten: false,
    arrangementChanged: false,
    soundSourcesNew: false,
    waveformRetained: false,
};
async function detectBpm(path) {
    try {
        const res = await invoke("analyze_segment_tempo", {
            path,
            windowStartMs: 0,
            windowEndMs: 600000,
            beatsPerBar: 4,
        });
        return res.bpm;
    }
    catch {
        return null;
    }
}
export function OriginalityPanel({ sourcePath, resultPath, onClose, }) {
    const { t } = useTranslation();
    const [srcBpm, setSrcBpm] = useState(null);
    const [outBpm, setOutBpm] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [checklist, setChecklist] = useState(DEFAULT_CHECKLIST);
    useEffect(() => {
        if (!sourcePath && !resultPath)
            return;
        setAnalyzing(true);
        void (async () => {
            const [a, b] = await Promise.all([
                sourcePath ? detectBpm(sourcePath) : Promise.resolve(null),
                resultPath ? detectBpm(resultPath) : Promise.resolve(null),
            ]);
            setSrcBpm(a);
            setOutBpm(b);
            setAnalyzing(false);
        })();
    }, [sourcePath, resultPath]);
    /** 参考分：BPM 差异（0~20）+ 清单勾选项（4×15）- 原波形保留惩罚（20） */
    const score = useMemo(() => {
        let s = 0;
        if (srcBpm != null && outBpm != null && srcBpm > 0) {
            const diff = Math.min(20, Math.round((Math.abs(outBpm - srcBpm) / srcBpm) * 100));
            s += diff;
        }
        const { waveformRetained, ...flags } = checklist;
        s += Object.values(flags).filter(Boolean).length * 15;
        if (waveformRetained)
            s = Math.max(0, s - 20);
        return Math.min(100, s);
    }, [srcBpm, outBpm, checklist]);
    const items = [
        { key: "voiceReplaced", label: t("songOriginality.voiceReplaced") },
        { key: "lyricsRewritten", label: t("songOriginality.lyricsRewritten") },
        { key: "arrangementChanged", label: t("songOriginality.arrangementChanged") },
        { key: "soundSourcesNew", label: t("songOriginality.soundSourcesNew") },
        { key: "waveformRetained", label: t("songOriginality.waveformRetained") },
    ];
    const exportReport = async () => {
        const path = await save({
            title: t("songOriginality.export"),
            defaultPath: "originality-report.json",
            filters: [{ name: "JSON", extensions: ["json"] }],
        });
        if (!path)
            return;
        const report = {
            generatedAt: new Date().toISOString(),
            disclaimer: t("songOriginality.disclaimer"),
            bpm: { source: srcBpm, result: outBpm },
            checklist,
            referenceScore: score,
            paths: { source: sourcePath, result: resultPath },
        };
        const { writeTextFile } = await import("@tauri-apps/plugin-fs");
        await writeTextFile(path, JSON.stringify(report, null, 2));
    };
    return (_jsx("div", { className: "confirm-overlay", onMouseDown: (e) => { if (e.target === e.currentTarget)
            onClose(); }, children: _jsxs("div", { className: "confirm-dialog orig-dialog", children: [_jsx("div", { className: "confirm-title", children: t("songOriginality.title") }), _jsxs("div", { className: "orig-bpm-row", children: [_jsxs("div", { className: "orig-bpm-card", children: [_jsx("span", { className: "orig-bpm-label", children: t("songOriginality.source") }), _jsx("span", { className: "orig-bpm-value", children: analyzing ? "…" : srcBpm != null ? `${srcBpm} BPM` : "—" })] }), _jsxs("div", { className: "orig-bpm-card", children: [_jsx("span", { className: "orig-bpm-label", children: t("songOriginality.result") }), _jsx("span", { className: "orig-bpm-value", children: analyzing ? "…" : outBpm != null ? `${outBpm} BPM` : "—" })] }), srcBpm != null && outBpm != null && srcBpm > 0 && (_jsxs("div", { className: "orig-bpm-card diff", children: [_jsx("span", { className: "orig-bpm-label", children: t("songOriginality.bpmDiff") }), _jsxs("span", { className: "orig-bpm-value", children: [Math.round((Math.abs(outBpm - srcBpm) / srcBpm) * 100), "%"] })] }))] }), _jsx("div", { className: "orig-checklist", children: items.map((it) => (_jsxs("label", { className: "orig-check", children: [_jsx("input", { type: "checkbox", checked: checklist[it.key], onChange: (e) => setChecklist((c) => ({ ...c, [it.key]: e.target.checked })) }), _jsx("span", { children: it.label })] }, it.key))) }), _jsxs("div", { className: "orig-score", children: [_jsx("span", { children: t("songOriginality.score") }), _jsx("strong", { className: score >= 60 ? "good" : "low", children: score }), _jsx("span", { children: "/ 100" })] }), _jsx("div", { className: "orig-disclaimer", children: t("songOriginality.disclaimer") }), _jsxs("div", { className: "orig-actions", children: [_jsx("button", { type: "button", className: "sow-btn", onClick: onClose, children: t("common.close") }), _jsx("button", { type: "button", className: "sow-btn primary", onClick: () => void exportReport(), children: t("songOriginality.export") })] })] }) }));
}
