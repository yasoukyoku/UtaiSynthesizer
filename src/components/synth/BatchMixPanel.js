import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
// BatchMixPanel — S9 多选批量「统一音量/声像」：对包含所选片段的每条轨道统一写入相同的
// 音量(dB) 与声像(pan)。复用 TempoStretchPanel 的浮层样式；应用走既有的 updateTrack 历史可见
// 写入 + updateTrackVolume/updateTrackPan 播放链路，保证与单轨推子行为一致。
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import * as playback from "../../lib/audio/playback";
import { ParamSlider } from "../workflow/nodes/ParamSlider";
import "./TempoStretchPanel.css";
const VOL_MIN = -24;
const VOL_MAX = 12;
const PAN_MIN = -1;
const PAN_MAX = 1;
export function BatchMixPanel({ x, y, trackIds, onClose }) {
    const { t } = useTranslation();
    const tracks = useProjectStore((s) => s.tracks);
    const [volumeDb, setVolumeDb] = useState(0);
    const [pan, setPan] = useState(0);
    const users = tracks.filter((tr) => trackIds.includes(tr.id));
    const canApply = users.length > 0;
    const apply = () => {
        for (const tr of users) {
            useProjectStore.getState().updateTrack(tr.id, { volumeDb, pan });
            playback.updateTrackVolume(tr.id, volumeDb);
            playback.updateTrackPan(tr.id, pan);
        }
        onClose();
    };
    const pw = 240;
    const px = Math.max(4, Math.min(x, window.innerWidth - pw - 8));
    const py = Math.max(4, Math.min(y, window.innerHeight - 190));
    return (_jsxs(_Fragment, { children: [_jsx("div", { className: "stretch-backdrop", onMouseDown: onClose, onContextMenu: (e) => { e.preventDefault(); onClose(); } }), _jsxs("div", { className: "stretch-panel", style: { left: px, top: py, width: pw }, onMouseDown: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "stretch-title", children: [t("menu.batchMixTitle"), " \u00B7 ", users.length] }), _jsx("div", { className: "stretch-readout", children: _jsxs("span", { children: [t("menu.batchMixTargets"), ": ", users.map((u) => u.name).join(", ") || "—"] }) }), _jsx(ParamSlider, { label: t("menu.resetVolume"), min: VOL_MIN, max: VOL_MAX, step: 0.5, value: volumeDb, onChange: setVolumeDb, format: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB` }), _jsx(ParamSlider, { label: t("menu.pan"), min: PAN_MIN, max: PAN_MAX, step: 0.05, value: pan, onChange: setPan, format: (v) => (v === 0 ? "C" : v < 0 ? `L${Math.round(-v * 100)}` : `R${Math.round(v * 100)}`) }), _jsx("div", { className: "stretch-actions", children: _jsx("button", { className: "stretch-btn", disabled: !canApply, onClick: apply, children: t("menu.apply") }) })] })] }));
}
