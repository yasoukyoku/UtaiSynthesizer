import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from "react";
import { useProjectStore } from "../../store/project";
import { useHistoryStore } from "../../store/history";
import { VolumeFader, formatDb, formatPan } from "../common/VolumeFader";
import { OutLevelMeter } from "../common/OutLevelMeter";
import { updateTrackVolume, updateTrackPan, updateTrackAudibility, getContext } from "../../lib/audio/playback";
import { getMasterVolumeDb, setMasterVolume } from "../../lib/audio/effectsBus";
import { loadSetting, saveSetting } from "../../lib/settings";
import { FADER_MIN_DB, FADER_MAX_DB } from "../../lib/constants";
import { trackTypeCssVar } from "../../lib/trackColors";
import "./MixerConsole.css";
/** dB 读数 — 点击变成输入框, 可手动输入精确值 (Enter 确认 / Esc 取消 / 失焦提交) */
function DbReadout({ value, min, max, onCommit, title }) {
    const [editing, setEditing] = useState(false);
    const [text, setText] = useState("");
    if (!editing) {
        return (_jsx("span", { className: "mixer-strip-db mixer-strip-db-click", title: title, onClick: () => { setText(String(Math.round(value * 10) / 10)); setEditing(true); }, children: formatDb(value, min) }));
    }
    const commit = () => {
        const v = parseFloat(text);
        if (!Number.isNaN(v)) {
            onCommit(Math.max(min, Math.min(max, Math.round(v * 10) / 10)));
        }
        setEditing(false);
    };
    return (_jsx("input", { className: "mixer-strip-db-input", type: "number", step: 0.1, min: min, max: max, autoFocus: true, value: text, onChange: (e) => setText(e.target.value), onBlur: commit, onKeyDown: (e) => {
            if (e.key === "Enter")
                commit();
            if (e.key === "Escape")
                setEditing(false);
        }, title: "\u8F93\u5165 dB \u503C, Enter \u786E\u8BA4, Esc \u53D6\u6D88" }));
}
/** 🎛 控制台 — 全轨道混音台 (Studio Pro Mixer 风格).
 *  底部停靠面板 (非弹窗): 打开后可与页面上的轨道/播放实时交互.
 *  左侧: 每条轨一条通道 (竖直推子 + 分轨电平表 + 静音/独奏 + 平衡);
 *  最右侧: 总输出条 (主音量推子 + 实时电平). */
export function MixerConsole({ onClose }) {
    const tracks = useProjectStore((s) => s.tracks);
    const updateTrack = useProjectStore((s) => s.updateTrack);
    // 主输出音量 (dB) — 持久化, 打开控制台时恢复
    const [masterDb, setMasterDb] = useState(() => getMasterVolumeDb());
    useEffect(() => {
        const db = loadSetting("utai.masterVolumeDb", 0);
        setMasterVolume(getContext(), db);
        setMasterDb(db);
    }, []);
    const setVol = (id, v) => {
        updateTrack(id, { volumeDb: v });
        updateTrackVolume(id, v);
    };
    const setPan = (id, v) => {
        updateTrack(id, { pan: v });
        updateTrackPan(id, v);
    };
    const toggleMute = (id, muted) => {
        useHistoryStore.getState().beginTransaction();
        updateTrack(id, { muted: !muted });
        useHistoryStore.getState().commitTransaction();
        updateTrackAudibility(useProjectStore.getState().tracks);
    };
    const toggleSolo = (id, solo) => {
        useHistoryStore.getState().beginTransaction();
        updateTrack(id, { solo: !solo });
        useHistoryStore.getState().commitTransaction();
        updateTrackAudibility(useProjectStore.getState().tracks);
    };
    const setMaster = (v) => {
        setMasterVolume(getContext(), v);
        saveSetting("utai.masterVolumeDb", v);
        setMasterDb(v);
    };
    return (_jsxs("div", { className: "mixer-dock", role: "region", "aria-label": "\u63A7\u5236\u53F0", children: [_jsxs("div", { className: "mixer-header", children: [_jsx("span", { className: "mixer-title", children: "\uD83C\uDF9B \u63A7\u5236\u53F0" }), _jsx("span", { className: "mixer-hint", children: "\u4E0A\u4E0B\u62D6\u52A8\u63A8\u5B50\u8C03\u97F3\u91CF \u00B7 \u9759\u97F3/\u72EC\u594F \u00B7 \u5E73\u8861=\u5DE6\u53F3\u58F0\u9053 \u00B7 \u64AD\u653E\u65F6\u53EF\u5B9E\u65F6\u64CD\u4F5C" }), _jsx("button", { className: "mixer-close", onClick: onClose, title: "\u6536\u8D77\u63A7\u5236\u53F0", children: "\u2715" })] }), _jsxs("div", { className: "mixer-strips", children: [tracks.map((t, i) => {
                        const color = t.color || trackTypeCssVar(t.trackType);
                        return (_jsxs("div", { className: "mixer-strip", style: { borderTopColor: color }, children: [_jsxs("div", { className: "mixer-strip-name", style: { color }, title: t.name, children: [_jsx("span", { className: "mixer-strip-no", style: { color }, children: i + 1 }), " ", t.name] }), _jsxs("div", { className: "mixer-strip-ms", children: [_jsx("button", { className: `mixer-ms-btn ${t.muted ? "active-mute" : ""}`, onClick: () => toggleMute(t.id, t.muted), title: "\u9759\u97F3", children: "\u9759\u97F3" }), _jsx("button", { className: `mixer-ms-btn ${t.solo ? "active-solo" : ""}`, onClick: () => toggleSolo(t.id, t.solo), title: "\u72EC\u594F", children: "\u72EC\u594F" })] }), _jsxs("div", { className: "mixer-strip-fader", children: [_jsxs("div", { className: "mixer-fader-meter", children: [_jsx(VolumeFader, { value: t.volumeDb, min: FADER_MIN_DB, max: FADER_MAX_DB, orientation: "vertical", width: 32, height: 170, onChange: (v) => setVol(t.id, v), onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction(), tip: "\u8F68\u9053\u97F3\u91CF" }), _jsx(OutLevelMeter, { trackId: t.id, width: 10, height: 170 })] }), _jsx("span", { className: "mixer-strip-db-row", children: _jsx(DbReadout, { value: t.volumeDb, min: FADER_MIN_DB, max: FADER_MAX_DB, onCommit: (v) => setVol(t.id, v), title: "\u70B9\u51FB\u624B\u52A8\u8F93\u5165\u97F3\u91CF dB" }) })] }), _jsxs("div", { className: "mixer-strip-pan", children: [_jsx("span", { className: "mixer-mini-label", children: "\u5E73\u8861" }), _jsx(VolumeFader, { value: t.pan, min: -1, max: 1, width: 84, step: 0.1, fillFrom: "center", format: formatPan, onChange: (v) => setPan(t.id, v), onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction(), tip: "\u5DE6\u53F3\u58F0\u9053\u5E73\u8861" })] })] }, t.id));
                    }), tracks.length === 0 && (_jsx("div", { className: "mixer-empty", children: "\u5DE5\u7A0B\u91CC\u8FD8\u6CA1\u6709\u8F68\u9053 \u2014 \u5148\u521B\u5EFA\u4E00\u6761\u8F68\u9053\u518D\u6765" })), _jsxs("div", { className: "mixer-strip mixer-master", children: [_jsx("div", { className: "mixer-strip-name", style: { color: "#c084fc" }, title: "\u603B\u8F93\u51FA", children: "\uD83D\uDD0A \u603B\u8F93\u51FA" }), _jsxs("div", { className: "mixer-master-body", children: [_jsxs("div", { className: "mixer-fader-meter", children: [_jsx(VolumeFader, { value: masterDb, min: FADER_MIN_DB, max: FADER_MAX_DB, orientation: "vertical", width: 32, height: 170, onChange: setMaster, onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction(), tip: "\u603B\u8F93\u51FA\u97F3\u91CF (\u6574\u4E2A\u8F6F\u4EF6\u7684\u76D1\u542C\u97F3\u91CF, \u4E0D\u5F71\u54CD\u5BFC\u51FA)" }), _jsx(OutLevelMeter, { width: 16, height: 170 })] }), _jsx("span", { className: "mixer-strip-db-row", children: _jsx(DbReadout, { value: masterDb, min: FADER_MIN_DB, max: FADER_MAX_DB, onCommit: setMaster, title: "\u70B9\u51FB\u624B\u52A8\u8F93\u5165\u603B\u8F93\u51FA\u97F3\u91CF dB" }) })] })] })] })] }));
}
