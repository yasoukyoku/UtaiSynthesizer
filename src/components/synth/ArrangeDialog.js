import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
// 智能自动编曲面板(三段式):① 歌曲身份证(实时 BPM/调/拍号/和弦) ② 风格情绪 + 乐器分组勾选
// ③ 底部[试听·清空·生成 N 轨]。引擎/动作层见 lib/arrangement/*;本组件只做交互与展示。
import { useEffect, useMemo, useRef, useState } from "react";
import { ARRANGE_STYLES, ARRANGE_MOODS, } from "../../lib/arrangement/styles";
import { runAutoArrange, previewAutoArrange, stopAutoArrangePreview, clearAutoArrange, analyzeSourceProfile, } from "../../lib/arrangement/autoArrange";
import { useProjectStore } from "../../store/project";
import "./ArrangeDialog.css";
/** 22 种风格中文名(轨道名语言无关,这里仅面板展示)。 */
const STYLE_LABELS = {
    pop: "流行", rock: "摇滚", jazz: "爵士", edm: "电子舞曲", latin: "拉丁",
    funk: "放克", rnb: "R&B", ballad: "抒情", country: "乡村", lofi: "Lo-Fi",
    trap: "Trap", reggae: "雷鬼", bossanova: "波萨诺瓦", kpop: "K-Pop",
    chinese: "中国风", ambient: "氛围", techno: "科技舞曲",
    folk: "民谣", citypop: "都市流行", hiphop: "嘻哈", house: "浩室", cinematic: "史诗影视",
};
/** 6 种情绪中文名(只影响力度/密度/铺底八度)。 */
const MOOD_LABELS = {
    neutral: "中性", happy: "明亮", sad: "忧伤",
    energetic: "有力度", chill: "松弛", epic: "宏大",
};
/** 11 类乐器分 4 组(节奏 / 和声 / 铺底 / 旋律)。 */
const INST_GROUPS = [
    {
        title: "节奏组",
        items: [
            { id: "drums", name: "鼓", desc: "节拍骨架" },
            { id: "bass", name: "贝斯", desc: "低音根基" },
        ],
    },
    {
        title: "和声组",
        items: [
            { id: "piano", name: "钢琴", desc: "柱式/琶音" },
            { id: "guitarArp", name: "吉他分解", desc: "拨弦" },
            { id: "guitarStrum", name: "吉他扫弦", desc: "柱式" },
            { id: "epiano", name: "电钢琴", desc: "柔和" },
        ],
    },
    {
        title: "铺底组",
        items: [
            { id: "strings", name: "弦乐", desc: "长音铺底" },
            { id: "chords", name: "铺底Pad", desc: "和弦垫" },
            { id: "synthPad", name: "合成Pad", desc: "空间感" },
        ],
    },
    {
        title: "旋律组",
        items: [
            { id: "pluck", name: "琶音", desc: "颗粒点缀" },
            { id: "melody", name: "AI旋律", desc: "自动加花" },
        ],
    },
];
/** 默认四大件:鼓 + 贝斯 + 钢琴 + 弦乐(主流流行最稳的组合)。 */
const DEFAULT_INST = ["drums", "bass", "piano", "strings"];
const ALL_INST = INST_GROUPS.flatMap((g) => g.items.map((i) => i.id));
/** localStorage key: 编曲最后一次成功生成的配置 — 新工程/重启自动恢复。 */
const LS_ARRANGE_LAST = "muno.arrange.last";
/** 5 个快捷预设 — 最常用的乐器组合,一键套用。 */
const QUICK_PRESETS = [
    { name: "流行四大件", emoji: "🎶", style: "pop", mood: "neutral", instruments: ["drums", "bass", "piano", "strings"] },
    { name: "民谣吉他组", emoji: "🎸", style: "folk", mood: "chill", instruments: ["drums", "bass", "guitarStrum", "strings", "melody"] },
    { name: "Lo-Fi Chill", emoji: "🌙", style: "lofi", mood: "chill", instruments: ["drums", "bass", "epiano", "synthPad", "pluck"] },
    { name: "摇滚齐奏", emoji: "🎸", style: "rock", mood: "energetic", instruments: ["drums", "bass", "guitarStrum", "guitarArp", "melody"] },
    { name: "Jazz Trio", emoji: "🎷", style: "jazz", mood: "happy", instruments: ["drums", "bass", "piano"] },
];
function loadLastPreset() {
    try {
        const raw = localStorage.getItem(LS_ARRANGE_LAST);
        if (!raw)
            return null;
        const rec = JSON.parse(raw);
        if (rec && rec.style && rec.mood && Array.isArray(rec.instruments))
            return rec;
        return null;
    }
    catch {
        return null;
    }
}
function saveLastPreset(p) {
    try {
        localStorage.setItem(LS_ARRANGE_LAST, JSON.stringify(p));
    }
    catch { /* ignore */ }
}
export function ArrangeDialog({ trackId, onClose }) {
    const last = loadLastPreset();
    const [style, setStyle] = useState(last?.style ?? "pop");
    const [mood, setMood] = useState(last?.mood ?? "neutral");
    const [instruments, setInstruments] = useState(last?.instruments ?? DEFAULT_INST);
    const [previewing, setPreviewing] = useState(false);
    const [busy, setBusy] = useState(false);
    // 订阅 tracks:源轨音符一旦变化(如刚转谱/编辑),身份证实时重算。
    const tracks = useProjectStore((s) => s.tracks);
    const track = tracks.find((t) => t.id === trackId);
    const profile = useMemo(() => (trackId ? analyzeSourceProfile(trackId) : null), [trackId, tracks]);
    // 卸载即停止试听,避免面板关了还在响。
    useEffect(() => () => stopAutoArrangePreview(), []);
    // 风格/情绪/勾选一旦变化,立即停掉正在播的旧预览,保证「听到的 = 当前选的」(首次挂载跳过)。
    const firstSel = useRef(true);
    useEffect(() => {
        if (firstSel.current) {
            firstSel.current = false;
            return;
        }
        stopAutoArrangePreview();
        setPreviewing(false);
    }, [style, mood, instruments]);
    const toggle = (id) => setInstruments((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
    const toggleGroup = (group, on) => setInstruments((prev) => {
        const ids = group.items.map((i) => i.id);
        const rest = prev.filter((x) => !ids.includes(x));
        return on ? [...rest, ...ids] : rest;
    });
    const handlePreview = async () => {
        if (previewing) {
            stopAutoArrangePreview();
            setPreviewing(false);
            return;
        }
        if (instruments.length === 0)
            return;
        setBusy(true);
        const ok = await previewAutoArrange(trackId, style, mood, {
            instruments,
            onEnded: () => setPreviewing(false),
        });
        setPreviewing(ok);
        setBusy(false);
    };
    const handleClear = () => {
        stopAutoArrangePreview();
        setPreviewing(false);
        clearAutoArrange(trackId);
    };
    const handleGenerate = async () => {
        if (instruments.length === 0)
            return;
        stopAutoArrangePreview();
        setBusy(true);
        const ok = await runAutoArrange(trackId, style, mood, { instruments });
        setBusy(false);
        if (ok) {
            saveLastPreset({ style, mood, instruments });
            onClose();
        }
    };
    return (_jsx("div", { className: "arrange-overlay", onClick: onClose, children: _jsxs("div", { className: "arrange-panel arrange-panel-wide", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "arrange-head", children: [_jsx("span", { className: "arrange-title", children: "\u667A\u80FD\u7F16\u66F2" }), _jsx("button", { className: "arrange-close", onClick: onClose, children: "\u2715" })] }), _jsxs("div", { className: "arrange-body", children: [_jsxs("div", { className: "idcard", children: [_jsx("div", { className: "idcard-title", children: "\u6B4C\u66F2\u8EAB\u4EFD\u8BC1 \u00B7 \u81EA\u52A8\u8BC6\u522B" }), profile ? (_jsxs(_Fragment, { children: [_jsxs("div", { className: "idcard-grid", children: [_jsxs("div", { className: "idcell", children: [_jsx("span", { className: "idcell-k", children: "\u901F\u5EA6" }), _jsx("span", { className: "idcell-v", children: profile.bpm }), _jsx("span", { className: "idcell-u", children: "BPM" })] }), _jsxs("div", { className: "idcell", children: [_jsx("span", { className: "idcell-k", children: "\u8C03\u6027" }), _jsx("span", { className: "idcell-v", children: profile.keyLabel })] }), _jsxs("div", { className: "idcell", children: [_jsx("span", { className: "idcell-k", children: "\u62CD\u53F7" }), _jsxs("span", { className: "idcell-v", children: [profile.beatsPerBar, "/", profile.beatUnit] })] }), _jsxs("div", { className: "idcell", children: [_jsx("span", { className: "idcell-k", children: "\u5C0F\u8282" }), _jsx("span", { className: "idcell-v", children: profile.bars })] })] }), _jsxs("div", { className: "idcard-chords", title: profile.chordLabels.join("  "), children: [_jsx("span", { className: "idcell-k", children: "\u548C\u5F26" }), _jsx("span", { className: "idcard-chordseq", children: profile.chordLabels.join(" · ") || "—" })] })] })) : (_jsx("div", { className: "idcard-empty", children: "\u8FD9\u6761\u8F68\u8FD8\u6CA1\u6709\u97F3\u7B26 \u2014\u2014 \u82E5\u662F\u97F3\u9891\u8BF7\u5148\u300CAI \u8F6C\u8C31\u300D\u518D\u7F16\u66F2" }))] }), _jsx("div", { style: { display: "flex", gap: 6, flexWrap: "wrap", margin: "0 0 8px 0" }, children: QUICK_PRESETS.map((p) => (_jsxs("button", { type: "button", style: {
                                    fontSize: 11, padding: "3px 8px", borderRadius: 14,
                                    border: "1px solid var(--border-default)", background: (style === p.style && mood === p.mood && instruments.length === p.instruments.length) ? "var(--accent-primary)" : "var(--bg-surface)",
                                    color: "var(--text-primary)", cursor: "pointer", fontFamily: "inherit",
                                }, onClick: () => { setStyle(p.style); setMood(p.mood); setInstruments(p.instruments); }, title: `${STYLE_LABELS[p.style]} · ${MOOD_LABELS[p.mood]} · ${p.instruments.length} 轨`, children: [p.emoji, " ", p.name] }, p.name))) }), _jsxs("div", { className: "arrange-row", children: [_jsxs("label", { className: "arrange-field", children: [_jsx("span", { children: "\u98CE\u683C" }), _jsx("select", { value: style, onChange: (e) => setStyle(e.target.value), children: ARRANGE_STYLES.map((s) => (_jsx("option", { value: s, children: STYLE_LABELS[s] }, s))) })] }), _jsxs("label", { className: "arrange-field", children: [_jsx("span", { children: "\u60C5\u7EEA" }), _jsx("select", { value: mood, onChange: (e) => setMood(e.target.value), children: ARRANGE_MOODS.map((m) => (_jsx("option", { value: m, children: MOOD_LABELS[m] }, m))) })] })] }), _jsxs("div", { className: "inst-block", children: [_jsxs("div", { className: "inst-block-head", children: [_jsx("span", { children: "\u4E50\u5668\u81EA\u7531\u7EC4\u5408" }), _jsxs("span", { className: "inst-block-actions", children: [_jsx("button", { type: "button", onClick: () => setInstruments(ALL_INST), children: "\u5168\u9009" }), _jsx("button", { type: "button", onClick: () => setInstruments([]), children: "\u6E05\u7A7A" })] })] }), INST_GROUPS.map((g) => {
                                    const ids = g.items.map((i) => i.id);
                                    const onCount = ids.filter((id) => instruments.includes(id)).length;
                                    return (_jsxs("div", { className: "inst-group", children: [_jsxs("div", { className: "inst-group-head", children: [_jsx("span", { className: "inst-group-title", children: g.title }), _jsx("button", { type: "button", className: "inst-group-toggle", onClick: () => toggleGroup(g, onCount !== ids.length), children: onCount === ids.length ? "取消" : "整组" })] }), _jsx("div", { className: "inst-chips", children: g.items.map((it) => {
                                                    const on = instruments.includes(it.id);
                                                    return (_jsxs("button", { type: "button", className: `inst-chip${on ? " on" : ""}`, "aria-pressed": on, onClick: () => toggle(it.id), title: it.desc, children: [_jsx("span", { className: "inst-chip-name", children: it.name }), _jsx("span", { className: "inst-chip-desc", children: it.desc })] }, it.id));
                                                }) })] }, g.title));
                                })] })] }), _jsxs("div", { className: "arrange-foot arrange-foot-split", children: [_jsx("button", { className: "arrange-btn", onClick: handleClear, children: "\u6E05\u7A7A\u4F34\u594F" }), _jsxs("div", { className: "arrange-foot-right", children: [_jsx("button", { className: `arrange-btn${previewing ? " active" : ""}`, disabled: busy || instruments.length === 0 || !profile, onClick: handlePreview, children: previewing ? "■ 停止" : "▶ 试听 8 小节" }), _jsx("button", { className: "arrange-btn primary", disabled: busy || instruments.length === 0 || !profile, onClick: handleGenerate, children: busy ? "处理中…" : `生成 ${instruments.length} 轨` })] })] }), track && _jsxs("div", { className: "arrange-source", children: ["\u7F16\u66F2\u6E90\uFF1A", _jsx("strong", { children: track.name })] })] }) }));
}
