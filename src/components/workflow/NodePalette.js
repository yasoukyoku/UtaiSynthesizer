import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useRef, useEffect, useState, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { useMsstModelStore } from "../../store/msst-models";
import { useAppStore } from "../../store/app";
import { MSST_CATALOG, ALL_CATEGORIES, CATEGORY_LABELS, CATEGORY_COLORS, t18 } from "../../lib/models/msst-catalog";
import { OUTPUT_NODE_COLOR } from "../../lib/constants";
import i18n from "../../i18n";
import "./NodePalette.css";
/** The full palette in sidebar order. Separation entries are filtered to INSTALLED models (a
 *  separation node is unusable without its weights on disk). Callers pass their own `lang`. */
/** The full palette in sidebar order, grouped into 4 categories. Callers pass their own lang. */
export function getPaletteDefs(lang, installedFiles) {
    const availableCategories = ALL_CATEGORIES.filter((cat) => MSST_CATALOG.some((e) => e.category === cat && installedFiles.has(e.filename)));
    return [
        // ─────────────── 💿 母带预设 — Phase 4-4: 8.6 节三档表, 一键插入预连线链 ───────────────
        {
            categoryKey: "workflow.catMastering",
            nodes: [
                { type: "masterPreset:transparent", label: i18n.t("workflow.presetTransparent"), icon: "💿", color: "#4ade80" },
                { type: "masterPreset:standard", label: i18n.t("workflow.presetStandard"), icon: "💿", color: "#facc15" },
                { type: "masterPreset:loud", label: i18n.t("workflow.presetLoud"), icon: "💿", color: "#fb923c" },
            ],
        },
        // ─────────────── 🎚 效果 / 音频处理 ───────────────
        {
            categoryKey: "workflow.catEffects",
            nodes: [
                { type: "transpose", label: "Signalsmith", icon: "🎛", color: "#fbbf24" },
                { type: "speedShift", label: i18n.t("workflow.nodeSpeedShift"), icon: "⏱", color: "#06b6d4" },
                // Phase 1 母带合规节点族
                { type: "merge", label: i18n.t("workflow.nodeMerge"), icon: "🧲", color: "#fb923c" },
                { type: "complianceCheck", label: i18n.t("workflow.nodeComplianceCheck"), icon: "🩺", color: "#22c55e" },
                { type: "lufsNormalize", label: i18n.t("workflow.nodeLufsNormalize"), icon: "📢", color: "#14b8a6" },
                { type: "dither", label: i18n.t("workflow.nodeDither"), icon: "🎚", color: "#64748b" },
                // Phase 4 母带补全节点族
                { type: "dcRemove", label: i18n.t("workflow.nodeDcRemove"), icon: "🧹", color: "#94a3b8" },
                { type: "busEq", label: i18n.t("workflow.nodeBusEq"), icon: "🎚", color: "#38bdf8" },
                { type: "stereoWidth", label: i18n.t("workflow.nodeStereoWidth"), icon: "🎧", color: "#a3e635" },
                { type: "saturate", label: i18n.t("workflow.nodeSaturate"), icon: "🔥", color: "#f97316" },
                { type: "phaseRotate", label: i18n.t("workflow.nodePhaseRotate"), icon: "🌀", color: "#c084fc" },
            ],
        },
        // ─────────────── 🔬 分析可视化 — Phase 5: 零损伤分析, 音频透传 + 报告旁路 ───────────────
        {
            categoryKey: "workflow.catAnalysis",
            nodes: [
                { type: "spectrogram", label: i18n.t("workflow.nodeSpectrogram"), icon: "🌊", color: "#38bdf8" },
                { type: "f0Curve", label: i18n.t("workflow.nodeF0Curve"), icon: "〰", color: "#22d3ee" },
                { type: "timbreMetrics", label: i18n.t("workflow.nodeTimbreMetrics"), icon: "🎨", color: "#fbbf24" },
                { type: "harmonicityCheck", label: i18n.t("workflow.nodeHarmonicityCheck"), icon: "🔬", color: "#4ade80" },
                { type: "spectralCompare", label: i18n.t("workflow.nodeSpectralCompare"), icon: "⚖", color: "#a78bfa" },
                { type: "dtwAlign", label: i18n.t("workflow.nodeDtwAlign"), icon: "🧭", color: "#f472b6" },
                { type: "abCompare", label: i18n.t("workflow.nodeAbCompare"), icon: "🔀", color: "#fb923c" },
                { type: "lufsAnalyze", label: i18n.t("workflow.nodeLufsAnalyze"), icon: "📊", color: "#8b5cf6" },
                { type: "melodySimilarity", label: i18n.t("workflow.nodeMelodySimilarity"), icon: "🔍", color: "#f59e0b" },
            ],
        },
        // ─────────────── 🎵 AI / 智能 ───────────────
        {
            categoryKey: "workflow.catAI",
            nodes: [
                { type: "rvc", label: i18n.t("workflow.nodeRvc"), icon: "🗣", color: "#39c5bb" },
                { type: "sovits", label: i18n.t("workflow.nodeSovits"), icon: "🎤", color: "#8b5cf6" },
                { type: "chordDetect", label: i18n.t("workflow.nodeChordDetect"), icon: "🎼", color: "#a78bfa" },
                { type: "autoArrange", label: i18n.t("workflow.nodeAutoArrange"), icon: "🥁", color: "#ec4899" },
                { type: "melodyGen", label: i18n.t("workflow.nodeMelodyGen"), icon: "🎵", color: "#10b981" },
                { type: "harmonizer", label: i18n.t("workflow.nodeHarmonizer"), icon: "🎶", color: "#06b6d4" },
                { type: "deepOriginal", label: i18n.t("workflow.nodeDeepOriginal"), icon: "✨", color: "#f59e0b" },
            ],
        },
        // ─────────────── 🎼 歌曲生成 (Song) — P2-14 歌曲节点族, 统一走 runSongTask ───────────────
        {
            categoryKey: "workflow.catSong",
            nodes: [
                { type: "songLyrics", label: i18n.t("songNode.lyrics"), icon: "✍", color: "#f472b6" },
                { type: "songPrompt", label: i18n.t("songNode.prompt"), icon: "🎨", color: "#fb7185" },
                { type: "songGen", label: i18n.t("songNode.gen"), icon: "🎵", color: "#ec4899" },
                { type: "songCover", label: i18n.t("songNode.cover"), icon: "🎤", color: "#f43f5e" },
                { type: "songRepaint", label: i18n.t("songNode.repaint"), icon: "🖌", color: "#e879f9" },
                { type: "songComplete", label: i18n.t("songNode.complete"), icon: "🧩", color: "#c084fc" },
                { type: "songExtract", label: i18n.t("songNode.extract"), icon: "✂", color: "#a78bfa" },
                { type: "songLego", label: i18n.t("songNode.lego"), icon: "🧱", color: "#818cf8" },
                { type: "songStems", label: i18n.t("songNode.stems"), icon: "🎚", color: "#60a5fa" },
                { type: "songSheet", label: i18n.t("songNode.sheet"), icon: "🎹", color: "#22d3ee" },
            ],
        },
        // ─────────────── ♾ 符号域原创化 — P2 符号域节点族 (确定性, 种子可复现) ───────────────
        {
            categoryKey: "workflow.catSymbol",
            nodes: [
                { type: "midiHumanize", label: i18n.t("workflow.nodeMidiHumanize"), icon: "🎹", color: "#38bdf8" },
                { type: "velocityCurve", label: i18n.t("workflow.nodeVelocityCurve"), icon: "📈", color: "#f472b6" },
                { type: "swingQuantize", label: i18n.t("workflow.nodeSwingQuantize"), icon: "🥁", color: "#a3e635" },
                { type: "melodyReharm", label: i18n.t("workflow.nodeMelodyReharm"), icon: "🎼", color: "#34d399" },
                { type: "rhythmRestructure", label: i18n.t("workflow.nodeRhythmRestructure"), icon: "🥁", color: "#fbbf24" },
                { type: "contourMorph", label: i18n.t("workflow.nodeContourMorph"), icon: "🎢", color: "#f97316" },
                { type: "motifDevelop", label: i18n.t("workflow.nodeMotifDevelop"), icon: "🧬", color: "#c084fc" },
                { type: "reharmonize", label: i18n.t("workflow.nodeReharmonize"), icon: "🎷", color: "#22d3ee" },
                { type: "rhythmVariation", label: i18n.t("workflow.nodeRhythmVariation"), icon: "🕹", color: "#e879f9" },
                { type: "structureEdit", label: i18n.t("workflow.nodeStructureEdit"), icon: "🏗", color: "#facc15" },
                { type: "breathPlanner", label: i18n.t("workflow.nodeBreathPlanner"), icon: "💨", color: "#60a5fa" },
            ],
        },
        // ─────────────── 🔊 音频分离 ───────────────
        {
            categoryKey: "workflow.catSeparation",
            nodes: [
                ...availableCategories.map((cat) => ({
                    type: "separation",
                    label: t18(CATEGORY_LABELS[cat], lang),
                    icon: t18(CATEGORY_LABELS[cat], lang).charAt(0),
                    color: CATEGORY_COLORS[cat],
                    extraParams: { category: cat },
                })),
                { type: "amtMidi", label: i18n.t("workflow.nodeAmtMidi"), icon: "♪", color: "#a855f7" },
            ],
        },
        // ─────────────── 🎹 MIDI / 输入 ───────────────
        {
            categoryKey: "workflow.catMidi",
            nodes: [
                { type: "audioInput", label: i18n.t("workflow.nodeAudioInput"), icon: "🎵", color: "#60a5fa" },
                { type: "midiFileIn", label: i18n.t("workflow.nodeMidiFileIn"), icon: "🎹", color: "#c084fc" },
                { type: "chordBlockIn", label: i18n.t("workflow.nodeChordBlock"), icon: "🎼", color: "#facc15" },
                { type: "soundfontRender", label: i18n.t("workflow.nodeSoundfontRender"), icon: "🎻", color: "#2dd4bf" },
            ],
        },
        // ─────────────── 📤 输出 ───────────────
        {
            categoryKey: "workflow.catIO",
            nodes: [
                { type: "split", label: i18n.t("workflow.nodeSplit"), icon: "✂", color: "#94a3b8" },
                { type: "audioOutput", label: i18n.t("workflow.nodeOutput"), icon: "📤", color: OUTPUT_NODE_COLOR },
            ],
        },
    ];
}
export function NodePalette({ onAddNode, onDropNode }) {
    const { t, i18n } = useTranslation();
    const lang = i18n.language;
    const installed = useMsstModelStore((s) => s.installed);
    const installedFiles = new Set(installed.map((m) => m.filename));
    const toggleModelManager = useAppStore((s) => s.toggleModelManager);
    const groups = getPaletteDefs(lang, installedFiles);
    const [searchQuery, setSearchQuery] = useState("");
    // 拼音首字母映射表
    const pinyinMap = useMemo(() => ({
        "变速": "bs", "移调": "yt", "合并": "hb", "母带": "md", "合规": "hg", "检测": "jc",
        "响度": "xd", "归一": "gy", "抖动": "dd", "直流": "zl", "移除": "yc", "均衡": "jh",
        "立体": "lt", "宽度": "kd", "饱和": "bh", "相位": "xw", "旋转": "xz", "频谱": "pp",
        "音高": "yg", "曲线": "qx", "音色": "ys", "指标": "zb", "谐波": "xb", "对比": "db",
        "时间": "sj", "规整": "gz", "相似": "xs", "分析": "fx", "和弦": "hx", "编曲": "bq",
        "旋律": "xl", "生成": "sc", "和声": "hs", "原创": "yc", "歌词": "gc", "提示": "ts",
        "翻唱": "fc", "重绘": "ch", "补全": "bq", "提取": "tq", "乐高": "lg", "分轨": "fg",
        "简谱": "jp", "人性": "rx", "力度": "ld", "摇摆": "yb", "量化": "lh", "复调": "fd",
        "节奏": "jz", "重组": "cz", "轮廓": "lk", "变形": "bx", "动机": "dj", "发展": "fz",
        "配和": "ph", "变化": "bh", "结构": "jg", "编辑": "bj", "换气": "hq", "规划": "gh",
        "分离": "fl", "输入": "sr", "输出": "sc", "拆分": "cf", "音频": "yp", "文件": "wj",
        "音源": "yy", "渲染": "xr", "预设": "ys", "透明": "tm", "标准": "bz", "响亮": "xl",
    }), []);
    // 搜索过滤逻辑
    const filteredGroups = useMemo(() => {
        if (!searchQuery.trim())
            return groups;
        const query = searchQuery.toLowerCase().trim();
        return groups.map(group => {
            const filteredNodes = group.nodes.filter(node => {
                const label = node.label.toLowerCase();
                const type = node.type.toLowerCase();
                // 直接匹配节点名称或类型
                if (label.includes(query) || type.includes(query))
                    return true;
                // 拼音首字母匹配
                for (const [word, pinyin] of Object.entries(pinyinMap)) {
                    if (label.includes(word) && pinyin.includes(query))
                        return true;
                }
                // 功能关键词匹配
                const keywords = {
                    "rvc": ["换声", "变声", "转换", "声音"],
                    "sovits": ["换声", "变声", "tts"],
                    "transpose": ["变调", "移调", "音高"],
                    "speedShift": ["变速", "加速", "减速"],
                    "merge": ["合并", "混合"],
                    "separation": ["分离", "提取", "分轨"],
                    "chordDetect": ["和弦", "识别", "检测"],
                    "autoArrange": ["编曲", "伴奏"],
                    "lufsNormalize": ["响度", "音量", "归一化"],
                    "busEq": ["均衡器", "eq", "频率"],
                    "stereoWidth": ["立体声", "声场", "宽度"],
                };
                const nodeKeywords = keywords[node.type] || [];
                return nodeKeywords.some(kw => kw.includes(query));
            });
            return {
                ...group,
                nodes: filteredNodes,
            };
        }).filter(group => group.nodes.length > 0);
    }, [groups, searchQuery, pinyinMap]);
    const dragRef = useRef(null);
    const ghostRef = useRef(null);
    useEffect(() => {
        const onMove = (e) => {
            if (!dragRef.current || !ghostRef.current)
                return;
            ghostRef.current.style.left = `${e.clientX - 40}px`;
            ghostRef.current.style.top = `${e.clientY - 12}px`;
        };
        const onUp = (e) => {
            if (!dragRef.current)
                return;
            const { type, label, extraParams } = dragRef.current;
            dragRef.current = null;
            if (ghostRef.current) {
                ghostRef.current.remove();
                ghostRef.current = null;
            }
            document.body.style.cursor = "";
            if (onDropNode) {
                onDropNode(type, label, e.clientX, e.clientY, extraParams);
            }
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        return () => {
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
        };
    }, [onDropNode]);
    const startDrag = useCallback((e, type, label, extraParams) => {
        e.preventDefault();
        dragRef.current = { type, label, extraParams };
        document.body.style.cursor = "grabbing";
        const ghost = document.createElement("div");
        ghost.className = "palette-drag-ghost";
        ghost.textContent = label;
        ghost.style.left = `${e.clientX - 40}px`;
        ghost.style.top = `${e.clientY - 12}px`;
        document.body.appendChild(ghost);
        ghostRef.current = ghost;
    }, []);
    return (_jsxs("aside", { className: "node-palette", children: [_jsx("div", { className: "palette-title", children: t("workflow.nodes") }), _jsxs("div", { className: "palette-search", children: [_jsx("input", { type: "text", className: "palette-search-input", placeholder: t("workflow.searchNodes") || "搜索节点...", value: searchQuery, onChange: (e) => setSearchQuery(e.target.value) }), searchQuery && (_jsx("button", { className: "palette-search-clear", onClick: () => setSearchQuery(""), title: t("workflow.clearSearch") || "清除搜索", children: "\u2715" }))] }), filteredGroups.length === 0 && searchQuery && (_jsx("div", { className: "palette-no-results", children: t("workflow.noSearchResults") || "未找到匹配的节点" })), filteredGroups.map((g) => (_jsxs("div", { className: "palette-category", children: [_jsx("div", { className: "palette-category-title", children: t(g.categoryKey) }), g.nodes.map((n) => (_jsx(PaletteItem, { type: n.type, label: n.label, icon: n.icon, color: n.color, extraParams: n.extraParams, onAdd: onAddNode, onDrag: startDrag }, `${n.type}-${n.label}`))), g.categoryKey === "workflow.catSeparation" && g.nodes.length === 0 && (_jsx("span", { className: "palette-empty", children: t18({ zh: "未安装分离模型", en: "No separation models", ja: "分離モデル未インストール" }, lang) })), g.categoryKey === "workflow.catSeparation" && (_jsx("button", { className: "palette-manage-btn", onClick: toggleModelManager, children: t18({ zh: "管理模型...", en: "Manage models...", ja: "モデル管理..." }, lang) }))] }, g.categoryKey)))] }));
}
function PaletteItem({ type, label, icon, color, extraParams, onAdd, onDrag }) {
    return (_jsxs("div", { className: "palette-node", role: "button", tabIndex: 0, onClick: () => onAdd(type, label, extraParams), onMouseDown: (e) => { if (e.button === 0)
            onDrag(e, type, label, extraParams); }, style: { "--node-color": color }, children: [_jsxs("span", { className: "palette-node-icon", children: ["[", icon, "]"] }), _jsx("span", { className: "palette-node-label", children: label })] }));
}
