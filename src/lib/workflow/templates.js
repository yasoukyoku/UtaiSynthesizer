/**
 * §user「超级原创」内置模板——把工程的既有节点（分离 / AMT 扒带 / RVC / 移调 / 输出）
 * 组装成三条一键流水线。模板是纯数据：生成 Workflow 对象后走与手动编辑器完全相同的
 * 执行引擎（preflightRun → executeWorkflow），因此缺失组件检测、进度、取消、轨道沉积、
 * 缓存全部免费复用。
 *
 *   transcribe  一键扒带：input → AMT(smart 多乐器) 。MIDI 自动落主时间线（换音源/修复的起点）。
 *   clone       声音复刻：input → 分离(人声) → [人声→RVC]→输出 + [伴奏]→输出。
 *   original    深度原创：clone + [伴奏→移调]→输出 + 平行 AMT 扒带（换音源起点）。
 *
 * 端口连线由向导按分离模型的真实 stem 顺序解析（resolveVocalPorts）；modelPath 不写入
 * params —— 引擎执行时用 modelFile 自愈（modelPathHeal.healMsstModelPath），天然可移植。
 */
import { MUSCRIPTOR_INSTRUMENTS } from "../models/muscriptor-instruments";
/** 分离模型真序 → 人声/伴奏端口索引（大小写不敏感；兜底 0/1）。 */
export function resolveVocalPorts(stemNames) {
    const names = (stemNames ?? []).map((s) => s.toLowerCase());
    if (names.length === 0)
        return { vocalPort: 0, instPort: 1 };
    let vocalPort = names.findIndex((s) => s.includes("vocal") || s.includes("voice"));
    if (vocalPort < 0)
        vocalPort = 0;
    let instPort = names.findIndex((s, i) => i !== vocalPort && (s.includes("instrument") || s.includes("accompan") || s.includes("other") || s.includes("no_vocal") || s.includes("karaoke")));
    if (instPort < 0)
        instPort = names.length > 1 ? (vocalPort === 0 ? 1 : 0) : vocalPort;
    return { vocalPort, instPort };
}
/** 人声分离模型 stem 端口数（供输出连线校验；兜底 2）。 */
export function stemCount(stemNames) {
    return Math.max(2, stemNames?.length ?? 2);
}
const node = (id, nodeType, x, y, params) => ({ id, nodeType, position: { x, y }, params });
const conn = (fromNode, fromPort, toNode, toPort) => ({ fromNode, fromPort, toNode, toPort });
/** 分离节点 params：modelFile（引擎自愈路径的唯一稳定标识）+ 真序 stemLabels。 */
function sepParams(ctx) {
    const p = { category: "vocals", device: "cuda" };
    if (ctx.separationModelFile)
        p.modelFile = ctx.separationModelFile;
    if (ctx.vocalStemNames?.length)
        p.stemLabels = ctx.vocalStemNames;
    return p;
}
/** AMT 节点 params（smart 多乐器模式；backend 默认 muscriptor = 最优转谱）。
 *  MuScriptor 默认全选全部 36 种乐器 —— 用户要精简可以在节点面板取消勾选，
 *  但「默认空数组导致只有 4 轨」是最常见的踩坑。 */
function amtParams(ctx) {
    return {
        midiMode: "smart",
        backend: ctx.amtBackend ?? "muscriptor",
        useGpu: true,
        midiTrackMode: "multi_track",
        quantizeGrid: "off",
        muscriptorInstruments: [...MUSCRIPTOR_INSTRUMENTS],
        muscriptorChain: "official",
    };
}
/** 一键扒带：AMT 多乐器转 MIDI（自动落轨）；MIDI 路径接到输出节点以便预览/导出。 */
export function buildTranscribeWorkflow(ctx) {
    return {
        nodes: [
            node("input", "input", 0, 0, {}),
            node("amt", "amtMidi", 300, 0, amtParams(ctx)),
            node("out", "output", 620, 0, { laneLabel: "一键扒带" }),
        ],
        connections: [
            conn("input", 0, "amt", 0),
            conn("amt", 0, "out", 0),
        ],
    };
}
/** 声音复刻：分离 → 人声 RVC + 伴奏直出。 */
export function buildCloneWorkflow(ctx) {
    const { vocalPort, instPort } = resolveVocalPorts(ctx.vocalStemNames);
    const vm = ctx.voiceModel;
    return {
        nodes: [
            node("input", "input", 0, 0, {}),
            node("sep", "msstSeparation", 300, 0, sepParams(ctx)),
            node("rvc", "rvc", 620, -60, vm ? { voiceName: vm.name, modelPath: vm.path } : {}),
            node("out", "output", 940, 0, { laneLabel: "超级原创" }),
        ],
        connections: [
            conn("input", 0, "sep", 0),
            conn("sep", vocalPort, "rvc", 0),
            conn("rvc", 0, "out", 0),
            conn("sep", instPort, "out", 1),
        ],
    };
}
/** 深度原创：复刻 + 伴奏移调 + 平行 AMT 扒带（换音源起点）。 */
export function buildOriginalWorkflow(ctx) {
    const { vocalPort, instPort } = resolveVocalPorts(ctx.vocalStemNames);
    const vm = ctx.voiceModel;
    const semis = ctx.transposeSemitones ?? 2;
    return {
        nodes: [
            node("input", "input", 0, 0, {}),
            node("sep", "msstSeparation", 280, 0, sepParams(ctx)),
            node("rvc", "rvc", 580, -90, vm ? { voiceName: vm.name, modelPath: vm.path } : {}),
            node("transpose", "transpose", 580, 110, { semitones: semis, formantFollow: 0, formantOffset: 0 }),
            node("amt", "amtMidi", 280, 240, amtParams(ctx)),
            node("out", "output", 900, 0, { laneLabel: "超级原创" }),
        ],
        connections: [
            conn("input", 0, "sep", 0),
            conn("input", 0, "amt", 0),
            conn("sep", vocalPort, "rvc", 0),
            conn("rvc", 0, "out", 0),
            conn("sep", instPort, "transpose", 0),
            conn("transpose", 0, "out", 1),
        ],
    };
}
export function buildWorkflow(template, ctx) {
    switch (template) {
        case "transcribe": return buildTranscribeWorkflow(ctx);
        case "clone": return buildCloneWorkflow(ctx);
        case "original": return buildOriginalWorkflow(ctx);
        case "rebuild": return buildRebuildWorkflow(ctx);
        case "repaint": return buildRepaintWorkflow(ctx);
        case "lyrics2song": return buildLyrics2SongWorkflow(ctx);
        case "segRepaint": return buildSegRepaintWorkflow(ctx);
        case "vocal2acc": return buildVocal2AccWorkflow(ctx);
        case "stemsRebuild": return buildStemsRebuildWorkflow(ctx);
    }
}
/** 规划 12.2/12.5 路线 A「原创重建」：人声换声 + 整曲 AMT 转 MIDI（伴奏侧不留原波形——
 *  伴奏由用户换音源重演奏；amtMidi 自动落主时间线，转谱后即可重编曲）。 */
export function buildRebuildWorkflow(ctx) {
    const { vocalPort } = resolveVocalPorts(ctx.vocalStemNames);
    const vm = ctx.voiceModel;
    return {
        nodes: [
            node("input", "input", 0, 0, {}),
            node("sep", "msstSeparation", 280, 0, sepParams(ctx)),
            node("rvc", "rvc", 580, -90, vm ? { voiceName: vm.name, modelPath: vm.path } : {}),
            node("amt", "amtMidi", 280, 240, amtParams(ctx)),
            node("out", "output", 900, 0, { laneLabel: "原创重建" }),
        ],
        connections: [
            conn("input", 0, "sep", 0),
            conn("input", 0, "amt", 0),
            conn("sep", vocalPort, "rvc", 0),
            conn("rvc", 0, "out", 0),
            conn("amt", 0, "out", 1),
        ],
    };
}
/** 规划 12.2/12.5 路线 B「快速重绘」：人声换声 + 伴奏移调 + 变速（songRepaint/songComplete
 *  局部重绘节点随 P2 歌曲节点上线后可插入伴奏链）。 */
export function buildRepaintWorkflow(ctx) {
    const { vocalPort, instPort } = resolveVocalPorts(ctx.vocalStemNames);
    const vm = ctx.voiceModel;
    const semis = ctx.transposeSemitones ?? 2;
    const speed = ctx.speedFactor ?? 1;
    const nodes = [
        node("input", "input", 0, 0, {}),
        node("sep", "msstSeparation", 280, 0, sepParams(ctx)),
        node("rvc", "rvc", 580, -90, vm ? { voiceName: vm.name, modelPath: vm.path } : {}),
        node("transpose", "transpose", 580, 110, { semitones: semis, formantFollow: 0, formantOffset: 0 }),
    ];
    const connections = [
        conn("input", 0, "sep", 0),
        conn("sep", vocalPort, "rvc", 0),
        conn("rvc", 0, "out", 0),
        conn("sep", instPort, "transpose", 0),
    ];
    if (Math.abs(speed - 1) > 0.001) {
        nodes.push(node("speed", "speedShift", 820, 110, { speed }));
        connections.push(conn("transpose", 0, "speed", 0));
        connections.push(conn("speed", 0, "out", 1));
    }
    else {
        connections.push(conn("transpose", 0, "out", 1));
    }
    nodes.push(node("out", "output", 1100, 0, { laneLabel: "快速重绘" }));
    return { nodes, connections };
}
/** 模板是否需要声音模型（前端组件检查用）。 */
export const templateNeedsVoiceModel = (t) => t === "clone" || t === "original" || t === "rebuild" || t === "repaint";
/** 模板是否需要 AMT(audio→MIDI) 转谱后端:只有图里含 amtMidi 节点的模板才检查
 *  (扒带/深度原创/原创重建);歌曲模板与纯换声模板不因缺 AMT 而被拦(规划 6-6)。 */
export const templateNeedsAmt = (t) => t === "transcribe" || t === "original" || t === "rebuild";
/** 模板是否需要人声分离模型。歌曲模板内部分轨由 song 节点自治（demucs/ACE），不走 MSST。 */
export const templateNeedsSeparation = (t) => t === "clone" || t === "original" || t === "rebuild" || t === "repaint";
/** stem 端口数（校验连线不越界）。 */
export function templateStemCount(template, ctx) {
    return templateNeedsSeparation(template) ? stemCount(ctx.vocalStemNames) : 0;
}
// ── P2-14 歌曲制作模板（规划 11.6）────────────────────────────────────────
// 与既有模板同一约定：多路结果共用一个 Output 节点（每条入边 = 一条沉积 lane，toPort 递增）。
/** 词曲一键成歌：歌词源 + 风格源 → songGen → 整首。模型/产物开关在节点面板里调。 */
export function buildLyrics2SongWorkflow(_ctx) {
    return {
        nodes: [
            // 图解析器与执行器硬性要求 input 节点存在（graph.inputNodeId）；此模板不消费音频，
            // 但引擎会把片段源音频灌进 input 节点，songGen 的参考音频口（口 2）留空即可。
            node("input", "input", 0, 90, {}),
            node("lyr", "songLyrics", 0, 0, {}),
            node("sty", "songPrompt", 0, 180, {}),
            node("gen", "songGen", 320, 40, {}),
            node("out", "output", 660, 40, { laneLabel: "词曲成歌" }),
        ],
        connections: [
            conn("lyr", 0, "gen", 0),
            conn("sty", 0, "gen", 1),
            conn("gen", 0, "out", 0),
        ],
    };
}
/** 片段重绘：源音频 + 新风格提示词 → songRepaint → 结果。起止秒在节点面板里调。 */
export function buildSegRepaintWorkflow(_ctx) {
    return {
        nodes: [
            node("input", "input", 0, 0, {}),
            node("sty", "songPrompt", 0, 180, {}),
            node("rep", "songRepaint", 320, 40, {}),
            node("out", "output", 660, 40, { laneLabel: "片段重绘" }),
        ],
        connections: [
            conn("input", 0, "rep", 0),
            conn("sty", 0, "rep", 1),
            conn("rep", 0, "out", 0),
        ],
    };
}
/** 人声转伴奏：源人声 → songComplete（人声转伴奏 + 补全声部）→ 混音 + 各声部分轨。 */
export function buildVocal2AccWorkflow(_ctx) {
    return {
        nodes: [
            node("input", "input", 0, 0, {}),
            node("comp", "songComplete", 320, 0, {}),
            node("out", "output", 660, 0, { laneLabel: "人声转伴奏" }),
        ],
        connections: [
            conn("input", 0, "comp", 0),
            conn("comp", 0, "out", 0),
            conn("comp", 1, "out", 1),
            conn("comp", 2, "out", 2),
            conn("comp", 3, "out", 3),
        ],
    };
}
/** songStems 版原创重建：源音频 → demucs 四分轨（人声/鼓/贝斯/其他）→ 各自落轨，重建起点。 */
export function buildStemsRebuildWorkflow(_ctx) {
    return {
        nodes: [
            node("input", "input", 0, 0, {}),
            node("st", "songStems", 320, 0, {}),
            node("out", "output", 660, 0, { laneLabel: "分轨重建" }),
        ],
        connections: [
            conn("input", 0, "st", 0),
            conn("st", 0, "out", 0),
            conn("st", 1, "out", 1),
            conn("st", 2, "out", 2),
            conn("st", 3, "out", 3),
        ],
    };
}
