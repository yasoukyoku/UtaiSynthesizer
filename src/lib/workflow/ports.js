/**
 * 节点端口类型表（规划 6-1）—— 每个节点类型的入/出口数据类型声明。
 * Record<WorkflowNodeType, …> 穷举：新增节点漏配直接编译报错。
 *
 * 用途（规划 6-2）：isValidConnection 在连线瞬间做类型守卫 ——
 * 同型相连 / 任意端 any 放行，其余（如 audio→midi、report→audio）直接拒绝，
 * 从源头杜绝「报告 JSON 接进音频链」这类哑线。
 *
 * 口径说明（与引擎/组件逐一核对过）：
 *   - output 节点入口为 any：转写模板把 amtMidi 的 midi 口接进 output 存档。
 *   - autoArrange/harmonizer 入口为 any：引擎同时接受 midi 文件 / 和弦 JSON / 音符数组。
 *   - msstSeparation/songExtract 声明 5/4 个 audio 出口，更多分轨（6-stem 模型等）经
 *     越界回退 any 放行；split/merge 动态端口同理。
 *   - reharmonize/structureEdit 有 2 个 midi 出口（原件 + 改件并存）。
 */
const AUDIO = "audio";
const MIDI = "midi";
const CHORDS = "chords";
const LYRICS = "lyrics";
const REPORT = "report";
const ANY = "any";
const audio = (n) => Array.from({ length: n }, () => AUDIO);
const midi = (n) => Array.from({ length: n }, () => MIDI);
export const NODE_PORTS = {
    // I/O 与母带链
    input: { inputs: [], outputs: [AUDIO] },
    output: { inputs: [ANY], outputs: [] },
    rvc: { inputs: [AUDIO], outputs: [AUDIO] },
    sovits: { inputs: [AUDIO], outputs: [AUDIO] },
    transpose: { inputs: [AUDIO], outputs: [AUDIO] },
    msstSeparation: { inputs: [AUDIO], outputs: audio(5) },
    split: { inputs: [AUDIO], outputs: audio(2) },
    merge: { inputs: audio(4), outputs: [AUDIO] },
    complianceCheck: { inputs: [AUDIO], outputs: [AUDIO, REPORT] },
    lufsNormalize: { inputs: [AUDIO], outputs: [AUDIO] },
    dither: { inputs: [AUDIO], outputs: [AUDIO] },
    busEq: { inputs: [AUDIO], outputs: [AUDIO] },
    stereoWidth: { inputs: [AUDIO], outputs: [AUDIO] },
    saturate: { inputs: [AUDIO], outputs: [AUDIO] },
    phaseRotate: { inputs: [AUDIO], outputs: [AUDIO] },
    dcRemove: { inputs: [AUDIO], outputs: [AUDIO] },
    // Phase 5 分析可视化：透传音频 + 报告旁路端口
    spectrogram: { inputs: [AUDIO], outputs: [AUDIO, REPORT, REPORT] },
    f0Curve: { inputs: [AUDIO], outputs: [AUDIO, REPORT] },
    timbreMetrics: { inputs: [AUDIO], outputs: [AUDIO, REPORT] },
    harmonicityCheck: { inputs: [AUDIO], outputs: [AUDIO, REPORT] },
    spectralCompare: { inputs: [AUDIO, AUDIO], outputs: [AUDIO, REPORT] },
    dtwAlign: { inputs: [AUDIO, AUDIO], outputs: [AUDIO, REPORT] },
    abCompare: { inputs: [AUDIO, AUDIO], outputs: [AUDIO, AUDIO] },
    lufsAnalyze: { inputs: [AUDIO], outputs: [AUDIO, REPORT] },
    // 符号域 / 自动编曲
    amtMidi: { inputs: [AUDIO], outputs: [MIDI, REPORT] },
    speedShift: { inputs: [AUDIO], outputs: [AUDIO] },
    chordDetect: { inputs: [MIDI], outputs: [CHORDS, REPORT] },
    // 11 条编曲分轨（引擎 tracks 表顺序）+ 末端口和弦摘要。曾声明 5 —— 引擎写 12 个端口、
    // 卡片只画 5 个把手，后 7 路（含扩充乐器与和弦摘要）根本连不出去，越界回退 any 还让
    // 仅存的连线绕过了类型守卫。
    autoArrange: { inputs: [ANY], outputs: [...midi(11), CHORDS] },
    deepOriginal: { inputs: [AUDIO], outputs: audio(5) },
    midiFileIn: { inputs: [], outputs: [MIDI] },
    chordBlockIn: { inputs: [], outputs: [CHORDS] },
    // 符号域 → 声音域：入口收 MIDI 文件路径或音符 JSON，出口是渲染好的 WAV。
    soundfontRender: { inputs: [MIDI], outputs: [AUDIO] },
    // 原创性闸门。口径跟 spectralCompare/dtwAlign 一族对齐：A(端口 0)是链路里正在走的那一路，
    // 原样透传出去，B(端口 1)是拿来比对的参考。只出报告不透传的话，它就只能挂在链路末端当死角。
    melodySimilarity: { inputs: [MIDI, MIDI], outputs: [MIDI, REPORT] },
    harmonizer: { inputs: [ANY], outputs: [MIDI, MIDI] },
    melodyGen: { inputs: [CHORDS], outputs: [MIDI, MIDI] },
    // Song Studio
    songGenYue2: { inputs: [ANY], outputs: [AUDIO] },
    songGenAceStep: { inputs: [ANY], outputs: [AUDIO] },
    songLyrics: { inputs: [], outputs: [LYRICS] },
    songPrompt: { inputs: [], outputs: [LYRICS] },
    songGen: { inputs: [LYRICS, LYRICS, AUDIO], outputs: [AUDIO] },
    songCover: { inputs: [AUDIO, ANY, LYRICS], outputs: [AUDIO] },
    songRepaint: { inputs: [AUDIO, LYRICS], outputs: [AUDIO] },
    songComplete: { inputs: [AUDIO, LYRICS], outputs: audio(4) },
    songExtract: { inputs: [AUDIO], outputs: audio(4) },
    songLego: { inputs: [AUDIO, LYRICS], outputs: [AUDIO] },
    songStems: { inputs: [AUDIO], outputs: audio(4) },
    // 两个输出口：0=ABC 文本、1=MIDI。少声明一个口不会报错——outputPortKind 越界回退 any，
    // 于是 MIDI 口曾能接进任何输入而绕过连线校验（与 autoArrange 的 5/12 口漏声明同一类）。
    songSheet: { inputs: [AUDIO], outputs: [REPORT, MIDI] },
    // P2 符号域原创化
    midiHumanize: { inputs: [MIDI], outputs: [MIDI] },
    velocityCurve: { inputs: [MIDI], outputs: [MIDI] },
    swingQuantize: { inputs: [MIDI], outputs: [MIDI] },
    melodyReharm: { inputs: [MIDI], outputs: [MIDI] },
    rhythmRestructure: { inputs: [MIDI], outputs: [MIDI] },
    contourMorph: { inputs: [MIDI], outputs: [MIDI] },
    motifDevelop: { inputs: [MIDI], outputs: [MIDI] },
    reharmonize: { inputs: [MIDI], outputs: [MIDI, MIDI] },
    rhythmVariation: { inputs: [MIDI], outputs: [MIDI] },
    structureEdit: { inputs: [MIDI], outputs: [MIDI, MIDI] },
    breathPlanner: { inputs: [MIDI], outputs: [MIDI, REPORT] },
};
/** 取某入端口类型；越界（动态端口）回退 any。 */
export function inputPortKind(nodeType, port) {
    return NODE_PORTS[nodeType]?.inputs[port] ?? ANY;
}
/** 取某出端口类型；越界（可变分轨数）回退 any。 */
export function outputPortKind(nodeType, port) {
    return NODE_PORTS[nodeType]?.outputs[port] ?? ANY;
}
/** 类型相容：同型或任一端为 any。 */
export function portsCompatible(from, to) {
    return from === ANY || to === ANY || from === to;
}
/** 连线合法性（规划 6-2）：isValidConnection 的域判定核心。 */
export function canConnect(fromType, fromPort, toType, toPort) {
    return portsCompatible(outputPortKind(fromType, fromPort), inputPortKind(toType, toPort));
}
