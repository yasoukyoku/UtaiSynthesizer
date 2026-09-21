/**
 * Muno 阶段4「自动编曲」动作层 —— 轨道右键 / AI 菜单的「自动编曲…」。
 *
 * 旋律轨 → arrange(风格模板引擎) → 在源轨下方插入 4 条 instrument 轨
 * (Drums / Bass / Piano / Pad),全部包进 ONE undo 事务(一次撤销整组移除),
 * 并按 defaultSoundfontFor 自动分配默认音色(没匹配到 → 不挂音色,
 * 轨道头显示占位 + toast 提示先导入;不打断流程)。
 *
 * 鼓轨名固定含英文 "Drums" —— isDrumLikeName 靠它切 channel 10 + 编辑器鼓件名形态
 * (与鼓点识别的命名约定一致)。
 */
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { useChordTrackStore } from "../../store/chordTrack";
import {
  useSoundfontStore,
  defaultSoundfontFor,
  firstPresetOf,
  pickInstrumentSoundfont,
  GM_PROGRAM,
  type ArrangeTrackKind,
  type Soundfont,
} from "../../store/soundfont";
import { blankTrack } from "../trackFactory";
import i18n from "../../i18n";
import type { Note } from "../../types/project";
import { invoke } from "@tauri-apps/api/core";
import { getPreviewContext, loadAudioBuffer } from "../audio/playback";
import { TICKS_PER_BEAT } from "../constants";
import { arrange, type ArrangeOutput } from "./arranger";
import { analyzeChords } from "../analysis/chordAnalysis";
import { generateChordMidi, type ChordMidiOptions, type ChordMidiStyle } from "./chordMidi";
import type { ArrangeMood, ArrangeStyle } from "./styles";

/** 收集整轨全部 notes 段的音符(绝对 tick)。 */
function collectTrackNotes(trackId: string): { notes: Array<{ tick: number; duration: number; pitch: number; velocity: number }> } | null {
  const track = useProjectStore.getState().tracks.find((t) => t.id === trackId);
  if (!track) return null;
  const notes: Array<{ tick: number; duration: number; pitch: number; velocity: number }> = [];
  for (const seg of track.segments) {
    if (seg.content.type !== "notes") continue;
    for (const n of seg.content.notes) {
      notes.push({ tick: seg.startTick + n.tick, duration: n.duration, pitch: n.pitch, velocity: n.velocity });
    }
  }
  return { notes };
}

/** 风格/情绪的中文名(轨道命名用;i18n 不进轨道名 —— 工程文件语言无关)。 */
const STYLE_ZH: Record<ArrangeStyle, string> = {
  pop: "流行", rock: "摇滚", jazz: "爵士", edm: "电子", latin: "拉丁",
  funk: "放克", rnb: "R&B", ballad: "抒情", country: "乡村",
  lofi: "Lo-Fi", trap: "Trap", reggae: "雷鬼", bossanova: "Bossa Nova",
  kpop: "K-Pop", chinese: "中国风", ambient: "氛围", techno: "Techno",
  folk: "民谣", citypop: "都市", hiphop: "嘻哈", house: "浩室", cinematic: "史诗",
};

/** 为一个编曲部件挑选可用音源:鼓走专用鼓音源 / GM 鼓组(bank128),其余走「专用 → GM 综合兜底」。
 *  返回挂轨/render_soundfont_notes 用的 font/preset;无任何可用音源时返回 undefined。 */
function pickPartFont(sfKind: ArrangeTrackKind, fonts: Soundfont[]) {
  if (sfKind === "drums") {
    const drumFont = defaultSoundfontFor("drums", fonts);
    if (drumFont) {
      const preset = firstPresetOf(drumFont);
      if (preset) return { fontId: drumFont.id, presetId: preset.id, presetName: preset.name };
    }
    const gm = fonts.find((f) => f.format === "sf2" && f.presets.some((p) => p.id === "128:0"));
    const dp = gm?.presets.find((p) => p.id === "128:0");
    if (gm && dp) return { fontId: gm.id, presetId: dp.id, presetName: dp.name };
    return undefined;
  }
  const got = pickInstrumentSoundfont(sfKind, fonts);
  return got ? { fontId: got.font.id, presetId: got.preset.id, presetName: got.preset.name } : undefined;
}

/** 11 类部件 → 音源种类(落轨/试听共用同一张映射)。 */
const PART_SF_KIND: Record<ArrangePartKind, ArrangeTrackKind> = {
  drums: "drums", bass: "bass", piano: "piano",
  guitarArp: "guitarArp", guitarStrum: "guitarStrum", epiano: "epiano",
  strings: "strings", chords: "chords", synthPad: "synthPad",
  pluck: "pluck", melody: "lead",
};

/** 从 arrange 结果取某部件的音符数组。 */
function partNotes(kind: ArrangePartKind, res: ArrangeOutput): Note[] {
  switch (kind) {
    case "drums": return res.drums;
    case "bass": return res.bass;
    case "piano": return res.piano;
    case "guitarArp": return res.guitarArp;
    case "guitarStrum": return res.guitarStrum;
    case "epiano": return res.epiano;
    case "strings": return res.strings;
    case "chords": return res.pad;
    case "synthPad": return res.synthPad;
    case "pluck": return res.pluck;
    case "melody": return res.melody;
  }
}

/** 自动编曲伴奏轨名判定:兼容 "🥁 Drums · 流行" 这种「emoji + 空格 + 英文词 + · + 风格」。
 *  开头的非单词字符(emoji/空格)跳过,随后必须是固定英文部件名 + " · "(SynthPad 先于 Pad 匹配)。 */
const AUTO_ARRANGE_RE = /^[^\w]*(Drums|Bass|Piano|Guitar|Strum|E\.Piano|Strings|SynthPad|Pad|Pluck|Lead) · /;
function isAutoArrangeTrackName(name: string): boolean {
  return AUTO_ARRANGE_RE.test(name);
}

/** 扫描源轨正下方连续紧邻的自动编曲伴奏轨 id(遇到第一条非伴奏轨即停)。
 *  识别规则: 新标记 `autoArrangeNodeId = autoArrange:<srcId>` 优先;
 *  旧命名正则 (emoji + English part · style) 作兜底 — 兼容升级前的工程。 */
function collectStaleArrangeIds(trackId: string): string[] {
  const proj = useProjectStore.getState();
  const at = proj.tracks.findIndex((t) => t.id === trackId);
  if (at < 0) return [];
  const groupMarker = `autoArrange:${trackId}`;
  const ids: string[] = [];
  for (let i = at + 1; i < proj.tracks.length; i++) {
    const cand = proj.tracks[i]!;
    if (cand.trackType !== "instrument") break;
    const isAuto = cand.autoArrangeNodeId === groupMarker || isAutoArrangeTrackName(cand.name);
    if (!isAuto) break;
    ids.push(cand.id);
  }
  return ids;
}

/** 一键清空:删除源轨正下方整组自动编曲伴奏轨(ONE undo 事务)。@returns 删除条数。 */
export function clearAutoArrange(trackId: string): number {
  const ids = collectStaleArrangeIds(trackId);
  if (ids.length === 0) return 0;
  const history = useHistoryStore.getState();
  history.beginTransaction();
  try {
    for (const id of [...ids].reverse()) useProjectStore.getState().removeTrack(id);
  } finally {
    history.commitTransaction();
  }
  useAppStore.getState().showToast(`🗑 已清空 ${ids.length} 条伴奏轨`, "info");
  return ids.length;
}

/** 歌曲身份证(实时分析,纯前端确定性):BPM / 调 / 拍号 / 和弦进行。面板实时展示用。 */
export interface ArrangeProfile {
  bpm: number;
  beatsPerBar: number;
  /** 拍号分母(4=四分音符为一拍,8=八分音符为一拍),来自工程拍号,不写死。 */
  beatUnit: number;
  keyLabel: string;
  /** 和弦 label 序列(按小节去重压缩,便于一行展示)。 */
  chordLabels: string[];
  noteCount: number;
  bars: number;
}

/** 实时计算源轨的「歌曲身份证」。无音符返回 null(UI 显示空态)。 */
export function analyzeSourceProfile(trackId: string): ArrangeProfile | null {
  const proj = useProjectStore.getState();
  const collected = collectTrackNotes(trackId);
  if (!collected || collected.notes.length === 0) return null;
  const beatsPerBar = proj.timeSignature[0] || 4;
  const beatUnit = proj.timeSignature[1] || 4;
  const analysis = analyzeChords(collected.notes, TICKS_PER_BEAT, beatsPerBar);
  const barTicks = beatsPerBar * TICKS_PER_BEAT;
  // 按小节压缩和弦:每小节取该小节起点命中的和弦 label,相邻重复去重。
  let minTick = Infinity, maxTick = 0;
  for (const n of collected.notes) {
    if (n.tick < minTick) minTick = n.tick;
    const e = n.tick + n.duration;
    if (e > maxTick) maxTick = e;
  }
  const startTick = Math.floor(minTick / barTicks) * barTicks;
  const bars = Math.max(1, Math.ceil((maxTick - startTick) / barTicks));
  const labels: string[] = [];
  for (let b = 0; b < bars; b++) {
    const t = startTick + b * barTicks;
    const seg = analysis.segments.filter((s) => s.startTick <= t).pop();
    const label = seg ? seg.label : analysis.key.label;
    if (labels[labels.length - 1] !== label) labels.push(label);
  }
  return {
    bpm: Math.round(proj.tempo * 10) / 10,
    beatsPerBar,
    beatUnit,
    keyLabel: analysis.key.label,
    chordLabels: labels.slice(0, 16), // 一行最多展示 16 个和弦,超出由 UI 提示
    noteCount: collected.notes.length,
    bars,
  };
}

/**
 * 对一条旋律轨跑自动编曲并落轨。
 * @returns 成功与否(失败已 toast,调用层无需再报)。
 */

/** 自动编曲可生成的 11 类部件(UI 勾选顺序的数据源)。 */
export type ArrangePartKind =
  | "drums" | "bass"
  | "piano" | "guitarArp" | "guitarStrum" | "epiano"
  | "strings" | "chords" | "synthPad"
  | "pluck" | "melody";

/** 可配置的自动编曲选项 —— 乐器勾选 + 自动旋律/和弦轨。 */
export interface ArrangeOptions {
  /** 勾选哪些乐器轨要生成. 默认 ["drums","bass","piano","chords"]. */
  instruments?: ArrangePartKind[];
  /** 没旋律时, 是否也生成一条 AI 自动旋律轨 (lead). 默认 true. */
  generateMelodyIfEmpty?: boolean;
  /** 额外生成一条和弦 MDI 轨 (只有 chord block, 方便用户看). 默认 false. */
  generateChordTrack?: boolean;
  /** 多随机 seed (0=确定性, 非 0=换一条旋律变体). */
  melodySeed?: number;
  /** 一键扒带传入的外部调性(优先于自估),保证与干音同调。 */
  externalKey?: import("../analysis/chordAnalysis").KeyEstimate;
  /** 一键扒带传入的外部和弦段(优先于自估),保证和声严格贴合干音。 */
  externalChords?: import("../analysis/chordAnalysis").ChordSegment[];
}

/**
 * 对一条旋律轨跑自动编曲并落轨。
 * @returns 成功与否(失败已 toast,调用层无需再报)。
 */
export async function runAutoArrange(trackId: string, style: ArrangeStyle, mood: ArrangeMood, opts: ArrangeOptions = {}): Promise<boolean> {
  const proj = useProjectStore.getState();
  const track = proj.tracks.find((t) => t.id === trackId);
  if (!track) return false;

  const collected = collectTrackNotes(trackId);
  if (!collected || collected.notes.length === 0) {
    useAppStore.getState().showToast(i18n.t("arrange.noNotes"), "info");
    return false;
  }

  const res = arrange({
    notes: collected.notes,
    tempo: proj.tempo,
    timeSignature: proj.timeSignature,
    style,
    mood,
    externalKey: opts.externalKey,
    externalChords: opts.externalChords,
  });
  if (!res || res.bars === 0) {
    useAppStore.getState().showToast(i18n.t("arrange.noNotes"), "info");
    return false;
  }

  // 音色自动分配:已导入的音源里按关键字匹配(阶段1 的推荐表规则)。
  try {
    await useSoundfontStore.getState().refresh();
  } catch {
    // 列不出音源 → 全部轨道不挂音色,走占位提示,不阻塞编曲落轨。
  }
  const fonts = useSoundfontStore.getState().fonts;
  // 为一个编曲部件选音色(专用音源 → GM 综合兜底),逻辑与试听完全一致。
  const pick = (sfKind: ArrangeTrackKind) => pickPartFont(sfKind, fonts);

  // ── 可选乐器动态组装: 11 类乐器自由组合, 用户勾几个出几个 ──
  // 顺序固定 = 落轨后的显示顺序(节奏组→和声组→铺底组→旋律组)。
  type PartKind = ArrangePartKind;
  type Part = { kind: PartKind; name: string; sfKind: ArrangeTrackKind; notes: Note[] };
  const selectedInstruments: PartKind[] = opts.instruments ?? ["drums", "bass", "piano", "chords"];
  const partsIndex: Record<PartKind, Omit<Part, "kind">> = {
    drums:       { name: `🥁 Drums · ${STYLE_ZH[style]}`,       sfKind: "drums",       notes: res.drums },
    bass:        { name: `🎸 Bass · ${STYLE_ZH[style]}`,        sfKind: "bass",        notes: res.bass },
    piano:       { name: `🎹 Piano · ${STYLE_ZH[style]}`,       sfKind: "piano",       notes: res.piano },
    guitarArp:   { name: `🪕 Guitar · ${STYLE_ZH[style]}`,      sfKind: "guitarArp",   notes: res.guitarArp },
    guitarStrum: { name: `🎶 Strum · ${STYLE_ZH[style]}`,       sfKind: "guitarStrum", notes: res.guitarStrum },
    epiano:      { name: `🎼 E.Piano · ${STYLE_ZH[style]}`,     sfKind: "epiano",      notes: res.epiano },
    strings:     { name: `🎻 Strings · ${STYLE_ZH[style]}`,     sfKind: "strings",     notes: res.strings },
    chords:      { name: `🪟 Pad · ${STYLE_ZH[style]}`,         sfKind: "chords",      notes: res.pad },
    synthPad:    { name: `🎛 SynthPad · ${STYLE_ZH[style]}`,    sfKind: "synthPad",    notes: res.synthPad },
    pluck:       { name: `💠 Pluck · ${STYLE_ZH[style]}`,       sfKind: "pluck",       notes: res.pluck },
    melody:      { name: `✨ Lead · ${STYLE_ZH[style]}`,        sfKind: "lead",        notes: res.melody },
  };
  const parts: Part[] = selectedInstruments
    .filter((k) => partsIndex[k] && partsIndex[k].notes.length > 0)
    .map((k) => ({ kind: k, ...partsIndex[k] }));

  const history = useHistoryStore.getState();
  history.beginTransaction();
  try {
    const insertIdx = proj.tracks.findIndex((t) => t.id === trackId);
    let at = insertIdx >= 0 ? insertIdx + 1 : proj.tracks.length;

    // ── 一键替换: 如果源轨下面紧邻着上一轮自动编曲的伴奏轨,先整组删掉再插回原位
    //    (用户再生成 → 旧的自动消失)。扫描规则与「一键清空」共用 collectStaleArrangeIds。 ──
    const staleIds = collectStaleArrangeIds(trackId);
    if (staleIds.length > 0) {
      for (const id of [...staleIds].reverse()) useProjectStore.getState().removeTrack(id);
      useAppStore.getState().showToast(`♻️ 已替换 ${staleIds.length} 条旧伴奏轨`, "info");
      // 删完后 at 保持不变 — 我们就是要插回原来的位置
    }

    for (const part of parts) {
      if (part.notes.length === 0) continue; // 空轨不建(如 ballad 的稀疏件)
      const notes = part.notes.map((n) => ({
        ...n, tick: n.tick - res.startTick,
      }));
      let maxTick = 0;
      for (const n of notes) maxTick = Math.max(maxTick, n.tick + n.duration);
      const t = blankTrack(crypto.randomUUID(), part.name, "instrument");
      t.expanded = false;
      // 🎯 OPT3: 标记为自动编曲轨 — TrackList 按此分组渲染浮动机头工具条 (整组换风格/单轨换音色/Solo)
      // 用一个稳定前缀 + source trackId, 便于 collectStaleArrangeIds 同时支持新标记/旧命名正则两种方式
      t.autoArrangeNodeId = `autoArrange:${trackId}`;
      t.segments = [{
        id: crypto.randomUUID(),
        startTick: res.startTick,
        durationTicks: Math.max(1, maxTick),
        content: { type: "notes", notes },
      }];
      const sf = pick(part.sfKind);
      if (sf) t.soundfont = sf;
      useProjectStore.getState().addTrack(t, at);
      at += 1;
    }
  } finally {
    history.commitTransaction();
  }

  const created = parts.filter((p) => p.notes.length > 0).length;
  const missingSf = parts.filter((p) => p.notes.length > 0 && !pick(p.sfKind)).length;
  useAppStore.getState().showToast(
    i18n.t("arrange.done", { count: created, bars: res.bars, style: STYLE_ZH[style], key: res.key.label }),
    "success",
  );
  if (missingSf > 0) {
    useAppStore.getState().showToast(i18n.t("arrange.noFontHint", { count: missingSf }), "info");
  }
  return true;
}

/** AI 菜单用的目标轨解析:活跃轨 → 选中段所在轨 → 第一条有音符的轨。 */
export function resolveMelodyTrackId(): string | null {
  const proj = useProjectStore.getState();
  const app = useAppStore.getState();
  const hasNotes = (id: string) => {
    const t = proj.tracks.find((tr) => tr.id === id);
    return !!t && t.segments.some((sg) => sg.content.type === "notes" && sg.content.notes.length > 0);
  };
  if (app.activeTrackId && hasNotes(app.activeTrackId)) return app.activeTrackId;
  const selTrack = app.selectedSegment?.trackId;
  if (selTrack && hasNotes(selTrack)) return selTrack;
  return proj.tracks.find((tr) => tr.segments.some((sg) => sg.content.type === "notes" && sg.content.notes.length > 0))?.id ?? null;
}

/** 和弦风格的中文名(轨道命名用;i18n 不进轨道名 —— 工程文件语言无关)。 */
const CHORD_MIDI_STYLE_ZH: Record<ChordMidiStyle, string> = {
  POP_STANDARD: "标准流行", POP_COMPLEX: "丰富流行", DARK: "暗黑", RANDB: "随机变化", NOCONSTRAINT: "自由分析",
};

/**
 * Muno 阶段5「生成和弦 MIDI(配和声)」动作层 —— AI 菜单 / 轨道右键的「生成和弦 MIDI…」。
 *
 * 旋律轨 → generateChordMidi(和声化引擎) → 源轨下方插入 1 条 instrument 和弦轨
 * (块状 voicing,音色按 chords 类自动分配),并把和声化后的和弦进行同步写进
 * 顶部和弦行(chordTrack store,会话态覆盖层)——规划里的「在时间线上方创建和弦轨道」。
 * 建轨包进 ONE undo 事务;失败已 toast。
 * @returns 成功与否(调用层无需再报)。
 */
export async function runChordMidi(trackId: string, opts: ChordMidiOptions): Promise<boolean> {
  const proj = useProjectStore.getState();
  const track = proj.tracks.find((t) => t.id === trackId);
  if (!track) return false;

  const collected = collectTrackNotes(trackId);
  if (!collected || collected.notes.length === 0) {
    useAppStore.getState().showToast(i18n.t("chordMidi.noNotes"), "info");
    return false;
  }

  const res = generateChordMidi({
    notes: collected.notes,
    timeSignature: proj.timeSignature,
    ...opts,
  });
  if (!res || res.segments.length === 0) return false;

  // 音色自动分配:chords 类(弦乐/铺底);列不出音源 → 不挂音色,占位提示,不阻塞落轨。
  try {
    await useSoundfontStore.getState().refresh();
  } catch {
    // 同 runAutoArrange:刷新失败不打断。
  }
  const fonts = useSoundfontStore.getState().fonts;
  const font = defaultSoundfontFor("chords", fonts);
  const preset = font ? firstPresetOf(font) : null;

  // 段内相对 tick + 独立 note id(引擎输出的 id 只在引擎内唯一)。
  const notes: Note[] = res.notes.map((n) => ({
    ...n, id: crypto.randomUUID(), tick: n.tick - res.startTick,
  }));
  let maxTick = 0;
  for (const n of notes) maxTick = Math.max(maxTick, n.tick + n.duration);
  const t = blankTrack(crypto.randomUUID(), `Chords · ${CHORD_MIDI_STYLE_ZH[opts.style]}`, "instrument");
  t.expanded = false;
  t.segments = [{
    id: crypto.randomUUID(),
    startTick: res.startTick,
    durationTicks: Math.max(1, maxTick),
    content: { type: "notes", notes },
  }];
  if (font && preset) t.soundfont = { fontId: font.id, presetId: preset.id, presetName: preset.name };

  const history = useHistoryStore.getState();
  history.beginTransaction();
  try {
    const idx = proj.tracks.findIndex((x) => x.id === trackId);
    useProjectStore.getState().addTrack(t, idx >= 0 ? idx + 1 : undefined);
  } finally {
    history.commitTransaction();
  }

  // 和弦行同步:显示和声化后的进行(阶段5 规划的「前端与和弦轨道对接」)。
  useChordTrackStore.getState().setAnalysis({
    sourceTrackId: track.id,
    sourceTrackName: track.name,
    segments: res.segments,
    key: res.key,
  });

  useAppStore.getState().showToast(
    i18n.t("chordMidi.done", { count: res.segments.length, bars: res.bars, key: res.key.label, track: t.name }),
    "success",
  );
  if (!font || !preset) {
    useAppStore.getState().showToast(i18n.t("arrange.noFontHint", { count: 1 }), "info");
  }
  return true;
}

// ── 自动编曲预览(试听前 8 小节)───────────────────────────────────────────
// ArrangeDialog 的「试听」按钮:选中的风格/情绪即时可听 —— 不落轨、不进
// 历史。同一 arrange 引擎只喂旋律前 8 小节(bar 计数随之 ≤ 8),四轨各自
// 渲染 WAV(音色分配与落轨同一张表;没音色 → GM/FluidSynth 试听通道,
// 单轨失败仅静音该轨)后经共享 AudioContext 同时起播。叠播/关弹窗/正式
// 落轨前都要 stopAutoArrangePreview()。

const PREVIEW_BARS = 8;
let previewSources: AudioBufferSourceNode[] = [];
let previewEndedCb: (() => void) | null = null;

/** 停止正在播放的编曲预览(幂等;不触发 onEnded —— 调用方自行更新 UI)。 */
export function stopAutoArrangePreview(): void {
  previewEndedCb = null;
  for (const s of previewSources) {
    try { s.stop(); } catch { /* already ended */ }
  }
  previewSources = [];
}

/** 试听前 8 小节:旋律轨 → arrange → 勾选部件各自 WAV → 同步起播。
 *  @param opts.instruments 勾选的部件(默认鼓/贝斯/钢琴/铺底)。
 *  @param opts.externalKey/externalChords 一键扒带的外部骨架(优先于自估)。
 *  @param opts.onEnded 自然播完回调(主动 stop 不触发)。
 *  @returns 是否开始播放(失败已 toast)。 */
export async function previewAutoArrange(
  trackId: string,
  style: ArrangeStyle,
  mood: ArrangeMood,
  opts?: {
    instruments?: ArrangePartKind[];
    externalKey?: import("../analysis/chordAnalysis").KeyEstimate;
    externalChords?: import("../analysis/chordAnalysis").ChordSegment[];
    onEnded?: () => void;
  },
): Promise<boolean> {
  const onEnded = opts?.onEnded;
  const proj = useProjectStore.getState();
  const track = proj.tracks.find((t) => t.id === trackId);
  if (!track) return false;

  const collected = collectTrackNotes(trackId);
  if (!collected || collected.notes.length === 0) {
    useAppStore.getState().showToast(i18n.t("arrange.noNotes"), "info");
    return false;
  }

  // 只喂前 8 小节的旋律 —— arrange 的 bar 计数随输入长度,输出即预览长度。
  const barTicks = proj.timeSignature[0] * TICKS_PER_BEAT;
  // reduce 求最小值,避免音符极多时 Math.min(...大数组) 的参数栈溢出。
  const firstTick = collected.notes.reduce((m, n) => (n.tick < m ? n.tick : m), Infinity);
  const previewEnd = Math.floor(firstTick / barTicks) * barTicks + PREVIEW_BARS * barTicks;
  const res = arrange({
    notes: collected.notes.filter((n) => n.tick < previewEnd),
    tempo: proj.tempo,
    timeSignature: proj.timeSignature,
    style,
    mood,
    externalKey: opts?.externalKey,
    externalChords: opts?.externalChords,
  });
  if (!res || res.bars === 0) {
    useAppStore.getState().showToast(i18n.t("arrange.noNotes"), "info");
    return false;
  }

  // 音色:已导入音源优先(与落轨同一分配表);没有 → GM/FluidSynth 通道(按 GM program 分音色)。
  try {
    await useSoundfontStore.getState().refresh();
  } catch { /* 同 runAutoArrange:刷新失败不打断 */ }
  const fonts = useSoundfontStore.getState().fonts;
  const msPerTick = 60000 / proj.tempo / TICKS_PER_BEAT;
  const selected = opts?.instruments ?? ["drums", "bass", "piano", "chords"];
  const parts: Array<{ sfKind: ArrangeTrackKind; notes: Note[] }> = selected
    .map((kind) => ({ sfKind: PART_SF_KIND[kind], notes: partNotes(kind, res) }))
    .filter((p) => p.notes.length > 0);

  const ctx = getPreviewContext();
  // 各轨并行渲染;单轨失败(如 FluidSynth 缺失且无音源)→ 该轨静音,其余照播。
  const wavs = await Promise.all(parts.map(async (p) => {
    if (p.notes.length === 0) return null;
    const picked = pickPartFont(p.sfKind, fonts);
    try {
      if (picked) {
        return await invoke<string>("render_soundfont_notes", {
          fontId: picked.fontId,
          presetId: picked.presetId,
          notes: p.notes.map((n) => ({
            start: (n.tick * msPerTick) / 1000,
            dur: (Math.max(1, n.duration) * msPerTick) / 1000,
            key: n.pitch,
            vel: n.velocity,
          })),
          sampleRate: 44100,
        });
      }
      return await invoke<string>("preview_notes_render", {
        bpm: proj.tempo,
        program: GM_PROGRAM[p.sfKind],
        channel: p.sfKind === "drums" ? 9 : 0,
        notes: p.notes.map((n) => ({
          tick: Math.round(n.tick),
          duration: Math.round(n.duration),
          pitch: n.pitch,
          velocity: n.velocity,
        })),
      });
    } catch {
      return null;
    }
  }));

  stopAutoArrangePreview(); // 保险:叠播前先清旧预览
  const started: AudioBufferSourceNode[] = [];
  let longest: AudioBufferSourceNode | null = null;
  let maxDur = -1;
  for (const wav of wavs) {
    if (!wav) continue;
    const buf = await loadAudioBuffer(wav);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start();
    started.push(src);
    // 结束回调要挂在「时长最长」的那轨上,否则较短轨先结束会让 UI 提前复位。
    if (buf.duration > maxDur) { maxDur = buf.duration; longest = src; }
  }
  if (started.length === 0) {
    useAppStore.getState().showToast(i18n.t("arrange.previewFail"), "error");
    return false;
  }
  previewSources = started;
  previewEndedCb = onEnded ?? null;
  (longest ?? started[started.length - 1]!).onended = () => {
    previewSources = [];
    previewEndedCb?.();
    previewEndedCb = null;
  };
  return true;
}



