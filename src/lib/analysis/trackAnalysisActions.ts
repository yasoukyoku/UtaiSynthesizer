/**
 * Muno 阶段3「轨道分析动作」——轨道右键菜单的两个入口：
 *
 *   · 识别和弦（analyzeTrackChords）：该轨全部 notes 段 → analyzeChords（确定性、纯前端）
 *     → 和弦轨标签行（chordTrack store，会话态覆盖层，不进 .usp）。
 *   · 识别鼓点（detectTrackDrums）：该轨音频源 → 解码（loadAudioBuffer，带缓存去重）
 *     → detectDrums（三频段能量包络法，无模型秒出）→ GM 鼓音符铺到新建乐器轨
 *     「Drums · 源轨名」（含英文 drums → VocalEditor 自动进鼓件名形态）。
 *
 * 音频对齐：鼓点秒相对源文件起点；audioClip segment 的 offsetMs 是窗口起点，
 * tick = seg.startTick + (hit − offsetMs) 的拍换算，窗口外的点丢弃。
 * 变速工程用全局 tempo 标量（drumHitsToNotes 的既有语义，快速模式精度足够）。
 */
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useChordTrackStore } from "../../store/chordTrack";
import { blankTrack } from "../trackFactory";
import { TICKS_PER_BEAT } from "../constants";
import { analyzeChords, type ChordAnalysisNote } from "./chordAnalysis";
import { detectDrums, drumHitsToNotes } from "./drumDetection";
import { loadAudioBuffer } from "../audio/playback";
import i18n from "../../i18n";

/** 识别该轨和弦并落到和弦轨。任何失败/空输入都以 toast 反馈（中文优先）。 */
export function analyzeTrackChords(trackId: string): void {
  const proj = useProjectStore.getState();
  const track = proj.tracks.find((t) => t.id === trackId);
  if (!track) return;

  // 收集全轨音符（绝对 tick = segment 起点 + 段内相对 tick）。
  const notes: ChordAnalysisNote[] = [];
  for (const seg of track.segments) {
    if (seg.content.type !== "notes") continue;
    for (const n of seg.content.notes) {
      notes.push({ tick: seg.startTick + n.tick, duration: n.duration, pitch: n.pitch, velocity: n.velocity });
    }
  }
  if (notes.length === 0) {
    useAppStore.getState().showToast(i18n.t("chordTrack.noNotes"), "info");
    return;
  }

  const beatsPerBar = proj.timeSignature[0] || 4;
  const res = analyzeChords(notes, TICKS_PER_BEAT, beatsPerBar);
  useChordTrackStore.getState().setAnalysis({
    sourceTrackId: track.id,
    sourceTrackName: track.name,
    segments: res.segments,
    key: res.key,
  });
  useAppStore.getState().showToast(
    i18n.t("chordTrack.done", { count: res.segments.length, key: res.key.label }),
    "success",
  );
}

/**
 * 识别该轨音频的鼓点并生成 GM 鼓音符轨。
 * @param audioPath 源音频文件（resolveTrackSourceAudio 解出的可播放路径）。
 */
export async function detectTrackDrums(trackId: string, audioPath: string): Promise<void> {
  const proj = useProjectStore.getState();
  const track = proj.tracks.find((t) => t.id === trackId);
  if (!track || !audioPath) return;

  try {
    const buf = await loadAudioBuffer(audioPath);
    const channels: Float32Array[] = [];
    for (let c = 0; c < buf.numberOfChannels; c++) channels.push(buf.getChannelData(c));
    const hits = detectDrums(channels, { sampleRate: buf.sampleRate });
    if (hits.length === 0) {
      useAppStore.getState().showToast(i18n.t("chordTrack.drumsNone"), "info");
      return;
    }

    // 对齐基准：第一个 audioClip 段的（startTick, offsetMs）—— 鼓点秒相对源文件起点。
    const clipSeg = track.segments.find((s) => s.content.type === "audioClip");
    const segStartTick = clipSeg?.startTick ?? 0;
    const offsetMs = clipSeg?.content.type === "audioClip" ? clipSeg.content.offsetMs : 0;
    const secPerTick = 60 / (proj.tempo * TICKS_PER_BEAT);
    const offsetTicks = offsetMs > 0 ? Math.round((offsetMs / 1000) / secPerTick) : 0;

    // drumHitsToNotes 从 0 起算 tick；这里平移到时间线绝对域并丢窗口外的点。
    const raw = drumHitsToNotes(hits, proj.tempo, TICKS_PER_BEAT);
    const notes = raw
      .map((n) => ({ ...n, tick: segStartTick + n.tick - offsetTicks }))
      .filter((n) => n.tick >= 0)
      .map((n) => ({
        id: crypto.randomUUID(),
        tick: n.tick,
        duration: n.duration,
        pitch: n.pitch,
        lyric: "",
        velocity: n.velocity,
      }));
    if (notes.length === 0) {
      useAppStore.getState().showToast(i18n.t("chordTrack.drumsNone"), "info");
      return;
    }

    let maxTick = 0;
    for (const n of notes) maxTick = Math.max(maxTick, n.tick + n.duration);
    // 「Drums · …」固定含英文 drums —— isDrumLikeName 只认英文，VocalEditor 靠它切鼓件名形态。
    const drumTrack = blankTrack(crypto.randomUUID(), `Drums · ${track.name}`, "instrument");
    drumTrack.expanded = false;
    drumTrack.segments = [
      {
        id: crypto.randomUUID(),
        startTick: 0,
        durationTicks: Math.max(1, maxTick),
        content: { type: "notes", notes },
      },
    ];
    const idx = proj.tracks.findIndex((t) => t.id === trackId);
    useProjectStore.getState().addTrack(drumTrack, idx >= 0 ? idx + 1 : undefined);
    useAppStore.getState().showToast(
      i18n.t("chordTrack.drumsDone", { count: notes.length, track: drumTrack.name }),
      "success",
    );
  } catch (e) {
    useAppStore.getState().showToast(
      i18n.t("chordTrack.drumsFailed", { msg: e instanceof Error ? e.message : String(e) }),
      "error",
    );
  }
}
