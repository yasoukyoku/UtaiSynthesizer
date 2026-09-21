// Muno 阶段2:乐器轨离线渲染(soundfont 原生合成 → WAV → processedOutputs 泳道)。
// 与人声 bake(vocalRender.ts)同一模式:签名判脏、Play 前顺序批量、结果作为普通
// ProcessedOutput 泳道 deposit——时间线播放 / 混音导出 / laneControls 全部复用现有机制。
// 渲染由 Rust 原生 SFZ/SF2 合成引擎完成(render_soundfont_notes),前端只做事件打包。
import { invoke } from "@tauri-apps/api/core";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import i18n from "../../i18n";
import { logToBackend } from "../log";
import { contentSig } from "../../store/history";
import { TICKS_PER_BEAT } from "../constants";
import { getPreviewContext, loadAudioBuffer } from "../audio/playback";
/** 乐器轨烘焙泳道的稳定标识(镜像 VOCAL_LANE_ID 的 Output-node 身份模式)。 */
export const INSTRUMENT_LANE_ID = "instrument";
/** tick(段落相对,480 PPQ)→ 秒。tempo ≤ 0 时按 120 兜底(与编辑器预览同一约定)。 */
function tickToSec(tick, tempo) {
    const bpm = tempo > 0 ? tempo : 120;
    return (tick / TICKS_PER_BEAT) * (60 / bpm);
}
/** 烘焙输入签名:音符内容(contentSig 单源,与人声共用,永不分叉)+ 音源选择 + BPM。
 *  presetName 是纯显示冗余,不进签名(改名不触发重渲染)。 */
export function instrumentRenderSig(track, seg, tempo) {
    const sf = track.soundfont;
    const layers = (track.soundfontLayers ?? [])
        .map((l) => `${l.fontId}/${l.presetId}@${l.gain},${l.pan}`)
        .join(";");
    // 3-9 引擎参与签名:换引擎 = 换渲染输出,必须重烘焙(fold-away,absent ≡ builtin)。
    const eng = sf?.backend === "fluidsynth" ? "fluidsynth" : "";
    return `ins:${contentSig(seg.content)}|sf:${sf ? `${sf.fontId}/${sf.presetId}` : ""}|ly:${layers}|bpm:${tempo}|eng:${eng}`;
}
/** 乐器轨段落是否需要(重)烘焙:有音符 + 已选音源,且(无烘焙 | 签名漂移)。
 *  未选音源的轨不判脏(渲染不了——由 Play 流程统一提示)。空音符段有残留烘焙 → 判脏
 *  (批量时清为静默,与人声「清空后残响」同一处置)。 */
export function isInstrumentDirty(track, seg, tempo) {
    if (track.trackType !== "instrument" || seg.content.type !== "notes")
        return false;
    const bake = seg.processedOutputs?.find((o) => o.laneId === INSTRUMENT_LANE_ID && !o.loading);
    if (seg.content.notes.length === 0)
        return !!bake;
    if (!track.soundfont)
        return false;
    if (!bake)
        return true;
    return bake.renderedSig !== instrumentRenderSig(track, seg, tempo);
}
/** 全工程脏乐器段落(实时读)。空 ⇒ Play 零附加延迟。 */
export function collectDirtyInstruments(tempo) {
    const out = [];
    for (const tr of useProjectStore.getState().tracks) {
        if (tr.trackType !== "instrument")
            continue;
        for (const sg of tr.segments) {
            if (isInstrumentDirty(tr, sg, tempo))
                out.push({ trackId: tr.id, segmentId: sg.id });
        }
    }
    return out;
}
/** 有音符但未选音源的乐器轨(Play 提示用;渲染批次之外,只提醒不阻塞播放)。 */
export function instrumentTracksMissingFont() {
    return useProjectStore
        .getState()
        .tracks.filter((tr) => tr.trackType === "instrument" &&
        !tr.soundfont &&
        tr.segments.some((sg) => sg.content.type === "notes" && sg.content.notes.length > 0));
}
/** 渲染一个乐器轨段落并 deposit 为泳道。laneLabel = 音色名(泳道行显示的来源)。 */
export async function renderInstrumentPart(track, seg, tempo) {
    if (seg.content.type !== "notes")
        return;
    const sf = track.soundfont;
    if (!sf)
        throw new Error(i18n.t("soundfont.trackNoneTip"));
    const segRef = () => useProjectStore.getState().tracks.find((t) => t.id === track.id)?.segments.find((s) => s.id === seg.id);
    const prevOutputs = segRef()?.processedOutputs; // 现有烘焙保留;失败时还原
    const deposit = (outs) => useProjectStore.getState().replaceProcessedOutputs(track.id, seg.id, outs ?? []);
    const laneLabel = sf.presetName || sf.fontId;
    // 首次渲染(无烘焙)→ loading 占位;重渲染 → 旧烘焙继续响到新 stem 落地(镜像人声)。
    if (!prevOutputs || prevOutputs.length === 0) {
        deposit([
            {
                laneId: INSTRUMENT_LANE_ID,
                laneLabel,
                group: laneLabel,
                audioPath: "",
                totalDurationMs: 0,
                waveformPeaks: [],
                outputNodeId: INSTRUMENT_LANE_ID,
                loading: true,
            },
        ]);
    }
    try {
        const notes = seg.content.notes.map((n) => ({
            start: tickToSec(n.tick, tempo),
            dur: tickToSec(n.duration, tempo),
            key: n.pitch,
            vel: n.velocity ?? 100,
        }));
        const wavPath = await invoke("render_soundfont_notes", {
            fontId: sf.fontId,
            presetId: sf.presetId,
            notes,
            layers: track.soundfontLayers ?? [],
            sampleRate: 44100,
            backend: sf.backend, // 3-9:undefined → None → 内置后端
        });
        const info = await invoke("load_audio_file", { path: wavPath });
        if (segRef()) {
            deposit([
                {
                    laneId: INSTRUMENT_LANE_ID,
                    laneLabel,
                    group: laneLabel,
                    audioPath: wavPath,
                    totalDurationMs: info.duration_ms,
                    waveformPeaks: info.peaks,
                    outputNodeId: INSTRUMENT_LANE_ID,
                    renderedSig: instrumentRenderSig(track, seg, tempo),
                },
            ]);
        }
    }
    catch (e) {
        if (segRef())
            deposit(prevOutputs);
        throw e;
    }
}
/** 顺序批量渲染脏乐器段落。每项渲染前实时复检(可能已被删除/被并发触发烘好)。
 *  `shouldCancel`(第二次按 Play)在条目间中止。失败:记日志 + 封顶 toast,不中断批次。 */
export async function renderDirtyInstruments(list, tempo, opts) {
    let rendered = 0;
    let failed = 0;
    const MAX_FAILURE_TOASTS = 3;
    for (const { trackId, segmentId } of list) {
        if (opts?.shouldCancel?.())
            return { rendered, failed, cancelled: true };
        const tr = useProjectStore.getState().tracks.find((t) => t.id === trackId);
        const sg = tr?.segments.find((s) => s.id === segmentId);
        if (!tr || !sg || !isInstrumentDirty(tr, sg, tempo))
            continue;
        // 音符被清空但烘焙残留 → 清为静默(无物可渲)。
        if (sg.content.type === "notes" && sg.content.notes.length === 0) {
            useProjectStore.getState().replaceProcessedOutputs(tr.id, sg.id, []);
            rendered++;
            continue;
        }
        try {
            await renderInstrumentPart(tr, sg, tempo);
            rendered++;
        }
        catch (e) {
            if (opts?.shouldCancel?.())
                return { rendered, failed, cancelled: true };
            failed++;
            logToBackend("error", `Instrument auto-render failed (${tr?.name ?? trackId}): ${String(e)}`);
            if (failed <= MAX_FAILURE_TOASTS) {
                useAppStore.getState().showToast(`${tr?.name ?? ""}: ${String(e)}`, "error");
            }
            else if (failed === MAX_FAILURE_TOASTS + 1) {
                useAppStore.getState().showToast(i18n.t("vocalEditor.render.moreFailures"), "error");
            }
        }
    }
    return { rendered, failed, cancelled: false };
}
// ── 单音试听(Muno 阶段2)───────────────────────────────────────────────────
// 点击/抓取乐器轨音符时用真实音色发声:优先走 Rust cpal 常驻实时流(乐器已缓存时
// 零文件 I/O);实时不可用时命令返回 WAV 缓存路径,由 WebAudio 播放。拖拽移调的
// 连续重发声仍走振荡器(setPreviewToneHz)——实时流也跟不上连续变调,两者互补。
let auditionSeq = 0;
export async function playSoundfontAudition(sf, key, vel) {
    const seq = ++auditionSeq;
    try {
        // Muno 阶段2:优先走 Rust cpal 常驻实时流(乐器已缓存时零延迟)——命令返回
        // null 表示已直接发声,前端无事可做;返回路径 = 实时不可用,回退 WAV+WebAudio。
        const wavPath = await invoke("audition_soundfont_note", {
            fontId: sf.fontId,
            presetId: sf.presetId,
            key: Math.max(0, Math.min(127, Math.round(key))),
            vel: vel ?? 100,
            dur: 0.6,
        });
        if (seq !== auditionSeq)
            return; // 已被更新的试听取代 → 丢弃
        if (!wavPath)
            return; // 实时引擎已直接发声
        const ctx = getPreviewContext();
        const buf = await loadAudioBuffer(wavPath);
        if (seq !== auditionSeq)
            return;
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.start();
    }
    catch {
        // 试听是点缀反馈:失败静默(渲染/播放失败的主路径在 Play 流程里会响亮报错)。
    }
}
