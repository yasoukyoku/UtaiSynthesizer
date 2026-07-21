// ② 自动音高调教(旋钮线 Phase A 立室,S73)— θ 获取 / 调教所有权判定 / 应用。
//
// 模型链:notes(tick→ms)→ Rust run_autotune(aux autotune_a1.onnx,乐句切分在 Rust)→
// per-note θ(绝对 ms/cents,SynthV 约定)→ 写 Note.transition/vibrato(全 6 字段覆盖,
// e1KarmDump K 臂耳测同款路径)→ 现有 evalF0Cents/overlay/渲染链零改动直接生效。
//
// 调教所有权(用户 S73 拍板;SynthV SV1 同构——「用户设过值的属性,自动过程跳过」):
//   - 用户调教 = 手设 vibrato/transition(无 autoTuned 标记,含 ustx 导入的显式覆盖)
//     或 pitchDev 在音符范围内有信号(手绘/ustx 烤入)。自动调教/Retake 一律绕行;
//     pitchDev 层永不被机器触碰(SynthV Instant-Mode 层隔离)。
//   - 机器调教 = autoTuned===true(auto-tune 自己写的),可自由改写/retake/重缩放。
//   - 侧栏手动改 transition/vibrato → 剥 autoTuned(所有权移交用户;VocalSidebar 负责)。
//
// Expressiveness k = 确定性缩放(vibrato depth + 滑音过冲 depthL/R ×k)——Phase A 的 det
// 近似;Phase B(CVAE retake)落地后升级为真采样温度。Retake = 逐音符随机 vibrato 相位
// (export_karm --phase random / KA3R 耳测同款语义;模型本身不预测 phase)。

import { invoke } from "@tauri-apps/api/core";
import type { Note, NoteTransition, PitchCurve, VocalTrackParams } from "../../types/project";
import { ticksToMs } from "../audio/laneOps";
import { evalCurveAt } from "../f0eval";
import { useProjectStore } from "../../store/project";
import { useHistoryStore, inGestureTransaction } from "../../store/history";

/** Rust run_autotune 的 θ 行(全 6+6 字段,单位 ms/cents/Hz;phase 恒 0,retake 在 TS 侧注入)。 */
export interface AutotuneTheta {
  transition: Required<NoteTransition>;
  vibrato: {
    depthCents: number;
    freqHz: number;
    phase: number;
    startMs: number;
    easeInMs: number;
    easeOutMs: number;
  };
}

/** fresh=侧栏按钮(强制重调教,phase 归 0=take 0)/ retake=换一版相位(仅机器音符)/
 *  refresh=常开 watcher 的静默跟随(fresh 资格 + 各音符现有相位保留——否则每次编辑都会
 *  把 Retake 的相位洗掉)。 */
export type AutoTuneMode = "fresh" | "retake" | "refresh";

/** S73b 双缩放:expr=整体表现力(过冲+起收滑幅+颤音都乘),vib=颤音单独再乘(总颤音深度
 *  = θ.depth × expr × vib)。持久在 VocalTrackParams(常开语义:改 k → 重调教 → 重渲染)。 */
export interface AutoTuneScales {
  expr: number;
  vib: number;
}

/** vocalParams → 缩放(单一读取点;absent = 默认 1)。 */
export function autoTuneScalesOf(p: VocalTrackParams | undefined): AutoTuneScales {
  return { expr: p?.autoTuneExpr ?? 1, vib: p?.autoTuneVib ?? 1 };
}

export interface AutoTuneResult {
  /** 本次写入 θ 的音符数。 */
  applied: number;
  /** 因用户调教被保留而绕行的音符数(scope 内)。 */
  skipped: number;
  /** await 期间内容变了(用户在途编辑/段被删)→ 整批丢弃未写入,请重试。 */
  stale?: boolean;
}

/** pitchDev 信号阈值:音符范围内 |dev|≥此值(cents)即视为「用户在此调教过」。 */
export const TUNED_DEV_EPS_CENTS = 2;

const round2 = (v: number): number => Math.round(v * 100) / 100;

/** 线性折线 pitchDev 在 [t0,t1](segment 相对 tick)内是否有 ≥eps 的信号——折线的最大
 *  绝对值只会出现在区间内控制点或区间端点的插值处,两类都查即充分。 */
export function curveHasSignal(
  c: PitchCurve | undefined,
  t0: number,
  t1: number,
  eps: number,
): boolean {
  if (!c || c.xs.length === 0) return false;
  if (Math.abs(evalCurveAt(c, t0)) >= eps || Math.abs(evalCurveAt(c, t1)) >= eps) return true;
  // 二分定位首个 x≥t0(S73b:烤入曲线数万点 × 编辑器着色逐音符判定,线性起扫会热)
  let lo = 0;
  let hi = c.xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((c.xs[mid] ?? Infinity) < t0) lo = mid + 1;
    else hi = mid;
  }
  for (let i = lo; i < c.xs.length; i++) {
    const x = c.xs[i];
    const y = c.ys[i];
    if (x === undefined || y === undefined) break;
    if (x > t1) break;
    if (Math.abs(y) >= eps) return true;
  }
  return false;
}

/** 用户调教判定(自动过程的绕行谓词)。
 *  ★顺序要紧(S73 审查):pitchDev 覆盖必须先于 autoTuned 短路——机器调教过的音符上用户又
 *  手绘了 dev,dev 是相对当时机器基线画的,机器再改基线=抽走手绘修正的地基 → 也算用户地盘。 */
export function isUserTuned(n: Note, pitchDev: PitchCurve | undefined): boolean {
  if (curveHasSignal(pitchDev, n.tick, n.tick + n.duration, TUNED_DEV_EPS_CENTS)) return true;
  if (n.autoTuned) return false;
  return !!(n.vibrato || n.transition);
}

/** θ → Note 字段(scales=表现力/颤音双缩放;phase 由 mode 决定)。expr 乘 过冲+起收滑幅+颤音
 *  (=「音符实不实」一并归表现力管,用户 S73b 拍板);vib 只再乘颤音。 */
function thetaToFields(th: AutotuneTheta, scales: AutoTuneScales, phase: number): Partial<Note> {
  const tr = th.transition;
  const k = scales.expr;
  // 缩放到 0 时垫 0.01¢(听阈外)保住 vibrato 容器:k 拖到端点再拖回,Retake 相位不被洗掉
  // (S73b 审查;θ.depth==0 的音符仍然干净地无 vibrato)。
  const scaled = th.vibrato.depthCents * k * scales.vib;
  const vibDepth = th.vibrato.depthCents > 0 ? Math.max(0.01, scaled) : 0;
  return {
    transition: {
      offsetMs: round2(tr.offsetMs),
      durLeftMs: round2(tr.durLeftMs),
      durRightMs: round2(tr.durRightMs),
      depthLeftCents: round2(tr.depthLeftCents * k),
      depthRightCents: round2(tr.depthRightCents * k),
      openEdgeCents: round2(tr.openEdgeCents * k),
    },
    // depth>0 一律保留(哪怕亚 cent):Expressiveness 拖低再拖高的往返若把 vibrato 省略掉,
    // Retake 相位会静默丢失归 0(S73 审查)。depth==0(k=0)由 normalizeNote 归一为 absent。
    vibrato:
      vibDepth > 0
        ? {
            depthCents: round2(vibDepth),
            freqHz: round2(th.vibrato.freqHz),
            phase: Math.round(phase * 1000) / 1000,
            startMs: round2(th.vibrato.startMs),
            easeInMs: round2(th.vibrato.easeInMs),
            easeOutMs: round2(th.vibrato.easeOutMs),
          }
        : undefined,
    autoTuned: true,
  };
}

/** 整段 sorted notes 喂模型(模型吃乐句上下文;θ 与 notes 同序同长)。 */
async function fetchTheta(notes: readonly Note[], tempo: number): Promise<AutotuneTheta[]> {
  const payload = notes.map((n) => ({
    startMs: ticksToMs(n.tick, tempo),
    durMs: ticksToMs(n.duration, tempo),
    pitch: n.pitch + (n.detune ?? 0) / 100,
  }));
  return await invoke<AutotuneTheta[]>("run_autotune", { notes: payload });
}

/**
 * 自动调教入口。selectedIds 空 = 整段。一次 applyNoteEdits = 一步 undo;
 * silent=true(常开 watcher)→ 走 history.runSilent:不进撤销栈、不砍 redo、基线同步——
 * 快照里 vocalParams(k)与 θ 一起恢复,undo 触发的重跑经 no-op 守卫收敛为零写入。
 */
export async function applyAutoTune(
  trackId: string,
  segmentId: string,
  selectedIds: readonly string[],
  scales: AutoTuneScales,
  mode: AutoTuneMode,
  opts: { silent?: boolean } = {},
): Promise<AutoTuneResult> {
  const store = useProjectStore.getState();
  const track = store.tracks.find((t) => t.id === trackId);
  const seg = track?.segments.find((s) => s.id === segmentId);
  if (!track || !seg || seg.content.type !== "notes" || seg.content.notes.length === 0) {
    return { applied: 0, skipped: 0 };
  }
  const notes = seg.content.notes; // 写入漏斗保证 (tick,id) 有序 = 命令的升序契约
  const pitchDev = seg.content.pitchDev;
  const scope = selectedIds.length > 0 ? new Set(selectedIds) : null;
  const targetIdx: number[] = [];
  let skipped = 0;
  notes.forEach((n, i) => {
    if (scope && !scope.has(n.id)) return;
    // ★资格必须整体过 isUserTuned(S73b 审查 HIGH):`machine || …` 的短路会架空谓词里
    //   「pitchDev 覆盖优先」规则——机器调教之上手绘了 dev 的音符,机器不得再动基线。
    const eligible =
      mode === "retake"
        ? n.autoTuned === true && !isUserTuned(n, pitchDev)
        : !isUserTuned(n, pitchDev);
    if (eligible) targetIdx.push(i);
    else skipped++;
  });
  if (targetIdx.length === 0) return { applied: 0, skipped };

  const theta = await fetchTheta(notes, store.tempo);
  if (theta.length !== notes.length) throw new Error("AUTOTUNE_SHAPE: theta/notes length");

  // ★await 窗口竞态守卫(S73 审查):eligibility/θ 对位都基于 await 前的快照;期间用户若
  //   手动编辑(所有权可能已移交)/增删移音符(θ 索引错位)/删段,旧补丁一律不许落盘——
  //   notes 数组引用不变 ⇔ 内容零变更(所有写入都经 normalizeNotesArray 产新数组)。
  const after = useProjectStore.getState();
  const segAfter = after.tracks
    .find((t) => t.id === trackId)
    ?.segments.find((s) => s.id === segmentId);
  if (!segAfter || segAfter.content.type !== "notes" || segAfter.content.notes !== notes) {
    return { applied: 0, skipped, stale: true };
  }

  const update: Record<string, Partial<Note>> = {};
  for (const i of targetIdx) {
    const n = notes[i];
    const th = theta[i];
    if (!n || !th) continue;
    const phase =
      mode === "retake"
        ? Math.random() - 0.5
        : mode === "refresh"
          ? (n.vibrato?.phase ?? 0)
          : 0;
    update[n.id] = thetaToFields(th, scales, phase);
  }
  // ★手势事务窗守卫(S73b 审查):txnDepth>0 期间落地 silent 写会被 commitTransaction 的
  //   sig 对比捕获——零变化手势变「纯机器 θ 的幻影撤销步」且 future=[](砍 redo)。
  //   await 期间手势可能才开始,所以在写入点(而非只在 watcher 入口)复查;按 stale 语义
  //   返回让 watcher 不入账、稍后重试。
  if (opts.silent && inGestureTransaction()) {
    return { applied: 0, skipped, stale: true };
  }
  // await 之后重新取 store(应用点用最新引用;update 按 id 命中,消失的音符自然 no-op)
  const write = () => useProjectStore.getState().applyNoteEdits(trackId, segmentId, { update });
  if (opts.silent) useHistoryStore.getState().runSilent(write);
  else write();
  return { applied: targetIdx.length, skipped };
}
