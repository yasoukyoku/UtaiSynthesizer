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
// 旋钮(S73d 定稿):Expressiveness=整体缩放(过冲+起收+颤音)/Rigidness(UI)↔vib=颤音维/
// Take=确定性唱法版本(相位=phaseForTake(take,noteId) 纯函数,替代随机 Retake 抽奖;
// 模型本身不预测 phase)。Phase B(CVAE)落地后 Expressiveness 升级为真采样温度。

import { invoke } from "@tauri-apps/api/core";
import type { Note, NoteTransition, VocalTrackParams } from "../../types/project";
import { ticksToMs } from "../audio/laneOps";
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

/** S73b/c/d 自动调教旋钮(持久在 VocalTrackParams;常开语义:改任一 → watcher 重调教 → 重渲染):
 *  expr=整体表现力(过冲+起收滑幅+颤音都乘);vib=颤音维度缩放(UI 以 Rigidness 反向百分比
 *  展示:+100%=vib 0 拉平 / 0%=vib 1 默认 / −100%=vib 2);take=唱法版本号(S73d,替代
 *  Retake 按钮的「抽奖」):相位 = 纯函数 phaseForTake(take, noteId)——确定可复现、可转回、
 *  存盘还原、新音符自动入座同一 take。 */
export interface AutoTuneScales {
  expr: number;
  vib: number;
  take: number;
}

/** vocalParams → 旋钮(单一读取点;absent 默认 = expr 2 / vib 1 / take 0)。 */
export function autoTuneScalesOf(p: VocalTrackParams | undefined): AutoTuneScales {
  return { expr: p?.autoTuneExpr ?? 2, vib: p?.autoTuneVib ?? 1, take: p?.autoTuneTake ?? 0 };
}

/** Take → 逐音符颤音相位(确定性;SynthV AI-Retakes 的可复现版):take 0 = 基准相位 0
 *  (KA3 耳测口径);take ≥1 = FNV-1a(`take:noteId`) 均匀散布到 [-0.5, 0.5)。 */
export function phaseForTake(take: number, noteId: string): number {
  if (take === 0) return 0;
  const s = `${take}:${noteId}`;
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) / 4294967296 - 0.5;
}

export interface AutoTuneResult {
  /** 本次写入 θ 的音符数。 */
  applied: number;
  /** 因用户调教被保留而绕行的音符数(scope 内)。 */
  skipped: number;
  /** await 期间内容变了(用户在途编辑/段被删)→ 整批丢弃未写入,请重试。 */
  stale?: boolean;
}

/** 音高线「手绘段染色」阈值:|dev|≥此值(cents)的采样点画成用户色(VocalEditor 消费)。 */
export const DEV_TINT_EPS_CENTS = 0.5;

const round2 = (v: number): number => Math.round(v * 100) / 100;

/** 用户调教判定(θ 维度的绕行谓词)。
 *  ★S73c 语义改版(用户拍板=SV 同构):pitchDev【不再】参与 θ 资格——手绘 deviation 是
 *  独立的用户叠加层,机器永不写它(层隔离),基线 θ 在它下面照常再生成(SV1 Sing 模式
 *  在既有 Pitch Deviation 之下重生成基线=同款手感)。「手绘段不吃自动调教」由层隔离
 *  天然成立,不需要把整个音符划出机器地盘。θ 维度的用户地盘 = 手设 vibrato/transition
 *  (含 ustx 烤入的显式 ZERO_TRANSITION=导入调教)。 */
export function isUserTuned(n: Note): boolean {
  if (n.autoTuned) return false;
  return !!(n.vibrato || n.transition);
}

/** θ → Note 字段(scales=旋钮;phase=phaseForTake 已由调用方算好传入)。expr 乘 过冲+起收
 *  滑幅+颤音(=「音符实不实」一并归表现力管,用户 S73b 拍板);vib 只再乘颤音。 */
function thetaToFields(th: AutotuneTheta, scales: AutoTuneScales, phase: number): Partial<Note> {
  const tr = th.transition;
  const k = scales.expr;
  // S73d:相位=纯函数 phaseForTake(take, noteId) → 不再需要「保住容器防相位丢失」的
  // 0.01¢ 垫底 hack;缩放到 0 就干净地无 vibrato,转回时相位从 take 重建。
  const vibDepth = th.vibrato.depthCents * k * scales.vib;
  return {
    transition: {
      offsetMs: round2(tr.offsetMs),
      durLeftMs: round2(tr.durLeftMs),
      durRightMs: round2(tr.durRightMs),
      depthLeftCents: round2(tr.depthLeftCents * k),
      depthRightCents: round2(tr.depthRightCents * k),
      openEdgeCents: round2(tr.openEdgeCents * k),
    },
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
 * 自动调教入口(S73d 起 watcher 专用,唯一模式=refresh 跟随)。一次 applyNoteEdits = 一步 undo;
 * silent=true → 走 history.runSilent:不进撤销栈、不砍 redo、基线同步——快照里 vocalParams
 * (旋钮)与 θ 一起恢复,undo 触发的重跑经 no-op 守卫收敛为零写入。
 */
export async function applyAutoTune(
  trackId: string,
  segmentId: string,
  scales: AutoTuneScales,
  opts: { silent?: boolean } = {},
): Promise<AutoTuneResult> {
  const store = useProjectStore.getState();
  const track = store.tracks.find((t) => t.id === trackId);
  const seg = track?.segments.find((s) => s.id === segmentId);
  if (!track || !seg || seg.content.type !== "notes" || seg.content.notes.length === 0) {
    return { applied: 0, skipped: 0 };
  }
  const notes = seg.content.notes; // 写入漏斗保证 (tick,id) 有序 = 命令的升序契约
  const targetIdx: number[] = [];
  let skipped = 0;
  notes.forEach((n, i) => {
    if (!isUserTuned(n)) targetIdx.push(i);
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
    update[n.id] = thetaToFields(th, scales, phaseForTake(scales.take, n.id));
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
