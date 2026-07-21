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

/** refresh=常开 watcher 的静默跟随(各音符现有相位保留——否则每次编辑都会把 Retake 的
 *  相位洗掉;新音符无相位 → 0)/ retake=换一版相位(仅机器音符)。S73c 起无手动按钮:
 *  「最开始也自动化」(用户拍板),follow 开关即模式切换。 */
export type AutoTuneMode = "refresh" | "retake";

/** S73b 双缩放:expr=整体表现力(过冲+起收滑幅+颤音都乘),vib=颤音单独再乘(总颤音深度
 *  = θ.depth × expr × vib)。持久在 VocalTrackParams(常开语义:改 k → 重调教 → 重渲染)。 */
export interface AutoTuneScales {
  expr: number;
  vib: number;
}

/** vocalParams → 缩放(单一读取点;absent 默认 = expr 2 / vib 1,S73c 用户拍板)。 */
export function autoTuneScalesOf(p: VocalTrackParams | undefined): AutoTuneScales {
  return { expr: p?.autoTuneExpr ?? 2, vib: p?.autoTuneVib ?? 1 };
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
  const scope = selectedIds.length > 0 ? new Set(selectedIds) : null;
  const targetIdx: number[] = [];
  let skipped = 0;
  notes.forEach((n, i) => {
    if (scope && !scope.has(n.id)) return;
    const eligible = mode === "retake" ? n.autoTuned === true : !isUserTuned(n);
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
    const phase = mode === "retake" ? Math.random() - 0.5 : (n.vibrato?.phase ?? 0);
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
