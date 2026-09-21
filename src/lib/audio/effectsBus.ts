// S12 — 轨道路由 / 总线 / 发送 (Track Routing / Aux Buses / Sends).
//
// 单一事实源:playback.ts 与 exportMixdown.ts 都用 `connectTrackOutput` 把每个音轨的 panner 输出
// **并行**叠进辅助总线——主路(干声)照旧 `panner → dry → destination`,效果发送(湿声)则按
// 每轨的 `reverbSend` / `delaySend` 抽头进入全局混响 / 延迟总线,返回后并回 `dry`。
//
// 「听导一致」保证:
//   1. playback 用实时 AudioContext,export 用 OfflineAudioContext,二者都满足 BaseAudioContext,
//      `getFxChain` 用同一个 builder 建图(仅首建),防止两份文件各自编一套而漂移。
//   2. 混响脉冲用**确定性 LCG** 生成(不用 Math.random),两条路径逐位一致。
//   3. 干声只是多经过一个增益为 1 的 GainNode(线性直通,数学上恒等于原来直接连 destination)。
//
// 总线参数(全局 wet / delay 时间 / 反馈)通过 `setFxBusConfig` 设定,`getFxChain` 在参数变化时
// 就地更新 AudioParam,无需重建图——播放中改参数也立即生效,且不影响已调度的源。

export interface FxBusConfig {
  /** 混响总返回(wet)增益 0..1。 */
  reverbWet: number;
  /** 延迟总返回(wet)增益 0..1。 */
  delayWet: number;
  /** 延迟时间,秒(>0)。 */
  delayTimeSec: number;
  /** 延迟反馈 0..<1(≥1 自激,UI 上限留 0.95)。 */
  delayFeedback: number;
}

export interface TrackFxSend {
  reverb?: number;
  delay?: number;
}

/** 默认总线参数:中性偏柔的混响 + 427ms 伴随延迟,反馈 0.42(可稳定的温暖尾音)。 */
export const DEFAULT_FX_BUS: FxBusConfig = {
  reverbWet: 0.9,
  delayWet: 0.7,
  delayTimeSec: 0.427,
  delayFeedback: 0.42,
};

interface FxChain {
  dry: GainNode;
  /** 主输出音量 (仅实时 AudioContext 生效; 导出用的 OfflineAudioContext 恒为 1, 不影响导出电平) */
  masterGain: GainNode;
  analyser: AnalyserNode;
  reverbInput: GainNode;
  delayInput: GainNode;
  delayNode: DelayNode;
  delayFb: GainNode;
  delayReturn: GainNode;
  reverbReturn: GainNode;
  cfgKey: string;
}

type BaseCtx = BaseAudioContext;
const chains = new WeakMap<BaseCtx, FxChain>();
/** 每轨分析器 (仅实时 AudioContext) — trackId → AnalyserNode, 供控制台分轨电平表读取 */
const trackAnalysers = new WeakMap<BaseCtx, Map<string, AnalyserNode>>();

/** Deterministic stereo reverb impulse — a fixed-seed LCG (NOT Math.random) so the live and offline
 *  render hear the SAME room. 2.2s, slow exponential decay. */
function reverbImpulse(ctx: BaseCtx): AudioBuffer {
  const rate = ctx.sampleRate;
  const len = Math.floor(rate * 2.2);
  const impulse = ctx.createBuffer(2, len, rate);
  let seed = 0x2f6e2b1;
  const rnd = () => {
    // xorshift32 — deterministic and fast
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  for (let ch = 0; ch < 2; ch++) {
    const d = impulse.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      const t = i / len;
      d[i] = (rnd() * 2 - 1) * Math.pow(1 - t, 2.6);
    }
  }
  return impulse;
}

/** Build (once per context) the aux-bus chain: dry sum → destination, plus two effect sends whose
 *  returns fold back into `dry`. Returns the cached chain, updating AudioParams if cfg changed. */
function getFxChain(ctx: BaseCtx, cfg: FxBusConfig): FxChain {
  const key = `${cfg.reverbWet}|${cfg.delayWet}|${cfg.delayTimeSec}|${cfg.delayFeedback}`;
  const existing = chains.get(ctx);
  if (existing) {
    if (existing.cfgKey !== key) {
      existing.cfgKey = key;
      existing.reverbReturn.gain.value = cfg.reverbWet;
      existing.reverbReturn.gain.setValueAtTime(cfg.reverbWet, ctx.currentTime);
      existing.delayReturn.gain.value = cfg.delayWet;
      existing.delayNode.delayTime.value = Math.max(0.02, cfg.delayTimeSec);
      existing.delayFb.gain.value = Math.min(0.95, Math.max(0, cfg.delayFeedback));
    }
    return existing;
  }

  const dry = ctx.createGain();
  dry.gain.value = 1;
  // 主输出音量: 实时上下文读取模块级 masterVolumeDb, 离线 (导出) 恒为 1 — 导出不受监听音量影响
  const masterGain = ctx.createGain();
  masterGain.gain.value = ctx instanceof AudioContext ? Math.pow(10, masterVolumeDb / 20) : 1;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.7;
  dry.connect(masterGain);
  masterGain.connect(analyser);
  analyser.connect(ctx.destination);

  const reverb = ctx.createConvolver();
  reverb.buffer = reverbImpulse(ctx);
  const reverbInput = ctx.createGain();
  const reverbReturn = ctx.createGain();
  reverbReturn.gain.value = cfg.reverbWet;
  reverbInput.connect(reverb).connect(reverbReturn).connect(dry);

  const delayNode = ctx.createDelay(3);
  delayNode.delayTime.value = Math.max(0.02, cfg.delayTimeSec);
  const delayInput = ctx.createGain();
  const delayFb = ctx.createGain();
  delayFb.gain.value = Math.min(0.95, Math.max(0, cfg.delayFeedback));
  const delayReturn = ctx.createGain();
  delayReturn.gain.value = cfg.delayWet;
  delayInput.connect(delayNode);
  delayNode.connect(delayReturn).connect(dry);
  delayNode.connect(delayFb).connect(delayInput);

  const chain: FxChain = { dry, masterGain, analyser, reverbInput, delayInput, delayNode, delayFb, delayReturn, reverbReturn, cfgKey: key };
  chains.set(ctx, chain);
  return chain;
}

function clamp01(v: number | undefined): number {
  if (v === undefined || !Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/**
 * Route one track's panner output: dry straight to the destination (unchanged behaviour) plus,
 * when the track sends a nonzero amount, a send-gain tap into each effect bus. Zero sends leave the
 * graph exactly as it was before S12 (no send node, dry only) — the export math stays byte-identical
 * to pre-bus projects.
 */
export function connectTrackOutput(
  ctx: BaseCtx,
  panner: StereoPannerNode,
  sends: TrackFxSend,
  cfg: FxBusConfig,
  /** 传入 trackId 时 (仅 playback 传入), 该轨输出会并行接一只每轨分析器供分轨电平表读取 */
  trackId?: string,
): void {
  const chain = getFxChain(ctx, cfg);
  panner.connect(chain.dry);
  // 分轨电平表抽头: 一条轨的多个来源都并接到同一只分析器上 (求和), OfflineAudioContext 不接
  if (trackId && ctx instanceof AudioContext) {
    let map = trackAnalysers.get(ctx);
    if (!map) { map = new Map(); trackAnalysers.set(ctx, map); }
    let an = map.get(trackId);
    if (!an) {
      an = ctx.createAnalyser();
      an.fftSize = 512;
      an.smoothingTimeConstant = 0.5;
      map.set(trackId, an);
    }
    panner.connect(an);
  }
  const rv = clamp01(sends.reverb);
  if (rv > 0) {
    const g = ctx.createGain();
    g.gain.value = rv;
    panner.connect(g);
    g.connect(chain.reverbInput);
  }
  const dl = clamp01(sends.delay);
  if (dl > 0) {
    const g = ctx.createGain();
    g.gain.value = dl;
    panner.connect(g);
    g.connect(chain.delayInput);
  }
}

/** Round a send to 2 decimals for byte-stable persistence. */
export function roundSend(v: number): number {
  return Math.round(Math.max(0, Math.min(1, v)) * 100) / 100;
}

// ── 全局总线参数(单例)。UI 经 setFxBusConfig 调整并持久化;playback/export 均在建图时读取
//   同一份,保证「听导一致」。 ──
let fxBusConfig: FxBusConfig = { ...DEFAULT_FX_BUS };

/** 当前总线参数——播放与导出的共同快照。 */
export function getFxBusConfig(): FxBusConfig {
  return fxBusConfig;
}

/** 更新总线参数(带钳制)。调用方负责持久化。 */
export function setFxBusConfig(c: Partial<FxBusConfig>): void {
  fxBusConfig = {
    reverbWet: clamp01(c.reverbWet ?? fxBusConfig.reverbWet),
    delayWet: clamp01(c.delayWet ?? fxBusConfig.delayWet),
    delayTimeSec: c.delayTimeSec !== undefined && Number.isFinite(c.delayTimeSec) ? Math.max(0.02, c.delayTimeSec) : fxBusConfig.delayTimeSec,
    delayFeedback: clamp01(c.delayFeedback ?? fxBusConfig.delayFeedback),
  };
}

/** 获取指定 AudioContext 的 AnalyserNode —— 用于频谱显示等可视化组件。
 *  仅当 FxChain 已建立(即至少有一条轨道播放过)时返回有效值;否则返回 null。 */
export function getAnalyserFor(ctx: BaseCtx): AnalyserNode | null {
  const chain = chains.get(ctx);
  return chain?.analyser ?? null;
}

/** 获取一条轨道的专用分析器 (分轨电平表用)。
 *  仅在该轨以 trackId 播放过 (playback 调 connectTrackOutput 时传入) 后才存在。 */
export function getTrackAnalyser(ctx: BaseCtx, trackId: string): AnalyserNode | null {
  return trackAnalysers.get(ctx)?.get(trackId) ?? null;
}

// ── 主输出音量 (总输出推子)。模块级单例 + 实时上下文的 masterGain;导出恒为 1 不受影响。 ──
let masterVolumeDb = 0;

/** 当前主输出音量 (dB)。 */
export function getMasterVolumeDb(): number {
  return masterVolumeDb;
}

/** 设置主输出音量 (dB, 钳制 -60..+6)。只影响实时监听, 离线导出不受影响。 */
export function setMasterVolume(ctx: BaseCtx, db: number): void {
  masterVolumeDb = Math.max(-60, Math.min(6, db));
  const chain = chains.get(ctx);
  if (chain && ctx instanceof AudioContext) {
    chain.masterGain.gain.setValueAtTime(Math.pow(10, masterVolumeDb / 20), ctx.currentTime);
  }
}