/**
 * 单音符预览 — VirtualPiano 点击时触发.
 * 用 WebAudio Oscillator + GainNode 直接发声, 不依赖 DAW engine.
 */
let cachedCtx: AudioContext | null = null;

function getCtx(): AudioContext {
  if (!cachedCtx) {
    const AC = (window as any).AudioContext || (window as any).webkitAudioContext;
    if (!AC) throw new Error("Web Audio API not supported");
    cachedCtx = new AC();
  }
  if ((cachedCtx as AudioContext).state === "suspended") {
    void (cachedCtx as AudioContext).resume();
  }
  return cachedCtx!;
}

function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function playSingleNote(midi: number): void {
  try {
    const ctx = getCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.value = midiToFreq(midi);
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.3, ctx.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.8);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.85);
  } catch (e) {
    console.warn("playSingleNote failed:", e);
  }
}
