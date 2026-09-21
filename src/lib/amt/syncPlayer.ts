/**
 * Sample-synchronized web port of music-to-midi's `SynchronizedPcmPlayer`
 * (src/gui/synchronized_pcm_player.py).
 *
 * The reference mixes the original audio, the pre-mixed MIDI render, and the
 * per-instrument buses through ONE audio output clock so they can never drift:
 *   - mix slider (0..1): mono crossfade  original*(1-mix) + midi*mix
 *   - stereo mode: LEFT = original, RIGHT = MIDI (A/B comparison)
 *   - per-instrument mute: when ANY instrument is muted, the MIDI side is
 *     rebuilt as the SUM of the unmuted instrument buses; with no mutes the
 *     pre-mixed transcription WAV plays (loudness-matched, exactly like the
 *     reference's midi_mix bus).
 *
 * Web Audio equivalent: all sources are scheduled with the same
 * `ctx.currentTime` + offset, which the audio thread starts sample-synchronously.
 */

import { readFile } from "@tauri-apps/plugin-fs";

export interface SyncMixState {
  /** 0 = pure original, 1 = pure MIDI. */
  mix: number;
  /** true = A/B mode (original left channel, MIDI right channel). */
  stereo: boolean;
  /** Muted instrument keys (stem names). */
  muted: Iterable<string>;
}

export interface SyncPlayerSources {
  /** The original audio (right side of the A/B, or the (1-mix) mono term). */
  originalPath?: string;
  /** Pre-mixed MIDI transcription WAV — used when NO instrument is muted. */
  midiMixPath?: string;
  /** Per-instrument buses keyed by instrument (stem) name. */
  instrumentPaths: Record<string, string>;
}

export class SynchronizedPlayer {
  private static _ctx: AudioContext | null = null;

  private static ctx(): AudioContext {
    if (!SynchronizedPlayer._ctx || SynchronizedPlayer._ctx.state === "closed") {
      SynchronizedPlayer._ctx = new AudioContext();
    }
    return SynchronizedPlayer._ctx;
  }

  private original: AudioBuffer | null = null;
  private midiMix: AudioBuffer | null = null;
  private instruments = new Map<string, AudioBuffer>();

  // Live wiring (rebuilt per play / per mix-state change).
  private sources: AudioBufferSourceNode[] = [];
  private origGain: GainNode | null = null;
  private midiGain: GainNode | null = null;
  private merger: ChannelMergerNode | null = null;

  private playing = false;
  private startCtxTime = 0;
  private startOffset = 0;
  /** Playback rate (project BPM ÷ detected BPM) — mirrors the reference's playback_rate. */
  private rate = 1;

  private mix = 0.5;
  private stereo = false;
  private muted = new Set<string>();

  /** Fired once when playback reaches the end (all sources exhausted). */
  onEnded: (() => void) | null = null;
  private endFired = false;

  get isConfigured(): boolean {
    return !!(this.original || this.midiMix || this.instruments.size > 0);
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get duration(): number {
    let d = 0;
    if (this.original) d = Math.max(d, this.original.duration);
    if (this.midiMix) d = Math.max(d, this.midiMix.duration);
    for (const b of this.instruments.values()) d = Math.max(d, b.duration);
    return d;
  }

  get position(): number {
    if (!this.playing) return this.startOffset;
    const ctx = SynchronizedPlayer.ctx();
    return Math.min(
      this.duration,
      this.startOffset + Math.max(0, ctx.currentTime - this.startCtxTime) * this.rate,
    );
  }

  private static async decode(path: string): Promise<AudioBuffer> {
    const bytes = await readFile(path);
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return await SynchronizedPlayer.ctx().decodeAudioData(ab as ArrayBuffer);
  }

  /** Load every bus. Mirrors `SynchronizedPcmPlayer.configure()`. */
  async configure(src: SyncPlayerSources): Promise<void> {
    this.stopSources();
    this.original = null;
    this.midiMix = null;
    this.instruments.clear();

    const jobs: Promise<void>[] = [];
    if (src.originalPath) {
      jobs.push(
        SynchronizedPlayer.decode(src.originalPath)
          .then((b) => { this.original = b; })
          .catch((e) => { console.warn("[syncPlayer] original decode failed:", e); }),
      );
    }
    if (src.midiMixPath) {
      jobs.push(
        SynchronizedPlayer.decode(src.midiMixPath)
          .then((b) => { this.midiMix = b; })
          .catch((e) => { console.warn("[syncPlayer] midi mix decode failed:", e); }),
      );
    }
    for (const [name, path] of Object.entries(src.instrumentPaths)) {
      if (!path) continue;
      jobs.push(
        SynchronizedPlayer.decode(path)
          .then((b) => { this.instruments.set(name, b); })
          .catch((e) => { console.warn(`[syncPlayer] instrument ${name} decode failed:`, e); }),
      );
    }
    await Promise.all(jobs);
    this.startOffset = 0;
    this.playing = false;
    this.endFired = false;
  }

  /** The instruments that exist for muting (stem names with buses). */
  instrumentNames(): string[] {
    return [...this.instruments.keys()];
  }

  private stopSources() {
    for (const s of this.sources) {
      try { s.stop(); s.disconnect(); } catch { /* already stopped */ }
    }
    this.sources = [];
    if (this.origGain) { try { this.origGain.disconnect(); } catch {} this.origGain = null; }
    if (this.midiGain) { try { this.midiGain.disconnect(); } catch {} this.midiGain = null; }
    if (this.merger) { try { this.merger.disconnect(); } catch {} this.merger = null; }
  }

  /** Build the output wiring for the CURRENT mix state and (re)start every
   *  source at `offsetSec` — the sample-synchronous core. */
  private buildGraph(offsetSec: number) {
    this.stopSources();
    const ctx = SynchronizedPlayer.ctx();
    const when = ctx.currentTime + 0.02; // one small scheduling headstart

    this.origGain = ctx.createGain();
    this.midiGain = ctx.createGain();

    if (this.stereo) {
      // A/B: original on LEFT, MIDI on RIGHT.
      this.merger = ctx.createChannelMerger(2);
      this.origGain.connect(this.merger, 0, 0);
      this.midiGain.connect(this.merger, 0, 1);
      this.merger.connect(ctx.destination);
      this.origGain.gain.value = 1;
      this.midiGain.gain.value = 1;
    } else {
      // Mono crossfade: original*(1-mix) + midi*mix.
      this.origGain.gain.value = 1 - this.mix;
      this.midiGain.gain.value = this.mix;
      this.origGain.connect(ctx.destination);
      this.midiGain.connect(ctx.destination);
    }

    let lastDur = 0;

    // Original side.
    if (this.original && offsetSec < this.original.duration) {
      const s = ctx.createBufferSource();
      s.buffer = this.original;
      s.playbackRate.value = this.rate;
      s.connect(this.origGain);
      s.start(when, offsetSec);
      this.sources.push(s);
      lastDur = Math.max(lastDur, this.original.duration - offsetSec);
    }

    // MIDI side: pre-mixed bus when nothing is muted, otherwise the sum of the
    // unmuted instrument buses (exactly the reference's mute semantics).
    const useBuses = this.muted.size > 0 || !this.midiMix;
    if (useBuses) {
      for (const [name, buf] of this.instruments) {
        if (this.muted.has(name)) continue;
        if (offsetSec >= buf.duration) continue;
        const s = ctx.createBufferSource();
        s.buffer = buf;
        s.playbackRate.value = this.rate;
        s.connect(this.midiGain!);
        s.start(when, offsetSec);
        this.sources.push(s);
        lastDur = Math.max(lastDur, buf.duration - offsetSec);
      }
    } else if (offsetSec < this.midiMix!.duration) {
      const s = ctx.createBufferSource();
      s.buffer = this.midiMix!;
      s.playbackRate.value = this.rate;
      s.connect(this.midiGain!);
      s.start(when, offsetSec);
      this.sources.push(s);
      lastDur = Math.max(lastDur, this.midiMix!.duration - offsetSec);
    }

    this.startCtxTime = when;
    this.startOffset = offsetSec;
    this.endFired = false;

    if (this.sources.length > 0) {
      // Longest source announces completion.
      this.sources[this.sources.length - 1]!.onended = () => {
        if (!this.playing || this.endFired) return;
        if (this.position >= this.duration - 0.05) {
          this.endFired = true;
          this.playing = false;
          this.startOffset = this.duration;
          this.onEnded?.();
        }
      };
      void lastDur; // (kept for clarity; end detection is position-based)
    }
  }

  /** Play from `fromSec` (defaults to the frozen position). */
  async play(fromSec?: number) {
    const ctx = SynchronizedPlayer.ctx();
    if (ctx.state === "suspended") await ctx.resume();
    if (!this.isConfigured) return;
    let pos = fromSec ?? this.startOffset;
    if (pos >= this.duration - 0.01) pos = 0; // restart from the top
    this.buildGraph(pos);
    this.playing = this.sources.length > 0;
    if (!this.playing) this.stopSources();
  }

  pause() {
    if (!this.playing) return;
    const pos = this.position;
    this.stopSources();
    this.playing = false;
    this.startOffset = pos;
  }

  seek(seconds: number) {
    const target = Math.min(this.duration, Math.max(0, seconds));
    if (this.playing) {
      this.buildGraph(target);
    } else {
      this.startOffset = target;
    }
  }

  /**
   * Live playback-rate change (project BPM ÷ detected BPM). A rate change while
   * playing rebuilds the sources at the CURRENT position — the reference's
   * `set_playback_rate` (reset sink + restart from the frozen position).
   */
  setPlaybackRate(rate: number) {
    const r = Math.min(20, Math.max(0.05, rate));
    if (Math.abs(r - this.rate) < 1e-9) return;
    if (this.playing) {
      const pos = this.position;
      this.rate = r;
      this.buildGraph(pos);
    } else {
      this.rate = r;
    }
  }

  /**
   * Live mix-state update. Gain-only changes apply WITHOUT restart; a change
   * that alters WHICH buffers feed the MIDI side (mute set emptiness flips)
   * rebuilds the sources at the current position — the reference calls this
   * "discard buffer at current position".
   */
  setMixState(state: SyncMixState) {
    const prevMuted = this.muted;
    const nextMuted = new Set(state.muted);
    const prevUseBuses = prevMuted.size > 0 || !this.midiMix;
    const nextUseBuses = nextMuted.size > 0 || !this.midiMix;
    let mutesChanged = prevMuted.size !== nextMuted.size;
    if (!mutesChanged) {
      for (const m of nextMuted) {
        if (!prevMuted.has(m)) { mutesChanged = true; break; }
      }
    }
    const busesFlipped = prevUseBuses !== nextUseBuses;

    this.mix = Math.min(1, Math.max(0, state.mix));
    this.stereo = !!state.stereo;
    this.muted = nextMuted;

    if (!this.playing || !this.origGain || !this.midiGain) return; // wiring is rebuilt on play anyway
    const prevStereo = this.merger !== null;
    const ctx = SynchronizedPlayer.ctx();

    if (prevStereo !== this.stereo || busesFlipped || (nextUseBuses && mutesChanged)) {
      // Structural change (A/B toggle, pre-mix↔bus flip, or a different bus set)
      // → rebuild at the current position, like the reference's
      // "_discard_buffer_at_current_position".
      this.buildGraph(this.position);
      return;
    }

    if (this.stereo) {
      this.origGain.gain.setTargetAtTime(1, ctx.currentTime, 0.01);
      this.midiGain.gain.setTargetAtTime(1, ctx.currentTime, 0.01);
    } else {
      this.origGain.gain.setTargetAtTime(1 - this.mix, ctx.currentTime, 0.01);
      this.midiGain.gain.setTargetAtTime(this.mix, ctx.currentTime, 0.01);
    }
  }

  dispose() {
    this.stopSources();
    this.playing = false;
    this.onEnded = null;
  }
}
