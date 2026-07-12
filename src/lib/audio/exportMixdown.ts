// S63 — Audio-export offline mixdown: re-builds THE playback schedule (playback.ts playAllTracks) on an
// OfflineAudioContext so the exported mix is rendered by the SAME WebAudio engine that playback uses —
// identical gain chain, StereoPanner pan law, crossfade ramps, loudness envelopes and stretch artifacts,
// with zero re-implementation of the audio math in Rust.
//
// PARITY CONTRACT (see the anchor comment on playAllTracks): the FORMULAS are imported from playback.ts /
// laneOps / trackLayout (single source), but the scheduling LOOP is mirrored here with the live-only
// machinery removed. The deliberate deltas, each equivalence-argued:
//   1. playheadTick = 0 → the "playhead inside a piece" branch collapses into the "piece ahead" branch
//      (secsInto = 0), so only the latter is kept. Byte-equal to playback started from tick 0.
//   2. No liveMix — mix values come from the passed-in snapshot (an export is a moment-in-time bounce;
//      there is nothing "live" to track mid-render).
//   3. No generation guard / onended counting / late compensation — all decodes happen BEFORE the graph
//      is built (the build loop is fully synchronous), and an OfflineAudioContext has no wall clock to
//      fall behind (currentTime stays 0 until startRendering).
//   4. Sources whose gain is provably constant-0 (muted/solo-excluded track, muted lane row) are SKIPPED
//      instead of scheduled at gain 0. Playback schedules them to support live un-mute; offline there is
//      no live toggle, and a 0-gain source contributes nothing to the sum — mathematically equivalent.
//   5. Decode failures are FATAL (loud error) instead of playback's skip-and-log: a partial export would
//      silently drop a stem — the "silent wrong audio" class of bug this project treats as worst-case.
import { readFile } from "@tauri-apps/plugin-fs";
import type { Track } from "../../types/project";
import type { AudioTrackData } from "../../store/audio";
import {
  applyFadeIn,
  applyFadeOut,
  clampPan,
  dbToLinear,
  loudnessEnvNode,
  ticksToSeconds,
} from "./playback";
import { laneGroupId, laneReachesSeam, laneRowKey, laneVisiblePieces, segStretch } from "./laneOps";
import { ensureStretched } from "./stretchCache";
import { contentEndTick, isLaneRowMuted, laneControlFor, segmentPlaysLanes } from "../trackLayout";

export interface MixdownResult {
  /** Interleaved stereo float32 (L R L R …), NOT clamped — the true float sum, like the WebAudio graph. */
  pcm: Float32Array;
  sampleRate: number;
  /** Max |sample| across both channels. > 1 means the fixed-point/lossy encode will clip (exactly what
   *  the live playback clamps at the hardware, so it still "sounds like what you heard") — the UI should
   *  surface a warning so the user can pull faders down if they care. */
  peak: number;
  durationSec: number;
}

/** Mirror of the playback loop's per-track segment admission (playback.ts sorted filter). */
function playableSegments(track: Track) {
  return [...track.segments]
    .filter((s) => (s.content.type === "audioClip" || segmentPlaysLanes(track, s)) && !s.loading)
    .sort((a, b) => a.startTick - b.startTick);
}

/**
 * Render the whole project (tick 0 → contentEndTick) to an interleaved stereo Float32 PCM buffer.
 * Throws Error whose message starts with a stable EXPORT_* code (the export dialog maps them):
 *   EXPORT_EMPTY            — no audible content
 *   EXPORT_SOURCE_LOADING   — a segment is still decoding (loading placeholder)
 *   EXPORT_SOURCE_MISSING   — an audioClip has no decoded entry (missing/unresolved source file)
 *   EXPORT_DECODE_FAIL      — a stem/source failed to decode or stretch
 */
export async function renderMixdown(
  tracks: Track[],
  audioFiles: Record<string, AudioTrackData>,
  tempo: number,
  sampleRate: number,
  onProgress?: (frac: number) => void,
): Promise<MixdownResult> {
  const hasSolo = tracks.some((t) => t.solo);
  const trackAudible = (t: Track) => !t.muted && (!hasSolo || t.solo);

  // ── Preflight: refuse to bounce a project that isn't fully materialized (silent-drop guard #5).
  // AUDIBLE tracks only (audit): a muted/solo-excluded track contributes zero samples — its missing
  // source must not block the export (playback plays such a project fine, delta #4's own rationale). ──
  for (const track of tracks) {
    if (!trackAudible(track)) continue;
    for (const seg of track.segments) {
      if (seg.loading) throw new Error("EXPORT_SOURCE_LOADING");
      if (
        seg.content.type === "audioClip" &&
        !segmentPlaysLanes(track, seg) &&
        !audioFiles[seg.content.sourcePath]
      ) {
        throw new Error(`EXPORT_SOURCE_MISSING: ${seg.content.sourcePath}`);
      }
    }
  }

  const endTick = contentEndTick(tracks);
  if (endTick <= 0) throw new Error("EXPORT_EMPTY");
  const durationSec = ticksToSeconds(endTick, tempo);
  // Hard ceiling (audit): a pathological timeline (stray far-out segment, extreme tempo) would allocate
  // rendered + interleaved float buffers in the GBs and OOM the WebView with no readable error. An hour
  // of stereo float is already ~2.8 GB across the two buffers — refuse loudly beyond that.
  if (durationSec > 3600) throw new Error("EXPORT_TOO_LONG");
  const frames = Math.max(1, Math.ceil(durationSec * sampleRate));
  const ctx = new OfflineAudioContext(2, frames, sampleRate);

  // ── Pass 1: resolve + decode every buffer the schedule will touch (stretch artifacts included), in
  // parallel, keyed by final path. Decoding on the OFFLINE context resamples straight to the export
  // rate (one resample pass — playback's live-ctx buffers are device-rate and would resample twice). ──
  const buffers = new Map<string, AudioBuffer>();
  /** "path r" → resolved final path (stretch artifact when r ≠ 1). Lets pass 2 stay fully
   *  synchronous — no awaits interleave the graph build. */
  const resolved = new Map<string, string>();
  const wanted = new Map<string, Promise<void>>();
  // COLD stretch artifacts regenerate here — SERIALIZED (audit): each Rust stretch job holds ~4
  // whole-file f32 copies, and S59 deliberately serialized the apply path for exactly that memory
  // spike; an unbounded parallel regen would reintroduce it. Decodes stay parallel.
  let stretchQueue: Promise<unknown> = Promise.resolve();
  const stretchSeq = (path: string, r: number) => {
    const next = stretchQueue.then(() => ensureStretched(path, r));
    stretchQueue = next.catch(() => {}); // a failed job must not wedge the queue
    return next;
  };
  const want = (path: string, r: number) => {
    const key = `${path} ${r}`;
    if (wanted.has(key)) return;
    wanted.set(
      key,
      (async () => {
        const p = r !== 1 ? await stretchSeq(path, r) : path;
        resolved.set(key, p);
        if (buffers.has(p)) return;
        const bytes = await readFile(p);
        const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        buffers.set(p, await ctx.decodeAudioData(ab));
      })().catch((e) => {
        throw new Error(`EXPORT_DECODE_FAIL: ${path} — ${e instanceof Error ? e.message : String(e)}`);
      }),
    );
  };
  for (const track of tracks) {
    if (!trackAudible(track)) continue; // delta #4 — constant-0 gain, no contribution
    for (const seg of playableSegments(track)) {
      const r = segStretch(seg);
      if (segmentPlaysLanes(track, seg)) {
        for (const out of seg.processedOutputs ?? []) {
          if (out.loading || isLaneRowMuted(track, laneRowKey(out), out.laneId)) continue;
          want(out.audioPath, r);
        }
      } else if (seg.content.type === "audioClip") {
        const meta = audioFiles[seg.content.sourcePath];
        if (meta) want(meta.playbackPath || seg.content.sourcePath, r);
      }
    }
  }
  await Promise.all(wanted.values());

  // ── Pass 2: build the graph synchronously — the exact playback loop with playhead = 0 (delta #1). ──
  const now = 0; // OfflineAudioContext.currentTime before startRendering
  for (const track of tracks) {
    if (!trackAudible(track)) continue; // delta #4
    const trackVol = dbToLinear(track.volumeDb);
    const sorted = playableSegments(track);

    for (let si = 0; si < sorted.length; si++) {
      const seg = sorted[si]!;
      const segEnd = seg.startTick + seg.durationTicks;

      if (segmentPlaysLanes(track, seg)) {
        for (const out of seg.processedOutputs!) {
          if (out.loading) continue;
          const rowKey = laneRowKey(out);
          const groupId = laneGroupId(out);
          if (isLaneRowMuted(track, rowKey, out.laneId)) continue; // delta #4 (lane row)
          const stretchR = segStretch(seg);
          const stemPath = resolved.get(`${out.audioPath} ${stretchR}`);
          const buf = stemPath ? buffers.get(stemPath) : undefined;
          if (!buf) throw new Error(`EXPORT_DECODE_FAIL: ${out.audioPath}`);

          const laneCtrl = laneControlFor(track, groupId, out.laneId);
          const laneVol = dbToLinear(laneCtrl?.volumeDb ?? 0);
          const group = groupId;
          const stemDurMs = seg.content.type === "audioClip" ? seg.content.totalDurationMs : out.totalDurationMs;
          const pieces = laneVisiblePieces(seg, seg.laneOps?.[group], stemDurMs, tempo, out.offsetMs ?? 0);

          for (const piece of pieces) {
            let audioOffset = (piece.startMs * stretchR) / 1000;
            let startDelay = ticksToSeconds(piece.startTick, tempo);
            let playDuration = ticksToSeconds(piece.endTick - piece.startTick, tempo);
            audioOffset = Math.max(0, Math.min(audioOffset, buf.duration));
            playDuration = Math.min(playDuration, buf.duration - audioOffset);
            if (playDuration <= 0) continue;

            const laneGainNode = ctx.createGain();
            laneGainNode.gain.setValueAtTime(laneVol, now);
            const trackGainNode = ctx.createGain();
            trackGainNode.gain.setValueAtTime(trackVol, now);
            const panner = ctx.createStereoPanner();
            const lanePan = laneCtrl?.pan ?? 0;
            panner.pan.value = clampPan(track.pan + lanePan);

            const fadeInNode = ctx.createGain();
            const fadeOutNode = ctx.createGain();
            const atSegEnd = Math.abs(piece.endTick - segEnd) < 1;
            const atSegStart = Math.abs(piece.startTick - seg.startTick) < 1;
            if (atSegEnd && si + 1 < sorted.length) {
              const next = sorted[si + 1]!;
              if (next.startTick < segEnd && laneReachesSeam(next, rowKey, tempo, "start")) {
                applyFadeOut(fadeOutNode, next.startTick, Math.min(segEnd, next.startTick + next.durationTicks), 0, tempo, now);
              }
            }
            if (atSegStart && si > 0) {
              const prev = sorted[si - 1]!;
              const prevEnd = prev.startTick + prev.durationTicks;
              if (seg.startTick < prevEnd && laneReachesSeam(prev, rowKey, tempo, "end")) {
                applyFadeIn(fadeInNode, seg.startTick, Math.min(prevEnd, segEnd), 0, tempo, now, startDelay);
              }
            }

            const source = ctx.createBufferSource();
            source.buffer = buf;
            const envNode = loudnessEnvNode(ctx, seg, 0, tempo, now, startDelay, playDuration, seg.laneLoudness?.[group]);
            const laneTail = source.connect(fadeInNode).connect(fadeOutNode);
            (envNode ? laneTail.connect(envNode) : laneTail).connect(laneGainNode).connect(trackGainNode).connect(panner).connect(ctx.destination);
            source.start(now + startDelay, audioOffset, playDuration);
          }
        }
        continue;
      }

      if (seg.content.type !== "audioClip") continue; // defensive mirror of playback.ts
      const meta = audioFiles[seg.content.sourcePath];
      if (!meta) throw new Error(`EXPORT_SOURCE_MISSING: ${seg.content.sourcePath}`);
      const origStretchR = segStretch(seg);
      const srcPath = meta.playbackPath || seg.content.sourcePath;
      const bufPath = resolved.get(`${srcPath} ${origStretchR}`);
      const buf = bufPath ? buffers.get(bufPath) : undefined;
      if (!buf) throw new Error(`EXPORT_DECODE_FAIL: ${srcPath}`);

      let audioOffset = (seg.content.offsetMs * origStretchR) / 1000;
      let startDelay = ticksToSeconds(seg.startTick, tempo);
      let playDuration = ticksToSeconds(seg.durationTicks, tempo);
      audioOffset = Math.max(0, Math.min(audioOffset, buf.duration));
      playDuration = Math.min(playDuration, buf.duration - audioOffset);
      if (playDuration <= 0) continue;

      const trackGainNode = ctx.createGain();
      trackGainNode.gain.setValueAtTime(trackVol, now);
      const panner = ctx.createStereoPanner();
      panner.pan.value = track.pan;

      const fadeInNode = ctx.createGain();
      const fadeOutNode = ctx.createGain();
      if (si + 1 < sorted.length) {
        const next = sorted[si + 1]!;
        if (next.startTick < segEnd) {
          applyFadeOut(fadeOutNode, next.startTick, Math.min(segEnd, next.startTick + next.durationTicks), 0, tempo, now);
        }
      }
      if (si > 0) {
        const prev = sorted[si - 1]!;
        const prevEnd = prev.startTick + prev.durationTicks;
        if (seg.startTick < prevEnd) {
          applyFadeIn(fadeInNode, seg.startTick, Math.min(prevEnd, segEnd), 0, tempo, now, startDelay);
        }
      }

      const source = ctx.createBufferSource();
      source.buffer = buf;
      const envNode = loudnessEnvNode(ctx, seg, 0, tempo, now, startDelay, playDuration);
      const origTail = source.connect(fadeInNode).connect(fadeOutNode);
      (envNode ? origTail.connect(envNode) : origTail).connect(trackGainNode).connect(panner).connect(ctx.destination);
      source.start(now + startDelay, audioOffset, playDuration);
    }
  }

  // ── Render, with coarse progress via pre-registered suspend points (only worth it on longer bounces;
  // same-time suspends after the 128-frame quantization reject — swallowed, they're progress-only). ──
  if (onProgress && durationSec > 2) {
    const steps = Math.min(100, Math.max(4, Math.floor(durationSec)));
    for (let k = 1; k < steps; k++) {
      const t = (k / steps) * durationSec;
      ctx
        .suspend(t)
        .then(() => {
          onProgress(t / durationSec);
          void ctx.resume();
        })
        .catch(() => {});
    }
  }
  const rendered = await ctx.startRendering();
  onProgress?.(1);

  const L = rendered.getChannelData(0);
  const R = rendered.getChannelData(1);
  const pcm = new Float32Array(L.length * 2);
  let peak = 0;
  for (let i = 0; i < L.length; i++) {
    const l = L[i]!;
    const r = R[i]!;
    pcm[2 * i] = l;
    pcm[2 * i + 1] = r;
    const a = Math.abs(l);
    const b = Math.abs(r);
    if (a > peak) peak = a;
    if (b > peak) peak = b;
  }
  return { pcm, sampleRate, peak, durationSec };
}
