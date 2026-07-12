// stretchCache.ts — S59 Tempo Slider: resolve (audio file, stretch factor) → the content-addressed
// stretched artifact produced by the Rust `stretch_segment_audio` command (32f WAV under
// audio_cache/stretch/). The Rust side is itself a cache (content hash + factor ⇒ same file), so
// this map is only a session-level memo + in-flight dedup; a cold entry (fresh .usp load, evicted
// cache) regenerates transparently on the first await — playback already awaits buffer decodes, a
// several-second first-play stretch is the same class of wait as a decode.

import { invoke } from "@tauri-apps/api/core";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { getLoadEpoch } from "../project/projectFile";

interface StretchResult {
  output_path: string;
  duration_ms: number;
  sample_rate: number;
  channels: number;
}

/** Canonical factor rounding — MUST match the store's setSegmentStretch rounding (1e-6) and the
 *  Rust cache key's {:.6} formatting, or the same factor would mint distinct artifacts. */
export function canonStretch(r: number): number {
  return Math.round(r * 1e6) / 1e6;
}

const resolved = new Map<string, string>(); // `${path}::${r}` → artifact path
const pending = new Map<string, Promise<string>>();

/** S61 cleanup support: every stretched-artifact path this session has resolved. The Settings
 *  render-cache sweep must NOT delete these — the memo above would keep returning the (deleted)
 *  path and stretched clips would fail to decode until an app restart. */
export function stretchedArtifactPaths(): string[] {
  return [...resolved.values()];
}

/** S61 cleanup support: a stretch is currently generating. Its content-addressed output path is
 *  minted Rust-side, so it CANNOT be added to the protected set — the cleanup must simply wait
 *  (a mid-flight sweep could delete the just-published wav before the memo records it, poisoning
 *  the memo with a dead path until restart; audit S61). */
export function stretchInFlight(): boolean {
  return pending.size > 0;
}

/** Apply a stretch factor to a segment: ARTIFACTS FIRST, THEN COMMIT — the source + every ready
 *  stem is stretched before the store write lands, so playback right after Apply never blocks on
 *  a cold stretch. Throws the Rust STRETCH_* CODE on failure (caller toasts); a stale segment
 *  (edited/removed while stretching) drops the commit silently. */
export async function applySegmentStretch(trackId: string, segmentId: string, rNew: number): Promise<void> {
  const factor = canonStretch(rNew);
  const p = useProjectStore.getState();
  const seg = p.tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
  if (!seg || seg.content.type !== "audioClip" || seg.loading) return;
  const c = seg.content;
  if (Math.abs(factor - (c.stretch ?? 1)) < 1e-9) return;
  const winSig = `${getLoadEpoch()}:${c.sourcePath}:${c.offsetMs}:${seg.durationTicks}:${c.stretch ?? 1}`;

  if (Math.abs(factor - 1) >= 1e-9) {
    const af = useAudioStore.getState().audioFiles[c.sourcePath];
    const paths = [
      af?.playbackPath ?? c.sourcePath,
      ...(seg.processedOutputs ?? []).filter((o) => !o.loading).map((o) => o.audioPath),
    ];
    // SEQUENTIAL on purpose: each stretch holds ~4 whole-file f32 copies in RAM (Rust in/out +
    // C++ planar in/out); a 5-stem long song in parallel spiked multiple GB (audit). Apply is an
    // explicit offline action — the extra wall time is fine.
    for (const path of paths) {
      await ensureStretched(path, factor);
    }
  }

  // stale re-check after the awaits (detectSegmentTempo's oovWatch pattern; epoch catches a
  // same-.usp reopen whose ids/values would otherwise match)
  const p2 = useProjectStore.getState();
  const seg2 = p2.tracks.find((t) => t.id === trackId)?.segments.find((s) => s.id === segmentId);
  if (!seg2 || seg2.content.type !== "audioClip") return;
  const c2 = seg2.content;
  if (`${getLoadEpoch()}:${c2.sourcePath}:${c2.offsetMs}:${seg2.durationTicks}:${c2.stretch ?? 1}` !== winSig) return;

  p2.setSegmentStretch(trackId, segmentId, factor);
  if (useAudioStore.getState().isPlaying) useAudioStore.getState().bumpSchedule();
}

/** Resolve the stretched artifact for `path` at factor `r` (passthrough when r ≈ 1). Throws the
 *  Rust STRETCH_* CODE on failure — callers map it (tempo.errFailed) or degrade. */
export async function ensureStretched(path: string, r: number): Promise<string> {
  const factor = canonStretch(r);
  if (Math.abs(factor - 1) < 1e-9) return path;
  const key = `${path}::${factor.toFixed(6)}`;
  const hit = resolved.get(key);
  if (hit) return hit;
  let p = pending.get(key);
  if (!p) {
    p = invoke<StretchResult>("stretch_segment_audio", { path, timeFactor: factor }).then(
      (res) => {
        resolved.set(key, res.output_path);
        pending.delete(key);
        return res.output_path;
      },
      (e) => {
        pending.delete(key);
        throw e;
      },
    );
    pending.set(key, p);
  }
  return p;
}
