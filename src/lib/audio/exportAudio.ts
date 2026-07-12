// S63 — Audio-export orchestration: dirty-vocal pre-render (THE same funnel Play uses) → offline
// mixdown (exportMixdown.ts) → one raw-body IPC hop → Rust encode. UI-free; the ExportAudioDialog
// drives it and renders the phases.
import { invoke } from "@tauri-apps/api/core";
import i18n from "../../i18n";
import { useProjectStore } from "../../store/project";
import { useAudioStore } from "../../store/audio";
import { collectDirtyVocals, renderDirtyVocals } from "../vocal/vocalRender";
import { scoreExportableTracks } from "../vocal/exportScore";
import { renderMixdown } from "./exportMixdown";
import { logToBackend } from "../log";

/** [exportDbg] probes write to the crash-proof file log — the S63 live crash died silently (no panic,
 *  no WER), so the last probe line IS the crash locator on a repro. Remove once export is verified. */
const dbg = (m: string) => logToBackend("info", `[exportDbg] ${m}`);

/** PCM chunk size for the IPC transfer. DELIBERATELY small (8MB): Tauri's IPC silently falls back
 *  from the custom-protocol fetch to postMessage on a rejected fetch, and that path Array.from()s
 *  the bytes into a JS number array + JSON string — for a single ~100MB body that fallback OOM-kills
 *  the WebView2 renderer (the S63 crash). Per-chunk, a stray fallback costs ~30MB transient, not GBs. */
const PCM_CHUNK_BYTES = 8 * 1024 * 1024;

export interface AudioExportParams {
  outPath: string;
  format: "wav" | "flac" | "mp3" | "ogg" | "opus" | "m4a";
  sampleRate: number;
  bitDepth: "16" | "24" | "32f";
  bitrateKbps: number;
}

export type ExportPhase =
  | { kind: "vocals"; total: number }
  | { kind: "mix"; frac: number }
  | { kind: "encode" };

export interface AudioExportOutcome {
  cancelled: boolean;
  /** True float peak of the bounce (pre-quantization). > 1 ⇒ fixed-point/lossy encodes clipped. */
  peak: number;
  fileBytes: number;
  durationSec: number;
}

interface EncodeResult {
  out_path: string;
  duration_ms: number;
  file_bytes: number;
}

/**
 * Run the full export. Throws with a stable EXPORT_* message (renderMixdown's own codes, or the Rust
 * encode codes via backendError mapping) — EXCEPT vocal-render failures, which renderDirtyVocals has
 * already toasted per-track; those surface here as EXPORT_VOCALS_FAILED so the dialog can show a
 * summary line without double-toasting details.
 */
export async function runAudioExport(
  params: AudioExportParams,
  onPhase: (p: ExportPhase) => void,
  shouldCancel: () => boolean,
): Promise<AudioExportOutcome> {
  // 1. Bake dirty vocal tracks first — the bounce must contain what the user WOULD hear on Play.
  //    Same collect/render funnel as Toolbar's pre-play batch (single source; sequential, cancellable).
  const tempo0 = useProjectStore.getState().tempo;
  const dirty = collectDirtyVocals(tempo0);
  dbg(`start: format=${params.format} sr=${params.sampleRate} dirtyVocals=${dirty.length}`);
  if (dirty.length > 0) {
    onPhase({ kind: "vocals", total: dirty.length });
    const res = await renderDirtyVocals(dirty, tempo0, i18n.t("vocalEditor.render.laneLabel"), {
      shouldCancel,
    });
    if (res.cancelled || shouldCancel()) return { cancelled: true, peak: 0, fileBytes: 0, durationSec: 0 };
    // A failed bake means the bounce would silently diverge from the project — abort loudly.
    if (res.failed > 0) throw new Error("EXPORT_VOCALS_FAILED");
  }
  if (shouldCancel()) return { cancelled: true, peak: 0, fileBytes: 0, durationSec: 0 };

  // 2. Offline mixdown — read FRESH state (the bakes just deposited; tempo may not change mid-dialog,
  //    but the same fresh-read discipline as Toolbar's post-render play costs nothing).
  const st = useProjectStore.getState();
  onPhase({ kind: "mix", frac: 0 });
  dbg("mixdown start");
  let mix;
  try {
    mix = await renderMixdown(
      st.tracks,
      useAudioStore.getState().audioFiles,
      st.tempo,
      params.sampleRate,
      (frac) => onPhase({ kind: "mix", frac }),
    );
  } catch (e) {
    // "No audio content" on a project that HAS vocal notes means the notes never became audio (no
    // singer configured / silently skipped by the dirty collector) — point the user at the real cause
    // instead of the misleading generic empty (audit). Same predicate as the score-export track list.
    if (String(e).includes("EXPORT_EMPTY") && scoreExportableTracks(st.tracks).length > 0) {
      throw new Error("EXPORT_VOCALS_UNRENDERED");
    }
    throw e;
  }
  dbg(`mixdown done: ${mix.pcm.length} samples, peak=${mix.peak.toFixed(3)}`);
  if (shouldCancel()) return { cancelled: true, peak: 0, fileBytes: 0, durationSec: 0 };

  // 3. Ship the PCM in raw-body CHUNKS (see PCM_CHUNK_BYTES — never one giant body, and never a
  //    Vec<f32> JSON round trip, the S59 O5 lesson) + encode.
  onPhase({ kind: "encode" });
  try {
    const bytes = new Uint8Array(mix.pcm.buffer, mix.pcm.byteOffset, mix.pcm.byteLength);
    dbg(`pcm transfer begin: ${bytes.byteLength} bytes, ${Math.ceil(bytes.byteLength / PCM_CHUNK_BYTES)} chunks`);
    await invoke("export_audio_pcm_begin", { totalBytes: bytes.byteLength });
    for (let off = 0; off < bytes.byteLength; off += PCM_CHUNK_BYTES) {
      // slice() copies the 8MB window into a fresh, offset-0 buffer — a subarray view's byteOffset
      // is NOT preserved through the raw-body transport, and a stale view would resend the head.
      const chunk = bytes.slice(off, Math.min(off + PCM_CHUNK_BYTES, bytes.byteLength));
      await invoke("export_audio_pcm_chunk", chunk);
    }
    dbg("pcm transfer done, encoding");
    const res = await invoke<EncodeResult>("export_audio_encode", {
      outPath: params.outPath,
      format: params.format,
      sampleRate: params.sampleRate,
      bitDepth: params.bitDepth,
      bitrateKbps: params.bitrateKbps,
    });
    dbg(`encode done: ${res.file_bytes} bytes at ${res.out_path}`);
    return { cancelled: false, peak: mix.peak, fileBytes: res.file_bytes, durationSec: mix.durationSec };
  } catch (e) {
    dbg(`FAILED: ${e instanceof Error ? e.message : String(e)}`);
    // Free the Rust-side PCM stash on any failure between the hops (best-effort, idempotent).
    void invoke("export_audio_discard").catch(() => {});
    throw e;
  }
}
