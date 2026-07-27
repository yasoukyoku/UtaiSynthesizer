// ② S58 OOV validation watcher (§9.5/§9.8.3): whenever a vocal segment's lyrics / languages / phoneme
// overrides / breath token / tempo change (and once on install, so a freshly LOADED project shows its
// real verdicts), re-classify the segment through the Rust `validate_lyrics` command — the SAME
// resolve pass the render uses (single classifier: the marking can never drift from what renders) —
// and publish the OOV note ids to `useAppStore.vocalOov`. That map drives all three marking levels:
// red notes (VocalEditor), the segment badge (Arrangement) and the track-header warning (TrackList).
//
// Passive + cheap: debounced (fixed-delay, so a long lyric-typing burst still validates every ~300ms),
// a whole pass is skipped when the tracks ref / tempo are unchanged (playhead-only store updates), and
// per-segment input signatures skip unchanged segments. The validation payload is built by the SHARED
// buildScoreTriples (identical breath mapping + gap rests — a rest breaks a zh phrase window, so its
// presence changes polyphone verdicts). Failures log only — the render path reports loudly on its own.
import { invoke } from "@tauri-apps/api/core";
import { useProjectStore, DEFAULT_VOCAL_PARAMS } from "../../store/project";
import { useAppStore } from "../../store/app";
import { buildScoreTriples } from "./vocalRender";
import type { Segment, Track } from "../../types/project";

const DEBOUNCE_MS = 300;

/** Everything a segment's verdict depends on (NOT pitch/curves — they can't change a lyric's class).
 *  JSON-encoded: lyrics may contain any delimiter character, so a hand-rolled join could collide two
 *  different note lists into one sig (audit — a collision silently skips revalidation). */
function oovSig(track: Track, seg: Segment, tempo: number): string {
  if (seg.content.type !== "notes") return "";
  const vp = track.vocalParams ?? DEFAULT_VOCAL_PARAMS;
  return JSON.stringify([
    seg.content.notes.map((n) => [n.tick, n.duration, n.lyric, n.lang ?? "", n.phonemeInput ?? ""]),
    vp.langId,
    vp.breathToken ?? "AP",
    tempo,
  ]);
}

interface Validated {
  /** ref-identity short-circuit (a drag re-runs the pass every DEBOUNCE_MS — never rebuild the sig
   *  string for segments whose notes/params/tempo refs did not change). */
  notes: unknown;
  vp: unknown;
  tempo: number;
  sig: string;
}
const validated = new Map<string, Validated>(); // segmentId → last VALIDATED inputs
let timer: number | null = null;
let running = false;
let lastTracks: unknown = null;
let lastTempo: number | null = null;

async function validatePass(): Promise<void> {
  if (running) {
    schedule(); // serialize: re-run after the in-flight pass settles
    return;
  }
  const st = useProjectStore.getState();
  if (st.tracks === lastTracks && st.tempo === lastTempo) return; // playhead-only updates
  running = true;
  try {
    const app = useAppStore.getState();
    const live = new Set<string>();
    for (const tr of st.tracks) {
      for (const seg of tr.segments) {
        if (seg.content.type !== "notes") continue;
        live.add(seg.id);
        const prev = validated.get(seg.id);
        if (prev && prev.notes === seg.content.notes && prev.vp === tr.vocalParams && prev.tempo === st.tempo) continue;
        const sig = oovSig(tr, seg, st.tempo);
        const stamp: Validated = { notes: seg.content.notes, vp: tr.vocalParams, tempo: st.tempo, sig };
        if (prev?.sig === sig) {
          validated.set(seg.id, stamp); // same content through different refs (e.g. undo) — refresh refs
          continue;
        }
        if (seg.content.notes.length === 0) {
          validated.set(seg.id, stamp);
          app.setVocalOov(seg.id, null);
          continue;
        }
        const vp = tr.vocalParams ?? DEFAULT_VOCAL_PARAMS;
        const { triples, tripleNoteIds, droppedNoteIds } = buildScoreTriples(seg.content.notes, st.tempo, vp.breathToken ?? "AP", vp.langId);
        try {
          const classes = await invoke<Array<{ kind: string }>>("validate_lyrics", {
            notes: triples.map((t) => ({ lyric: t.lyric, lang: t.lang, phoneme_input: t.phoneme_input ?? null })),
            defaultLang: vp.langId,
          });
          // stale guard: the segment changed while we awaited → leave it to the rescheduled pass
          const now = useProjectStore.getState();
          const trNow = now.tracks.find((x) => x.id === tr.id);
          const segNow = trNow?.segments.find((x) => x.id === seg.id);
          if (!trNow || !segNow || oovSig(trNow, segNow, now.tempo) !== sig) continue;
          const oov: string[] = [];
          classes.forEach((c, i) => {
            const id = tripleNoteIds[i];
            if (id && c.kind === "unknown") oov.push(id);
          });
          // S84 D 刀: notes whose span rounded to ZERO frames will not sound — mark them through
          // the same channel (the marking's meaning is "this note cannot sing", true for both).
          oov.push(...droppedNoteIds);
          validated.set(seg.id, stamp);
          app.setVocalOov(seg.id, oov.length ? oov : null);
        } catch (e) {
          console.warn("[oovWatch] validate_lyrics failed:", e);
          validated.set(seg.id, stamp); // don't hot-loop on a persistent backend error
        }
      }
    }
    // prune verdicts of deleted segments (undo/load/delete)
    for (const id of [...validated.keys()]) {
      if (!live.has(id)) {
        validated.delete(id);
        app.setVocalOov(id, null);
      }
    }
    lastTracks = st.tracks;
    lastTempo = st.tempo;
    // a store change landed while the pass ran → catch it on the next tick
    if (useProjectStore.getState().tracks !== st.tracks) schedule();
  } finally {
    running = false;
  }
}

/** FIXED-DELAY debounce (no reset on retrigger): a continuous edit burst still validates every
 *  DEBOUNCE_MS instead of being starved until the burst ends. */
function schedule(): void {
  if (timer !== null) return;
  timer = window.setTimeout(() => {
    timer = null;
    void validatePass();
  }, DEBOUNCE_MS);
}

/** Install the watcher (App mounts it once). Returns the uninstall. */
export function installOovWatch(): () => void {
  const unsub = useProjectStore.subscribe(schedule);
  schedule(); // initial pass — a loaded project shows real verdicts immediately (§9.8.3)
  return () => {
    unsub();
    if (timer !== null) {
      window.clearTimeout(timer);
      timer = null;
    }
  };
}
