import type { EsDialectId, Note } from "../../types/project";
import type { VocalTokens } from "../vocalNotes";
import { buildScoreTriples, type ScoreTriple } from "./vocalRender";

/**
 * EVERY input the read-only phoneme lane's preview depends on, as ONE named object.
 *
 * ⚠ Why this exists rather than an inline `JSON.stringify([...])` in the component: the lane caches its
 * last result under a signature and skips the IPC when the signature is unchanged. If an input reaches
 * `preview_vocal_phonemes` but NOT the signature, the lane keeps painting the previous answer — the
 * switch looks broken, or worse, silently shows the other arm's layout. S88 shipped exactly that shape
 * once (`AutoTuneWatcher`'s skip-sig omitted the two lyric tokens, so changing a token never
 * re-tuned — "the fix was dormant at precisely the moment it mattered"), and the S89 preroll switch is
 * the same hazard class: a mutation test proved the inline signature stayed green with the switch
 * removed from it.
 *
 * The defence is structural: both the signature and the IPC payload are derived from THIS object, and
 * `phonemeLane.test.ts` asserts that perturbing ANY field moves the signature. A future input can
 * therefore only be added by adding a field here — which the property test immediately holds to
 * account.
 */
export interface PhonemeLaneInputs {
  notes: Note[];
  tempo: number;
  /** Both lyric triggers: re-pointing either one re-resolves which notes produce phones at all. */
  tokens: VocalTokens;
  /** Track default language (per-note overrides ride on the notes). */
  langId: number;
  /** S89 「自动咬字时序」: changes WHERE every onset consonant sits. */
  consonantPreroll: boolean;
  /** S91 「音素约定」: changes WHICH phones every English note has at all. */
  phonemeSet?: string;
  /** S167 (§E4): changes WHICH phones every Spanish note has (θ/s · ʎ/ʝ). */
  esDialect?: EsDialectId;
}

/** The per-note phone-timing term of the lane signature — the edit changes the SPLIT the preview
 *  returns, so it must move the cache key (same hazard class as the S89 switch). */
function phoneTimingSig(n: Note): string {
  const pt = n.phoneTiming;
  return pt ? `${pt.phones.join(",")}~${pt.scale.join(",")}~${(pt.gainDb ?? []).join(",")}` : "";
}

/** The lane's cache key. Cheap: reads the notes' fields directly, no triple building. */
export function phonemeLaneSig(i: PhonemeLaneInputs): string {
  return JSON.stringify([
    i.notes.map((n) => [n.tick, n.duration, n.lyric, n.pitch, n.lang ?? "", n.phonemeInput ?? "", phoneTimingSig(n)]),
    i.langId,
    i.tokens.breath,
    i.tokens.rest,
    i.tempo,
    i.consonantPreroll,
    i.phonemeSet ?? "",
    i.esDialect ?? "",
  ]);
}

/** The `preview_vocal_phonemes` payload + the parallel arrays the lane needs to map spans back to notes.
 *  Built from the SAME inputs object as the signature — that is the whole point. */
export function phonemeLaneRequest(i: PhonemeLaneInputs): {
  args: {
    score: ScoreTriple[];
    defaultLang: number;
    consonantPreroll: boolean;
    phonemeSet: string | null;
    esDialect: string | null;
  };
  tripleNoteIds: (string | null)[];
  ticksPerFrame: number;
} {
  const { triples, tripleNoteIds, ticksPerFrame } = buildScoreTriples(i.notes, i.tempo, i.tokens, i.langId);
  return {
    args: {
      score: triples,
      defaultLang: i.langId,
      consonantPreroll: i.consonantPreroll,
      phonemeSet: i.phonemeSet ?? null,
      esDialect: i.esDialect ?? null,
    },
    tripleNoteIds,
    ticksPerFrame,
  };
}
