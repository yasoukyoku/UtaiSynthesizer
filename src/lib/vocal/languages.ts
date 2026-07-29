// ② Multi-language (S58 §3.7): THE single source for the 7 ScoreToCV languages. Everything language-
// related on the TS side goes through this module — track-header badge, sidebar selects, per-note
// overrides, buildVocalScore's per-note lang resolution — so the id↔code mapping can never drift from
// the Rust `Lang` enum (inference/g2p.rs) it mirrors. Labels are i18n keys (`langs.<code>`).

export interface VocalLanguage {
  /** ScoreToCV lang_id (the wire value — zh0 en1 ja2 de3 fr4 es5 it6). */
  id: number;
  /** The Note.lang override code (persisted on notes). */
  code: string;
  /** Two-letter badge for the crowded track header. */
  short: string;
  /** Default lyric for a NEWLY DRAWN note — must be singable in THIS language (a ja "あ" on a zh/en
   *  track would be instant OOV; audit MAJOR). Every value is verified against its dictionary. */
  defaultLyric: string;
}

export const VOCAL_LANGUAGES: readonly VocalLanguage[] = [
  { id: 0, code: "zh", short: "ZH", defaultLyric: "啊" },
  { id: 1, code: "en", short: "EN", defaultLyric: "a" },
  { id: 2, code: "ja", short: "JA", defaultLyric: "あ" },
  { id: 3, code: "de", short: "DE", defaultLyric: "a" },
  { id: 4, code: "fr", short: "FR", defaultLyric: "a" },
  { id: 5, code: "es", short: "ES", defaultLyric: "a" },
  { id: 6, code: "it", short: "IT", defaultLyric: "a" },
];

export const DEFAULT_LANG_ID = 2; // ja — the historical default (DEFAULT_VOCAL_PARAMS.langId)

const BY_CODE = new Map(VOCAL_LANGUAGES.map((l) => [l.code, l]));
const BY_ID = new Map(VOCAL_LANGUAGES.map((l) => [l.id, l]));

/** S91: default lyric for a newly drawn / emptied ENGLISH note on a track using a UTAU alias
 *  convention. Same contract as `defaultLyric` — it must be singable — but the constraint is now the
 *  CONVENTION, not the dictionary: the English default `a` is not an ARPABET symbol, so on an
 *  ARPAsing track the pen tool would mint notes that fail the whole segment render with VOCAL_ALIAS
 *  (review S91). `aa` is ARPABET for the same vowel; `a` is already legal in both alias tables. */
export function aliasDefaultLyric(set: string | undefined): string {
  return set === "arpasing" ? "aa" : "a";
}

/** True iff `code` is one of the 7 language codes (the Note.lang sanitize whitelist). */
export function isVocalLangCode(code: string): boolean {
  return BY_CODE.has(code);
}

/** Per-note effective lang id: the note's override code (when valid) else the track default id. */
export function effLangId(noteLang: string | undefined, defaultLangId: number): number {
  return (noteLang ? BY_CODE.get(noteLang)?.id : undefined) ?? defaultLangId;
}

/** Language for a lang_id (out-of-range falls back to ja — mirrors the Rust-side clamp). */
export function langById(id: number): VocalLanguage {
  return BY_ID.get(id) ?? BY_ID.get(DEFAULT_LANG_ID)!;
}
