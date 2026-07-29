// S88 — the two lyric TRIGGERS (breath / rest). These predicates are THE single source for "what does
// this lyric mean", shared by the render feed, the OOV watcher, the editor overlay, the preview tone and
// the score export; a drift between any two of them shows up as "the drawn line does not match the audio".
import { describe, it, expect } from "vitest";
import {
  isBreathLyric,
  isRestLyric,
  isSilentLyric,
  splitLyricTokens,
  vocalTokens,
  sanitizeVocalParams,
  DEFAULT_BREATH_TOKEN,
  DEFAULT_REST_TOKEN,
} from "./vocalNotes";
import type { VocalTrackParams } from "../types/project";

const TK = (breath = DEFAULT_BREATH_TOKEN, rest = DEFAULT_REST_TOKEN) => ({ breath, rest });

describe("isRestLyric — the canonical set Rust hard-wires, plus the track trigger", () => {
  it("always accepts R / r / empty (g2p.rs token_class), whatever the trigger is set to", () => {
    for (const l of ["R", "r", "", "  "]) {
      expect(isRestLyric(l, DEFAULT_REST_TOKEN), l).toBe(true);
      expect(isRestLyric(l, "休"), l).toBe(true); // a custom trigger never REVOKES the canonical ones
    }
  });

  it("the words S86 freed stay singable — they are ordinary lyrics unless chosen as the trigger", () => {
    // S86 narrowed the reserved set precisely because these are real words (and word fragments: sil|ver,
    // pau|se). A regression here silently swallows notes again, which is what that round was about.
    for (const l of ["rest", "sil", "pau", "Rest", "ら"]) expect(isRestLyric(l, DEFAULT_REST_TOKEN), l).toBe(false);
  });

  it("the track trigger classifies exactly its own token (trimmed), nothing else", () => {
    expect(isRestLyric("休", "休")).toBe(true);
    expect(isRestLyric("  休 ", " 休")).toBe(true); // both sides are trimmed before comparing
    expect(isRestLyric("休符", "休")).toBe(false); // exact match only — never a prefix
    expect(isRestLyric("あ", "休")).toBe(false);
  });

  it("an EMPTY trigger disables only the custom half (clearing the box ≠ everything is a rest)", () => {
    expect(isRestLyric("あ", "")).toBe(false);
    expect(isRestLyric("R", "")).toBe(true);
  });
});

describe("isBreathLyric — unchanged by S88 (the rest token must not have moved its boundaries)", () => {
  it("still accepts only AP / ap / its own trigger", () => {
    for (const l of ["AP", "ap"]) expect(isBreathLyric(l, DEFAULT_BREATH_TOKEN), l).toBe(true);
    expect(isBreathLyric("呼", "呼")).toBe(true);
    for (const l of ["Ap", "aP", "apple", "", "R"]) expect(isBreathLyric(l, DEFAULT_BREATH_TOKEN), l).toBe(false);
  });
});

describe("isSilentLyric — the frontend twin of Rust g2p::is_silent_token", () => {
  it("covers both silences and nothing else", () => {
    expect(isSilentLyric("R", TK())).toBe(true);
    expect(isSilentLyric("AP", TK())).toBe(true);
    expect(isSilentLyric("か", TK())).toBe(false);
    expect(isSilentLyric("休", TK(undefined, "休"))).toBe(true);
    expect(isSilentLyric("呼", TK("呼"))).toBe(true);
  });

  it("both triggers pointed at the SAME word resolves to REST (the quieter reading)", () => {
    // Not an academic case: the sidebar has two free-text boxes and nothing stops a user typing the same
    // thing in both. The tie-break must be ONE decision — mapLyric in vocalRender makes the identical
    // choice, so the triple and the pitch chain can never disagree about such a note.
    expect(isRestLyric("同", "同")).toBe(true);
    expect(isSilentLyric("同", TK("同", "同"))).toBe(true);
  });
});

describe("vocalTokens — the one place the defaults live", () => {
  it("absent params / absent fields fall back to the canonical tokens", () => {
    expect(vocalTokens(undefined)).toEqual({ breath: "AP", rest: "R" });
    expect(vocalTokens({} as VocalTrackParams)).toEqual({ breath: "AP", rest: "R" });
  });
  it("an explicitly EMPTY token stays empty (only the canonical forms then trigger)", () => {
    const tk = vocalTokens({ breathToken: "", restToken: "" } as VocalTrackParams);
    expect(tk).toEqual({ breath: "", rest: "" });
    expect(isRestLyric("あ", tk.rest)).toBe(false);
  });
});

describe("sanitizeVocalParams — the two triggers from an untrusted .usp", () => {
  const base = { backend: "sovits", speakerId: 49, langId: 2, transpose: 0, formant: 0, transition: {} } as unknown as VocalTrackParams;

  it("materializes the rest token exactly like the breath token beside it (absent → canonical)", () => {
    const p = sanitizeVocalParams(base)!;
    expect(p.restToken).toBe("R");
    expect(p.breathToken).toBe("AP");
  });

  it("keeps a custom token, and falls back when nothing printable survives", () => {
    expect(sanitizeVocalParams({ ...base, restToken: "休" })!.restToken).toBe("休");
    expect(sanitizeVocalParams({ ...base, restToken: "   " })!.restToken).toBe("R"); // blank ≠ "no trigger"
    expect(sanitizeVocalParams({ ...base, restToken: 42 as unknown as string })!.restToken).toBe("R");
  });

  it("strips control/format code points, which a lyric could never contain anyway", () => {
    // Note lyrics all pass through sanitizeText, so an unsanitized token would be a trigger that can
    // never fire — a silent "my rest token does nothing" report with no visible cause.
    // built from a code point on purpose — an invisible literal in source is unreviewable and unstable.
    const zeroWidth = "休" + String.fromCharCode(0x200b);
    expect(sanitizeVocalParams({ ...base, restToken: zeroWidth })!.restToken).toBe("休");
  });
});

// S90 — the lyric splitter (§9.2 auto-distribute). It moved here from VocalEditor so it can be tested,
// and it had to learn about OpenUtau phonetic hints: a hint contains spaces but is ONE note's content.
describe("splitLyricTokens — whole-phrase distribution", () => {
  it("keeps the pre-S90 behaviour byte-for-byte when no bracket is involved", () => {
    expect(splitLyricTokens("and you don't", "あ")).toEqual(["and", "you", "don't"]);
    expect(splitLyricTokens("  a   b  ", "あ")).toEqual(["a", "b"]); // collapses runs, drops the edges
    expect(splitLyricTokens("長大", "あ")).toEqual(["長", "大"]); // one hanzi per note (S58)
    expect(splitLyricTokens("きゃっと", "あ")).toEqual(["きゃっ", "と"]); // per mora; EVERY small kana (っ too) attaches
    expect(splitLyricTokens("beautiful", "あ")).toEqual(["beautiful"]); // latin needs explicit spaces
    expect(splitLyricTokens("", "あ")).toEqual(["あ"]);
  });

  it("★ a phonetic hint stays ONE token — it contains spaces but belongs to a single note", () => {
    // Without this, typing what a UST file imports (`[dh ae dh]`) scattered `[dh` / `ae` / `dh]` across
    // three notes and painted all three OOV-red, while the identical text arriving through import
    // stayed whole. Two paths, two behaviours, no error message: exactly the asymmetry S88 warned about.
    expect(splitLyricTokens("[dh ae dh]", "あ")).toEqual(["[dh ae dh]"]);
    expect(splitLyricTokens("[ae n] you know [w ah dh]", "あ")).toEqual(["[ae n]", "you", "know", "[w ah dh]"]);
    expect(splitLyricTokens("read[r iy d] this", "あ")).toEqual(["read[r iy d]", "this"]);
    expect(splitLyricTokens("[k ae n d ah l ih t]", "あ")).toEqual(["[k ae n d ah l ih t]"]);
    expect(splitLyricTokens("［k ae］ you", "あ")).toEqual(["［k ae］", "you"]); // CJK IME brackets
  });

  it("an UNCLOSED bracket is not a hint here either — it splits like ordinary text (Rust then says OOV)", () => {
    expect(splitLyricTokens("[k aa}", "あ")).toEqual(["[k", "aa}"]);
    // the group must CLOSE its token, exactly like Rust's phoneme_hint — otherwise this splitter would
    // manufacture a hint out of text the renderer refuses (review S90)
    expect(splitLyricTokens("pre[a b]post", "あ")).toEqual(["pre[a", "b]post"]);
  });

  it("all-whitespace: `match` yields null, so the fallback token appears (commitLyric maps both to the default)", () => {
    expect(splitLyricTokens("   ", "あ")).toEqual(["あ"]); // the old split() gave [] — same note in the end
  });
});
