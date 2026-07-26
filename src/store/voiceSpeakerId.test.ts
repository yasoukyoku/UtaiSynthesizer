import { describe, expect, it } from "vitest";

import { governingSpeakerId, vocalTrackSpeakerId, voiceHasRangeRecord } from "./voice-models";

/** A vocal_range record as the range test writes it: keyed by the SVC speaker (emb_g index). */
const tested = {
  config: { vocal_range: { speakers: { "0": { usable: [36, 77] } } } },
} as never;

describe("speaker id semantics (S81 drift guard)", () => {
  it("a vocal track resolves the SVC speaker, NOT the ScoreToCV one", () => {
    // VocalTrackParams carries BOTH ids one field apart:
    //   speakerId   = ScoreToCV conditioning speaker, 0-76, default 49 — content only,
    //                 decoupled from pitch, so it has no vocal_range record and never will;
    //   sovits/rvc.speaker_id = the SVC voice's emb_g speaker, which the record IS keyed by.
    // Passing the first where the second belongs looked up speakers["49"] and hid the
    // range-extend toggle on EVERY model, silently — the S81 field bug this pins.
    const vp = { backend: "sovits", speakerId: 49, sovits: {}, rvc: {} } as never;
    expect(vocalTrackSpeakerId(vp)).toBe(0);
    expect(voiceHasRangeRecord(tested, vocalTrackSpeakerId(vp))).toBe(true);
    // …and the shape of the original bug stays failing, so nobody reintroduces it.
    expect(voiceHasRangeRecord(tested, 49)).toBe(false);
  });

  it("honours an explicit SVC speaker and the dominant blend entry", () => {
    expect(vocalTrackSpeakerId({ backend: "sovits", sovits: { speaker_id: 3 } } as never)).toBe(3);
    expect(vocalTrackSpeakerId({ backend: "rvc", rvc: { speaker_id: 2 } } as never)).toBe(2);
    // a genuine blend is governed by its max-weight entry (mirrors Rust dominant_speaker)
    const blended = {
      backend: "sovits",
      sovits: { speaker_id: 0, spk_mix: [{ id: 1, weight: 0.2 }, { id: 5, weight: 0.8 }] },
    } as never;
    expect(vocalTrackSpeakerId(blended)).toBe(5);
    expect(governingSpeakerId(0, [{ id: 1, weight: 0.2 }, { id: 5, weight: 0.8 }])).toBe(5);
  });

  it("reads the backend's OWN options, not the other backend's", () => {
    // The two option bags coexist on the track; switching backend must switch which one governs.
    const vp = { backend: "rvc", sovits: { speaker_id: 7 }, rvc: { speaker_id: 2 } } as never;
    expect(vocalTrackSpeakerId(vp)).toBe(2);
    expect(vocalTrackSpeakerId({ ...(vp as object), backend: "sovits" } as never)).toBe(7);
  });

  it("an untested speaker of a partially tested model still reads as no record", () => {
    // Rust speaker_range never borrows another speaker's record — the gate must not either.
    expect(voiceHasRangeRecord(tested, 1)).toBe(false);
  });
});
