import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
export const useVoiceModelStore = create((set, get) => ({
    models: { rvc: [], sovits: [], vocoder: [] },
    error: null,
    rangeTesting: {},
    setRangeTesting: (name, progress) => set((s) => {
        const next = { ...s.rangeTesting };
        if (progress === null)
            delete next[name];
        else
            next[name] = progress;
        return { rangeTesting: next };
    }),
    rangeBatch: null,
    setRangeBatch: (b) => set({ rangeBatch: b }),
    bumpRangeBatch: (failedName) => set((s) => s.rangeBatch
        ? {
            rangeBatch: {
                ...s.rangeBatch,
                done: s.rangeBatch.done + 1,
                failed: failedName ? [...s.rangeBatch.failed, failedName] : s.rangeBatch.failed,
            },
        }
        : {}),
    cancelRangeBatch: () => set((s) => (s.rangeBatch ? { rangeBatch: { ...s.rangeBatch, cancel: true } } : {})),
    // Keep the finished shape around ONLY when something failed: the run's whole point is that the
    // user can see which models still need attention after a multi-minute unattended job. A clean
    // run clears itself so the row returns to its idle state with no dismissal chore.
    finishRangeBatch: () => set((s) => ({
        rangeBatch: s.rangeBatch && s.rangeBatch.failed.length ? { ...s.rangeBatch, finished: true } : null,
    })),
    auditionState: null,
    setAuditionState: (s) => set({ auditionState: s }),
    fetchModels: async () => {
        try {
            const [rvc, sovits, vocoder] = await Promise.all([
                invoke("list_models", { modelType: "rvc" }),
                invoke("list_models", { modelType: "sovits" }),
                invoke("list_models", { modelType: "vocoder" }),
            ]);
            set({ models: { rvc, sovits, vocoder } });
        }
        catch (e) {
            set({ error: String(e) });
        }
    },
    deleteModel: async (name, voiceType) => {
        try {
            await invoke("delete_model", { name, modelType: voiceType });
            await get().fetchModels();
        }
        catch (e) {
            set({ error: String(e) });
        }
    },
    setAvatar: async (name, avatarPath) => {
        try {
            await invoke("set_model_avatar", { name, avatarPath });
            await get().fetchModels();
        }
        catch (e) {
            set({ error: String(e) });
        }
    },
    clearError: () => set({ error: null }),
}));
/**
 * Display version badge: RVC "v1"/"v2", SoVITS "4.0"/"4.1" — straight from the model json's
 * `version`; RVC falls back to the features_dim heuristic (768 = v2, 256 = v1) for jsons
 * without one. Null = render no badge (never show "unknown").
 */
export function voiceVersionBadge(m) {
    const v = m.config?.version;
    if (v && v !== "unknown")
        return v;
    if (m.model_type === "Rvc") {
        if (m.config?.features_dim === 768)
            return "v2";
        if (m.config?.features_dim === 256)
            return "v1";
    }
    return null;
}
/**
 * A model's ContentVec feature dim (speech_encoder wins, features_dim
 * fallback) — the pairing key for shallow-diffusion companions: the diffusion
 * card derives its training version from it and the attach flow filters
 * candidates by it. Null = unknown (let the Rust side validate).
 */
export function voiceFeatureDim(m) {
    if (m.config?.speech_encoder === "vec768l12")
        return 768;
    if (m.config?.speech_encoder === "vec256l9")
        return 256;
    return m.config?.features_dim ?? null;
}
/**
 * Speaker dropdown options — EMPTY unless the model really is multi-speaker, driven by the
 * `speakers` NAME→id map (NOT n_speakers: that's the emb_g embedding-table row count, e.g. 109
 * for a single-speaker RVC voice / 200 for akiko — selecting a phantom id gathers an UNTRAINED
 * embedding row → silent garbage). RVC sidecars carry no map → single-speaker → no dropdown.
 * Options are sorted by id and labelled with the real speaker name.
 */
/** Minimum comfort/usable span (semitones) a range record must offer to be applicable — a
 *  degenerate zone becomes a centering target for everything (S60d: comfort=[42,42] →
 *  whole-song -27 st renders). THE single TS source (rangeTest.ts re-exports it); mirrors
 *  MIN_COMFORT_SPAN in src-tauri/src/inference/vocal_range.rs (read-side healing). */
export const MIN_COMFORT_SPAN = 5;
/** S60c/S62c: the model carries a USABLE tested range for the GOVERNING speaker — gates the
 * range-extend UI. Per-SPEAKER (S62c user-caught: the old any-speaker check showed the toggle
 * for untested speakers of a partially-tested model — Rust speaker_range never borrows another
 * speaker's record, so that toggle was the exact confusing no-op the gate exists to prevent)
 * and span-checked (mirrors the Rust read-side healing: usable narrower than MIN_COMFORT_SPAN
 * reads as "no record"). */
export function voiceHasRangeRecord(m, speakerId) {
    const rec = m?.config?.vocal_range;
    const sp = rec?.speakers?.[String(speakerId ?? 0)];
    const u = sp?.usable;
    if (!Array.isArray(u) || u.length < 2)
        return false;
    const lo = Number(u[0]);
    const hi = Number(u[1]);
    return Number.isFinite(lo) && Number.isFinite(hi) && hi - lo >= MIN_COMFORT_SPAN;
}
/** The speaker whose range record governs a render: the max-weight blend entry when a genuine
 *  spk_mix blend is set, else the plain speaker selection (mirrors Rust dominant_speaker). */
export function governingSpeakerId(speakerId, spkMix) {
    if (spkMix && spkMix.length > 0) {
        return spkMix.reduce((a, b) => (b.weight > a.weight ? b : a)).id;
    }
    return speakerId ?? 0;
}
/** THE SVC speaker a VOCAL TRACK (自己唱) renders with — the mirror of the Rust selection in
 *  commands/inference.rs (`dominant_speaker(options.<backend>.spk_mix, ....speaker_id)`).
 *
 *  ★ NOT `VocalTrackParams.speakerId`. That one is the **ScoreToCV** conditioning speaker
 *  (0–76, default 49): content only, deliberately decoupled from pitch, so it has no
 *  vocal_range record and never will. Passing it where an SVC speaker id belongs looks up
 *  `speakers["49"]`, which cannot exist — S81 field bug: every vocal track hid its range-extend
 *  toggle even for models whose range HAD been tested, on every model, silently. The two ids
 *  live in the same struct one field apart, so resolve the SVC one HERE and nowhere else. */
export function vocalTrackSpeakerId(vp) {
    const o = vp.backend === "sovits" ? vp.sovits : vp.rvc;
    return governingSpeakerId(o?.speaker_id, o?.spk_mix);
}
export function voiceSpeakerOptions(m) {
    const map = m.config?.speakers;
    if (!map || typeof map !== "object" || Array.isArray(map))
        return [];
    const entries = Object.entries(map).map(([label, id]) => ({ id: Number(id), label }));
    if (entries.length <= 1)
        return []; // single real speaker → default 0, no dropdown
    return entries.sort((a, b) => a.id - b.id);
}
/** "40kHz" / "44.1kHz" — shared by the node meta rows and the resource manager list. */
export function formatSampleRateKhz(sr) {
    return sr % 1000 === 0 ? `${sr / 1000}kHz` : `${(sr / 1000).toFixed(1)}kHz`;
}
/** Whether the model has a converted `.diffusion/` attachment (gates 浅扩散/仅扩散 UI). */
export function voiceHasDiffusion(m) {
    return !!m?.diffusion_path;
}
/** Whether the model's export includes the automatic-f0 predictor (gates 自动音高预测 UI).
 * Truth source = converter-written sidecar key (weight-derived), NOT the model config. */
export function voiceHasAutoF0(m) {
    return m?.config?.auto_f0?.available === true;
}
/** ①c: whether the model is a GENUINE multi-speaker export whose ONNX graph takes a `spk_mix`
 * blend input (converter writes `spk_mix.available` ONLY for len(speakers) > 1). This — NOT
 * merely voiceSpeakerOptions().length > 1 — gates the blend-stack UI: a pre-①c multi-speaker
 * model has a speaker map but a scalar `sid` input, so it uses the plain SpeakerSelect instead.
 * Requires ≥2 named speakers too (the blend needs real names to pick from). */
export function voiceHasSpkMix(m) {
    return m?.config?.spk_mix?.available === true && voiceSpeakerOptions(m).length > 1;
}
/** 一期唯一声码器格式类 = the OpenVPI standard (the aux default vocoder's recipe;
 * every SoVITS diffusion attachment / the enhancer mel is anchored to it).
 * MIRRORS the Rust constants in commands/inference.rs (VOCODER_STD_*) — the Rust
 * side re-validates strictly at run time, this filter only decides visibility. */
export const VOCODER_STD_FORMAT = {
    sample_rate: 44100,
    hop_size: 512,
    n_fft: 2048,
    win_size: 2048,
    num_mels: 128,
    fmin: 40,
    fmax: 16000,
};
/** Whether an installed vocoder's sidecar recipe matches the standard format class —
 * mismatches are HIDDEN from the node dropdown (不能选隐藏), the resource list
 * explains the format instead. Missing fields = unverifiable = no match. */
export function vocoderFormatMatches(m) {
    const c = m.config;
    if (!c)
        return false;
    return Object.keys(VOCODER_STD_FORMAT).every((k) => Number(c[k]) === VOCODER_STD_FORMAT[k]);
}
/** "44.1kHz · hop 512 · 128 mel" — the resource-list format line for a vocoder row. */
export function vocoderFormatLabel(m) {
    const c = m.config ?? {};
    const sr = typeof c.sample_rate === "number" ? c.sample_rate : m.sample_rate;
    const hop = c.hop_size ?? "?";
    const mels = c.num_mels ?? "?";
    return `${formatSampleRateKhz(sr)} · hop ${hop} · ${mels} mel`;
}
