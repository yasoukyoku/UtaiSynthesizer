/**
 * THE voice-node parameter contract — the single source of truth shared by the node UIs
 * (RvcNode / SoVitsNode) and the workflow engine, and the exact shape the Rust pipeline's
 * options deserialization must mirror (run_rvc / run_sovits in src-tauri).
 *
 * The engine serializes EXACTLY these snake_case keys as the `options` object of the invoke
 * payload `{ voiceName, modelPath, audioPath, options }` — nothing else. (S36: the SoVITS
 * quality path — shallow diffusion / only_diffusion / second_encoding / NSF enhancer /
 * auto-f0 — is now wired. S46 ①c: `spk_mix` (speaker-blend) is wired for genuine
 * multi-speaker SoVITS exports — see SpkMixEntry below. Rust also accepts a test-only
 * `debug_zero_noise` key that deliberately has NO entry here — gate harnesses only.)
 *
 * Node params store the SAME snake_case keys (plus `voiceName` / `modelPath`), so there is no
 * UI-key → wire-key mapping layer to drift: an absent key means "use the default below".
 * f0 method is rmvpe-only for now — no selector param.
 */
/** Diffusion sampler ids — EXACT wire strings the Rust pipeline matches on (original
 * so-vits-svc method names; "naive" = the plain DDPM p_sample fallback branch). */
export const DIFFUSION_METHODS = [
    "dpm-solver++",
    "dpm-solver",
    "unipc",
    "pndm",
    "ddim",
    "naive",
];
export const RVC_DEFAULTS = {
    f0_shift: 0,
    speaker_id: null,
    spk_mix: [],
    range_extend: false,
    range_formant_follow: 0,
    index_ratio: 0.75,
    protect: 0.33,
    noise_scale: 0.66666,
    rms_mix_rate: 0.25,
    l2_normalize: false,
    resample_sr: 0,
    seed: 0,
    gpu_extract: false,
    formant: 0,
};
export const SOVITS_DEFAULTS = {
    f0_shift: 0,
    speaker_id: null,
    spk_mix: [],
    range_extend: false,
    range_formant_follow: 0,
    noise_scale: 0.4,
    cluster_ratio: 0,
    loudness_envelope: 1.0,
    seed: 0,
    // Quality path (S36). k_step 100 / dpm-solver++ / speedup 10 = the original template
    // defaults (Svc.infer k_step + configs_template/diffusion_template.yaml infer block).
    shallow_diffusion: false,
    k_step: 100,
    diffusion_method: "dpm-solver++",
    diffusion_speedup: 10,
    only_diffusion: false,
    second_encoding: false,
    nsf_enhance: false,
    enhancer_adaptive_key: 0,
    auto_f0: false,
    gpu_extract: false,
    vocoder_name: null,
    formant: 0,
};
/**
 * Build the wire `options` object: contract defaults overlaid with any contract keys the node
 * params carry. ONLY keys present in `defaults` are emitted — node-side extras (voiceName,
 * modelPath, ...) never leak into the options payload.
 */
export function buildVoiceOptions(defaults, params) {
    const out = { ...defaults };
    for (const key of Object.keys(defaults)) {
        if (params[key] !== undefined)
            out[key] = params[key];
    }
    return out;
}
