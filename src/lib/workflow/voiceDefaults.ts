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
] as const;
export type DiffusionMethod = (typeof DIFFUSION_METHODS)[number];

/** ①c speaker-blend entry: emb_g row `id` weighted by `weight` (≥0). The Rust pipeline
 * normalizes the stack to sum 1 and builds a dense `spk_mix` [1, n_spk] f32 vector fed in
 * place of the scalar `sid` — ONLY for a genuine multi-speaker export (the model's ONNX graph
 * carries a "spk_mix" input; single-speaker / pre-①c models ignore this and use speaker_id).
 * An empty stack degrades to a one-hot on `speaker_id` (byte-identical to picking one speaker). */
export interface SpkMixEntry {
  id: number;
  weight: number;
}

export interface RvcOptions {
  /** Pitch shift in semitones, -24..24. */
  f0_shift: number;
  /** Target speaker index for multi-speaker models; null = 0 (single-speaker default). */
  speaker_id: number | null;
  /** ①c speaker blend — non-empty ONLY for a genuine multi-speaker RVC export (α′, ONNX
   * "spk_mix" input). Empty = use speaker_id (single-speaker / pre-①c: byte-identical). */
  spk_mix: SpkMixEntry[];
  /** KNN index feature blend, 0..1. */
  index_ratio: number;
  /** Voiceless-consonant/breath protection, 0..0.5 — 0.5 means OFF. */
  protect: number;
  /** Synthesis randomness, 0..1. */
  noise_scale: number;
  /** Output-loudness envelope mix vs the input's, 0..1. */
  rms_mix_rate: number;
  /** L2-normalize ContentVec features before the index lookup (official pipeline does NOT). */
  l2_normalize: boolean;
  /** Output resample rate; 0 = keep the model's sample rate. */
  resample_sr: number;
  seed: number;
  /** Run the aux feature/f0 extractors (ContentVec + RMVPE) on the global GPU device instead
   * of the S35 forced-CPU default. Faster, but costs VRAM. */
  gpu_extract: boolean;
  /** ② 共振腔/formant — node-level SCALAR in semitones (post-decode formant_warp). 0 = no shift. */
  formant: number;
}

export interface SovitsOptions {
  /** Pitch shift in semitones. */
  f0_shift: number;
  /** Target speaker index for multi-speaker models; null = 0. */
  speaker_id: number | null;
  /** ①c speaker blend — a stack of {id, weight} rows. Non-empty ONLY for a genuine
   * multi-speaker export (ONNX "spk_mix" input). Empty = use speaker_id (byte-identical). */
  spk_mix: SpkMixEntry[];
  /** Synthesis randomness, 0..1. */
  noise_scale: number;
  /** Cluster-model / feature-index blend, 0..1; 0 = off. */
  cluster_ratio: number;
  /** Input-loudness-envelope replacement mix, 0..1 — 1.0 means OFF (keep output loudness). */
  loudness_envelope: number;
  seed: number;
  /** Shallow diffusion: VITS output → mel → k_step-noised → denoised → NSF-HiFiGAN vocoder.
   * Requires the model's `.diffusion/` attachment; mutually exclusive with nsf_enhance. */
  shallow_diffusion: boolean;
  /** Diffusion depth 1..1000 (≤ the diffusion model's k_step_max — Rust validates).
   * IGNORED by only_diffusion (full-depth generation, original semantics). */
  k_step: number;
  /** Sampler — one of DIFFUSION_METHODS. */
  diffusion_method: string;
  /** Step-skip factor: solver steps ≈ k_step / speedup. 1 disables acceleration
   * (falls back to the plain DDPM loop, same as "naive"). */
  diffusion_speedup: number;
  /** Skip VITS entirely — pure from-noise diffusion of the input (needs a full-depth
   * diffusion model, k_step_max == timesteps). */
  only_diffusion: boolean;
  /** Re-extract ContentVec from the VITS output before diffusing (原版「玄学选项」). */
  second_encoding: boolean;
  /** NSF-HiFiGAN enhancer on the plain VITS path — force-disabled while any diffusion
   * mode is on (original mutual exclusion). */
  nsf_enhance: boolean;
  /** Enhancer high-range adaptation in semitones (原版 enhancer_adaptive_key). */
  enhancer_adaptive_key: number;
  /** Automatic f0 prediction via the model's f0_decoder (`<stem>.f0.onnx`). Speech only —
   * singing will drift badly (original warning); f0_shift is largely neutralized. */
  auto_f0: boolean;
  /** Same as RvcOptions.gpu_extract. */
  gpu_extract: boolean;
  /** S40 vocoder resource: registry NAME of an installed NSF-HiFiGAN vocoder to use for
   * shallow diffusion + the enhancer; null = the built-in aux default. An unknown name is a
   * loud Rust error (never a silent fallback to the default). */
  vocoder_name: string | null;
  /** ② 共振腔/formant — node-level SCALAR in semitones (post-decode formant_warp). 0 = no shift. */
  formant: number;
}

export const RVC_DEFAULTS: RvcOptions = {
  f0_shift: 0,
  speaker_id: null,
  spk_mix: [],
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

export const SOVITS_DEFAULTS: SovitsOptions = {
  f0_shift: 0,
  speaker_id: null,
  spk_mix: [],
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
 * The params object persisted on an rvc/sovits WorkflowNode (`WorkflowNode.params` in
 * types/project.ts — untyped Record there; this documents the shape):
 *   - `voiceName`  — registry name of the model (list_models entry `.name`), invoke `voiceName`
 *   - `modelPath`  — the entry's `.path`, invoke `modelPath`
 *   - plus any subset of RvcOptions / SovitsOptions keys VERBATIM (absent = default above).
 */
export interface VoiceNodeParams extends Partial<RvcOptions & SovitsOptions> {
  voiceName?: string;
  modelPath?: string;
}

/**
 * Build the wire `options` object: contract defaults overlaid with any contract keys the node
 * params carry. ONLY keys present in `defaults` are emitted — node-side extras (voiceName,
 * modelPath, ...) never leak into the options payload.
 */
export function buildVoiceOptions<T extends object>(
  defaults: T,
  params: Record<string, unknown>,
): T {
  const out = { ...defaults } as Record<string, unknown>;
  for (const key of Object.keys(defaults)) {
    if (params[key] !== undefined) out[key] = params[key];
  }
  return out as T;
}
