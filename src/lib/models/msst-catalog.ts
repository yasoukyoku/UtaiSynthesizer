export type MsstArchitecture = "bs_roformer" | "mel_band_roformer" | "mdx23c" | "htdemucs" | "uvr_vr" | "mdx_net";
export type MsstCategory = "vocals" | "instrumental" | "denoise" | "dereverb" | "karaoke" | "multistem" | "special";

export interface I18nText { zh: string; en: string; ja: string }

export interface MsstCatalogEntry {
  id: string;
  name: I18nText;
  description: I18nText;
  category: MsstCategory;
  architecture: MsstArchitecture;
  filename: string;
  fileSize: number;
  stems: string[];
  sdrScore?: number;
  downloadUrl: string;
  configUrl?: string;
  source: "official" | "community";
}

/** Pick the active-language string from an I18nText. ONE source of truth for the zh/ja/en dispatch
 *  (previously copy-pasted as t18/l in SeparationNode, NodePalette, MsstModelManager). */
export function t18(text: I18nText, lang: string): string {
  if (lang === "zh") return text.zh;
  if (lang === "ja") return text.ja;
  return text.en;
}

export const CATEGORY_LABELS: Record<MsstCategory, I18nText> = {
  vocals:       { zh: "提取人声", en: "Vocals", ja: "ボーカル抽出" },
  instrumental: { zh: "提取伴奏", en: "Instrumental", ja: "伴奏抽出" },
  denoise:      { zh: "去噪", en: "Denoise", ja: "ノイズ除去" },
  dereverb:     { zh: "去混响", en: "De-Reverb", ja: "リバーブ除去" },
  karaoke:      { zh: "卡拉OK", en: "Karaoke", ja: "カラオケ" },
  multistem:    { zh: "多轨道", en: "Multi-Stem", ja: "マルチステム" },
  special:      { zh: "特殊", en: "Special", ja: "特殊" },
};

/** Per-category accent color (was duplicated identically in SeparationNode + NodePalette). */
export const CATEGORY_COLORS: Record<MsstCategory, string> = {
  vocals: "#ec4899",
  instrumental: "#60a5fa",
  denoise: "#4ade80",
  dereverb: "#a78bfa",
  karaoke: "#fbbf24",
  multistem: "#f97316",
  special: "#94a3b8",
};

export const ARCHITECTURE_LABELS: Record<MsstArchitecture, string> = {
  bs_roformer: "BS-Roformer",
  mel_band_roformer: "MelBand Roformer",
  mdx23c: "MDX23C",
  htdemucs: "HTDemucs",
  uvr_vr: "VR Arch",
  mdx_net: "MDX-Net",
};

/** Per-architecture default `num_overlap` — MUST mirror the converter/convert.py FALLBACK it
 *  writes into each model's JSON when the yaml has no explicit inference.num_overlap. Since S34
 *  this is ONLY the pre-install display fallback: for INSTALLED models the node's slider shows
 *  the json's real num_overlap (list_msst_models returns it), so yaml-carrying models (e.g.
 *  Kim-family melbands, yaml=2) no longer display a value Rust doesn't actually run.
 *  htdemucs = 2: official demucs weights have signature-only yamls, and the authors' own
 *  apply_model runs overlap=0.25 (~1.33x coverage) — ov4 was 2.95x that compute for nothing. */
export const MSST_DEFAULT_NUM_OVERLAP: Record<MsstArchitecture, number> = {
  bs_roformer: 4,
  mel_band_roformer: 4,
  mdx23c: 4,
  htdemucs: 2,
  // uvr_vr uses window-stride inference — the node hides the overlap slider, so this value is
  // unused (present only to keep the Record exhaustive).
  uvr_vr: 4,
  mdx_net: 2,
};

export type MsstPrecision = "fp32" | "fp16";

/** Per-architecture default inference precision. fp16 ≈ 2x faster + half VRAM/size, 63-70 dB vs
 *  fp32 (inaudible in practice; fused-attention exports, S33 re-gate). MelBand keeps fp16 as the
 *  default for speed — since the S33 attention fusion its fp32 also FITS 12GB cards (~6.3GB peak;
 *  pre-fusion exports saturated 12GB into WDDM paging, so old installs need a re-download/补转 to
 *  get the fused graph). mdx23c/htdemucs stay fp32 by default (conv archs, smaller fp16 gain). */
export const MSST_DEFAULT_PRECISION: Record<MsstArchitecture, MsstPrecision> = {
  bs_roformer: "fp32",
  mel_band_roformer: "fp16",
  mdx23c: "fp32",
  htdemucs: "fp32",
  uvr_vr: "fp32",
  mdx_net: "fp32",
};

/** Archs the converter can produce fp16 for (SNR-gated per arch on CUDA: bs 65.8/70.2 dB,
 *  melband 67.8/63.0 dB — S33 fused-attention exports, ~+9-12 dB over the old decomposed graphs
 *  whose fp16 computed rotary angles in fp16; mdx23c 71.0/75.5 dB, htdemucs 52.9-56.8 dB
 *  non-quiet stems vs fp32).
 *  S68c re-gate (norm-stats fp32 protection recipe, onnx_fp16.py): bs 72.8/67.9 dB,
 *  melband 68.7/63.5 dB (≤0.1 dB vs the unprotected recipe); mdx23c byte-identical;
 *  htdemucs untouched (own S31 path, collapse-exempt). Roformer fp16 files converted by
 *  older builds can go ALL-NaN on silent/near-silent chunks on GPU EPs (fp16 norm stats +
 *  a Clip floor squashed to 0 en route) — the manager's 重转 fp16 button regenerates them.
 *  ONE source of truth for every fp16 choice in the UI (download precision, 补转 actions).
 *  MUST mirror converter/convert.py FP16_VERIFIED_TYPES. Gate rule: fp16 verification MUST run
 *  on the CUDA EP — the CPU EP emulates fp16 in fp32 and false-passes (htdemucs NaN case).
 *  uvr_vr / mdx_net deliberately excluded — fp16 not CUDA-gated for them yet. */
export const MSST_FP16_ARCHS: ReadonlySet<MsstArchitecture> = new Set(["bs_roformer", "mel_band_roformer", "mdx23c", "htdemucs"]);

/** fp16 tradeoff copy shared by the model manager (download / 补转 tooltips) and the separation
 *  node's precision row — measured facts, keep in ONE place. */
export const MSST_FP16_TIP: I18nText = {
  zh: "fp16：速度约 2 倍、显存与体积减半；与 fp32 相比 63-70 dB（听感无差异）",
  en: "fp16 — about 2x faster at half the VRAM/size; 63-70 dB vs fp32 (inaudible)",
  ja: "fp16 — 約2倍高速、VRAM/サイズ半減。fp32 比 63-70 dB（聴感上の差なし）",
};

export const ALL_CATEGORIES: MsstCategory[] = [
  "vocals", "instrumental", "denoise", "dereverb", "karaoke", "multistem", "special",
];

const HF = "https://huggingface.co";
const MSST = `${HF}/Eddycrack864/Music-Source-Separation-Training/resolve/main`;
const UVR = `${HF}/Politrees/UVR_resources/resolve/main/models/MDX23C`;
const GH_CFG = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs";
// nomadkaraoke's audio-separator model-configs release — mirror host for several models whose
// original repos are dead/gated, and the canonical surviving source for the jarredou VR model.
const ASEP_GH = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs";
const FB_CDN = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer";
// Official UVR download backend (the UVR app itself downloads from here) — VR/MDX-Net weights
// plus the demucs yaml configs.
const UVR_GH = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models";
const GH_ZFT = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download";
const GH_ZFT_CFG = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs";

export const MSST_CATALOG: MsstCatalogEntry[] = [
  // ════════════════════════════════════════
  //  VOCALS — 提取人声
  // ════════════════════════════════════════
  {
    id: "bs_ep317", category: "vocals", architecture: "bs_roformer", source: "official",
    name: { zh: "BS-Roformer 12.98", en: "BS-Roformer 12.98", ja: "BS-Roformer 12.98" },
    description: { zh: "最佳人声分离模型，SDR 12.98，社区标准", en: "Best vocal model, SDR 12.98, community standard", ja: "最高性能ボーカル分離、SDR 12.98、コミュニティ標準" },
    filename: "model_bs_roformer_ep_317_sdr_12.9755.ckpt", fileSize: 1_597_000_000,
    stems: ["Vocals", "Instrumental"], sdrScore: 12.98,
    downloadUrl: `${MSST}/model_bs_roformer_ep_317_sdr_12.9755.ckpt`,
    configUrl: `${GH_CFG}/model_bs_roformer_ep_317_sdr_12.9755.yaml`,
  },
  {
    id: "bs_ep368", category: "vocals", architecture: "bs_roformer", source: "official",
    name: { zh: "BS-Roformer 12.96", en: "BS-Roformer 12.96", ja: "BS-Roformer 12.96" },
    description: { zh: "人声分离 V2，SDR 12.96", en: "Vocal separation V2, SDR 12.96", ja: "ボーカル分離 V2、SDR 12.96" },
    filename: "model_bs_roformer_ep_368_sdr_12.9628.ckpt", fileSize: 1_597_000_000,
    stems: ["Vocals", "Instrumental"], sdrScore: 12.96,
    downloadUrl: `${MSST}/model_bs_roformer_ep_368_sdr_12.9628.ckpt`,
    configUrl: `${GH_CFG}/model_bs_roformer_ep_368_sdr_12.9628.yaml`,
  },
  {
    id: "mel_vocals_3005", category: "vocals", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 人声 11.44", en: "MelBand Vocals 11.44", ja: "MelBand ボーカル 11.44" },
    description: { zh: "标准 MelBand 人声模型，SDR 11.44", en: "Standard MelBand vocal model, SDR 11.44", ja: "標準 MelBand ボーカル、SDR 11.44" },
    filename: "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt", fileSize: 1_600_000_000,
    stems: ["Vocals", "Instrumental"], sdrScore: 11.44,
    downloadUrl: `${MSST}/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt`,
    configUrl: `${MSST}/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml`,
  },
  {
    id: "mel_vocals_kim", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 人声 Kim", en: "MelBand Vocals Kim", ja: "MelBand ボーカル Kim" },
    description: { zh: "KimberleyJSN 原版人声模型", en: "KimberleyJSN original vocal model", ja: "KimberleyJSN オリジナルボーカルモデル" },
    filename: "MelBandRoformer.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt`,
    // Author repo hosts no config; this is the pairing documented by ZFTurbo's pretrained_models.md.
    configUrl: `${GH_ZFT_CFG}/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml`,
  },
  {
    id: "mel_vocals_kim_ft", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 人声 Kim FT", en: "MelBand Vocals Kim FT", ja: "MelBand ボーカル Kim FT" },
    description: { zh: "Kim 微调版 (unwa)，适合流行音乐", en: "Kim fine-tuned by unwa, good for pop", ja: "Kim ファインチューン版 (unwa)、ポップス向け" },
    filename: "kimmel_unwa_ft.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt`,
    // Author recipe: chunk 485100 / num_overlap 8 (slower than the Kim original's 352800/2).
    configUrl: `${HF}/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml`,
  },
  {
    id: "mel_vocals_big_syhft_v1", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Big SYHFT V1", en: "MelBand Big SYHFT V1", ja: "MelBand Big SYHFT V1" },
    description: { zh: "大型模型，更高精度人声分离", en: "Large model, higher precision vocals", ja: "大型モデル、高精度ボーカル分離" },
    filename: "MelBandRoformerBigSYHFTV1.ckpt", fileSize: 3_200_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/MelBandRoformerBigSYHFTV1.ckpt`,
    configUrl: `${HF}/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/config.yaml`,
  },
  {
    id: "mel_vocals_big_beta4", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Big Beta4", en: "MelBand Big Beta4", ja: "MelBand Big Beta4" },
    description: { zh: "大型模型 Beta4 版 (unwa)", en: "Large model Beta4 by unwa", ja: "大型モデル Beta4 版 (unwa)" },
    filename: "melband_roformer_big_beta4.ckpt", fileSize: 3_200_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt`,
    // beta4 is the depth-12 odd-one-out in this repo — the generic config_melbandroformer_big.yaml
    // (depth 6) would silently misconvert it. This yaml is named for beta4 specifically.
    configUrl: `${HF}/pcunwa/Mel-Band-Roformer-big/resolve/main/config_melbandroformer_big_beta4.yaml`,
  },
  {
    id: "mel_vocals_big_beta5e", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Big Beta5e", en: "MelBand Big Beta5e", ja: "MelBand Big Beta5e" },
    description: { zh: "大型模型最新版 (unwa)，推荐", en: "Latest large model by unwa, recommended", ja: "大型モデル最新版 (unwa)、推奨" },
    filename: "big_beta5e.ckpt", fileSize: 3_200_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt`,
    configUrl: `${HF}/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml`,
  },
  {
    id: "mel_vocals_syhft_v2", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "SYHFT V2", en: "SYHFT V2", ja: "SYHFT V2" },
    description: { zh: "SYH99999 人声模型 V2", en: "SYH99999 vocal model V2", ja: "SYH99999 ボーカルモデル V2" },
    filename: "MelBandRoformerSYHFTV2.ckpt", fileSize: 1_600_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/SYH99999/MelBandRoformerSYHFTV2/resolve/main/MelBandRoformerSYHFTV2.ckpt`,
    // Family config hosted by the same author in the SYHFT V1 repo (V2/V2.5 repos ship no yaml).
    configUrl: `${HF}/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml`,
  },
  {
    id: "mel_vocals_syhft_v25", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "SYHFT V2.5", en: "SYHFT V2.5", ja: "SYHFT V2.5" },
    description: { zh: "SYH99999 人声模型 V2.5，最新", en: "SYH99999 vocal model V2.5, latest", ja: "SYH99999 ボーカルモデル V2.5、最新" },
    filename: "MelBandRoformerSYHFTV2.5.ckpt", fileSize: 1_600_000_000,
    stems: ["Vocals", "Instrumental"],
    // The author's HF repo is malformed (ckpt nested inside a same-named folder → the flat URL
    // 404s); this GH mirror is byte-identical and flat.
    downloadUrl: `${ASEP_GH}/MelBandRoformerSYHFTV2.5.ckpt`,
    configUrl: `${HF}/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml`,
  },
  {
    id: "mel_vocals_duality_v1", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Duality V1", en: "MelBand Duality V1", ja: "MelBand Duality V1" },
    description: { zh: "Duality 人声模型 V1 (unwa)", en: "Duality vocal model V1 by unwa", ja: "Duality ボーカルモデル V1 (unwa)" },
    filename: "melband_roformer_instvoc_duality_v1.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt`,
    configUrl: `${HF}/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml`,
  },
  {
    id: "mel_vocals_duality_v2", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Duality V2", en: "MelBand Duality V2", ja: "MelBand Duality V2" },
    description: { zh: "Duality 人声模型 V2 (unwa)", en: "Duality vocal model V2 by unwa", ja: "Duality ボーカルモデル V2 (unwa)" },
    filename: "melband_roformer_instvox_duality_v2.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt`,
    // One family config covers v1+v2 (identical arch, author hosts a single yaml).
    configUrl: `${HF}/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml`,
  },
  {
    id: "mdx23c_hq", category: "vocals", architecture: "mdx23c", source: "community",
    name: { zh: "MDX23C 人声 HQ", en: "MDX23C Vocals HQ", ja: "MDX23C ボーカル HQ" },
    description: { zh: "MDX23C 8K FFT 高品质人声分离", en: "MDX23C 8K FFT high-quality vocals", ja: "MDX23C 8K FFT 高品質ボーカル" },
    filename: "MDX23C-8KFFT-InstVoc_HQ.ckpt", fileSize: 900_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${UVR}/MDX23C-8KFFT-InstVoc_HQ.ckpt`,
    configUrl: `${GH_CFG}/model_2_stem_full_band_8k.yaml`,
  },
  {
    id: "mdx23c_hq2", category: "vocals", architecture: "mdx23c", source: "community",
    name: { zh: "MDX23C 人声 HQ 2", en: "MDX23C Vocals HQ 2", ja: "MDX23C ボーカル HQ 2" },
    description: { zh: "MDX23C 高品质 V2", en: "MDX23C HQ V2, improved", ja: "MDX23C 高品質 V2" },
    filename: "MDX23C-8KFFT-InstVoc_HQ_2.ckpt", fileSize: 900_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${UVR}/MDX23C-8KFFT-InstVoc_HQ_2.ckpt`,
    configUrl: `${GH_CFG}/model_2_stem_full_band_8k.yaml`,
  },
  {
    id: "mdx23c_d1581", category: "vocals", architecture: "mdx23c", source: "community",
    name: { zh: "MDX23C D1581", en: "MDX23C D1581", ja: "MDX23C D1581" },
    description: { zh: "MDX23C D1581 人声模型", en: "MDX23C D1581 vocal model", ja: "MDX23C D1581 ボーカル" },
    filename: "MDX23C_D1581.ckpt", fileSize: 900_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${UVR}/MDX23C_D1581.ckpt`,
    // UVR pairs D1581 with model_2_stem_061321 (12288-FFT family) — the 8K-FFT yaml previously
    // referenced here belongs to the InstVoc HQ models and would misconvert this one.
    configUrl: `${GH_CFG}/model_2_stem_061321.yaml`,
  },

  // ════════════════════════════════════════
  //  INSTRUMENTAL — 提取伴奏
  // ════════════════════════════════════════
  {
    id: "mel_inst_v2", category: "instrumental", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 伴奏 V2", en: "MelBand Inst V2", ja: "MelBand 伴奏 V2" },
    description: { zh: "最佳伴奏提取，SDR 16.1，推荐", en: "Best instrumental, SDR 16.1, recommended", ja: "最高性能伴奏抽出、SDR 16.1、推奨" },
    filename: "melband_roformer_inst_v2.ckpt", fileSize: 1_500_000_000,
    stems: ["Instrumental", "Vocals"], sdrScore: 16.1,
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt`,
    configUrl: `${HF}/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst_v2.yaml`,
  },
  {
    id: "mel_inst_v1e", category: "instrumental", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 伴奏 V1e", en: "MelBand Inst V1e", ja: "MelBand 伴奏 V1e" },
    description: { zh: "伴奏提取 V1 增强版 (unwa)", en: "Instrumental V1 enhanced by unwa", ja: "伴奏抽出 V1 強化版 (unwa)" },
    filename: "inst_v1e.ckpt", fileSize: 1_500_000_000,
    stems: ["Instrumental", "Vocals"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt`,
    // v1 family shares the depth-6 config; the repo's other yaml (inst_v2) is depth-12 — wrong here.
    configUrl: `${HF}/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml`,
  },
  {
    id: "mel_inst_v1", category: "instrumental", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 伴奏 V1", en: "MelBand Inst V1", ja: "MelBand 伴奏 V1" },
    description: { zh: "伴奏提取 V1 (unwa)", en: "Instrumental V1 by unwa", ja: "伴奏抽出 V1 (unwa)" },
    filename: "melband_roformer_inst_v1.ckpt", fileSize: 1_500_000_000,
    stems: ["Instrumental", "Vocals"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt`,
    configUrl: `${HF}/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml`,
  },
  {
    id: "mel_bleed_suppress", category: "instrumental", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 泄漏抑制", en: "MelBand Bleed Suppressor", ja: "MelBand リーク抑制" },
    description: { zh: "抑制分离后的人声泄漏，后处理用", en: "Suppress vocal bleed in separated stems", ja: "分離後のボーカルリーク抑制" },
    filename: "mel_band_roformer_bleed_suppressor_v1.ckpt", fileSize: 1_500_000_000,
    stems: ["Clean", "Bleed"],
    downloadUrl: `${ASEP_GH}/mel_band_roformer_bleed_suppressor_v1.ckpt`,
    configUrl: `${ASEP_GH}/config_mel_band_roformer_bleed_suppressor_v1.yaml`,
  },

  // ════════════════════════════════════════
  //  DENOISE — 去噪
  // ════════════════════════════════════════
  {
    id: "mel_denoise", category: "denoise", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 去噪 27.99", en: "MelBand Denoise 27.99", ja: "MelBand ノイズ除去 27.99" },
    description: { zh: "去噪模型 (aufr33)，SDR 27.99", en: "Denoise by aufr33, SDR 27.99", ja: "ノイズ除去 (aufr33)、SDR 27.99" },
    filename: "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", fileSize: 1_500_000_000,
    stems: ["Clean", "Noise"], sdrScore: 27.99,
    // The jarredou HF account (original host) is gone — 401 on every URL. ZFTurbo's release is
    // the authoritative surviving origin (docs/pretrained_models.md pairs these exact files).
    downloadUrl: `${GH_ZFT}/v.1.0.7/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`,
    configUrl: `${GH_ZFT}/v.1.0.7/model_mel_band_roformer_denoise.yaml`,
  },
  {
    id: "mel_denoise_aggr", category: "denoise", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 去噪 (激进)", en: "MelBand Denoise (Aggressive)", ja: "MelBand ノイズ除去 (アグレッシブ)" },
    description: { zh: "激进去噪 (aufr33)，SDR 27.98", en: "Aggressive denoise by aufr33, SDR 27.98", ja: "アグレッシブノイズ除去 (aufr33)、SDR 27.98" },
    filename: "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt", fileSize: 1_500_000_000,
    stems: ["Clean", "Noise"], sdrScore: 27.98,
    // Same dead jarredou host as mel_denoise; one shared config covers both denoise variants.
    downloadUrl: `${GH_ZFT}/v.1.0.7/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt`,
    configUrl: `${GH_ZFT}/v.1.0.7/model_mel_band_roformer_denoise.yaml`,
  },
  {
    // VR-arch entries carry no configUrl: params come from the converter's embedded registry.
    id: "vr_denoise", category: "denoise", architecture: "uvr_vr", source: "official",
    name: { zh: "UVR DeNoise (VR)", en: "UVR DeNoise (VR)", ja: "UVR DeNoise (VR)" },
    description: { zh: "去噪 (FoxJoy)：注意主输出口是Noise（噪声本体），干净音频在第二个口No Noise", en: "Denoise (FoxJoy): note the FIRST port is Noise (the noise itself) — the clean audio is on the second port, No Noise", ja: "ノイズ除去 (FoxJoy)：第1ポートはNoise（ノイズ本体）、クリーン音声は第2ポートNo Noiseに出力" },
    filename: "UVR-DeNoise.pth", fileSize: 127_139_365,
    // TRUE model output order — port 0 is the NOISE stem, port 1 the clean audio. Do not swap.
    stems: ["Noise", "No Noise"],
    downloadUrl: `${UVR_GH}/UVR-DeNoise.pth`,
  },
  {
    id: "vr_denoise_lite", category: "denoise", architecture: "uvr_vr", source: "official",
    name: { zh: "UVR DeNoise Lite (VR)", en: "UVR DeNoise Lite (VR)", ja: "UVR DeNoise Lite (VR)" },
    description: { zh: "轻量单频带去噪版，速度快但质量略低", en: "Lightweight single-band denoise — fast, slightly lower quality", ja: "軽量シングルバンド版ノイズ除去。高速だが品質はやや低め" },
    filename: "UVR-DeNoise-Lite.pth", fileSize: 17_922_277,
    // Same TRUE order as vr_denoise: port 0 = noise, port 1 = clean.
    stems: ["Noise", "No Noise"],
    downloadUrl: `${UVR_GH}/UVR-DeNoise-Lite.pth`,
  },

  // ════════════════════════════════════════
  //  DEREVERB — 去混响
  // ════════════════════════════════════════
  {
    id: "mel_dereverb", category: "dereverb", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 去混响 19.17", en: "MelBand De-Reverb 19.17", ja: "MelBand リバーブ除去 19.17" },
    description: { zh: "最佳去混响 (anvuew)，SDR 19.17", en: "Best de-reverb by anvuew, SDR 19.17", ja: "最高性能リバーブ除去 (anvuew)、SDR 19.17" },
    filename: "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", fileSize: 1_500_000_000,
    stems: ["Dry", "Reverb"], sdrScore: 19.17,
    downloadUrl: `${HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt`,
    // Stereo config (the repo's mono variant has its own separate ckpt+yaml — do not mix).
    configUrl: `${HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml`,
  },
  {
    id: "mel_dereverb_less", category: "dereverb", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 去混响 (温和)", en: "MelBand De-Reverb (Mild)", ja: "MelBand リバーブ除去 (マイルド)" },
    description: { zh: "温和去混响 (anvuew)，保留更多原始音色", en: "Less aggressive by anvuew, preserves tone", ja: "マイルドリバーブ除去 (anvuew)、原音をより保持" },
    filename: "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt", fileSize: 1_500_000_000,
    stems: ["Dry", "Reverb"], sdrScore: 18.81,
    downloadUrl: `${HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt`,
    // Mid-training checkpoint of the same run as mel_dereverb — same arch, same config.
    configUrl: `${HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml`,
  },
  {
    id: "mel_dereverb_echo", category: "dereverb", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 去回声", en: "MelBand De-Echo", ja: "MelBand エコー除去" },
    description: { zh: "去回声+混响 (Sucial)，SDR 10.02", en: "De-echo + de-reverb by Sucial, SDR 10.02", ja: "エコー+リバーブ除去 (Sucial)、SDR 10.02" },
    filename: "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt", fileSize: 1_500_000_000,
    stems: ["Dry", "Reverb+Echo"], sdrScore: 10.02,
    downloadUrl: `${HF}/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt`,
    // dim-256/depth-8 V1 config; the repo's config_dereverb_echo_mbr_v2.yaml belongs to the
    // v2/big/fused models and would misconvert this ckpt.
    configUrl: `${HF}/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml`,
  },
  {
    id: "bs_dereverb", category: "dereverb", architecture: "bs_roformer", source: "official",
    name: { zh: "BS-Roformer 去混响", en: "BS-Roformer De-Reverb", ja: "BS-Roformer リバーブ除去" },
    description: { zh: "BS-Roformer 架构去混响 (anvuew)", en: "BS-Roformer de-reverb by anvuew", ja: "BS-Roformer リバーブ除去 (anvuew)" },
    filename: "deverb_bs_roformer_8_384dim_10depth.ckpt", fileSize: 1_200_000_000,
    stems: ["Dry", "Reverb"],
    downloadUrl: `${HF}/anvuew/deverb_bs_roformer/resolve/main/archive/deverb_bs_roformer_8_384dim_10depth.ckpt`,
    configUrl: `${HF}/anvuew/deverb_bs_roformer/resolve/main/archive/deverb_bs_roformer_8_384dim_10depth.yaml`,
  },
  {
    id: "mdx23c_dereverb", category: "dereverb", architecture: "mdx23c", source: "community",
    name: { zh: "MDX23C 去混响", en: "MDX23C De-Reverb", ja: "MDX23C リバーブ除去" },
    description: { zh: "MDX23C 架构去混响 (aufr33+jarredou)", en: "MDX23C de-reverb by aufr33+jarredou", ja: "MDX23C リバーブ除去 (aufr33+jarredou)" },
    filename: "MDX23C-De-Reverb-aufr33-jarredou.ckpt", fileSize: 900_000_000,
    stems: ["Dry", "Reverb"],
    downloadUrl: `${UVR}/MDX23C-De-Reverb-aufr33-jarredou.ckpt`,
    configUrl: `${UVR.replace("models/MDX23C", "models/MDX23C")}/config_dereverb_mdx23c.yaml`,
  },
  {
    id: "vr_deecho_normal", category: "dereverb", architecture: "uvr_vr", source: "official",
    name: { zh: "UVR De-Echo Normal (VR)", en: "UVR De-Echo Normal (VR)", ja: "UVR De-Echo Normal (VR)" },
    description: { zh: "去回声（普通强度），常用于人声stem后处理（作者 FoxJoy）", en: "De-echo (normal strength), a common vocal-stem post-processing step (by FoxJoy)", ja: "エコー除去（通常強度）、ボーカルstemの後処理の定番（作者 FoxJoy）" },
    filename: "UVR-De-Echo-Normal.pth", fileSize: 127_139_365,
    stems: ["No Echo", "Echo"],
    downloadUrl: `${UVR_GH}/UVR-De-Echo-Normal.pth`,
  },
  {
    id: "vr_deecho_aggressive", category: "dereverb", architecture: "uvr_vr", source: "official",
    name: { zh: "UVR De-Echo Aggressive (VR)", en: "UVR De-Echo Aggressive (VR)", ja: "UVR De-Echo Aggressive (VR)" },
    description: { zh: "去回声（强力档），残留更少但对原音更激进（作者 FoxJoy）", en: "De-echo (aggressive) — less residual echo, harder on the source (by FoxJoy)", ja: "エコー除去（強力）、残留は少ないが原音への影響も大きめ（作者 FoxJoy）" },
    filename: "UVR-De-Echo-Aggressive.pth", fileSize: 127_139_365,
    stems: ["No Echo", "Echo"],
    downloadUrl: `${UVR_GH}/UVR-De-Echo-Aggressive.pth`,
  },
  {
    id: "vr_deecho_dereverb", category: "dereverb", architecture: "uvr_vr", source: "official",
    name: { zh: "UVR DeEcho-DeReverb (VR)", en: "UVR DeEcho-DeReverb (VR)", ja: "UVR DeEcho-DeReverb (VR)" },
    description: { zh: "同时去回声+去混响，AI翻唱人声清理的社区首选之一（作者 FoxJoy）", en: "Removes echo AND reverb in one pass — a community favorite for AI-cover vocal cleanup (by FoxJoy)", ja: "エコー+リバーブを同時除去。AIカバーのボーカルクリーニングで人気（作者 FoxJoy）" },
    filename: "UVR-DeEcho-DeReverb.pth", fileSize: 223_650_277,
    stems: ["No Reverb", "Reverb"],
    downloadUrl: `${UVR_GH}/UVR-DeEcho-DeReverb.pth`,
  },
  {
    // NOT in UVR's own download list (hence source community); the original author links
    // (jarredou GitHub/HF) are dead — the audio-separator release is the canonical surviving host.
    id: "vr_dereverb_aufr33", category: "dereverb", architecture: "uvr_vr", source: "community",
    name: { zh: "De-Reverb aufr33+jarredou (VR)", en: "De-Reverb aufr33+jarredou (VR)", ja: "De-Reverb aufr33+jarredou (VR)" },
    description: { zh: "全频带mid-side去混响（aufr33+jarredou 合作模型）", en: "Full-band mid-side de-reverb (aufr33+jarredou collaboration)", ja: "フルバンドmid-sideリバーブ除去（aufr33+jarredou 共同モデル）" },
    filename: "UVR-De-Reverb-aufr33-jarredou.pth", fileSize: 58_928_133,
    stems: ["Dry", "Reverb"],
    downloadUrl: `${ASEP_GH}/UVR-De-Reverb-aufr33-jarredou.pth`,
  },

  // ════════════════════════════════════════
  //  KARAOKE — 卡拉OK
  // ════════════════════════════════════════
  {
    id: "mel_karaoke", category: "karaoke", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 卡拉OK", en: "MelBand Karaoke", ja: "MelBand カラオケ" },
    description: { zh: "去和声/背景人声 (aufr33+viperx)，SDR 10.20", en: "Remove backing vocals by aufr33+viperx, SDR 10.20", ja: "バッキングボーカル除去 (aufr33+viperx)、SDR 10.20" },
    filename: "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", fileSize: 1_500_000_000,
    stems: ["Lead Vocal", "Backing"], sdrScore: 10.20,
    // Original jarredou HF repo is now gated (401); this GH mirror ckpt is byte-identical and its
    // paired yaml matches the original config on every model/audio/inference param (cross-checked
    // against a second independent mirror). Mirror instrument STRINGS are Vocals/Instrumental
    // (original said karaoke/other) — same order/content, only naming.
    downloadUrl: `${ASEP_GH}/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`,
    configUrl: `${ASEP_GH}/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml`,
  },
  {
    id: "vr_5hp_karaoke", category: "karaoke", architecture: "uvr_vr", source: "official",
    name: { zh: "5_HP Karaoke (VR)", en: "5_HP Karaoke (VR)", ja: "5_HP Karaoke (VR)" },
    description: { zh: "经典VR卡拉OK模型：Instrumental保留和声/伴唱，Vocals仅主音。常用两步法：先分离人声再用它切主音/和声（作者 Anjok07/aufr33）", en: "Classic VR karaoke model: Instrumental keeps backing vocals/harmonies, Vocals is lead only. Common 2-pass: separate vocals first, then split lead/backing (by Anjok07/aufr33)", ja: "定番VRカラオケモデル：Instrumentalはコーラスを保持、Vocalsはリードのみ。2段階処理でリード/コーラス分離に（作者 Anjok07/aufr33）" },
    filename: "5_HP-Karaoke-UVR.pth", fileSize: 126_782_699,
    stems: ["Instrumental", "Vocals"],
    downloadUrl: `${UVR_GH}/5_HP-Karaoke-UVR.pth`,
  },
  {
    id: "vr_6hp_karaoke", category: "karaoke", architecture: "uvr_vr", source: "official",
    name: { zh: "6_HP Karaoke (VR)", en: "6_HP Karaoke (VR)", ja: "6_HP Karaoke (VR)" },
    description: { zh: "与5_HP训练不同的姊妹模型，社区反馈伴奏略更干净，建议两者对比取优", en: "Sibling of 5_HP with different training; community reports slightly cleaner instrumentals — try both and keep the better result", ja: "5_HPとは学習の異なる姉妹モデル。伴奏がやや綺麗との評、両方試して良い方を採用" },
    filename: "6_HP-Karaoke-UVR.pth", fileSize: 126_782_699,
    stems: ["Instrumental", "Vocals"],
    downloadUrl: `${UVR_GH}/6_HP-Karaoke-UVR.pth`,
  },
  {
    id: "mdx_kara", category: "karaoke", architecture: "mdx_net", source: "official",
    name: { zh: "UVR MDX-NET Karaoke", en: "UVR MDX-NET Karaoke", ja: "UVR MDX-NET Karaoke" },
    description: { zh: "主输出是主音Vocals，伴奏+和声在第二口（与KARA_2主副相反）；原生ONNX免转换", en: "First port is the LEAD vocal; instrumental+backing on the second (opposite of KARA_2). Native ONNX, no conversion needed", ja: "第1ポートがリードVocals、伴奏+コーラスは第2ポート（KARA_2と主従が逆）。ONNXネイティブで変換不要" },
    filename: "UVR_MDXNET_KARA.onnx", fileSize: 29_704_436,
    // TRUE order — opposite of mdx_kara_2: port 0 = lead vocal, port 1 = instrumental+backing.
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${UVR_GH}/UVR_MDXNET_KARA.onnx`,
  },
  {
    id: "mdx_kara_2", category: "karaoke", architecture: "mdx_net", source: "official",
    name: { zh: "UVR MDX-NET Karaoke 2", en: "UVR MDX-NET Karaoke 2", ja: "UVR MDX-NET Karaoke 2" },
    description: { zh: "直接输出卡拉OK轨（伴奏+和声），主音在第二口；社区常用两步法：先跑人声分离，再用它把人声stem切成主音/和声", en: "Outputs the karaoke track (instrumental+backing) first, lead vocal on the second port. Common 2-pass: separate vocals first, then split the vocal stem into lead/backing with this", ja: "カラオケトラック（伴奏+コーラス）を第1ポートに出力、リードは第2ポート。先にボーカル分離→これでリード/コーラス分割の2段階処理が定番" },
    filename: "UVR_MDXNET_KARA_2.onnx", fileSize: 52_786_726,
    stems: ["Instrumental", "Vocals"],
    downloadUrl: `${UVR_GH}/UVR_MDXNET_KARA_2.onnx`,
  },

  // ════════════════════════════════════════
  //  MULTI-STEM — 多轨道
  // ════════════════════════════════════════
  {
    id: "htdemucs_std", category: "multistem", architecture: "htdemucs", source: "official",
    name: { zh: "HTDemucs 标准版", en: "HTDemucs Standard", ja: "HTDemucs 標準版" },
    description: { zh: "4轨: 鼓/贝斯/其他/人声, Meta 官方, ~80MB", en: "4-stem: drums/bass/other/vocals, by Meta, ~80MB", ja: "4ステム: ドラム/ベース/その他/ボーカル、Meta 公式、~80MB" },
    filename: "955717e8-8726e21a.th", fileSize: 84_000_000,
    stems: ["Drums", "Bass", "Other", "Vocals"],
    downloadUrl: `${FB_CDN}/955717e8-8726e21a.th`,
    configUrl: `${UVR_GH}/htdemucs.yaml`,
  },
  {
    id: "htdemucs_6s", category: "multistem", architecture: "htdemucs", source: "official",
    name: { zh: "HTDemucs 6轨", en: "HTDemucs 6-Stem", ja: "HTDemucs 6ステム" },
    description: { zh: "6轨: 鼓/贝斯/吉他/钢琴/其他/人声, ~52MB", en: "6-stem: drums/bass/guitar/piano/other/vocals, ~52MB", ja: "6ステム: ドラム/ベース/ギター/ピアノ/その他/ボーカル、~52MB" },
    // TRUE weight output order (ckpt kwargs) — the model card lists guitar/piano mid-list, but
    // labeling ports from that order put VOCALS on the Piano port. Port labels now come from the
    // installed json anyway; this is the pre-install display fallback and must still be correct.
    filename: "5c90dfd2-34c22ccb.th", fileSize: 55_000_000,
    stems: ["Drums", "Bass", "Other", "Vocals", "Guitar", "Piano"],
    downloadUrl: `${FB_CDN}/5c90dfd2-34c22ccb.th`,
    configUrl: `${UVR_GH}/htdemucs_6s.yaml`,
  },
  {
    id: "hdemucs_mmi", category: "multistem", architecture: "htdemucs", source: "official",
    name: { zh: "HDemucs MMI", en: "HDemucs MMI", ja: "HDemucs MMI" },
    description: { zh: "混合密度互信息，4轨, ~160MB", en: "Mixed Mutual Information, 4-stem, ~160MB", ja: "混合相互情報量、4ステム、~160MB" },
    filename: "75fc33f5-1941ce65.th", fileSize: 167_000_000,
    stems: ["Drums", "Bass", "Other", "Vocals"],
    downloadUrl: `${FB_CDN}/75fc33f5-1941ce65.th`,
    configUrl: `${UVR_GH}/hdemucs_mmi.yaml`,
  },
  {
    id: "mdx23c_drumsep", category: "multistem", architecture: "mdx23c", source: "community",
    name: { zh: "MDX23C 鼓组分离", en: "MDX23C Drum Separator", ja: "MDX23C ドラム分離" },
    description: { zh: "6种鼓组分离 (aufr33+jarredou)", en: "6 drum types by aufr33+jarredou", ja: "6種ドラム分離 (aufr33+jarredou)" },
    filename: "MDX23C-DrumSep-aufr33-jarredou.ckpt", fileSize: 900_000_000,
    // TRUE training order per its own configUrl yaml: [kick, snare, toms, hh, ride, crash] —
    // the old hand-written list swapped toms/hh and invented "Cymbals"/"Other".
    stems: ["Kick", "Snare", "Toms", "Hi-hat", "Ride", "Crash"],
    downloadUrl: `${UVR}/MDX23C-DrumSep-aufr33-jarredou.ckpt`,
    configUrl: `${UVR}/config_drumsep_mdx23c.yaml`,
  },

  // ════════════════════════════════════════
  //  SPECIAL — 特殊
  // ════════════════════════════════════════
  {
    id: "mel_crowd", category: "special", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 人群噪声", en: "MelBand Crowd Noise", ja: "MelBand クラウドノイズ" },
    description: { zh: "去除人群/现场录音背景声，SDR 8.71", en: "Remove crowd/live background, SDR 8.71", ja: "クラウドノイズ除去、SDR 8.71" },
    filename: "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt", fileSize: 1_500_000_000,
    // target_instrument = crowd → port 0 is the CROWD noise, port 1 the clean residual.
    stems: ["Crowd", "Clean"], sdrScore: 8.71,
    downloadUrl: `${MSST}/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt`,
    // Byte-identical to the authoritative ZFTurbo v.1.0.4 release copy; confirms [crowd, other].
    configUrl: `${MSST}/model_mel_band_roformer_crowd.yaml`,
  },
  {
    id: "bs_chorus", category: "special", architecture: "bs_roformer", source: "official",
    name: { zh: "BS-Roformer 合唱分离", en: "BS-Roformer Chorus Split", ja: "BS-Roformer コーラス分離" },
    description: { zh: "男女声合唱分离 (Sucial)，SDR 24.13", en: "Male/female chorus separation by Sucial, SDR 24.13", ja: "男女コーラス分離 (Sucial)、SDR 24.13" },
    filename: "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt", fileSize: 1_500_000_000,
    stems: ["Male", "Female"], sdrScore: 24.13,
    downloadUrl: `${HF}/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt`,
    configUrl: `${HF}/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml`,
  },
  {
    id: "mel_aspiration", category: "special", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 气息检测", en: "MelBand Aspiration", ja: "MelBand 気息検出" },
    description: { zh: "检测分离呼吸声 (Sucial)，SDR 18.98", en: "Detect breath sounds by Sucial, SDR 18.98", ja: "呼吸音検出 (Sucial)、SDR 18.98" },
    filename: "aspiration_mel_band_roformer_sdr_18.9845.ckpt", fileSize: 1_500_000_000,
    // direct 2-stem [aspiration, other] → port 0 is the BREATH stem, port 1 the clean track.
    stems: ["Breath", "Clean"], sdrScore: 18.98,
    downloadUrl: `${HF}/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_sdr_18.9845.ckpt`,
    configUrl: `${HF}/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/config_aspiration_mel_band_roformer.yaml`,
  },
  {
    id: "bs_drum_bass", category: "special", architecture: "bs_roformer", source: "official",
    name: { zh: "BS-Roformer 鼓+贝斯", en: "BS-Roformer Drum+Bass", ja: "BS-Roformer ドラム+ベース" },
    description: { zh: "分离鼓和贝斯，SDR 10.53", en: "Separate drums+bass, SDR 10.53", ja: "ドラム+ベース分離、SDR 10.53" },
    filename: "model_bs_roformer_ep_937_sdr_10.5309.ckpt", fileSize: 1_500_000_000,
    // Its configUrl yaml (authoritative — the ARCHIVE copy for this ckpt is itself wrong):
    // target_instrument = "No Drum-Bass" → port 0 is everything EXCEPT drums+bass (vocals
    // included), port 1 is the drums+bass stem.
    stems: ["Other", "Drums+Bass"], sdrScore: 10.53,
    downloadUrl: `${MSST}/model_bs_roformer_ep_937_sdr_10.5309.ckpt`,
    configUrl: `${GH_CFG}/model_bs_roformer_ep_937_sdr_10.5309.yaml`,
  },
  {
    id: "vr_17hp_wind", category: "special", architecture: "uvr_vr", source: "official",
    name: { zh: "17_HP 管乐提取 (VR)", en: "17_HP Wind Inst (VR)", ja: "17_HP 管楽器抽出 (VR)" },
    description: { zh: "提取长笛/萨克斯等管乐：Woodwinds口是管乐本体，No Woodwinds是其余伴奏", en: "Extracts flutes/sax and other wind instruments: Woodwinds port is the winds, No Woodwinds the rest", ja: "フルート/サックス等の管楽器を抽出：Woodwindsが管楽器本体、No Woodwindsが残り" },
    filename: "17_HP-Wind_Inst-UVR.pth", fileSize: 223_661_285,
    stems: ["No Woodwinds", "Woodwinds"],
    downloadUrl: `${UVR_GH}/17_HP-Wind_Inst-UVR.pth`,
  },
];

export function applyMirror(url: string, mirror: MirrorSource): string {
  if (mirror.type === "huggingface" || !url.includes("huggingface.co")) return url;
  if (mirror.type === "hf-mirror") return url.replace("https://huggingface.co", "https://hf-mirror.com");
  if (mirror.type === "custom" && mirror.customUrl) return url.replace("https://huggingface.co", mirror.customUrl);
  return url;
}

export interface MirrorSource {
  type: "huggingface" | "hf-mirror" | "custom";
  customUrl: string;
}

export const DEFAULT_MIRROR: MirrorSource = { type: "huggingface", customUrl: "" };

/** The HF host-replacement BASE for Rust commands that take an `hf_base` (asset packs):
 *  the chosen mirror host, or null for the official default. SINGLE SOURCE — Settings'
 *  download button and the S66 one-click dialogs must derive it identically. */
export function hfBaseForMirror(mirror: MirrorSource): string | null {
  if (mirror.type === "hf-mirror") return "https://hf-mirror.com";
  if (mirror.type === "custom" && mirror.customUrl.trim()) return mirror.customUrl.trim();
  return null;
}

/** GitHub direct-link mirror (mainland-China acceleration) — the GH counterpart of
 *  `MirrorSource` above. Orthogonal axes: `applyMirror` only rewrites huggingface.co
 *  hosts, `applyGhMirror` only prefixes github.com-family hosts, so chaining them is
 *  always safe. S66: presets are DATA (a list), not enum members — public prefixes rot
 *  in 6-18 months, so the live list is refreshed remotely (mirrors.json on the
 *  utai-runtimes HF dataset → store.refreshGhPresets) and these are the offline fallback. */
export interface GhMirror {
  type: "direct" | "preset" | "custom";
  /** Chosen entry in the preset list (type "preset"); unknown ids fall back to the first preset. */
  presetId?: string;
  customUrl: string;
}

export interface GhPreset {
  id: string;
  prefix: string;
}

/** 2026-07 live-verified (raw + release-asset 302 follow + Range 206): gh-proxy.com,
 *  ghproxy.net, gh.llkk.cc all functional; ghfast.top kept last (blocked reports from
 *  mainland testers). */
export const BUILTIN_GH_PRESETS: GhPreset[] = [
  { id: "gh-proxy.com", prefix: "https://gh-proxy.com" },
  { id: "ghproxy.net", prefix: "https://ghproxy.net" },
  { id: "gh.llkk.cc", prefix: "https://gh.llkk.cc" },
  { id: "ghfast.top", prefix: "https://ghfast.top" },
];

export const DEFAULT_GH_MIRROR: GhMirror = { type: "direct", customUrl: "" };

/** Migrate a persisted pre-S66 value ({type:"ghfast"|"ghproxy"}) onto the preset shape —
 *  localStorage survives updates, so the old enum members must keep meaning forever.
 *  Shape-tolerant to CORRUPT persisted values incl. null/non-object (review S66: this runs at
 *  store-module init — a throw here white-screens the whole app). */
export function migrateGhMirror(gh: unknown): GhMirror {
  if (!gh || typeof gh !== "object") return { ...DEFAULT_GH_MIRROR };
  const g = gh as { type?: unknown; presetId?: unknown; customUrl?: unknown };
  const customUrl = typeof g.customUrl === "string" ? g.customUrl : "";
  const presetId = typeof g.presetId === "string" ? g.presetId : undefined;
  if (g.type === "ghfast") return { type: "preset", presetId: "ghfast.top", customUrl };
  if (g.type === "ghproxy") return { type: "preset", presetId: "gh-proxy.com", customUrl };
  if (g.type === "direct" || g.type === "preset" || g.type === "custom") {
    return { type: g.type, presetId, customUrl };
  }
  return { ...DEFAULT_GH_MIRROR };
}

/** Proxy prefix for the selected GH mirror, or null for direct / blank custom.
 *  The presets are community-run public services — they come and go without notice;
 *  the custom option is the user's fallback when a preset dies. */
export function ghProxyPrefix(gh: GhMirror, presets: GhPreset[] = BUILTIN_GH_PRESETS): string | null {
  if (gh.type === "preset") {
    return presets.find((p) => p.id === gh.presetId)?.prefix ?? presets[0]?.prefix ?? null;
  }
  if (gh.type === "custom") {
    // localStorage shape-tolerance (audit): a corrupt persisted value without customUrl must not
    // throw here (it would wedge the Settings test button) — same posture as applyMirror's truthiness guard.
    let u = (gh.customUrl ?? "").trim().replace(/\/+$/, "");
    // Scheme normalization (S64 audit): downstream consumers require a well-formed https prefix —
    // the updater's endpoint validation hard-rejects non-https in RELEASE builds, and Rust-side
    // sanitize_gh_prefix degrades schemeless/http prefixes to "no proxy". Prepending https here
    // keeps a bare "ghfast.top"-style entry working everywhere instead of silently doing nothing.
    if (u && !/^https?:\/\//i.test(u)) u = "https://" + u;
    return u || null;
  }
  return null; // direct
}

/** Prefix-style proxying — the standard usage of public GH accelerators is
 *  `<prefix>/<full original URL>`. GH release assets 302 to
 *  objects.githubusercontent.com; the proxy follows redirects itself, so only the
 *  INITIAL URL needs rewriting. Exact-hostname match (plus subdomain fallback) so
 *  lookalikes such as "notgithub.com" are never touched. */
const GH_HOSTS = new Set([
  "github.com",
  "raw.githubusercontent.com",
  "objects.githubusercontent.com",
  "codeload.github.com",
]);

function isGhUrl(url: string): boolean {
  let host: string;
  try {
    host = new URL(url).hostname;
  } catch {
    return false;
  }
  return GH_HOSTS.has(host) || host.endsWith(".github.com") || host.endsWith(".githubusercontent.com");
}

export function applyGhMirror(url: string, gh: GhMirror, presets: GhPreset[] = BUILTIN_GH_PRESETS): string {
  const prefix = ghProxyPrefix(gh, presets);
  if (!prefix || !isGhUrl(url)) return url;
  return `${prefix}/${url}`;
}

/** Ordered GH ROUTE list for Rust-side consumers that build their own URLs (updater /
 *  GAME): proxy prefixes with an "" marker at the DIRECT position — the chosen proxy
 *  (if any) rides before direct, the remaining presets after. Rust maps "" to the
 *  unprefixed URL and prefixes the rest. Full-chain fallback is safe HERE because both
 *  consumers verify content (GAME sha256 / updater minisign). */
export function ghRouteOrder(gh: GhMirror, presets: GhPreset[] = BUILTIN_GH_PRESETS): string[] {
  const out: string[] = [];
  const chosen = ghProxyPrefix(gh, presets);
  if (chosen) out.push(chosen);
  out.push("");
  for (const p of presets) {
    if (!out.includes(p.prefix)) out.push(p.prefix);
  }
  return out;
}

/** S66 failover candidates for a GH-hosted url: the chosen prefix first, then the direct
 *  url — and, ONLY when `presetFallbacks` is true, every other preset as a tail.
 *
 *  presetFallbacks MUST stay false for UN-HASHED downloads (MSST models — review S66):
 *  without a sha256 commit gate, silently routing a direct-mode user's bytes through
 *  community proxies would let a rogue proxy's wrong bytes commit as the model file.
 *  Hashed / signature-verified consumers take the full chain via ghRouteOrder instead.
 *  Non-GH urls pass through as a single candidate. */
export function ghMirrorCandidates(
  url: string,
  gh: GhMirror,
  presets: GhPreset[] = BUILTIN_GH_PRESETS,
  presetFallbacks = false,
): string[] {
  if (!isGhUrl(url)) return [url];
  const out: string[] = [];
  const push = (u: string) => {
    if (!out.includes(u)) out.push(u);
  };
  const chosen = ghProxyPrefix(gh, presets);
  if (chosen) push(`${chosen}/${url}`);
  push(url);
  if (presetFallbacks) {
    for (const p of presets) push(`${p.prefix}/${url}`);
  }
  return out;
}
