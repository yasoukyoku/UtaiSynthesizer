export type MsstArchitecture = "bs_roformer" | "mel_band_roformer" | "mdx23c" | "htdemucs";
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
 *  (previously copy-pasted as t18/l in SeparationNode, NodePalette, MsstModelManager, EffectsNode). */
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
};

/** Per-architecture default `num_overlap` — MUST mirror what converter/convert.py writes into each
 *  model's JSON. Used ONLY to DISPLAY the model's true default in the node when the user hasn't
 *  overridden it; the actual value still comes from the model JSON on the Rust side. Keep in sync
 *  with the converter (all 4 use the original MSST inference recipe: num_overlap 4). */
export const MSST_DEFAULT_NUM_OVERLAP: Record<MsstArchitecture, number> = {
  bs_roformer: 4,
  mel_band_roformer: 4,
  mdx23c: 4,
  htdemucs: 4,
};

export const ALL_CATEGORIES: MsstCategory[] = [
  "vocals", "instrumental", "denoise", "dereverb", "karaoke", "multistem", "special",
];

const HF = "https://huggingface.co";
const MSST = `${HF}/Eddycrack864/Music-Source-Separation-Training/resolve/main`;
const UVR = `${HF}/Politrees/UVR_resources/resolve/main/models/MDX23C`;
const GH_CFG = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs";
const GH_SEP = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs";
const FB_CDN = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer";
const GH_DEMUCS = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models";

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
  },
  {
    id: "mel_vocals_kim", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 人声 Kim", en: "MelBand Vocals Kim", ja: "MelBand ボーカル Kim" },
    description: { zh: "KimberleyJSN 原版人声模型", en: "KimberleyJSN original vocal model", ja: "KimberleyJSN オリジナルボーカルモデル" },
    filename: "MelBandRoformer.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt`,
  },
  {
    id: "mel_vocals_kim_ft", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 人声 Kim FT", en: "MelBand Vocals Kim FT", ja: "MelBand ボーカル Kim FT" },
    description: { zh: "Kim 微调版 (unwa)，适合流行音乐", en: "Kim fine-tuned by unwa, good for pop", ja: "Kim ファインチューン版 (unwa)、ポップス向け" },
    filename: "kimmel_unwa_ft.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt`,
  },
  {
    id: "mel_vocals_big_syhft_v1", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Big SYHFT V1", en: "MelBand Big SYHFT V1", ja: "MelBand Big SYHFT V1" },
    description: { zh: "大型模型，更高精度人声分离", en: "Large model, higher precision vocals", ja: "大型モデル、高精度ボーカル分離" },
    filename: "MelBandRoformerBigSYHFTV1.ckpt", fileSize: 3_200_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/MelBandRoformerBigSYHFTV1.ckpt`,
  },
  {
    id: "mel_vocals_big_beta4", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Big Beta4", en: "MelBand Big Beta4", ja: "MelBand Big Beta4" },
    description: { zh: "大型模型 Beta4 版 (unwa)", en: "Large model Beta4 by unwa", ja: "大型モデル Beta4 版 (unwa)" },
    filename: "melband_roformer_big_beta4.ckpt", fileSize: 3_200_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt`,
  },
  {
    id: "mel_vocals_big_beta5e", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Big Beta5e", en: "MelBand Big Beta5e", ja: "MelBand Big Beta5e" },
    description: { zh: "大型模型最新版 (unwa)，推荐", en: "Latest large model by unwa, recommended", ja: "大型モデル最新版 (unwa)、推奨" },
    filename: "big_beta5e.ckpt", fileSize: 3_200_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt`,
  },
  {
    id: "mel_vocals_syhft_v2", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "SYHFT V2", en: "SYHFT V2", ja: "SYHFT V2" },
    description: { zh: "SYH99999 人声模型 V2", en: "SYH99999 vocal model V2", ja: "SYH99999 ボーカルモデル V2" },
    filename: "MelBandRoformerSYHFTV2.ckpt", fileSize: 1_600_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/SYH99999/MelBandRoformerSYHFTV2/resolve/main/MelBandRoformerSYHFTV2.ckpt`,
  },
  {
    id: "mel_vocals_syhft_v25", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "SYHFT V2.5", en: "SYHFT V2.5", ja: "SYHFT V2.5" },
    description: { zh: "SYH99999 人声模型 V2.5，最新", en: "SYH99999 vocal model V2.5, latest", ja: "SYH99999 ボーカルモデル V2.5、最新" },
    filename: "MelBandRoformerSYHFTV2.5.ckpt", fileSize: 1_600_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/SYH99999/MelBandRoformerSYHFTV2.5/resolve/main/MelBandRoformerSYHFTV2.5.ckpt`,
  },
  {
    id: "mel_vocals_duality_v1", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Duality V1", en: "MelBand Duality V1", ja: "MelBand Duality V1" },
    description: { zh: "Duality 人声模型 V1 (unwa)", en: "Duality vocal model V1 by unwa", ja: "Duality ボーカルモデル V1 (unwa)" },
    filename: "melband_roformer_instvoc_duality_v1.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt`,
  },
  {
    id: "mel_vocals_duality_v2", category: "vocals", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand Duality V2", en: "MelBand Duality V2", ja: "MelBand Duality V2" },
    description: { zh: "Duality 人声模型 V2 (unwa)", en: "Duality vocal model V2 by unwa", ja: "Duality ボーカルモデル V2 (unwa)" },
    filename: "melband_roformer_instvox_duality_v2.ckpt", fileSize: 1_500_000_000,
    stems: ["Vocals", "Instrumental"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt`,
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
    configUrl: `${GH_CFG}/model_2_stem_full_band_8k.yaml`,
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
  },
  {
    id: "mel_inst_v1", category: "instrumental", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 伴奏 V1", en: "MelBand Inst V1", ja: "MelBand 伴奏 V1" },
    description: { zh: "伴奏提取 V1 (unwa)", en: "Instrumental V1 by unwa", ja: "伴奏抽出 V1 (unwa)" },
    filename: "melband_roformer_inst_v1.ckpt", fileSize: 1_500_000_000,
    stems: ["Instrumental", "Vocals"],
    downloadUrl: `${HF}/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt`,
  },
  {
    id: "mel_bleed_suppress", category: "instrumental", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 泄漏抑制", en: "MelBand Bleed Suppressor", ja: "MelBand リーク抑制" },
    description: { zh: "抑制分离后的人声泄漏，后处理用", en: "Suppress vocal bleed in separated stems", ja: "分離後のボーカルリーク抑制" },
    filename: "mel_band_roformer_bleed_suppressor_v1.ckpt", fileSize: 1_500_000_000,
    stems: ["Clean", "Bleed"],
    downloadUrl: `${GH_SEP}/mel_band_roformer_bleed_suppressor_v1.ckpt`,
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
    downloadUrl: `${HF}/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`,
  },
  {
    id: "mel_denoise_aggr", category: "denoise", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 去噪 (激进)", en: "MelBand Denoise (Aggressive)", ja: "MelBand ノイズ除去 (アグレッシブ)" },
    description: { zh: "激进去噪 (aufr33)，SDR 27.98", en: "Aggressive denoise by aufr33, SDR 27.98", ja: "アグレッシブノイズ除去 (aufr33)、SDR 27.98" },
    filename: "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt", fileSize: 1_500_000_000,
    stems: ["Clean", "Noise"], sdrScore: 27.98,
    downloadUrl: `${HF}/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt`,
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
  },
  {
    id: "mel_dereverb_less", category: "dereverb", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 去混响 (温和)", en: "MelBand De-Reverb (Mild)", ja: "MelBand リバーブ除去 (マイルド)" },
    description: { zh: "温和去混响 (anvuew)，保留更多原始音色", en: "Less aggressive by anvuew, preserves tone", ja: "マイルドリバーブ除去 (anvuew)、原音をより保持" },
    filename: "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt", fileSize: 1_500_000_000,
    stems: ["Dry", "Reverb"], sdrScore: 18.81,
    downloadUrl: `${HF}/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt`,
  },
  {
    id: "mel_dereverb_echo", category: "dereverb", architecture: "mel_band_roformer", source: "official",
    name: { zh: "MelBand 去回声", en: "MelBand De-Echo", ja: "MelBand エコー除去" },
    description: { zh: "去回声+混响 (Sucial)，SDR 10.02", en: "De-echo + de-reverb by Sucial, SDR 10.02", ja: "エコー+リバーブ除去 (Sucial)、SDR 10.02" },
    filename: "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt", fileSize: 1_500_000_000,
    stems: ["Dry", "Reverb+Echo"], sdrScore: 10.02,
    downloadUrl: `${HF}/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt`,
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

  // ════════════════════════════════════════
  //  KARAOKE — 卡拉OK
  // ════════════════════════════════════════
  {
    id: "mel_karaoke", category: "karaoke", architecture: "mel_band_roformer", source: "community",
    name: { zh: "MelBand 卡拉OK", en: "MelBand Karaoke", ja: "MelBand カラオケ" },
    description: { zh: "去和声/背景人声 (aufr33+viperx)，SDR 10.20", en: "Remove backing vocals by aufr33+viperx, SDR 10.20", ja: "バッキングボーカル除去 (aufr33+viperx)、SDR 10.20" },
    filename: "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", fileSize: 1_500_000_000,
    stems: ["Lead Vocal", "Backing"], sdrScore: 10.20,
    downloadUrl: `${HF}/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`,
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
    configUrl: `${GH_DEMUCS}/htdemucs.yaml`,
  },
  {
    id: "htdemucs_6s", category: "multistem", architecture: "htdemucs", source: "official",
    name: { zh: "HTDemucs 6轨", en: "HTDemucs 6-Stem", ja: "HTDemucs 6ステム" },
    description: { zh: "6轨: 鼓/贝斯/吉他/钢琴/其他/人声, ~52MB", en: "6-stem: drums/bass/guitar/piano/other/vocals, ~52MB", ja: "6ステム: ドラム/ベース/ギター/ピアノ/その他/ボーカル、~52MB" },
    filename: "5c90dfd2-34c22ccb.th", fileSize: 55_000_000,
    stems: ["Drums", "Bass", "Guitar", "Piano", "Other", "Vocals"],
    downloadUrl: `${FB_CDN}/5c90dfd2-34c22ccb.th`,
    configUrl: `${GH_DEMUCS}/htdemucs_6s.yaml`,
  },
  {
    id: "hdemucs_mmi", category: "multistem", architecture: "htdemucs", source: "official",
    name: { zh: "HDemucs MMI", en: "HDemucs MMI", ja: "HDemucs MMI" },
    description: { zh: "混合密度互信息，4轨, ~160MB", en: "Mixed Mutual Information, 4-stem, ~160MB", ja: "混合相互情報量、4ステム、~160MB" },
    filename: "75fc33f5-1941ce65.th", fileSize: 167_000_000,
    stems: ["Drums", "Bass", "Other", "Vocals"],
    downloadUrl: `${FB_CDN}/75fc33f5-1941ce65.th`,
    configUrl: `${GH_DEMUCS}/hdemucs_mmi.yaml`,
  },
  {
    id: "mdx23c_drumsep", category: "multistem", architecture: "mdx23c", source: "community",
    name: { zh: "MDX23C 鼓组分离", en: "MDX23C Drum Separator", ja: "MDX23C ドラム分離" },
    description: { zh: "6种鼓组分离 (aufr33+jarredou)", en: "6 drum types by aufr33+jarredou", ja: "6種ドラム分離 (aufr33+jarredou)" },
    filename: "MDX23C-DrumSep-aufr33-jarredou.ckpt", fileSize: 900_000_000,
    stems: ["Kick", "Snare", "Hi-hat", "Toms", "Cymbals", "Other"],
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
    stems: ["Clean", "Crowd"], sdrScore: 8.71,
    downloadUrl: `${MSST}/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt`,
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
    stems: ["Clean", "Breath"], sdrScore: 18.98,
    downloadUrl: `${HF}/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_sdr_18.9845.ckpt`,
  },
  {
    id: "bs_drum_bass", category: "special", architecture: "bs_roformer", source: "official",
    name: { zh: "BS-Roformer 鼓+贝斯", en: "BS-Roformer Drum+Bass", ja: "BS-Roformer ドラム+ベース" },
    description: { zh: "分离鼓和贝斯，SDR 10.53", en: "Separate drums+bass, SDR 10.53", ja: "ドラム+ベース分離、SDR 10.53" },
    filename: "model_bs_roformer_ep_937_sdr_10.5309.ckpt", fileSize: 1_500_000_000,
    stems: ["Drums+Bass", "Other"], sdrScore: 10.53,
    downloadUrl: `${MSST}/model_bs_roformer_ep_937_sdr_10.5309.ckpt`,
    configUrl: `${GH_CFG}/model_bs_roformer_ep_937_sdr_10.5309.yaml`,
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
