import { I18nText, applyMirror, MirrorSource } from "./msst-catalog";

export type AmtArchitecture = "yourmt3_plus" | "transkun" | "aria_amt" | "bytedance_piano" | "beat_this" | "fluidsynth" | "soundfont" | "bs_roformer" | "polarformer" | "musescore" | "miros" | "whisper" | "ffmpeg";

/** A small companion file a multi-file model needs besides its main artifact. */
export interface AmtExtraFile {
  filename: string;
  downloadUrl: string;
  fileSize: number;
}

export interface AmtCatalogEntry {
  id: string;
  name: I18nText;
  description: I18nText;
  architecture: AmtArchitecture;
  filename: string;
  fileSize: number;
  downloadUrl: string;
  repoId: string;
  revision?: string;
  repoType?: "space" | "model" | "dataset";
  sha256: string;
  ui_features?: I18nText[];
  rating?: number;
  newbie_tip?: I18nText;
  isEssential?: boolean;
  /** Companion files downloaded together with the main artifact (e.g. Whisper's
   * config.json / tokenizer.json). Downloaded sequentially, same progress card. */
  extraFiles?: AmtExtraFile[];
}

const HF = "https://huggingface.co";

export const AMT_CATALOG: AmtCatalogEntry[] = [
  {
    id: "yptf_moe_multi_nops",
    architecture: "yourmt3_plus",
    name: { zh: "YourMT3+ MoE (默认)", en: "YourMT3+ MoE (Default)", ja: "YourMT3+ MoE (デフォルト)" },
    description: { 
      zh: "默认高精度模型，采用 MoE 多通道解码，支持多种乐器。", 
      en: "Default high-precision model using MoE multi-channel decoding, supports multiple instruments.",
      ja: "デフォルトの高精度モデル、MoEマルチチャネルデコードを採用、多楽器に対応。"
    },
    filename: "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt",
    fileSize: 561_544_628,
    repoId: "mimbres/YourMT3",
    repoType: "space",
    revision: "5e66c1ea173a8186e0d20432b841d3180cc015b5",
    downloadUrl: `${HF}/spaces/mimbres/YourMT3/resolve/5e66c1ea173a8186e0d20432b841d3180cc015b5/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt`,
    sha256: "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25",
    rating: 5,
    newbie_tip: { zh: "最强全能模型，支持多种乐器，识别最精准，但速度较慢。", en: "Best all-around model, supports multiple instruments, most accurate, but slower.", ja: "最強の万能モデル、多楽器対応、最も高精度ですが速度は遅めです。" },
    ui_features: [
      { zh: "MoE 架构", en: "MoE Architecture", ja: "MoE アーキテクチャ" },
      { zh: "多通道解码", en: "Multi-channel decoding", ja: "マルチチャネルデコード" }
    ],
    isEssential: true
  },
  {
    id: "transkun_v2_aug",
    architecture: "transkun",
    name: { zh: "TransKun 2.0 (钢琴大师)", en: "TransKun 2.0 (Piano Master)", ja: "TransKun 2.0 (ピアノマスター)" },
    description: { 
      zh: "目前最强的钢琴/踏板识别模型，支持高精度力度还原。", 
      en: "The strongest piano/pedal recognition model, supporting high-precision velocity restoration.",
      ja: "現在最強のピアノ/ペダル認識モデル、高精度のベロシティ復元に対応。"
    },
    filename: "transkun_v2_aug/checkpointMSimplerAug/checkpoint.pt",
    fileSize: 56_423_254,
    repoId: "py-transkun/transkun-v2-aug",
    downloadUrl: `${HF}/py-transkun/transkun-v2-aug/resolve/main/checkpointMSimplerAug/checkpoint.pt`,
    sha256: "8bd6b4b5ddf9ce8c5f296a57859eec9f166cd337c35245ec2a2576d90be68c4c",
    rating: 5,
    newbie_tip: { zh: "钢琴转谱天花板，极其精准，支持踏板识别。", en: "Piano transcription ceiling, extremely precise, supports pedal detection.", ja: "ピアノ採譜の最高峰、極めて高精度、ペダル検出対応。" },
  },
  {
    id: "aria_amt",
    architecture: "aria_amt",
    name: { zh: "Aria-AMT (轻量钢琴)", en: "Aria-AMT (Lightweight Piano)", ja: "Aria-AMT (軽量ピアノ)" },
    description: { 
      zh: "响应极快，适合钢琴快速草稿。", 
      en: "Extremely fast response, suitable for quick piano drafts.",
      ja: "非常に高速なレスポンス、ピアノの素早いドラフトに適しています。"
    },
    filename: "aria_amt/piano-medium-double-1.0.safetensors",
    fileSize: 446_577_344,
    repoId: "loubb/aria-midi",
    repoType: "dataset",
    revision: "8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b",
    downloadUrl: `${HF}/datasets/loubb/aria-midi/resolve/8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b/piano-medium-double-1.0.safetensors`,
    sha256: "089d3129dbe93246aeda55efe668c8a48af08afaf9dd15c64cef0a07c0fb30a4",
    rating: 4,
    newbie_tip: { zh: "响应极快，适合钢琴快速草稿。", en: "Extremely fast response, suitable for quick piano drafts.", ja: "極めて高速な応答、ピアノのクイックドラフトに最適。" },
  },
  {
    id: "bytedance_piano",
    architecture: "bytedance_piano",
    name: { zh: "ByteDance Piano (工业级)", en: "ByteDance Piano (Industrial)", ja: "ByteDance Piano (工業級)" },
    description: { 
      zh: "字节跳动出品，力度还原度极高。", 
      en: "Produced by ByteDance, with extremely high velocity restoration.",
      ja: "ByteDance製、ベロシティの再現性が非常に高い。"
    },
    filename: "bytedance_piano/note_F1=0.9677_pedal_F1=0.9186.pth",
    fileSize: 171_966_578,
    repoId: "bytedance/piano-transcription",
    downloadUrl: `https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1`,
    sha256: "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141",
    rating: 4,
    newbie_tip: { zh: "老牌工业级模型，稳健且力度还原好。", en: "Classic industrial model, robust with good velocity restoration.", ja: "老舗の産業用モデル、堅牢でベロシティの復元が良い。" },
  },
  {
    id: "beat_this",
    architecture: "beat_this",
    name: { zh: "Beat-This (节拍追踪)", en: "Beat-This (Beat Tracking)", ja: "Beat-This (ビートトラッキング)" },
    description: { 
      zh: "决定 MIDI 节拍网格准不准的关键模型。", 
      en: "Key model determining the accuracy of the MIDI beat grid.",
      ja: "MIDIビートグリッドの正確さを决定する重要なモデル。"
    },
    filename: "beat_this/final0.ckpt",
    fileSize: 80_000_000,
    repoId: "mason369/music-to-midi-assets",
    downloadUrl: `${HF}/mason369/music-to-midi-assets/resolve/main/beat_this/final0.ckpt`,
    sha256: "verified_by_runtime",
    isEssential: true
  },
  {
    id: "fluidsynth_runtime",
    architecture: "fluidsynth",
    name: { zh: "FluidSynth 运行时 (2.5.6)", en: "FluidSynth Runtime (2.5.6)", ja: "FluidSynth 実行環境 (2.5.6)" },
    description: { 
      zh: "必需组件。用于 MIDI 音频合成、导出和实时播放预览。", 
      en: "Essential component. Used for MIDI audio synthesis, export, and real-time playback preview.",
      ja: "必須コンポーネント。MIDIオーディオ合成、エクスポート、およびリアルタイム再生プレビューに使用されます。"
    },
    filename: "fluidsynth/2.5.6/bin/fluidsynth.exe",
    fileSize: 26_000_000,
    repoId: "FluidSynth/fluidsynth",
    downloadUrl: `https://github.com/FluidSynth/fluidsynth/releases/download/v2.5.6/fluidsynth-v2.5.6-win10-x64-cpp11.zip`,
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "音频合成", en: "Audio Synthesis", ja: "オーディオ合成" },
      { zh: "WAV 导出", en: "WAV Export", ja: "WAV エクスポート" }
    ],
    isEssential: true
  },
  {
    id: "ffmpeg_runtime",
    architecture: "ffmpeg",
    name: { zh: "FFmpeg 音频编解码器 (9.0.1)", en: "FFmpeg Audio Codec (9.0.1)", ja: "FFmpeg オーディオコーデック (9.0.1)" },
    description: {
      zh: "必备工具。负责压缩音频格式的解码（MP3/FLAC/M4A/OGG/Opus/WMA 等）、音频导出与时长探测；WAV 无需它也能用。安装包不捆绑，用到时在此按需下载。",
      en: "Essential tool. Decodes compressed audio formats (MP3/FLAC/M4A/OGG/Opus/WMA...), encodes exports and probes durations; WAV works without it. Not bundled in the installer — download here on demand.",
      ja: "必須ツール。圧縮音声フォーマット（MP3/FLAC/M4A/OGG/Opus/WMA など）のデコード、書き出し、長さ検出を担当。WAV は不要で動作します。インストーラーに同梱せず、必要時にここからダウンロードします。"
    },
    filename: "ffmpeg/bin/ffmpeg.exe",
    fileSize: 111_253_802,
    repoId: "GyanD/codexffmpeg",
    downloadUrl: "https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-essentials_build.zip",
    sha256: "fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9",
    ui_features: [
      { zh: "全格式解码", en: "All-format decode", ja: "全フォーマット対応" },
      { zh: "按需下载", en: "On-demand download", ja: "オンデマンド" }
    ],
    isEssential: true
  },
  {
    id: "musescore_general_sf2",
    architecture: "soundfont",
    name: { zh: "MuseScore General 音源", en: "MuseScore General SoundFont", ja: "MuseScore General 音源" },
    description: { 
      zh: "必需组件。提供高质量的乐器音色库，使 MIDI 听起来像真实的乐器。", 
      en: "Essential component. Provides a high-quality instrument sound library, making MIDI sound like real instruments.",
      ja: "必須コンポーネント。高品質な楽器音源ライブラリを提供し、MIDIを本物の楽器のように聴かせます。"
    },
    filename: "MuseScore_General.sf2",
    fileSize: 216_000_000,
    repoId: "MuScriptor/assets",
    downloadUrl: `${HF}/MuScriptor/assets/resolve/main/MuseScore_General.sf2`,
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "通用音色库", en: "General MIDI", ja: "General MIDI" },
      { zh: "高质量采样", en: "High Quality", ja: "高品質サンプリング" }
    ],
    isEssential: true
  },
  {
    id: "soundfont_fluidr3_gm",
    architecture: "soundfont",
    name: { zh: "FluidR3 GM 音源", en: "FluidR3 GM SoundFont", ja: "FluidR3 GM 音源" },
    description: {
      zh: "经典行业标准 GM 音源（Frank Wen 出品），音色均衡温暖，钢琴与鼓组尤为出色，MIDI 播放的事实标准之一。",
      en: "Classic industry-standard GM soundfont (by Frank Wen) — balanced, warm tones with especially fine piano and drums.",
      ja: "業界標準のクラシックGM音源（Frank Wen 製）。バランスの取れた温かみのある音色で、ピアノとドラムが特に優秀です。"
    },
    filename: "soundfonts/FluidR3_GM.sf2",
    fileSize: 148_398_306,
    repoId: "Jacalz/fluid-soundfont",
    downloadUrl: "https://github.com/Jacalz/fluid-soundfont/raw/master/original-files/FluidR3_GM.sf2",
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "行业经典", en: "Industry classic", ja: "定番" },
      { zh: "钢琴/鼓组出色", en: "Piano & drums", ja: "ピアノ・ドラム得意" }
    ],
    rating: 5,
    newbie_tip: { zh: "想换掉默认音色又怕踩坑：选它。", en: "Safe first replacement for the default sound.", ja: "デフォルトから乗り換える最初の一択。" }
  },
  {
    id: "soundfont_generaluser_gs",
    architecture: "soundfont",
    name: { zh: "GeneralUser GS 音源", en: "GeneralUser GS SoundFont", ja: "GeneralUser GS 音源" },
    description: {
      zh: "仅 30MB 的高效 GS 音源（261 种音色 + 13 套鼓组），深度调制让它小体积也有大表现，与 FluidSynth 兼容性极佳。",
      en: "A highly efficient 30 MB GS soundfont (261 presets + 13 drum kits) — small size, big expressiveness, excellent FluidSynth compatibility.",
      ja: "わずか30MBの高効率GS音源（261音色＋13ドラムキット）。小さなサイズながら表現力豊かで、FluidSynthとの互換性も抜群です。"
    },
    filename: "soundfonts/GeneralUser_GS.sf2",
    fileSize: 32_319_396,
    repoId: "mrbumpy409/GeneralUser-GS",
    downloadUrl: "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2",
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "超小体积", en: "Ultra compact", ja: "超コンパクト" },
      { zh: "GS 兼容", en: "GS compatible", ja: "GS互換" }
    ],
    rating: 4,
    newbie_tip: { zh: "硬盘紧张时的最佳选择，音质远超体积。", en: "Best pick when disk space is tight.", ja: "ディスク節約に最適な選択肢。" }
  },
  {
    id: "soundfont_arachno",
    architecture: "soundfont",
    name: { zh: "Arachno SoundFont 复古音源", en: "Arachno SoundFont (Retro)", ja: "Arachno SoundFont（レトロ）" },
    description: {
      zh: "为复古游戏 MIDI 打造的 128 音色 GM 音源，采样源自 Roland/Korg/Yamaha 经典合成器，怀旧游戏风必备。",
      en: "A 128-preset GM soundfont built for retro game MIDI — samples from classic Roland/Korg/Yamaha synths. Nostalgia essential.",
      ja: "レトロゲームMIDIのために作られた128音色GM音源。Roland/Korg/Yamahaの名機サンプルで懐かしのゲーム音楽に最適です。"
    },
    filename: "soundfonts/Arachno_v1.0.sf2",
    fileSize: 155_405_818,
    repoId: "rwtnb/Drumsthesia",
    downloadUrl: "https://github.com/rwtnb/Drumsthesia/raw/refs/heads/main/Arachno%20SoundFont%20-%20Version%201.0.sf2",
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "复古游戏风", en: "Retro game flavor", ja: "レトロゲーム風" },
      { zh: "经典合成器采样", en: "Classic synth samples", ja: "名機サンプリング" }
    ],
    rating: 4,
    newbie_tip: { zh: "做 8-bit/怀旧编曲时格外对味。", en: "Great flavor for 8-bit / nostalgic arrangements.", ja: "8ビット・懐かしいアレンジにぴったり。" }
  },
  {
    id: "bs_roformer_sw_fixed",
    architecture: "bs_roformer",
    name: { zh: "BS-RoFormer 六声部分离", en: "BS-RoFormer 6-Stem Separation", ja: "BS-RoFormer 6ステム分離" },
    description: { 
      zh: "高精度音频分离模型，可将音频分为人声、鼓、贝斯、钢琴、吉他和其他。", 
      en: "High-precision audio separation model for vocals, drums, bass, piano, guitar, and others.",
      ja: "ボーカル、ドラム、ベース、ピアノ、ギター、その他に分離する高精度モデル。"
    },
    filename: "audio-separator/BS-Rofo-SW-Fixed.ckpt",
    fileSize: 699_412_152,
    repoId: "noblebarkrr/mvsepless_resources",
    downloadUrl: `https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/370198fbb6997e3f5774778254698794e7b1267d/bs_roformer/bs_6stem_fixed.ckpt`,
    sha256: "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
    ui_features: [
      { zh: "六声部分离", en: "6-Stem Separation", ja: "6ステム分離" },
      { zh: "高保真", en: "High Fidelity", ja: "高忠実度" }
    ]
  },
  {
    id: "bs_roformer_leap_xe",
    architecture: "bs_roformer",
    name: { zh: "BS-RoFormer Leap XE 人声提取", en: "BS-RoFormer Leap XE Vocals", ja: "BS-RoFormer Leap XE ボーカル" },
    description: { 
      zh: "SOTA 级别的人声分离模型，适用于极致清晰的人声提取。", 
      en: "SOTA vocal separation model for extremely clear vocal extraction.",
      ja: "極めてクリアなボーカル抽出のためのSOTAボーカル分離モデル。"
    },
    filename: "audio-separator/bs_leap_xe_voc.ckpt",
    fileSize: 267_796_851,
    repoId: "pcunwa/BS-Roformer-Leap",
    downloadUrl: `https://huggingface.co/pcunwa/BS-Roformer-Leap/resolve/4e47d6662ae82eaa8b4ac4329fe66099a843b48e/Xe/bs_leap_xe_voc.ckpt`,
    sha256: "b739c1d2d87a81cd3dd3844ed9ad0bd678708c7a0a761a03a1aaff9af79a096d",
    ui_features: [
      { zh: "人声极致清晰", en: "Crystal Clear Vocals", ja: "超クリアボーカル" }
    ]
  },
  {
    id: "bs_polarformer_accompaniment",
    architecture: "polarformer",
    name: { zh: "BS-PolarFormer 伴奏提取", en: "BS-PolarFormer Accompaniment", ja: "BS-PolarFormer 伴奏抽出" },
    description: { 
      zh: "专门用于伴奏和人声分离的 ONNX 模型，性能优异。", 
      en: "ONNX model specialized for accompaniment and vocal separation.",
      ja: "伴奏とボーカル分離に特化したONNXモデル。"
    },
    filename: "audio-separator/bs_polarformer_fp16.onnx",
    fileSize: 108_325_429,
    repoId: "bgkb/bs_polarformer",
    downloadUrl: `https://huggingface.co/bgkb/bs_polarformer/resolve/9158719ee2173edd480a735764627526506fe4af/bs_polarformer_fp16.onnx`,
    sha256: "76424289ea586bae4bbdb289383b0269b099416471e2b05068d02aa0b0c01467",
    ui_features: [
      { zh: "ONNX 加速", en: "ONNX Accelerated", ja: "ONNX加速" }
    ]
  },
  {
    id: "musescore_studio_4",
    architecture: "musescore",
    name: { zh: "MuseScore Studio 4 运行时", en: "MuseScore Studio 4 Runtime", ja: "MuseScore Studio 4 実行環境" },
    description: { 
      zh: "必需组件。用于将 MIDI 转换为乐谱（PDF）或 MusicXML 格式。", 
      en: "Essential component. Used to convert MIDI to sheet music (PDF) or MusicXML format.",
      ja: "必須コンポーネント。MIDIを楽譜（PDF）またはMusicXML形式に変換するために使用されます。"
    },
    filename: "musescore/4.7.4/bin/MuseScore4.exe",
    fileSize: 127_807_488,
    repoId: "musescore/MuseScore",
    downloadUrl: `https://github.com/musescore/MuseScore/releases/download/v4.7.4/MuseScore-Studio-4.7.4.260706075-x86_64.msi`,
    sha256: "64fe70e5cb9ffe159d047d1e88db567bd101f60d36b0de28feb674716929a378",
    ui_features: [
      { zh: "乐谱生成", en: "Sheet Music", ja: "楽譜生成" },
      { zh: "PDF 导出", en: "PDF Export", ja: "PDF エクスポート" }
    ],
    isEssential: true
  },
  {
    id: "miros_streaming_amt",
    architecture: "miros",
    name: { zh: "MIROS 流式转谱模型", en: "MIROS Streaming AMT", ja: "MIROS ストリーミング AMT" },
    description: { 
      zh: "最新的流式全自动音乐转谱模型，支持实时处理预览。", 
      en: "Latest streaming automatic music transcription model, supporting real-time processing.",
      ja: "最新のストリーミング自動音楽転写モデル。"
    },
    filename: "amt/miros/pretrained_msd.pt",
    fileSize: 1_316_802_088,
    repoId: "minzwon/MusicFM",
    downloadUrl: `${HF}/minzwon/MusicFM/resolve/546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c/pretrained_msd.pt`,
    sha256: "218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded",
    ui_features: [
      { zh: "流式处理", en: "Streaming", ja: "ストリーミング" }
    ]
  },
  {
    id: "transkun_v2_aug_official",
    architecture: "transkun",
    name: { zh: "TransKun V2 Aug (官方增强版)", en: "TransKun V2 Aug (Official)", ja: "TransKun V2 Aug (公式)" },
    description: { 
      zh: "TransKun 的官方增强版本，包含更强的泛化能力和数据增强支持。", 
      en: "Official augmented version of TransKun with better generalization and data augmentation support.",
      ja: "より優れた汎用性とデータ拡張サポートを備えたTransKunの公式拡張バージョン。"
    },
    filename: "amt/transkun_v2_aug/checkpoints/transkun_v2_aug.pt",
    fileSize: 100_000_000,
    repoId: "mason369/music-to-midi-assets",
    downloadUrl: `${HF}/mason369/music-to-midi-assets/resolve/main/transkun_v2_aug/transkun_v2_aug.zip`,
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "增强型转谱", en: "Augmented AMT", ja: "拡張 AMT" }
    ]
  },
  {
    id: "muscriptor_large",
    architecture: "miros", // MuScriptor models are often categorized under MIROS or similar for backend purposes
    name: { zh: "MuScriptor Large (推荐)", en: "MuScriptor Large (Recommended)", ja: "MuScriptor Large (推奨)" },
    description: { 
      zh: "MuScriptor 系列最强模型，支持极其精细的力度和时值还原。", 
      en: "Strongest MuScriptor model, supporting extremely fine velocity and duration restoration.",
      ja: "MuScriptorシリーズ最強モデル、極めて精細なベロシティ与デュレーションの復元に対応。"
    },
    filename: "amt/muscriptor/large/model.safetensors",
    fileSize: 5_465_642_136,
    repoId: "cocktailpeanut/muscriptor-large",
    revision: "main",
    downloadUrl: `${HF}/cocktailpeanut/muscriptor-large/resolve/main/model.safetensors`,
    sha256: "ac4eb6ea87dfc26b6ca6b954c6b967ab87ad4c7d08e078b25214f13ed051f397",
    rating: 5,
    newbie_tip: { zh: "多乐器转谱首选，除了慢没毛病。", en: "First choice for multi-instrument transcription, perfect except for speed.", ja: "多楽器採譜の第一選択、速度以外は完璧です。" }
  },
  {
    id: "muscriptor_medium",
    architecture: "miros",
    name: { zh: "MuScriptor Medium", en: "MuScriptor Medium", ja: "MuScriptor Medium" },
    description: { 
      zh: "平衡速度与精度的 MuScriptor 模型。", 
      en: "Balanced MuScriptor model for speed and accuracy.",
      ja: "速度と精度のバランスが取れたMuScriptorモデル。"
    },
    filename: "amt/muscriptor/medium/model.safetensors",
    fileSize: 1_228_144_472,
    repoId: "cocktailpeanut/muscriptor-medium",
    revision: "main",
    downloadUrl: `${HF}/cocktailpeanut/muscriptor-medium/resolve/main/model.safetensors`,
    sha256: "ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec",
    rating: 4,
  },
  {
    id: "muscriptor_small",
    architecture: "miros",
    name: { zh: "MuScriptor Small", en: "MuScriptor Small", ja: "MuScriptor Small" },
    description: {
      zh: "极速 MuScriptor 模型，适合低配电脑或快速草稿。",
      en: "Fast MuScriptor model, suitable for low-end PCs or quick drafts.",
      ja: "高速MuScriptorモデル、低スペックPCや素早いドラフトに適しています。"
    },
    filename: "amt/muscriptor/small/model.safetensors",
    fileSize: 411_888_600,
    repoId: "cocktailpeanut/muscriptor-small",
    revision: "main",
    downloadUrl: `${HF}/cocktailpeanut/muscriptor-small/resolve/main/model.safetensors`,
    sha256: "bbd482c786b895cf7d8f44185073d951adae2ebb8a66f82ca84cd1f84569549c",
    rating: 3,
  },
  {
    id: "whisper_small",
    architecture: "whisper",
    name: { zh: "Whisper Small 歌词识别（推荐）", en: "Whisper Small Lyrics (Recommended)", ja: "Whisper Small 歌詞認識（推奨）" },
    description: {
      zh: "自动识别中文 / 英文 / 日文等歌曲歌词并带时间轴，可在结果工作台中自由编辑后写回 MIDI。",
      en: "Recognizes Chinese / English / Japanese lyrics with timestamps; edit freely in the result workbench and write back to MIDI.",
      ja: "中国語・英語・日本語の歌詞をタイムスタンプ付きで自動認識し、結果ワークベンチで自由に編集してMIDIに書き戻せます。"
    },
    filename: "whisper/small/model.bin",
    fileSize: 483_546_902,
    repoId: "Systran/faster-whisper-small",
    downloadUrl: `${HF}/Systran/faster-whisper-small/resolve/main/model.bin`,
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "中英日识别", en: "ZH/EN/JA Recognition", ja: "中英日認識" },
      { zh: "时间轴歌词", en: "Timed Lyrics", ja: "タイム付き歌詞" }
    ],
    rating: 5,
    newbie_tip: { zh: "歌词功能推荐型号：中文、英文、日文识别均衡。", en: "Recommended for lyrics: balanced ZH/EN/JA accuracy.", ja: "歌詞機能の推奨モデル：中英日のバランスが良い。" },
    isEssential: true,
    extraFiles: [
      { filename: "whisper/small/config.json", downloadUrl: `${HF}/Systran/faster-whisper-small/resolve/main/config.json`, fileSize: 2_370 },
      { filename: "whisper/small/tokenizer.json", downloadUrl: `${HF}/Systran/faster-whisper-small/resolve/main/tokenizer.json`, fileSize: 2_203_239 },
      { filename: "whisper/small/vocabulary.txt", downloadUrl: `${HF}/Systran/faster-whisper-small/resolve/main/vocabulary.txt`, fileSize: 459_861 },
    ]
  },
  {
    id: "whisper_medium",
    architecture: "whisper",
    name: { zh: "Whisper Medium 歌词识别（高精度）", en: "Whisper Medium Lyrics (High Accuracy)", ja: "Whisper Medium 歌詞認識（高精度）" },
    description: {
      zh: "歌词识别精度更高，速度较慢、体积较大，适合对歌词准确率要求极高的场景。",
      en: "Higher lyric accuracy but slower and larger; for cases demanding the best recognition quality.",
      ja: "歌詞認識の精度がより高い一方、速度は遅くサイズも大きい。最高精度が必要な場合に。"
    },
    filename: "whisper/medium/model.bin",
    fileSize: 1_530_578_521,
    repoId: "Systran/faster-whisper-medium",
    downloadUrl: `${HF}/Systran/faster-whisper-medium/resolve/main/model.bin`,
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "最高精度", en: "Highest Accuracy", ja: "最高精度" }
    ],
    rating: 4,
    extraFiles: [
      { filename: "whisper/medium/config.json", downloadUrl: `${HF}/Systran/faster-whisper-medium/resolve/main/config.json`, fileSize: 2_370 },
      { filename: "whisper/medium/tokenizer.json", downloadUrl: `${HF}/Systran/faster-whisper-medium/resolve/main/tokenizer.json`, fileSize: 2_203_239 },
      { filename: "whisper/medium/vocabulary.txt", downloadUrl: `${HF}/Systran/faster-whisper-medium/resolve/main/vocabulary.txt`, fileSize: 459_861 },
    ]
  },
  {
    id: "whisper_base",
    architecture: "whisper",
    name: { zh: "Whisper Base 歌词识别（轻量）", en: "Whisper Base Lyrics (Lightweight)", ja: "Whisper Base 歌詞認識（軽量）" },
    description: {
      zh: "轻量快速，适合纯英文或日语等拼写规整歌词的快速提取。",
      en: "Lightweight and fast; good for quick extraction of regularly-spelled lyrics such as English or Japanese.",
      ja: "軽量・高速。英語や日本語など綴りが規則的な歌詞の素早い抽出に適します。"
    },
    filename: "whisper/base/model.bin",
    fileSize: 145_433_585,
    repoId: "Systran/faster-whisper-base",
    downloadUrl: `${HF}/Systran/faster-whisper-base/resolve/main/model.bin`,
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "极速提取", en: "Fast Extraction", ja: "高速抽出" }
    ],
    rating: 3,
    extraFiles: [
      { filename: "whisper/base/config.json", downloadUrl: `${HF}/Systran/faster-whisper-base/resolve/main/config.json`, fileSize: 2_370 },
      { filename: "whisper/base/tokenizer.json", downloadUrl: `${HF}/Systran/faster-whisper-base/resolve/main/tokenizer.json`, fileSize: 2_203_239 },
      { filename: "whisper/base/vocabulary.txt", downloadUrl: `${HF}/Systran/faster-whisper-base/resolve/main/vocabulary.txt`, fileSize: 459_861 },
    ]
  },
  {
    id: "whisper_tiny",
    architecture: "whisper",
    name: { zh: "Whisper Tiny 歌词识别（最小体积）", en: "Whisper Tiny Lyrics (Smallest)", ja: "Whisper Tiny 歌詞認識（最小）" },
    description: {
      zh: "体积最小的歌词模型，速度最快但精度有限，适合低配电脑试玩歌词功能。",
      en: "Smallest lyrics model — fastest but limited accuracy; for low-end machines trying the lyrics feature.",
      ja: "最小サイズの歌詞モデル。最速だが精度は限定的。低スペックPCでの歌詞機能体験に。"
    },
    filename: "whisper/tiny/model.bin",
    fileSize: 75_487_731,
    repoId: "Systran/faster-whisper-tiny",
    downloadUrl: `${HF}/Systran/faster-whisper-tiny/resolve/main/model.bin`,
    sha256: "verified_by_runtime",
    ui_features: [
      { zh: "最小体积", en: "Smallest Size", ja: "最小サイズ" }
    ],
    rating: 2,
    extraFiles: [
      { filename: "whisper/tiny/config.json", downloadUrl: `${HF}/Systran/faster-whisper-tiny/resolve/main/config.json`, fileSize: 2_370 },
      { filename: "whisper/tiny/tokenizer.json", downloadUrl: `${HF}/Systran/faster-whisper-tiny/resolve/main/tokenizer.json`, fileSize: 2_203_239 },
      { filename: "whisper/tiny/vocabulary.txt", downloadUrl: `${HF}/Systran/faster-whisper-tiny/resolve/main/vocabulary.txt`, fileSize: 459_861 },
    ]
  }
];

export function applyAmtMirror(url: string, mirror: MirrorSource): string {
  return applyMirror(url, mirror);
}
