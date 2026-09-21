// 歌曲生成模型目录（内嵌方案）——三个模型家族：YuE-2 / ACE-Step / HeartMuLa。
// 每个模型是"多文件清单"：下载到 <song_models_dir>/<id>/<相对路径>，保持 HuggingFace
// 仓库的子目录结构，这样 Python 侧可直接 from_pretrained(<模型目录>) 加载（含
// trust_remote_code 自定义代码模型）。URL 首选 hf-mirror（国内直连），回退官方 HF。
import { I18nText, t18 } from "./msst-catalog";

export type SongModelFamily = "yue2" | "acestep" | "heartmula";

export interface SongCatalogFile {
  /** 相对模型目录的路径（保留 HF 仓库子目录结构） */
  path: string;
  /** 下载直链（按顺序尝试：hf-mirror → huggingface） */
  urls: string[];
  size: number;
  sha256?: string;
  /**
   * 等价替代路径（相对模型目录）。安装校验时主路径或任一替代路径存在即算就绪
   * （用于同一模型的不同 DiT 变体，如 acestep 的 xl-turbo / turbo 二选一）。
   * 仅在主路径缺失时才会下载主路径文件。
   */
  alts?: string[];
}

export interface SongCatalogModel {
  id: string;
  family: SongModelFamily;
  label: I18nText;
  description: I18nText;
  license: string;
  sourceRepo: string;
  /** 显存需求提示（GB） */
  vramGb: number;
  /** 模型能力标签 */
  supportsMultiTrack: boolean;
  supportsMidi: boolean;
  languages: string;
  files: SongCatalogFile[];
}

export const SONG_FAMILY_LABELS: Record<SongModelFamily, I18nText> = {
  yue2:     { zh: "YuE-2 · 歌词成曲（双语）", en: "YuE-2 · Lyrics-to-Song", ja: "YuE-2 · 歌詞から楽曲" },
  acestep:  { zh: "ACE-Step · 文本成曲", en: "ACE-Step · Text-to-Music", ja: "ACE-Step · テキストから音楽" },
  heartmula:{ zh: "HeartMuLa · 歌词成曲", en: "HeartMuLa · Lyrics-to-Song", ja: "HeartMuLa · 歌詞から楽曲" },
};

// ---------------------------------------------------------------------------
// HF 直链构造（镜像优先）
// ---------------------------------------------------------------------------

const HF_MIRROR = "https://hf-mirror.com";
const HF_OFFICIAL = "https://huggingface.co";

function hf(repo: string, path: string): string[] {
  return [`${HF_MIRROR}/${repo}/resolve/main/${path}`, `${HF_OFFICIAL}/${repo}/resolve/main/${path}`];
}

// ---------------------------------------------------------------------------
// YuE-2（m-a-p/YuE2-3B + m-a-p/YuE2-Vae，license: CC-BY-NC-4.0）
// 主模型 + VAE 统一装在 yue2-3b/ 下（vae/ 子目录），生成时一并需要。
// ---------------------------------------------------------------------------

const YUE2_3B_REPO = "m-a-p/YuE2-3B";
const YUE2_VAE_REPO = "m-a-p/YuE2-Vae";

// ---------------------------------------------------------------------------
// ACE-Step（Apache-2.0 / MIT）
// ---------------------------------------------------------------------------

const ACE_V15_REPO = "ACE-Step/Ace-Step1.5";

// ---------------------------------------------------------------------------
// HeartMuLa：已从目录移除（16GB 显卡上主模型+Codec 显存需求超出物理显存，
// 生成中会触发驱动级崩溃）。sidecar 里的 heartmula 入口保留但不可达。
// ---------------------------------------------------------------------------

function repoFiles(repo: string, files: Array<[string, number, string?]>): SongCatalogFile[] {
  return files.map(([path, size, sha256]) => ({ path, size, sha256, urls: hf(repo, path) }));
}

export const SONG_MODEL_CATALOG: SongCatalogModel[] = [
  {
    id: "yue2-3b",
    family: "yue2",
    label: { zh: "YuE2-3B + VAE", en: "YuE2-3B + VAE", ja: "YuE2-3B + VAE" },
    description: {
      zh: "m-a-p 出品 3B 歌词成曲模型：支持中文/英文歌词，人声+伴奏双轨输出，音质出色。含 VAE 解码器，共约 7.3 GB。内置能力：ABC 乐谱自动转 MIDI（依赖 music21，已集成到运行时）；可选分轨分离（依赖 demucs，首次使用时自动安装）；LRC 歌词文件。",
      en: "m-a-p 3B lyrics-to-song model: Chinese/English lyrics, vocal+accompaniment dual-track output. Includes VAE decoder, ~7.3 GB total. Built-in: ABC score to MIDI conversion (music21, bundled in runtime); optional stem separation (demucs, auto-installed on first use); LRC lyrics.",
      ja: "m-a-p製 3B 歌詞から楽曲生成モデル：中国語/英語の歌詞に対応、ボーカル+伴奏の2トラック出力。VAE デコーダ含む、合計約 7.3 GB。ABC 楽譜の MIDI 変換（music21 同梱）、分離トラック（demucs、初回使用時に自動インストール）、LRC 歌詞に対応。",
    },
    license: "CC-BY-NC-4.0",
    sourceRepo: YUE2_3B_REPO,
    vramGb: 12,
    supportsMultiTrack: true,
    supportsMidi: true,
    languages: "zh / en",
    files: [
      ...repoFiles(YUE2_3B_REPO, [
        ["model.safetensors", 7261441640, "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9"],
        ["config.json", 959],
        ["generation_config.json", 204],
        ["yue2_generation_config.json", 466],
        ["weights_manifest.json", 179],
        ["modeling_yue2.py", 32983],
        ["qwen.tiktoken", 2561218],
        ["yue2_infer-0.1.5-py3-none-any.whl", 66117],
      ]),
      ...repoFiles(YUE2_VAE_REPO, [
        ["model.safetensors", 530512720, "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346"],
        ["config.json", 1378],
        ["weights_manifest.json", 178],
        ["modeling_vae.py", 25265],
      ]).map((f) => ({ ...f, path: `vae/${f.path}` })),
    ],
  },
  {
    id: "acestep-v1.5",
    family: "acestep",
    label: { zh: "ACE-Step v1.5 Turbo", en: "ACE-Step v1.5 Turbo", ja: "ACE-Step v1.5 Turbo" },
    description: {
      zh: "ACE-Step 1.5 官方 Turbo 模型（MIT 许可，可商用）：5Hz LM + DiT + VAE 全套官方权重，约 10.1 GB，支持中英文等多语言歌词。若本地已存在 XL 变体（acestep-v15-xl-turbo）会自动优先使用。",
      en: "ACE-Step 1.5 official Turbo model (MIT license): full official weights for 5Hz LM + DiT + VAE, ~10.1 GB, multi-language lyrics. Automatically prefers the XL variant (acestep-v15-xl-turbo) when present locally.",
      ja: "ACE-Step 1.5 公式 Turbo モデル（MIT ライセンス）：5Hz LM + DiT + VAE の公式ウェイト一式、約 10.1 GB、多言語歌詞対応。XL バリアント（acestep-v15-xl-turbo）があれば自動的に優先使用。",
    },
    license: "MIT",
    sourceRepo: ACE_V15_REPO,
    vramGb: 12,
    supportsMultiTrack: false,
    supportsMidi: false,
    languages: "zh / en / multi",
    // 全部来自官方主仓库 ACE-Step/Ace-Step1.5，落盘为
    // <song_models_dir>/acestep-v1.5/checkpoints/<path>（与 HF 仓库结构一致）
    files: [
      // DiT 权重：本机若已装 XL 变体（acestep-v15-xl-turbo，质量更好），安装校验
      // 会把下面的 turbo 文件视为等价就绪，不再重复下载 4.8GB 的 turbo 权重。
      // 注意：.py 文件不在 LISTED_EXTS 中，不参与校验，但仍会随清单下载。
      ...repoFiles(ACE_V15_REPO, [
        ["acestep-v15-turbo/model.safetensors", 4787825604],
        ["acestep-v15-turbo/config.json", 1968],
        ["acestep-v15-turbo/configuration_acestep_v15.py", 13130],
        ["acestep-v15-turbo/modeling_acestep_v15_turbo.py", 96036],
        ["acestep-v15-turbo/silence_latent.pt", 3841215],
        ["vae/config.json", 425],
        ["vae/diffusion_pytorch_model.safetensors", 337431388],
        ["Qwen3-Embedding-0.6B/model.safetensors", 1191586416],
        ["Qwen3-Embedding-0.6B/config.json", 1359],
        ["Qwen3-Embedding-0.6B/tokenizer.json", 11423705],
        ["Qwen3-Embedding-0.6B/tokenizer_config.json", 5404],
        ["Qwen3-Embedding-0.6B/special_tokens_map.json", 613],
        ["Qwen3-Embedding-0.6B/added_tokens.json", 707],
        ["Qwen3-Embedding-0.6B/chat_template.jinja", 4116],
        ["Qwen3-Embedding-0.6B/vocab.json", 2776833],
        ["Qwen3-Embedding-0.6B/merges.txt", 1671853],
        ["acestep-5Hz-lm-1.7B/model.safetensors", 3708521528],
        ["acestep-5Hz-lm-1.7B/config.json", 1385],
        ["acestep-5Hz-lm-1.7B/tokenizer.json", 24321939],
        ["acestep-5Hz-lm-1.7B/tokenizer_config.json", 14072925],
        ["acestep-5Hz-lm-1.7B/special_tokens_map.json", 1824199],
        ["acestep-5Hz-lm-1.7B/added_tokens.json", 2217787],
        ["acestep-5Hz-lm-1.7B/chat_template.jinja", 4168],
        ["acestep-5Hz-lm-1.7B/vocab.json", 2776833],
        ["acestep-5Hz-lm-1.7B/merges.txt", 1671853],
        // base 模型：多轨任务（叠加音轨 lego / 分轨分离 extract / 补全 complete）
        // 仅 base 支持（官方 TASK_TYPES_TURBO 不含这三个），turbo/XL 不需要它
        ["acestep-v15-base/model.safetensors", 4787825604],
        ["acestep-v15-base/config.json", 1940],
        ["acestep-v15-base/configuration_acestep_v15.py", 13130],
        ["acestep-v15-base/modeling_acestep_v15_base.py", 95545],
        ["acestep-v15-base/silence_latent.pt", 3841215],
      ]).map((f) => {
        // 为 turbo 的 5 个 DiT 文件补充 XL 变体等价路径（参与校验的 3 个 +
        // 2 个 .py 代码文件；xl 目录的 modeling 文件名不同）
        const altMap: Record<string, string[]> = {
          "acestep-v15-turbo/model.safetensors": ["checkpoints/acestep-v15-xl-turbo/model.safetensors"],
          "acestep-v15-turbo/config.json": ["checkpoints/acestep-v15-xl-turbo/config.json"],
          "acestep-v15-turbo/configuration_acestep_v15.py": ["checkpoints/acestep-v15-xl-turbo/configuration_acestep_v15.py"],
          "acestep-v15-turbo/modeling_acestep_v15_turbo.py": ["checkpoints/acestep-v15-xl-turbo/modeling_acestep_v15_xl_turbo.py"],
          "acestep-v15-turbo/silence_latent.pt": ["checkpoints/acestep-v15-xl-turbo/silence_latent.pt"],
          // base 模型：多轨任务（lego/extract/complete）必需，无替代路径
        };
        // 统一加 checkpoints/ 前缀（落盘为 <model>/checkpoints/<path>）
        const prefixed = { ...f, path: `checkpoints/${f.path}` };
        return altMap[f.path] ? { ...prefixed, alts: altMap[f.path] } : prefixed;
      }),
    ],
  },
];

export function songModelTotalSize(model: SongCatalogModel): number {
  return model.files.reduce((acc, f) => acc + f.size, 0);
}

// ---------------------------------------------------------------------------
// 安装状态判定：list_song_models 只回报这些扩展名的文件（与 Rust 端白名单一致），
// 因此只依据它们判断安装完整性；py/tiktoken 等小文件不参与校验但会随下载落盘。
// ---------------------------------------------------------------------------

const LISTED_EXTS = ["safetensors", "ckpt", "pt", "pth", "onnx", "whl", "bin", "yaml", "json", "sf2", "mid", "wav"];

export interface SongInstallStatus {
  installed: boolean;
  /** 已就绪的关键文件数 / 需要的关键文件数 */
  ready: number;
  required: number;
  missing: string[];
}

export function getSongInstallStatus(
  installedFiles: Array<{ filename: string; size: number }>,
  model: SongCatalogModel,
): SongInstallStatus {
  const listed = new Set(installedFiles.map((f) => f.filename.replace(/\\/g, "/")));
  const prefix = `${model.id}/`;
  const required: string[] = [];
  const missing: string[] = [];
  let ready = 0;
  for (const f of model.files) {
    const ext = f.path.split(".").pop()?.toLowerCase() ?? "";
    if (!LISTED_EXTS.includes(ext)) continue; // 不可见的文件不参与校验
    required.push(f.path);
    // 主路径或任一等价替代路径存在即算就绪（如 xl-turbo/turbo 变体二选一）。
    // 替代路径只按文件名匹配（变体权重体积不同，不能用 size 匹配）
    const hasAlt = (f.alts ?? []).some((alt) => listed.has(prefix + alt));
    if (listed.has(prefix + f.path) || hasAlt) ready += 1;
    else missing.push(f.path);
  }
  return { installed: required.length > 0 && missing.length === 0, ready, required: required.length, missing };
}

export function songModelById(id: string): SongCatalogModel | undefined {
  return SONG_MODEL_CATALOG.find((m) => m.id === id);
}

export function songCatalogLabel(model: SongCatalogModel, lang: string): string {
  return t18(model.label, lang);
}
