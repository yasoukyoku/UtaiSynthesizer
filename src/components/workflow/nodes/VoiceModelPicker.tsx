import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../../../store/app";
import {
  useVoiceModelStore,
  voiceVersionBadge,
  voiceSpeakerOptions,
  formatSampleRateKhz,
  vocoderFormatMatches,
  type VoiceModelEntry,
  type VoiceType,
} from "../../../store/voice-models";
import { ParamSlider } from "./ParamSlider";
import type { SpkMixEntry } from "../../../lib/workflow/voiceDefaults";
import { t18, type I18nText } from "../../../lib/models/msst-catalog";

/** Strings shared by BOTH voice nodes (RVC + SoVITS) — node-specific ones stay in the nodes. */
export const VOICE_STRINGS = {
  f0Shift: { zh: "变调", en: "Pitch", ja: "ピッチ" },
  f0ShiftTip: { zh: "音高平移（半音），+12 = 升一个八度", en: "Pitch shift in semitones, +12 = one octave up", ja: "ピッチシフト（半音）、+12 = 1オクターブ上" },
  noise: { zh: "噪声", en: "Noise", ja: "ノイズ" },
  noiseTip: { zh: "合成随机性（noise_scale）", en: "Synthesis randomness (noise_scale)", ja: "合成のランダム性（noise_scale）" },
  off: { zh: "关", en: "Off", ja: "オフ" },
  gpuExtract: { zh: "GPU特征提取", en: "GPU extraction", ja: "GPU特徴抽出" },
  gpuExtractTip: { zh: "特征/音高提取（ContentVec+RMVPE）改在 GPU 上跑：更快但占显存；默认 CPU 更稳更省显存", en: "Run ContentVec+RMVPE extraction on the GPU: faster but uses VRAM; the CPU default is safer", ja: "特徴/ピッチ抽出（ContentVec+RMVPE）を GPU で実行：高速だが VRAM を消費。既定の CPU が安全" },
  gpuExtractLowVram: { zh: "显卡显存不足 12GB，已禁用——GPU 特征提取实测峰值约 9.4GB，低显存卡会爆显存（CPU 提取不受影响）", en: "Disabled: needs a ≥12GB GPU — GPU extraction peaks at ~9.4GB in practice and would exhaust smaller cards (CPU extraction is unaffected)", ja: "無効：12GB 以上の GPU が必要です — GPU 抽出は実測ピーク約 9.4GB で、それ未満のカードでは VRAM が枯渇します（CPU 抽出は影響なし）" },
  diffBadgeTip: { zh: "已附带扩散模型（可用浅扩散）", en: "Diffusion attachment present (shallow diffusion available)", ja: "拡散モデルあり（浅い拡散が利用可能）" },
  blendTitle: { zh: "声线混合", en: "Voice blend", ja: "声質ブレンド" },
  blendTip: { zh: "混合多个歌手的音色生成新声线：各歌手权重会自动归一化为占比；未添加时使用默认歌手", en: "Blend multiple speakers' timbres into a new voice — weights are auto-normalized to a share; the default speaker is used when empty", ja: "複数話者の声質を混ぜて新しい声を作る — 各重みは自動で比率に正規化。未追加時は既定の話者を使用" },
  blendEmpty: { zh: "未混合 — 使用默认歌手", en: "No blend — default speaker", ja: "ブレンドなし — 既定の話者" },
  blendAdd: { zh: "＋ 添加歌手", en: "+ Add speaker", ja: "＋ 話者を追加" },
  blendWeight: { zh: "占比", en: "Weight", ja: "比重" },
} satisfies Record<string, I18nText>;

/** S66 GPU-extraction VRAM gate: the feature's measured steady peak is ~9.4 GB (user, two
 *  runs), so enabling it needs a ≥12 GB card. nvidia-smi truth, cached module-level (one
 *  subprocess per session, not per node mount); undetermined / non-NVIDIA fails OPEN (the
 *  variant_supported convention — DirectML cards can't be queried reliably). */
const GPU_EXTRACT_MIN_VRAM_MB = 12_000;
let vramProbe: Promise<number | null> | null = null;
function nvidiaVramMb(): Promise<number | null> {
  vramProbe ??= invoke<{ nvidia_vram_mb: number | null }>("get_hardware_info")
    .then((h) => h.nvidia_vram_mb)
    .catch(() => null);
  return vramProbe;
}

/** The aux-extractor device toggle row — identical on BOTH voice nodes. */
export function GpuExtractRow({ value, lang, onChange }: {
  value: boolean;
  lang: string;
  onChange: (v: boolean) => void;
}) {
  const [vram, setVram] = useState<number | null>(null);
  useEffect(() => {
    let alive = true;
    void nvidiaVramMb().then((v) => {
      if (alive) setVram(v);
    });
    return () => {
      alive = false;
    };
  }, []);
  // Gate only the ENABLE direction: a pre-existing true (older project / user insisting)
  // stays visible and can always be turned OFF.
  const lowVram = vram !== null && vram < GPU_EXTRACT_MIN_VRAM_MB;
  const blocked = lowVram && !value;
  const tip = blocked ? t18(VOICE_STRINGS.gpuExtractLowVram, lang) : t18(VOICE_STRINGS.gpuExtractTip, lang);
  return (
    <div className="sep-param-row" style={blocked ? { opacity: 0.55 } : undefined}>
      <label title={tip}>
        {t18(VOICE_STRINGS.gpuExtract, lang)}
      </label>
      <input
        type="checkbox"
        checked={value}
        disabled={blocked}
        title={tip}
        onChange={(e) => onChange(e.target.checked)}
      />
    </div>
  );
}

/**
 * Resolve a voice node's selected model from its GRAPH params against the installed list, and
 * keep the persisted `voiceName` / `modelPath` in sync. Derived from params, never mirrored
 * into local state — same rule as SeparationNode: the modal-local undo restores node params via
 * setNodes WITHOUT remounting, and a useState mirror would keep showing (and re-committing) the
 * undone selection.
 */
export function useVoiceModelSelection(
  voiceType: VoiceType,
  params: Record<string, unknown>,
  updateParams: (updates: Record<string, unknown>) => void,
  /** Params force-reset on ANY real model switch (incl. the silent deleted-model fallback
   * below, which bypasses the node's own onSelect) — e.g. speaker_id, and the asset-gated
   * SoVITS toggles whose stale `true` on a model without the asset is a runtime error. */
  switchResets: Record<string, unknown> = { speaker_id: null },
): { models: VoiceModelEntry[]; selected: VoiceModelEntry | undefined } {
  const models = useVoiceModelStore((s) => s.models[voiceType]);
  useEffect(() => { void useVoiceModelStore.getState().fetchModels(); }, []);

  const selectedName = (params.voiceName as string) ?? models[0]?.name ?? "";
  const selected = models.find((m) => m.name === selectedName) ?? models[0];

  // Persist the RESOLVED selection whenever it drifts from the params: first mount (auto-pick
  // the first installed model), a deleted model falling back, or a moved models dir changing
  // `path`. A real model SWITCH also applies switchResets — a stale index from the previous
  // model could exceed the new one's n_speakers / reference assets it doesn't have.
  useEffect(() => {
    if (!selected) return;
    if (params.voiceName !== selected.name || params.modelPath !== selected.path) {
      updateParams({
        voiceName: selected.name,
        modelPath: selected.path,
        ...(params.voiceName !== undefined && params.voiceName !== selected.name
          ? switchResets
          : {}),
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.name, selected?.path, params.voiceName, params.modelPath]);

  return { models, selected };
}

/**
 * Model dropdown + meta row (version badge / sample rate / index tag / speaker count), or the
 * "no models installed → go import" empty state. onSelect gets the full entry so the node can
 * write voiceName + modelPath (+ reset speaker_id) in one params update.
 */
export function VoiceModelPicker({ models, selected, lang, onSelect }: {
  models: VoiceModelEntry[];
  selected: VoiceModelEntry | undefined;
  lang: string;
  onSelect: (m: VoiceModelEntry) => void;
}) {
  const toggleModelManager = useAppStore((s) => s.toggleModelManager);

  if (models.length === 0) {
    return (
      <div className="voice-no-model">
        <span className="sep-no-model">
          {t18({ zh: "未安装模型", en: "No models installed", ja: "モデル未インストール" }, lang)}
        </span>
        <button className="voice-manage-btn" onClick={(e) => { e.stopPropagation(); toggleModelManager(); }}>
          {t18({ zh: "去资源管理导入", en: "Import in Resource Manager", ja: "リソース管理で取り込む" }, lang)}
        </button>
      </div>
    );
  }

  const badge = selected ? voiceVersionBadge(selected) : null;
  const speakerCount = selected ? voiceSpeakerOptions(selected).length : 0;

  return (
    <>
      <select
        className="sep-model-select"
        value={selected?.name ?? ""}
        onChange={(e) => {
          const m = models.find((x) => x.name === e.target.value);
          if (m) onSelect(m);
        }}
      >
        {models.map((m) => (
          <option key={m.name} value={m.name}>{m.name}</option>
        ))}
      </select>
      {selected && (
        <div className="voice-model-meta">
          {badge && <span className="ver-badge">{badge}</span>}
          <span>{formatSampleRateKhz(selected.sample_rate)}</span>
          {selected.index_path && (
            <span className="ver-badge" title={t18({ zh: "已附带检索/聚类文件", en: "Index/cluster asset present", ja: "インデックス/クラスタあり" }, lang)}>
              IDX
            </span>
          )}
          {selected.diffusion_path && (
            <span className="ver-badge" title={t18(VOICE_STRINGS.diffBadgeTip, lang)}>
              DIFF
            </span>
          )}
          {speakerCount > 1 && (
            <span>{speakerCount} {t18({ zh: "歌手", en: "speakers", ja: "話者" }, lang)}</span>
          )}
        </div>
      )}
    </>
  );
}

/** Speaker dropdown row — renders NOTHING for single-speaker models (contract: null = 0). */
export function SpeakerSelect({ model, value, onChange, lang }: {
  model: VoiceModelEntry | undefined;
  value: number | null;
  onChange: (id: number) => void;
  lang: string;
}) {
  const opts = model ? voiceSpeakerOptions(model) : [];
  if (opts.length === 0) return null;
  return (
    <div className="sep-param-row">
      <label title={t18({ zh: "多歌手模型的目标歌手", en: "Target speaker of a multi-speaker model", ja: "マルチスピーカーモデルの話者" }, lang)}>
        {t18({ zh: "歌手", en: "Speaker", ja: "話者" }, lang)}
      </label>
      <select value={String(value ?? 0)} onChange={(e) => onChange(parseInt(e.target.value, 10))}>
        {opts.map((o) => (
          <option key={o.id} value={o.id}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

/**
 * ①c speaker-blend stack — the multi-speaker replacement for SpeakerSelect on a GENUINE spk_mix
 * export (gate: voiceHasSpkMix). A list of {id, weight} rows (Rust normalizes to sum 1 and builds
 * the dense spk_mix vector); each row shows the speaker name + a weight slider reading its
 * effective blend %. SHARED by RvcNode + SoVitsNode. Modeled on the old Effects node's stack
 * pattern — derived from GRAPH params (never a useState mirror; the modal-local undo restores
 * params via setNodes WITHOUT remounting, so a mirror would re-commit undone rows) and every
 * mutation is a single onChange for clean single-step JSON-diff undo. An EMPTY stack degrades to
 * the default speaker 0 (byte-identical to picking that one speaker). Reuses the `.fx-*` CSS.
 */
export function SpeakerBlend({ model, value, onChange, lang }: {
  model: VoiceModelEntry | undefined;
  value: SpkMixEntry[];
  onChange: (rows: SpkMixEntry[]) => void;
  lang: string;
}) {
  const opts = model ? voiceSpeakerOptions(model) : [];
  // Keep only rows whose id still exists on THIS model. A same-name re-import can drop a speaker
  // whose id is still < n_speakers (emb_g width, e.g. 3 < 109) — Rust would blend that UNTRAINED
  // emb_g row (subtly wrong timbre) even though the row is never shown. SWITCH_RESETS only fires
  // on a NAME change, so we ACTIVELY prune stale ids from params here (mirrors SoVitsNode's
  // stale-flag effect). Fires once when stale rows exist, then converges (pruned value has none).
  const rows = value.filter((r) => opts.some((o) => o.id === r.id));
  useEffect(() => {
    if (rows.length !== value.length) onChange(rows);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows.length, value.length]);
  if (opts.length <= 1) return null;
  const nameOf = (id: number) => opts.find((o) => o.id === id)?.label ?? `#${id}`;
  const total = rows.reduce((s, r) => s + Math.max(0, r.weight), 0);
  const used = new Set(rows.map((r) => r.id));
  const addable = opts.filter((o) => !used.has(o.id));

  const addSpeaker = (id: number) => onChange([...rows, { id, weight: 1 }]);
  const removeSpeaker = (id: number) => onChange(rows.filter((r) => r.id !== id));
  const updateWeight = (id: number, weight: number) =>
    onChange(rows.map((r) => (r.id === id ? { ...r, weight } : r)));

  return (
    <div className="fx-stack">
      <span className="fx-stack-title" title={t18(VOICE_STRINGS.blendTip, lang)}>
        {t18(VOICE_STRINGS.blendTitle, lang)}
      </span>
      {rows.length === 0 && (
        <span className="fx-empty">{t18(VOICE_STRINGS.blendEmpty, lang)}</span>
      )}
      {rows.map((r) => (
        <div key={r.id} className="fx-entry">
          <div className="fx-entry-header">
            <span className="fx-entry-type">{nameOf(r.id)}</span>
            <button className="fx-remove" title={t18({ zh: "移除", en: "Remove", ja: "削除" }, lang)}
              onClick={() => removeSpeaker(r.id)}>x</button>
          </div>
          <ParamSlider
            label={t18(VOICE_STRINGS.blendWeight, lang)}
            min={0} max={1} step={0.01} value={r.weight}
            format={() => (total > 0 ? `${Math.round((Math.max(0, r.weight) / total) * 100)}%` : "—")}
            onChange={(v) => updateWeight(r.id, v)}
          />
        </div>
      ))}
      {addable.length > 0 && (
        <div className="fx-add-row">
          <select value="" onChange={(e) => { if (e.target.value !== "") addSpeaker(parseInt(e.target.value, 10)); }}>
            <option value="">{t18(VOICE_STRINGS.blendAdd, lang)}</option>
            {addable.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}

/** Shared NSF-HiFiGAN vocoder picker (S40) — used by the 翻唱 SoVITS node AND the ② vocal sidebar (Item-1:
 *  a fine-tuned vocoder applies to the MIDI track too — the backend `resolve_sovits_quality` already honors
 *  `vocoder_name` for both). Only meaningful on the mel→audio paths (shallow diffusion / enhancer); the
 *  CALLER gates visibility. `rowClass`/`selectClass` let each host use its own layout. A deleted/format-
 *  mismatched pick stays visible as「已缺失」/「格式不匹配」rather than auto-cleared (the async list would
 *  transiently flip a valid pick to the default — Rust errors loudly on a dangling name as the backstop). */
export function VocoderSelect({ value, lang, onChange, rowClass = "sep-param-row", selectClass, labelClass }: {
  value: string | null;
  lang: string;
  onChange: (v: string | null) => void;
  rowClass?: string;
  selectClass?: string;
  labelClass?: string;
}) {
  const vocoders = useVoiceModelStore((s) => s.models.vocoder);
  const matched = vocoders.filter(vocoderFormatMatches);
  const dangling =
    value && !matched.some((v) => v.name === value)
      ? value +
        (vocoders.some((v) => v.name === value)
          ? t18({ zh: "（格式不匹配）", en: " (format mismatch)", ja: "（フォーマット不一致）" }, lang)
          : t18({ zh: "（已缺失）", en: " (missing)", ja: "（欠落）" }, lang))
      : null;
  return (
    <div className={rowClass}>
      <label className={labelClass} title={t18({
        zh: "浅扩散/增强器使用的 NSF-HiFiGAN 声码器。可选资源管理中已微调/导入的声码器（同一歌手的声码器可被其所有 SoVITS 模型共享；仅列出频谱格式一致的）；默认 = 内置通用声码器",
        en: "NSF-HiFiGAN vocoder used by shallow diffusion / the enhancer. Pick a fine-tuned/imported vocoder from the Resource Manager (one singer's vocoder is shared by all their SoVITS models; only format-matching ones are listed); default = the built-in general vocoder",
        ja: "浅い拡散/エンハンサーが使う NSF-HiFiGAN ボコーダー。リソース管理で微調整/取り込んだボコーダーを選択可能（同じ歌手のボコーダーは全 SoVITS モデルで共有可。フォーマット一致のもののみ表示）。既定 = 内蔵汎用ボコーダー",
      }, lang)}>
        {t18({ zh: "声码器", en: "Vocoder", ja: "ボコーダー" }, lang)}
      </label>
      <select className={selectClass} value={value ?? ""} onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}>
        <option value="">{t18({ zh: "默认声码器", en: "Default vocoder", ja: "既定ボコーダー" }, lang)}</option>
        {matched.map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
        {dangling && <option value={value ?? ""}>{dangling}</option>}
      </select>
    </div>
  );
}
