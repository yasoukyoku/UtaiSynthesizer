import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../../../store/app";
import { useVoiceModelStore, voiceVersionBadge, voiceSpeakerOptions, formatSampleRateKhz, vocoderFormatMatches, } from "../../../store/voice-models";
import { ParamSlider } from "./ParamSlider";
import { t18 } from "../../../lib/models/msst-catalog";
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
};
/** S66 GPU-extraction VRAM gate: the feature's measured steady peak is ~9.4 GB (user, two
 *  runs), so enabling it needs a ≥12 GB card. nvidia-smi truth, cached module-level (one
 *  subprocess per session, not per node mount); undetermined / non-NVIDIA fails OPEN — the
 *  probe is NVIDIA-only while the extractors run on the GLOBAL device, so denying here would
 *  take the feature away from every card we simply cannot measure (DirectML included).
 *  ⛔ S115: NOT "the variant_supported convention" — S74b made that one fail-CLOSED. Why this
 *  gate points the other way is documented on `nvidia_total_vram_mb` (commands/settings.rs). */
const GPU_EXTRACT_MIN_VRAM_MB = 12_000;
let vramProbe = null;
function nvidiaVramMb() {
    vramProbe ??= invoke("get_hardware_info")
        .then((h) => h.nvidia_vram_mb)
        .catch(() => null);
    return vramProbe;
}
/** The aux-extractor device toggle row — identical on BOTH voice nodes. */
export function GpuExtractRow({ value, lang, onChange }) {
    const [vram, setVram] = useState(null);
    useEffect(() => {
        let alive = true;
        void nvidiaVramMb().then((v) => {
            if (alive)
                setVram(v);
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
    return (_jsxs("div", { className: "sep-param-row", style: blocked ? { opacity: 0.55 } : undefined, children: [_jsx("label", { title: tip, children: t18(VOICE_STRINGS.gpuExtract, lang) }), _jsx("input", { type: "checkbox", checked: value, disabled: blocked, title: tip, onChange: (e) => onChange(e.target.checked) })] }));
}
/**
 * Resolve a voice node's selected model from its GRAPH params against the installed list, and
 * keep the persisted `voiceName` / `modelPath` in sync. Derived from params, never mirrored
 * into local state — same rule as SeparationNode: the modal-local undo restores node params via
 * setNodes WITHOUT remounting, and a useState mirror would keep showing (and re-committing) the
 * undone selection.
 */
export function useVoiceModelSelection(voiceType, params, updateParams, 
/** Params force-reset on ANY real model switch (incl. the silent deleted-model fallback
 * below, which bypasses the node's own onSelect) — e.g. speaker_id, and the asset-gated
 * SoVITS toggles whose stale `true` on a model without the asset is a runtime error. */
switchResets = { speaker_id: null }) {
    const models = useVoiceModelStore((s) => s.models[voiceType]);
    useEffect(() => { void useVoiceModelStore.getState().fetchModels(); }, []);
    const selectedName = params.voiceName ?? models[0]?.name ?? "";
    const selected = models.find((m) => m.name === selectedName) ?? models[0];
    // Persist the RESOLVED selection whenever it drifts from the params: first mount (auto-pick
    // the first installed model), a deleted model falling back, or a moved models dir changing
    // `path`. A real model SWITCH also applies switchResets — a stale index from the previous
    // model could exceed the new one's n_speakers / reference assets it doesn't have.
    useEffect(() => {
        if (!selected)
            return;
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
export function VoiceModelPicker({ models, selected, lang, onSelect }) {
    const toggleModelManager = useAppStore((s) => s.toggleModelManager);
    if (models.length === 0) {
        return (_jsxs("div", { className: "voice-no-model", children: [_jsx("span", { className: "sep-no-model", children: t18({ zh: "未安装模型", en: "No models installed", ja: "モデル未インストール" }, lang) }), _jsx("button", { className: "voice-manage-btn", onClick: (e) => { e.stopPropagation(); toggleModelManager(); }, children: t18({ zh: "去资源管理导入", en: "Import in Resource Manager", ja: "リソース管理で取り込む" }, lang) })] }));
    }
    const badge = selected ? voiceVersionBadge(selected) : null;
    const speakerCount = selected ? voiceSpeakerOptions(selected).length : 0;
    return (_jsxs(_Fragment, { children: [_jsx("select", { className: "sep-model-select", value: selected?.name ?? "", onChange: (e) => {
                    const m = models.find((x) => x.name === e.target.value);
                    if (m)
                        onSelect(m);
                }, children: models.map((m) => (_jsx("option", { value: m.name, children: m.name }, m.name))) }), selected && (_jsxs("div", { className: "voice-model-meta", children: [badge && _jsx("span", { className: "ver-badge", children: badge }), _jsx("span", { children: formatSampleRateKhz(selected.sample_rate) }), selected.index_path && (_jsx("span", { className: "ver-badge", title: t18({ zh: "已附带检索/聚类文件", en: "Index/cluster asset present", ja: "インデックス/クラスタあり" }, lang), children: "IDX" })), selected.diffusion_path && (_jsx("span", { className: "ver-badge", title: t18(VOICE_STRINGS.diffBadgeTip, lang), children: "DIFF" })), speakerCount > 1 && (_jsxs("span", { children: [speakerCount, " ", t18({ zh: "歌手", en: "speakers", ja: "話者" }, lang)] }))] }))] }));
}
/** Speaker dropdown row — renders NOTHING for single-speaker models (contract: null = 0). */
export function SpeakerSelect({ model, value, onChange, lang }) {
    const opts = model ? voiceSpeakerOptions(model) : [];
    if (opts.length === 0)
        return null;
    return (_jsxs("div", { className: "sep-param-row", children: [_jsx("label", { title: t18({ zh: "多歌手模型的目标歌手", en: "Target speaker of a multi-speaker model", ja: "マルチスピーカーモデルの話者" }, lang), children: t18({ zh: "歌手", en: "Speaker", ja: "話者" }, lang) }), _jsx("select", { value: String(value ?? 0), onChange: (e) => onChange(parseInt(e.target.value, 10)), children: opts.map((o) => (_jsx("option", { value: o.id, children: o.label }, o.id))) })] }));
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
export function SpeakerBlend({ model, value, onChange, lang }) {
    const opts = model ? voiceSpeakerOptions(model) : [];
    // Keep only rows whose id still exists on THIS model. A same-name re-import can drop a speaker
    // whose id is still < n_speakers (emb_g width, e.g. 3 < 109) — Rust would blend that UNTRAINED
    // emb_g row (subtly wrong timbre) even though the row is never shown. SWITCH_RESETS only fires
    // on a NAME change, so we ACTIVELY prune stale ids from params here (mirrors SoVitsNode's
    // stale-flag effect). Fires once when stale rows exist, then converges (pruned value has none).
    const rows = value.filter((r) => opts.some((o) => o.id === r.id));
    useEffect(() => {
        if (rows.length !== value.length)
            onChange(rows);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rows.length, value.length]);
    if (opts.length <= 1)
        return null;
    const nameOf = (id) => opts.find((o) => o.id === id)?.label ?? `#${id}`;
    const total = rows.reduce((s, r) => s + Math.max(0, r.weight), 0);
    const used = new Set(rows.map((r) => r.id));
    const addable = opts.filter((o) => !used.has(o.id));
    const addSpeaker = (id) => onChange([...rows, { id, weight: 1 }]);
    const removeSpeaker = (id) => onChange(rows.filter((r) => r.id !== id));
    const updateWeight = (id, weight) => onChange(rows.map((r) => (r.id === id ? { ...r, weight } : r)));
    return (_jsxs("div", { className: "fx-stack", children: [_jsx("span", { className: "fx-stack-title", title: t18(VOICE_STRINGS.blendTip, lang), children: t18(VOICE_STRINGS.blendTitle, lang) }), rows.length === 0 && (_jsx("span", { className: "fx-empty", children: t18(VOICE_STRINGS.blendEmpty, lang) })), rows.map((r) => (_jsxs("div", { className: "fx-entry", children: [_jsxs("div", { className: "fx-entry-header", children: [_jsx("span", { className: "fx-entry-type", children: nameOf(r.id) }), _jsx("button", { className: "fx-remove", title: t18({ zh: "移除", en: "Remove", ja: "削除" }, lang), onClick: () => removeSpeaker(r.id), children: "x" })] }), _jsx(ParamSlider, { label: t18(VOICE_STRINGS.blendWeight, lang), min: 0, max: 1, step: 0.01, value: r.weight, format: () => (total > 0 ? `${Math.round((Math.max(0, r.weight) / total) * 100)}%` : "—"), onChange: (v) => updateWeight(r.id, v) })] }, r.id))), addable.length > 0 && (_jsx("div", { className: "fx-add-row", children: _jsxs("select", { value: "", onChange: (e) => { if (e.target.value !== "")
                        addSpeaker(parseInt(e.target.value, 10)); }, children: [_jsx("option", { value: "", children: t18(VOICE_STRINGS.blendAdd, lang) }), addable.map((o) => (_jsx("option", { value: o.id, children: o.label }, o.id)))] }) }))] }));
}
/** Shared NSF-HiFiGAN vocoder picker (S40) — used by the 翻唱 SoVITS node AND the ② vocal sidebar (Item-1:
 *  a fine-tuned vocoder applies to the MIDI track too — the backend `resolve_sovits_quality` already honors
 *  `vocoder_name` for both). Only meaningful on the mel→audio paths (shallow diffusion / enhancer); the
 *  CALLER gates visibility. `rowClass`/`selectClass` let each host use its own layout. A deleted/format-
 *  mismatched pick stays visible as「已缺失」/「格式不匹配」rather than auto-cleared (the async list would
 *  transiently flip a valid pick to the default — Rust errors loudly on a dangling name as the backstop). */
export function VocoderSelect({ value, lang, onChange, rowClass = "sep-param-row", selectClass, labelClass }) {
    const vocoders = useVoiceModelStore((s) => s.models.vocoder);
    const matched = vocoders.filter(vocoderFormatMatches);
    const dangling = value && !matched.some((v) => v.name === value)
        ? value +
            (vocoders.some((v) => v.name === value)
                ? t18({ zh: "（格式不匹配）", en: " (format mismatch)", ja: "（フォーマット不一致）" }, lang)
                : t18({ zh: "（已缺失）", en: " (missing)", ja: "（欠落）" }, lang))
        : null;
    return (_jsxs("div", { className: rowClass, children: [_jsx("label", { className: labelClass, title: t18({
                    zh: "浅扩散/增强器使用的 NSF-HiFiGAN 声码器。可选资源管理中已微调/导入的声码器（同一歌手的声码器可被其所有 SoVITS 模型共享；仅列出频谱格式一致的）；默认 = 内置通用声码器",
                    en: "NSF-HiFiGAN vocoder used by shallow diffusion / the enhancer. Pick a fine-tuned/imported vocoder from the Resource Manager (one singer's vocoder is shared by all their SoVITS models; only format-matching ones are listed); default = the built-in general vocoder",
                    ja: "浅い拡散/エンハンサーが使う NSF-HiFiGAN ボコーダー。リソース管理で微調整/取り込んだボコーダーを選択可能（同じ歌手のボコーダーは全 SoVITS モデルで共有可。フォーマット一致のもののみ表示）。既定 = 内蔵汎用ボコーダー",
                }, lang), children: t18({ zh: "声码器", en: "Vocoder", ja: "ボコーダー" }, lang) }), _jsxs("select", { className: selectClass, value: value ?? "", onChange: (e) => onChange(e.target.value === "" ? null : e.target.value), children: [_jsx("option", { value: "", children: t18({ zh: "默认声码器", en: "Default vocoder", ja: "既定ボコーダー" }, lang) }), matched.map((v) => _jsx("option", { value: v.name, children: v.name }, v.name)), dangling && _jsx("option", { value: value ?? "", children: dangling })] })] }));
}
