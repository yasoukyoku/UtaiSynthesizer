import { useCallback, useEffect } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider, formatRatio } from "./ParamSlider";
import { VoiceModelPicker, SpeakerSelect, useVoiceModelSelection, GpuExtractRow, VOICE_STRINGS } from "./VoiceModelPicker";
import { SOVITS_DEFAULTS, DIFFUSION_METHODS } from "../../../lib/workflow/voiceDefaults";
import { voiceHasDiffusion, voiceHasAutoF0, vocoderFormatMatches, useVoiceModelStore } from "../../../store/voice-models";
import type { VoiceModelEntry } from "../../../store/voice-models";
import { t18 } from "../../../lib/models/msst-catalog";

/** Reset on ANY model switch (dropdown or silent fallback): stale speaker ids and asset-gated
 * toggles (diffusion attachment / f0 predictor) from the previous model are runtime errors. */
const SOVITS_SWITCH_RESETS = {
  speaker_id: null,
  shallow_diffusion: false,
  only_diffusion: false,
  auto_f0: false,
} as const;

export function SoVitsNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [params, updateParams] = useNodeParams(props);
  const { models, selected } = useVoiceModelSelection("sovits", params, updateParams, SOVITS_SWITCH_RESETS);

  // Param keys ARE the wire contract keys (see voiceDefaults.ts) — absent = contract default.
  const f0Shift = (params.f0_shift as number) ?? SOVITS_DEFAULTS.f0_shift;
  const noiseScale = (params.noise_scale as number) ?? SOVITS_DEFAULTS.noise_scale;
  const clusterRatio = (params.cluster_ratio as number) ?? SOVITS_DEFAULTS.cluster_ratio;
  const loudnessEnvelope = (params.loudness_envelope as number) ?? SOVITS_DEFAULTS.loudness_envelope;
  const speakerId = (params.speaker_id as number | null) ?? SOVITS_DEFAULTS.speaker_id;
  const shallowDiffusion = (params.shallow_diffusion as boolean) ?? SOVITS_DEFAULTS.shallow_diffusion;
  const kStep = (params.k_step as number) ?? SOVITS_DEFAULTS.k_step;
  const diffusionMethod = (params.diffusion_method as string) ?? SOVITS_DEFAULTS.diffusion_method;
  const diffusionSpeedup = (params.diffusion_speedup as number) ?? SOVITS_DEFAULTS.diffusion_speedup;
  const onlyDiffusion = (params.only_diffusion as boolean) ?? SOVITS_DEFAULTS.only_diffusion;
  const secondEncoding = (params.second_encoding as boolean) ?? SOVITS_DEFAULTS.second_encoding;
  const nsfEnhance = (params.nsf_enhance as boolean) ?? SOVITS_DEFAULTS.nsf_enhance;
  const enhancerAdaptiveKey = (params.enhancer_adaptive_key as number) ?? SOVITS_DEFAULTS.enhancer_adaptive_key;
  const autoF0 = (params.auto_f0 as boolean) ?? SOVITS_DEFAULTS.auto_f0;
  const gpuExtract = (params.gpu_extract as boolean) ?? SOVITS_DEFAULTS.gpu_extract;
  const vocoderName = (params.vocoder_name as string | null) ?? SOVITS_DEFAULTS.vocoder_name;
  // S40 vocoder resources — shared across every 44.1k SoVITS model of a singer;
  // format-mismatched entries are hidden (不能选隐藏), the Rust side re-validates.
  // Dangling pick sentinel (审查修复): the select's value must ALWAYS have an
  // option or the control renders blank — deleted entry vs still-installed-but-
  // format-mismatched (a same-name re-import can change the recipe) get
  // distinct labels; NOT auto-cleared (the list loads async — clearing on a
  // transient empty list would silently flip the run to the default vocoder;
  // Rust errors loudly on a dangling name as the backstop).
  const vocoders = useVoiceModelStore((s) => s.models.vocoder);
  const vocoderMatched = vocoders.filter(vocoderFormatMatches);
  const vocoderDangling =
    vocoderName && !vocoderMatched.some((v) => v.name === vocoderName)
      ? vocoderName +
        (vocoders.some((v) => v.name === vocoderName)
          ? t18({ zh: "（格式不匹配）", en: " (format mismatch)", ja: "（フォーマット不一致）" }, lang)
          : t18({ zh: "（已缺失）", en: " (missing)", ja: "（欠落）" }, lang))
      : null;
  // Cluster/index asset presence comes from the SAME ModelEntry field RVC's index uses (Rust
  // scan() picks up any sibling .npy regardless of model type).
  const hasCluster = !!selected?.index_path;
  const hasDiffusion = voiceHasDiffusion(selected);
  const hasAutoF0 = voiceHasAutoF0(selected);
  const diffusionOn = shallowDiffusion && hasDiffusion;

  const handleSelect = useCallback((m: VoiceModelEntry) => {
    updateParams({ voiceName: m.name, modelPath: m.path, ...SOVITS_SWITCH_RESETS });
  }, [updateParams]);

  // Stale asset-gated flags are actively CLEARED, not just visually masked: switchResets
  // only fires on a NAME change, so a same-name re-import that dropped the diffusion
  // attachment / f0 predictor would otherwise leave shallow_diffusion/only_diffusion/
  // auto_f0 stranded true on the wire (disabled checkboxes can't be unchecked) and the
  // next run hard-errors in Rust. `selected` undefined = models still loading — don't
  // clear on transient state.
  useEffect(() => {
    if (!selected) return;
    const stale: Record<string, unknown> = {};
    if (!hasDiffusion && (params.shallow_diffusion || params.only_diffusion)) {
      stale.shallow_diffusion = false;
      stale.only_diffusion = false;
    }
    if (!hasAutoF0 && params.auto_f0) stale.auto_f0 = false;
    if (Object.keys(stale).length > 0) updateParams(stale);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.name, hasDiffusion, hasAutoF0, params.shallow_diffusion, params.only_diffusion, params.auto_f0]);

  return (
    <NodeShell nodeId={props.id} label="SoVITS" icon="[S]" color="#8b5cf6" inputs={1} outputs={1}>
      <div className="sep-node-body">
        <VoiceModelPicker models={models} selected={selected} lang={lang} onSelect={handleSelect} />
        {models.length > 0 && (
          <div className="sep-params">
            <ParamSlider
              label={t18(VOICE_STRINGS.f0Shift, lang)}
              title={t18(VOICE_STRINGS.f0ShiftTip, lang)}
              min={-24} max={24} step={1} value={f0Shift}
              onChange={(v) => updateParams({ f0_shift: v })}
            />
            <ParamSlider
              label={t18(VOICE_STRINGS.noise, lang)}
              title={t18(VOICE_STRINGS.noiseTip, lang)}
              min={0} max={1} step={0.01} value={noiseScale} format={formatRatio}
              onChange={(v) => updateParams({ noise_scale: v })}
            />
            {/* hidden without the cluster asset — 不能选择的控件一律不渲染（S36 用户拍板） */}
            {hasCluster && (
            <ParamSlider
              label={t18({ zh: "聚类占比", en: "Cluster ratio", ja: "クラスタ率" }, lang)}
              title={t18({ zh: "聚类/特征检索混合比例：越高越像目标音色，咬字可能变糊；0 = 关闭", en: "Cluster / feature-index blend — higher = closer to the target timbre, may slur articulation; 0 = off", ja: "クラスタ/特徴検索の混合比 — 高いほど目標声質に近づくが発音が不明瞭になることも。0 = 無効" }, lang)}
              min={0} max={1} step={0.01} value={clusterRatio} format={formatRatio}
              onChange={(v) => updateParams({ cluster_ratio: v })}
            />
            )}
            <ParamSlider
              label={t18({ zh: "响度包络", en: "Loudness env", ja: "音量包絡" }, lang)}
              title={t18({ zh: "用输入响度包络替换输出的混合比例，1 = 不替换（关）", en: "Input-loudness-envelope replacement mix; 1 = no replacement (off)", ja: "入力のラウドネス包絡で出力を置き換える比率。1 = 置き換えなし（オフ）" }, lang)}
              min={0} max={1} step={0.01} value={loudnessEnvelope}
              format={(v) => (v >= 1 ? t18(VOICE_STRINGS.off, lang) : v.toFixed(2))}
              onChange={(v) => updateParams({ loudness_envelope: v })}
            />
            <SpeakerSelect model={selected} value={speakerId} lang={lang}
              onChange={(id) => updateParams({ speaker_id: id })} />

            {/* ---- 浅扩散 group (S36 quality path) — HIDDEN entirely without the
                 .diffusion/ attachment (user call: disabled-grey rows read as clutter) ---- */}
            {hasDiffusion && (
            <div className="sep-param-row">
              <label title={t18({ zh: "VITS 输出经扩散模型精修：可缓解电音，细节更自然；开启时 NSF 增强器不可用（原版互斥）", en: "Refines the VITS output with the attached diffusion model — reduces artifacts, adds detail; the NSF enhancer is unavailable while on (original mutual exclusion)", ja: "VITS 出力を拡散モデルでリファイン — ノイズ軽減・質感向上。オン中は NSF エンハンサー不可（原版準拠）" }, lang)}>
                {t18({ zh: "浅扩散", en: "Shallow diffusion", ja: "浅い拡散" }, lang)}
              </label>
              <input type="checkbox" checked={diffusionOn}
                onChange={(e) => updateParams(
                  // Unchecking must ALSO clear only_diffusion: its checkbox lives inside
                  // this group and unmounts, so a stranded true would silently keep the
                  // run in diffusion-only mode with all diffusion UI hidden.
                  e.target.checked
                    ? { shallow_diffusion: true }
                    : { shallow_diffusion: false, only_diffusion: false },
                )} />
            </div>
            )}
            {diffusionOn && (
              <>
                {/* k_step 在仅扩散下被原版语义忽略 → 整行隐藏而非灰显 */}
                {!onlyDiffusion && (
                <ParamSlider
                  label={t18({ zh: "扩散步数", en: "k_step", ja: "拡散ステップ" }, lang)}
                  title={t18({ zh: "越大越接近纯扩散结果；上限受扩散模型 k_step_max 限制（超限会报错）", en: "Higher = closer to the pure-diffusion result; capped by the diffusion model's k_step_max (errors past it)", ja: "大きいほど拡散寄りの結果に。上限は拡散モデルの k_step_max（超えるとエラー）" }, lang)}
                  min={10} max={1000} step={10} value={kStep}
                  onChange={(v) => updateParams({ k_step: v })}
                />
                )}
                <div className="sep-param-row">
                  <label title={t18({ zh: "采样算法：dpm-solver++ 为原版默认；naive = 原始 DDPM 全步采样（慢）", en: "Sampler: dpm-solver++ is the original default; naive = plain full-step DDPM (slow)", ja: "サンプラー：dpm-solver++ が原版デフォルト。naive はフルステップ DDPM（低速）" }, lang)}>
                    {t18({ zh: "采样器", en: "Sampler", ja: "サンプラー" }, lang)}
                  </label>
                  <select value={diffusionMethod}
                    onChange={(e) => updateParams({ diffusion_method: e.target.value })}>
                    {DIFFUSION_METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
                <ParamSlider
                  label={t18({ zh: "加速倍数", en: "Speedup", ja: "加速倍率" }, lang)}
                  title={t18({ zh: "跳步加速：实际采样步 ≈ 扩散步数 ÷ 加速倍数；1 = 不加速（逐步采样）", en: "Step skipping: solver steps ≈ k_step ÷ speedup; 1 = no acceleration", ja: "ステップスキップ：実サンプル数 ≈ k_step ÷ 倍率。1 = 加速なし" }, lang)}
                  min={1} max={100} step={1} value={diffusionSpeedup}
                  onChange={(v) => updateParams({ diffusion_speedup: v })}
                />
                {/* second_encoding 只在浅扩散路径有意义（原版 guard）→ 仅扩散时隐藏 */}
                {!onlyDiffusion && (
                <div className="sep-param-row">
                  <label title={t18({ zh: "浅扩散前对 VITS 输出重提特征（原版「玄学选项」：时好时坏）", en: "Re-extract ContentVec from the VITS output before diffusing (original: sometimes better, sometimes worse)", ja: "拡散前に VITS 出力から特徴を再抽出（原版いわく「オカルト設定」）" }, lang)}>
                    {t18({ zh: "二次编码", en: "2nd encoding", ja: "二次エンコード" }, lang)}
                  </label>
                  <input type="checkbox" checked={secondEncoding}
                    onChange={(e) => updateParams({ second_encoding: e.target.checked })} />
                </div>
                )}
                <div className="sep-param-row">
                  <label title={t18({ zh: "跳过 VITS，纯扩散生成整段（需要全步数训练的扩散模型；忽略扩散步数）", en: "Skip VITS entirely — pure diffusion generation (needs a full-depth diffusion model; k_step is ignored)", ja: "VITS をスキップし拡散のみで生成（フルステップ学習の拡散モデルが必要。k_step は無視）" }, lang)}>
                    {t18({ zh: "仅扩散", en: "Diffusion only", ja: "拡散のみ" }, lang)}
                  </label>
                  <input type="checkbox" checked={onlyDiffusion}
                    onChange={(e) => updateParams({ only_diffusion: e.target.checked })} />
                </div>
              </>
            )}

            {/* ---- NSF-HiFiGAN enhancer — HIDDEN while any diffusion mode is on
                 (original mutual exclusion; disabled-grey read as clutter) ---- */}
            {!diffusionOn && (
              <>
                <div className="sep-param-row">
                  <label title={t18({ zh: "NSF-HiFiGAN 增强器：对训练不足的模型有音质增强，训练充分的模型可能有反效果（原版说明）；需要 aux/nsf_hifigan.onnx", en: "NSF-HiFiGAN enhancer — helps under-trained models, may hurt well-trained ones (original note); needs aux/nsf_hifigan.onnx", ja: "NSF-HiFiGAN エンハンサー — 学習不足のモデルに有効、十分学習済みだと逆効果の場合も（原版注記）。aux/nsf_hifigan.onnx が必要" }, lang)}>
                    {t18({ zh: "NSF增强器", en: "NSF enhancer", ja: "NSFエンハンサー" }, lang)}
                  </label>
                  <input type="checkbox" checked={nsfEnhance}
                    onChange={(e) => updateParams({ nsf_enhance: e.target.checked })} />
                </div>
                {nsfEnhance && (
                  <ParamSlider
                    label={t18({ zh: "音域适应", en: "Adaptive key", ja: "音域適応" }, lang)}
                    title={t18({ zh: "使增强器适应更高的音域（单位：半音）", en: "Adapts the enhancer to a higher range (semitones)", ja: "エンハンサーを高い音域に適応させる（半音単位）" }, lang)}
                    min={-12} max={12} step={1} value={enhancerAdaptiveKey}
                    onChange={(v) => updateParams({ enhancer_adaptive_key: v })}
                  />
                )}
              </>
            )}

            {/* ---- S40 vocoder pick — only meaningful on the two mel→audio paths
                 (shallow diffusion / enhancer), hidden otherwise. A deleted pick
                 stays visible as「已缺失」instead of being auto-cleared: the
                 vocoder list loads async, and clearing on a transient empty list
                 would silently flip the run to the default vocoder — the Rust
                 side errors loudly on a dangling name as the backstop. ---- */}
            {(diffusionOn || nsfEnhance) && (
              <div className="sep-param-row">
                <label title={t18({
                  zh: "浅扩散/增强器使用的 NSF-HiFiGAN 声码器。可选资源管理中已微调/导入的声码器（同一歌手的声码器可被其所有 SoVITS 模型共享；仅列出频谱格式一致的）；默认 = 内置通用声码器",
                  en: "NSF-HiFiGAN vocoder used by shallow diffusion / the enhancer. Pick a fine-tuned/imported vocoder from the Resource Manager (one singer's vocoder is shared by all their SoVITS models; only format-matching ones are listed); default = the built-in general vocoder",
                  ja: "浅い拡散/エンハンサーが使う NSF-HiFiGAN ボコーダー。リソース管理で微調整/取り込んだボコーダーを選択可能（同じ歌手のボコーダーは全 SoVITS モデルで共有可。フォーマット一致のもののみ表示）。既定 = 内蔵汎用ボコーダー",
                }, lang)}>
                  {t18({ zh: "声码器", en: "Vocoder", ja: "ボコーダー" }, lang)}
                </label>
                {/* native select — the node's own control style (sampler/
                    speaker are native too); the training-page Dropdown reads
                    oversized inside a node (user call) */}
                <select
                  value={vocoderName ?? ""}
                  onChange={(e) =>
                    updateParams({ vocoder_name: e.target.value === "" ? null : e.target.value })
                  }
                >
                  <option value="">
                    {t18({ zh: "默认声码器", en: "Default vocoder", ja: "既定ボコーダー" }, lang)}
                  </option>
                  {vocoderMatched.map((v) => (
                    <option key={v.name} value={v.name}>{v.name}</option>
                  ))}
                  {vocoderDangling && (
                    <option value={vocoderName ?? ""}>{vocoderDangling}</option>
                  )}
                </select>
              </div>
            )}

            {/* ---- auto-f0 — HIDDEN unless the export carries the f0 predictor ---- */}
            {hasAutoF0 && (
            <div className="sep-param-row">
              <label title={t18({ zh: "由模型自动预测音高（语音转换用）——转换歌声会严重跑调，且变调基本失效（原版警告）", en: "Model-predicted f0 (for speech conversion) — singing will drift badly and f0 shift is mostly neutralized (original warning)", ja: "モデルによる自動ピッチ予測（話し声向け）— 歌声では大きく音痴になり、キー変更もほぼ無効（原版警告）" }, lang)}>
                {t18({ zh: "自动音高预测", en: "Auto f0", ja: "自動ピッチ予測" }, lang)}
              </label>
              <input type="checkbox" checked={autoF0}
                onChange={(e) => updateParams({ auto_f0: e.target.checked })} />
            </div>
            )}

            <GpuExtractRow value={gpuExtract} lang={lang}
              onChange={(v) => updateParams({ gpu_extract: v })} />
          </div>
        )}
      </div>
    </NodeShell>
  );
}
