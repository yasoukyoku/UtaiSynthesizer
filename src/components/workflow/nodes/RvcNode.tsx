import { useCallback } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider, formatRatio } from "./ParamSlider";
import { VoiceModelPicker, SpeakerSelect, useVoiceModelSelection, GpuExtractRow, VOICE_STRINGS } from "./VoiceModelPicker";
import { RVC_DEFAULTS } from "../../../lib/workflow/voiceDefaults";
import type { VoiceModelEntry } from "../../../store/voice-models";
import { t18 } from "../../../lib/models/msst-catalog";

export function RvcNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [params, updateParams] = useNodeParams(props);
  const { models, selected } = useVoiceModelSelection("rvc", params, updateParams);

  // Param keys ARE the wire contract keys (see voiceDefaults.ts) — absent = contract default.
  const f0Shift = (params.f0_shift as number) ?? RVC_DEFAULTS.f0_shift;
  const indexRatio = (params.index_ratio as number) ?? RVC_DEFAULTS.index_ratio;
  const protect = (params.protect as number) ?? RVC_DEFAULTS.protect;
  const noiseScale = (params.noise_scale as number) ?? RVC_DEFAULTS.noise_scale;
  const rmsMixRate = (params.rms_mix_rate as number) ?? RVC_DEFAULTS.rms_mix_rate;
  const l2Normalize = (params.l2_normalize as boolean) ?? RVC_DEFAULTS.l2_normalize;
  const gpuExtract = (params.gpu_extract as boolean) ?? RVC_DEFAULTS.gpu_extract;
  const speakerId = (params.speaker_id as number | null) ?? RVC_DEFAULTS.speaker_id;
  const hasIndex = !!selected?.index_path;

  const handleSelect = useCallback((m: VoiceModelEntry) => {
    // Model switch resets the speaker — a stale index could exceed the new model's n_speakers.
    updateParams({ voiceName: m.name, modelPath: m.path, speaker_id: null });
  }, [updateParams]);

  return (
    <NodeShell nodeId={props.id} label="RVC" icon="[R]" color="#39c5bb" inputs={1} outputs={1}>
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
            {/* hidden without the KNN index — 不能选择的控件一律不渲染（S36 用户拍板） */}
            {hasIndex && (
            <ParamSlider
              label={t18({ zh: "检索占比", en: "Index ratio", ja: "インデックス率" }, lang)}
              title={t18({ zh: "检索特征替换比例：越高越像目标音色，过高咬字可能发糊", en: "KNN index feature blend — higher = closer to the target timbre, too high can slur articulation", ja: "検索特徴の置換率 — 高いほど目標声質に近づくが、上げすぎると発音が不明瞭に" }, lang)}
              min={0} max={1} step={0.01} value={indexRatio} format={formatRatio}
              onChange={(v) => updateParams({ index_ratio: v })}
            />
            )}
            <ParamSlider
              label={t18({ zh: "清辅音保护", en: "Protect", ja: "無声子音保護" }, lang)}
              title={t18({ zh: "保护清辅音和呼吸声，防止电音撕裂；0.5 = 关闭", en: "Protects voiceless consonants & breaths from artifacts; 0.5 = off", ja: "無声子音と息を保護しアーティファクトを防ぐ。0.5 = 無効" }, lang)}
              min={0} max={0.5} step={0.01} value={protect}
              format={(v) => (v >= 0.5 ? t18(VOICE_STRINGS.off, lang) : v.toFixed(2))}
              onChange={(v) => updateParams({ protect: v })}
            />
            <ParamSlider
              label={t18(VOICE_STRINGS.noise, lang)}
              title={t18(VOICE_STRINGS.noiseTip, lang)}
              min={0} max={1} step={0.01} value={noiseScale} format={formatRatio}
              onChange={(v) => updateParams({ noise_scale: v })}
            />
            <ParamSlider
              label={t18({ zh: "响度混合", en: "RMS mix", ja: "音量ミックス" }, lang)}
              title={t18({ zh: "响度包络混合比例：0 = 完全跟随输入响度，1 = 完全用转换后响度", en: "Loudness envelope mix: 0 = follow the input's loudness, 1 = use the converted output's", ja: "ラウドネス包絡の混合比：0 = 入力の音量に追従、1 = 変換後の音量を使用" }, lang)}
              min={0} max={1} step={0.01} value={rmsMixRate} format={formatRatio}
              onChange={(v) => updateParams({ rms_mix_rate: v })}
            />
            <SpeakerSelect model={selected} value={speakerId} lang={lang}
              onChange={(id) => updateParams({ speaker_id: id })} />
            {/* retrieval-metric option — meaningless without an index, hidden with it */}
            {hasIndex && (
            <div className="sep-param-row">
              <label title={t18({ zh: "检索改按余弦（方向）选近邻，忽略特征幅度；官方按 L2 距离。索引咬字发糊时可尝试", en: "Pick index neighbors by cosine (direction) instead of the official L2 distance; try it if the index slurs articulation", ja: "インデックス近傍を公式の L2 距離でなくコサイン（方向）で選択。インデックスで発音が不明瞭なときに試してください" }, lang)}>
                {t18({ zh: "L2 归一化", en: "L2 normalize", ja: "L2 正規化" }, lang)}
              </label>
              <input type="checkbox" checked={l2Normalize}
                onChange={(e) => updateParams({ l2_normalize: e.target.checked })} />
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
