import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
import { t18 } from "../../../lib/models/msst-catalog";

/** Fidelity transpose (spectral pitch-shift, Signalsmith Stretch — tonality-aware,
 *  polyphonic-safe). Built for INSTRUMENTAL transposition: the voice nodes already transpose
 *  model-side (f0_shift), but nothing could shift the accompaniment until this node.
 *  S82 formant controls (engine-native, Signalsmith 1.3.2): "keep formants" pins the spectral
 *  envelope to the source (big shifts stop sounding cartoonish/muddy; OFF = the classic
 *  full-spectrum shift = the node's pre-S82 behavior, bit-path identical), and the formant
 *  slider shifts timbre on top (works even at 0 semitones).
 *  All-defaults = exact passthrough (the engine skips the invoke entirely). */
export function TransposeNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [params, updateParams] = useNodeParams(props);

  const semitones = (params.semitones as number) ?? 0;
  const preserveFormants = params.preserveFormants === true;
  const formantOffset = (params.formantOffset as number) ?? 0;

  return (
    <NodeShell nodeId={props.id} label={t18({ zh: "移调", en: "Transpose", ja: "移調" }, lang)} icon="[T]" color="#fbbf24" inputs={1} outputs={1}>
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSlider
            label={t18({ zh: "半音", en: "Semitones", ja: "半音" }, lang)}
            title={t18({
              zh: "频谱域保真移调（保持时长与高频自然感，适合给伴奏整体移调）；0 = 原样直通",
              en: "Spectral-domain fidelity pitch shift (length & natural highs preserved — made for transposing accompaniment); 0 = exact passthrough",
              ja: "スペクトル領域の高品質ピッチシフト（長さと高域の自然さを保持、伴奏の移調向け）。0 = そのまま通過",
            }, lang)}
            min={-24} max={24} step={1} value={semitones}
            format={(v) => (v > 0 ? `+${v}` : `${v}`)}
            onChange={(v) => updateParams({ semitones: v })}
          />
          <div className="sep-param-row">
            <label title={t18({
              zh: "移调时把共振峰（音色骨架）钉在原位：大幅移调不再卡通感/发闷。关闭 = 传统整谱移调（共振峰跟着音高走）",
              en: "Pin the formants (timbral skeleton) to the source while transposing: big shifts stop sounding cartoonish or muddy. Off = classic full-spectrum shift (formants follow the pitch)",
              ja: "移調時にフォルマント（音色の骨格）を元の位置に固定：大きな移調でもカートゥーン的/こもった音になりません。オフ = 従来のフルスペクトル移調（フォルマントは音高に追従）",
            }, lang)}>
              {t18({ zh: "保持共振峰", en: "Keep formants", ja: "フォルマント維持" }, lang)}
            </label>
            <input type="checkbox" checked={preserveFormants}
              onChange={(e) => updateParams({ preserveFormants: e.target.checked })} />
          </div>
          <ParamSlider
            label={t18({ zh: "共振峰偏移", en: "Formant shift", ja: "フォルマントシフト" }, lang)}
            title={t18({
              zh: "在上面基准之上再偏移共振峰（半音）：正 = 更亮/更年轻，负 = 更暗/更浑厚。音高不变，0 半音时也可单独用来改音色",
              en: "Shift the formants on top of the base above (semitones): higher = brighter/younger, lower = darker/fuller. Pitch is untouched — usable as a pure timbre control even at 0 semitones",
              ja: "上の基準に加えてフォルマントをシフト（半音）：高い = 明るい/若い、低い = 暗い/太い。音高は不変。0 半音でも音色調整として単独で使えます",
            }, lang)}
            min={-24} max={24} step={1} value={formantOffset}
            format={(v) => (v > 0 ? `+${v}` : `${v}`)}
            onChange={(v) => updateParams({ formantOffset: v })}
          />
        </div>
      </div>
    </NodeShell>
  );
}
