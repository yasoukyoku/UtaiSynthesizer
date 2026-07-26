import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
import { t18 } from "../../../lib/models/msst-catalog";

/** The Signalsmith node (spectral pitch-shift + engine-native formant controls,
 *  tonality-aware, polyphonic-safe). Built for INSTRUMENTAL transposition: the voice nodes
 *  already transpose model-side (f0_shift), but nothing could shift the accompaniment until
 *  this node. Node TYPE stays "transpose" (persisted in projects); only the display name is
 *  Signalsmith (S82 — it stopped being just a transpose).
 *  Formant controls: 「共振峰跟随」 0..1 (the same κ vocabulary as the range-extension
 *  slider) — 1 = classic full-spectrum shift (pre-S82 behavior, bit-path identical),
 *  0 = envelope pinned to the source (big shifts stop sounding cartoonish/muddy);
 *  「共振峰偏移」 shifts timbre on top (works even at 0 semitones).
 *  All-defaults = exact passthrough (the engine skips the invoke entirely). */
export function TransposeNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [params, updateParams] = useNodeParams(props);

  const semitones = (params.semitones as number) ?? 0;
  // legacy same-session shape: preserveFormants=true reads as follow 0
  const formantFollow = typeof params.formantFollow === "number"
    ? (params.formantFollow as number)
    : params.preserveFormants === true ? 0 : 1;
  const formantOffset = (params.formantOffset as number) ?? 0;

  return (
    <NodeShell nodeId={props.id} label="Signalsmith" icon="[S]" color="#fbbf24" inputs={1} outputs={1}>
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
          <ParamSlider
            label={t18({ zh: "共振峰跟随", en: "Formant follow", ja: "フォルマント追従" }, lang)}
            title={t18({
              zh: "共振峰（音色骨架）跟随移调的比例：1＝传统整谱移调（旧行为），0＝钉在原位（大幅移调不再卡通感/发闷）。仅在半音≠0 时有影响",
              en: "How much the formants (timbral skeleton) follow the transpose: 1 = classic full-spectrum shift (old behavior), 0 = pinned to the source (big shifts stop sounding cartoonish or muddy). Only matters when semitones ≠ 0",
              ja: "フォルマント（音色の骨格）が移調に追従する割合：1＝従来のフルスペクトル移調（旧動作）、0＝元の位置に固定（大きな移調でもカートゥーン的/こもった音になりません）。半音≠0 のときのみ影響します",
            }, lang)}
            min={0} max={1} step={0.01} value={formantFollow}
            format={(v) => v.toFixed(2)}
            onChange={(v) => updateParams({ formantFollow: v })}
          />
          <ParamSlider
            label={t18({ zh: "共振峰偏移", en: "Formant shift", ja: "フォルマントシフト" }, lang)}
            title={t18({
              zh: "在跟随基准之上再偏移共振峰（半音）：正 = 更亮/更年轻，负 = 更暗/更浑厚。音高不变，0 半音时也可单独用来改音色",
              en: "Shift the formants on top of the follow base (semitones): higher = brighter/younger, lower = darker/fuller. Pitch is untouched — usable as a pure timbre control even at 0 semitones",
              ja: "追従基準に加えてフォルマントをシフト（半音）：高い = 明るい/若い、低い = 暗い/太い。音高は不変。0 半音でも音色調整として単独で使えます",
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
