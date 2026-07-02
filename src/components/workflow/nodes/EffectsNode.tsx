import { useCallback } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { t18, type I18nText } from "../../../lib/models/msst-catalog";

type EffectType = "pitchShift" | "formantShift" | "enhance";

interface EffectEntry {
  id: string;
  type: EffectType;
  params: Record<string, unknown>;
}

const EFFECT_DEFS: Record<EffectType, { label: I18nText; defaults: Record<string, unknown> }> = {
  pitchShift: {
    label: { zh: "变调", en: "Pitch Shift", ja: "ピッチシフト" },
    defaults: { semitones: 0, vocoder: "world" },
  },
  formantShift: {
    label: { zh: "共振峰", en: "Formant Shift", ja: "フォルマントシフト" },
    defaults: { ratio: 1.0, vocoder: "world" },
  },
  enhance: {
    label: { zh: "增强", en: "Enhance", ja: "エンハンス" },
    defaults: {},
  },
};

const EFFECT_STRINGS: Record<string, I18nText> = {
  add: { zh: "+ 添加效果", en: "+ Add Effect", ja: "+ エフェクト追加" },
  empty: { zh: "无效果 — 音频直通", en: "No effects — audio passthrough", ja: "エフェクトなし — パススルー" },
  semitones: { zh: "半音", en: "Semitones", ja: "半音" },
  ratio: { zh: "比例", en: "Ratio", ja: "比率" },
  vocoder: { zh: "声码器", en: "Vocoder", ja: "ボコーダー" },
  enhanceDesc: { zh: "NSF-HiFiGAN 后处理", en: "NSF-HiFiGAN post-process", ja: "NSF-HiFiGAN 後処理" },
};

export function EffectsNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [nodeParams, updateParams] = useNodeParams(props);
  // Derived from the GRAPH params, never mirrored into local state: the modal-local undo restores node
  // params via setNodes WITHOUT remounting this component — a useState mirror kept rendering the undone
  // effect stack, and the next add/remove/update silently re-committed it over the restored params.
  const effects = (nodeParams.effects as EffectEntry[]) ?? [];
  const l = (key: string) => {
    const s = EFFECT_STRINGS[key];
    return s ? t18(s, lang) : key;
  };

  const syncToNode = useCallback((newEffects: EffectEntry[]) => {
    updateParams({ effects: newEffects });
  }, [updateParams]);

  const addEffect = useCallback((type: EffectType) => {
    const entry: EffectEntry = {
      id: crypto.randomUUID().slice(0, 8),
      type,
      params: { ...EFFECT_DEFS[type].defaults },
    };
    syncToNode([...effects, entry]);
  }, [effects, syncToNode]);

  const removeEffect = useCallback((id: string) => {
    syncToNode(effects.filter((e) => e.id !== id));
  }, [effects, syncToNode]);

  const updateEffect = useCallback((id: string, updates: Record<string, unknown>) => {
    syncToNode(effects.map((e) =>
      e.id === id ? { ...e, params: { ...e.params, ...updates } } : e,
    ));
  }, [effects, syncToNode]);

  return (
    <NodeShell nodeId={props.id} label={t18({ zh: "效果器", en: "Effects", ja: "エフェクト" }, lang)} icon="FX" color="#fbbf24" inputs={1} outputs={1}>
      <div className="fx-stack">
        {effects.length === 0 && (
          <span className="fx-empty">{l("empty")}</span>
        )}
        {effects.map((fx) => (
          <div key={fx.id} className="fx-entry">
            <div className="fx-entry-header">
              <span className="fx-entry-type">{t18(EFFECT_DEFS[fx.type].label, lang)}</span>
              <button className="fx-remove" onClick={() => removeEffect(fx.id)}>x</button>
            </div>
            <EffectParams fx={fx} onChange={(u) => updateEffect(fx.id, u)} l={l} />
          </div>
        ))}
        <div className="fx-add-row">
          <select
            value=""
            onChange={(e) => { if (e.target.value) addEffect(e.target.value as EffectType); }}
          >
            <option value="">{l("add")}</option>
            {(Object.keys(EFFECT_DEFS) as EffectType[]).map((tp) => (
              <option key={tp} value={tp}>{t18(EFFECT_DEFS[tp].label, lang)}</option>
            ))}
          </select>
        </div>
      </div>
    </NodeShell>
  );
}

/** WORLD / NSF-HiFiGAN vocoder select — was pasted in both the pitch- and formant-shift branches. */
function VocoderSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="world">WORLD</option>
      <option value="nsf">NSF-HiFiGAN</option>
    </select>
  );
}

function EffectParams({ fx, onChange, l }: { fx: EffectEntry; onChange: (u: Record<string, unknown>) => void; l: (k: string) => string }) {
  if (fx.type === "pitchShift") {
    return (
      <div className="fx-params">
        <div className="sep-param-row">
          <label>{l("semitones")}</label>
          <input type="number" min={-24} max={24} step={0.5}
            value={(fx.params.semitones as number) ?? 0}
            onChange={(e) => onChange({ semitones: parseFloat(e.target.value) || 0 })} />
        </div>
        <div className="sep-param-row">
          <label>{l("vocoder")}</label>
          <VocoderSelect value={(fx.params.vocoder as string) ?? "world"} onChange={(v) => onChange({ vocoder: v })} />
        </div>
      </div>
    );
  }
  if (fx.type === "formantShift") {
    return (
      <div className="fx-params">
        <div className="sep-param-row">
          <label>{l("ratio")}</label>
          <input type="number" min={0.5} max={2.0} step={0.05}
            value={(fx.params.ratio as number) ?? 1.0}
            onChange={(e) => onChange({ ratio: parseFloat(e.target.value) || 1.0 })} />
        </div>
        <div className="sep-param-row">
          <label>{l("vocoder")}</label>
          <VocoderSelect value={(fx.params.vocoder as string) ?? "world"} onChange={(v) => onChange({ vocoder: v })} />
        </div>
      </div>
    );
  }
  if (fx.type === "enhance") {
    return <span className="fx-desc">{l("enhanceDesc")}</span>;
  }
  return null;
}
