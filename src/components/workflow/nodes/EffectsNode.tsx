import { useState, useCallback } from "react";
import { type NodeProps, useReactFlow } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useTranslation } from "react-i18next";

type EffectType = "pitchShift" | "formantShift" | "enhance";

interface EffectEntry {
  id: string;
  type: EffectType;
  params: Record<string, unknown>;
}

const EFFECT_DEFS: Record<EffectType, { label: Record<string, string>; defaults: Record<string, unknown> }> = {
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

export function EffectsNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const { setNodes } = useReactFlow();
  const nodeParams = (props.data?.params as Record<string, unknown>) ?? {};
  const [effects, setEffects] = useState<EffectEntry[]>(
    (nodeParams.effects as EffectEntry[]) ?? [],
  );

  const syncToNode = useCallback((newEffects: EffectEntry[]) => {
    setEffects(newEffects);
    setNodes((nds) => nds.map((n) =>
      n.id === props.id
        ? { ...n, data: { ...n.data, params: { ...nodeParams, effects: newEffects } } }
        : n,
    ));
  }, [props.id, nodeParams, setNodes]);

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

  const l = (key: string) => {
    const map: Record<string, Record<string, string>> = {
      add: { zh: "+ 添加效果", en: "+ Add Effect", ja: "+ エフェクト追加" },
      empty: { zh: "无效果 — 音频直通", en: "No effects — audio passthrough", ja: "エフェクトなし — パススルー" },
      semitones: { zh: "半音", en: "Semitones", ja: "半音" },
      ratio: { zh: "比例", en: "Ratio", ja: "比率" },
      vocoder: { zh: "声码器", en: "Vocoder", ja: "ボコーダー" },
      enhanceDesc: { zh: "NSF-HiFiGAN 后处理", en: "NSF-HiFiGAN post-process", ja: "NSF-HiFiGAN 後処理" },
    };
    return map[key]?.[lang] ?? map[key]?.en ?? key;
  };

  return (
    <NodeShell nodeId={props.id} label={lang === "zh" ? "效果器" : "Effects"} icon="FX" color="#fbbf24" inputs={1} outputs={1}>
      <div className="fx-stack">
        {effects.length === 0 && (
          <span className="fx-empty">{l("empty")}</span>
        )}
        {effects.map((fx) => (
          <div key={fx.id} className="fx-entry">
            <div className="fx-entry-header">
              <span className="fx-entry-type">{EFFECT_DEFS[fx.type].label[lang] ?? EFFECT_DEFS[fx.type].label.en}</span>
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
            {(Object.keys(EFFECT_DEFS) as EffectType[]).map((t) => (
              <option key={t} value={t}>{EFFECT_DEFS[t].label[lang] ?? EFFECT_DEFS[t].label.en}</option>
            ))}
          </select>
        </div>
      </div>
    </NodeShell>
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
          <select value={(fx.params.vocoder as string) ?? "world"}
            onChange={(e) => onChange({ vocoder: e.target.value })}>
            <option value="world">WORLD</option>
            <option value="nsf">NSF-HiFiGAN</option>
          </select>
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
          <select value={(fx.params.vocoder as string) ?? "world"}
            onChange={(e) => onChange({ vocoder: e.target.value })}>
            <option value="world">WORLD</option>
            <option value="nsf">NSF-HiFiGAN</option>
          </select>
        </div>
      </div>
    );
  }
  if (fx.type === "enhance") {
    return <span className="fx-desc">{l("enhanceDesc")}</span>;
  }
  return null;
}
