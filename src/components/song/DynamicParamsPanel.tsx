/**
 * 动态参数面板 - 根据选择的模型显示对应的参数控件
 * 悬停任意参数可查看：推荐值 + 效果说明 + 示例场景
 */
import { getModelParams, type ParamConfig } from "../../lib/models/song-params";
import type { SongModelFamily } from "../../lib/models/song-catalog";
import { ParamTooltip } from "./ParamTooltip";

interface DynamicParamsPanelProps {
  modelFamily: SongModelFamily;
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
  lang?: "zh" | "en";
}

export function DynamicParamsPanel({
  modelFamily,
  params,
  onChange,
  lang = "zh",
}: DynamicParamsPanelProps) {
  const paramConfigs = getModelParams(modelFamily);
  const sectionTitle = modelFamily === "yue2" ? "YuE2 专属参数" : modelFamily === "acestep" ? "ACE-Step v1.5 专属参数" : "HeartMuLa 专属参数";
  const isAce = modelFamily === "acestep";
  const getRange = (config: ParamConfig) => {
    if (isAce && config.key === "inference_steps") {
      return { min: 4, max: 8 };
    }
    return { min: config.min, max: config.max };
  };

  const wrapWithTooltip = (config: ParamConfig, control: React.ReactNode) => (
    <ParamTooltip
      key={config.key}
      paramName={config.label[lang]}
      recommendedValue={config.recommended}
      description={config.description?.[lang] ?? ""}
      example={config.example?.[lang]}
    >
      {control}
    </ParamTooltip>
  );

  const renderParam = (config: ParamConfig) => {
    const value = params[config.key] ?? config.default;
    const label = config.label[lang];

    const range = getRange(config);

    switch (config.type) {
      case "number":
        return wrapWithTooltip(
          config,
          <div className="ss-field-compact">
            <label className="ss-field-label">{label}</label>
            <input
              className="ss-input"
              type="number"
              min={range.min}
              max={range.max}
              step={config.step ?? 1}
              value={value}
              onChange={(e) => onChange(config.key, Number(e.target.value))}
            />
          </div>,
        );

      case "select":
        return wrapWithTooltip(
          config,
          <div className="ss-field-compact">
            <label className="ss-field-label">{label}</label>
            <select
              className="ss-input"
              value={value}
              onChange={(e) => onChange(config.key, e.target.value)}
            >
              {config.options?.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>,
        );

      case "boolean":
        return wrapWithTooltip(
          config,
          <div className="ss-field-compact">
            <label className="ss-field-label">
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => onChange(config.key, e.target.checked)}
                style={{ marginRight: "8px" }}
              />
              {label}
            </label>
          </div>,
        );

      case "text":
        return wrapWithTooltip(
          config,
          <div className="ss-field-compact">
            <label className="ss-field-label">{label}</label>
            <input
              className="ss-input"
              type="text"
              value={value}
              onChange={(e) => onChange(config.key, e.target.value)}
            />
          </div>,
        );

      default:
        return null;
    }
  };

  return (
    <section className="ss-params-section">
      <div className="ss-params-section-title">{sectionTitle}</div>
      <div className="ss-params-grid">
        {paramConfigs.map((config) => renderParam(config))}
      </div>
    </section>
  );
}
