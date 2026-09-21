import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * 动态参数面板 - 根据选择的模型显示对应的参数控件
 * 悬停任意参数可查看：推荐值 + 效果说明 + 示例场景
 */
import { getModelParams } from "../../lib/models/song-params";
import { ParamTooltip } from "./ParamTooltip";
export function DynamicParamsPanel({ modelFamily, params, onChange, lang = "zh", }) {
    const paramConfigs = getModelParams(modelFamily);
    const wrapWithTooltip = (config, control) => (_jsx(ParamTooltip, { paramName: config.label[lang], recommendedValue: config.recommended, description: config.description?.[lang] ?? "", example: config.example?.[lang], children: control }, config.key));
    const renderParam = (config) => {
        const value = params[config.key] ?? config.default;
        const label = config.label[lang];
        switch (config.type) {
            case "number":
                return wrapWithTooltip(config, _jsxs("div", { className: "ss-field-compact", children: [_jsx("label", { className: "ss-field-label", children: label }), _jsx("input", { className: "ss-input", type: "number", min: config.min, max: config.max, step: config.step ?? 1, value: value, onChange: (e) => onChange(config.key, Number(e.target.value)) })] }));
            case "select":
                return wrapWithTooltip(config, _jsxs("div", { className: "ss-field-compact", children: [_jsx("label", { className: "ss-field-label", children: label }), _jsx("select", { className: "ss-input", value: value, onChange: (e) => onChange(config.key, e.target.value), children: config.options?.map((opt) => (_jsx("option", { value: opt.value, children: opt.label }, opt.value))) })] }));
            case "boolean":
                return wrapWithTooltip(config, _jsx("div", { className: "ss-field-compact", children: _jsxs("label", { className: "ss-field-label", children: [_jsx("input", { type: "checkbox", checked: value, onChange: (e) => onChange(config.key, e.target.checked), style: { marginRight: "8px" } }), label] }) }));
            case "text":
                return wrapWithTooltip(config, _jsxs("div", { className: "ss-field-compact", children: [_jsx("label", { className: "ss-field-label", children: label }), _jsx("input", { className: "ss-input", type: "text", value: value, onChange: (e) => onChange(config.key, e.target.value) })] }));
            default:
                return null;
        }
    };
    return (_jsx("div", { className: "ss-params-grid", children: paramConfigs.map((config) => renderParam(config)) }));
}
