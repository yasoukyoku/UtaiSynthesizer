import { useCallback, useEffect } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { useSoundfontStore } from "../../../store/soundfont";

const SAMPLE_RATES = [44100, 48000] as const;

const selStyle: React.CSSProperties = {
  marginTop: 2,
  padding: "4px 6px",
  borderRadius: 6,
  border: "1px solid var(--border-subtle)",
  background: "var(--bg-surface)",
  color: "var(--text-primary)",
  fontSize: 12,
};

const labelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 2,
  fontSize: 12,
  color: "var(--text-secondary)",
};

export function SoundfontRenderNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const fonts = useSoundfontStore((s) => s.fonts);
  const refresh = useSoundfontStore((s) => s.refresh);

  useEffect(() => {
    if (fonts.length === 0) void refresh();
  }, [fonts.length, refresh]);

  const fontId = (params.fontId as string) ?? "";
  const presetId = (params.presetId as string) ?? "";
  const backend = (params.backend as string) ?? "builtin";
  const sampleRate = (params.sampleRate as number) ?? 44100;

  const font = fonts.find((f) => f.id === fontId);
  const presets = font?.presets ?? [];

  const handleFontChange = useCallback(
    (id: string) => {
      const next = fonts.find((f) => f.id === id);
      updateParams({ fontId: id, presetId: next?.presets[0]?.id ?? "" });
    },
    [fonts, updateParams],
  );
  const handlePresetChange = useCallback(
    (id: string) => updateParams({ presetId: id }),
    [updateParams],
  );
  const handleBackendChange = useCallback(
    (b: string) => updateParams({ backend: b }),
    [updateParams],
  );
  const handleSampleRateChange = useCallback(
    (sr: string) => updateParams({ sampleRate: Number(sr) }),
    [updateParams],
  );

  return (
    <NodeShell
      nodeId={props.id}
      label={t("workflow.nodeSoundfontRender")}
      icon="🎻"
      color="#2dd4bf"
      inputs={1}
      outputLabels={[t("workflow.sfRenderAudioOut")]}
    >
      <div
        style={{
          padding: 8,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          minWidth: 200,
        }}
      >
        {fonts.length === 0 ? (
          <div style={{ fontSize: 11, color: "var(--text-tertiary)", lineHeight: 1.4 }}>
            {t("workflow.sfRenderNoFont")}
          </div>
        ) : (
          <>
            <label style={labelStyle}>
              {t("workflow.sfRenderFont")}
              <select
                value={fontId}
                onChange={(e) => handleFontChange(e.target.value)}
                style={selStyle}
              >
                <option value="">{t("workflow.sfRenderPickFont")}</option>
                {fonts.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name} ({f.format.toUpperCase()})
                  </option>
                ))}
              </select>
            </label>

            <label style={labelStyle}>
              {t("workflow.sfRenderPreset")}
              <select
                value={presetId}
                onChange={(e) => handlePresetChange(e.target.value)}
                style={selStyle}
                disabled={presets.length === 0}
              >
                <option value="">{t("workflow.sfRenderPickPreset")}</option>
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}

        <label style={labelStyle}>
          {t("workflow.sfRenderBackend")}
          <select
            value={backend}
            onChange={(e) => handleBackendChange(e.target.value)}
            style={selStyle}
          >
            <option value="builtin">{t("workflow.sfRenderBackendBuiltin")}</option>
            <option value="fluidsynth">{t("workflow.sfRenderBackendFluidsynth")}</option>
          </select>
        </label>

        <label style={labelStyle}>
          {t("workflow.sfRenderSampleRate")}
          <select
            value={String(sampleRate)}
            onChange={(e) => handleSampleRateChange(e.target.value)}
            style={selStyle}
          >
            {SAMPLE_RATES.map((sr) => (
              <option key={sr} value={sr}>
                {sr} Hz
              </option>
            ))}
          </select>
        </label>

        <div style={{ fontSize: 11, color: "var(--text-tertiary)", lineHeight: 1.4 }}>
          {t("workflow.sfRenderDesc")}
        </div>
      </div>
    </NodeShell>
  );
}
