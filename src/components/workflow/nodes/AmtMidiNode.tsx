import { useCallback, useState } from "react";
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { useTranslation } from "react-i18next";
import { MUSCRIPTOR_INSTRUMENTS, MUSCRIPTOR_ZH_LABELS } from "../../../lib/models/muscriptor-instruments";

type ModeValue =
  | "smart"
  | "vocal_split"
  | "six_stem_split"
  | "piano_transkun"
  | "piano_transkun_v2_aug"
  | "piano_aria_amt"
  | "piano_bytedance_pedal";

interface ModeDef {
  value: ModeValue;
  labelKey: string;
  multi: boolean;
}

const MIDI_MODES: ModeDef[] = [
  { value: "smart", labelKey: "amt.modeSmart", multi: true },
  { value: "vocal_split", labelKey: "amt.modeVocalSplit", multi: true },
  { value: "six_stem_split", labelKey: "amt.modeSixStem", multi: true },
  { value: "piano_transkun", labelKey: "amt.modePiano", multi: false },
  { value: "piano_transkun_v2_aug", labelKey: "amt.modePianoAug", multi: false },
  { value: "piano_aria_amt", labelKey: "amt.modeAria", multi: false },
  { value: "piano_bytedance_pedal", labelKey: "amt.modeBytedance", multi: false },
];

const BACKENDS = [
  { value: "yourmt3", labelKey: "amt.backendYourmt3" },
  { value: "miros", labelKey: "amt.backendMiros" },
  { value: "muscriptor", labelKey: "amt.backendMuscriptor" },
] as const;

const QUANTIZE_GRIDS = ["off", "1/4", "1/8", "1/16", "1/32", "1/64"] as const;

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

export function AmtMidiNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);

  const midiMode = (params.midiMode as ModeValue) ?? "smart";
  const useGpu = (params.useGpu as boolean) ?? true;
  const backend = (params.backend as string) ?? "yourmt3";
  const quantizeGrid = (params.quantizeGrid as string) ?? "off";
  // 轨道布局：multi_track=全轨道(按乐器拆分为多条)，single_track=单轨道(合并)
  const midiTrackMode = (params.midiTrackMode as string) ?? "multi_track";
  // MuScriptor 专属：输出乐器(勾选) + 分段衔接方式
  const muscriptorInstruments = (params.muscriptorInstruments as string[]) ?? [];
  // Python contract: only "official" | "telknet" (never the legacy "sustain_connect").
  const muscriptorChain = (params.muscriptorChain as string) ?? "official";

  const handleModeChange = useCallback(
    (mode: string) => updateParams({ midiMode: mode }),
    [updateParams],
  );
  const handleBackendChange = useCallback(
    (b: string) => updateParams({ backend: b }),
    [updateParams],
  );
  const handleQuantizeChange = useCallback(
    (g: string) => updateParams({ quantizeGrid: g }),
    [updateParams],
  );
  const handleGpuToggle = useCallback(
    () => updateParams({ useGpu: !useGpu }),
    [updateParams, useGpu],
  );
  const handleTrackModeChange = useCallback(
    (m: string) => updateParams({ midiTrackMode: m }),
    [updateParams],
  );
  const handleChainChange = useCallback(
    (c: string) => updateParams({ muscriptorChain: c }),
    [updateParams],
  );
  const toggleInstrument = useCallback(
    (id: string) => {
      const next = muscriptorInstruments.includes(id)
        ? muscriptorInstruments.filter((x) => x !== id)
        : [...muscriptorInstruments, id];
      updateParams({ muscriptorInstruments: next });
    },
    [updateParams, muscriptorInstruments],
  );
  const selectAllInstruments = useCallback(
    () => updateParams({ muscriptorInstruments: [...MUSCRIPTOR_INSTRUMENTS] }),
    [updateParams],
  );

  const currentMode = MIDI_MODES.find((m) => m.value === midiMode) ?? MIDI_MODES[0]!;
  const showBackend = currentMode.multi;
  const isMuscriptor = showBackend && backend === "muscriptor";
  // 「输出乐器」默认折叠 — 节点面板空间有限，展开后才勾选/取消
  const [instrumentsOpen, setInstrumentsOpen] = useState(false);

  return (
    <NodeShell
      nodeId={props.id}
      label={t("amt.nodeLabel")}
      icon="♪"
      color="#a855f7"
      inputs={1}
      outputLabels={[t("amt.midiOutput")]}
    >
      <div
        className="amt-node-body"
        style={{
          padding: 8,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          minWidth: 200,
        }}
      >
        <label style={labelStyle}>
          {t("amt.modeLabel")}
          <select
            value={midiMode}
            onChange={(e) => handleModeChange(e.target.value)}
            style={selStyle}
          >
            {MIDI_MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {t(m.labelKey)}
              </option>
            ))}
          </select>
        </label>

        {showBackend && (
          <label style={labelStyle}>
            {t("amt.backendLabel")}
            <select
              value={backend}
              onChange={(e) => handleBackendChange(e.target.value)}
              style={selStyle}
            >
              {BACKENDS.map((b) => (
                <option key={b.value} value={b.value}>
                  {t(b.labelKey)}
                </option>
              ))}
            </select>
          </label>
        )}

        <label style={labelStyle}>
          {t("amt.quantizeLabel")}
          <select
            value={quantizeGrid}
            onChange={(e) => handleQuantizeChange(e.target.value)}
            style={selStyle}
          >
            {QUANTIZE_GRIDS.map((g) => (
              <option key={g} value={g}>
                {g === "off" ? t("amt.quantizeOff") : g}
              </option>
            ))}
          </select>
        </label>

        {/* 轨道布局：全轨道（按乐器拆分为多条）/ 单轨道（合并） */}
        <label style={labelStyle}>
          {t("amt.trackLayout")}
          <select
            value={midiTrackMode}
            onChange={(e) => handleTrackModeChange(e.target.value)}
            style={selStyle}
          >
            <option value="multi_track">{t("amt.trackLayoutAll")}</option>
            <option value="single_track">{t("amt.trackLayoutSingle")}</option>
          </select>
        </label>

        {isMuscriptor && (
          <>
            <label style={labelStyle}>
              {t("amt.processingChain")}
              <select
                value={muscriptorChain}
                onChange={(e) => handleChainChange(e.target.value)}
                style={selStyle}
              >
                <option value="official">{t("amt.chainOfficial")}</option>
                <option value="telknet">{t("amt.chainTelknet", "Telk-Net 增强 (实验)")}</option>
              </select>
            </label>
            <div style={{ ...labelStyle, gap: 4 }}>
              <div
                style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", userSelect: "none" }}
                onClick={() => setInstrumentsOpen((o) => !o)}
              >
                <span>
                  {instrumentsOpen ? "▼" : "▶"} {t("amt.nodeInstruments")}
                  <span style={{ color: "var(--text-tertiary)", fontSize: 10, marginLeft: 4 }}>
                    ({muscriptorInstruments.length}/{MUSCRIPTOR_INSTRUMENTS.length})
                  </span>
                </span>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); selectAllInstruments(); }}
                  style={{
                    fontSize: 10,
                    padding: "1px 6px",
                    borderRadius: 4,
                    border: "1px solid var(--border-subtle)",
                    background: "transparent",
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                  }}
                  title="全选"
                >
                  {t("common.selectAll")}
                </button>
              </div>
              {instrumentsOpen && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, maxHeight: 140, overflowY: "auto" }}>
                  {MUSCRIPTOR_INSTRUMENTS.map((id) => (
                    <label
                      key={id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 3,
                        fontSize: 10,
                        padding: "2px 6px",
                        borderRadius: 10,
                        border: "1px solid var(--border-subtle)",
                        cursor: "pointer",
                        background: muscriptorInstruments.includes(id)
                          ? "rgba(168,85,247,0.25)"
                          : "transparent",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={muscriptorInstruments.includes(id)}
                        onChange={() => toggleInstrument(id)}
                        style={{ accentColor: "#a855f7" }}
                      />
                      {MUSCRIPTOR_ZH_LABELS[id] ?? id}
                    </label>
                  ))}
                </div>
              )}
            </div>
          </>
        )}

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 12,
            color: "var(--text-secondary)",
            cursor: "pointer",
          }}
        >
          <input type="checkbox" checked={useGpu} onChange={handleGpuToggle} />
          {t("amt.useGpu")}
        </label>

        <div
          style={{
            fontSize: 11,
            color: "var(--text-tertiary)",
            lineHeight: 1.4,
            marginTop: 4,
          }}
        >
          {t("amt.modeDesc")}: {t(currentMode.labelKey)}
        </div>
      </div>
    </NodeShell>
  );
}
