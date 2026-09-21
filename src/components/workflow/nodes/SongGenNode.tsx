// P2-14 songGen 歌曲模型核心节点（规划 11.2/11.3）：
// 0:歌词文本(可空) 1:提示词文本(可空) 2:参考音频(可空) → 0:整首 1:MIDI 2:字幕 3+i:分轨。
// 歌词/提示词优先取连线（songLyrics/songPrompt），未连线时用节点内文本框。
import { useState } from "react";
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
import { ACE_TRACK_CLASSES } from "../../../lib/models/song-tasks";
import { trackClassLabel, songNodeColor, songTextAreaStyle } from "./SongNodeShared";

export function SongGenNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const [showAdv, setShowAdv] = useState(false);

  const model = (params.model as string) ?? "acestep-v1.5";
  const isYue2 = model.startsWith("yue2");
  const songTask = (params.songTask as string) ?? "generate";
  const wantStems = (params.wantStems as boolean) ?? false;
  const lyricsConn = (params.lyrics as string) ?? "";
  const promptConn = (params.prompt as string) ?? "";

  // 输出端口：0=整首 1=MIDI 2=LRC；勾选分轨后 3+i 按轨种展开（与引擎映射一致）
  const outLabels = [
    t("songNode.portMix"),
    t("songNode.portMidi"),
    t("songNode.portLrc"),
    ...(wantStems ? ACE_TRACK_CLASSES.map(trackClassLabel) : []),
  ];

  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.gen")}
      icon="🎵"
      color={songNodeColor}
      inputs={3}
      outputLabels={outLabels}
      width={280}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div className="sep-param-row">
          <label>{t("songNode.model")}</label>
          <select value={model} onChange={(e) => updateParams({ model: e.target.value })} className="nodrag">
            <option value="acestep-v1.5">ACE-Step v1.5</option>
            <option value="yue2-3b">YuE2-3B</option>
          </select>
        </div>
        <div className="sep-param-row">
          <label>{t("songNode.genTask")}</label>
          <select value={songTask} onChange={(e) => updateParams({ songTask: e.target.value })} className="nodrag">
            <option value="generate">{t("songNode.taskWrite")}</option>
            <option value="instrumental">{t("songNode.taskInst")}</option>
          </select>
        </div>
        <ParamSlider
          label={t("songNode.duration")}
          min={0} max={300} step={5}
          value={(params.durationSec as number) ?? 0}
          onChange={(v) => updateParams({ durationSec: v })}
          format={(v) => (v === 0 ? t("songNode.durationAuto") : `${v}s`)}
        />
        <div className="sep-param-row">
          <label>{t("songNode.seed")}</label>
          <input type="number" min={0} step={1} className="nodrag"
            value={(params.seed as number) ?? 0}
            onChange={(e) => updateParams({ seed: Math.max(0, parseInt(e.target.value) || 0) })} />
        </div>
        <div className="sep-param-row">
          <label>{t("songNode.outputs")}</label>
          <span className="nodrag" style={{ display: "flex", gap: 6 }}>
            <label style={{ display: "flex", gap: 2, alignItems: "center", cursor: "pointer" }}>
              {t("songNode.wantStems")}
              <input type="checkbox" checked={wantStems}
                onChange={(e) => updateParams({ wantStems: e.target.checked })} />
            </label>
            <label style={{ display: "flex", gap: 2, alignItems: "center", cursor: "pointer" }}>
              MIDI
              <input type="checkbox" checked={(params.wantMidi as boolean) ?? false}
                onChange={(e) => updateParams({ wantMidi: e.target.checked })} />
            </label>
            <label style={{ display: "flex", gap: 2, alignItems: "center", cursor: "pointer" }}>
              LRC
              <input type="checkbox" checked={(params.wantLrc as boolean) ?? false}
                onChange={(e) => updateParams({ wantLrc: e.target.checked })} />
            </label>
          </span>
        </div>
        {/* 歌词/提示词：优先取连线输入，未连线时用节点内文本框（11.3） */}
        <textarea
          value={lyricsConn}
          onChange={(e) => updateParams({ lyrics: e.target.value })}
          rows={2}
          className="nodrag"
          style={songTextAreaStyle}
          placeholder={t("songNode.lyricsPlaceholder")}
        />
        <textarea
          value={promptConn}
          onChange={(e) => updateParams({ prompt: e.target.value })}
          rows={2}
          className="nodrag"
          style={songTextAreaStyle}
          placeholder={t("songNode.promptPlaceholder")}
        />
        <button type="button" className="nodrag"
          onClick={() => setShowAdv((s) => !s)}
          style={{ fontSize: 10, background: "none", border: "none", color: "#f9a8d4", cursor: "pointer", textAlign: "left", padding: 0 }}>
          {showAdv ? "▾" : "▸"} {t("songNode.advanced")}
        </button>
        {showAdv && (isYue2 ? (
          <div className="sep-param-row">
            <label>CoT</label>
            <select value={((params.cot as string) ?? "full")} onChange={(e) => updateParams({ cot: e.target.value })} className="nodrag">
              <option value="full">full</option>
              <option value="off">off</option>
            </select>
          </div>
        ) : (
          <>
            <ParamSlider label="guidance" min={0} max={15} step={0.5}
              value={(params.guidanceScale as number) ?? 0}
              onChange={(v) => updateParams({ guidanceScale: v || undefined })}
              format={(v) => (v === 0 ? t("songNode.default") : v.toFixed(1))} />
            <ParamSlider label="steps" min={0} max={100} step={5}
              value={(params.numInferenceSteps as number) ?? 0}
              onChange={(v) => updateParams({ numInferenceSteps: v || undefined })}
              format={(v) => (v === 0 ? t("songNode.default") : String(v))} />
          </>
        ))}
      </div></div>
    </NodeShell>
  );
}
