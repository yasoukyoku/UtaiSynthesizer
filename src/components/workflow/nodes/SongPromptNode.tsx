// P2-14 songPrompt 风格提示词源节点：风格/情绪/乐器/BPM 描述 + 灵感模板（规划 11.2）。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor, songTextAreaStyle } from "./SongNodeShared";

export function SongPromptNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const text = (params.text as string) ?? "";
  const presets = [
    { key: "presetPop", v: "pop, upbeat, catchy chorus, modern drums, warm bass" },
    { key: "presetRock", v: "rock, driving guitar, powerful drums, energetic" },
    { key: "presetBallad", v: "ballad, emotional, piano, strings, slow tempo" },
    { key: "presetElectronic", v: "electronic, synth lead, punchy beat, 128 bpm" },
    { key: "presetFolk", v: "folk, acoustic guitar, gentle, storytelling" },
  ];
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.prompt")}
      icon="🎨"
      color={songNodeColor}
      inputs={0}
      outputs={1}
      outputLabels={[t("songNode.portPrompt")]}
      width={280}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div style={{ display: "flex", gap: 3, flexWrap: "wrap", paddingBottom: 4 }}>
          {presets.map((p) => (
            <button key={p.key} type="button"
              onClick={() => updateParams({ text: p.v })}
              className="nodrag"
              style={{ fontSize: 9, padding: "2px 5px", borderRadius: 4, border: "1px solid #9f1239", background: "#4c0519", color: "#fda4af", cursor: "pointer", fontFamily: "inherit" }}>
              {t(`songNode.${p.key}`)}
            </button>
          ))}
        </div>
        <textarea
          value={text}
          onChange={(e) => updateParams({ text: e.target.value })}
          rows={4}
          className="nodrag"
          style={songTextAreaStyle}
          placeholder={t("songNode.promptPlaceholder")}
        />
      </div></div>
    </NodeShell>
  );
}
