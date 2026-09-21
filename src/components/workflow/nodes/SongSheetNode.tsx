// P2-14 songSheet 乐谱节点：0:源音频(可空) → 0:ABC 文本 1:MIDI（YuE2 生成式乐谱 + abc_to_midi）。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor } from "./SongNodeShared";

export function SongSheetNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.sheet")}
      icon="🎹"
      color={songNodeColor}
      inputs={1}
      outputLabels={[t("songNode.portAbc"), t("songNode.portMidi")]}
    >
      <div className="sep-node-body"><div className="sep-params">
        <textarea
          value={(params.prompt as string) ?? ""}
          onChange={(e) => updateParams({ prompt: e.target.value })}
          rows={2}
          className="nodrag"
          style={{ width: "100%", fontSize: 11, padding: 6, borderRadius: 6, border: "1px solid #ec4899", background: "#1c1917", color: "#fbcfe8", resize: "none", boxSizing: "border-box" }}
          placeholder={t("songNode.sheetPromptPlaceholder")}
        />
      </div></div>
    </NodeShell>
  );
}
