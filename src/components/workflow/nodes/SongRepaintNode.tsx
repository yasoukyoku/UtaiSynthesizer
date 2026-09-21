// P2-14 songRepaint 局部重绘节点：0:源音频 1:提示词或歌词文本 → 0:结果 WAV。参数：起止秒。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor } from "./SongNodeShared";

function SecInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input type="number" min={0} step={0.5} className="nodrag"
      value={value} onChange={(e) => onChange(Math.max(0, parseFloat(e.target.value) || 0))}
      style={{ width: 56 }} />
  );
}

export function SongRepaintNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.repaint")}
      icon="🖌"
      color={songNodeColor}
      inputs={2}
      outputs={1}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div className="sep-param-row">
          <label>{t("songNode.startSec")}</label>
          <SecInput value={(params.repaintStart as number) ?? 0}
            onChange={(v) => updateParams({ repaintStart: v })} />
        </div>
        <div className="sep-param-row">
          <label>{t("songNode.endSec")}</label>
          <SecInput value={(params.repaintEnd as number) ?? 0}
            onChange={(v) => updateParams({ repaintEnd: v })} />
        </div>
        <div className="sep-param-row">
          <label>{t("songNode.songName")}</label>
          <input type="text" className="nodrag" value={(params.songName as string) ?? ""}
            onChange={(e) => updateParams({ songName: e.target.value })} style={{ width: 96 }} />
        </div>
      </div></div>
    </NodeShell>
  );
}
