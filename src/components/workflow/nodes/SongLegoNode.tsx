// P2-14 songLego 叠加音轨节点：0:源音频 1:提示词文本(可空) → 0:结果 WAV。参数选叠加轨种。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { TrackClassChips, songNodeColor } from "./SongNodeShared";

export function SongLegoNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const trackName = (params.trackName as string) ?? "guitar";
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.lego")}
      icon="🧱"
      color={songNodeColor}
      inputs={2}
      outputs={1}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div className="sep-param-row" style={{ marginBottom: 2 }}>
          <label>{t("songNode.legoTrack")}</label>
        </div>
        <TrackClassChips value={[trackName]} multi={false}
          onChange={(next) => updateParams({ trackName: next[0] })} />
      </div></div>
    </NodeShell>
  );
}
