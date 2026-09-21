// P2-14 songStems 快速四轨节点（demucs）：0:源音频 → 0:vocals 1:drums 2:bass 3:other。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { DEMUCS_STEM_KINDS } from "../../../lib/models/song-tasks";
import { trackClassLabel, songNodeColor } from "./SongNodeShared";

export function SongStemsNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.stems")}
      icon="🎚"
      color={songNodeColor}
      inputs={1}
      outputLabels={DEMUCS_STEM_KINDS.map(trackClassLabel)}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div className="sep-param-row">
          <label>{t("songNode.songName")}</label>
          <input type="text" className="nodrag" value={(params.songName as string) ?? ""}
            onChange={(e) => updateParams({ songName: e.target.value })} style={{ width: 96 }} />
        </div>
      </div></div>
    </NodeShell>
  );
}
