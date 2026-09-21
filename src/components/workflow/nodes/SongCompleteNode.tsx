// P2-14 songComplete 音轨补全节点：0:源音频 1:提示词(可空) → 0:结果 WAV + 1..n 各补全声部分轨。
// 输出端口数 = 1 + 勾选声部数（与引擎映射一致，勾选顺序 = 端口顺序）。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { TrackClassChips, trackClassLabel, songNodeColor } from "./SongNodeShared";

export function SongCompleteNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const classes = (params.trackClasses as string[]) ?? ["drums", "bass", "guitar"];
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.complete")}
      icon="🧩"
      color={songNodeColor}
      inputs={2}
      outputLabels={[t("songNode.portMix"), ...classes.map(trackClassLabel)]}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div className="sep-param-row" style={{ marginBottom: 2 }}>
          <label>{t("songNode.trackClasses")}</label>
        </div>
        <TrackClassChips value={classes} multi onChange={(next) => updateParams({ trackClasses: next })} />
      </div></div>
    </NodeShell>
  );
}
