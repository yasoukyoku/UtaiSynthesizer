// P2-14 songExtract 分轨提取节点：0:源音频 → 0..n 动态端口（按勾选轨种，逐轨顺序执行）。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { TrackClassChips, trackClassLabel, songNodeColor } from "./SongNodeShared";

export function SongExtractNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const classes = (params.trackClasses as string[]) ?? ["vocals", "drums", "bass", "guitar", "piano", "other"];
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.extract")}
      icon="✂"
      color={songNodeColor}
      inputs={1}
      outputLabels={classes.map(trackClassLabel)}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div className="sep-param-row" style={{ marginBottom: 2 }}>
          <label>{t("songNode.extractClasses")}</label>
        </div>
        <TrackClassChips value={classes} multi onChange={(next) => updateParams({ trackClasses: next })} />
      </div></div>
    </NodeShell>
  );
}
