// P2-14 songCover 翻唱节点：0:源音频 1:参考音频(可空) 2:提示词(可空) → 0:结果 WAV。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider, formatRatio } from "./ParamSlider";
import { songNodeColor } from "./SongNodeShared";

export function SongCoverNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.cover")}
      icon="🎤"
      color={songNodeColor}
      inputs={3}
      outputs={1}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div className="sep-param-row">
          <label>{t("songNode.songName")}</label>
          <input type="text" className="nodrag" value={(params.songName as string) ?? ""}
            onChange={(e) => updateParams({ songName: e.target.value })} style={{ width: 110 }} />
        </div>
        <ParamSlider
          label={t("songNode.strength")}
          title={t("songNode.strengthTip")}
          min={0} max={1} step={0.05}
          value={(params.coverStrength as number) ?? 0.5}
          onChange={(v) => updateParams({ coverStrength: v })}
          format={formatRatio}
        />
      </div></div>
    </NodeShell>
  );
}
