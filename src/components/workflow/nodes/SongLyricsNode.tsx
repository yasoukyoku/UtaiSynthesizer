// P2-14 songLyrics 歌词源节点：多行歌词编辑 + 段落结构标签工具条（规划 11.2）。
// 引擎输出以 lyrics:// 前缀的虚拟文本（11.3 约定），下游 songGen 0 口消费。
import { type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { songNodeColor, songTextAreaStyle } from "./SongNodeShared";

const STRUCT_TAGS = ["[Verse]", "[Chorus]", "[Bridge]", "[Outro]", "[Instrumental]"];

export function SongLyricsNode(props: NodeProps) {
  const { t } = useTranslation();
  const [params, updateParams] = useNodeParams(props);
  const text = (params.text as string) ?? "";
  return (
    <NodeShell
      nodeId={props.id}
      label={t("songNode.lyrics")}
      icon="✍"
      color={songNodeColor}
      inputs={0}
      outputs={1}
      outputLabels={[t("songNode.portLyrics")]}
      width={280}
    >
      <div className="sep-node-body"><div className="sep-params">
        <div style={{ display: "flex", gap: 3, flexWrap: "wrap", paddingBottom: 4 }}>
          {STRUCT_TAGS.map((tag) => (
            <button key={tag} type="button"
              onClick={() => updateParams({ text: text ? `${text}\n${tag}\n` : `${tag}\n` })}
              className="nodrag"
              style={{ fontSize: 9, padding: "2px 5px", borderRadius: 4, border: "1px solid #9d174d", background: "#500724", color: "#f9a8d4", cursor: "pointer", fontFamily: "inherit" }}>
              {tag}
            </button>
          ))}
        </div>
        <textarea
          value={text}
          onChange={(e) => updateParams({ text: e.target.value })}
          rows={6}
          className="nodrag"
          style={songTextAreaStyle}
          placeholder={t("songNode.lyricsPlaceholder")}
        />
      </div></div>
    </NodeShell>
  );
}
