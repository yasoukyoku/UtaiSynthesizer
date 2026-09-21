// P2-14 歌曲节点族的公共零件（规划 11.5「可抽公共 SongTaskNodeShell」）：
// 轨种选择 chips、轨种本地化名、源文本节点的正文编辑区样式常量。
// 所有文案走 i18n（songNode.*）；轨种名复用 midiExtract.stemNames.*（人声/鼓/贝斯…）。
import { ACE_TRACK_CLASSES } from "../../../lib/models/song-tasks";
import i18n from "../../../i18n";
import type { CSSProperties } from "react";

/** 歌曲节点族统一配色（粉色系，与面板 catSong 一致）。 */
export const songNodeColor = "#ec4899";

/** 轨种 id → 本地化名（vocals→人声；未知轨种原样返回）。 */
export function trackClassLabel(cls: string): string {
  return i18n.t(`midiExtract.stemNames.${cls}`, { defaultValue: cls });
}

/** 单选/多选轨种 chips（lego=单选 select；complete/extract=多选 checkbox）。 */
export function TrackClassChips({ value, onChange, multi }: {
  value: string[];
  onChange: (next: string[]) => void;
  multi: boolean;
}) {
  const toggle = (cls: string) => {
    if (!multi) return onChange([cls]);
    const has = value.includes(cls);
    // 多选至少保留一个：取消最后一项时忽略
    const next = has ? value.filter((c) => c !== cls) : [...value, cls];
    if (next.length > 0) onChange(next);
  };
  return (
    <div className="sep-params" style={{ gap: 2 }}>
      {ACE_TRACK_CLASSES.map((cls) => (
        <label key={cls} className="sep-param-row" style={{ cursor: "pointer" }}>
          <span>{trackClassLabel(cls)}</span>
          <input
            type={multi ? "checkbox" : "radio"}
            name={multi ? undefined : "song-lego-track"}
            checked={value.includes(cls)}
            onChange={() => toggle(cls)}
            className="nodrag"
          />
        </label>
      ))}
    </div>
  );
}

/** 源文本节点正文 textarea 的统一样式（对齐 ChordBlockInNode 的内联样式）。 */
export const songTextAreaStyle: CSSProperties = {
  width: "100%",
  fontSize: 12,
  padding: 6,
  borderRadius: 6,
  border: "1px solid #ec4899",
  background: "#1c1917",
  color: "#fbcfe8",
  fontFamily: "monospace",
  resize: "none",
  boxSizing: "border-box",
};
