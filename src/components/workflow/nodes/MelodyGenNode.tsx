/**
 * MelodyGenNode — AI 旋律生成 (纯前端, 确定性).
 * 输入: chordBlockIn 或 chordDetect 的和弦输出.
 * 输出: 旋律轨 MIDI notes (JSON, 确定性 seed 同输入同输出).
 * 引擎: arrange() 的 melody 生成逻辑 (复用 arranger.ts).
 */
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

const STYLES = [
  "pop", "rock", "jazz", "edm", "latin", "funk", "rnb", "ballad", "country",
  "lofi", "trap", "reggae", "bossanova", "kpop", "chinese", "ambient", "techno",
];
const STYLE_ZH: Record<string, string> = {
  pop: "流行", rock: "摇滚", jazz: "爵士", edm: "电子", latin: "拉丁",
  funk: "放克", rnb: "R&B", ballad: "抒情", country: "乡村",
  lofi: "Lo-Fi", trap: "Trap", reggae: "雷鬼", bossanova: "Bossa Nova",
  kpop: "K-Pop", chinese: "中国风", ambient: "氛围", techno: "Techno",
};
const MOODS = ["neutral", "happy", "sad", "energetic", "chill", "epic"];
const MOOD_ZH: Record<string, string> = {
  neutral: "中性", happy: "开心", sad: "悲伤", energetic: "活力", chill: "放松", epic: "史诗",
};

export function MelodyGenNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const style = (params.style as string) ?? "pop";
  const mood = (params.mood as string) ?? "neutral";

  return (
    <NodeShell nodeId={props.id} label="AI 旋律" icon="🎵" color="#10b981" inputs={1} outputs={2}>
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">风格</span>
            <select
              className="sep-select"
              value={style}
              onChange={(e) => updateParams({ style: e.target.value })}
            >
              {STYLES.map((s) => <option key={s} value={s}>{STYLE_ZH[s]}</option>)}
            </select>
          </div>
          <div className="sep-label-row">
            <span className="sep-label">情绪</span>
            <select
              className="sep-select"
              value={mood}
              onChange={(e) => updateParams({ mood: e.target.value })}
            >
              {MOODS.map((m) => <option key={m} value={m}>{MOOD_ZH[m]}</option>)}
            </select>
          </div>
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            纯前端 · 和弦驱动确定性 seed · 同输入同输出
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
