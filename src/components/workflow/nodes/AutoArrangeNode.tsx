/**
 * AutoArrangeNode — AI 自动编曲 (纯前端, 不需要 Tauri 后端).
 * 输入: 旋律 MIDI 轨 (或音频, 自动先 AMT 转谱 — 需要 Tauri).
 * 输出: 11 条伴奏轨 + 末端口和弦摘要 (顺序与引擎 tracks 表、NODE_PORTS 三处严格对齐).
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
const MOODS = ["neutral", "happy", "sad", "energetic", "chill", "dark"];
const MOOD_ZH: Record<string, string> = {
  neutral: "中性", happy: "开心", sad: "悲伤", energetic: "活力", chill: "放松", dark: "暗黑",
};
// 与引擎 autoArrange 的 tracks 表逐位对齐；末位是和弦摘要（chords 端口，内联 JSON）。
const OUT_LABELS = [
  "🥁 Drums", "🎸 Bass", "🎹 Piano", "🪕 GuitarArp", "🎶 Strum", "🎼 E.Piano",
  "🎻 Strings", "🪟 Pad", "🎛 SynthPad", "💠 Pluck", "✨ Lead", "🎵 Chords",
];

export function AutoArrangeNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const style = (params.style as string) ?? "pop";
  const mood = (params.mood as string) ?? "neutral";
  const autoReplace = (params.autoReplace as boolean) ?? true;

  return (
    <NodeShell nodeId={props.id} label="AI 自动编曲" icon="🥁" color="#ec4899" inputs={1} outputLabels={OUT_LABELS}>
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
          <label className="sep-checkbox-row">
            <input
              type="checkbox"
              checked={autoReplace}
              onChange={(e) => updateParams({ autoReplace: e.target.checked })}
            />
            <span>再生成时自动替换旧轨</span>
          </label>
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            纯前端 · 22 种风格 · 11 类乐器自由组合 · 确定性输出
          </div>
        </div>
      </div>
    </NodeShell>
  );
}

