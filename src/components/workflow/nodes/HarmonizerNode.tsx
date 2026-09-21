/**
 * HarmonizerNode — 人声和声 (纯前端).
 * 输入: 旋律 MIDI (或 chordDetect 输出的音符数组).
 * 输出: 1 轨和声块 MIDI (JSON, notes 数组).
 * 引擎: generateChordMidi() — 和声化引擎 (复用菜单层同款).
 */
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

const CHORD_STYLES = ["POP_STANDARD", "POP_COMPLEX", "DARK", "RANDB", "NOCONSTRAINT"];
const STYLE_ZH: Record<string, string> = {
  POP_STANDARD: "标准流行", POP_COMPLEX: "丰富流行", DARK: "暗黑", RANDB: "随机变化", NOCONSTRAINT: "自由分析",
};

export function HarmonizerNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const chordStyle = (params.chordStyle as string) ?? "POP_STANDARD";
  const chordsPerBar = (params.chordsPerBar as number) ?? 1;
  const key = (params.key as string) ?? "auto";

  return (
    <NodeShell nodeId={props.id} label="和声层" icon="🎶" color="#06b6d4" inputs={1} outputs={2}>
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">和弦风格</span>
            <select
              className="sep-select"
              value={chordStyle}
              onChange={(e) => updateParams({ chordStyle: e.target.value })}
            >
              {CHORD_STYLES.map((s) => <option key={s} value={s}>{STYLE_ZH[s]}</option>)}
            </select>
          </div>
          <div className="sep-label-row">
            <span className="sep-label">每小节和弦数</span>
            <select
              className="sep-select"
              value={chordsPerBar}
              onChange={(e) => updateParams({ chordsPerBar: Number(e.target.value) })}
            >
              <option value={1}>1 (整小节)</option>
              <option value={2}>2 (半小节)</option>
            </select>
          </div>
          <div className="sep-label-row">
            <span className="sep-label">调性</span>
            <input
              type="text"
              className="sep-input"
              style={{ width: 80 }}
              value={key}
              placeholder="auto"
              onChange={(e) => updateParams({ key: e.target.value })}
            />
          </div>
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            纯前端 · MIDI-SAG 和声化引擎 · 确定性输出
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
