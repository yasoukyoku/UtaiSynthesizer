/**
 * ChordDetectNode — 和弦识别 (纯前端, 确定性).
 * 引擎: analyzeChords — MIDI 音符 → 时长加权音级直方图 → 和弦模板匹配 → 惯性平滑.
 * 输入: MIDI 文件 / chordDetect JSON / chordBlockIn JSON.
 * 输出: slot0 = 和弦段 JSON (含 key), slot1 = 和弦简写串 (C | Am | F | G).
 */
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

export function ChordDetectNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const beatsPerBar = (params.beatsPerBar as number) ?? 4;

  return (
    <NodeShell nodeId={props.id} label="和弦识别" icon="🎼" color="#a78bfa" inputs={1} outputs={2}>
      <div className="sep-node-body">
        <div className="sep-params">
          <div className="sep-label-row">
            <span className="sep-label">每拍窗数</span>
            <select
              className="sep-select"
              value={beatsPerBar}
              onChange={(e) => updateParams({ beatsPerBar: Number(e.target.value) })}
            >
              <option value={4}>4/4 拍 (默认)</option>
              <option value={3}>3/4 拍</option>
              <option value={6}>6/8 拍</option>
            </select>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 0 4px" }}>
            MIDI 音符驱动 · analyzeChords · 确定性输出
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "2px 0 0 4px" }}>
            💡 接 MIDI File In 或 AMT 转谱节点
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
