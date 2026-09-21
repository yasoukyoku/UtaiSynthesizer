/**
 * SpeedShiftNode — 保音高变速.
 * 引擎: Rust stretch_segment_audio (Signalsmith 频谱时间拉伸, 保音高).
 * 范围: 0.25x ~ 4.0x. 1.0 = 原样直通. 超出范围报错, 后端失败时 passthrough 兜底.
 */
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSliderWithUnit } from "./ParamSliderWithUnit";

export function SpeedShiftNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const rate = (params.rate as number) ?? 1.0;

  return (
    <NodeShell nodeId={props.id} label="变速" icon="⏱" color="#06b6d4" inputs={1} outputs={1}>
      <div className="sep-node-body">
        <div className="sep-params">
          <ParamSliderWithUnit
            label="播放速度"
            unitType="ratio"
            title="Signalsmith 频谱时间拉伸 — 变速但保音高. 0.5x = 半速, 2.0x = 倍速 (音高不变). 范围 0.25x ~ 4.0x."
            min={0.25} max={4.0} step={0.05} value={rate}
            onChange={(v) => updateParams({ rate: v })}
          />
          <div style={{ fontSize: 10, color: "#888", padding: "4px 0 0 4px" }}>
            Rust Signalsmith · 频谱时间拉伸 · 保音高
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
