import { type NodeProps, useReactFlow } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

export function AudioOutputNode(props: NodeProps) {
  const { setNodes } = useReactFlow();
  const params = (props.data?.params as Record<string, unknown>) ?? {};
  const label = (params.laneLabel as string) ?? "Main";

  return (
    <NodeShell label="Output" icon="[>]" color="#4ade80" inputs={1} outputs={0}>
      <label>Lane</label>
      <input
        type="text"
        value={label}
        onChange={(e) => {
          setNodes((nds) =>
            nds.map((n) =>
              n.id === props.id
                ? { ...n, data: { ...n.data, params: { ...params, laneLabel: e.target.value } } }
                : n,
            ),
          );
        }}
      />
    </NodeShell>
  );
}
