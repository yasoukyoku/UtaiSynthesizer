import { useCallback } from "react";
import { type NodeProps, useReactFlow } from "@xyflow/react";

/**
 * Per-node params + an immutable updater. Replaces the verbatim incantation that every node had:
 *   setNodes(nds => nds.map(n => n.id === props.id
 *     ? { ...n, data: { ...n.data, params: { ...params, ...updates } } } : n))
 * Behavior is identical to the old per-node code (merges onto the render-time params), just in one place.
 */
export function useNodeParams(
  props: NodeProps,
): [Record<string, unknown>, (updates: Record<string, unknown>) => void] {
  const { setNodes } = useReactFlow();
  const params = (props.data?.params as Record<string, unknown>) ?? {};
  const update = useCallback(
    (updates: Record<string, unknown>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === props.id
            ? { ...n, data: { ...n.data, params: { ...params, ...updates } } }
            : n,
        ),
      );
    },
    [props.id, params, setNodes],
  );
  return [params, update];
}
