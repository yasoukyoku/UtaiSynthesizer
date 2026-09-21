import { useCallback } from "react";
import { useReactFlow } from "@xyflow/react";
/**
 * Per-node params + an immutable updater. Replaces the verbatim incantation that every node had:
 *   setNodes(nds => nds.map(n => n.id === props.id
 *     ? { ...n, data: { ...n.data, params: { ...params, ...updates } } } : n))
 * Behavior is identical to the old per-node code (merges onto the render-time params), just in one place.
 */
export function useNodeParams(props) {
    const { setNodes } = useReactFlow();
    const params = props.data?.params ?? {};
    const update = useCallback((updates) => {
        setNodes((nds) => nds.map((n) => n.id === props.id
            ? { ...n, data: { ...n.data, params: { ...params, ...updates } } }
            : n));
    }, [props.id, params, setNodes]);
    return [params, update];
}
