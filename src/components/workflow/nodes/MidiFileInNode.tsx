import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

/** MIDI File In — undefined. */
export function MidiFileInNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const filePath = (params.filePath as string) ?? "";
  return (
    <NodeShell nodeId={props.id} label="MIDI File In" icon="🎹" color="#a855f7" inputs={0} outputs={1}>
      <div className="sep-node-body"><div className="sep-params">
        <div style={{ fontSize: 10, color: "#a855f7", padding: "0 0 6px 2px" }}>导入 .mid 文件 → 送入下游</div>
        <button onClick={() => {
          if (typeof window !== "undefined" && typeof window !== "undefined" && (window as any).__TAURI__) {
            import("@tauri-apps/plugin-dialog").then(d => d.open({ filters: [{ name: "MIDI", extensions: ["mid", "midi"] }] }))
              .then(p => { if (p) updateParams({ filePath: String(p) }); }).catch(() => {});
          }
        }} style={{ fontSize: 11, padding: "6px 10px", borderRadius: 6, border: "1px solid #a855f7", background: "#2e1065", color: "#c4b5fd", cursor: "pointer", width: "100%", fontFamily: "inherit" }}>
          {filePath ? "📄 " + filePath.split(/[\\/]/).pop() : "📂 选择 MIDI 文件..."}
        </button>
        {!filePath && <div style={{ fontSize: 10, color: "var(--color-error)", padding: "4px 0 0 2px" }}>⚠️ 需要先选择文件再运行</div>}
      </div></div>
    </NodeShell>
  );
}
