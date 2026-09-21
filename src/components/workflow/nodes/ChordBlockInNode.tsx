import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";

/** Chord Block In — undefined. */
export function ChordBlockInNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const chords = (params.chords as string) ?? "C | Am | F | G";
  const bpm = (params.bpm as number) ?? 120;
  const PRESETS = [
    { name: "🎵 流行", v: "C | Am | F | G" },
    { name: "🌧 伤感", v: "Am | F | C | G" },
    { name: "🖤 情感", v: "Am | F | C | E" },
    { name: "🎸 摇滚", v: "E | A | B | E" },
    { name: "🌊 Lo-Fi", v: "Dm | Bb | F | C" },
    { name: "🎩 Jazz", v: "Dm7 | G7 | Cmaj7" },
    { name: "💜 vi-IV-I-V", v: "Am | F | C | G" },
  ];
  return (
    <NodeShell nodeId={props.id} label="Chord Block In" icon="🎼" color="#eab308" inputs={0} outputs={1} width={280}>
      <div className="sep-node-body"><div className="sep-params">
        <div style={{ fontSize: 10, color: "#eab308", padding: "0 0 6px 2px" }}>手动和弦 → 自动生成旋律/编曲</div>
        <textarea value={chords} onChange={(e)=>updateParams({chords:e.target.value})} rows={2}
          style={{ width:"100%", fontSize:12, padding:6, borderRadius:6, border:"1px solid #eab308", background:"#1c1917", color:"#fde68a", fontFamily:"monospace", resize:"none", boxSizing:"border-box" }}
          placeholder="C | Am | F | G" />
        <div style={{ display:"flex", gap:3, flexWrap:"wrap", marginTop:4 }}>
          {PRESETS.map(p => (
            <button key={p.v} onClick={()=>updateParams({chords:p.v})} type="button"
              style={{ fontSize:9, padding:"2px 5px", borderRadius:4, border:"1px solid #a16207", background:"#451a03", color:"#fbbf24", cursor:"pointer", fontFamily:"inherit" }}>
              {p.name}
            </button>
          ))}
        </div>
        <div style={{ display:"flex", gap:6, marginTop:6 }}>
          <label style={{ fontSize:10, color:"var(--text-primary)" }}>BPM <input type="number" min={40} max={220} value={bpm}
            onChange={(e)=>updateParams({bpm:+e.target.value})}
            style={{ width:50, marginLeft:4, padding:2, borderRadius:4, border:"1px solid var(--border-strong)", background:"var(--bg-deep)", color:"var(--text-primary)" }} /></label>
        </div>
      </div></div>
    </NodeShell>
  );
}
