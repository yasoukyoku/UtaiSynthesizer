/**
 * DeepOriginalNode — 深度原创 (一键链).
 * 把现有歌曲改得面目全非变成你自己的:
 *   输入音频 → 变速 → 变调 → AMT 转谱 → 和弦识别 → 自动编曲 → 可选 RVC 换声
 * 输出: 5 轨整首歌 (旋律 + 鼓 + 贝斯 + 钢琴 + 铺底).
 * 需要 Tauri 后端 (AMT 转谱 + RVC).
 */
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";

export function DeepOriginalNode(props: NodeProps) {
  const [params, updateParams] = useNodeParams(props);
  const semitones = (params.semitones as number) ?? 3;
  const rate = (params.rate as number) ?? 1.0;
  const style = (params.style as string) ?? "trap";
  const enableRVC = (params.enableRVC as boolean) ?? false;
  const randomSeed = (params.randomSeed as number) ?? 0;
  const autoInstruments = (params.instruments as string[]) ?? ["drums", "bass", "piano", "chords", "melody"];

  const STYLES = [
    { v: "trap", label: "Trap (常用)" },
    { v: "lofi", label: "Lo-Fi" },
    { v: "techno", label: "Techno" },
    { v: "edm", label: "电子 EDM" },
    { v: "pop", label: "流行" },
    { v: "kpop", label: "K-Pop" },
    { v: "chinese", label: "中国风" },
    { v: "reggae", label: "雷鬼" },
    { v: "ambient", label: "氛围" },
    { v: "rock", label: "摇滚" },
  ];

  const ALL_INSTRUMENTS = [
    { v: "drums", label: "🥁 鼓" },
    { v: "bass", label: "🎸 贝斯" },
    { v: "piano", label: "🎹 钢琴" },
    { v: "chords", label: "🪟 Pad" },
    { v: "melody", label: "✨ AI Lead" },
  ];

  const toggleInst = (v: string) => {
    if (autoInstruments.includes(v)) {
      updateParams({ instruments: autoInstruments.filter((x) => x !== v) });
    } else {
      updateParams({ instruments: [...autoInstruments, v] });
    }
  };

  return (
    <NodeShell nodeId={props.id} label="⚡深度原创" icon="✨" color="#f59e0b" inputs={1} outputs={5}>
      <div className="sep-node-body">
        <div className="sep-params">
          <div style={{ fontSize: 11, color: "var(--color-warning)", padding: "0 0 6px 2px" }}>
            一键链: 音频→变速→变调→AMT→和弦→编曲
          </div>
          <ParamSlider
            label="变调 (半音)"
            title="把原曲移调. ±3/4/5/7 半音通常不会被识别为原曲"
            min={-12} max={12} step={1} value={semitones}
            format={(v) => (v > 0 ? `+${v}` : `${v}`)}
            onChange={(v) => updateParams({ semitones: v })}
          />
          <ParamSlider
            label="变速 (x)"
            title="播放速率. 0.8x / 1.25x 改节奏不调式"
            min={0.7} max={1.5} step={0.05} value={rate}
            format={(v) => `${v.toFixed(2)}x`}
            onChange={(v) => updateParams({ rate: v })}
          />
          <ParamSlider
            label="🎲 随机种子"
            title="0 = 确定性 (同输入同输出). 改值 = 每次生成不同旋律/节奏变体"
            min={0} max={9999} step={1} value={randomSeed}
            format={(v) => (v === 0 ? "🎯 确定" : `#${v}`)}
            onChange={(v) => updateParams({ randomSeed: v })}
          />
          <div className="sep-label-row">
            <span className="sep-label">编曲风格</span>
            <select className="sep-select" value={style} onChange={(e) => updateParams({ style: e.target.value })}>
              {STYLES.map((s) => <option key={s.v} value={s.v}>{s.label}</option>)}
            </select>
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", padding: "4px 0 2px 2px" }}>乐器轨:</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 3, padding: "0 2px" }}>
            {ALL_INSTRUMENTS.map((inst) => {
              const on = autoInstruments.includes(inst.v);
              return (
                <button
                  key={inst.v}
                  type="button"
                  className={`sep-inst-chip ${on ? "on" : ""}`}
                  onClick={() => toggleInst(inst.v)}
                  style={{
                    fontSize: 10,
                    padding: "2px 6px",
                    borderRadius: 6,
                    border: `1px solid ${on ? "#f59e0b" : "var(--border-strong)"}`,
                    background: on ? "#78350f" : "transparent",
                    color: on ? "var(--color-warning)" : "var(--text-muted)",
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {inst.label}
                </button>
              );
            })}
          </div>
          <label className="sep-checkbox-row" style={{ marginTop: 6 }}>
            <input type="checkbox" checked={enableRVC} onChange={(e) => updateParams({ enableRVC: e.target.checked })} />
            <span style={{ fontSize: 11 }}>RVC 换声 (用你训练的音色替原唱)</span>
          </label>
          <div style={{ fontSize: 10, color: "var(--color-error)", padding: "4px 0 0 4px" }}>
            ⚠️ 需要 Tauri 桌面版 + AMT 模型
          </div>
        </div>
      </div>
    </NodeShell>
  );
}
