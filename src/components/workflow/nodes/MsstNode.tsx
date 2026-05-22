import { useState, useCallback } from "react";
import { type NodeProps, useReactFlow } from "@xyflow/react";
import { NodeShell } from "./NodeShell";

const MODEL_PRESETS: Record<string, { label: string; stems: string[] }> = {
  bs_roformer: { label: "BS Roformer (2-stem)", stems: ["Vocals", "Instrumental"] },
  melband_roformer: { label: "MelBand Roformer (2-stem)", stems: ["Vocals", "Instrumental"] },
  mdx23c: { label: "MDX23C (2-stem)", stems: ["Vocals", "Instrumental"] },
  htdemucs: { label: "HTDemucs (4-stem)", stems: ["Drums", "Bass", "Other", "Vocals"] },
  htdemucs_6s: { label: "HTDemucs (6-stem)", stems: ["Drums", "Bass", "Guitar", "Piano", "Other", "Vocals"] },
};

export function MsstNode(props: NodeProps) {
  const { setNodes } = useReactFlow();
  const params = (props.data?.params as Record<string, unknown>) ?? {};
  const [model, setModel] = useState<string>((params.modelPreset as string) ?? "bs_roformer");

  const preset = MODEL_PRESETS[model] ?? MODEL_PRESETS.bs_roformer!;

  const handleModelChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      setModel(val);
      const p = MODEL_PRESETS[val]!;
      setNodes((nds) =>
        nds.map((n) =>
          n.id === props.id
            ? { ...n, data: { ...n.data, params: { ...params, modelPreset: val, stemLabels: p.stems, modelPath: "" } } }
            : n,
        ),
      );
    },
    [props.id, params, setNodes],
  );

  return (
    <NodeShell label="MSST" icon="[M]" color="#ec4899" inputs={1} outputLabels={preset.stems}>
      <label>Model Type</label>
      <select value={model} onChange={handleModelChange}>
        {Object.entries(MODEL_PRESETS).map(([key, val]) => (
          <option key={key} value={key}>{val.label}</option>
        ))}
      </select>
      <label>Model File</label>
      <input
        type="text"
        placeholder="model.onnx or model_name.ckpt"
        defaultValue={(params.modelPath as string) ?? ""}
        onChange={(e) => {
          setNodes((nds) =>
            nds.map((n) =>
              n.id === props.id
                ? { ...n, data: { ...n.data, params: { ...params, modelPath: e.target.value } } }
                : n,
            ),
          );
        }}
      />
    </NodeShell>
  );
}
