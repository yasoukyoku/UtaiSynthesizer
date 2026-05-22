import { useState, useCallback, useEffect, useRef } from "react";
import { type NodeProps, useReactFlow } from "@xyflow/react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { NodeShell } from "./NodeShell";
import { useMsstModelStore } from "../../../store/msst-models";
import { useWorkflowStore } from "../../../store/workflow";
import {
  MSST_CATALOG,
  type MsstCategory,
  type MsstArchitecture,
  CATEGORY_LABELS,
} from "../../../lib/models/msst-catalog";
import { useTranslation } from "react-i18next";

function t18(text: { zh: string; en: string; ja: string }, lang: string) {
  if (lang === "zh") return text.zh;
  if (lang === "ja") return text.ja;
  return text.en;
}

const CATEGORY_COLORS: Record<MsstCategory, string> = {
  vocals: "#ec4899",
  instrumental: "#60a5fa",
  denoise: "#4ade80",
  dereverb: "#a78bfa",
  karaoke: "#fbbf24",
  multistem: "#f97316",
  special: "#94a3b8",
};

interface InstalledOption { filename: string; displayName: string; arch: MsstArchitecture; stems: string[] }

function getModelsForCategory(category: MsstCategory, installedFiles: Set<string>): InstalledOption[] {
  return MSST_CATALOG
    .filter((e) => e.category === category && installedFiles.has(e.filename))
    .map((e) => ({ filename: e.filename, displayName: e.name.en, arch: e.architecture, stems: e.stems }));
}

export function SeparationNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const { setNodes } = useReactFlow();
  const installed = useMsstModelStore((s) => s.installed);
  const modelsDir = useMsstModelStore((s) => s.modelsDir);
  const installedFiles = new Set(installed.map((m) => m.filename));

  const params = (props.data?.params as Record<string, unknown>) ?? {};
  const category = (params.category as MsstCategory) ?? "vocals";
  const models = getModelsForCategory(category, installedFiles);

  const [selectedModel, setSelectedModel] = useState<string>((params.modelFile as string) ?? models[0]?.filename ?? "");
  const currentModel = models.find((m) => m.filename === selectedModel) ?? models[0];
  const stems = currentModel?.stems ?? ["Output"];
  const arch = currentModel?.arch ?? "bs_roformer";

  const updateParams = useCallback((updates: Record<string, unknown>) => {
    setNodes((nds) => nds.map((n) =>
      n.id === props.id
        ? { ...n, data: { ...n.data, params: { ...params, ...updates } } }
        : n,
    ));
  }, [props.id, params, setNodes]);

  useEffect(() => {
    if (!modelsDir) return;
    if (currentModel && (params.modelFile !== currentModel.filename || !params.modelPath || params.modelPath === `/${currentModel.filename.replace(/\.(ckpt|th|pth)$/, ".onnx")}`)) {
      updateParams({
        modelFile: currentModel.filename,
        stemLabels: currentModel.stems,
        modelPath: resolveOnnxPath(currentModel.filename),
      });
    }
  }, [modelsDir]);

  const resolveOnnxPath = useCallback((filename: string) => {
    const onnxName = filename.replace(/\.(ckpt|th|pth)$/, ".onnx");
    const dir = modelsDir.replace(/\\/g, "/");
    return `${dir}/${onnxName}`;
  }, [modelsDir]);

  const handleModelChange = useCallback((filename: string) => {
    setSelectedModel(filename);
    const m = models.find((x) => x.filename === filename);
    if (m) {
      updateParams({ modelFile: filename, stemLabels: m.stems, modelPath: resolveOnnxPath(filename) });
    }
  }, [models, updateParams, resolveOnnxPath]);

  const catLabel = t18(CATEGORY_LABELS[category], lang);

  const nodeOutputs = useWorkflowStore((s) => s.nodeOutputs);
  const outputPaths = Object.values(nodeOutputs)
    .map((segs) => segs[props.id])
    .find((p) => p && p.length > 0);

  return (
    <NodeShell
      nodeId={props.id}
      label={catLabel}
      icon={catLabel.charAt(0)}
      color={CATEGORY_COLORS[category] ?? "#ec4899"}
      inputs={1}
      outputLabels={stems}
    >
      <div className="sep-node-body">
        {models.length === 0 ? (
          <span className="sep-no-model">
            {lang === "zh" ? "未安装模型" : "No models installed"}
          </span>
        ) : (
          <>
            <select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              className="sep-model-select"
            >
              {models.map((m) => (
                <option key={m.filename} value={m.filename}>{m.displayName}</option>
              ))}
            </select>

            <SepParams arch={arch} params={params} onChange={updateParams} lang={lang} />
          </>
        )}
        {outputPaths && outputPaths.length > 0 && (
          <AudioPreview paths={outputPaths} stems={stems} />
        )}
      </div>
    </NodeShell>
  );
}

function AudioPreview({ paths, stems }: { paths: string[]; stems: string[] }) {
  const [playing, setPlaying] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef(0);

  const toggle = useCallback((idx: number) => {
    if (playing === idx) {
      audioRef.current?.pause();
      setPlaying(null);
      cancelAnimationFrame(rafRef.current);
      return;
    }
    if (audioRef.current) { audioRef.current.pause(); }
    const path = paths[idx];
    if (!path) return;
    const audio = new Audio(convertFileSrc(path));
    audioRef.current = audio;
    audio.onended = () => { setPlaying(null); setProgress(0); cancelAnimationFrame(rafRef.current); };
    audio.play();
    setPlaying(idx);
    const tick = () => {
      if (audio.duration) setProgress(audio.currentTime / audio.duration);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [playing, paths]);

  useEffect(() => () => {
    audioRef.current?.pause();
    cancelAnimationFrame(rafRef.current);
  }, []);

  return (
    <div className="sep-preview">
      {paths.map((_, i) => (
        <div key={i} className="sep-preview-row">
          <button
            className={`sep-preview-btn ${playing === i ? "playing" : ""}`}
            onClick={() => toggle(i)}
          >
            {playing === i ? "||" : ">>"}
          </button>
          <span className="sep-preview-label">{stems[i] ?? `#${i}`}</span>
          {playing === i && (
            <div className="sep-preview-bar">
              <div className="sep-preview-fill" style={{ width: `${progress * 100}%` }} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function SepParams({ arch, params, onChange, lang }: {
  arch: MsstArchitecture;
  params: Record<string, unknown>;
  onChange: (u: Record<string, unknown>) => void;
  lang: string;
}) {
  const overlap = (params.overlap as number) ?? 0.25;
  const normalize = (params.normalize as boolean) ?? false;

  return (
    <div className="sep-params">
      <div className="sep-param-row">
        <label>{lang === "zh" ? "重叠" : "Overlap"}</label>
        <input type="number" min={0} max={0.99} step={0.05} value={overlap}
          onChange={(e) => onChange({ overlap: parseFloat(e.target.value) || 0.25 })} />
      </div>

      {(arch === "bs_roformer" || arch === "mel_band_roformer" || arch === "mdx23c") && (
        <div className="sep-param-row">
          <label>{lang === "zh" ? "归一化" : "Normalize"}</label>
          <input type="checkbox" checked={normalize}
            onChange={(e) => onChange({ normalize: e.target.checked })} />
        </div>
      )}

      {arch === "htdemucs" && (
        <div className="sep-param-row">
          <label>{lang === "zh" ? "TTA 偏移" : "Shifts (TTA)"}</label>
          <input type="number" min={0} max={5} step={1}
            value={(params.shifts as number) ?? 0}
            onChange={(e) => onChange({ shifts: parseInt(e.target.value) || 0 })} />
        </div>
      )}
    </div>
  );
}
