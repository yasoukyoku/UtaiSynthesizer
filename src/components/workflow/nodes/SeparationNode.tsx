import { useCallback, useEffect } from "react";
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { useMsstModelStore } from "../../../store/msst-models";
import {
  MSST_CATALOG,
  type MsstCategory,
  type MsstArchitecture,
  CATEGORY_LABELS,
  CATEGORY_COLORS,
  MSST_DEFAULT_NUM_OVERLAP,
  t18,
} from "../../../lib/models/msst-catalog";
import { useTranslation } from "react-i18next";

interface InstalledOption { filename: string; displayName: string; arch: MsstArchitecture; stems: string[] }

function getModelsForCategory(category: MsstCategory, installedFiles: Set<string>): InstalledOption[] {
  return MSST_CATALOG
    .filter((e) => e.category === category && installedFiles.has(e.filename))
    .map((e) => ({ filename: e.filename, displayName: e.name.en, arch: e.architecture, stems: e.stems }));
}

export function SeparationNode(props: NodeProps) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const [params, updateParams] = useNodeParams(props);
  const installed = useMsstModelStore((s) => s.installed);
  const modelsDir = useMsstModelStore((s) => s.modelsDir);
  const installedFiles = new Set(installed.map((m) => m.filename));

  const category = (params.category as MsstCategory) ?? "vocals";
  const models = getModelsForCategory(category, installedFiles);

  // Derived from the GRAPH params, never mirrored into local state: the modal-local undo restores node
  // params via setNodes WITHOUT remounting this component — a useState mirror kept displaying (and, on
  // the next edit, silently re-committing) the undone model while Run used the restored one.
  const selectedModel = (params.modelFile as string) ?? models[0]?.filename ?? "";
  const currentModel = models.find((m) => m.filename === selectedModel) ?? models[0];
  const stems = currentModel?.stems ?? ["Output"];
  const arch = currentModel?.arch ?? "bs_roformer";

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
    const m = models.find((x) => x.filename === filename);
    if (m) {
      updateParams({ modelFile: filename, stemLabels: m.stems, modelPath: resolveOnnxPath(filename) });
    }
  }, [models, updateParams, resolveOnnxPath]);

  const catLabel = t18(CATEGORY_LABELS[category], lang);

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
            {t18({ zh: "未安装模型", en: "No models installed", ja: "モデル未インストール" }, lang)}
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
      </div>
    </NodeShell>
  );
}

function SepParams({ arch, params, onChange, lang }: {
  arch: MsstArchitecture;
  params: Record<string, unknown>;
  onChange: (u: Record<string, unknown>) => void;
  lang: string;
}) {
  const numOverlap = (params.numOverlap as number) ?? MSST_DEFAULT_NUM_OVERLAP[arch] ?? 2;
  const normalize = (params.normalize as boolean) ?? false;
  const useTta = (params.useTta as boolean) ?? false;
  const shifts = (params.shifts as number) ?? 0;
  const isSpectral = arch === "bs_roformer" || arch === "mel_band_roformer" || arch === "mdx23c";

  return (
    <div className="sep-params">
      <div className="sep-param-row">
        <label title={t18({ zh: "重叠窗口数，越大越精细也越慢（MSST num_overlap）", en: "Overlap windows — higher = finer & slower (MSST num_overlap)", ja: "オーバーラップ数 — 大きいほど高精度・低速（MSST num_overlap）" }, lang)}>
          {t18({ zh: "重叠次数", en: "Overlap", ja: "オーバーラップ" }, lang)}
        </label>
        <span className="sep-overlap nodrag">
          <input
            className="sep-overlap-range nodrag"
            type="range" min={2} max={8} step={1} value={numOverlap}
            onPointerDown={(e) => e.stopPropagation()}
            onChange={(e) => onChange({ numOverlap: parseInt(e.target.value, 10) })}
          />
          <span className="sep-overlap-val">{numOverlap}</span>
        </span>
      </div>

      {isSpectral && (
        <div className="sep-param-row">
          <label title={t18({ zh: "批大小：一次喂多个 chunk 给 GPU（需重导出的新模型；旧模型自动按 1 跑）。显存不够会报错，调低即可", en: "Batch size — feed several chunks to the GPU at once (needs a re-exported model; old models run at 1). Lower it on out-of-memory", ja: "バッチサイズ — 複数チャンクを同時に GPU へ（再エクスポート済みモデルが必要。旧モデルは 1 で動作）。VRAM 不足なら下げてください" }, lang)}>
            {t18({ zh: "批大小", en: "Batch", ja: "バッチ" }, lang)}
          </label>
          <input type="number" min={1} max={16} step={1} value={(params.batch as number) ?? 1}
            onChange={(e) => onChange({ batch: Math.max(1, Math.min(16, parseInt(e.target.value) || 1)) })} />
        </div>
      )}

      {isSpectral && (
        <div className="sep-param-row">
          <label title={t18({ zh: "按 mean/std 归一化输入再推理（安静/过响素材有用）", en: "Mean/std-normalize the input before inference", ja: "推論前に mean/std で入力を正規化（音量が極端な素材に有効）" }, lang)}>
            {t18({ zh: "归一化", en: "Normalize", ja: "正規化" }, lang)}
          </label>
          <input type="checkbox" checked={normalize}
            onChange={(e) => onChange({ normalize: e.target.checked })} />
        </div>
      )}

      <div className="sep-param-row">
        <label title={t18({ zh: "测试时增强：原始/反相/声道交换三遍平均，更准但约慢 3 倍", en: "Test-time augmentation — averages original / polarity / channel-swap passes (~3× slower)", ja: "テスト時拡張 — 原音/極性反転/チャンネル入替の平均（約3倍遅い）" }, lang)}>
          TTA
        </label>
        <input type="checkbox" checked={useTta}
          onChange={(e) => onChange({ useTta: e.target.checked })} />
      </div>

      {arch === "htdemucs" && (
        <div className="sep-param-row">
          <label title={t18({ zh: "随机时移次数（仅 Demucs，0 = 关闭）", en: "Random time-shift passes (Demucs only, 0 = off)", ja: "タイムシフト回数（Demucs のみ、0 = 無効）" }, lang)}>
            {t18({ zh: "时移", en: "Shifts", ja: "シフト" }, lang)}
          </label>
          <input type="number" min={0} max={5} step={1} value={shifts}
            onChange={(e) => onChange({ shifts: Math.max(0, Math.min(5, parseInt(e.target.value) || 0)) })} />
        </div>
      )}
    </div>
  );
}
