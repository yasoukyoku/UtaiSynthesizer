import { useCallback, useEffect } from "react";
import { type NodeProps } from "@xyflow/react";
import { NodeShell } from "./NodeShell";
import { useNodeParams } from "./useNodeParams";
import { ParamSlider } from "./ParamSlider";
import { useMsstModelStore } from "../../../store/msst-models";
import { msstOnnxPath } from "../../../lib/workflow/modelPathHeal";
import {
  MSST_CATALOG,
  type MsstCategory,
  type MsstArchitecture,
  CATEGORY_LABELS,
  CATEGORY_COLORS,
  MSST_DEFAULT_NUM_OVERLAP,
  MSST_DEFAULT_PRECISION,
  MSST_FP16_TIP,
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
  const arch = currentModel?.arch ?? "bs_roformer";
  // Precision is choosable only when BOTH onnx variants are on disk — otherwise Rust already
  // runs the only one that exists (with graceful fallback), so a selector would be a lie.
  const installedModel = installed.find((m) => m.filename === currentModel?.filename);
  const hasBothPrecisions = !!installedModel?.has_onnx && !!installedModel?.has_fp16;
  // Output ports MUST follow the model's TRUE order from its json (stem_names [+ residual]) —
  // Rust deposits stems by index in that order. Hand-written catalog `stems` lists are only the
  // pre-conversion display fallback: htdemucs_6s's model card says drums/bass/guitar/piano/other/
  // vocals but the weights output [drums,bass,other,vocals,guitar,piano] — labeling ports from
  // the catalog put VOCALS on the Piano port.
  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  const stems = installedModel?.stem_names?.length
    ? [...installedModel.stem_names, ...(installedModel.residual_name ? [installedModel.residual_name] : [])].map(cap)
    : currentModel?.stems ?? ["Output"];

  useEffect(() => {
    if (!modelsDir) return;
    if (currentModel && (params.modelFile !== currentModel.filename || !params.modelPath || params.modelPath === msstOnnxPath("", currentModel.filename))) {
      updateParams({
        modelFile: currentModel.filename,
        stemLabels: stems,
        modelPath: resolveOnnxPath(currentModel.filename),
      });
    }
  }, [modelsDir]);

  // Keep the persisted port labels in sync with the model's true order — the json's stem_names
  // only become available after install/convert, which can be AFTER the params were first written
  // (and lane naming reads params.stemLabels, not this component).
  useEffect(() => {
    const saved = params.stemLabels as string[] | undefined;
    if (currentModel && saved && JSON.stringify(saved) !== JSON.stringify(stems)) {
      updateParams({ stemLabels: stems });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stems.join("")]);

  // The extension rule lives in modelPathHeal.msstOnnxPath (shared with the engine's use-time heal).
  const resolveOnnxPath = useCallback((filename: string) => msstOnnxPath(modelsDir, filename), [modelsDir]);

  const handleModelChange = useCallback((filename: string) => {
    const m = models.find((x) => x.filename === filename);
    if (m) {
      // Same true-order rule as above: prefer the installed json's stem_names over catalog stems.
      const inst = installed.find((x) => x.filename === filename);
      const labels = inst?.stem_names?.length
        ? [...inst.stem_names, ...(inst.residual_name ? [inst.residual_name] : [])].map(cap)
        : m.stems;
      updateParams({ modelFile: filename, stemLabels: labels, modelPath: resolveOnnxPath(filename) });
    }
  }, [models, installed, updateParams, resolveOnnxPath]);

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

            <SepParams arch={arch} params={params} onChange={updateParams} lang={lang} showPrecision={hasBothPrecisions} installedNumOverlap={installedModel?.num_overlap ?? undefined} />
          </>
        )}
      </div>
    </NodeShell>
  );
}

function SepParams({ arch, params, onChange, lang, showPrecision, installedNumOverlap }: {
  arch: MsstArchitecture;
  params: Record<string, unknown>;
  onChange: (u: Record<string, unknown>) => void;
  lang: string;
  showPrecision: boolean;
  /** The installed model json's real num_overlap — what Rust actually runs when the user
   *  never touches the slider. Display truth beats the per-arch catalog constant. */
  installedNumOverlap?: number;
}) {
  const numOverlap = (params.numOverlap as number) ?? installedNumOverlap
    ?? MSST_DEFAULT_NUM_OVERLAP[arch] ?? 2;
  const normalize = (params.normalize as boolean) ?? false;
  const useTta = (params.useTta as boolean) ?? false;
  const shifts = (params.shifts as number) ?? 0;
  const aggression = (params.aggression as number) ?? 5;
  const postProcess = (params.postProcess as boolean) ?? false;
  const isSpectral = arch === "bs_roformer" || arch === "mel_band_roformer" || arch === "mdx23c";

  return (
    <div className="sep-params">
      {/* VR has no overlap-add — its window stride is fixed, so the slider would be a lie. */}
      {arch !== "uvr_vr" && (
        <ParamSlider
          label={t18({ zh: "重叠次数", en: "Overlap", ja: "オーバーラップ" }, lang)}
          title={t18({ zh: "重叠窗口数，越大越精细也越慢（MSST num_overlap）", en: "Overlap windows — higher = finer & slower (MSST num_overlap)", ja: "オーバーラップ数 — 大きいほど高精度・低速（MSST num_overlap）" }, lang)}
          min={2} max={8} step={1} value={numOverlap}
          onChange={(v) => onChange({ numOverlap: v })}
        />
      )}

      {showPrecision && (
        <div className="sep-param-row">
          <label title={t18(MSST_FP16_TIP, lang)}>
            {t18({ zh: "推理精度", en: "Precision", ja: "推論精度" }, lang)}
          </label>
          <select
            value={(params.precision as string) ?? MSST_DEFAULT_PRECISION[arch]}
            onChange={(e) => onChange({ precision: e.target.value })}
          >
            <option value="fp32">fp32</option>
            <option value="fp16">fp16</option>
          </select>
        </div>
      )}

      {/* uvr_vr also batches (dynamic-batch windows, Rust default 4); mdx_net does not. */}
      {(isSpectral || arch === "uvr_vr") && (
        <div className="sep-param-row">
          <label title={t18({ zh: "批大小：一次喂多个 chunk 给 GPU（需重导出的新模型；旧模型自动按 1 跑）。显存不够会报错，调低即可", en: "Batch size — feed several chunks to the GPU at once (needs a re-exported model; old models run at 1). Lower it on out-of-memory", ja: "バッチサイズ — 複数チャンクを同時に GPU へ（再エクスポート済みモデルが必要。旧モデルは 1 で動作）。VRAM 不足なら下げてください" }, lang)}>
            {t18({ zh: "批大小", en: "Batch", ja: "バッチ" }, lang)}
          </label>
          <input type="number" min={1} max={16} step={1} value={(params.batch as number) ?? (arch === "uvr_vr" ? 4 : 1)}
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

      {arch === "uvr_vr" && (
        <ParamSlider
          label={t18({ zh: "强度", en: "Aggression", ja: "強度" }, lang)}
          title={t18({ zh: "主输出提取强度，UVR同款参数；人声/伴奏类默认5，数值过大可能发闷", en: "Extraction intensity for the primary stem — same parameter as UVR; default 5 for vocal/instrumental models, too high can sound muffled", ja: "主出力の抽出強度、UVRと同じパラメータ。ボーカル/伴奏系は既定5、上げすぎると籠った音に" }, lang)}
          min={-100} max={100} step={5} value={aggression}
          onChange={(v) => onChange({ aggression: v })}
        />
      )}

      {arch === "uvr_vr" && (
        <div className="sep-param-row">
          <label title={t18({ zh: "UVR的merge_artifacts后处理：抑制主输出中的残留伪影，个别歌曲有奇效，默认关", en: "UVR's merge_artifacts post-processing: suppresses residual artifacts in the primary stem — works wonders on some songs, off by default", ja: "UVRのmerge_artifacts後処理：主出力の残留アーティファクトを抑制。曲によっては劇的に効く、既定はオフ" }, lang)}>
            {t18({ zh: "后处理", en: "Post-process", ja: "後処理" }, lang)}
          </label>
          <input type="checkbox" checked={postProcess}
            onChange={(e) => onChange({ postProcess: e.target.checked })} />
        </div>
      )}

      {arch === "uvr_vr" && postProcess && (
        <div className="sep-param-row">
          <label>{t18({ zh: "阈值", en: "Threshold", ja: "しきい値" }, lang)}</label>
          <select
            value={String((params.postProcessThreshold as number) ?? 0.2)}
            onChange={(e) => onChange({ postProcessThreshold: parseFloat(e.target.value) })}
          >
            <option value="0.1">0.1</option>
            <option value="0.2">0.2</option>
            <option value="0.3">0.3</option>
          </select>
        </div>
      )}

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
