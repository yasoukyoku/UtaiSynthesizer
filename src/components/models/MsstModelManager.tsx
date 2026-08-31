import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import { useMsstModelStore, setupDownloadListener } from "../../store/msst-models";
import { useAppStore } from "../../store/app";
import {
  MSST_CATALOG,
  ALL_CATEGORIES,
  CATEGORY_LABELS,
  ARCHITECTURE_LABELS,
  MSST_DEFAULT_PRECISION,
  MSST_FP16_ARCHS,
  MSST_FP16_TIP,
  ghRouteOrder,
  t18,
  type MsstArchitecture,
  type MsstCatalogEntry,
  type MsstCategory,
  type MsstPrecision,
} from "../../lib/models/msst-catalog";
import { useFloatingPanel } from "../../lib/useFloatingPanel";
import { PanelResizeHandles } from "../common/PanelResizeHandles";
import { backendErrorMessage, isBusyError, isCancelError } from "../../lib/backendError";
import { maybeShowErrorModal } from "../../lib/errorDisplay";
import { logToBackend } from "../../lib/log";
import { VOICE_STRINGS } from "../workflow/nodes/VoiceModelPicker";
import {
  useVoiceModelStore,
  voiceVersionBadge,
  voiceSpeakerOptions,
  formatSampleRateKhz,
  vocoderFormatMatches,
  vocoderFormatLabel,
  type VoiceModelEntry,
  type VoiceType,
} from "../../store/voice-models";
import { runRangeTest, runRangeTestBatch, collectRangeTestTargets, midiName, deriveCautionZones, SCAN_VERSION, type SpeakerRangeRecord } from "../../lib/vocal/rangeTest";
import { targetRange } from "../../lib/vocal/rangeBounds";
import { preview } from "../common/previewPlayer";
import { RangeBoundsEditor } from "../vocal/RangeBoundsEditor";
import { readFile } from "@tauri-apps/plugin-fs";
import "./MsstModelManager.css";

type TopTab = "separation" | "voice" | "tools";

export function MsstModelManager({ onClose }: { onClose: () => void }) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const {
    installed, downloading, error,
    fetchInstalled, fetchModelsDir, modelsDir,
    clearError, deleteModel, downloadEntry, convertPrecision,
  } = useMsstModelStore();

  const { style: panelStyle, startDrag, startResize } = useFloatingPanel({
    storageKey: "utai.msstManagerRect",
    initial: () => ({ x: 100, y: 96, w: 440, h: Math.round(window.innerHeight * 0.72) }),
    minW: 380,
    minH: 320,
  });

  const [topTab, setTopTab] = useState<TopTab>("separation");
  const [category, setCategory] = useState<MsstCategory>("vocals");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  // Download-time precision choice per catalog entry (roformers only); absent = arch default.
  const [dlPrecision, setDlPrecision] = useState<Record<string, MsstPrecision>>({});

  useEffect(() => {
    fetchModelsDir();
    fetchInstalled();
    setupDownloadListener();
  }, [fetchModelsDir, fetchInstalled]);

  const installedFilenames = new Set(installed.map((m) => m.filename));
  const filtered = MSST_CATALOG.filter((m) => m.category === category);

  const handleDownload = useCallback(async (entry: MsstCatalogEntry) => {
    // Only the fp16-verified roformers get a precision choice; other archs download as before.
    const precision = MSST_FP16_ARCHS.has(entry.architecture)
      ? (dlPrecision[entry.id] ?? MSST_DEFAULT_PRECISION[entry.architecture])
      : undefined;
    await downloadEntry(entry, precision);
  }, [downloadEntry, dlPrecision]);

  const handleMsstImport = useCallback(async () => {
    const path = await open({
      title: lang === "zh" ? "选择 MSST 模型文件" : "Select MSST Model File",
      filters: [{ name: "Model", extensions: ["ckpt", "th", "pth", "onnx"] }],
    });
    if (path) await useMsstModelStore.getState().importLocal(path as string);
  }, [lang]);

  const handleDelete = useCallback(async (filename: string) => { await deleteModel(filename); setConfirmDelete(null); }, [deleteModel]);

  return (
    <aside className="msst-model-manager" style={panelStyle}>
      <div className="panel-header" onMouseDown={startDrag}>
        <span className="panel-title">{lang === "zh" ? "资源管理" : lang === "ja" ? "リソース管理" : "Resource Manager"}</span>
        <button className="panel-close" onClick={onClose}>X</button>
      </div>
      <PanelResizeHandles start={startResize} />

      {error && <div className="msst-error" onClick={clearError}>{backendErrorMessage(error) ?? error}</div>}

      <div className="rm-top-tabs">
        <button className={topTab === "separation" ? "active" : ""} onClick={() => setTopTab("separation")}>
          {lang === "zh" ? "音频分离" : lang === "ja" ? "音声分離" : "Separation"}
        </button>
        <button className={topTab === "voice" ? "active" : ""} onClick={() => setTopTab("voice")}>
          {lang === "zh" ? "声音模型" : lang === "ja" ? "ボイスモデル" : "Voice Models"}
        </button>
        <button className={topTab === "tools" ? "active" : ""} onClick={() => setTopTab("tools")}>
          {lang === "zh" ? "工具模型" : lang === "ja" ? "ツールモデル" : "Tool Models"}
        </button>
      </div>

      {topTab === "voice" && <VoiceModelsTab lang={lang} />}

      {topTab === "tools" && <GameEngineTab lang={lang} />}

      {topTab === "separation" && (
        <>
          <div className="msst-filter">
            {ALL_CATEGORIES.map((cat) => (
              <button key={cat} className={category === cat ? "active" : ""} onClick={() => setCategory(cat)}>
                {t18(CATEGORY_LABELS[cat], lang)}
              </button>
            ))}
          </div>

          <div className="msst-model-list">
            {filtered.map((entry) => {
              const isInstalled = installedFilenames.has(entry.filename);
              const dl = downloading[entry.filename];
              const isDownloading = !!dl;
              const fp16Capable = MSST_FP16_ARCHS.has(entry.architecture);
              const chosenPrecision = dlPrecision[entry.id] ?? MSST_DEFAULT_PRECISION[entry.architecture];
              return (
                <div key={entry.id} className={`msst-model-card-wrap ${isInstalled ? "installed" : ""}`}>
                  {!isInstalled && !isDownloading && (
                    <div className="msst-model-card-slide">
                      <button className="primary" onClick={() => handleDownload(entry)} title={lang === "zh" ? "下载" : "Download"}>
                        ↓
                      </button>
                    </div>
                  )}
                  <div className="msst-model-card">
                    <div className="model-card-header">
                      <span className="model-card-name">{t18(entry.name, lang)}</span>
                      <span className="model-card-arch">
                        {ARCHITECTURE_LABELS[entry.architecture]}
                        {entry.source === "community" && <span className="model-card-community"> *</span>}
                      </span>
                    </div>
                    <p className="model-card-desc">{t18(entry.description, lang)}</p>
                    <div className="model-card-meta">
                      <span className="model-card-stems">{entry.stems.join(" / ")}</span>
                      {entry.sdrScore && <span className="model-card-sdr">SDR {entry.sdrScore}</span>}
                      <span className="model-card-size">{formatSize(entry.fileSize)}</span>
                    </div>
                    {!isInstalled && !isDownloading && fp16Capable && (
                      <div className="model-card-precision">
                        <span className="model-precision-label">
                          {t18({ zh: "下载精度", en: "Precision", ja: "精度" }, lang)}
                        </span>
                        <div className="model-precision-seg" title={t18(MSST_FP16_TIP, lang)}>
                          {(["fp32", "fp16"] as const).map((p) => (
                            <button
                              key={p}
                              className={chosenPrecision === p ? "active" : ""}
                              onClick={() => setDlPrecision((s) => ({ ...s, [entry.id]: p }))}
                            >
                              {p}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {isDownloading && <DownloadBar dl={dl} lang={lang} />}
                    {isInstalled && (
                      <div className="model-card-actions">
                        <span className="model-status-installed">{lang === "zh" ? "已安装" : "Installed"}</span>
                        {confirmDelete === entry.filename ? (
                          <div className="model-confirm-delete">
                            <button className="danger" onClick={() => handleDelete(entry.filename)}>{lang === "zh" ? "确认" : "OK"}</button>
                            <button onClick={() => setConfirmDelete(null)}>{lang === "zh" ? "取消" : "Cancel"}</button>
                          </div>
                        ) : (
                          <button className="model-delete-btn" onClick={() => setConfirmDelete(entry.filename)}>{lang === "zh" ? "删除" : "Delete"}</button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
            {filtered.length === 0 && <p className="msst-empty">{lang === "zh" ? "此分类暂无模型" : "No models in this category"}</p>}
          </div>

          <div className="msst-installed-section">
            <div className="msst-installed-header">
              <span>{lang === "zh" ? "已安装文件" : "Installed Files"} <span className="mono">{modelsDir}</span></span>
              <button className="msst-import-btn" onClick={handleMsstImport}>{lang === "zh" ? "导入" : "Import"}</button>
            </div>
            {installed.length === 0 ? (
              <p className="msst-empty">{lang === "zh" ? "暂无模型" : "No models installed"}</p>
            ) : (
              <div className="msst-installed-list">
                {installed.map((m) => {
                  const isConverting = downloading[m.filename]?.stage === "converting";
                  // S66: conversions are single-flight app-wide (Rust convert slot is the
                  // authority) — gray every other convert button while one runs.
                  const anyConverting = Object.values(downloading).some((d) => d.stage === "converting");
                  // Catalog arch wins: hash-named official weights (demucs .th) defeat Rust's
                  // filename detection, which reports "unknown" for them.
                  const arch = (MSST_CATALOG.find((e) => e.filename === m.filename)?.architecture
                    ?? m.architecture) as MsstArchitecture;
                  const archHint = arch === ("unknown" as string) ? undefined : arch;
                  const fp16Capable = MSST_FP16_ARCHS.has(arch);
                  return (
                    <div key={m.filename} className="msst-installed-item">
                      <span className="msst-installed-name" title={m.filename}>{m.filename}</span>
                      <span className="msst-installed-meta">
                        {m.has_onnx && <span className="msst-onnx-ok">fp32</span>}
                        {m.has_fp16 && <span className="msst-onnx-ok">fp16</span>}
                        {isConverting ? (
                          <span className="msst-converting">...</span>
                        ) : !m.has_onnx && !m.has_fp16 ? (
                          <button className="msst-convert-btn" disabled={anyConverting} onClick={() => convertPrecision(m.filename, undefined, archHint)}>Convert</button>
                        ) : fp16Capable && !m.has_fp16 ? (
                          <button
                            className="msst-convert-btn"
                            disabled={anyConverting}
                            title={t18(MSST_FP16_TIP, lang)}
                            onClick={() => convertPrecision(m.filename, "fp16", archHint)}
                          >
                            {t18({ zh: "补转 fp16", en: "Convert to fp16", ja: "fp16に変換" }, lang)}
                          </button>
                        ) : fp16Capable && !m.has_onnx ? (
                          <>
                            {/* S68c review: fp16-ONLY installs (the MelBand download default) are the
                                LARGEST victim group of the pre-protection fp16 recipe — they must get
                                the cure here too. "重转 fp16" without an fp32 on disk goes through the
                                full ckpt export (slower) and lands back in fp16-only shape. */}
                            {m.has_fp16 && !m.fp16_recipe_ok && (
                              <button
                                className="msst-convert-btn"
                                disabled={anyConverting}
                                title={t18({ zh: "用当前转换配方从 ckpt 重新生成 fp16（较慢；旧版转换的 fp16 在部分显卡上有数值问题）", en: "Regenerate fp16 from the ckpt with the current recipe (slower; older fp16 conversions can misbehave numerically on some GPUs)", ja: "現在のレシピで ckpt から fp16 を再生成（時間がかかります。旧版の fp16 は一部の GPU で数値問題があります）" }, lang)}
                                onClick={() => convertPrecision(m.filename, "fp16", archHint)}
                              >
                                {t18({ zh: "重转 fp16", en: "Redo fp16", ja: "fp16再変換" }, lang)}
                              </button>
                            )}
                            <button
                              className="msst-convert-btn"
                              disabled={anyConverting}
                              title={t18({ zh: "从 ckpt 完整导出 fp32（较慢）", en: "Full fp32 export from the ckpt (slower)", ja: "ckpt から fp32 を完全エクスポート（時間がかかります）" }, lang)}
                              onClick={() => convertPrecision(m.filename, "fp32", archHint)}
                            >
                              {t18({ zh: "补转 fp32", en: "Convert to fp32", ja: "fp32に変換" }, lang)}
                            </button>
                          </>
                        ) : fp16Capable && !m.fp16_recipe_ok ? (
                          // S68c: both variants installed but the fp16 lacks the current-recipe stamp
                          // (`<stem>.fp16.recipe`) → offer a REFRESH (cheap, from the fp32 on disk).
                          // Older builds converted roformer fp16 without the fp32 norm-stats
                          // protection — those files can NaN on true-fp16 GPU kernels. A successful
                          // reconvert stamps the recipe and this button disappears (§user).
                          <button
                            className="msst-convert-btn"
                            disabled={anyConverting}
                            title={t18({ zh: "用当前转换配方重新生成 fp16（旧版转换的 fp16 在部分显卡上有数值问题）", en: "Regenerate fp16 with the current recipe (older fp16 conversions can misbehave numerically on some GPUs)", ja: "現在のレシピで fp16 を再生成（旧版で変換した fp16 は一部の GPU で数値問題があります）" }, lang)}
                            onClick={() => convertPrecision(m.filename, "fp16", archHint)}
                          >
                            {t18({ zh: "重转 fp16", en: "Redo fp16", ja: "fp16再変換" }, lang)}
                          </button>
                        ) : null}
                        {" "}{formatSize(m.size)}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}

// ─── Import Dialog ──────────────────────────────────────────

interface ImportDialogProps {
  lang: string;
  voiceType: VoiceType;
  onClose: () => void;
  onDone: () => void;
}

function ImportDialog({ lang, voiceType, onClose, onDone }: ImportDialogProps) {
  const [modelPath, setModelPath] = useState("");
  const [indexPath, setIndexPath] = useState("");
  const [diffusionPath, setDiffusionPath] = useState("");
  const [diffusionConfigPath, setDiffusionConfigPath] = useState("");
  const [avatarPath, setAvatarPath] = useState("");
  const [vocoderConfigPath, setVocoderConfigPath] = useState("");
  const [modelName, setModelName] = useState("");
  const [importing, setImporting] = useState(false);
  const [err, setErr] = useState("");
  const isVocoder = voiceType === "vocoder";

  const browse = useCallback(async (title: string, exts: string[]) => {
    // "*" filter: community vocoder checkpoints are often extensionless
    // (so-vits pretrain names the file just "model")
    const filters = exts.includes("*")
      ? [{ name: "File", extensions: exts.filter((e) => e !== "*") }, { name: "All", extensions: ["*"] }]
      : [{ name: "File", extensions: exts }];
    const path = await open({ title, filters });
    return path ? (path as string) : "";
  }, []);

  const handleBrowseModel = useCallback(async () => {
    const p = isVocoder
      ? await browse(
          lang === "zh" ? "选择声码器权重 (.ckpt / .pt / .onnx)" : "Select vocoder checkpoint (.ckpt / .pt / .onnx)",
          ["ckpt", "pt", "onnx", "*"],
        )
      : await browse(lang === "zh" ? "选择模型文件 (.pth)" : "Select model file (.pth)", ["pth", "onnx"]);
    if (p) {
      setModelPath(p);
      const filename = p.split(/[/\\]/).pop() ?? "";
      setModelName(filename.replace(/\.(pth|onnx|ckpt|pt)$/i, ""));
    }
  }, [browse, lang, isVocoder]);

  const handleBrowseVocoderConfig = useCallback(async () => {
    const p = await browse(lang === "zh" ? "选择声码器配置 (config.json)" : "Select vocoder config (config.json)", ["json"]);
    if (p) setVocoderConfigPath(p);
  }, [browse, lang]);

  const handleBrowseIndex = useCallback(async () => {
    // RVC: FAISS .index / pre-extracted .npy. SoVITS: cluster kmeans .pt / feature-retrieval
    // .pkl / pre-converted .npy — the backend routes by model type + file extension.
    const isRvcPick = voiceType === "rvc";
    const title = isRvcPick
      ? (lang === "zh" ? "选择索引文件 (.index)" : "Select index file (.index)")
      : (lang === "zh" ? "选择聚类/检索模型 (.pt / .pkl)" : "Select cluster/retrieval model (.pt / .pkl)");
    const exts = isRvcPick ? ["index", "npy"] : ["pt", "pkl", "pickle", "npy"];
    const p = await browse(title, exts);
    if (p) setIndexPath(p);
  }, [browse, lang, voiceType]);

  // SoVITS only: the separate shallow-diffusion model pair (.pt + config .yaml). The .yaml is
  // optional here — export_diffusion.py auto-resolves it next to the .pt (same stem → unique
  // .yaml in dir → config.yaml) and errors in Chinese when ambiguous.
  const handleBrowseDiffusion = useCallback(async () => {
    const p = await browse(lang === "zh" ? "选择扩散模型 (.pt)" : "Select diffusion model (.pt)", ["pt"]);
    if (p) setDiffusionPath(p);
  }, [browse, lang]);

  const handleBrowseDiffusionConfig = useCallback(async () => {
    const p = await browse(lang === "zh" ? "选择扩散配置 (.yaml)" : "Select diffusion config (.yaml)", ["yaml", "yml"]);
    if (p) setDiffusionConfigPath(p);
  }, [browse, lang]);

  const handleBrowseAvatar = useCallback(async () => {
    const p = await browse(lang === "zh" ? "选择角色头图" : "Select character avatar", ["png", "jpg", "jpeg", "bmp", "webp"]);
    if (p) setAvatarPath(p);
  }, [browse, lang]);

  const handleImport = useCallback(async () => {
    if (!modelPath || !modelName) return;
    // S60 audit: a running range test stamps THIS name's sidecar at its TAIL (after the render
    // guard released) — a REPLACE import racing that window would get the OLD model's record
    // stamped onto the NEW files. Block while the test runs.
    if (useVoiceModelStore.getState().rangeTesting[modelName] !== undefined) {
      setErr(t18({ zh: "该模型正在音域测试中，请稍后再导入", en: "This model's range test is running — import later", ja: "このモデルは音域テスト中です。後で取り込んでください" }, lang));
      return;
    }
    setImporting(true);
    setErr("");
    try {
      const outcome = await invoke<{ entry: { name: string; path: string } | null; warnings: string[] }>("import_model", {
        name: modelName,
        path: modelPath,
        modelType: voiceType,
        indexPath: indexPath || null,
        diffusionPath: diffusionPath || null,
        diffusionConfigPath: diffusionConfigPath || null,
        avatarPath: avatarPath || null,
        vocoderConfigPath: vocoderConfigPath || null,
      });
      for (const w of outcome?.warnings ?? []) {
        // Import warnings arrive as "WARN_X: detail" CODE strings — localize known ones.
        useAppStore.getState().showToast(backendErrorMessage(w) ?? w, "info");
      }
      // S60-2: fresh import → background range test (default speaker; the record died with any
      // REPLACEd sidecar). Fire-and-forget — failures/busy toast from rangeTest itself.
      if ((voiceType === "rvc" || voiceType === "sovits") && outcome?.entry) {
        void runRangeTest(outcome.entry.name, voiceType, outcome.entry.path);
      }
      onDone();
    } catch (e) {
      const msg = String(e);
      setErr(backendErrorMessage(msg) ?? msg);
    }
    setImporting(false);
  }, [modelPath, modelName, voiceType, indexPath, diffusionPath, diffusionConfigPath, avatarPath, vocoderConfigPath, onDone, lang]);

  const isRvc = voiceType === "rvc";
  const Z = (key: string) => {
    const map: Record<string, Record<string, string>> = {
      title: isVocoder
        ? { zh: "导入声码器", en: "Import Vocoder", ja: "ボコーダー取り込み" }
        : { zh: `导入 ${voiceType.toUpperCase()} 模型`, en: `Import ${voiceType.toUpperCase()} Model`, ja: `${voiceType.toUpperCase()} モデル取り込み` },
      model: isVocoder
        ? { zh: "声码器权重 (.ckpt / .pt / .onnx，社区包内常为无后缀的 model 文件)", en: "Vocoder checkpoint (.ckpt / .pt / .onnx; community zips often name it just \"model\")", ja: "ボコーダー重み (.ckpt / .pt / .onnx。コミュニティ配布では拡張子なしの model の場合あり)" }
        : { zh: "模型文件 (.pth)", en: "Model file (.pth)", ja: "モデルファイル (.pth)" },
      vocoderCfg: { zh: "声码器配置 (config.json)  — 可留空自动查找", en: "Vocoder config (config.json) — blank = auto-detect", ja: "ボコーダー設定 (config.json) — 空欄で自動検出" },
      vocoderNote: {
        zh: "支持经典 NSF-HiFiGAN（如 openvpi 2022.12/2024.02 社区声码器及其微调产物）；PC-NSF（mini_nsf）暂不支持。导入后在 SoVITS 推理节点的「声码器」下拉中选用。",
        en: "Classic NSF-HiFiGAN only (openvpi 2022.12/2024.02 community vocoders and their fine-tunes); PC-NSF (mini_nsf) is not supported yet. After import, pick it in the SoVITS node's Vocoder dropdown.",
        ja: "クラシック NSF-HiFiGAN のみ対応（openvpi 2022.12/2024.02 コミュニティボコーダーとその微調整版）。PC-NSF（mini_nsf）は未対応。取り込み後、SoVITS ノードの「ボコーダー」で選択できます。",
      },
      index: { zh: "索引文件 (.index)  — 可选", en: "Index file (.index) — optional", ja: "インデックス (.index) — 任意" },
      cluster: { zh: "聚类/检索模型 (.pt / .pkl)  — 可选", en: "Cluster/retrieval model (.pt / .pkl) — optional", ja: "クラスタ/検索モデル (.pt / .pkl) — 任意" },
      diffusion: { zh: "扩散模型 (.pt)  — 可选，启用浅扩散", en: "Diffusion model (.pt) — optional, enables shallow diffusion", ja: "拡散モデル (.pt) — 任意、浅い拡散を有効化" },
      diffusionCfg: { zh: "扩散配置 (.yaml)  — 可留空自动查找", en: "Diffusion config (.yaml) — blank = auto-detect", ja: "拡散設定 (.yaml) — 空欄で自動検出" },
      avatar: { zh: "角色头图 — 可选", en: "Character avatar — optional", ja: "キャラクター画像 — 任意" },
      name: { zh: "模型名称", en: "Model name", ja: "モデル名" },
      import: { zh: "导入", en: "Import", ja: "取り込み" },
      cancel: { zh: "取消", en: "Cancel", ja: "キャンセル" },
      importing: { zh: "导入并转换中...", en: "Importing & converting...", ja: "取り込み・変換中..." },
      browseBtn: { zh: "浏览", en: "Browse", ja: "参照" },
      required: { zh: "必填", en: "Required", ja: "必須" },
    };
    return map[key]?.[lang] ?? map[key]?.en ?? key;
  };

  return (
    <div className="rm-import-overlay" onClick={onClose}>
      <div className="rm-import-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="rm-import-title">{Z("title")}</div>

        {err && <div className="rm-import-error">{err}</div>}

        <div className="rm-import-field">
          <label>{Z("model")} <span className="rm-required">{Z("required")}</span></label>
          <div className="rm-import-row">
            <input type="text" readOnly value={modelPath} placeholder="..." className="rm-import-path" />
            <button onClick={handleBrowseModel}>{Z("browseBtn")}</button>
          </div>
        </div>

        {isVocoder && (
          <>
            <div className="rm-import-field">
              <label>{Z("vocoderCfg")}</label>
              <div className="rm-import-row">
                <input type="text" readOnly value={vocoderConfigPath} placeholder="..." className="rm-import-path" />
                <button onClick={handleBrowseVocoderConfig}>{Z("browseBtn")}</button>
              </div>
            </div>
            <p className="rm-voice-hint">{Z("vocoderNote")}</p>
          </>
        )}

        {!isVocoder && (
        <div className="rm-import-field">
          <label>{isRvc ? Z("index") : Z("cluster")}</label>
          <div className="rm-import-row">
            <input type="text" readOnly value={indexPath} placeholder="..." className="rm-import-path" />
            <button onClick={handleBrowseIndex}>{Z("browseBtn")}</button>
          </div>
        </div>
        )}

        {!isRvc && !isVocoder && (
          <>
            <div className="rm-import-field">
              <label>{Z("diffusion")}</label>
              <div className="rm-import-row">
                <input type="text" readOnly value={diffusionPath} placeholder="..." className="rm-import-path" />
                <button onClick={handleBrowseDiffusion}>{Z("browseBtn")}</button>
              </div>
            </div>
            {diffusionPath && (
              <div className="rm-import-field">
                <label>{Z("diffusionCfg")}</label>
                <div className="rm-import-row">
                  <input type="text" readOnly value={diffusionConfigPath} placeholder="..." className="rm-import-path" />
                  <button onClick={handleBrowseDiffusionConfig}>{Z("browseBtn")}</button>
                </div>
              </div>
            )}
          </>
        )}

        {!isVocoder && (
        <div className="rm-import-field">
          <label>{Z("avatar")}</label>
          <div className="rm-import-row">
            <input type="text" readOnly value={avatarPath} placeholder="..." className="rm-import-path" />
            <button onClick={handleBrowseAvatar}>{Z("browseBtn")}</button>
          </div>
        </div>
        )}

        <div className="rm-import-field">
          <label>{Z("name")}</label>
          <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} className="rm-import-name" />
        </div>

        <div className="rm-import-actions">
          <button onClick={onClose} disabled={importing}>{Z("cancel")}</button>
          <button className="primary" onClick={handleImport} disabled={importing || !modelPath || !modelName}>
            {importing ? Z("importing") : Z("import")}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Sub-components ─────────────────────────────────────────

function DownloadBar({ dl, lang }: { dl: { downloaded: number; total: number; stage: string }; lang: string }) {
  return (
    <div className="model-download-progress">
      {dl.stage === "converting" ? (
        <>
          <div className="model-download-bar model-convert-bar" style={{ width: "100%" }} />
          <span className="model-download-text">{lang === "zh" ? "转换为 ONNX..." : "Converting to ONNX..."}</span>
        </>
      ) : (
        <>
          <div className="model-download-bar" style={{ width: dl.total > 0 ? `${(dl.downloaded / dl.total) * 100}%` : "0%" }} />
          <span className="model-download-text">{formatSize(dl.downloaded)} / {dl.total > 0 ? formatSize(dl.total) : "..."}</span>
        </>
      )}
    </div>
  );
}

// ─── S60: Tool models tab — the GAME 人声→MIDI engine (downloaded on demand: CC BY-NC-SA
// weights must not ship in the bundle; GitHub release primary → HF mirror fallback in Rust) ───

interface GameDlProgress {
  stage: string; // download | extract | done
  downloaded: number;
  total: number;
}

function GameEngineTab({ lang }: { lang: string }) {
  const [installed, setInstalled] = useState<boolean | null>(null);
  const [dl, setDl] = useState<GameDlProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const showToast = useAppStore((s) => s.showToast);
  const unlistenRef = useRef<UnlistenFn | null>(null);

  const refresh = useCallback(async () => {
    try {
      const st = await invoke<{ installed: boolean; downloading: boolean }>("midi_extract_status");
      setInstalled(st.installed);
      // a download started before an unmount is still running (Rust single-flight) —
      // restore the busy view instead of offering a second download (audit S60)
      if (st.downloading) setBusy(true);
    } catch {
      setInstalled(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    let disposed = false;
    void listen<GameDlProgress>("game-download-progress", (e) => {
      setDl(e.payload);
      // remounted mid-download: no pending invoke here, so the terminal event drives the state
      if (e.payload.stage === "done") {
        setBusy(false);
        setDl(null);
        void refresh();
      } else {
        setBusy(true);
      }
    }).then((un) => {
      if (disposed) un();
      else unlistenRef.current = un;
    });
    return () => {
      disposed = true;
      unlistenRef.current?.();
      unlistenRef.current = null;
    };
  }, [refresh]);

  const handleDownload = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setDl({ stage: "download", downloaded: 0, total: 0 });
    try {
      // ghRoutes → Rust gh_routes: the full ordered GH failover chain (chosen proxy →
      // direct → other presets, S66); the backend interleaves it with its static rotation.
      const { ghMirror, ghPresets } = useMsstModelStore.getState();
      const st = await invoke<{ installed: boolean }>("download_game_package", {
        ghRoutes: ghRouteOrder(ghMirror, ghPresets),
      });
      setInstalled(st.installed);
      showToast(t18({ zh: "GAME 引擎已安装", en: "GAME engine installed", ja: "GAME エンジンをインストールしました" }, lang), "success");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("GAME_DL_BUSY")) return; // another flight is running — its events drive the UI
      if (isCancelError(msg)) return; // user cancelled the download — silent settle
      const base = msg.includes("GAME_DL_EXTRACT")
        ? t18({ zh: "解压安装失败", en: "Extraction failed", ja: "展開に失敗しました" }, lang)
        : t18({ zh: "下载失败", en: "Download failed", ja: "ダウンロードに失敗しました" }, lang);
      showToast(`${base}: ${backendErrorMessage(msg) ?? msg}`, "error");
    } finally {
      setBusy(false);
      setDl(null);
    }
  }, [busy, lang, showToast]);

  const handleDelete = useCallback(async () => {
    setConfirmDelete(false);
    try {
      const st = await invoke<{ installed: boolean }>("delete_game_package");
      setInstalled(st.installed);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const base = t18({ zh: "删除失败", en: "Delete failed", ja: "削除に失敗しました" }, lang);
      showToast(msg.includes("GAME_DELETE_FAILED") ? `${base}: ${msg}` : msg, "error");
    }
  }, [lang, showToast]);

  const stageText = (p: GameDlProgress): string => {
    if (p.stage === "extract") return t18({ zh: "解压安装中...", en: "Extracting...", ja: "展開中..." }, lang);
    return `${formatSize(p.downloaded)} / ${p.total > 0 ? formatSize(p.total) : "..."}`;
  };

  return (
    <div className="msst-model-list">
      <div className={`msst-model-card-wrap ${installed ? "installed" : ""}`}>
        {installed === false && !busy && (
          <div className="msst-model-card-slide">
            <button className="primary" onClick={handleDownload} title={lang === "zh" ? "下载" : lang === "ja" ? "ダウンロード" : "Download"}>↓</button>
          </div>
        )}
        <div className="msst-model-card">
          <div className="model-card-header">
            <span className="model-card-name">GAME · {t18({ zh: "人声转 MIDI", en: "Vocal-to-MIDI", ja: "歌声→MIDI" }, lang)}</span>
            <span className="model-card-arch">openvpi · 1.0.3 medium</span>
          </div>
          <p className="model-card-desc">
            {t18({
              zh: "从人声干声/分离声提取音符（右键子轨道 →「提取 MIDI」）。识别为无歌词音符，自动填入占位词供改词翻唱。",
              en: "Transcribes vocal stems into notes (right-click a sub-lane → \"Extract MIDI\"). Notes carry no lyrics; placeholder lyrics are filled in for re-lyric covers.",
              ja: "ボーカルステムからノートを抽出します（サブレーン右クリック →「MIDI 抽出」）。歌詞なしのノートとして認識され、置き換え用のプレースホルダー歌詞が入ります。",
            }, lang)}
          </p>
          <div className="model-card-meta">
            <span className="model-card-stems">en / ja / yue / zh</span>
            <span className="model-card-size">{formatSize(179775226)}</span>
          </div>
          <p className="model-card-desc">
            {t18({
              zh: "模型权重按 CC BY-NC-SA 4.0 由 openvpi 发布（代码 MIT），因此不随本体分发、需在此下载。",
              en: "Weights are released by openvpi under CC BY-NC-SA 4.0 (code MIT), so they are downloaded here instead of shipping with the app.",
              ja: "モデル重みは openvpi が CC BY-NC-SA 4.0 で公開しています（コードは MIT）。そのためアプリには同梱されず、ここでダウンロードします。",
            }, lang)}
          </p>
          {busy && dl && (
            <div className="model-download-progress">
              <div
                className={`model-download-bar ${dl.stage !== "download" ? "model-convert-bar" : ""}`}
                style={{ width: dl.stage !== "download" ? "100%" : dl.total > 0 ? `${(dl.downloaded / dl.total) * 100}%` : "0%" }}
              />
              <span className="model-download-text">{stageText(dl)}</span>
            </div>
          )}
          {installed && (
            <div className="model-card-actions">
              <span className="model-status-installed">{lang === "zh" ? "已安装" : lang === "ja" ? "インストール済み" : "Installed"}</span>
              {confirmDelete ? (
                <div className="model-confirm-delete">
                  <button className="danger" onClick={handleDelete}>{lang === "zh" ? "确认" : "OK"}</button>
                  <button onClick={() => setConfirmDelete(false)}>{lang === "zh" ? "取消" : lang === "ja" ? "キャンセル" : "Cancel"}</button>
                </div>
              ) : (
                <button className="model-delete-btn" onClick={() => setConfirmDelete(true)}>{lang === "zh" ? "删除" : lang === "ja" ? "削除" : "Delete"}</button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function VoiceAvatar({ path, name, onSet }: { path: string | null; name: string; onSet: () => void }) {
  if (path) {
    return (
      <div className="rm-voice-avatar" onClick={onSet} title={name}>
        <img src={convertFileSrc(path)} alt={name} />
      </div>
    );
  }
  return (
    <div className="rm-voice-avatar rm-voice-avatar-empty" onClick={onSet} title="Set avatar">
      <span>{name.charAt(0).toUpperCase()}</span>
    </div>
  );
}

/** Facts about the BUILT-IN default vocoder (Rust get_default_vocoder_info):
 *  aux infrastructure, shown as a pinned read-only row in the vocoder tab. */
interface DefaultVocoderInfo {
  present: boolean;
  missing: string[];
  sample_rate: number | null;
  hop_size: number | null;
  num_mels: number | null;
}

function VoiceModelsTab({ lang }: { lang: string }) {
  // S82d: ONE active speaker per model row, shared by the audition button AND the range row.
  // The two rows used to carry their own selects (each with its own state, able to point at
  // DIFFERENT singers = semantic drift + the crowding the user reported); the single selector
  // lives in the model meta line, replacing the static "N speakers" badge.
  const [voiceSpk, setVoiceSpk] = useState<Record<string, number>>({});
  const [voiceType, setVoiceType] = useState<VoiceType>("rvc");
  const [showImport, setShowImport] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  // S146f: 音域边界编辑器展开时,四条滑条会占满整行 —— 试听按钮与它挤在同一行里视觉重叠
  // (用户实机报的)。编辑态提到这一层,让同排的动作按钮能让位。
  const [rangeEditing, setRangeEditing] = useState<string | null>(null);
  // S167c: which model's EXPORT format chooser is open — the chooser owns its row (audition /
  // range row / delete yield), same tab-level pattern as rangeEditing (S146f: 同排按钮要跟着让位).
  const [exportPick, setExportPick] = useState<string | null>(null);
  // Shared store — the SAME list the RVC/SoVITS workflow nodes read (one source of truth).
  const models = useVoiceModelStore((s) => s.models[voiceType]);
  const voiceError = useVoiceModelStore((s) => s.error);
  const { fetchModels, deleteModel, setAvatar, clearError } = useVoiceModelStore();
  // built-in default vocoder facts — refetched on tab entry (cheap disk stat)
  const [defaultVoc, setDefaultVoc] = useState<DefaultVocoderInfo | null>(null);
  useEffect(() => {
    if (voiceType !== "vocoder") return;
    void invoke<DefaultVocoderInfo>("get_default_vocoder_info")
      .then(setDefaultVoc)
      .catch(() => setDefaultVoc(null));
  }, [voiceType]);

  useEffect(() => { void fetchModels(); }, [fetchModels]);

  // S60-4: tab unmount = the audition UI is gone — stop OUR playback (ownership proven
  // against preview.path; a foreign consumer's playback is untouched) and clear the state.
  useEffect(() => () => {
    const a = useVoiceModelStore.getState().auditionState;
    if (a) {
      if (a.phase === "playing" && preview.path === a.path) {
        preview.onEnd = null;
        preview.stop();
      }
      useVoiceModelStore.getState().setAuditionState(null);
    }
  }, []);

  const handleDelete = useCallback(async (name: string) => {
    // S60 audit: a running range test writes this model's sidecar at its tail (and an
    // audition writes a wav beside it — Rust also guards that one); block the delete.
    const vm = useVoiceModelStore.getState();
    if (vm.rangeTesting[name] !== undefined || vm.auditionState?.name === name) {
      useAppStore.getState().showToast(
        t18({ zh: "该模型正在测试/试听中，稍后再删除", en: "This model is being tested/auditioned — delete later", ja: "このモデルはテスト/試聴中です。後で削除してください" }, lang),
        "info",
      );
      setDeleteConfirm(null);
      return;
    }
    // type-scoped: same-name entries across types are standard (rvc+sovits pair
    // + a vocoder named after the singer) — an untyped delete hits the first
    // scan match, i.e. potentially the WRONG model's files (S40 红队 A5)
    await deleteModel(name, voiceType); // errors land in voiceError
    setDeleteConfirm(null);
  }, [deleteModel, voiceType, lang]);

  // S78 batch 7: import a `.zip` model package (the Export counterpart). Rust figures the registry
  // type from the manifest, so afterwards we jump to that tab so the imported model is visible.
  const handleImportPackage = useCallback(async () => {
    const file = await open({
      title: t18({ zh: "导入模型包 (.zip)", en: "Import model package (.zip)", ja: "モデルパッケージを取り込み (.zip)" }, lang),
      filters: [{ name: "Utai Model / Zip", extensions: ["zip"] }],
    });
    if (!file || typeof file !== "string") return;
    try {
      const outcome = await invoke<{ entry: VoiceModelEntry; warnings: string[] }>(
        "import_model_package",
        { packagePath: file },
      );
      await fetchModels();
      const vt = MODEL_TYPE_TO_VOICE[outcome.entry.model_type];
      if (vt) setVoiceType(vt);
      useAppStore.getState().showToast(
        t18({ zh: `已导入 · ${outcome.entry.name}`, en: `Imported · ${outcome.entry.name}`, ja: `取り込み完了 · ${outcome.entry.name}` }, lang),
        "success",
      );
      // Non-fatal import warnings ride the same mapper the ImportDialog uses.
      (outcome.warnings ?? []).forEach((w) =>
        useAppStore.getState().showToast(backendErrorMessage(w) ?? w, "info"),
      );
    } catch (e) {
      // Busy/interlock rejections (CONVERT_BUSY / MODEL_BUSY_AUDITION) are transient → info, not error
      // (the app-wide isBusyError funnel discipline; matches VoiceAuditionButton).
      useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
    }
  }, [lang, fetchModels]);

  return (
    <div className="rm-voice-tab">
      {voiceError && <div className="msst-error" onClick={clearError}>{backendErrorMessage(voiceError) ?? voiceError}</div>}
      <div className="msst-filter">
        <button className={voiceType === "rvc" ? "active" : ""} onClick={() => setVoiceType("rvc")}>RVC</button>
        <button className={voiceType === "sovits" ? "active" : ""} onClick={() => setVoiceType("sovits")}>SoVITS</button>
        <button className={voiceType === "vocoder" ? "active" : ""} onClick={() => setVoiceType("vocoder")}>
          {t18({ zh: "声码器", en: "Vocoder", ja: "ボコーダー" }, lang)}
        </button>
        <div className="rm-filter-spacer" />
        <button
          className="rm-import-top-btn"
          onClick={handleImportPackage}
          title={t18({
            zh: "从 .zip 模型包导入（本软件「导出」生成的包，含索引/聚类/扩散/头像）",
            en: "Import from a .zip model package (produced by Export — includes index / cluster / diffusion / avatar)",
            ja: "「書き出し」で作成した .zip モデルパッケージから取り込み（インデックス/クラスタ/拡散/アバターを含む）",
          }, lang)}
        >
          {t18({ zh: "导入模型包", en: "Import Package", ja: "パッケージ取り込み" }, lang)}
        </button>
        <button className="primary rm-import-top-btn" onClick={() => setShowImport(true)}>
          + {lang === "zh" ? "导入模型" : lang === "ja" ? "モデル取り込み" : "Import Model"}
        </button>
      </div>
      {voiceType !== "vocoder" && <RangeBatchRow lang={lang} />}

      <div className="rm-voice-list">
        {models.length === 0 && voiceType !== "vocoder" && (
          <p className="msst-empty">
            {lang === "zh"
              ? `暂无 ${voiceType.toUpperCase()} 模型`
              : `No ${voiceType.toUpperCase()} models`}
          </p>
        )}
        {voiceType === "vocoder" && (
          // zero-knowledge banner: THE answer to "为什么我的声码器不能用于某个模型"
          <p className="rm-voice-hint">
            {t18({
              zh: "声码器供 SoVITS 浅扩散/增强器使用（在 SoVITS 推理节点里选择）；同一歌手微调的声码器可被其所有 SoVITS 模型共享。仅频谱格式一致（44.1kHz / hop 512 / 128 mel）的声码器可被选用；RVC 模型无外部声码器接口，不适用。",
              en: "Vocoders serve SoVITS shallow diffusion / the enhancer (picked inside the SoVITS node); one singer's fine-tuned vocoder is shared by all their SoVITS models. Only format-matching vocoders (44.1kHz / hop 512 / 128 mel) are selectable; RVC models have no external vocoder interface.",
              ja: "ボコーダーは SoVITS の浅い拡散/エンハンサー用（SoVITS ノード内で選択）。同じ歌手のボコーダーは全 SoVITS モデルで共有可。フォーマット一致（44.1kHz / hop 512 / 128 mel）のもののみ選択可能。RVC には外部ボコーダーの接続点がありません。",
            }, lang)}
          </p>
        )}
        {voiceType === "vocoder" && defaultVoc && (
          // pinned read-only row: what the node dropdown's「默认声码器」IS —
          // the built-in aux vocoder; its facts come from disk (get_default_
          // vocoder_info), so a missing aux install surfaces HERE as a loud
          // chip instead of only erroring at render time
          <div
            className="rm-voice-item rm-voice-item-builtin"
            title={t18({
              zh: "随应用分发的 OpenVPI 社区通用声码器——未选择自定义声码器时，浅扩散/增强器使用它；也是声码器格式类的基准。不可删除。",
              en: "The OpenVPI community general vocoder shipped with the app — shallow diffusion / the enhancer use it unless a custom vocoder is picked; also the format-class reference. Not deletable.",
              ja: "アプリ同梱の OpenVPI コミュニティ汎用ボコーダー。カスタム未選択時に浅い拡散/エンハンサーが使用。フォーマットの基準でもあります。削除不可。",
            }, lang)}
          >
            <div className="rm-voice-item-info">
              <span className="rm-voice-item-name">
                {t18({ zh: "默认声码器", en: "Default vocoder", ja: "既定ボコーダー" }, lang)}
              </span>
              <span className="rm-voice-item-meta">
                <span className="ver-badge">NSF-HiFiGAN</span>
                <span className="msst-onnx-ok">
                  {t18({ zh: "内置", en: "Built-in", ja: "内蔵" }, lang)}
                </span>
                {defaultVoc.present ? (
                  <>
                    <span>
                      {formatSampleRateKhz(defaultVoc.sample_rate ?? 44100)} · hop{" "}
                      {defaultVoc.hop_size ?? "?"} · {defaultVoc.num_mels ?? "?"} mel
                    </span>
                    <span
                      className="msst-onnx-ok"
                      title={t18({
                        zh: "标准格式：可用于所有 SoVITS 模型的浅扩散/增强器",
                        en: "Standard format: usable by every SoVITS model's shallow diffusion / enhancer",
                        ja: "標準フォーマット：全 SoVITS モデルの浅い拡散/エンハンサーで使用可能",
                      }, lang)}
                    >
                      {t18({ zh: "SoVITS 扩散/增强", en: "SoVITS diff/enhance", ja: "SoVITS 拡散/強化" }, lang)}
                    </span>
                  </>
                ) : (
                  <span
                    className="rm-voice-item-warn"
                    title={t18({
                      zh: `缺少文件：${defaultVoc.missing.join("、")}——请到 设置→模型资产 下载推理核心包（或手动放入 data/models/auxiliary/），否则浅扩散/增强器无法运行`,
                      en: `Missing: ${defaultVoc.missing.join(", ")} — download the core inference pack in Settings → Model Assets (or place them in data/models/auxiliary/), or shallow diffusion / the enhancer cannot run`,
                      ja: `欠落ファイル：${defaultVoc.missing.join("、")} — 設定→モデルアセット で推論コアパックをダウンロード（または data/models/auxiliary/ に配置）してください。ないと浅い拡散/エンハンサーは動きません`,
                    }, lang)}
                  >
                    {t18({ zh: "缺失", en: "Missing", ja: "欠落" }, lang)}
                  </span>
                )}
              </span>
            </div>
          </div>
        )}
        {voiceType === "vocoder" && models.length === 0 && (
          <p className="msst-empty">
            {t18({
              zh: "尚无自定义声码器——可在训练页微调后保存，或导入社区声码器（ckpt/onnx）",
              en: "No custom vocoders yet — fine-tune one on the training page, or import a community vocoder (ckpt/onnx)",
              ja: "カスタムボコーダーはまだありません — トレーニングページで微調整して保存するか、コミュニティボコーダー（ckpt/onnx）を取り込めます",
            }, lang)}
          </p>
        )}
        {models.map((m) => {
          const isVocoder = voiceType === "vocoder";
          const ver = isVocoder ? null : voiceVersionBadge(m);
          const speakerOpts = isVocoder ? [] : voiceSpeakerOptions(m);
          // a re-import can shrink the speaker list — a stale stored id falls back to 0
          const stored = voiceSpk[m.name] ?? 0;
          const spk = speakerOpts.some((s) => s.id === stored) ? stored : 0;
          const vocFormatOk = isVocoder ? vocoderFormatMatches(m) : true;
          return (
            <div key={m.name} className="rm-voice-item">
              {!isVocoder && (
              <VoiceAvatar path={m.avatar_path} name={m.name} onSet={async () => {
                const file = await open({ title: lang === "zh" ? "选择角色头图" : "Select avatar", filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "bmp", "webp"] }] });
                if (file) await setAvatar(m.name, file as string);
              }} />
              )}
              <div className="rm-voice-item-info">
                <span className="rm-voice-item-name">{m.name}</span>
                {isVocoder ? (
                  <>
                  {/* S82d r6: vocoder meta is DELIBERATELY two lines — identity badges first,
                      the format string on its own full-width line (§user: ellipsizing it hid
                      the info, and squeezing it into one line wrapped chaotically). Matches
                      the voice rows' three-line rhythm (name / badges / range). */}
                  <span className="rm-voice-item-meta">
                    <span className="ver-badge" title={t18({ zh: "经典 NSF-HiFiGAN 架构", en: "Classic NSF-HiFiGAN architecture", ja: "クラシック NSF-HiFiGAN アーキテクチャ" }, lang)}>
                      NSF-HiFiGAN
                    </span>
                    {vocFormatOk ? (
                      <span className="msst-onnx-ok" title={t18({
                        zh: "标准格式：可用于所有 SoVITS 模型的浅扩散/增强器",
                        en: "Standard format: usable by every SoVITS model's shallow diffusion / enhancer",
                        ja: "標準フォーマット：全 SoVITS モデルの浅い拡散/エンハンサーで使用可能",
                      }, lang)}>
                        {t18({ zh: "SoVITS 扩散/增强", en: "SoVITS diff/enhance", ja: "SoVITS 拡散/強化" }, lang)}
                      </span>
                    ) : (
                      <span className="rm-voice-item-warn" title={t18({
                        zh: "梅尔频谱格式与标准格式（44.1kHz / hop 512 / 128 mel / 40-16000Hz）不一致——不会出现在推理节点的声码器列表中",
                        en: "Mel format differs from the standard (44.1kHz / hop 512 / 128 mel / 40-16000Hz) — will not appear in the node's vocoder list",
                        ja: "メルフォーマットが標準（44.1kHz / hop 512 / 128 mel / 40-16000Hz）と不一致 — ノードのボコーダー一覧に表示されません",
                      }, lang)}>
                        {t18({ zh: "格式不匹配", en: "Format mismatch", ja: "フォーマット不一致" }, lang)}
                      </span>
                    )}
                  </span>
                  <span className="rm-voice-item-meta">
                    <span className="rm-meta-shrink" title={vocoderFormatLabel(m)}>{vocoderFormatLabel(m)}</span>
                  </span>
                  </>
                ) : (
                <span className="rm-voice-item-meta">
                  {ver && <span className="ver-badge">{ver}</span>}
                  {m.format === "Onnx" ? <span className="msst-onnx-ok">ONNX</span> : <span>{m.format}</span>}
                  {m.index_path && (
                    // SoVITS carries ONE of two mutually exclusive asset kinds
                    // (inference prefers retrieval): `*.index_vectors.npy` =
                    // the retrieval matrix (training default), anything else
                    // in `.cluster/` = kmeans centers — labelling both 聚类
                    // told users their non-kmeans runs produced kmeans
                    <span
                      className="msst-onnx-ok"
                      title={t18(
                        voiceType === "rvc"
                          ? { zh: "已附带检索索引", en: "Retrieval index present", ja: "検索インデックスあり" }
                          : m.index_path.endsWith(".index_vectors.npy")
                            ? { zh: "已附带检索特征库", en: "Retrieval feature bank present", ja: "検索特徴バンクあり" }
                            : { zh: "已附带聚类中心 (kmeans)", en: "Kmeans cluster centers present", ja: "クラスタ中心 (kmeans) あり" },
                        lang,
                      )}
                    >
                      {voiceType === "rvc"
                        ? "IDX"
                        : m.index_path.endsWith(".index_vectors.npy")
                          ? t18({ zh: "检索", en: "RETR", ja: "検索" }, lang)
                          : t18({ zh: "聚类", en: "KMEANS", ja: "クラスタ" }, lang)}
                    </span>
                  )}
                  {/* companion-asset badges — the label matches the inference
                      node's badge verbatim so users can pattern-match across
                      the two surfaces. (S39 had reserved a per-model "VOC"
                      attachment chip here — SUPERSEDED by S40's standalone
                      vocoder resource class, see the 声码器 tab.) */}
                  {m.diffusion_path && (
                    <span className="msst-onnx-ok" title={t18(VOICE_STRINGS.diffBadgeTip, lang)}>
                      DIFF
                    </span>
                  )}
                  <span>{formatSampleRateKhz(m.sample_rate)}</span>
                  {typeof m.config?.features_dim === "number" && (
                    <span>{m.config.features_dim} {t18({ zh: "维", en: "dim", ja: "次元" }, lang)}</span>
                  )}
                </span>
                )}
                {!isVocoder && exportPick !== m.name && (
                  <VoiceRangeRow
                    m={m}
                    voiceType={voiceType as "rvc" | "sovits"}
                    lang={lang}
                    spk={spk}
                    onSpk={(id) => setVoiceSpk((s) => ({ ...s, [m.name]: id }))}
                    editing={rangeEditing === m.name}
                    onEditing={(on) => setRangeEditing(on ? m.name : null)}
                  />
                )}
              </div>
              {/* S167c: while the export chooser is open it OWNS the row (user: the extra
                  buttons burst the row) — audition, the range row and delete all yield,
                  leaving only 包 / 社区格式 / 取消. Same yield pattern as rangeEditing. */}
              {!isVocoder && rangeEditing !== m.name && exportPick !== m.name && (
                <VoiceAuditionButton m={m} voiceType={voiceType as "rvc" | "sovits"} lang={lang} spk={spk} />
              )}
              {rangeEditing !== m.name && (
                <VoiceExportButton
                  m={m}
                  voiceType={voiceType}
                  lang={lang}
                  picking={exportPick === m.name}
                  onPicking={(on) => { setExportPick(on ? m.name : null); if (on) setDeleteConfirm(null); }}
                />
              )}
              {exportPick !== m.name && (deleteConfirm === m.name ? (
                <div className="model-confirm-delete">
                  <button className="danger" onClick={() => handleDelete(m.name)}>{lang === "zh" ? "确认" : "OK"}</button>
                  <button onClick={() => setDeleteConfirm(null)}>{lang === "zh" ? "取消" : "Cancel"}</button>
                </div>
              ) : (
                <button className="model-delete-btn" onClick={() => setDeleteConfirm(m.name)}>{lang === "zh" ? "删除" : "Delete"}</button>
              ))}
            </div>
          );
        })}
      </div>

      {showImport && (
        <ImportDialog
          lang={lang}
          voiceType={voiceType}
          onClose={() => setShowImport(false)}
          onDone={() => { setShowImport(false); fetchModels(); }}
        />
      )}
    </div>
  );
}

// Rust `ModelType` enum variant name (list_models payload) → the frontend voice-tab vocabulary.
// Used to jump to the imported model's tab after a package import.
const MODEL_TYPE_TO_VOICE: Record<string, VoiceType | undefined> = {
  Rvc: "rvc",
  SoVits: "sovits",
  NsfHifigan: "vocoder",
};

/// Windows-safe default filename for the export save dialog (keeps CJK, strips illegal chars +
/// trailing dots/spaces) — mirrors Rust sanitize_file_stem's intent so the suggested name never
/// contains a character the OS save dialog rejects. Empty → "model".
function sanitizeExportName(name: string): string {
  // eslint-disable-next-line no-control-regex
  const cleaned = name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "").replace(/[. ]+$/, "").trim();
  return cleaned || "model";
}

// S78 batch 7: export ONE installed voice model as a portable `.zip` package (re-importable via
// "Import Package"). Per-row so each has its own busy state; guards against a live test/audition
// (Rust re-checks) before opening the native save dialog.
function VoiceExportButton({ m, voiceType, lang, picking, onPicking }: {
  m: VoiceModelEntry; voiceType: VoiceType; lang: string;
  /** S167c: the chooser OWNS the row while open (user: the extra buttons burst the row) — the
   *  open state lives at the tab level (the S146f rangeEditing pattern) so the row can hide its
   *  audition / range / delete controls and show ONLY 包 / 社区格式 / 取消. */
  picking: boolean; onPicking: (on: boolean) => void;
}) {
  const [busy, setBusy] = useState(false);
  // S167: on Export the user picks the format inline: UTAI .zip (lossless re-importable package)
  // vs community-standard files into a plain folder. The community choice is enabled only when
  // the training-side source still exists — installed models are ONNX-only, so the `.pth` must
  // come from the export ledger (has_community_source).
  const [communityOk, setCommunityOk] = useState<boolean | null>(null);

  const guardBusyModel = useCallback((): boolean => {
    const vm = useVoiceModelStore.getState();
    if (vm.rangeTesting[m.name] !== undefined || vm.auditionState?.name === m.name) {
      useAppStore.getState().showToast(
        t18({ zh: "该模型正在测试/试听中，稍后再导出", en: "This model is being tested/auditioned — export later", ja: "このモデルはテスト/試聴中です。後で書き出してください" }, lang),
        "info",
      );
      return false;
    }
    return true;
  }, [m.name, lang]);

  const zipFlow = useCallback(async () => {
    if (busy || !guardBusyModel()) return;
    const dest = await save({
      title: t18({ zh: "导出模型为 .zip", en: "Export model as .zip", ja: "モデルを .zip に書き出し" }, lang),
      defaultPath: `${sanitizeExportName(m.name)}.zip`,
      filters: [{ name: "Zip", extensions: ["zip"] }],
    });
    if (!dest || typeof dest !== "string") return;
    setBusy(true);
    try {
      await invoke("export_model", { name: m.name, modelType: voiceType, destPath: dest });
      useAppStore.getState().showToast(
        t18({ zh: `已导出 · ${m.name}`, en: `Exported · ${m.name}`, ja: `書き出し完了 · ${m.name}` }, lang),
        "success",
      );
    } catch (e) {
      useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
    } finally {
      setBusy(false);
    }
  }, [busy, guardBusyModel, m.name, voiceType, lang]);

  const communityFlow = useCallback(async () => {
    if (busy || !guardBusyModel()) return;
    // community format = plain files, no zip (user 2026-08-31) ⇒ a folder picker
    const dest = await open({
      directory: true,
      title: t18({ zh: "选择社区格式的导出文件夹", en: "Pick a folder for the community-format files", ja: "コミュニティ形式の書き出し先フォルダーを選択" }, lang),
    });
    if (!dest || typeof dest !== "string") return;
    setBusy(true);
    try {
      const files = await invoke<string[]>("export_model_community", { name: m.name, modelType: voiceType, destDir: dest });
      useAppStore.getState().showToast(
        t18({ zh: `已按社区格式导出 ${files.length} 个文件 · ${m.name}`, en: `Exported ${files.length} community-format file(s) · ${m.name}`, ja: `コミュニティ形式で ${files.length} 個のファイルを書き出し · ${m.name}` }, lang),
        "success",
      );
    } catch (e) {
      useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
    } finally {
      setBusy(false);
    }
  }, [busy, guardBusyModel, m.name, voiceType, lang]);

  const onExport = useCallback(async () => {
    if (busy || !guardBusyModel()) return;
    if (voiceType === "vocoder") {
      // vocoders have no community-standard format — go straight to the .zip package
      void zipFlow();
      return;
    }
    let ok = false;
    try {
      ok = await invoke<boolean>("has_community_source", { name: m.name, modelType: voiceType });
    } catch {
      ok = false;
    }
    setCommunityOk(ok);
    onPicking(true);
  }, [busy, guardBusyModel, m.name, voiceType, zipFlow, onPicking]);

  if (picking) {
    return (
      <div className="model-confirm-delete">
        <button
          className="model-export-btn"
          disabled={busy}
          onClick={() => { onPicking(false); void zipFlow(); }}
          title={t18({ zh: "UTAI 模型包，可在其它设备导入", en: "UTAI package — import on another device", ja: "UTAI パッケージ。他のデバイスで取り込めます" }, lang)}
        >
          {t18({ zh: "UTAI 包 (.zip)", en: "UTAI package (.zip)", ja: "UTAI パッケージ (.zip)" }, lang)}
        </button>
        <button
          className="model-export-btn"
          disabled={busy || communityOk !== true}
          onClick={() => { onPicking(false); void communityFlow(); }}
          title={communityOk === true
            ? t18({ zh: "导出社区通用格式到文件夹（不打包）", en: "Community-standard files into a plain folder (no zip)", ja: "コミュニティ標準形式でフォルダーに書き出し（zip なし）" }, lang)
            : t18({ zh: "没有可用的社区源：v0.12 起导入模型会保留源 .pth（此模型更早导入，或当初就是 ONNX）——重新导入一次即可解锁；本机训练的模型请在训练页的存档里导出", en: "No community source: since v0.12 importing a model retains its source .pth (this one predates that, or was imported as bare ONNX) — re-import it once to unlock; locally trained models export from the training page's archive", ja: "コミュニティソースがありません：v0.12 以降のインポートは元の .pth を保持します（このモデルはそれ以前、または ONNX 直接インポート）——一度再インポートすると有効になります。ローカル学習モデルは学習ページのアーカイブから書き出してください" }, lang)}
        >
          {t18({ zh: "社区格式（文件夹）", en: "Community format (folder)", ja: "コミュニティ形式（フォルダー）" }, lang)}
        </button>
        <button className="model-export-btn" disabled={busy} onClick={() => onPicking(false)}>
          {t18({ zh: "取消", en: "Cancel", ja: "キャンセル" }, lang)}
        </button>
      </div>
    );
  }

  return (
    <button
      className="model-export-btn"
      disabled={busy}
      onClick={onExport}
      title={voiceType === "vocoder"
        ? t18({
            zh: "导出为 .zip 模型包，可在其它设备导入",
            en: "Export as a .zip package to import on another device",
            ja: "他のデバイスで取り込める .zip パッケージに書き出し",
          }, lang)
        : t18({
            zh: "导出为 .zip 模型包，可在其它设备导入（含索引/聚类/扩散/头像）",
            en: "Export as a .zip package to import on another device (includes index / cluster / diffusion / avatar)",
            ja: "他のデバイスで取り込める .zip パッケージに書き出し（インデックス/クラスタ/拡散/アバターを含む）",
          }, lang)}
    >
      {busy
        ? t18({ zh: "导出中…", en: "Exporting…", ja: "書き出し中…" }, lang)
        : t18({ zh: "导出", en: "Export", ja: "書き出し" }, lang)}
    </button>
  );
}

// ─── S60-4: per-model audition (resource manager) — the training-audition bare recipe on an
// INSTALLED model via render_model_audition (per-speaker cache in the stem family). Playback
// through the shared preview singleton (contract: stop + assign onEnd on takeover, stop +
// null onEnd on unmount — previewPlayer.ts header). ───

// The S41-era auditionBusyMessage (Chinese substring matchers for the busy guards) is GONE: the S62
// sweep converted every Rust emitter to stable CODEs, so busy classification + localization now live
// entirely in the app-wide mapper (backendErrorMessage / isBusyError — the single source).

function VoiceAuditionButton({ m, voiceType, lang, spk }: { m: VoiceModelEntry; voiceType: "rvc" | "sovits"; lang: string; spk: number }) {
  // shared audition state (audit S60): the preview player is a singleton — per-row local
  // state desyncs on takeover; ownership of a stop() is proven against preview.path.
  // S82d: `spk` comes from the model row's single speaker selector (shared with the range row).
  const audition = useVoiceModelStore((s) => s.auditionState);
  const speakers = voiceSpeakerOptions(m);
  const showToast = useAppStore((s) => s.showToast);
  const phase = audition?.name === m.name ? audition.phase : "idle";

  const start = useCallback(async () => {
    const st = useVoiceModelStore.getState();
    const cur = st.auditionState;
    if (cur?.name === m.name) {
      if (cur.phase === "playing") {
        if (preview.path === cur.path) {
          // we still own the player — a foreign consumer (training page) may have taken over
          preview.onEnd = null;
          preview.stop();
        }
        st.setAuditionState(null);
      }
      return; // rendering → ignore (Rust FlightGuard is the real gate anyway)
    }
    if (cur) return; // another row is busy
    st.setAuditionState({ name: m.name, phase: "rendering" });
    try {
      const path = await invoke<string>("render_model_audition", {
        name: m.name,
        modelType: voiceType,
        speakerId: speakers.length > 1 ? spk : null,
      });
      // the manager may have closed / the state may have been torn down mid-render
      if (useVoiceModelStore.getState().auditionState?.name !== m.name) return;
      const bytes = await readFile(path);
      const buf = await preview.decode(new Uint8Array(bytes));
      if (useVoiceModelStore.getState().auditionState?.name !== m.name) return;
      preview.stop(); // explicit user intent — supersede whatever was playing
      preview.onEnd = () => {
        preview.onEnd = null;
        const a = useVoiceModelStore.getState().auditionState;
        if (a?.name === m.name) useVoiceModelStore.getState().setAuditionState(null);
      };
      await preview.play(path, buf);
      useVoiceModelStore.getState().setAuditionState({ name: m.name, phase: "playing", path });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const mapped = backendErrorMessage(msg);
      const busy = isBusyError(msg);
      // S74: audition runs render_model_audition (blocking inference) — leave a copyable log trace.
      logToBackend(busy ? "warn" : "error", `Model audition failed (${m.name}): ${msg}`);
      // S67c: fatal modal-class errors (INFERENCE_LOW_MEMORY) open the alert dialog instead.
      if (!(mapped && maybeShowErrorModal(msg, mapped))) {
        showToast(
          busy && mapped ? mapped : `${t18({ zh: "试听失败", en: "Audition failed", ja: "試聴に失敗しました" }, lang)}: ${mapped ?? msg}`,
          busy ? "info" : "error",
        );
      }
      if (useVoiceModelStore.getState().auditionState?.name === m.name) {
        useVoiceModelStore.getState().setAuditionState(null);
      }
    }
  }, [m.name, voiceType, spk, speakers.length, lang, showToast]);

  return (
    <span className="rm-audition">
      <button
        className="rm-range-btn rm-audition-btn"
        title={t18(
          phase === "playing"
            ? { zh: "停止", en: "Stop", ja: "停止" }
            : { zh: "试听（同训练页口径：裸配方渲染打包干声片段）", en: "Audition (training-page recipe: bare render of the bundled dry clip)", ja: "試聴（トレーニングページと同条件：バンドル済みドライ音声を素の設定でレンダリング）" },
          lang,
        )}
        onClick={() => void start()}
      >
        {phase === "rendering" ? "…" : phase === "playing" ? "■" : "▶"}
      </button>
    </span>
  );
}

// ─── S81: batch range test. A PERMANENT row, never a one-shot button (§user) — it is worth
// having whenever models are imported in bulk or the criteria change again, and a run can take
// minutes, so its state has to be visible and interruptible rather than a fire-and-forget click.
// The work list comes from the same pure `collectRangeTestTargets` the count does, so the
// number shown and the work performed can never disagree. ───

function RangeBatchRow({ lang }: { lang: string }) {
  const models = useVoiceModelStore((s) => s.models);
  const batch = useVoiceModelStore((s) => s.rangeBatch);
  const cancelBatch = useVoiceModelStore((s) => s.cancelRangeBatch);
  const clearBatch = useVoiceModelStore((s) => s.setRangeBatch);
  const targets = useMemo(() => collectRangeTestTargets(models), [models]);

  if (batch && !batch.finished) {
    return (
      <div className="rm-range-batch">
        <span className="rm-range-batch-text">
          {t18({ zh: "重测音域", en: "Re-testing ranges", ja: "音域を再測定中" }, lang)} {batch.done}/{batch.total}
          {batch.cancel && ` · ${t18({ zh: "正在停止", en: "stopping", ja: "停止中" }, lang)}`}
        </span>
        <div className="rm-range-batch-bar">
          <div style={{ width: `${batch.total ? (batch.done / batch.total) * 100 : 0}%` }} />
        </div>
        <button className="rm-range-btn" disabled={batch.cancel} onClick={cancelBatch}>
          {t18({ zh: "停止", en: "Stop", ja: "停止" }, lang)}
        </button>
      </div>
    );
  }
  if (batch?.finished) {
    // Only reachable when something failed — a clean run clears itself (store.finishRangeBatch).
    return (
      <div className="rm-range-batch">
        <span className="rm-range-batch-text rm-range-missing">
          {t18({ zh: "以下模型未能测出音域", en: "These models could not be measured", ja: "以下のモデルは測定できませんでした" }, lang)}
          {`: ${batch.failed.join("、")}`}
        </span>
        <button className="rm-range-btn" onClick={() => clearBatch(null)}>
          {t18({ zh: "知道了", en: "Dismiss", ja: "閉じる" }, lang)}
        </button>
      </div>
    );
  }
  if (!targets.length) return null; // nothing to offer → no row at all
  return (
    <div className="rm-range-batch">
      <span className="rm-range-batch-text">
        {t18({
          zh: `${targets.length} 个歌手的音域待测或建议重测`,
          en: `${targets.length} singer(s) need a range test or a re-test`,
          ja: `${targets.length} 名の話者が音域測定または再測定を必要としています`,
        }, lang)}
      </span>
      <button
        className="rm-range-btn"
        title={t18({
          zh: "逐个渲染音阶并测量，可能需要几分钟；期间可以随时停止，渲染/播放不会被长时间占用。",
          en: "Renders and measures a scale per singer; may take minutes. Stoppable at any time, and it never holds the render lock across models.",
          ja: "話者ごとに音階をレンダリングして測定します（数分かかる場合があります）。いつでも停止でき、レンダリングロックを跨いで保持しません。",
        }, lang)}
        onClick={() => void runRangeTestBatch(targets)}
      >
        {t18({ zh: "全部重测", en: "Test all", ja: "すべて測定" }, lang)}
      </button>
    </div>
  );
}

// ─── S60-2: per-model vocal-range row (v1 session20/21 UX: auto label + comfort editor
// clamped inside usable + Reset + retest; missing record → 补做 button) ───

function VoiceRangeRow({ m, voiceType, lang, spk, onSpk, editing, onEditing }: {
  m: VoiceModelEntry; voiceType: "rvc" | "sovits"; lang: string; spk: number;
  onSpk: (id: number) => void;
  /** S146f: 受控 —— 编辑态住在 tab 层,因为同排的试听/导出按钮要跟着让位。 */
  editing: boolean;
  onEditing: (on: boolean) => void;
}) {
  const progress = useVoiceModelStore((s) => s.rangeTesting[m.name]);
  // S81: the record is keyed PER SPEAKER on every read side (Rust speaker_range, the node
  // gates, the vocal sidebar) but only speaker 0 was ever writable here, so a multi-speaker
  // model's other singers could never get a record — and their range-extend toggle stayed
  // hidden forever with no way to fix it. Co-trained speakers genuinely differ in range (that
  // is the point of co-training), and borrowing speaker 0's ceiling for another singer is
  // actively wrong, not merely imprecise.
  // S82d: ONE speaker selector per model, state lifted to the tab (the audition button
  // follows it — two per-row selects pointing at different singers were semantic drift, and
  // the row was visibly overcrowded, §user). It renders HERE at the range row's start (the
  // row reads as a sentence: singer ▾ comfort … — and this row has slack + wraps), as a
  // 16px chip matching the row rhythm (a full-height select dwarfed the xs lines = the
  // crowding, §user round 2). Switching speaker closes an open comfort edit: the lo/hi
  // sliders were seeded from the previous singer's record.
  useEffect(() => onEditing(false), [spk]); // eslint-disable-line react-hooks/exhaustive-deps
  const speakers = voiceSpeakerOptions(m);
  const speakerPicker = speakers.length > 1 && (
    <select
      className="sep-model-select rm-range-spk"
      value={spk}
      title={t18({
        zh: "当前歌手——试听与音域记录都指它（多歌手模型的每位歌手各有自己的音域记录）",
        en: "Active speaker — both audition and the range record point at it (each singer of a multi-speaker model has its own range record)",
        ja: "現在の話者——試聴と音域記録の両方が対象とします（多話者モデルは話者ごとに音域記録を持ちます）",
      }, lang)}
      onChange={(e) => onSpk(Number(e.target.value))}
    >
      {speakers.map((s) => (
        <option key={s.id} value={s.id}>{s.label}</option>
      ))}
    </select>
  );
  const rec = (m.config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } }).vocal_range;
  const sp = rec?.speakers?.[String(spk)];
  // what the render layer will actually target (degenerate stored comfort heals to
  // comfort_auto/usable — mirror of the Rust read side); display + slider seed use THIS
  const shown = sp ? targetRange(sp) : null;
  // model-quirk chips from the stored scan: artifact zones + in-range weak notes, so a
  // weird render at those pitches reads as the MODEL's doing (§user S60d2)
  const caution = sp ? deriveCautionZones(sp.semitones ?? {}, sp.usable, targetRange(sp)) : null;
  // S81 F1: a record measured before the timbre dimension existed still WORKS (its damage curve
  // just can't see timbre), so this is an invitation, never a rejection.
  const stale = sp !== undefined && (sp.scan_version ?? 0) < SCAN_VERSION;
  if (progress !== undefined) {
    return (
      <span className="rm-range-row rm-range-testing">
        {t18({ zh: "音域测试中", en: "Testing range", ja: "音域テスト中" }, lang)} {Math.round(progress * 100)}%
      </span>
    );
  }
  if (!sp) {
    // no record (never tested / lost to a re-import / app crash) → the 补做 entry point
    return (
      <span className="rm-range-row">
        {speakerPicker}
        <span className="rm-range-missing">{t18({ zh: "无音域记录", en: "No range record", ja: "音域記録なし" }, lang)}</span>
        <button className="rm-range-btn" onClick={() => void runRangeTest(m.name, voiceType, m.path, spk)}>
          {t18({ zh: "测音域", en: "Detect range", ja: "音域を測定" }, lang)}
        </button>
      </span>
    );
  }
  return (
    <>
    <span className="rm-range-row">
      {speakerPicker}
      <span
        className="rm-range-text"
        title={t18({
          zh: `目标范围 = 被救的音落在哪里。初值由扫描给出（音准 + 浊音 + 音色三项达标的区间），之后以你设的为准。可用范围 ${midiName(sp.usable[0])}–${midiName(sp.usable[1])} 决定哪些音要救，其上沿通常已经很勉强。`,
          en: `Target is where rescued notes land (seeded by the scan — pitch + voicing + timbre all pass — and yours to override). Usable ${midiName(sp.usable[0])}–${midiName(sp.usable[1])} decides WHICH notes get rescued; its top edge is typically already strained.`,
          ja: `目標範囲＝救済された音の着地先。初期値は測定値（音程・有声・音色の三項目を満たす範囲）で、以後はユーザー設定が優先されます。使用可能域 ${midiName(sp.usable[0])}–${midiName(sp.usable[1])} はどの音を救済するかを決めます。`,
        }, lang)}
      >
        {/* S81: the headline is COMFORT. Leading with `usable` advertised a range whose top few
            semitones measurably sing badly — the number the user reads should be the one the
            render aims at. Usable stays available in the tooltip. */}
        {t18({ zh: "目标范围", en: "Target", ja: "目標範囲" }, lang)} {midiName(shown![0])}–{midiName(shown![1])}
      </span>
      {stale && (
        <span
          className="rm-range-missing"
          title={t18({
            zh: "这条记录是在加入音色检测之前测的，仍然可用；重测一次会让目标范围更准。",
            en: "This record predates the timbre measurement. It still works; re-testing makes the target range more accurate.",
            ja: "この記録は音色測定の追加前に取得されたものです。引き続き使用できますが、再測定すると目標範囲がより正確になります。",
          }, lang)}
        >
          {t18({ zh: "建议重测", en: "re-test suggested", ja: "再測定推奨" }, lang)}
        </span>
      )}
      {editing ? (
        // S146e: 四个滑条(可用范围 + 目标范围)现在都在共用编辑器里 —— 人声侧栏挂的是**同一个**
        // 组件。⛔ 别在这里重写一份:两个入口的夹取规则一旦分叉,一边写出去的记录另一边
        // 读起来就是错的,而这条 UI 线**零渲染测试**接得住。
        <span className="rm-range-edit">
          <RangeBoundsEditor
            sp={sp}
            modelName={m.name}
            backend={voiceType}
            speakerId={spk}
            speakerLabel={speakers.find((s) => s.id === spk)?.label}
            lang={lang}
            onClose={() => onEditing(false)}
          />
        </span>
      ) : (
        <>
          <button
            className="rm-range-btn"
            title={t18({ zh: "调整可用范围与目标范围（MIDI 音号）", en: "Adjust the usable and target ranges (MIDI numbers)", ja: "使用可能域と目標範囲を調整（MIDI 番号）" }, lang)}
            onClick={() => onEditing(true)}
          >
            {t18({ zh: "调整", en: "Adjust", ja: "調整" }, lang)}
          </button>
          <button className="rm-range-btn" onClick={() => void runRangeTest(m.name, voiceType, m.path, spk)}>
            {t18({ zh: "重测", en: "Retest", ja: "再測定" }, lang)}
          </button>
        </>
      )}
    </span>
    {(caution!.artifact.length > 0 || caution!.weak.length > 0) && (
      <span className="rm-range-row rm-range-caution-row">
        {caution!.artifact.length > 0 && (
          <span
            className="rm-range-caution"
            title={`${caution!.artifact.map(([a, b]) => `${midiName(a)}–${midiName(b)}`).join(", ")} — ${t18({
              zh: "模型在这些音高会发声但明显走音（中位误差≥200¢）——模型自身的伪影区，不是程序或算法问题；此区间谨慎使用",
              en: "the model voices these pitches but lands ≥200¢ off — model-side artifact zones, not a program/algorithm issue; use with caution",
              ja: "モデルはこの音高で発声しますが大きく音を外します（中央誤差≥200¢）——モデル自体のアーティファクト域です。プログラムの問題ではありません",
            }, lang)}`}
          >
            {t18({ zh: "伪影", en: "artifacts", ja: "偽影" }, lang)}{" "}
            {caution!.artifact.map(([a, b]) => `${midiName(a)}–${midiName(b)}`).join(", ")}
          </span>
        )}
        {caution!.weak.length > 0 && (
          <span
            className="rm-range-caution"
            title={`${caution!.weak.map((n) => midiName(n)).join(", ")} — ${t18({
              zh: "可用区内部的孤立弱音（测试未达标、推导范围时被桥接跳过）——这些音上出怪声属模型自身问题，谨慎使用",
              en: "isolated weak notes inside the usable range (failed the probe, bridged over when deriving) — oddities at these pitches are the model's own; use with caution",
              ja: "使用可能域内の孤立した弱点（測定不合格・範囲導出時にブリッジ）——この音高での異音はモデル由来です",
            }, lang)}`}
          >
            {t18({ zh: "弱点", en: "weak", ja: "弱点" }, lang)}{" "}
            {caution!.weak.slice(0, 3).map((n) => midiName(n)).join(", ")}
            {caution!.weak.length > 3 ? ` +${caution!.weak.length - 3}` : ""}
          </span>
        )}
      </span>
    )}
    </>
  );
}

function formatSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  return `${bytes} B`;
}
