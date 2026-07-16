import { useEffect, useState, useCallback, useRef } from "react";
import { open } from "@tauri-apps/plugin-dialog";
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
import { runRangeTest, setComfortRange, midiName, effectiveComfort, deriveCautionZones, MIN_COMFORT_SPAN, type SpeakerRangeRecord } from "../../lib/vocal/rangeTest";
import { preview } from "../common/previewPlayer";
import { ParamSlider } from "../workflow/nodes/ParamSlider";
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
                          <button
                            className="msst-convert-btn"
                            disabled={anyConverting}
                            title={t18({ zh: "从 ckpt 完整导出 fp32（较慢）", en: "Full fp32 export from the ckpt (slower)", ja: "ckpt から fp32 を完全エクスポート（時間がかかります）" }, lang)}
                            onClick={() => convertPrecision(m.filename, "fp32", archHint)}
                          >
                            {t18({ zh: "补转 fp32", en: "Convert to fp32", ja: "fp32に変換" }, lang)}
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
  const [voiceType, setVoiceType] = useState<VoiceType>("rvc");
  const [showImport, setShowImport] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
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
        <button className="primary rm-import-top-btn" onClick={() => setShowImport(true)}>
          + {lang === "zh" ? "导入模型" : lang === "ja" ? "モデル取り込み" : "Import Model"}
        </button>
      </div>

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
          const speakerCount = isVocoder ? 0 : voiceSpeakerOptions(m).length;
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
                  <span className="rm-voice-item-meta">
                    <span className="ver-badge" title={t18({ zh: "经典 NSF-HiFiGAN 架构", en: "Classic NSF-HiFiGAN architecture", ja: "クラシック NSF-HiFiGAN アーキテクチャ" }, lang)}>
                      NSF-HiFiGAN
                    </span>
                    <span>{vocoderFormatLabel(m)}</span>
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
                  {speakerCount > 1 && (
                    <span>{speakerCount} {t18({ zh: "歌手", en: "speakers", ja: "話者" }, lang)}</span>
                  )}
                </span>
                )}
                {!isVocoder && <VoiceRangeRow m={m} voiceType={voiceType as "rvc" | "sovits"} lang={lang} />}
              </div>
              {!isVocoder && <VoiceAuditionButton m={m} voiceType={voiceType as "rvc" | "sovits"} lang={lang} />}
              {deleteConfirm === m.name ? (
                <div className="model-confirm-delete">
                  <button className="danger" onClick={() => handleDelete(m.name)}>{lang === "zh" ? "确认" : "OK"}</button>
                  <button onClick={() => setDeleteConfirm(null)}>{lang === "zh" ? "取消" : "Cancel"}</button>
                </div>
              ) : (
                <button className="model-delete-btn" onClick={() => setDeleteConfirm(m.name)}>{lang === "zh" ? "删除" : "Delete"}</button>
              )}
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

// ─── S60-4: per-model audition (resource manager) — the training-audition bare recipe on an
// INSTALLED model via render_model_audition (per-speaker cache in the stem family). Playback
// through the shared preview singleton (contract: stop + assign onEnd on takeover, stop +
// null onEnd on unmount — previewPlayer.ts header). ───

// The S41-era auditionBusyMessage (Chinese substring matchers for the busy guards) is GONE: the S62
// sweep converted every Rust emitter to stable CODEs, so busy classification + localization now live
// entirely in the app-wide mapper (backendErrorMessage / isBusyError — the single source).

function VoiceAuditionButton({ m, voiceType, lang }: { m: VoiceModelEntry; voiceType: "rvc" | "sovits"; lang: string }) {
  // shared audition state (audit S60): the preview player is a singleton — per-row local
  // state desyncs on takeover; ownership of a stop() is proven against preview.path.
  const audition = useVoiceModelStore((s) => s.auditionState);
  const [spk, setSpk] = useState(0);
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
      {speakers.length > 1 && phase === "idle" && (
        <select
          className="sep-model-select rm-audition-spk"
          value={spk}
          title={t18({ zh: "试听歌手", en: "Audition speaker", ja: "試聴する話者" }, lang)}
          onChange={(e) => setSpk(Number(e.target.value))}
        >
          {speakers.map((s) => (
            <option key={s.id} value={s.id}>{s.label}</option>
          ))}
        </select>
      )}
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

// ─── S60-2: per-model vocal-range row (v1 session20/21 UX: auto label + comfort editor
// clamped inside usable + Reset + retest; missing record → 补做 button) ───

function VoiceRangeRow({ m, voiceType, lang }: { m: VoiceModelEntry; voiceType: "rvc" | "sovits"; lang: string }) {
  const progress = useVoiceModelStore((s) => s.rangeTesting[m.name]);
  const [editing, setEditing] = useState(false);
  const [lo, setLo] = useState(0);
  const [hi, setHi] = useState(0);
  const rec = (m.config as { vocal_range?: { speakers?: Record<string, SpeakerRangeRecord> } }).vocal_range;
  const sp = rec?.speakers?.["0"];
  // what the render layer will actually target (degenerate stored comfort heals to
  // comfort_auto/usable — mirror of the Rust read side); display + slider seed use THIS
  const shown = sp ? effectiveComfort(sp) : null;
  // model-quirk chips from the stored scan: artifact zones + in-range weak notes, so a
  // weird render at those pitches reads as the MODEL's doing (§user S60d2)
  const caution = sp ? deriveCautionZones(sp.semitones ?? {}, sp.usable) : null;
  const commit = async () => {
    if (!sp) { setEditing(false); return; }
    await setComfortRange(m.name, voiceType, 0, [lo, hi]); // clampComfort enforces span
    setEditing(false);
  };

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
        <span className="rm-range-missing">{t18({ zh: "无音域记录", en: "No range record", ja: "音域記録なし" }, lang)}</span>
        <button className="rm-range-btn" onClick={() => void runRangeTest(m.name, voiceType, m.path)}>
          {t18({ zh: "测音域", en: "Detect range", ja: "音域を測定" }, lang)}
        </button>
      </span>
    );
  }
  return (
    <>
    <span className="rm-range-row">
      <span
        className="rm-range-text"
        title={t18({
          zh: `可用（<100¢ 且浊音>50%）${midiName(sp.usable[0])}–${midiName(sp.usable[1])}；舒适（<50¢ 且浊音>80%）为渲染的目标区间`,
          en: `Usable (<100¢, voiced>50%) ${midiName(sp.usable[0])}–${midiName(sp.usable[1])}; comfort (<50¢, voiced>80%) is the render target zone`,
          ja: `使用可能（<100¢・有声>50%）${midiName(sp.usable[0])}–${midiName(sp.usable[1])}。快適域（<50¢・有声>80%）がレンダリングの目標域です`,
        }, lang)}
      >
        {t18({ zh: "音域", en: "Range", ja: "音域" }, lang)} {midiName(sp.usable[0])}–{midiName(sp.usable[1])}
        {" · "}
        {t18({ zh: "舒适", en: "comfort", ja: "快適" }, lang)} {midiName(shown![0])}–{midiName(shown![1])}
      </span>
      {editing ? (
        <span className="rm-range-edit">
          <ParamSlider
            label={t18({ zh: "下限", en: "Low", ja: "下限" }, lang)}
            min={sp.usable[0]} max={sp.usable[1]} step={1} value={lo}
            onChange={(v) => setLo(Math.max(sp.usable[0], Math.min(v, hi - MIN_COMFORT_SPAN)))}
            format={(v) => midiName(v)}
          />
          <ParamSlider
            label={t18({ zh: "上限", en: "High", ja: "上限" }, lang)}
            min={sp.usable[0]} max={sp.usable[1]} step={1} value={hi}
            onChange={(v) => setHi(Math.min(sp.usable[1], Math.max(v, lo + MIN_COMFORT_SPAN)))}
            format={(v) => midiName(v)}
          />
          <button className="rm-range-btn" onClick={() => void commit()}>OK</button>
          <button
            className="rm-range-btn"
            title={t18({ zh: "还原为自动检测值", en: "Reset to the detected value", ja: "自動検出値に戻す" }, lang)}
            onClick={() => { void setComfortRange(m.name, voiceType, 0, sp.comfort_auto).then(() => setEditing(false)); }}
          >
            {t18({ zh: "还原", en: "Reset", ja: "リセット" }, lang)}
          </button>
        </span>
      ) : (
        <>
          <button
            className="rm-range-btn"
            title={t18({ zh: "在可用区间内微调舒适区（MIDI 音号）", en: "Adjust the comfort zone within usable (MIDI numbers)", ja: "使用可能域の中で快適域を調整（MIDI 番号）" }, lang)}
            onClick={() => { setLo(shown![0]); setHi(shown![1]); setEditing(true); }}
          >
            {t18({ zh: "调整", en: "Adjust", ja: "調整" }, lang)}
          </button>
          <button className="rm-range-btn" onClick={() => void runRangeTest(m.name, voiceType, m.path)}>
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
