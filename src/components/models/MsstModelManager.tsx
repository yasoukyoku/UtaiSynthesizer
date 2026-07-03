import { useEffect, useState, useCallback } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useMsstModelStore, setupDownloadListener } from "../../store/msst-models";
import {
  MSST_CATALOG,
  ALL_CATEGORIES,
  CATEGORY_LABELS,
  ARCHITECTURE_LABELS,
  MSST_DEFAULT_PRECISION,
  MSST_FP16_ARCHS,
  MSST_FP16_TIP,
  t18,
  type MsstArchitecture,
  type MsstCatalogEntry,
  type MsstCategory,
  type MsstPrecision,
  type MirrorSource,
} from "../../lib/models/msst-catalog";
import { useDraggable } from "../../lib/useDraggable";
import "./MsstModelManager.css";

type TopTab = "separation" | "voice";
type VoiceType = "rvc" | "sovits";

interface VoiceModelEntry {
  name: string;
  model_type: string;
  format: string;
  path: string;
  sample_rate: number;
  index_path: string | null;
  avatar_path: string | null;
}

export function MsstModelManager({ onClose }: { onClose: () => void }) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const {
    installed, downloading, error, mirror,
    fetchInstalled, fetchModelsDir, modelsDir,
    clearError, deleteModel, setMirror, downloadEntry, convertPrecision,
  } = useMsstModelStore();

  const { pos, startDrag } = useDraggable(() => ({ x: 100, y: 96 }));

  const [topTab, setTopTab] = useState<TopTab>("separation");
  const [category, setCategory] = useState<MsstCategory>("vocals");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  // Download-time precision choice per catalog entry (roformers only); absent = arch default.
  const [dlPrecision, setDlPrecision] = useState<Record<string, MsstPrecision>>({});
  const [showMirrorConfig, setShowMirrorConfig] = useState(false);

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

  const mirrorLabel = mirror.type === "hf-mirror" ? "HF Mirror" : mirror.type === "custom" ? "Custom" : "HuggingFace";

  return (
    <aside className="msst-model-manager" style={{ left: pos.x, top: pos.y }}>
      <div className="panel-header" onMouseDown={startDrag}>
        <span className="panel-title">{lang === "zh" ? "资源管理" : lang === "ja" ? "リソース管理" : "Resource Manager"}</span>
        <button className="panel-close" onClick={onClose}>X</button>
      </div>

      {error && <div className="msst-error" onClick={clearError}>{error}</div>}

      <div className="rm-top-tabs">
        <button className={topTab === "separation" ? "active" : ""} onClick={() => setTopTab("separation")}>
          {lang === "zh" ? "音频分离" : lang === "ja" ? "音声分離" : "Separation"}
        </button>
        <button className={topTab === "voice" ? "active" : ""} onClick={() => setTopTab("voice")}>
          {lang === "zh" ? "声音模型" : lang === "ja" ? "ボイスモデル" : "Voice Models"}
        </button>
        <div className="rm-top-spacer" />
        {topTab === "separation" && (
          <button
            className={`rm-mirror-btn ${showMirrorConfig ? "active" : ""}`}
            onClick={() => setShowMirrorConfig(!showMirrorConfig)}
            title={mirrorLabel}
          >
            {lang === "zh" ? "源" : lang === "ja" ? "ソース" : "Source"}: {mirrorLabel}
          </button>
        )}
      </div>

      {showMirrorConfig && topTab === "separation" && (
        <MirrorConfig mirror={mirror} onChange={setMirror} lang={lang} onClose={() => setShowMirrorConfig(false)} />
      )}

      {topTab === "voice" && <VoiceModelsTab lang={lang} />}

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
                          <button className="msst-convert-btn" onClick={() => convertPrecision(m.filename, undefined, archHint)}>Convert</button>
                        ) : fp16Capable && !m.has_fp16 ? (
                          <button
                            className="msst-convert-btn"
                            title={t18(MSST_FP16_TIP, lang)}
                            onClick={() => convertPrecision(m.filename, "fp16", archHint)}
                          >
                            {t18({ zh: "补转 fp16", en: "Convert to fp16", ja: "fp16に変換" }, lang)}
                          </button>
                        ) : fp16Capable && !m.has_onnx ? (
                          <button
                            className="msst-convert-btn"
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
  const [avatarPath, setAvatarPath] = useState("");
  const [modelName, setModelName] = useState("");
  const [importing, setImporting] = useState(false);
  const [err, setErr] = useState("");

  const browse = useCallback(async (title: string, exts: string[]) => {
    const path = await open({ title, filters: [{ name: "File", extensions: exts }] });
    return path ? (path as string) : "";
  }, []);

  const handleBrowseModel = useCallback(async () => {
    const p = await browse(lang === "zh" ? "选择模型文件 (.pth)" : "Select model file (.pth)", ["pth", "onnx"]);
    if (p) {
      setModelPath(p);
      const filename = p.split(/[/\\]/).pop() ?? "";
      setModelName(filename.replace(/\.(pth|onnx)$/i, ""));
    }
  }, [browse, lang]);

  const handleBrowseIndex = useCallback(async () => {
    const p = await browse(lang === "zh" ? "选择索引文件 (.index)" : "Select index file (.index)", ["index", "npy"]);
    if (p) setIndexPath(p);
  }, [browse, lang]);

  const handleBrowseAvatar = useCallback(async () => {
    const p = await browse(lang === "zh" ? "选择角色头图" : "Select character avatar", ["png", "jpg", "jpeg", "bmp", "webp"]);
    if (p) setAvatarPath(p);
  }, [browse, lang]);

  const handleImport = useCallback(async () => {
    if (!modelPath || !modelName) return;
    setImporting(true);
    setErr("");
    try {
      await invoke("import_model", {
        name: modelName,
        path: modelPath,
        modelType: voiceType,
        indexPath: indexPath || null,
        avatarPath: avatarPath || null,
      });
      onDone();
    } catch (e) {
      setErr(String(e));
    }
    setImporting(false);
  }, [modelPath, modelName, voiceType, indexPath, avatarPath, onDone]);

  const isRvc = voiceType === "rvc";
  const Z = (key: string) => {
    const map: Record<string, Record<string, string>> = {
      title: { zh: `导入 ${voiceType.toUpperCase()} 模型`, en: `Import ${voiceType.toUpperCase()} Model`, ja: `${voiceType.toUpperCase()} モデル取り込み` },
      model: { zh: "模型文件 (.pth)", en: "Model file (.pth)", ja: "モデルファイル (.pth)" },
      index: { zh: "索引文件 (.index)  — 可选", en: "Index file (.index) — optional", ja: "インデックス (.index) — 任意" },
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

        {isRvc && (
          <div className="rm-import-field">
            <label>{Z("index")}</label>
            <div className="rm-import-row">
              <input type="text" readOnly value={indexPath} placeholder="..." className="rm-import-path" />
              <button onClick={handleBrowseIndex}>{Z("browseBtn")}</button>
            </div>
          </div>
        )}

        <div className="rm-import-field">
          <label>{Z("avatar")}</label>
          <div className="rm-import-row">
            <input type="text" readOnly value={avatarPath} placeholder="..." className="rm-import-path" />
            <button onClick={handleBrowseAvatar}>{Z("browseBtn")}</button>
          </div>
        </div>

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

function MirrorConfig({ mirror, onChange, lang, onClose }: { mirror: MirrorSource; onChange: (m: MirrorSource) => void; lang: string; onClose: () => void }) {
  return (
    <div className="rm-mirror-config">
      <div className="rm-mirror-title">
        {lang === "zh" ? "下载源设置" : lang === "ja" ? "ダウンロードソース設定" : "Download Source"}
        <button className="rm-mirror-close" onClick={onClose}>X</button>
      </div>
      <label className={mirror.type === "huggingface" ? "active" : ""}>
        <input type="radio" name="mirror" checked={mirror.type === "huggingface"} onChange={() => onChange({ type: "huggingface", customUrl: mirror.customUrl })} />
        HuggingFace ({lang === "zh" ? "默认" : "Default"})
      </label>
      <label className={mirror.type === "hf-mirror" ? "active" : ""}>
        <input type="radio" name="mirror" checked={mirror.type === "hf-mirror"} onChange={() => onChange({ type: "hf-mirror", customUrl: mirror.customUrl })} />
        HF Mirror (hf-mirror.com) — {lang === "zh" ? "中国大陆加速" : "China mainland"}
      </label>
      <label className={mirror.type === "custom" ? "active" : ""}>
        <input type="radio" name="mirror" checked={mirror.type === "custom"} onChange={() => onChange({ type: "custom", customUrl: mirror.customUrl })} />
        {lang === "zh" ? "自定义" : "Custom"}
      </label>
      {mirror.type === "custom" && (
        <input type="text" className="rm-mirror-url" placeholder="https://your-mirror.com"
          value={mirror.customUrl} onChange={(e) => onChange({ type: "custom", customUrl: e.target.value })} />
      )}
    </div>
  );
}

function VoiceAvatar({ path, name, onSet }: { path: string | null; name: string; onSet: () => void }) {
  if (path) {
    return (
      <div className="rm-voice-avatar" onClick={onSet} title={name}>
        <img src={`https://asset.localhost/${path.replace(/\\/g, "/")}`} alt={name} />
      </div>
    );
  }
  return (
    <div className="rm-voice-avatar rm-voice-avatar-empty" onClick={onSet} title="Set avatar">
      <span>{name.charAt(0).toUpperCase()}</span>
    </div>
  );
}

function VoiceModelsTab({ lang }: { lang: string }) {
  const [voiceType, setVoiceType] = useState<VoiceType>("rvc");
  const [models, setModels] = useState<VoiceModelEntry[]>([]);
  const [showImport, setShowImport] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      const list = await invoke<VoiceModelEntry[]>("list_models", { modelType: voiceType });
      setModels(list);
    } catch { setModels([]); }
  }, [voiceType]);

  useEffect(() => { fetchModels(); }, [fetchModels]);

  const handleDelete = useCallback(async (name: string) => {
    try {
      await invoke("delete_model", { name });
      setDeleteConfirm(null);
      await fetchModels();
    } catch (e) { useMsstModelStore.setState({ error: String(e) }); }
  }, [fetchModels]);

  return (
    <div className="rm-voice-tab">
      <div className="msst-filter">
        <button className={voiceType === "rvc" ? "active" : ""} onClick={() => setVoiceType("rvc")}>RVC</button>
        <button className={voiceType === "sovits" ? "active" : ""} onClick={() => setVoiceType("sovits")}>SoVITS</button>
        <div className="rm-filter-spacer" />
        <button className="primary rm-import-top-btn" onClick={() => setShowImport(true)}>
          + {lang === "zh" ? "导入模型" : lang === "ja" ? "モデル取り込み" : "Import Model"}
        </button>
      </div>

      <div className="rm-voice-list">
        {models.length === 0 && (
          <p className="msst-empty">
            {lang === "zh"
              ? `暂无 ${voiceType.toUpperCase()} 模型`
              : `No ${voiceType.toUpperCase()} models`}
          </p>
        )}
        {models.map((m) => (
          <div key={m.name} className="rm-voice-item">
            <VoiceAvatar path={m.avatar_path} name={m.name} onSet={async () => {
              const file = await open({ title: lang === "zh" ? "选择角色头图" : "Select avatar", filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "bmp", "webp"] }] });
              if (file) { await invoke("set_model_avatar", { name: m.name, avatarPath: file as string }); await fetchModels(); }
            }} />
            <div className="rm-voice-item-info">
              <span className="rm-voice-item-name">{m.name}</span>
              <span className="rm-voice-item-meta">
                {m.format === "Onnx" ? <span className="msst-onnx-ok">ONNX</span> : <span>{m.format}</span>}
                {m.index_path && <span className="msst-onnx-ok">IDX</span>}
                {" "}{m.sample_rate / 1000}kHz
              </span>
            </div>
            {deleteConfirm === m.name ? (
              <div className="model-confirm-delete">
                <button className="danger" onClick={() => handleDelete(m.name)}>{lang === "zh" ? "确认" : "OK"}</button>
                <button onClick={() => setDeleteConfirm(null)}>{lang === "zh" ? "取消" : "Cancel"}</button>
              </div>
            ) : (
              <button className="model-delete-btn" onClick={() => setDeleteConfirm(m.name)}>{lang === "zh" ? "删除" : "Delete"}</button>
            )}
          </div>
        ))}
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

function formatSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  return `${bytes} B`;
}
