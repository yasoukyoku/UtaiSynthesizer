import { useEffect, useState, useCallback } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
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
  t18,
  type MsstArchitecture,
  type MsstCatalogEntry,
  type MsstCategory,
  type MsstPrecision,
} from "../../lib/models/msst-catalog";
import { useDraggable } from "../../lib/useDraggable";
import { VOICE_STRINGS } from "../workflow/nodes/VoiceModelPicker";
import {
  useVoiceModelStore,
  voiceVersionBadge,
  voiceSpeakerOptions,
  formatSampleRateKhz,
  vocoderFormatMatches,
  vocoderFormatLabel,
  type VoiceType,
} from "../../store/voice-models";
import "./MsstModelManager.css";

type TopTab = "separation" | "voice";

export function MsstModelManager({ onClose }: { onClose: () => void }) {
  const { i18n } = useTranslation();
  const lang = i18n.language;
  const {
    installed, downloading, error,
    fetchInstalled, fetchModelsDir, modelsDir,
    clearError, deleteModel, downloadEntry, convertPrecision,
  } = useMsstModelStore();

  const { pos, startDrag } = useDraggable(() => ({ x: 100, y: 96 }));

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
      </div>

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
    setImporting(true);
    setErr("");
    try {
      const outcome = await invoke<{ entry: unknown; warnings: string[] }>("import_model", {
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
        useAppStore.getState().showToast(w, "info");
      }
      onDone();
    } catch (e) {
      setErr(String(e));
    }
    setImporting(false);
  }, [modelPath, modelName, voiceType, indexPath, diffusionPath, diffusionConfigPath, avatarPath, vocoderConfigPath, onDone]);

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

  const handleDelete = useCallback(async (name: string) => {
    // type-scoped: same-name entries across types are standard (rvc+sovits pair
    // + a vocoder named after the singer) — an untyped delete hits the first
    // scan match, i.e. potentially the WRONG model's files (S40 红队 A5)
    await deleteModel(name, voiceType); // errors land in voiceError
    setDeleteConfirm(null);
  }, [deleteModel, voiceType]);

  return (
    <div className="rm-voice-tab">
      {voiceError && <div className="msst-error" onClick={clearError}>{voiceError}</div>}
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
                      zh: `缺少文件：${defaultVoc.missing.join("、")}——请将其放入 data/models/aux/，否则浅扩散/增强器无法运行`,
                      en: `Missing: ${defaultVoc.missing.join(", ")} — place them in data/models/aux/ or shallow diffusion / the enhancer cannot run`,
                      ja: `欠落ファイル：${defaultVoc.missing.join("、")} — data/models/aux/ に配置してください。ないと浅い拡散/エンハンサーは動きません`,
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

function formatSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  return `${bytes} B`;
}
