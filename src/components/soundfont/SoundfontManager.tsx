import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import { useAppStore } from "../../store/app";
import { useSoundfontStore, RECOMMENDED_SOUNDFONTS, type Soundfont } from "../../store/soundfont";
import { useAmtModelStore, setupAmtDownloadListener } from "../../store/amt-models";
import { AMT_CATALOG, type AmtCatalogEntry } from "../../lib/models/amt-catalog";
import { t18 } from "../../lib/models/msst-catalog";
import { useFloatingPanel } from "../../lib/useFloatingPanel";
import { PanelResizeHandles } from "../common/PanelResizeHandles";
import { backendErrorMessage } from "../../lib/backendError";
import "./SoundfontManager.css";

function formatSize(bytes: number): string {
  if (!bytes || bytes <= 0) return "—";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/** 可一键下载并自动安装的通用音源（来自 AMT 目录，architecture = soundfont）。 */
const DOWNLOADABLE_SOUNDFONTS = AMT_CATALOG.filter((e) => e.architecture === "soundfont");

type TabKey = "installed" | "download";

export function SoundfontManager() {
  const { t, i18n } = useTranslation();
  const lang = i18n.language;
  const close = useCallback(() => useAppStore.getState().toggleSoundfontManager(), []);
  const { fonts, scanning, importingPaths, refresh, importFont, removeFont } = useSoundfontStore();
  const {
    installed: amtInstalled,
    downloading: amtProgress,
    fetchInstalled: fetchAmtInstalled,
    downloadEntry: downloadSoundfont,
  } = useAmtModelStore();
  // 稳定 action 从 getState 取,避免整 store 订阅(toast/滚动等任意变化都会重渲染本面板)。
  const { showToast, showConfirm } = useAppStore.getState();

  const [tab, setTab] = useState<TabKey>("installed");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const { style, startDrag, startResize } = useFloatingPanel({
    storageKey: "muno.soundfontManagerRect",
    initial: () => ({ x: 160, y: 120, w: 620, h: Math.round(window.innerHeight * 0.75) }),
    minW: 520,
    minH: 360,
  });

  // 打开即扫一次 + 拉取可下载音源的安装状态
  useEffect(() => {
    void refresh();
    void fetchAmtInstalled();
    void setupAmtDownloadListener();
  }, [refresh, fetchAmtInstalled]);

  const handleImport = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const picked = await open({
        multiple: true,
        filters: [
          { name: "SoundFont", extensions: ["sf2", "sf3"] },
          { name: "SFZ", extensions: ["sfz"] },
        ],
        title: t("soundfont.importDialogTitle"),
      });
      if (!picked) return;
      const paths = Array.isArray(picked) ? picked : [picked];
      for (const p of paths) {
        try {
          const font = await importFont(p);
          if (font) showToast(t("soundfont.imported", { name: font.name }), "success");
        } catch (e) {
          showToast(backendErrorMessage(e) ?? String(e), "error");
        }
      }
    } catch (e) {
      showToast(backendErrorMessage(e) ?? String(e), "error");
    }
  };

  const handleImportDir = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const dir = await open({ directory: true, multiple: false, title: t("soundfont.importDirTitle") });
      if (!dir || Array.isArray(dir)) return;
      try {
        const font = await importFont(dir);
        if (font) showToast(t("soundfont.imported", { name: font.name }), "success");
      } catch (e) {
        showToast(backendErrorMessage(e) ?? String(e), "error");
      }
    } catch (e) {
      showToast(backendErrorMessage(e) ?? String(e), "error");
    }
  };

  const handleDelete = async (font: Soundfont) => {
    setConfirmDelete(font.id);
    const choice = await showConfirm({
      title: t("soundfont.deleteTitle"),
      body: t("soundfont.deleteBody", { name: font.name }),
      buttons: [
        { id: "cancel", label: t("common.cancel") },
        { id: "delete", label: t("soundfont.deleteConfirm"), kind: "danger" },
      ],
    });
    setConfirmDelete(null);
    if (choice !== "delete") return;
    const ok = await removeFont(font.id);
    if (ok) showToast(t("soundfont.deleted", { name: font.name }), "success");
    else showToast(t("soundfont.deleteFailed"), "error");
  };

  const handleOpenFolder = async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("open_soundfonts_dir");
    } catch (e) {
      showToast(backendErrorMessage(e) ?? String(e), "error");
    }
  };

  const handleOpenLink = async (url: string) => {
    try {
      await openUrl(url);
    } catch {
      /* openUrl 权限失败时静默 */
    }
  };

  const handleRescan = () => void refresh();

  return (
    <aside className="soundfont-manager" style={style}>
      <div className="panel-header" onMouseDown={startDrag}>
        <span className="panel-title">{t("soundfont.title")}</span>
        <button className="panel-close" onClick={close}>X</button>
      </div>

      {/* 顶部标签页 */}
      <div className="sfm-tabs">
        <button
          className={`sfm-tab ${tab === "installed" ? "active" : ""}`}
          onClick={() => setTab("installed")}
        >
          {t("soundfont.tabInstalled")}
          {fonts.length > 0 && <span className="sfm-tab-count">{fonts.length}</span>}
        </button>
        <button
          className={`sfm-tab ${tab === "download" ? "active" : ""}`}
          onClick={() => setTab("download")}
        >
          {t("soundfont.tabDownload")}
          <span className="sfm-tab-count sfm-tab-primary">
            {DOWNLOADABLE_SOUNDFONTS.length + RECOMMENDED_SOUNDFONTS.length}
          </span>
        </button>
      </div>

      <div className="sfm-body">
        {tab === "installed" && (
          <div className="sfm-tab-pane">
            <div className="sfm-toolbar">
              <button className="sfm-btn primary" onClick={handleImport} disabled={importingPaths.length > 0}>
                {t("soundfont.importFile")}
              </button>
              <button className="sfm-btn" onClick={handleImportDir} disabled={importingPaths.length > 0}>
                {t("soundfont.importDir")}
              </button>
              <button className="sfm-btn" onClick={handleOpenFolder}>{t("soundfont.openFolder")}</button>
              <button className="sfm-btn" onClick={handleRescan} disabled={scanning}>
                {scanning ? t("soundfont.scanning") : t("soundfont.rescan")}
              </button>
            </div>

            <div className="sfm-list">
              {fonts.length === 0 && !scanning ? (
                <div className="sfm-empty">
                  <p>{t("soundfont.empty")}</p>
                  <p className="sfm-hint">{t("soundfont.emptyHint")}</p>
                  <button className="sfm-btn primary" style={{ marginTop: 12 }} onClick={() => setTab("download")}>
                    → {t("soundfont.tabDownload")}
                  </button>
                </div>
              ) : (
                fonts.map((f) => (
                  <div key={f.id} className={`sfm-font-card ${confirmDelete === f.id ? "confirming" : ""}`}>
                    <div className="sfm-font-main">
                      <span className={`sfm-format ${f.format}`}>{f.format.toUpperCase()}</span>
                      <span className="sfm-font-name" title={f.id}>{f.name}</span>
                      <span className="sfm-size">{formatSize(f.sizeBytes)}</span>
                    </div>
                    <div className="sfm-font-sub">
                      {f.presets.length > 0 ? (
                        <span className="sfm-presets" title={f.presets.map((p) => p.name).join(" · ")}>
                          {t("soundfont.presetCount", { count: f.presets.length })}
                        </span>
                      ) : (
                        <span className="sfm-presets none">{t("soundfont.noPresets")}</span>
                      )}
                    </div>
                    <button className="sfm-delete" onClick={() => void handleDelete(f)} title={t("soundfont.deleteTitle")}>
                      ✕
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {tab === "download" && (
          <div className="sfm-tab-pane sfm-download-pane">
            {/* Section A: 可一键下载的 SF2 通用音源 */}
            <div className="sfm-section">
              <div className="sfm-section-title">
                <span className="sfm-section-dot" />
                {t("soundfont.oneClick")}
              </div>
              <div className="sfm-section-hint">{t("soundfont.recommendedNote")}</div>
              <div className="sfm-rec-grid">
                {DOWNLOADABLE_SOUNDFONTS.map((entry) => (
                  <AmtSoundfontCard
                    key={entry.id}
                    entry={entry}
                    lang={lang}
                    isInstalled={!!amtInstalled.find((m) => m.id === entry.id)?.is_available}
                    dl={amtProgress[entry.id]}
                    onDownload={() => void downloadSoundfont(entry)}
                  />
                ))}
              </div>
            </div>

            {/* Section B: 免费 SFZ 音源（需到官网下载后导入） */}
            <div className="sfm-section sfm-section-sfz">
              <div className="sfm-section-title">
                <span className="sfm-section-dot sfm-section-dot-sfz" />
                {t("soundfont.sfzSection")}
              </div>
              <div className="sfm-section-hint">{t("soundfont.sfzSectionHint")}</div>
              <div className="sfm-rec-grid">
                {RECOMMENDED_SOUNDFONTS.map((r) => {
                  // 根据名称模糊匹配已导入的音源（不区分大小写）
                  const lower = (s: string) => s.toLowerCase();
                  const matched = fonts.find((f) => lower(f.name).includes(lower(r.name).substring(0, 8))
                    || r.matchKeywords.some((kw) => lower(f.id).includes(lower(kw))));
                  return (
                    <SfzSoundfontCard
                      key={r.name}
                      rec={r}
                      isImported={!!matched}
                      onOpenLink={() => void handleOpenLink(r.url)}
                      onImportDir={handleImportDir}
                      purposeLabel={t(r.purposeKey)}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      <PanelResizeHandles start={startResize} />
    </aside>
  );
}

/** 可一键下载 SF2 音源卡片（来自 AMT_CATALOG，复用 amt-models 的下载/进度） */
function AmtSoundfontCard({
  entry, lang, isInstalled, dl, onDownload,
}: {
  entry: AmtCatalogEntry;
  lang: string;
  isInstalled: boolean;
  dl: { downloaded: number; total: number; stage: string } | undefined;
  onDownload: () => void;
}) {
  const { t } = useTranslation();
  const isDownloading = !!dl;
  return (
    <div className={`sfm-rec-card ${isInstalled ? "installed" : ""} ${isDownloading ? "downloading" : ""}`}>
      <div className="sfm-rec-card-header">
        <span className="sfm-rec-name">{t18(entry.name, lang)}</span>
        <span className="sfm-rec-format">{entry.architecture.toUpperCase()}</span>
      </div>
      <p className="sfm-rec-desc">{t18(entry.description, lang)}</p>
      {entry.newbie_tip && (
        <div className="sfm-rec-tip">💡 {t18(entry.newbie_tip, lang)}</div>
      )}
      <div className="sfm-rec-meta">
        <span>{formatSize(entry.fileSize)}</span>
        {entry.rating && (
          <span className="sfm-rec-rating">
            {"★".repeat(entry.rating)}{"☆".repeat(5 - entry.rating)}
          </span>
        )}
        {entry.ui_features && (
          <span>{entry.ui_features.map((f) => t18(f, lang)).join(" / ")}</span>
        )}
      </div>
      {isDownloading && (
        <div className="sfm-rec-progress">
          {dl!.stage === "extract" ? (
            <>
              <div className="sfm-rec-progress-bar" style={{ width: "100%" }} />
              <span className="sfm-rec-progress-text">{t("soundfont.extracting")}</span>
            </>
          ) : (
            <>
              <div
                className="sfm-rec-progress-bar"
                style={{ width: dl!.total > 0 ? `${(dl!.downloaded / dl!.total) * 100}%` : "0%" }}
              />
              <span className="sfm-rec-progress-text">
                {formatSize(dl!.downloaded)} / {dl!.total > 0 ? formatSize(dl!.total) : "..."}
              </span>
            </>
          )}
        </div>
      )}
      <button
        className="sfm-rec-btn"
        disabled={isDownloading || isInstalled}
        onClick={onDownload}
      >
        {isInstalled
          ? `✓ ${t("soundfont.installed")}`
          : isDownloading
            ? t("soundfont.downloading")
            : t("soundfont.download")}
      </button>
    </div>
  );
}

/** SFZ 目录音源卡片（需手动下载后导入） */
function SfzSoundfontCard({
  rec, isImported, onOpenLink, onImportDir, purposeLabel,
}: {
  rec: { name: string; format: string; size: string; url: string; license: string; matchKeywords: string[] };
  isImported: boolean;
  onOpenLink: () => void;
  onImportDir: () => void;
  purposeLabel: string;
}) {
  const { t } = useTranslation();
  return (
    <div className={`sfm-rec-card sfz ${isImported ? "installed" : ""}`}>
      <div className="sfm-rec-card-header">
        <span className="sfm-rec-name">{rec.name}</span>
        <span className="sfm-rec-format sfz">SFZ</span>
      </div>
      <div className="sfm-rec-desc sfz-desc">
        <span className="sfz-purpose-badge">{purposeLabel}</span>
        <span className="sfz-meta-text">
          {rec.size} · {rec.license}
        </span>
      </div>
      <div className="sfm-rec-actions sfz-actions">
        {isImported ? (
          <span className="sfm-rec-installed-tag">✓ {t("soundfont.importedStatus")}</span>
        ) : (
          <>
            <button className="sfm-rec-btn sfz-download" onClick={onOpenLink}>
              {t("soundfont.sourceDownload")}
            </button>
            <button className="sfm-rec-btn sfz-import" onClick={onImportDir}>
              {t("soundfont.importDir")}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
