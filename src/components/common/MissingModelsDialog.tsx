import { useEffect } from "react";
import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useAppStore } from "../../store/app";
import { useMsstModelStore } from "../../store/msst-models";
import { hfBaseForMirror } from "../../lib/models/msst-catalog";
import { backendErrorMessage } from "../../lib/backendError";
import "./ConfirmDialog.css";
import "./MissingModelsDialog.css";

/**
 * S66 pre-run model dialog — the "don't make users guess" surface. The workflow/vocal
 * preflight (engine.collectMissingModels / preflightVocalModels) aborts the run and opens
 * this instead of letting MSST_MODEL_NOT_CONVERTED / AUX_FILE_MISSING explode mid-run.
 * Each row carries its own one-click action:
 *   - unconverted MSST model → convert (serial app-wide via the Rust convert slot)
 *   - model file not installed → open the resource manager
 *   - core asset pack missing → start the download (progress lives in Settings → Model Assets)
 * The user re-runs after the fixes — the dialog deliberately does NOT auto-restart the run.
 */
export function MissingModelsDialog() {
  const items = useAppStore((s) => s.missingModels);
  const close = useAppStore((s) => s.closeMissingModels);
  const { t } = useTranslation();
  const installed = useMsstModelStore((s) => s.installed);
  const downloading = useMsstModelStore((s) => s.downloading);
  const error = useMsstModelStore((s) => s.error);
  const convertPrecision = useMsstModelStore((s) => s.convertPrecision);
  const fetchInstalled = useMsstModelStore((s) => s.fetchInstalled);
  const mirror = useMsstModelStore((s) => s.mirror);
  const [auxStarted, setAuxStarted] = useState(false);

  // Fresh conversion state for the rows (the store may never have fetched this session).
  useEffect(() => {
    if (items) void fetchInstalled();
  }, [items, fetchInstalled]);

  useEffect(() => {
    if (!items) return;
    const onKey = (e: KeyboardEvent) => {
      e.stopPropagation();
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [items, close]);

  if (!items) return null;

  const anyConverting = Object.values(downloading).some((d) => d.stage === "converting");

  return (
    <div className="confirm-overlay" onMouseDown={close}>
      <div className="confirm-dialog mm-dialog" role="dialog" aria-modal="true" onMouseDown={(e) => e.stopPropagation()}>
        <div className="confirm-title">{t("missingModels.title")}</div>
        <div className="confirm-body">{t("missingModels.body")}</div>
        <div className="mm-list">
          {items.map((it, i) => {
            if (it.kind === "msstConvert") {
              const entry = installed.find((m) => m.filename === it.filename);
              const done = !!entry && (entry.has_onnx || entry.has_fp16);
              const converting = it.filename ? downloading[it.filename]?.stage === "converting" : false;
              return (
                <div key={i} className="mm-row">
                  <span className="mm-label" title={it.label}>{it.label}</span>
                  {done ? (
                    <span className="mm-status mm-done">{t("missingModels.converted")}</span>
                  ) : converting ? (
                    <span className="mm-status">{t("missingModels.converting")}</span>
                  ) : (
                    <button
                      className="mm-btn"
                      disabled={anyConverting}
                      onClick={() => {
                        if (it.filename) void convertPrecision(it.filename, it.precision, it.architecture);
                      }}
                    >
                      {t("missingModels.convert")}
                    </button>
                  )}
                </div>
              );
            }
            if (it.kind === "msstMissing") {
              return (
                <div key={i} className="mm-row">
                  <span className="mm-label" title={it.label}>{it.label}</span>
                  <span className="mm-status">{t("missingModels.notInstalled")}</span>
                  <button
                    className="mm-btn"
                    onClick={() => {
                      if (!useAppStore.getState().modelManagerOpen) useAppStore.getState().toggleModelManager();
                    }}
                  >
                    {t("missingModels.openManager")}
                  </button>
                </div>
              );
            }
            // auxPack — label 即 pack id(aux-inference / aux-autotune,S73 泛化;
            // 显示文案按 pack 选,下载走同一 download_asset_pack 漏斗)
            const packId = it.label || "aux-inference";
            return (
              <div key={i} className="mm-row">
                <span className="mm-label">
                  {packId === "aux-autotune" ? t("missingModels.autotunePack") : t("startup.compAux")}
                </span>
                {auxStarted ? (
                  <span className="mm-status">{t("missingModels.downloadStarted")}</span>
                ) : (
                  <button
                    className="mm-btn"
                    onClick={() => {
                      setAuxStarted(true);
                      void invoke("download_asset_pack", {
                        id: packId,
                        hfBase: hfBaseForMirror(mirror),
                      }).catch(() => {
                        /* busy/cancel/fail surface in Settings → Model Assets */
                      });
                    }}
                  >
                    {t("missingModels.download")}
                  </button>
                )}
              </div>
            );
          })}
        </div>
        {error && <div className="mm-error">{backendErrorMessage(error) ?? error}</div>}
        <div className="confirm-body mm-hint">{t("missingModels.hint")}</div>
        <div className="confirm-buttons">
          <button className="confirm-btn neutral" onClick={close}>
            {t("missingModels.close")}
          </button>
        </div>
      </div>
    </div>
  );
}
