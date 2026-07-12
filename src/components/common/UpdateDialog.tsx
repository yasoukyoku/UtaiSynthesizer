import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../../store/app";
import {
  installUpdate,
  UPDATE_PROGRESS_EVENT,
  UPDATE_INSTALLING_EVENT,
  type UpdateProgress,
} from "../../lib/update";
import { backendErrorMessage, isCancelError } from "../../lib/backendError";
import "./ConfirmDialog.css";

// S64 — "New version available" modal (startup auto-check + Settings manual check both open it via
// app store `updateDialog`). Shell borrows the ConfirmDialog confirm-* frame (the ExportAudioDialog
// NO-duplication pattern). Download progress streams in over the Rust "update-progress" events; a
// successful install EXITS the app (the NSIS updater relaunches the new version), so the only ways
// out of this dialog are Later, an error, or the restart itself.

type Phase = "idle" | "downloading" | "installing" | "error";

function fmtMB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UpdateDialog() {
  const { t } = useTranslation();
  const info = useAppStore((s) => s.updateDialog);
  const closeUpdateDialog = useAppStore((s) => s.closeUpdateDialog);

  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<UpdateProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fresh state each time the dialog opens on a (new) update — the component stays mounted across
  // open/close, so stale "error" from a failed attempt must not leak into the next opening.
  const openSeq = info ? `${info.version}` : null;
  useEffect(() => {
    setPhase("idle");
    setProgress(null);
    setError(null);
  }, [openSeq]);

  const busy = phase === "downloading" || phase === "installing";
  const busyRef = useRef(busy);
  busyRef.current = busy;

  // Publish busy to the store — the quit flows (window X / tray quit) consult it so a mid-flight
  // update is never silently abandoned behind an unreachable confirm (audit S64).
  const setUpdateBusy = useAppStore((s) => s.setUpdateBusy);
  useEffect(() => {
    setUpdateBusy(busy);
    return () => setUpdateBusy(false);
  }, [busy, setUpdateBusy]);

  // Own the keyboard while open (the ConfirmDialog capture pattern) — Esc closes unless mid-update.
  useEffect(() => {
    if (!info) return;
    const onKey = (e: KeyboardEvent) => {
      e.stopPropagation();
      if (e.key === "Escape" && !busyRef.current) closeUpdateDialog();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [info, closeUpdateDialog]);

  // Progress events from update_install (throttled Rust-side).
  useEffect(() => {
    if (!info) return;
    let disposed = false;
    const unlistens: (() => void)[] = [];
    void listen<UpdateProgress>(UPDATE_PROGRESS_EVENT, (e) => setProgress(e.payload)).then((u) => {
      if (disposed) u();
      else unlistens.push(u);
    });
    void listen(UPDATE_INSTALLING_EVENT, () => setPhase("installing")).then((u) => {
      if (disposed) u();
      else unlistens.push(u);
    });
    return () => {
      disposed = true;
      unlistens.forEach((u) => u());
    };
  }, [info]);

  if (!info) return null;

  const handleInstall = async () => {
    if (busyRef.current) return;
    setPhase("downloading");
    setProgress(null);
    setError(null);
    try {
      await installUpdate();
      // Unreachable on Windows: a successful install exits the process (NSIS /P /R relaunches).
    } catch (e) {
      if (isCancelError(e)) {
        // User cancel — back to the offer, silently (the app-wide cancel-sentinel convention).
        setPhase("idle");
        setProgress(null);
        return;
      }
      setError(backendErrorMessage(e) ?? String(e));
      setPhase("error");
    }
  };

  // Downloads are cancellable Rust-side (update_cancel flips a flag; the stall-watchdog loop drops
  // the transfer and restores the pending handle for a later retry). Install is not (point of no return).
  const handleCancelDownload = () => {
    void invoke("update_cancel").catch(() => {});
  };

  const pct = progress && progress.total ? Math.min(100, Math.round((progress.downloaded / progress.total) * 100)) : null;
  const progressLine =
    phase === "installing"
      ? t("update.installing")
      : progress
        ? pct !== null
          ? `${t("update.downloading")} ${pct}% (${fmtMB(progress.downloaded)} / ${fmtMB(progress.total!)})`
          : `${t("update.downloading")} ${fmtMB(progress.downloaded)}`
        : t("update.starting");

  return (
    <div className="confirm-overlay" onMouseDown={busy ? undefined : closeUpdateDialog}>
      <div className="confirm-dialog" role="dialog" aria-modal="true" onMouseDown={(e) => e.stopPropagation()}>
        <div className="confirm-title">{t("update.title")}</div>

        <div className="confirm-body" style={{ marginBottom: 8 }}>
          {t("update.versionLine", { from: info.currentVersion, to: info.version })}
        </div>

        {info.notes && (
          <div
            style={{
              maxHeight: 180,
              overflowY: "auto",
              whiteSpace: "pre-line",
              fontSize: "var(--font-size-sm)",
              color: "var(--text-secondary)",
              background: "var(--bg-base)",
              border: "1px solid var(--border-subtle)",
              padding: "6px 8px",
              marginBottom: 10,
            }}
          >
            {info.notes}
          </div>
        )}

        {!busy && phase !== "error" && (
          <div className="confirm-body" style={{ fontSize: "var(--font-size-sm)", color: "var(--text-muted)", marginBottom: 10 }}>
            {t("update.restartNote")}
          </div>
        )}

        {busy && (
          <div style={{ margin: "6px 0 12px" }}>
            <div style={{ fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginBottom: 5 }}>{progressLine}</div>
            <div style={{ height: 4, background: "var(--bg-base)", border: "1px solid var(--border-strong)" }}>
              <div
                style={{
                  height: "100%",
                  width: phase === "installing" ? "100%" : `${pct ?? 0}%`,
                  background: "var(--accent-primary)",
                  transition: "width 0.12s linear",
                }}
              />
            </div>
          </div>
        )}

        {phase === "error" && error && (
          <div className="confirm-body" style={{ color: "var(--color-error)", marginBottom: 10 }}>
            {error}
          </div>
        )}

        <div className="confirm-buttons">
          {/* Downloading → a REAL cancel (Rust watchdog drops the transfer). Installing is the
              atomic tail: cancel there would be silently ignored — disabled for honesty (S63 lesson). */}
          {phase === "downloading" ? (
            <button className="confirm-btn neutral" onClick={handleCancelDownload}>
              {t("common.cancel")}
            </button>
          ) : (
            <button className="confirm-btn neutral" disabled={busy} onClick={closeUpdateDialog}>
              {t("update.later")}
            </button>
          )}
          <button className="confirm-btn primary" disabled={busy} onClick={() => void handleInstall()}>
            {phase === "error" ? t("update.retry") : t("update.installNow")}
          </button>
        </div>
      </div>
    </div>
  );
}
