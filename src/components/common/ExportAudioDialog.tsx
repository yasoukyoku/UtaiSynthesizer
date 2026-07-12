import { useEffect, useRef, useState, type ReactNode } from "react";
import { save as saveDialog } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { useWorkflowStore } from "../../store/workflow";
import { useVoiceModelStore } from "../../store/voice-models";
import { runAudioExport, type AudioExportParams, type ExportPhase } from "../../lib/audio/exportAudio";
import { backendErrorMessage, isBusyError } from "../../lib/backendError";
import "./ConfirmDialog.css";

// S63 — Audio-export dialog (File → Export Audio). Styling deliberately JOINS existing selector groups
// instead of forking new ones (NO-duplication): the modal shell reuses ConfirmDialog's confirm-* frame,
// the option pills reuse Settings' settings-source-opt radio-label group. The mixdown itself lives in
// lib/audio/exportAudio.ts (render) + Rust export_audio.rs (encode) — this file is pure form + phases.

const FORMATS = ["wav", "flac", "mp3", "ogg", "opus", "m4a"] as const;
type Format = (typeof FORMATS)[number];
const RATES = [44100, 48000] as const;
const WAV_DEPTHS = ["16", "24", "32f"] as const;
const FLAC_DEPTHS = ["16", "24"] as const;
const BITRATES = [128, 160, 192, 256, 320] as const;

const isLossy = (f: Format) => f === "mp3" || f === "ogg" || f === "opus" || f === "m4a";

/** Map the export pipeline's stable codes (frontend renderMixdown EXPORT_* + Rust encode codes) to a
 *  localized line — payload-first is irrelevant here (no user-content payloads), so plain substring
 *  checks then delegation to THE shared backend mapper (the vocalRenderErrorMessage pattern). */
function exportErrorMessage(e: unknown, t: (k: string) => string): string {
  const msg = e instanceof Error ? e.message : String(e);
  const local: Record<string, string> = {
    EXPORT_VOCALS_FAILED: "export.errVocals",
    EXPORT_VOCALS_UNRENDERED: "export.errUnrendered",
    EXPORT_SOURCE_LOADING: "export.errLoading",
    EXPORT_SOURCE_MISSING: "export.errMissing",
    EXPORT_DECODE_FAIL: "export.errDecode",
    EXPORT_TOO_LONG: "export.errTooLong",
    EXPORT_EMPTY: "export.errEmpty",
  };
  for (const code of Object.keys(local)) {
    if (msg.includes(code)) return t(local[code]!);
  }
  return backendErrorMessage(e) ?? `${t("export.errFailed")}: ${msg}`;
}

export function ExportAudioDialog({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation();
  const projectName = useProjectStore((s) => s.name);

  const [format, setFormat] = useState<Format>("wav");
  const [rate, setRate] = useState<number>(44100);
  const [depth, setDepth] = useState<string>("16");
  const [bitrate, setBitrate] = useState<number>(320);
  const [phase, setPhase] = useState<ExportPhase | null>(null);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef(false);

  // Same live gates as Settings' cacheCleanBlocked family — an export must not overlap a job that is
  // mid-write into the render cache / holds the voice guard (the dirty-vocal pre-render needs it free).
  const isPlaying = useAudioStore((s) => s.isPlaying);
  const vocalRenderActive = useAppStore((s) => s.vocalRenderActive);
  const anyWorkflowRunning = useWorkflowStore((s) => Object.values(s.executions).some((e) => e.status === "running"));
  const midiExtracting = useAppStore((s) => Object.keys(s.midiExtracting).length > 0);
  const rangeTesting = useVoiceModelStore((s) => Object.keys(s.rangeTesting).length > 0);
  const decoding = useAudioStore((s) => s.loadingPaths.length > 0);
  const blocked = isPlaying || vocalRenderActive || anyWorkflowRunning || midiExtracting || rangeTesting || decoding;

  // Own the keyboard while open (the ConfirmDialog capture pattern) — Esc closes unless mid-export.
  const busyRef = useRef(busy);
  busyRef.current = busy;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      e.stopPropagation();
      if (e.key === "Escape" && !busyRef.current) onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const depths: readonly string[] = format === "wav" ? WAV_DEPTHS : format === "flac" ? FLAC_DEPTHS : [];
  const pickFormat = (f: Format) => {
    setFormat(f);
    // Keep the depth valid for the new format (flac has no 32f).
    if (f === "flac" && depth === "32f") setDepth("24");
  };

  const handleExport = async () => {
    if (busy || blocked) return;
    const base = (projectName || "export").replace(/[<>:"/\\|?*]/g, "_");
    const outPath = await saveDialog({
      title: t("export.audioTitle"),
      defaultPath: `${base}.${format}`,
      filters: [{ name: format.toUpperCase(), extensions: [format] }],
    });
    if (!outPath || typeof outPath !== "string") return;

    setBusy(true);
    abortRef.current = false;
    try {
      const params: AudioExportParams = {
        outPath,
        format,
        sampleRate: rate,
        bitDepth: depth as AudioExportParams["bitDepth"],
        bitrateKbps: bitrate,
      };
      const res = await runAudioExport(params, setPhase, () => abortRef.current);
      if (res.cancelled) {
        useAppStore.getState().showToast(t("export.cancelled"), "info");
        return;
      }
      const name = outPath.replace(/^.*[\\/]/, "");
      useAppStore.getState().showToast(`${t("export.done")} · ${name}`, "success");
      if (res.peak > 1) useAppStore.getState().showToast(t("export.clipWarn"), "info");
      onClose();
    } catch (e) {
      useAppStore.getState().showToast(exportErrorMessage(e, t), isBusyError(e) ? "info" : "error");
    } finally {
      setBusy(false);
      setPhase(null);
    }
  };

  const handleCancel = () => {
    if (!busy) return onClose();
    abortRef.current = true;
    // cancel_voice ONLY during the vocal pre-render phase (the Toolbar pattern — aborts the in-flight
    // GPU render). Past that phase the latch is not ours to pull: a manual sidebar render started while
    // we're in mix/encode would be killed by a stray global cancel (the flag alone ends us between stages).
    if (phase?.kind === "vocals") void invoke("cancel_voice").catch(() => {});
  };

  const phaseLine = !phase
    ? null
    : phase.kind === "vocals"
      ? `${t("export.phaseVocals")} (${phase.total})`
      : phase.kind === "mix"
        ? t("export.phaseMix")
        : t("export.phaseEncode");
  const mixFrac = phase?.kind === "mix" ? phase.frac : phase?.kind === "encode" ? 1 : 0;

  const optRow = (label: string, children: ReactNode) => (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginBottom: 4 }}>{label}</div>
      {/* NOT .settings-source (that container is flex-COLUMN — Settings' vertical radio list); the
          pills themselves reuse .settings-source-opt, laid out as a wrapping row here. */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>{children}</div>
    </div>
  );
  const pill = (name: string, active: boolean, label: string, onPick: () => void) => (
    <label key={label} className={`settings-source-opt ${active ? "active" : ""}`}>
      <input type="radio" name={name} checked={active} disabled={busy} onChange={onPick} />
      <span>{label}</span>
    </label>
  );

  return (
    <div className="confirm-overlay" onMouseDown={busy ? undefined : onClose}>
      <div className="confirm-dialog" role="dialog" aria-modal="true" onMouseDown={(e) => e.stopPropagation()}>
        <div className="confirm-title">{t("export.audioTitle")}</div>

        {optRow(t("export.format"), FORMATS.map((f) => pill("exp-fmt", format === f, f.toUpperCase(), () => pickFormat(f))))}
        {optRow(t("export.sampleRate"), RATES.map((r) => pill("exp-rate", rate === r, `${r / 1000} kHz`, () => setRate(r))))}
        {depths.length > 0 &&
          optRow(t("export.bitDepth"), depths.map((d) => pill("exp-depth", depth === d, d === "32f" ? "32-bit float" : `${d}-bit`, () => setDepth(d))))}
        {isLossy(format) &&
          optRow(t("export.bitrate"), BITRATES.map((b) => pill("exp-rate-k", bitrate === b, `${b} kbps`, () => setBitrate(b))))}

        {blocked && !busy && (
          <div className="confirm-body" style={{ marginBottom: 12, color: "var(--accent-tertiary)" }}>
            {isPlaying ? t("export.blockedPlaying") : t("common.busyRetry")}
          </div>
        )}

        {busy && (
          <div style={{ margin: "6px 0 12px" }}>
            <div style={{ fontSize: "var(--font-size-sm)", color: "var(--text-secondary)", marginBottom: 5 }}>{phaseLine}</div>
            <div style={{ height: 4, background: "var(--bg-base)", border: "1px solid var(--border-strong)" }}>
              <div
                style={{
                  height: "100%",
                  width: `${Math.round(mixFrac * 100)}%`,
                  background: "var(--accent-primary)",
                  transition: "width 0.12s linear",
                }}
              />
            </div>
          </div>
        )}

        <div className="confirm-buttons">
          {/* The encode tail is atomic (raw-body hop + blocking encode, seconds) — a cancel there would
              be silently ignored and still end in a success toast; disable it for honesty (audit). */}
          <button className="confirm-btn neutral" disabled={busy && phase?.kind === "encode"} onClick={handleCancel}>
            {t("common.cancel")}
          </button>
          <button className="confirm-btn primary" disabled={busy || blocked} onClick={() => void handleExport()}>
            {t("export.start")}
          </button>
        </div>
      </div>
    </div>
  );
}
