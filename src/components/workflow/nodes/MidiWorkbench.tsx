import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

import { useTranslation } from "react-i18next";
import { useAmtStore } from "../../../store/amt";
import { useAppStore } from "../../../store/app";
import { t18 } from "../../../lib/models/msst-catalog";
import "../../common/ConfirmDialog.css";

interface AmtMidiMetadata {
  track_count: number;
  total_notes: number;
  ppq: number;
  bpm: number | null;
  time_signature: [number, number] | null;
  tracks: { name: string; note_count: number; program: number | null }[];
  size_bytes: number;
}

const QUANTIZE_GRIDS = ["off", "1/4", "1/8", "1/16", "1/32", "1/64"] as const;

export function MidiWorkbench({ nodeId, onClose }: { nodeId: string; onClose: () => void }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language;
  const run = useAmtStore((s) => s.runs[nodeId]);
  const setRun = useAmtStore((s) => s.setRun);

  const [meta, setMeta] = useState<AmtMidiMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [quantizeGrid, setQuantizeGrid] = useState<string>("off");
  const [customBpm, setCustomBpm] = useState<string>("");
  const [rerunning, setRerunning] = useState(false);
  const [reProgress, setReProgress] = useState<number | null>(null);

  const midiPath = run?.midiPath ?? "";

  const loadMeta = useCallback(async () => {
    if (!midiPath) return;
    setLoading(true);
    setError(null);
    try {
      const m = await invoke<AmtMidiMetadata>("amt_midi_metadata", { midiPath });
      setMeta(m);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [midiPath]);

  useEffect(() => {
    void loadMeta();
  }, [loadMeta, midiPath]);

  const rerun = useCallback(
    async (opts: { quantizeGrid?: string; customBpm?: number | null }) => {
      if (!run) return;
      setRerunning(true);
      setReProgress(0);
      setError(null);
      const outDir = midiPath.substring(0, midiPath.lastIndexOf("\\")) ||
        midiPath.substring(0, midiPath.lastIndexOf("/")) || ".";
      try {
        const res = await invoke<{ midi_path: string; total_notes: number | null; processing_time_secs: number }>(
          "run_amt_midi",
          {
            audioPath: run.audioPath,
            midiMode: run.mode,
            transcriptionBackend: ["smart", "vocal_split", "six_stem_split"].includes(run.mode)
              ? run.backend
              : null,
            yourmt3Model: null,
            muscriptorModel: null,
            midiTrackMode: null,
            tempoMode: opts.customBpm != null ? "fixed_manual" : "fixed_auto",
            customBpm: opts.customBpm,
            quantizeNotes: opts.quantizeGrid !== "off",
            quantizeGrid: opts.quantizeGrid === "off" ? null : opts.quantizeGrid,
            useGpu: true,
            gpuDevice: 0,
            outputDir: outDir,
            nodeId,
          },
        );
        setRun({
          ...run,
          midiPath: res.midi_path,
          totalNotes: res.total_notes,
          processingTimeSecs: res.processing_time_secs,
        });
        setReProgress(1);
      } catch (e) {
        setError(String(e));
      } finally {
        setRerunning(false);
        setTimeout(() => setReProgress(null), 800);
      }
    },
    [run, midiPath, nodeId, setRun],
  );

  // Export EVERY track of the merged MIDI as its own .mid into a user-chosen folder.
  // Fallback: if the merged file has a single track, one file is still written into that folder.
  const handleExportFolder = useCallback(async () => {
    if (!midiPath) return;
    const { open } = await import("@tauri-apps/plugin-dialog");
    let dir = await open({ directory: true, title: t("amt.exportFolderTitle") });
    if (!dir) return;
    // multi-selection off → returns a single string; normalize an array just in case.
    if (Array.isArray(dir)) dir = dir[0];
    if (typeof dir !== "string") return;

    try {
      const written = await invoke<string[]>("export_midi_tracks_to_folder", {
        midiPath,
        folder: dir,
      });
      useAppStore
        .getState()
        .showToast(
          t18(
            { zh: `已导出 ${written.length} 个轨道`, en: `Exported ${written.length} track(s)`, ja: `${written.length} トラックをエクスポート` },
            lang,
          ),
          "success",
        );
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
    }
  }, [midiPath, lang, t]);

  const fmtSize = (b: number) => {
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / 1024 / 1024).toFixed(2)} MB`;
  };

  return (
    <div className="confirm-overlay" onMouseDown={onClose}>
      <div
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
        style={{ maxWidth: 560 }}
      >
        <div className="confirm-title">{t("amt.workbenchTitle")}</div>

        {loading && <div className="confirm-body">{t("amt.loadingMeta")}</div>}

        {error && (
          <div className="confirm-body" style={{ color: "var(--danger)" }}>
            {error}
          </div>
        )}

        {meta && (
          <div className="confirm-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 13 }}>
              <div>
                <span style={{ color: "var(--text-tertiary)" }}>{t("amt.totalNotes")}: </span>
                <b>{meta.total_notes}</b>
              </div>
              <div>
                <span style={{ color: "var(--text-tertiary)" }}>{t("amt.tracks")}: </span>
                <b>{meta.track_count}</b>
              </div>
              <div>
                <span style={{ color: "var(--text-tertiary)" }}>{t("amt.bpm")}: </span>
                <b>{meta.bpm != null ? meta.bpm.toFixed(1) : "—"}</b>
              </div>
              <div>
                <span style={{ color: "var(--text-tertiary)" }}>{t("amt.timeSig")}: </span>
                <b>{meta.time_signature ? `${meta.time_signature[0]}/${meta.time_signature[1]}` : "—"}</b>
              </div>
              <div>
                <span style={{ color: "var(--text-tertiary)" }}>PPQ: </span>
                <b>{meta.ppq}</b>
              </div>
              <div>
                <span style={{ color: "var(--text-tertiary)" }}>{t("amt.fileSize")}: </span>
                <b>{fmtSize(meta.size_bytes)}</b>
              </div>
            </div>

            <div style={{ maxHeight: 160, overflowY: "auto", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: 6 }}>
              {meta.tracks.map((tr, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "2px 4px" }}>
                  <span style={{ color: "var(--text-secondary)" }}>{tr.name}</span>
                  <span>{tr.note_count} {t("amt.notes")}</span>
                </div>
              ))}
            </div>

            {/* Re-quantize */}
            <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>{t("amt.requantize")}</div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <select
                  value={quantizeGrid}
                  onChange={(e) => setQuantizeGrid(e.target.value)}
                  style={{ flex: 1, padding: "6px 8px", borderRadius: 6, border: "1px solid var(--border-subtle)", background: "var(--bg-surface)", color: "var(--text-primary)" }}
                >
                  {QUANTIZE_GRIDS.map((g) => (
                    <option key={g} value={g}>{g === "off" ? t("amt.quantizeOff") : g}</option>
                  ))}
                </select>
                <button
                  className="confirm-btn neutral"
                  disabled={rerunning}
                  onClick={() => void rerun({ quantizeGrid })}
                >
                  {t("amt.apply")}
                </button>
              </div>
            </div>

            {/* Re-tempo */}
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>{t("amt.retempo")}</div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input
                  type="number"
                  min={20}
                  max={300}
                  step={0.1}
                  placeholder={t("amt.bpmPlaceholder")}
                  value={customBpm}
                  onChange={(e) => setCustomBpm(e.target.value)}
                  style={{ flex: 1, padding: "6px 8px", borderRadius: 6, border: "1px solid var(--border-subtle)", background: "var(--bg-surface)", color: "var(--text-primary)" }}
                />
                <button
                  className="confirm-btn neutral"
                  disabled={rerunning || !customBpm}
                  onClick={() => void rerun({ customBpm: customBpm ? parseFloat(customBpm) : null })}
                >
                  {t("amt.apply")}
                </button>
                <button
                  className="confirm-btn neutral"
                  disabled={rerunning}
                  onClick={() => void rerun({ customBpm: null })}
                  title={t("amt.resetTempo")}
                >
                  {t("amt.auto")}
                </button>
              </div>
            </div>

            {reProgress != null && (
              <div style={{ fontSize: 12, color: "var(--accent)" }}>
                {rerunning ? t("amt.processing") : t("amt.done")}
              </div>
            )}
          </div>
        )}

        <div className="confirm-buttons">
          <button className="confirm-btn neutral" onClick={onClose}>
            {t18({ zh: "关闭", en: "Close", ja: "閉じる" }, lang)}
          </button>
          <button className="confirm-btn primary" onClick={() => void handleExportFolder()} disabled={!midiPath}>
            {t("amt.exportAll")}
          </button>
        </div>
      </div>
    </div>
  );
}
