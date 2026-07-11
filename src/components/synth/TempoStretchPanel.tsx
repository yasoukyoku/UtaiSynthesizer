// TempoStretchPanel — S59 Tempo Slider (速度推子): per-segment offline time-stretch, BPM-anchored
// when the clip has a detected grid (target-BPM slider), plain speed-% otherwise. Apply is
// artifacts-first-then-commit (applySegmentStretch), so the button shows a processing state and
// the store change (undoable, reschedules playback) lands only when every stem is stretched.

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { applySegmentStretch, canonStretch } from "../../lib/audio/stretchCache";
import { tempoErrorMessage } from "../../lib/audio/tempoDetect";
import { ParamSlider } from "../workflow/nodes/ParamSlider";
import "./TempoStretchPanel.css";

/** Duration-factor travel. Deliberately WIDE (⅓×–3×, §user: 自由度给用户打开,推坏了自己会降) —
 *  the engine's official sweet spot is 0.75–1.5×, beyond it artifacts grow but nothing breaks;
 *  the backend still hard-clamps [0.25, 4]. */
const F_MIN = 1 / 3;
const F_MAX = 3;

export function TempoStretchPanel({ x, y, trackId, segId, onClose }: {
  x: number;
  y: number;
  trackId: string;
  segId: string;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const seg = useProjectStore((s) => s.tracks.find((tr) => tr.id === trackId)?.segments.find((sg) => sg.id === segId));
  const clip = seg?.content.type === "audioClip" ? seg.content : null;
  const r = clip?.stretch ?? 1;
  const td = clip?.tempoDetect;
  // target = effective BPM (grid detected) or playback speed % (no grid); seeded from the current factor
  const [target, setTarget] = useState<number>(() => (td ? td.bpm / r : 100 / r));
  const [busy, setBusy] = useState(false);
  // re-seed when the underlying values change EXTERNALLY while the panel is open (global Ctrl+Z
  // reaches the app through the backdrop) — a stale target would display one value and apply
  // another after the clamp (audit).
  useEffect(() => {
    setTarget(td ? td.bpm / r : 100 / r);
  }, [td, td?.bpm, r]);

  if (!clip || !seg || seg.loading) return null;

  // duration factor from the chosen target (r = played/source duration)
  const factor = canonStretch(Math.min(F_MAX, Math.max(F_MIN, td ? td.bpm / target : 100 / target)));
  const changed = Math.abs(factor - r) >= 1e-6;

  const run = (f: number) => {
    setBusy(true);
    applySegmentStretch(trackId, segId, f)
      .then(() => onClose())
      .catch((e) => {
        useAppStore.getState().showToast(tempoErrorMessage(e));
        setBusy(false);
      });
  };

  const min = td ? td.bpm / F_MAX : 100 / F_MAX;
  const max = td ? td.bpm / F_MIN : 100 / F_MIN;
  const pw = 240;
  const px = Math.max(4, Math.min(x, window.innerWidth - pw - 8));
  const py = Math.max(4, Math.min(y, window.innerHeight - 150));

  return (
    <>
      <div className="stretch-backdrop" onMouseDown={onClose} onContextMenu={(e) => { e.preventDefault(); onClose(); }} />
      <div className="stretch-panel" style={{ left: px, top: py, width: pw }} onMouseDown={(e) => e.stopPropagation()}>
        <div className="stretch-title">{t("tempo.stretchTitle")}</div>
        <ParamSlider
          label={td ? t("tempo.targetBpm") : t("tempo.speedPct")}
          min={Math.round(min * 10) / 10}
          max={Math.round(max * 10) / 10}
          step={td ? 0.1 : 0.5}
          value={Math.round(target * 10) / 10}
          onChange={setTarget}
          format={(v) => (td ? v.toFixed(1) : `${v.toFixed(0)}%`)}
          disabled={busy}
        />
        <div className="stretch-readout">
          {td && <span>{t("tempo.detectedBpm")}: {td.bpm.toFixed(1)}</span>}
          <span>{t("tempo.durationPct")}: {(factor * 100).toFixed(1)}%</span>
          {r !== 1 && <span>{t("tempo.current")}: ×{r.toFixed(3)}</span>}
        </div>
        <div className="stretch-actions">
          <button className="stretch-btn" disabled={busy || !changed} onClick={() => run(factor)}>
            {busy ? t("tempo.processing") : t("tempo.apply")}
          </button>
          <button className="stretch-btn" disabled={busy || r === 1} onClick={() => run(1)}>
            {t("tempo.reset")}
          </button>
        </div>
      </div>
    </>
  );
}
