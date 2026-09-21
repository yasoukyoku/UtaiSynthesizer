// BatchMixPanel — S9 多选批量「统一音量/声像」：对包含所选片段的每条轨道统一写入相同的
// 音量(dB) 与声像(pan)。复用 TempoStretchPanel 的浮层样式；应用走既有的 updateTrack 历史可见
// 写入 + updateTrackVolume/updateTrackPan 播放链路，保证与单轨推子行为一致。

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useProjectStore } from "../../store/project";
import * as playback from "../../lib/audio/playback";
import { ParamSlider } from "../workflow/nodes/ParamSlider";
import "./TempoStretchPanel.css";

const VOL_MIN = -24;
const VOL_MAX = 12;
const PAN_MIN = -1;
const PAN_MAX = 1;

export function BatchMixPanel({ x, y, trackIds, onClose }: {
  x: number;
  y: number;
  /** 包含所选片段的轨道 id（统一写入这些轨道的音量/声像）。 */
  trackIds: string[];
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const tracks = useProjectStore((s) => s.tracks);
  const [volumeDb, setVolumeDb] = useState(0);
  const [pan, setPan] = useState(0);

  const users = tracks.filter((tr) => trackIds.includes(tr.id));
  const canApply = users.length > 0;

  const apply = () => {
    for (const tr of users) {
      useProjectStore.getState().updateTrack(tr.id, { volumeDb, pan });
      playback.updateTrackVolume(tr.id, volumeDb);
      playback.updateTrackPan(tr.id, pan);
    }
    onClose();
  };

  const pw = 240;
  const px = Math.max(4, Math.min(x, window.innerWidth - pw - 8));
  const py = Math.max(4, Math.min(y, window.innerHeight - 190));

  return (
    <>
      <div className="stretch-backdrop" onMouseDown={onClose} onContextMenu={(e) => { e.preventDefault(); onClose(); }} />
      <div className="stretch-panel" style={{ left: px, top: py, width: pw }} onMouseDown={(e) => e.stopPropagation()}>
        <div className="stretch-title">{t("menu.batchMixTitle")} · {users.length}</div>
        <div className="stretch-readout">
          <span>{t("menu.batchMixTargets")}: {users.map((u) => u.name).join(", ") || "—"}</span>
        </div>
        <ParamSlider
          label={t("menu.resetVolume")}
          min={VOL_MIN}
          max={VOL_MAX}
          step={0.5}
          value={volumeDb}
          onChange={setVolumeDb}
          format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`}
        />
        <ParamSlider
          label={t("menu.pan")}
          min={PAN_MIN}
          max={PAN_MAX}
          step={0.05}
          value={pan}
          onChange={setPan}
          format={(v) => (v === 0 ? "C" : v < 0 ? `L${Math.round(-v * 100)}` : `R${Math.round(v * 100)}`)}
        />
        <div className="stretch-actions">
          <button className="stretch-btn" disabled={!canApply} onClick={apply}>
            {t("menu.apply")}
          </button>
        </div>
      </div>
    </>
  );
}