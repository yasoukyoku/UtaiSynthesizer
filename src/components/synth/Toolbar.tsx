import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useTranslation } from "react-i18next";
import "./Toolbar.css";

export function Toolbar() {
  const { t } = useTranslation();
  const { openWorkflow } = useAppStore();
  const { tempo, setTempo, playheadTick, timeSignature } = useProjectStore();

  return (
    <div className="toolbar">
      <div className="toolbar-section transport">
        <button className="transport-btn" data-tooltip={t("transport.stop")}>
          ⏹
        </button>
        <button className="transport-btn play" data-tooltip={t("transport.play")}>
          ▶
        </button>
        <button className="transport-btn" data-tooltip={t("transport.record")}>
          ⏺
        </button>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section tempo-section">
        <label className="toolbar-label">{t("toolbar.bpm")}</label>
        <input
          type="number"
          className="tempo-input mono"
          value={tempo}
          min={20}
          max={400}
          step={1}
          onChange={(e) => setTempo(Number(e.target.value))}
        />
      </div>

      <div className="toolbar-section time-sig">
        <span className="mono time-display">
          {timeSignature[0]}/{timeSignature[1]}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section position-section">
        <label className="toolbar-label">{t("toolbar.position")}</label>
        <span className="mono position-display">
          {formatPosition(playheadTick, tempo, timeSignature)}
        </span>
      </div>

      <div className="toolbar-divider" />

      <div className="toolbar-section snap-section">
        <label className="toolbar-label">{t("toolbar.snap")}</label>
        <select className="snap-select">
          <option value="4">1/4</option>
          <option value="8">1/8</option>
          <option value="16" selected>1/16</option>
          <option value="32">1/32</option>
          <option value="triplet">Triplet</option>
          <option value="free">{t("toolbar.free")}</option>
        </select>
      </div>

      <div className="toolbar-spacer" />

      <div className="toolbar-section">
        <button className="toolbar-btn" data-tooltip={t("toolbar.addTrack")}>
          + Track
        </button>
        <button
          className="toolbar-btn"
          onClick={() => openWorkflow("demo-segment")}
          data-tooltip={t("workflow.title")}
        >
          ⚙ Workflow
        </button>
      </div>
    </div>
  );
}

function formatPosition(
  tick: number,
  _tempo: number,
  timeSig: [number, number]
): string {
  const ticksPerBeat = 480;
  const ticksPerBar = ticksPerBeat * timeSig[0];
  const bar = Math.floor(tick / ticksPerBar) + 1;
  const beat = Math.floor((tick % ticksPerBar) / ticksPerBeat) + 1;
  const sub = Math.floor(((tick % ticksPerBar) % ticksPerBeat) / (ticksPerBeat / 4));
  return `${bar}:${beat}:${sub.toString().padStart(2, "0")}`;
}
