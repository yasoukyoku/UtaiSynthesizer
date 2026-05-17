import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useTranslation } from "react-i18next";
import type { Track } from "../../types/project";
import "./TrackList.css";

interface Props {
  width: number;
}

export function TrackList({ width }: Props) {
  const { t } = useTranslation();
  const { tracks, updateTrack } = useProjectStore();
  const { activeTrackId, setActiveTrack } = useAppStore();

  return (
    <div className="track-list" style={{ width }}>
      <div className="track-list-header">
        <span>{t("tracks.title")}</span>
      </div>
      <div className="track-list-body">
        {tracks.length === 0 && (
          <div className="track-list-empty">
            <span className="text-muted">{t("tracks.empty")}</span>
          </div>
        )}
        {tracks.map((track) => (
          <TrackItem
            key={track.id}
            track={track}
            active={track.id === activeTrackId}
            onSelect={() => setActiveTrack(track.id)}
            onMute={() => updateTrack(track.id, { muted: !track.muted })}
            onSolo={() => updateTrack(track.id, { solo: !track.solo })}
          />
        ))}
      </div>
    </div>
  );
}

interface TrackItemProps {
  track: Track;
  active: boolean;
  onSelect: () => void;
  onMute: () => void;
  onSolo: () => void;
}

function TrackItem({ track, active, onSelect, onMute, onSolo }: TrackItemProps) {
  const colorVar =
    track.trackType === "vocal"
      ? "var(--track-vocal)"
      : track.trackType === "audio"
        ? "var(--track-audio)"
        : "var(--track-instrument)";

  return (
    <div
      className={`track-item ${active ? "active" : ""}`}
      onClick={onSelect}
    >
      <div className="track-color-bar" style={{ background: colorVar }} />
      <div className="track-info">
        <span className="track-name">{track.name}</span>
        <span className="track-type text-muted">{track.trackType}</span>
      </div>
      <div className="track-controls">
        <button
          className={`track-btn ${track.muted ? "active-mute" : ""}`}
          onClick={(e) => { e.stopPropagation(); onMute(); }}
        >
          M
        </button>
        <button
          className={`track-btn ${track.solo ? "active-solo" : ""}`}
          onClick={(e) => { e.stopPropagation(); onSolo(); }}
        >
          S
        </button>
      </div>
    </div>
  );
}
