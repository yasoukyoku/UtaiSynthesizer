import { useState, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useTranslation } from "react-i18next";
import { TRACK_HEIGHT } from "../../lib/constants";
import { VolumeFader } from "../common/VolumeFader";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import * as playback from "../../lib/audio/playback";
import type { Track } from "../../types/project";
import "./TrackList.css";

interface Props {
  width: number;
  scrollY: number;
}

export function TrackList({ width, scrollY }: Props) {
  const { t } = useTranslation();
  const { tracks, updateTrack, removeTrack } = useProjectStore();
  const { activeTrackId, setActiveTrack } = useAppStore();
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; trackId: string } | null>(null);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent, trackId: string) => {
      e.preventDefault();
      e.stopPropagation();
      setCtxMenu({ x: e.clientX, y: e.clientY, trackId });
    },
    [],
  );

  const ctxItems: MenuItem[] = ctxMenu
    ? [
        {
          label: t("tracks.delete"),
          danger: true,
          onClick: () => removeTrack(ctxMenu.trackId),
        },
      ]
    : [];

  return (
    <div className="track-list" style={{ width }}>
      <div className="track-list-scroll" style={{ transform: `translateY(${-scrollY}px)` }}>
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
            onVolumeChange={(v) => {
              updateTrack(track.id, { volumeDb: v });
              playback.updateTrackVolume(track.id, v);
            }}
            onContextMenu={(e) => handleContextMenu(e, track.id)}
          />
        ))}
      </div>

      {ctxMenu && (
        <ContextMenu
          x={ctxMenu.x}
          y={ctxMenu.y}
          items={ctxItems}
          onClose={() => setCtxMenu(null)}
        />
      )}
    </div>
  );
}

interface TrackItemProps {
  track: Track;
  active: boolean;
  onSelect: () => void;
  onMute: () => void;
  onSolo: () => void;
  onVolumeChange: (v: number) => void;
  onContextMenu: (e: React.MouseEvent) => void;
}

function TrackItem({
  track,
  active,
  onSelect,
  onMute,
  onSolo,
  onVolumeChange,
  onContextMenu,
}: TrackItemProps) {
  const colorVar =
    track.trackType === "vocal"
      ? "var(--track-vocal)"
      : track.trackType === "audio"
        ? "var(--track-audio)"
        : "var(--track-instrument)";

  return (
    <div
      className={`track-item ${active ? "active" : ""}`}
      style={{ height: TRACK_HEIGHT }}
      onClick={onSelect}
      onContextMenu={onContextMenu}
    >
      <div className="track-color-bar" style={{ background: colorVar }} />
      <div className="track-info">
        <span className="track-name">{track.name}</span>
      </div>
      <div className="track-controls">
        <VolumeFader value={track.volumeDb} min={-24} max={6} onChange={onVolumeChange} />
        <button
          className={`track-btn ${track.muted ? "active-mute" : ""}`}
          onClick={(e) => {
            e.stopPropagation();
            onMute();
          }}
        >
          M
        </button>
        <button
          className={`track-btn ${track.solo ? "active-solo" : ""}`}
          onClick={(e) => {
            e.stopPropagation();
            onSolo();
          }}
        >
          S
        </button>
      </div>
    </div>
  );
}
