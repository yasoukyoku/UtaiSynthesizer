import { useState, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useTranslation } from "react-i18next";
import { LANE_HEIGHT } from "../../lib/constants";
import { computeTrackHeight, getLaneLabels } from "../../lib/trackLayout";
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
  const { tracks, updateTrack, removeTrack, toggleTrackExpanded, updateLaneControl } = useProjectStore();
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
            onMute={() => {
              updateTrack(track.id, { muted: !track.muted });
              playback.updateTrackAudibility(useProjectStore.getState().tracks);
            }}
            onSolo={() => {
              updateTrack(track.id, { solo: !track.solo });
              playback.updateTrackAudibility(useProjectStore.getState().tracks);
            }}
            onVolumeChange={(v) => {
              updateTrack(track.id, { volumeDb: v });
              playback.updateTrackVolume(track.id, v);
            }}
            onToggleExpand={() => toggleTrackExpanded(track.id)}
            onLaneMute={(label) => {
              const ctrl = track.laneControls[label];
              const newMuted = !(ctrl?.muted ?? false);
              updateLaneControl(track.id, label, { muted: newMuted });
              playback.updateLaneMute(track.id, label, newMuted, ctrl?.volumeDb ?? 0);
            }}
            onLaneVolumeChange={(label, v) => {
              updateLaneControl(track.id, label, { volumeDb: v });
              playback.updateLaneVolume(track.id, label, v);
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
  onToggleExpand: () => void;
  onLaneMute: (label: string) => void;
  onLaneVolumeChange: (label: string, v: number) => void;
  onContextMenu: (e: React.MouseEvent) => void;
}

function TrackItem({
  track,
  active,
  onSelect,
  onMute,
  onSolo,
  onVolumeChange,
  onToggleExpand,
  onLaneMute,
  onLaneVolumeChange,
  onContextMenu,
}: TrackItemProps) {
  const colorVar =
    track.trackType === "vocal"
      ? "var(--track-vocal)"
      : track.trackType === "audio"
        ? "var(--track-audio)"
        : "var(--track-instrument)";

  const laneLabels = getLaneLabels(track);
  const hasLanes = laneLabels.length > 0;
  const totalHeight = computeTrackHeight(track);

  return (
    <div
      className={`track-item-group ${active ? "active" : ""}`}
      style={{ height: totalHeight }}
      onContextMenu={onContextMenu}
    >
      <div className="track-item" onClick={onSelect}>
        {hasLanes && (
          <button
            className="track-expand-btn"
            onClick={(e) => { e.stopPropagation(); onToggleExpand(); }}
          >
            {track.expanded ? "▼" : "▶"}
          </button>
        )}
        <div className="track-color-bar" style={{ background: colorVar }} />
        {track.voiceModelAvatar && (
          <div className="track-avatar">
            <img src={`https://asset.localhost/${track.voiceModelAvatar.replace(/\\/g, "/")}`} alt="" />
          </div>
        )}
        <div className="track-info">
          <span className="track-name">{track.name}</span>
        </div>
        <div className="track-controls">
          <VolumeFader value={track.volumeDb} min={-24} max={6} onChange={onVolumeChange} />
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
      {track.expanded && laneLabels.map((label) => {
        const ctrl = track.laneControls[label];
        return (
          <div key={label} className="lane-item" style={{ height: LANE_HEIGHT }}>
            <span className="lane-label">{label}</span>
            <div className="track-controls">
              <VolumeFader value={ctrl?.volumeDb ?? 0} min={-24} max={6} onChange={(v) => onLaneVolumeChange(label, v)} />
              <button
                className={`track-btn ${ctrl?.muted ? "active-mute" : ""}`}
                onClick={(e) => { e.stopPropagation(); onLaneMute(label); }}
              >
                M
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
