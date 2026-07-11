import { useState, useCallback, useEffect, useRef, Fragment } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useHistoryStore } from "../../store/history";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { convertFileSrc } from "@tauri-apps/api/core";
import { LANE_HEIGHT, LANE_GROUP_BAR_HEIGHT, TRACK_HEADER_HEIGHT, FADER_MIN_DB, FADER_MAX_DB, AUDIO_EXTENSIONS } from "../../lib/constants";
import { computeTrackHeight, computeTrackYOffsets, computeTotalTracksHeight, findTrackAtY, getLanes, getLaneLayout, isLaneRowMuted, laneControlFor, type LaneGroupRun, type LaneMember } from "../../lib/trackLayout";
import { laneLabelParts } from "../../lib/audio/laneOps";
import { trackTypeCssVar, LANE_COLORS } from "../../lib/trackColors";
import { importAudioToNewTrack } from "../../lib/audio/import";
import { blankTrack } from "../../lib/trackFactory";
import { VolumeFader, formatPan, formatDb } from "../common/VolumeFader";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import * as playback from "../../lib/audio/playback";
import type { Track } from "../../types/project";
import "./TrackList.css";

interface Props {
  width: number;
}

/** Context menu in the track-header column: per-track actions, or "add material" at a boundary. */
type Menu = { x: number; y: number } & (
  | { kind: "track"; trackId: string }
  | { kind: "add"; index: number }
);

export function TrackList({ width }: Props) {
  const { t } = useTranslation();
  // Per-field selectors so this column re-renders only on values it shows — NOT on playheadTick
  // (every playback frame) or scrollX (horizontal scroll). scrollY self-subscribed for the
  // vertical transform.
  const tracks = useProjectStore((s) => s.tracks);
  const updateTrack = useProjectStore((s) => s.updateTrack);
  const removeTrack = useProjectStore((s) => s.removeTrack);
  const toggleTrackExpanded = useProjectStore((s) => s.toggleTrackExpanded);
  const updateLaneControl = useProjectStore((s) => s.updateLaneControl);
  const setLaneMute = useProjectStore((s) => s.setLaneMute);
  const setTrackPlayOriginal = useProjectStore((s) => s.setTrackPlayOriginal);
  const addTrack = useProjectStore((s) => s.addTrack);
  const activeTrackId = useAppStore((s) => s.activeTrackId);
  const setActiveTrack = useAppStore((s) => s.setActiveTrack);
  const scrollY = useAppStore((s) => s.scrollY);
  const vZoom = useAppStore((s) => s.vZoom);
  const ghostInsert = useAppStore((s) => s.ghostInsert);
  const [menu, setMenu] = useState<Menu | null>(null);
  const [hoverBoundary, setHoverBoundary] = useState<number | null>(null);
  const [addMenuOpen, setAddMenuOpen] = useState(false);
  const [editingTrackId, setEditingTrackId] = useState<string | null>(null);
  const [draggingTrackId, setDraggingTrackId] = useState<string | null>(null);
  const addRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const trackDragRef = useRef<{ trackId: string; startY: number; dragging: boolean } | null>(null);

  useEffect(() => {
    if (!addMenuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (addRef.current && !addRef.current.contains(e.target as Node)) setAddMenuOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [addMenuOpen]);

  // The boundary hint is computed from the cursor vs. the track layout; when the layout shifts
  // under a stationary cursor (vertical zoom or scroll), drop the stale hint until the pointer
  // moves again — otherwise the line sticks to the wrong boundary.
  useEffect(() => { setHoverBoundary(null); }, [vZoom, scrollY]);

  // Drag-reorder tracks by their header. Starts only past a small threshold (so a plain click still
  // selects); live-reorders as the cursor crosses track midpoints.
  const onTrackHeaderMouseDown = useCallback((e: React.MouseEvent, trackId: string) => {
    if (e.button !== 0) return;
    const el = e.target as HTMLElement;
    if (el.closest(".track-btn, .track-expand-btn, .track-name, .track-name-input, .vol-fader, .track-row-bot")) return;
    trackDragRef.current = { trackId, startY: e.clientY, dragging: false };
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = trackDragRef.current;
      if (!d) return;
      if (!d.dragging) {
        if (Math.abs(e.clientY - d.startY) <= 4) return;
        d.dragging = true;
        setDraggingTrackId(d.trackId);
        document.body.style.cursor = "grabbing";
        document.body.style.userSelect = "none"; // no stray text selection across track names
        // Reorder fires live per midpoint cross — coalesce the whole drag into one undo step.
        useHistoryStore.getState().beginTransaction();
      }
      const el = listRef.current;
      if (!el) return;
      const proj = useProjectStore.getState();
      const trks = proj.tracks;
      const fromIdx = trks.findIndex((t) => t.id === d.trackId);
      if (fromIdx < 0) return;
      const vz = useAppStore.getState().vZoom;
      const contentY = e.clientY - el.getBoundingClientRect().top + useAppStore.getState().scrollY;
      // Target = how many OTHER tracks' midpoints sit above the cursor. Excluding the dragged track
      // gives proper hysteresis (using its own slot as the only dead-band oscillates when it is
      // shorter than the track it crosses).
      const offs = computeTrackYOffsets(trks, vz);
      let target = 0;
      for (let i = 0; i < trks.length; i++) {
        if (i === fromIdx) continue;
        if (contentY >= offs[i]! + computeTrackHeight(trks[i]!, vz) / 2) target++;
      }
      if (target !== fromIdx) proj.reorderTrack(fromIdx, target);
    };
    const onUp = () => {
      if (trackDragRef.current) {
        const wasDragging = trackDragRef.current.dragging;
        trackDragRef.current = null;
        setDraggingTrackId(null);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        if (wasDragging) useHistoryStore.getState().commitTransaction();
      }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, []);

  const commitRename = useCallback((trackId: string, name: string) => {
    const n = name.trim();
    if (n) updateTrack(trackId, { name: n });
    setEditingTrackId(null);
  }, [updateTrack]);

  const offsets = computeTrackYOffsets(tracks, vZoom);
  const totalH = computeTotalTracksHeight(tracks, vZoom);

  // Reusable track-creation actions. `insertIndex` positions the new track at a boundary (the
  // right-click "add material" menu); omitting it appends at the bottom (the "+" menu).
  const createAudioTrack = useCallback((insertIndex?: number) => {
    setAddMenuOpen(false);
    const n = useProjectStore.getState().tracks.filter((tk) => tk.trackType === "audio").length + 1;
    addTrack(blankTrack(crypto.randomUUID(), `Audio ${n}`, "audio"), insertIndex);
  }, [addTrack]);

  const createVocalTrack = useCallback((insertIndex?: number) => {
    setAddMenuOpen(false);
    const n = useProjectStore.getState().tracks.filter((tk) => tk.trackType === "vocal").length + 1;
    addTrack(blankTrack(crypto.randomUUID(), `Vocal ${n}`, "vocal"), insertIndex);
  }, [addTrack]);

  const importAudioAt = useCallback(async (insertIndex?: number) => {
    setAddMenuOpen(false);
    const path = await open({
      title: t("toolbar.importAudio"),
      filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
    });
    if (!path) return;
    // import.ts creates the track + loading segment immediately + owns decode/error handling.
    void importAudioToNewTrack(path as string, useProjectStore.getState().playheadTick, undefined, insertIndex);
  }, [t]);

  // Boundary nearest the cursor (insert index 0..tracks.length) within a small threshold, else null.
  const boundaryAt = useCallback((clientY: number): number | null => {
    const el = listRef.current;
    if (!el) return null;
    const contentY = clientY - el.getBoundingClientRect().top + scrollY;
    for (let i = 0; i <= tracks.length; i++) {
      const by = i < tracks.length ? offsets[i]! : totalH;
      if (Math.abs(contentY - by) <= 5) return i;
    }
    return null;
  }, [tracks.length, offsets, totalH, scrollY]);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const b = boundaryAt(e.clientY);
      if (b !== null) {
        setMenu({ kind: "add", index: b, x: e.clientX, y: e.clientY });
        return;
      }
      const el = listRef.current;
      if (!el) return;
      const contentY = e.clientY - el.getBoundingClientRect().top + scrollY;
      const idx = findTrackAtY(offsets, contentY);
      if (idx >= 0 && idx < tracks.length && contentY <= totalH) {
        setMenu({ kind: "track", trackId: tracks[idx]!.id, x: e.clientX, y: e.clientY });
      } else {
        // Empty area below the last track → "add material", appending at the bottom.
        setMenu({ kind: "add", index: tracks.length, x: e.clientX, y: e.clientY });
      }
    },
    [boundaryAt, offsets, totalH, scrollY, tracks],
  );

  const menuItems: MenuItem[] = (() => {
    if (!menu) return [];
    if (menu.kind === "track") {
      return [
        { label: t("tracks.rename"), onClick: () => setEditingTrackId(menu.trackId) },
        { label: t("tracks.delete"), danger: true, onClick: () => removeTrack(menu.trackId) },
      ];
    }
    return [
      { label: t("toolbar.importAudio"), onClick: () => importAudioAt(menu.index) },
      { label: t("toolbar.addAudio"), onClick: () => createAudioTrack(menu.index) },
      { label: t("toolbar.addMidi"), onClick: () => createVocalTrack(menu.index) },
    ];
  })();

  return (
    <div
      className="track-list"
      style={{ width }}
      ref={listRef}
      onContextMenu={handleContextMenu}
      onMouseMove={(e) => {
        if (trackDragRef.current) { if (hoverBoundary !== null) setHoverBoundary(null); return; }
        const b = boundaryAt(e.clientY);
        setHoverBoundary((prev) => (prev === b ? prev : b));
      }}
      onMouseLeave={() => setHoverBoundary(null)}
    >
      <div className="track-list-scroll" style={{ transform: `translateY(${-scrollY}px)` }}>
        {tracks.length === 0 && (
          <div className="track-list-empty">
            <span className="text-muted">{t("tracks.empty")}</span>
          </div>
        )}
        {hoverBoundary !== null && (
          <div
            className="track-boundary-hint"
            style={{ top: hoverBoundary < tracks.length ? offsets[hoverBoundary]! : totalH }}
          />
        )}
        {tracks.map((track, i) => (
          <Fragment key={track.id}>
            {ghostInsert && ghostInsert.index === i && (
              <div
                className="track-ghost-slot"
                style={{ height: ghostInsert.count * TRACK_HEADER_HEIGHT * vZoom }}
              />
            )}
          <TrackItem
            track={track}
            vZoom={vZoom}
            hasSolo={tracks.some((tk) => tk.solo)}
            active={track.id === activeTrackId}
            dragging={track.id === draggingTrackId}
            editing={track.id === editingTrackId}
            onHeaderMouseDown={(e) => onTrackHeaderMouseDown(e, track.id)}
            onStartRename={() => setEditingTrackId(track.id)}
            onCommitRename={(name) => commitRename(track.id, name)}
            onCancelRename={() => setEditingTrackId(null)}
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
            onPanChange={(v) => {
              updateTrack(track.id, { pan: v });
              playback.updateTrackPan(track.id, v);
            }}
            onToggleExpand={() => toggleTrackExpanded(track.id)}
            onTogglePlayOriginal={() => setTrackPlayOriginal(track.id, !track.playOriginal)}
            onLaneMute={(members) => {
              // A merged row toggles as ONE: "muted" reads as all-members-muted, and the write fans
              // out over every member rowKey — inside one transaction so the click is one undo step
              // (each setLaneMute is a separate store set that would otherwise auto-capture).
              const newMuted = !members.every((m) => isLaneRowMuted(track, m.rowKey, m.laneId));
              useHistoryStore.getState().beginTransaction();
              for (const m of members) {
                setLaneMute(track.id, m.rowKey, newMuted);
                playback.updateLaneMute(track.id, m.rowKey, newMuted, laneControlFor(track, m.groupId, m.laneId)?.volumeDb ?? 0);
              }
              useHistoryStore.getState().commitTransaction();
            }}
            onLaneVolumeChange={(run, v) => {
              // Fan out over every member 组 under this bar (merged rows stay in lockstep; the legacy
              // laneId seed applies only to the primary 组 — other members' legacy entries key by
              // THEIR laneIds and simply start fresh, converging on this first touch).
              for (const gid of run.groupIds) {
                updateLaneControl(track.id, gid, { volumeDb: v }, gid === run.groupId ? run.laneId : undefined);
                playback.updateLaneVolume(track.id, gid, v);
              }
            }}
            onLanePanChange={(run, v) => {
              for (const gid of run.groupIds) {
                updateLaneControl(track.id, gid, { pan: v }, gid === run.groupId ? run.laneId : undefined);
                playback.updateLanePan(track.id, gid, v);
              }
            }}
          />
          </Fragment>
        ))}

        <div className="track-add" ref={addRef}>
          <button className="track-add-btn" onClick={() => setAddMenuOpen((o) => !o)}>
            <span className="track-add-icon">+</span>
            <span>{t("toolbar.addTrack")}</span>
          </button>
          {addMenuOpen && (
            <div className="track-add-menu">
              <button className="track-add-option" onClick={() => importAudioAt()}>
                {t("toolbar.importAudio")}
              </button>
              <button className="track-add-option" onClick={() => createAudioTrack()}>
                {t("toolbar.addAudio")}
              </button>
              <button className="track-add-option" onClick={() => createVocalTrack()}>
                {t("toolbar.addMidi")}
              </button>
            </div>
          )}
        </div>
      </div>

      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          items={menuItems}
          onClose={() => setMenu(null)}
        />
      )}
    </div>
  );
}

interface TrackItemProps {
  track: Track;
  vZoom: number;
  hasSolo: boolean;
  active: boolean;
  dragging: boolean;
  editing: boolean;
  onHeaderMouseDown: (e: React.MouseEvent) => void;
  onStartRename: () => void;
  onCommitRename: (name: string) => void;
  onCancelRename: () => void;
  onSelect: () => void;
  onMute: () => void;
  onSolo: () => void;
  onVolumeChange: (v: number) => void;
  onPanChange: (v: number) => void;
  onToggleExpand: () => void;
  onTogglePlayOriginal: () => void;
  onLaneMute: (members: LaneMember[]) => void;
  onLaneVolumeChange: (run: LaneGroupRun, v: number) => void;
  onLanePanChange: (run: LaneGroupRun, v: number) => void;
}

function TrackItem({
  track,
  vZoom,
  hasSolo,
  active,
  dragging,
  editing,
  onHeaderMouseDown,
  onStartRename,
  onCommitRename,
  onCancelRename,
  onSelect,
  onMute,
  onSolo,
  onVolumeChange,
  onPanChange,
  onToggleExpand,
  onTogglePlayOriginal,
  onLaneMute,
  onLaneVolumeChange,
  onLanePanChange,
}: TrackItemProps) {
  const { t } = useTranslation();
  const colorVar = trackTypeCssVar(track.trackType);

  const lanes = getLanes(track);
  const laneLayout = getLaneLayout(track);
  const hasLanes = lanes.length > 0;
  const totalHeight = computeTrackHeight(track, vZoom);
  const isEmpty = track.segments.length === 0;
  // The left "indicator light": lit + glowing when the track is actually AUDIBLE (has content, not
  // muted, and — when any track is soloed — this track is the/one of the soloed ones); off (dimmed)
  // when muted, empty, or silenced by another track's solo.
  const lit = !isEmpty && !track.muted && (!hasSolo || track.solo);

  return (
    <div
      className={`track-item-group ${active ? "active" : ""} ${isEmpty ? "empty" : ""} ${dragging ? "dragging" : ""} ${track.playOriginal ? "play-original" : ""}`}
      style={{ height: totalHeight }}
    >
      <div
        className="track-item"
        onClick={onSelect}
        onMouseDown={onHeaderMouseDown}
        style={{ height: TRACK_HEADER_HEIGHT * vZoom }}
      >
        {/* Two rows inside the fixed 48px header: the NAME owns the TOP row (it was squeezed to nothing
            between the expand button and the faders on expandable tracks), the V/P faders + readouts sit
            side-by-side on the BOTTOM row. Total height unchanged, so canvas/lane geometry is untouched. */}
        <div className="track-row-top">
          {hasLanes && (
            <button
              className="track-expand-btn"
              onClick={(e) => { e.stopPropagation(); onToggleExpand(); }}
            >
              {track.expanded ? "▼" : "▶"}
            </button>
          )}
          <div
            className="track-color-bar"
            style={{
              background: colorVar,
              opacity: lit ? 1 : 0.28,
              boxShadow: lit ? `0 0 6px ${colorVar}` : "none",
            }}
          />
          {track.voiceModelAvatar && (
            <div className="track-avatar">
              <img src={convertFileSrc(track.voiceModelAvatar)} alt="" />
            </div>
          )}
          <div className="track-info">
            {editing ? (
              <RenameInput initial={track.name} onCommit={onCommitRename} onCancel={onCancelRename} />
            ) : (
              <span
                className="track-name"
                title={track.name}
                onDoubleClick={(e) => { e.stopPropagation(); onStartRename(); }}
              >
                {track.name}
              </span>
            )}
          </div>
          {/* SOURCE selector (only meaningful once lanes exist): lit = the ORIGINAL audio plays and the
              sub-lanes leave the output entirely — a Mute/Solo-class state, not just monitoring. */}
          {hasLanes && (
            <button
              className={`track-btn ${track.playOriginal ? "active-orig" : ""}`}
              title={t("tracks.playOriginal")}
              onClick={(e) => { e.stopPropagation(); onTogglePlayOriginal(); }}
            >
              O
            </button>
          )}
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
        <div className="track-row-bot">
          <div className="fader-row">
            <span className="fader-tag">V</span>
            <VolumeFader
              value={track.volumeDb}
              min={FADER_MIN_DB}
              max={FADER_MAX_DB}
              onChange={onVolumeChange}
              onGestureStart={() => useHistoryStore.getState().beginTransaction()}
              onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
            />
            <span className="fader-val">{formatDb(track.volumeDb, FADER_MIN_DB)}</span>
          </div>
          <div className="fader-row">
            <span className="fader-tag">P</span>
            <VolumeFader
              value={track.pan}
              min={-1}
              max={1}
              step={0.1}
              fillFrom="center"
              format={formatPan}
              onChange={onPanChange}
              onGestureStart={() => useHistoryStore.getState().beginTransaction()}
              onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
            />
            <span className="fader-val">{formatPan(track.pan)}</span>
          </div>
        </div>
      </div>
      {track.expanded && laneLayout.runs.map((run) => {
        // One GROUP BLOCK per 组+名 run (getLaneLayout — same geometry the canvas rows use): a slim
        // group BAR carrying the 轨道组 name + the group-level volume/pan (keyed by the 组 via
        // laneControlFor — all rows of one 组 share the mix; 解组 for independent control), then the
        // member rows with just the stem name + per-ROW mute (isLaneRowMuted, loose row semantics).
        const ctrl = laneControlFor(track, run.groupId, run.laneId);
        const laneRgb = LANE_COLORS[run.colorIndex % LANE_COLORS.length]!;
        return (
          <div key={run.key} className="lane-group" style={{ "--lane-rgb": laneRgb } as React.CSSProperties}>
            <div className="lane-group-bar" style={{ height: LANE_GROUP_BAR_HEIGHT * vZoom }}>
              <span className="lane-group-swatch" />
              <span className="lane-group-name" title={run.name}>{run.name}</span>
              <div className="track-controls lane-group-controls">
                <div className="fader-row">
                  <span className="fader-tag">V</span>
                  <VolumeFader
                    value={ctrl?.volumeDb ?? 0}
                    min={FADER_MIN_DB}
                    max={FADER_MAX_DB}
                    width={42}
                    onChange={(v) => onLaneVolumeChange(run, v)}
                    onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                    onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                  />
                  <span className="fader-val">{formatDb(ctrl?.volumeDb ?? 0, FADER_MIN_DB)}</span>
                </div>
                <div className="fader-row">
                  <span className="fader-tag">P</span>
                  <VolumeFader
                    value={ctrl?.pan ?? 0}
                    min={-1}
                    max={1}
                    step={0.1}
                    fillFrom="center"
                    format={formatPan}
                    width={28}
                    onChange={(v) => onLanePanChange(run, v)}
                    onGestureStart={() => useHistoryStore.getState().beginTransaction()}
                    onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
                  />
                  <span className="fader-val">{formatPan(ctrl?.pan ?? 0)}</span>
                </div>
              </div>
            </div>
            {lanes.slice(run.start, run.start + run.count).map(({ id, label, members }) => {
              // A merged row reads muted only when ALL members are (the toggle fans out, so they
              // only diverge via legacy state — the canvas dims each piece by its own member truth).
              const muted = members.every((m) => isLaneRowMuted(track, m.rowKey, m.laneId));
              // Show only the sub-name within the group (labels are "Group · stem") — the group name
              // lives on the bar above, and the bracket ties the rows to it (members, not peers).
              const subName = laneLabelParts(label).stem ?? label;
              return (
                <div key={id} className="lane-item" style={{ height: LANE_HEIGHT * vZoom }}>
                  <span className="lane-label" title={label}>{subName}</span>
                  <div className="track-controls">
                    <button
                      className={`track-btn ${muted ? "active-mute" : ""}`}
                      onClick={(e) => { e.stopPropagation(); onLaneMute(members); }}
                    >
                      M
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

/** Inline track-name editor. Commits once on blur (Enter blurs → commit; Escape cancels). */
function RenameInput({ initial, onCommit, onCancel }: {
  initial: string;
  onCommit: (name: string) => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const doneRef = useRef(false);
  return (
    <input
      ref={ref}
      className="track-name-input"
      autoFocus
      defaultValue={initial}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === "Enter") ref.current?.blur();
        else if (e.key === "Escape") { doneRef.current = true; onCancel(); }
      }}
      onBlur={(e) => { if (doneRef.current) return; doneRef.current = true; onCommit(e.target.value); }}
    />
  );
}
