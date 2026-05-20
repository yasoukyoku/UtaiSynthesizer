import { useRef, useEffect, useCallback, useState } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { useTranslation } from "react-i18next";
import { TICKS_PER_BEAT, PIXELS_PER_TICK, TRACK_HEIGHT } from "../../lib/constants";
import { importAudioToTrack } from "../../lib/audio/import";
import { ContextMenu, type MenuItem } from "../common/ContextMenu";
import "./Arrangement.css";

const EDGE_ZONE = 6;
const AUTOSCROLL_ZONE = 40;
const AUTOSCROLL_SPEED = 8;

type DragMode = null | "playhead" | "move" | "resizeL" | "resizeR";

interface DragState {
  mode: DragMode;
  trackIdx: number;
  segId: string;
  startMouseX: number;
  startMouseY: number;
  origStartTick: number;
  origDurationTicks: number;
  origOffsetMs: number;
}

interface CtxState {
  x: number;
  y: number;
  trackIdx: number;
  segId: string | null;
}

export function Arrangement() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const {
    tracks, timeSignature, playheadTick, setPlayhead, updateTrack, tempo,
    addTrack, splitSegment, deleteSegment,
  } = useProjectStore();
  const {
    zoom, scrollX, scrollY, setScroll,
    openWorkflow, selectedSegment, selectSegment, clearSelection,
  } = useAppStore();
  const { audioFiles, loadAudioFile } = useAudioStore();
  const { t } = useTranslation();
  const [cursor, setCursor] = useState("default");
  const [dragOver, setDragOver] = useState(false);
  const [ctxMenu, setCtxMenu] = useState<CtxState | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const mouseXRef = useRef(-9999);
  const mouseClientXRef = useRef(0);
  const autoScrollRef = useRef<number>(0);
  const drawRef = useRef(() => {});

  const ppt = PIXELS_PER_TICK * zoom;

  // Refs for latest values (used in document-level handlers)
  const scrollXRef = useRef(scrollX);
  scrollXRef.current = scrollX;
  const pptRef = useRef(ppt);
  pptRef.current = ppt;
  const tracksRef = useRef(tracks);
  tracksRef.current = tracks;
  const tempoRef = useRef(tempo);
  tempoRef.current = tempo;

  const canvasToTick = useCallback(
    (clientX: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return 0;
      const rect = canvas.getBoundingClientRect();
      return Math.max(0, Math.round((clientX - rect.left + scrollXRef.current) / pptRef.current));
    },
    [],
  );

  const hitTest = useCallback(
    (clientX: number, clientY: number):
      { trackIdx: number; segId: string; zone: "body" | "left" | "right" } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left + scrollX;
      const y = clientY - rect.top + scrollY;

      for (let i = 0; i < tracks.length; i++) {
        const track = tracks[i];
        if (!track) continue;
        const trackY = i * TRACK_HEIGHT;
        if (y < trackY || y > trackY + TRACK_HEIGHT) continue;

        // Iterate in reverse: later segments (higher z-order) are hit first
        for (let si = track.segments.length - 1; si >= 0; si--) {
          const seg = track.segments[si]!;
          const sx = seg.startTick * ppt;
          const sw = seg.durationTicks * ppt;
          if (x >= sx && x <= sx + sw) {
            if (x - sx < EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "left" };
            if (sx + sw - x < EDGE_ZONE) return { trackIdx: i, segId: seg.id, zone: "right" };
            return { trackIdx: i, segId: seg.id, zone: "body" };
          }
        }
      }
      return null;
    },
    [tracks, ppt, scrollX, scrollY],
  );

  const hitTrackIdx = useCallback(
    (clientY: number): number => {
      const canvas = canvasRef.current;
      if (!canvas) return -1;
      const rect = canvas.getBoundingClientRect();
      const y = clientY - rect.top + scrollY;
      return Math.floor(y / TRACK_HEIGHT);
    },
    [scrollY],
  );

  // Auto-scroll during drag near edges
  const startAutoScroll = useCallback(() => {
    cancelAnimationFrame(autoScrollRef.current);
    const tick = () => {
      if (!dragRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const localX = mouseClientXRef.current - rect.left;
      let dx = 0;
      if (localX < AUTOSCROLL_ZONE) {
        dx = -AUTOSCROLL_SPEED * (1 - localX / AUTOSCROLL_ZONE);
      } else if (localX > rect.width - AUTOSCROLL_ZONE) {
        dx = AUTOSCROLL_SPEED * (1 - (rect.width - localX) / AUTOSCROLL_ZONE);
      }
      if (dx !== 0) {
        const newX = Math.max(0, scrollXRef.current + dx);
        useAppStore.getState().setScroll(newX, useAppStore.getState().scrollY);

        if (dragRef.current.mode === "playhead") {
          const t = canvasToTick(mouseClientXRef.current);
          useProjectStore.getState().setPlayhead(t);
        }
      }
      autoScrollRef.current = requestAnimationFrame(tick);
    };
    autoScrollRef.current = requestAnimationFrame(tick);
  }, [canvasToTick]);

  const stopAutoScroll = useCallback(() => {
    cancelAnimationFrame(autoScrollRef.current);
  }, []);

  // Document-level drag handlers
  useEffect(() => {
    const onDocMove = (e: MouseEvent) => {
      const drag = dragRef.current;
      mouseClientXRef.current = e.clientX;
      if (!drag) return;

      if (drag.mode === "playhead") {
        setPlayhead(canvasToTick(e.clientX));
        return;
      }

      const deltaPx = e.clientX - drag.startMouseX;
      const deltaTicks = Math.round(deltaPx / pptRef.current);
      const trks = tracksRef.current;
      const srcTrack = trks[drag.trackIdx];
      if (!srcTrack) return;

      // Cross-track move: detect vertical track change
      if (drag.mode === "move") {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const scrollYNow = useAppStore.getState().scrollY;
        const mouseY = e.clientY - rect.top + scrollYNow;
        const targetIdx = Math.max(0, Math.min(trks.length - 1, Math.floor(mouseY / TRACK_HEIGHT)));
        const targetTrack = trks[targetIdx];

        if (targetTrack && targetIdx !== drag.trackIdx
          && srcTrack.trackType === "audio" && targetTrack.trackType === "audio") {
          // Move segment to different track
          const seg = srcTrack.segments.find((s) => s.id === drag.segId);
          if (seg) {
            const movedSeg = { ...seg, startTick: Math.max(0, drag.origStartTick + deltaTicks) };
            updateTrack(srcTrack.id, { segments: srcTrack.segments.filter((s) => s.id !== drag.segId) });
            updateTrack(targetTrack.id, { segments: [...targetTrack.segments, movedSeg] });
            drag.trackIdx = targetIdx;
            useAppStore.getState().selectSegment(targetTrack.id, drag.segId);
          }
          return;
        }

        // Same-track move
        const updated = srcTrack.segments.map((seg) =>
          seg.id === drag.segId
            ? { ...seg, startTick: Math.max(0, drag.origStartTick + deltaTicks) }
            : seg,
        );
        updateTrack(srcTrack.id, { segments: updated });
        return;
      }

      // Resize (no cross-track)
      const updated = srcTrack.segments.map((seg) => {
        if (seg.id !== drag.segId) return seg;
        if (drag.mode === "resizeL") {
          const newStart = Math.max(0, drag.origStartTick + deltaTicks);
          const shrink = newStart - drag.origStartTick;
          const newDur = Math.max(TICKS_PER_BEAT / 4, drag.origDurationTicks - shrink);
          const newOff =
            seg.content.type === "audioClip"
              ? Math.max(0, drag.origOffsetMs + (shrink / TICKS_PER_BEAT) * (60000 / tempoRef.current))
              : 0;
          return {
            ...seg, startTick: newStart, durationTicks: newDur,
            content: seg.content.type === "audioClip" ? { ...seg.content, offsetMs: newOff } : seg.content,
          };
        }
        if (drag.mode === "resizeR") {
          return { ...seg, durationTicks: Math.max(TICKS_PER_BEAT / 4, drag.origDurationTicks + deltaTicks) };
        }
        return seg;
      });
      updateTrack(srcTrack.id, { segments: updated });
    };

    const onDocUp = () => {
      if (dragRef.current) {
        dragRef.current = null;
        stopAutoScroll();
      }
    };

    document.addEventListener("mousemove", onDocMove);
    document.addEventListener("mouseup", onDocUp);
    return () => {
      document.removeEventListener("mousemove", onDocMove);
      document.removeEventListener("mouseup", onDocUp);
    };
  }, [canvasToTick, setPlayhead, updateTrack, stopAutoScroll]);

  // Canvas mousedown (starts drag)
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      const hit = hitTest(e.clientX, e.clientY);
      if (hit) {
        const track = tracks[hit.trackIdx];
        const seg = track?.segments.find((s) => s.id === hit.segId);
        if (!track || !seg) return;

        selectSegment(track.id, seg.id);
        const mode: DragMode = hit.zone === "left" ? "resizeL" : hit.zone === "right" ? "resizeR" : "move";
        const offsetMs = seg.content.type === "audioClip" ? seg.content.offsetMs : 0;
        dragRef.current = {
          mode, trackIdx: hit.trackIdx, segId: hit.segId,
          startMouseX: e.clientX, startMouseY: e.clientY,
          origStartTick: seg.startTick, origDurationTicks: seg.durationTicks, origOffsetMs: offsetMs,
        };
      } else {
        clearSelection();
        dragRef.current = {
          mode: "playhead", trackIdx: -1, segId: "",
          startMouseX: e.clientX, startMouseY: e.clientY,
          origStartTick: 0, origDurationTicks: 0, origOffsetMs: 0,
        };
        setPlayhead(canvasToTick(e.clientX));
      }
      startAutoScroll();
    },
    [hitTest, tracks, setPlayhead, canvasToTick, selectSegment, clearSelection, startAutoScroll],
  );

  // Canvas mousemove (cursor only, drag is handled at document level)
  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (canvas) mouseXRef.current = e.clientX - canvas.getBoundingClientRect().left;

      if (!dragRef.current) {
        const hit = hitTest(e.clientX, e.clientY);
        setCursor(!hit ? "crosshair" : hit.zone !== "body" ? "ew-resize" : "grab");
        drawRef.current();
      }
    },
    [hitTest],
  );

  const handleMouseLeave = useCallback(() => {
    mouseXRef.current = -9999;
    drawRef.current();
  }, []);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit) openWorkflow(hit.segId);
    },
    [hitTest, openWorkflow],
  );

  // Wheel: scroll = horizontal, shift = vertical, ctrl = zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.stopPropagation();
      if (e.ctrlKey) {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        useAppStore.getState().setZoom(zoom * factor);
      } else if (e.shiftKey) {
        setScroll(scrollX, Math.max(0, scrollY + e.deltaY));
      } else {
        setScroll(Math.max(0, scrollX + e.deltaY), scrollY);
      }
    },
    [scrollX, scrollY, zoom, setScroll],
  );

  // Right-click context menu
  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const hit = hitTest(e.clientX, e.clientY);
      const trackIdx = hitTrackIdx(e.clientY);
      if (hit) {
        const track = tracks[hit.trackIdx];
        if (track) selectSegment(track.id, hit.segId);
        setCtxMenu({ x: e.clientX, y: e.clientY, trackIdx: hit.trackIdx, segId: hit.segId });
      } else if (trackIdx >= 0 && trackIdx < tracks.length) {
        setCtxMenu({ x: e.clientX, y: e.clientY, trackIdx, segId: null });
      }
    },
    [hitTest, hitTrackIdx, tracks, selectSegment],
  );

  const ctxItems: MenuItem[] = (() => {
    if (!ctxMenu) return [];
    const track = tracks[ctxMenu.trackIdx];
    if (!track) return [];
    const items: MenuItem[] = [];
    if (ctxMenu.segId) {
      const seg = track.segments.find((s) => s.id === ctxMenu.segId);
      const canSplit = seg && playheadTick > seg.startTick && playheadTick < seg.startTick + seg.durationTicks;
      items.push({
        label: t("toolbar.split"), shortcut: "Ctrl+K", disabled: !canSplit,
        onClick: () => { if (canSplit) splitSegment(track.id, ctxMenu.segId!, playheadTick); },
      });
      items.push({
        label: t("toolbar.delete"), shortcut: "Del",
        onClick: () => { deleteSegment(track.id, ctxMenu.segId!); clearSelection(); },
      });
    }
    items.push({
      label: t("tracks.delete"), danger: true,
      onClick: () => useProjectStore.getState().removeTrack(track.id),
    });
    return items;
  })();

  // Drag-and-drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    setDragOver(true);
  }, []);
  const handleDragLeave = useCallback(() => setDragOver(false), []);
  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const files = Array.from(e.dataTransfer.files).filter((f) => /\.(wav|mp3|flac|ogg)$/i.test(f.name));
      if (!files.length) return;
      const dropTick = canvasToTick(e.clientX);
      for (const file of files) {
        const filePath = (file as any).path as string | undefined;
        if (!filePath) continue;
        try {
          await importAudioToTrack(filePath, tempo, dropTick, tracks, addTrack, loadAudioFile, updateTrack);
        } catch (err) {
          console.error("Drop import failed:", err);
        }
      }
    },
    [canvasToTick, tempo, tracks, addTrack, loadAudioFile, updateTrack],
  );

  // Drawing
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const ticksPerBar = TICKS_PER_BEAT * timeSignature[0];

    ctx.fillStyle = "#0d1220";
    ctx.fillRect(0, 0, width, height);

    if (dragOver) {
      ctx.fillStyle = "rgba(57, 197, 187, 0.06)";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(57, 197, 187, 0.3)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(2, 2, width - 4, height - 4);
      ctx.setLineDash([]);
    }

    const startTick = Math.floor(scrollX / ppt);
    const endTick = Math.ceil((scrollX + width) / ppt);
    for (let tick = startTick - (startTick % TICKS_PER_BEAT); tick < endTick; tick += TICKS_PER_BEAT) {
      const x = tick * ppt - scrollX;
      const isBar = tick % ticksPerBar === 0;
      ctx.strokeStyle = isBar ? "rgba(57,197,187,0.18)" : "rgba(57,197,187,0.05)";
      ctx.lineWidth = isBar ? 1 : 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }

    for (let i = 0; i < tracks.length; i++) {
      const track = tracks[i];
      if (!track) continue;
      const y = i * TRACK_HEIGHT - scrollY;

      ctx.strokeStyle = "rgba(30,42,69,0.8)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, y + TRACK_HEIGHT); ctx.lineTo(width, y + TRACK_HEIGHT); ctx.stroke();

      const c = track.trackType === "audio" ? [96, 165, 250]
        : track.trackType === "vocal" ? [57, 197, 187] : [167, 139, 250];

      // Draw in array order: later entries render on top (higher z-order)
      for (const seg of track.segments) {
        const sx = seg.startTick * ppt - scrollX;
        const sw = seg.durationTicks * ppt;
        const sy = y + 2;
        const sh = TRACK_HEIGHT - 4;

        const isSel = selectedSegment?.trackId === track.id && selectedSegment?.segmentId === seg.id;

        ctx.fillStyle = isSel ? `rgba(${c[0]},${c[1]},${c[2]},0.28)` : `rgba(${c[0]},${c[1]},${c[2]},0.15)`;
        ctx.fillRect(sx, sy, sw, sh);
        ctx.strokeStyle = isSel ? `rgba(${c[0]},${c[1]},${c[2]},0.9)` : `rgba(${c[0]},${c[1]},${c[2]},0.5)`;
        ctx.lineWidth = isSel ? 1.5 : 1;
        ctx.strokeRect(sx, sy, sw, sh);

        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.3)`;
        ctx.fillRect(sx, sy, 3, sh);
        ctx.fillRect(sx + sw - 3, sy, 3, sh);

        if (seg.content.type === "audioClip" && sw > 2) {
          const audio = audioFiles[seg.content.sourcePath];
          if (audio && audio.peaks.length > 0) {
            const offMs = seg.content.offsetMs;
            const totalMs = seg.content.totalDurationMs;
            const segMs = (seg.durationTicks / TICKS_PER_BEAT) * (60000 / tempo);
            drawWaveform(ctx, audio.peaks, sx, sy, sw, sh, `rgba(${c[0]},${c[1]},${c[2]},0.6)`, offMs, totalMs, segMs);
          }
        }

        if (track.muted) {
          ctx.fillStyle = "rgba(13, 18, 32, 0.5)";
          ctx.fillRect(sx, sy, sw, sh);
        }
      }

      // Draw crossfade X markers (need time-sorted order to detect overlaps)
      const timeSorted = [...track.segments].sort((a, b) => a.startTick - b.startTick);
      for (let si = 0; si + 1 < timeSorted.length; si++) {
        const seg = timeSorted[si]!;
        const next = timeSorted[si + 1]!;
        const segEnd = seg.startTick + seg.durationTicks;
        if (next.startTick < segEnd) {
          const overlapEnd = Math.min(segEnd, next.startTick + next.durationTicks);
          const ox = next.startTick * ppt - scrollX;
          const ow = (overlapEnd - next.startTick) * ppt;
          drawCrossfade(ctx, ox, y + 2, ow, TRACK_HEIGHT - 4, c);
        }
      }
    }

    // Playhead
    const phx = playheadTick * ppt - scrollX;
    if (phx >= -1 && phx <= width + 1) {
      const near = Math.abs(mouseXRef.current - phx) < 10;
      if (near) {
        ctx.save(); ctx.shadowColor = "#ff6b9d"; ctx.shadowBlur = 12;
        ctx.strokeStyle = "#ffadc8"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(phx, 0); ctx.lineTo(phx, height); ctx.stroke();
        ctx.restore();
      } else {
        ctx.strokeStyle = "#ff6b9d"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(phx, 0); ctx.lineTo(phx, height); ctx.stroke();
      }
      ctx.fillStyle = near ? "#ffadc8" : "#ff6b9d";
      ctx.beginPath(); ctx.moveTo(phx - 6, 0); ctx.lineTo(phx + 6, 0); ctx.lineTo(phx, 8); ctx.closePath(); ctx.fill();
    }
  }, [tracks, audioFiles, ppt, scrollX, scrollY, playheadTick, timeSignature, tempo, selectedSegment, dragOver]);

  drawRef.current = draw;

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  return (
    <div className="arrangement">
      <canvas
        ref={canvasRef}
        className="arrangement-canvas"
        style={{ cursor }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleContextMenu}
        onWheel={handleWheel}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      />
      {ctxMenu && <ContextMenu x={ctxMenu.x} y={ctxMenu.y} items={ctxItems} onClose={() => setCtxMenu(null)} />}
    </div>
  );
}

function drawWaveform(
  ctx: CanvasRenderingContext2D, peaks: number[],
  x: number, y: number, w: number, h: number, color: string,
  offsetMs: number, totalDurationMs: number, segDurationMs: number,
) {
  const midY = y + h / 2;
  const amp = h / 2 - 1;
  if (totalDurationMs <= 0 || peaks.length === 0) return;
  const startRatio = offsetMs / totalDurationMs;
  const endRatio = Math.min(1, (offsetMs + segDurationMs) / totalDurationMs);
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
  for (let px = 0; px < w; px++) {
    const ratio = startRatio + (px / w) * (endRatio - startRatio);
    const peakIdx = Math.min(Math.floor(ratio * peaks.length), peaks.length - 1);
    if (peakIdx < 0) continue;
    const peak = peaks[peakIdx] ?? 0;
    ctx.moveTo(x + px, midY - peak * amp); ctx.lineTo(x + px, midY + peak * amp);
  }
  ctx.stroke();
}

function drawCrossfade(
  ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, c: number[],
) {
  if (w < 2) return;
  ctx.save(); ctx.globalAlpha = 0.25;
  ctx.strokeStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + w, y + h);
  ctx.moveTo(x, y + h); ctx.lineTo(x + w, y); ctx.stroke();
  ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.08)`; ctx.fillRect(x, y, w, h);
  ctx.restore();
}
