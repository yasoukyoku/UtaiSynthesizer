import { useRef, useEffect, useCallback } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { TICKS_PER_BEAT, PIXELS_PER_TICK } from "../../lib/constants";
import { collectSnapTicks, snapTick, SNAP_PX } from "../../lib/snapping";
import { drawBeatGrid, drawPlayhead, CANVAS_BORDER } from "../../lib/canvasDraw";
import "./TimelineRuler.css";

// Edge auto-scroll while dragging the playhead near the ruler's left/right edge (mirrors Arrangement).
const AUTOSCROLL_ZONE = 48;
const AUTOSCROLL_SPEED = 14; // px/frame at the edge; ramps up to 2× when the pointer goes past it

export function TimelineRuler() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Per-field selectors — re-render only on tempo/timeSignature change (NOT on playheadTick,
  // which would re-render every playback frame). scroll/zoom/playhead drive the canvas via subs.
  const tempo = useProjectStore((s) => s.tempo);
  const timeSignature = useProjectStore((s) => s.timeSignature);
  const setPlayhead = useProjectStore((s) => s.setPlayhead);
  const dragging = useRef(false);
  const mouseClientXRef = useRef(0); // latest pointer X, so the auto-scroll tick can re-seek while held still
  const autoScrollRef = useRef(0);

  // View refs — driven imperatively by store subscriptions, not React render.
  const scrollXRef = useRef(useAppStore.getState().scrollX);
  const zoomRef = useRef(useAppStore.getState().zoom);
  const pptRef = useRef(PIXELS_PER_TICK * zoomRef.current);
  const playheadRef = useRef(useProjectStore.getState().playheadTick);
  const drawRef = useRef(() => {});

  const redrawRafRef = useRef(0);
  const requestRedraw = useCallback(() => {
    if (redrawRafRef.current) return;
    redrawRafRef.current = requestAnimationFrame(() => {
      redrawRafRef.current = 0;
      drawRef.current();
    });
  }, []);

  const clickToTick = useCallback(
    (clientX: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return 0;
      const rect = canvas.getBoundingClientRect();
      return Math.max(0, Math.round((clientX - rect.left + scrollXRef.current) / pptRef.current));
    },
    [],
  );

  // Seek tick at a pointer X — snapped to clip edges when playhead-snap is enabled.
  const seekTickAt = useCallback(
    (clientX: number) => {
      let tick = clickToTick(clientX);
      if (useAppStore.getState().snapPlayhead) {
        tick = snapTick(tick, collectSnapTicks(useProjectStore.getState().tracks), SNAP_PX / pptRef.current);
      }
      return Math.max(0, tick);
    },
    [clickToTick],
  );

  // Edge auto-scroll while the playhead is dragged to the ruler's edge: scroll the view and keep the
  // playhead pinned to the edge (the pointer is held still, so no mousemove fires — the rAF re-seeks).
  const startAutoScroll = useCallback(() => {
    cancelAnimationFrame(autoScrollRef.current);
    const tick = () => {
      if (!dragging.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const localX = mouseClientXRef.current - rect.left;
      let dx = 0;
      if (localX < AUTOSCROLL_ZONE) {
        dx = -AUTOSCROLL_SPEED * Math.min(2, (AUTOSCROLL_ZONE - localX) / AUTOSCROLL_ZONE);
      } else if (localX > rect.width - AUTOSCROLL_ZONE) {
        dx = AUTOSCROLL_SPEED * Math.min(2, (localX - (rect.width - AUTOSCROLL_ZONE)) / AUTOSCROLL_ZONE);
      }
      if (dx !== 0) {
        const st = useAppStore.getState();
        // No content-based max clamp — allow scrubbing the playhead past the last clip into empty space
        // (mouseup always stops this rAF, so there's no runaway). Minimap/scrollbar stay content-based.
        const newX = Math.max(0, st.scrollX + dx);
        if (newX !== st.scrollX) {
          st.setScroll(newX, st.scrollY); // sub updates scrollXRef synchronously → seekTickAt sees newX
          setPlayhead(seekTickAt(mouseClientXRef.current));
        }
      }
      autoScrollRef.current = requestAnimationFrame(tick);
    };
    autoScrollRef.current = requestAnimationFrame(tick);
  }, [seekTickAt, setPlayhead]);

  const stopAutoScroll = useCallback(() => {
    cancelAnimationFrame(autoScrollRef.current);
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      dragging.current = true;
      mouseClientXRef.current = e.clientX;
      setPlayhead(seekTickAt(e.clientX));
      // Seek during playback: set the seeking flag so the Toolbar rAF stops auto-advancing
      // (otherwise it clobbers the drag every frame). Audio reschedules from the new position
      // when the drag is released.
      if (useAudioStore.getState().isPlaying) {
        useAudioStore.getState().setSeeking(true);
      }
      startAutoScroll();
    },
    [seekTickAt, setPlayhead, startAutoScroll],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      mouseClientXRef.current = e.clientX;
      if (!dragging.current) return;
      // Playback may START mid-drag (Space is global): pin `seeking` so the transport rAF doesn't
      // fight the drag and the release reschedules — mousedown only set it if already playing
      // (mirrors OverviewMap's drag handler).
      const a = useAudioStore.getState();
      if (a.isPlaying && !a.seeking) a.setSeeking(true);
      setPlayhead(seekTickAt(e.clientX));
    };
    const onUp = () => {
      if (!dragging.current) return;
      dragging.current = false;
      stopAutoScroll();
      if (useAudioStore.getState().seeking) {
        useAudioStore.getState().setSeeking(false);
      }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      stopAutoScroll();
    };
  }, [seekTickAt, setPlayhead, stopAutoScroll]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = devicePixelRatio;
    const { width, height } = canvas.getBoundingClientRect();
    const cw = Math.round(width * dpr);
    const ch = Math.round(height * dpr);
    // Only reallocate the backing store on size change — setting canvas.width every draw
    // (i.e. every scroll frame) forces a clear + GPU realloc and is a major scroll-jank source.
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const scrollX = scrollXRef.current;
    const ppt = pptRef.current;
    const playheadTick = playheadRef.current;

    const ticksPerBar = TICKS_PER_BEAT * timeSignature[0];
    const secsPerBar = (60.0 / tempo) * timeSignature[0];

    ctx.fillStyle = "#1a2236";
    ctx.fillRect(0, 0, width, height);

    const startTick = Math.floor(scrollX / ppt);
    const endTick = Math.ceil((scrollX + width) / ppt);
    const startBar = Math.floor(startTick / ticksPerBar);
    const endBar = Math.ceil(endTick / ticksPerBar);

    drawBeatGrid(ctx, { ppt, scrollX, width, height, ticksPerBar, barAlpha: 0.4, beatAlpha: 0.15, beatTop: height - 6 });

    for (let bar = startBar; bar <= endBar; bar++) {
      const tick = bar * ticksPerBar;
      const x = tick * ppt - scrollX;
      ctx.fillStyle = "#e8ecf4";
      ctx.font = "bold 10px monospace";
      ctx.fillText(String(bar + 1), x + 3, 10);
      ctx.fillStyle = "#556b94";
      ctx.font = "9px monospace";
      ctx.fillText(formatTime(bar * secsPerBar), x + 3, 20);
    }

    // Playhead marker on ruler
    const phx = playheadTick * ppt - scrollX;
    if (phx >= 0 && phx <= width) {
      drawPlayhead(ctx, { x: phx, height, cap: "bottom", capHalfWidth: 5, capDepth: 6 });
    }

    ctx.strokeStyle = CANVAS_BORDER;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 0.5);
    ctx.lineTo(width, height - 0.5);
    ctx.stroke();
  }, [tempo, timeSignature]);

  drawRef.current = draw;

  // Redraw when content (tempo/timeSignature) changes. scroll/zoom/playhead redraws come from
  // the store subscriptions below.
  useEffect(() => { draw(); }, [draw]);

  // Subscribe to scroll/zoom (app) + playhead (project); update refs and repaint imperatively.
  useEffect(() => {
    const unsubApp = useAppStore.subscribe((s) => {
      let changed = false;
      if (s.scrollX !== scrollXRef.current) { scrollXRef.current = s.scrollX; changed = true; }
      if (s.zoom !== zoomRef.current) { zoomRef.current = s.zoom; pptRef.current = PIXELS_PER_TICK * s.zoom; changed = true; }
      if (changed) requestRedraw();
    });
    const unsubProj = useProjectStore.subscribe((s) => {
      if (s.playheadTick !== playheadRef.current) { playheadRef.current = s.playheadTick; requestRedraw(); }
    });
    return () => { unsubApp(); unsubProj(); cancelAnimationFrame(redrawRafRef.current); };
  }, [requestRedraw]);

  // Observe canvas size once — don't tear down / rebuild the observer on every draw change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => drawRef.current());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="timeline-ruler"
      style={{ cursor: "pointer" }}
      onMouseDown={handleMouseDown}
    />
  );
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}
