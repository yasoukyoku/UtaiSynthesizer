import { useRef, useCallback, useEffect } from "react";
import { Toolbar } from "./Toolbar";
import { TrackList } from "./TrackList";
import { TimelineRuler } from "./TimelineRuler";
import { Arrangement } from "./Arrangement";
import { HScrollbar } from "./HScrollbar";
import { useAppStore } from "../../store/app";
import { useProjectStore, useTimeAxis } from "../../store/project";
import { PIXELS_PER_TICK, TRACK_ADD_FOOTER } from "../../lib/constants";
import { computeTotalTracksHeight, computeTotalTicks } from "../../lib/trackLayout";
import "./DawView.css";

const HEADER_WIDTH = 200;

export function DawView() {
  // scrollX/scrollY are intentionally NOT subscribed here — horizontal/vertical scroll must not
  // re-render the whole DAW subtree. The canvases (Arrangement/TimelineRuler) self-subscribe and
  // repaint imperatively; HScrollbar/TrackList self-subscribe their own scroll value. vZoom is
  // likewise NOT subscribed (zoom gestures would re-render the subtree); it's read via getState.
  const zoom = useAppStore((s) => s.zoom);
  const canvasWidth = useAppStore((s) => s.canvasWidth);
  const canvasHeight = useAppStore((s) => s.canvasHeight);
  const setCanvasWidth = useAppStore((s) => s.setCanvasWidth);
  const setCanvasHeight = useAppStore((s) => s.setCanvasHeight);
  const tracks = useProjectStore((s) => s.tracks);
  const timeAxis = useTimeAxis();
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  // Vertical-zoom wheel events are coalesced into one setVZoom per frame (high-res wheels fire many
  // per frame; applying each = many re-renders/redraws = sluggish).
  const vZoomFactorRef = useRef(1);
  const vZoomRafRef = useRef(0);
  useEffect(() => () => cancelAnimationFrame(vZoomRafRef.current), []);

  useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setCanvasWidth(entry.contentRect.width);
        setCanvasHeight(entry.contentRect.height);
      }
    });
    observer.observe(el);
    setCanvasWidth(el.clientWidth);
    setCanvasHeight(el.clientHeight);
    return () => observer.disconnect();
  }, [setCanvasWidth, setCanvasHeight]);

  // Keep scrollY within content bounds: when the track stack is shorter than the viewport (few
  // tracks, or vertical-zoom shrank it), don't allow scrolling past it. vZoom read via getState
  // (its own change sites clamp); this covers track add/remove + viewport resize.
  useEffect(() => {
    const st = useAppStore.getState();
    const maxY = Math.max(0, computeTotalTracksHeight(tracks, st.vZoom) + TRACK_ADD_FOOTER - canvasHeight);
    if (st.scrollY > maxY) st.setScroll(st.scrollX, maxY);
  }, [tracks, canvasHeight]);

  const totalWidth = computeTotalTicks(tracks, timeAxis) * PIXELS_PER_TICK * zoom;

  // Wheel over the track-header column: Alt or Ctrl → vertical (track-height) zoom; plain →
  // vertical scroll, clamped so a partially-filled column can't scroll its content out of view.
  const handleTrackListWheel = useCallback((e: React.WheelEvent) => {
    e.stopPropagation();
    const st = useAppStore.getState();
    if (e.altKey || e.ctrlKey) {
      e.preventDefault(); // stop the browser's Ctrl+wheel page-zoom / Alt+wheel default
      vZoomFactorRef.current *= e.deltaY > 0 ? 0.9 : 1.1;
      if (!vZoomRafRef.current) {
        vZoomRafRef.current = requestAnimationFrame(() => {
          vZoomRafRef.current = 0;
          const s = useAppStore.getState();
          s.setVZoom(s.vZoom * vZoomFactorRef.current);
          vZoomFactorRef.current = 1;
          const maxY = Math.max(0, computeTotalTracksHeight(useProjectStore.getState().tracks, useAppStore.getState().vZoom) + TRACK_ADD_FOOTER - s.canvasHeight);
          s.setScroll(s.scrollX, Math.min(s.scrollY, maxY));
        });
      }
      return;
    }
    const maxY = Math.max(0, computeTotalTracksHeight(useProjectStore.getState().tracks, st.vZoom) + TRACK_ADD_FOOTER - st.canvasHeight);
    st.setScroll(st.scrollX, Math.max(0, Math.min(maxY, st.scrollY + e.deltaY)));
  }, []);

  return (
    <div
      className="daw-view"
      onPointerDownCapture={() => useAppStore.getState().setActivePane("timeline")}
    >
      <Toolbar />
      <div className="daw-grid">
        <div className="daw-corner" style={{ width: HEADER_WIDTH }} />
        <TimelineRuler />
        <div className="daw-tracklist-wrap" onWheel={handleTrackListWheel}>
          <TrackList width={HEADER_WIDTH} />
        </div>
        <div className="daw-canvas-container" ref={canvasContainerRef}>
          <Arrangement />
        </div>
        <div className="daw-scrollbar-corner" style={{ width: HEADER_WIDTH }} />
        <HScrollbar
          totalWidth={totalWidth}
          viewWidth={canvasWidth}
          onChange={(x) => { const st = useAppStore.getState(); st.setScroll(x, st.scrollY); }}
        />
      </div>
    </div>
  );
}
