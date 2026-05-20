import { useRef, useCallback } from "react";
import { Toolbar } from "./Toolbar";
import { TrackList } from "./TrackList";
import { TimelineRuler } from "./TimelineRuler";
import { Arrangement } from "./Arrangement";
import { HScrollbar } from "./HScrollbar";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { TICKS_PER_BEAT, PIXELS_PER_TICK } from "../../lib/constants";
import "./DawView.css";

const HEADER_WIDTH = 200;

export function DawView() {
  const { scrollX, scrollY, setScroll, zoom } = useAppStore();
  const { tracks, timeSignature } = useProjectStore();
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  const totalTicks = Math.max(
    TICKS_PER_BEAT * timeSignature[0] * 32,
    ...tracks.flatMap((t) => t.segments.map((s) => s.startTick + s.durationTicks)),
  );
  const totalWidth = totalTicks * PIXELS_PER_TICK * zoom;

  // Track header area: scroll always vertical
  const handleTrackListWheel = useCallback(
    (e: React.WheelEvent) => {
      e.stopPropagation();
      setScroll(scrollX, Math.max(0, scrollY + e.deltaY));
    },
    [scrollX, scrollY, setScroll],
  );

  return (
    <div className="daw-view">
      <Toolbar />
      <div className="daw-grid">
        <div className="daw-corner" style={{ width: HEADER_WIDTH }} />
        <TimelineRuler scrollX={scrollX} zoom={zoom} />
        <div className="daw-tracklist-wrap" onWheel={handleTrackListWheel}>
          <TrackList width={HEADER_WIDTH} scrollY={scrollY} />
        </div>
        <div className="daw-canvas-container" ref={canvasContainerRef}>
          <Arrangement />
        </div>
        <div className="daw-scrollbar-corner" style={{ width: HEADER_WIDTH }} />
        <HScrollbar
          scrollX={scrollX}
          totalWidth={totalWidth}
          viewWidth={800}
          onChange={(x) => setScroll(x, scrollY)}
        />
      </div>
    </div>
  );
}
