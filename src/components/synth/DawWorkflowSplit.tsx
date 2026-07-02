import { useRef, useEffect, useState, useCallback } from "react";
import { DawView } from "./DawView";
import { WorkflowEditor } from "../workflow/WorkflowEditor";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { saveSetting } from "../../lib/settings";
import "./DawWorkflowSplit.css";

// SynthV/OpenUTAU-style vertical split: the track arrangement (DawView) on top, the per-segment node
// workflow editor docked as a RESIZABLE bottom panel (not a full-screen overlay). The track area
// REALLY shrinks when the panel opens/grows, so DawView's ResizeObserver chain self-corrects scroll
// and hit-test geometry — no special wiring needed here beyond owning the divider + panel height.

const MIN_PANEL = 160; // the editor never smaller than this (ReactFlow needs room to be usable)
const MIN_TRACKS = 120; // the track area never smaller than this (so it can't be dragged away)
const DIVIDER_H = 6;

export function DawWorkflowSplit() {
  const workflowSegmentId = useAppStore((s) => s.workflowSegmentId);
  const closeWorkflow = useAppStore((s) => s.closeWorkflow);
  const activePane = useAppStore((s) => s.activePane); // drives the active-pane cue on the divider
  const panelHeight = useAppStore((s) => s.workflowPanelHeight); // the DESIRED (persisted) height
  const setPanelHeight = useAppStore((s) => s.setWorkflowPanelHeight);
  const tracks = useProjectStore((s) => s.tracks);
  const splitRef = useRef<HTMLDivElement>(null);
  const [splitH, setSplitH] = useState(0); // measured container height; 0 = not yet measured
  const dragRef = useRef<{ startY: number; startH: number; splitH: number } | null>(null);

  // Close the docked editor if its segment vanishes out from under it (the user deletes that segment, or
  // its whole track, or loads/news a project). The editor + tracks coexist now, so this is no longer
  // handled implicitly by a full-screen overlay being dismissed before any track edit.
  useEffect(() => {
    if (workflowSegmentId && !tracks.some((t) => t.segments.some((s) => s.id === workflowSegmentId))) {
      closeWorkflow();
    }
  }, [tracks, workflowSegmentId, closeWorkflow]);

  // Measure the split container so we can CLAMP the RENDERED panel height to what fits, WITHOUT ever
  // mutating the user's stored preference. (An earlier version let the observer write the height down
  // and never restored it — a one-way ratchet on window-shrink, plus a startup-0 clobber while the
  // window was still hidden. Keeping the preference intact + clamping only at render fixes both.)
  useEffect(() => {
    const el = splitRef.current;
    if (!el) return;
    const measure = () => setSplitH(el.clientHeight);
    measure();
    const ob = new ResizeObserver(measure);
    ob.observe(el);
    return () => ob.disconnect();
  }, []);

  // Effective height = the preference, clamped to what currently fits. While unmeasured (splitH 0 =
  // window still hidden at startup) render the raw preference so a transient 0 never shrinks anything.
  const maxPanel = Math.max(MIN_PANEL, splitH - MIN_TRACKS - DIVIDER_H);
  const effectiveHeight = splitH > 0 ? Math.min(panelHeight, maxPanel) : panelHeight;

  const endDrag = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return; // already ended (pointerup → lostpointercapture fires too)
    dragRef.current = null;
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
    saveSetting("utai.workflowPanelHeight", useAppStore.getState().workflowPanelHeight); // persist on release only
  }, []);

  const onDividerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    dragRef.current = {
      startY: e.clientY,
      startH: useAppStore.getState().workflowPanelHeight,
      splitH: splitRef.current?.clientHeight ?? 0,
    };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  }, []);

  const onDividerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      if (e.buttons === 0) { endDrag(e); return; } // self-heal if a pointerup/cancel was ever missed
      const dy = e.clientY - d.startY; // dragging UP (dy<0) grows the bottom panel
      const max = Math.max(MIN_PANEL, d.splitH - MIN_TRACKS - DIVIDER_H);
      setPanelHeight(Math.max(MIN_PANEL, Math.min(max, d.startH - dy)));
    },
    [setPanelHeight, endDrag],
  );

  return (
    <div className={`daw-split pane-${activePane}${workflowSegmentId ? " panel-open" : ""}`} ref={splitRef}>
      <DawView />
      {workflowSegmentId && (
        <>
          <div
            className="workflow-divider"
            onPointerDown={onDividerDown}
            onPointerMove={onDividerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onLostPointerCapture={endDrag}
          />
          {/* key={segmentId}: switching to another segment REMOUNTS the editor, so the node graph, the
              modal-local undo stack, and the debounced-save scope all reset to the new segment. Without
              this, an open→open segment switch reused one instance that (a) kept the previous segment's
              graph + undo history and (b) wrote the OLD graph into the NEW segment via the [nodes,edges,
              segmentId] save effect — silent cross-segment corruption + "undo stack is confused". */}
          <WorkflowEditor key={workflowSegmentId} segmentId={workflowSegmentId} onClose={closeWorkflow} style={{ height: effectiveHeight }} />
        </>
      )}
    </div>
  );
}
