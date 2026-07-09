import { useRef, useEffect, useState, useCallback } from "react";
import { DawView } from "./DawView";
import { WorkflowEditor } from "../workflow/WorkflowEditor";
import { VocalEditor } from "./VocalEditor";
import { useAppStore } from "../../store/app";
import { useProjectStore } from "../../store/project";
import { saveSetting } from "../../lib/settings";
import "./DawWorkflowSplit.css";

// SynthV/OpenUTAU-style vertical split: the track arrangement (DawView) on top, and ONE per-segment editor
// docked as a RESIZABLE bottom panel — the node workflow editor (audioClip segment) OR the ② vocal
// piano-roll editor (notes segment). They are mutually exclusive (the app store opens one and closes the
// other), so a single divider + clamp serves whichever is open; each keeps its OWN persisted height. The
// track area REALLY shrinks when the panel opens/grows, so DawView's ResizeObserver chain self-corrects.

const MIN_PANEL = 160; // the editor never smaller than this
const MIN_TRACKS = 120; // the track area never smaller than this
const DIVIDER_H = 6;

export function DawWorkflowSplit() {
  const workflowSegmentId = useAppStore((s) => s.workflowSegmentId);
  const vocalSegmentId = useAppStore((s) => s.vocalSegmentId);
  const closeWorkflow = useAppStore((s) => s.closeWorkflow);
  const closeVocalEditor = useAppStore((s) => s.closeVocalEditor);
  const activePane = useAppStore((s) => s.activePane); // drives the active-pane cue on the divider
  const workflowPanelHeight = useAppStore((s) => s.workflowPanelHeight);
  const vocalPanelHeight = useAppStore((s) => s.vocalPanelHeight);
  const setWorkflowPanelHeight = useAppStore((s) => s.setWorkflowPanelHeight);
  const setVocalPanelHeight = useAppStore((s) => s.setVocalPanelHeight);
  const tracks = useProjectStore((s) => s.tracks);
  const splitRef = useRef<HTMLDivElement>(null);
  const [splitH, setSplitH] = useState(0); // measured container height; 0 = not yet measured
  const dragRef = useRef<{ startY: number; startH: number; splitH: number } | null>(null);

  const isVocal = vocalSegmentId != null;
  const panelOpen = workflowSegmentId != null || vocalSegmentId != null;
  const panelHeight = isVocal ? vocalPanelHeight : workflowPanelHeight;
  const setPanelHeight = isVocal ? setVocalPanelHeight : setWorkflowPanelHeight;

  // Close a docked editor if its segment vanishes out from under it (deleting it / its track / a load).
  useEffect(() => {
    if (workflowSegmentId && !tracks.some((t) => t.segments.some((s) => s.id === workflowSegmentId))) closeWorkflow();
    if (vocalSegmentId && !tracks.some((t) => t.segments.some((s) => s.id === vocalSegmentId))) closeVocalEditor();
  }, [tracks, workflowSegmentId, vocalSegmentId, closeWorkflow, closeVocalEditor]);

  // Measure the split container so we can CLAMP the rendered panel height to what fits, WITHOUT mutating
  // the stored preference (see the earlier one-way-ratchet fix).
  useEffect(() => {
    const el = splitRef.current;
    if (!el) return;
    const measure = () => setSplitH(el.clientHeight);
    measure();
    const ob = new ResizeObserver(measure);
    ob.observe(el);
    return () => ob.disconnect();
  }, []);

  const maxPanel = Math.max(MIN_PANEL, splitH - MIN_TRACKS - DIVIDER_H);
  const effectiveHeight = splitH > 0 ? Math.min(panelHeight, maxPanel) : panelHeight;

  const endDrag = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
    // Persist the height of WHICHEVER editor is open (read live so a mid-drag pane swap can't cross wires).
    const st = useAppStore.getState();
    if (st.vocalSegmentId != null) saveSetting("utai.vocalPanelHeight", st.vocalPanelHeight);
    else saveSetting("utai.workflowPanelHeight", st.workflowPanelHeight);
  }, []);

  const onDividerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      dragRef.current = { startY: e.clientY, startH: panelHeight, splitH: splitRef.current?.clientHeight ?? 0 };
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    },
    [panelHeight],
  );

  const onDividerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      if (e.buttons === 0) { endDrag(e); return; }
      const dy = e.clientY - d.startY; // dragging UP (dy<0) grows the bottom panel
      const max = Math.max(MIN_PANEL, d.splitH - MIN_TRACKS - DIVIDER_H);
      setPanelHeight(Math.max(MIN_PANEL, Math.min(max, d.startH - dy)));
    },
    [setPanelHeight, endDrag],
  );

  return (
    <div className={`daw-split pane-${activePane}${panelOpen ? " panel-open" : ""}`} ref={splitRef}>
      <DawView />
      {panelOpen && (
        <>
          <div
            className="workflow-divider"
            onPointerDown={onDividerDown}
            onPointerMove={onDividerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onLostPointerCapture={endDrag}
          />
          {/* key={segmentId}: switching segments REMOUNTS the editor so its view/undo/save scope reset. */}
          {isVocal ? (
            <VocalEditor key={vocalSegmentId} segmentId={vocalSegmentId!} onClose={closeVocalEditor} style={{ height: effectiveHeight }} />
          ) : (
            <WorkflowEditor key={workflowSegmentId} segmentId={workflowSegmentId!} onClose={closeWorkflow} style={{ height: effectiveHeight }} />
          )}
        </>
      )}
    </div>
  );
}
