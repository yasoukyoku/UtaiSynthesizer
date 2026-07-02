import type { Edge, Node } from "@xyflow/react";

export interface GraphSnap {
  nodes: Node[];
  edges: Edge[];
}

/** One segment's node-graph undo state. Fields are mutated in place by the editor (the map entry IS
 *  the live stack), so a remount resumes exactly where the previous instance left off. */
export interface NodeHistory {
  past: GraphSnap[];
  future: GraphSnap[];
  /** The CURRENT committed graph — the state an undo steps back FROM. */
  commit: GraphSnap;
  /** sigOfGraph of `commit` — the auto-capture change detector. */
  sig: string;
}

/**
 * Per-SEGMENT node-graph undo histories — they SURVIVE the docked editor's `key={segmentId}` remounts.
 * The stacks used to live in component refs, so switching the open segment destroyed the previous
 * segment's history (detach on piece 1 → detach on piece 2 → back to piece 1: its 解组 was no longer
 * undoable; and pre-P6 the resulting empty-stack Ctrl+Z even fell through to the TIMELINE stack — the
 * phantom-undo bug). Resuming is safe because a segment's graph is only ever mutated by its OWN open
 * editor (the unmount cleanup flushes the graph synchronously; headless deposit / RenderLinkWatcher
 * never touch graphs), so the graph at remount is exactly the graph at unmount.
 *
 * Entries for deleted segments simply become unreachable (tiny; reclaimed on load). Cleared wholesale
 * by teardownForLoad on project new/open/recover — you can't undo across a document swap.
 */
const histories = new Map<string, NodeHistory>();

/** The segment's resumable history, creating it from `init` on first open. */
export function nodeHistoryFor(segmentId: string, init: () => NodeHistory): NodeHistory {
  let h = histories.get(segmentId);
  if (!h) {
    h = init();
    histories.set(segmentId, h);
  }
  return h;
}

/** Drop every segment's node history (project new/open/recover — the document is being replaced). */
export function clearNodeHistories() {
  histories.clear();
}
