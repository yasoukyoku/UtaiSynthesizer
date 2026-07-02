import { useEffect } from "react";
import { useProjectStore } from "../../store/project";
import { useWorkflowStore } from "../../store/workflow";
import { useAudioStore } from "../../store/audio";
import { useAppStore } from "../../store/app";
import { depositFromCache } from "../../lib/workflow/engine";
import type { Workflow } from "../../types/project";

/** Segments currently being headless-deposited (async decode) — guards against re-firing the deposit every
 *  watcher tick during the decode window. */
const depositing = new Set<string>();

/** Fire-and-forget headless deposit of a segment's lanes from the render cache, using the segment's OWN
 *  persisted graph — no open editor needed. Guarded against overlapping decodes; bumps playback on change. */
function headlessDeposit(trackId: string, segmentId: string, workflow: Workflow | undefined): void {
  if (depositing.has(segmentId)) return;
  depositing.add(segmentId);
  void depositFromCache(trackId, segmentId, workflow ?? { nodes: [], connections: [] })
    .then((ch) => {
      if (ch) useAudioStore.getState().bumpSchedule();
    })
    // finally, not then: a rejection that skipped the delete would permanently poison this segment in
    // the in-flight set — every future headless deposit would no-op (the spin-forever bug, reborn).
    .finally(() => depositing.delete(segmentId));
}

/**
 * Split-mid-render inheritance (Option B). When a segment is split WHILE its render is in flight, the new
 * (right) half is `renderLink`-ed to the source (left) half and carries the source's loading placeholders.
 * The render is a SINGLE global job keyed to the source id — it keeps running and deposits onto the source
 * via the open editor's reconciler. This watcher mirrors that result onto the linked half:
 *
 *  - source SETTLED (completed/error) AND its lanes fully deposited (none loading) → copy the source's final
 *    lanes onto the new half (each half windows the SAME whole-source stem by its own offsetMs, so no
 *    re-render), clone the now-settled render cache/badges (so reconnecting an Output works), then unlink.
 *  - source segment GONE (deleted/merged) → drop the new half's pending placeholders so they don't spin
 *    forever, then unlink.
 *  - new half GONE → just unlink.
 *
 * ALSO the app's generic RENDER-SETTLE watcher (not just split links): deposit is otherwise
 * open-editor-only (the WorkflowEditor reconciler), so a render whose editor was closed mid-run would
 * complete into the session cache but never land on the track — its loading placeholders would spin
 * forever and the finished render would be silently lost on app close (save strips loading lanes, the
 * cache is runtime-only). The settle pass below gives any such segment the same headless deposit a
 * link half gets.
 *
 * Renders null; lives in App so it runs regardless of which (if any) workflow editor is open.
 */
export function RenderLinkWatcher() {
  const renderLinks = useWorkflowStore((s) => s.renderLinks);
  const executions = useWorkflowStore((s) => s.executions);
  const tracks = useProjectStore((s) => s.tracks);
  // The open editor's segment is EXCLUDED from the settle pass (its mounted reconciler owns deposit);
  // subscribing means closing the editor right after a settle re-runs the effect and we take over.
  const workflowSegmentId = useAppStore((s) => s.workflowSegmentId);

  useEffect(() => {
    const links = Object.entries(renderLinks);

    const wf = useWorkflowStore.getState();
    const proj = useProjectStore.getState();

    // Locate against the LIVE store, not the subscribed `tracks` prop: at the editor-close boundary the
    // editor's unmount cleanup flushes its final graph into the project store DURING the same commit
    // whose effect closure captured the pre-flush tracks — depositing with that stale seg.workflow could
    // resurrect a just-deleted Output node's lane as a persistent ghost. The prop stays in the dep array
    // purely as the re-fire trigger.
    const locate = (segId: string) => {
      for (const t of proj.tracks) {
        const seg = t.segments.find((s) => s.id === segId);
        if (seg) return { trackId: t.id, seg };
      }
      return null;
    };

    let changed = false;
    for (const [toId, fromId] of links) {
      const to = locate(toId);
      if (!to) {
        wf.unlinkRender(toId); // right half deleted
        continue;
      }
      const from = locate(fromId);
      if (!from) {
        // source gone before it finished — drop the right half's pending placeholders (keep any real lanes)
        proj.replaceProcessedOutputs(to.trackId, toId, (to.seg.processedOutputs ?? []).filter((o) => !o.loading));
        wf.unlinkRender(toId);
        changed = true;
        continue;
      }
      const exec = executions[fromId];
      const settled = exec !== undefined && exec.status !== "running";
      if (settled) {
        // Render DONE → deposit BOTH halves from the render cache, EACH using its OWN persisted graph
        // (headless — needs NO open editor). This REPLACES the old "mirror the source's lanes onto the half",
        // whose failures the user hit: (a) it required the SOURCE's editor to be open to become "ready" (else
        // the linked halves stayed stuck "loading" forever); (b) it blindly copied the source's lanes,
        // ignoring the half's OWN graph — so a lane the half keeps (but the source deleted) never appeared
        // until you opened that half, and a lane deleted only on the source wrongly vanished from the half.
        // The single global render cache is whole-source, so each half windows the SAME stems by its own
        // offsetMs at draw/play. cloneSegmentState warms the half's cache first so its deposit finds the stems.
        headlessDeposit(from.trackId, fromId, from.seg.workflow); // source (its editor may be closed)
        wf.cloneSegmentState(fromId, toId);                       // warm the link half's cache from the settled source
        headlessDeposit(to.trackId, toId, to.seg.workflow);       // link half (its OWN graph)
        wf.unlinkRender(toId);
        changed = true;
      }
    }

    // SETTLE pass for NON-linked segments (see the component doc): any settled execution whose segment
    // still carries a loading lane, is not the open editor's segment, and is not a pending link target
    // (handled above) gets a headless settle-deposit. depositFromCache's settle-only contract holds —
    // the execution has settled — so cached branches land and dead placeholders are pruned. Repeat
    // effect runs are cheap: once the lanes are settled the `.some(loading)` guard skips the segment,
    // and headlessDeposit's in-flight set dedupes the decode window. Playback bumps via headlessDeposit
    // itself (the deposit is async — bumping here would fire before any lane actually changed).
    for (const [segId, exec] of Object.entries(executions)) {
      if (exec.status === "running" || segId === workflowSegmentId || renderLinks[segId]) continue;
      const loc = locate(segId);
      if (!loc || !loc.seg.processedOutputs?.some((o) => o.loading)) continue;
      headlessDeposit(loc.trackId, segId, loc.seg.workflow);
    }

    if (changed) useAudioStore.getState().bumpSchedule(); // live playback follows the new lanes
  }, [renderLinks, executions, tracks, workflowSegmentId]);

  return null;
}
