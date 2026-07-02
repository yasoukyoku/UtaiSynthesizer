import type { Track, Segment, ProcessedOutput, Workflow, LaneControl } from "../types/project";
import { TRACK_HEADER_HEIGHT, LANE_HEIGHT, LANE_GROUP_BAR_HEIGHT, TICKS_PER_BEAT } from "./constants";
import { laneRowKey, laneGroupName, laneGroupId, laneOpsSig, materializeClips } from "./audio/laneOps";
import { peaksSignature } from "./waveformCache";

/** Order a segment's deposited lanes by the position of their producing Output node in the workflow, so
 *  lane ROWS stay in a DETERMINISTIC workflow order instead of the async deposit-COMPLETION order (which
 *  let a later-finishing group — e.g. "Main2" — jump above an earlier one's stems). STABLE: the original
 *  index is the tiebreaker, so edge order WITHIN a group (and legacy lanes with no outputNodeId, parked at
 *  the end) is preserved. Keyed by outputNodeId only — never touches laneId/laneOps identity. */
export function orderProcessedOutputs(outputs: ProcessedOutput[], workflow?: Workflow): ProcessedOutput[] {
  if (!workflow || outputs.length < 2) return outputs;
  const order = new Map<string, number>();
  let i = 0;
  for (const n of workflow.nodes) if (n.nodeType === "output") order.set(n.id, i++);
  const rank = (o: ProcessedOutput) =>
    o.outputNodeId !== undefined ? order.get(o.outputNodeId) ?? Number.MAX_SAFE_INTEGER : Number.MAX_SAFE_INTEGER;
  return outputs
    .map((o, idx) => [o, idx] as const)
    .sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1])
    .map(([o]) => o);
}

/** Total tick span of the arrangement: at least 32 bars of scroll headroom, or the last segment end.
 *  Shared by DawView (scroll width / HScrollbar) and OverviewMap so the minimap's viewport box and
 *  drag map 1:1 to the real scrollable range. */
export function computeTotalTicks(tracks: Track[], beatsPerBar: number): number {
  return Math.max(
    TICKS_PER_BEAT * beatsPerBar * 32,
    ...tracks.flatMap((t) => t.segments.map((s) => s.startTick + s.durationTicks)),
  );
}

/** Tick where the last PLAYABLE (audio-clip, non-loading) segment ends — the real audible content
 *  end (no 32-bar scroll floor). 0 when there's none. Used to detect "playhead at the end" so Play
 *  restarts from 0; counts only audio clips since playback schedules only those (a trailing notes
 *  segment must not push the natural-end playhead snap past where the audio actually stopped). */
export function contentEndTick(tracks: Track[]): number {
  let end = 0;
  for (const t of tracks) {
    for (const s of t.segments) {
      if (s.loading || s.content.type !== "audioClip") continue;
      const e = s.startTick + s.durationTicks;
      if (e > end) end = e;
    }
  }
  return end;
}

export interface LaneInfo {
  /** ROW identity = `laneRowKey` (轨道组 name + laneId). Keys the header rows, canvas row positioning,
   *  and crossfade pairing. Including the NAME lets two split halves that share a laneId (a split copies
   *  the graph, so node ids match) separate onto their own rows when one half's Output node is renamed —
   *  instead of the first-seen label swallowing the sibling. NOT the mixer key (that's `laneId`). */
  id: string;
  /** Display name; same-label rows are de-collided with a " 2"/" 3" suffix in first-seen order. */
  label: string;
  /** The row's 轨道组 name (laneRowKey's name part) — for group visuals / bracket rendering. */
  group: string;
  /** The row's ProcessedOutput.laneId — used for the LEGACY mixer fallback + mute's legacy fallback. */
  laneId: string;
  /** The row's 组 id (`laneGroupId` = producing Output node) — the laneControls (volume/pan) key.
   *  All rows of one 组 share the mix by design (解组 to control independently). */
  groupId: string;
}

/** Per-track memo for getLanes: tracks are replaced immutably on every mutation, so the track object
 *  identity IS the cache key — this keeps getLanes effectively free in the per-frame draw/height/hit
 *  paths (it is called per track per frame, and per segment in the loading-overlay pass). */
const lanesMemo = new WeakMap<Track, LaneInfo[]>();

/** A row's visual-GROUP membership key: 组 (groupId) + 轨道组 name. One group bar per contiguous run of
 *  this key. Two runs CAN share a groupId (diverged split halves — same node id, renamed group): they
 *  get separate bars (separate names) whose mixer controls mirror ONE laneControls entry, by design. */
function laneRunKey(l: Pick<LaneInfo, "groupId" | "group">): string {
  return `${l.groupId}\u0000${l.group}`;
}

/** The track's distinct lanes in first-seen order, keyed by laneRowKey (NOT laneLabel — two Output
 *  nodes can share a label yet be different lanes). Drives row layout, the header column, and
 *  hit-testing. Two rows can still end up with the SAME display label (two same-group Output nodes
 *  fed by the same stem name, across segments/workflows) — those get numbered, display-only. */
export function getLanes(track: Track): LaneInfo[] {
  const memo = lanesMemo.get(track);
  if (memo) return memo;
  const seen = new Map<string, { label: string; group: string; laneId: string; groupId: string }>();
  for (const seg of track.segments) {
    for (const out of seg.processedOutputs ?? []) {
      const key = laneRowKey(out);
      if (!seen.has(key)) {
        seen.set(key, { label: out.laneLabel, group: laneGroupName(out), laneId: out.laneId, groupId: laneGroupId(out) });
      }
    }
  }
  const lanes = Array.from(seen, ([id, { label, group, laneId, groupId }]) => ({ id, label, group, laneId, groupId }));
  // Cluster rows so every 组+名 run is CONTIGUOUS (one group bar / one bracket per run — P5). Rows
  // arrive in per-segment deposit order, which can interleave groups across heterogeneous segments;
  // a STABLE sort by first-seen run key keeps within-run row order. The header column and the canvas
  // both consume THIS order, so their row geometry stays aligned by construction.
  const runOrder = new Map<string, number>();
  for (const l of lanes) {
    const k = laneRunKey(l);
    if (!runOrder.has(k)) runOrder.set(k, runOrder.size);
  }
  lanes.sort((a, b) => runOrder.get(laneRunKey(a))! - runOrder.get(laneRunKey(b))!);
  // Display de-collision: number duplicate labels ("instr", "instr 2", …) in first-seen order.
  // Purely cosmetic — identity is `id`; numbers may shift when rows come/go (like file managers).
  // Generated names skip values that ALREADY exist as literal labels (a row the user named "… 2").
  const labelCounts = new Map<string, number>();
  for (const l of lanes) labelCounts.set(l.label, (labelCounts.get(l.label) ?? 0) + 1);
  if (lanes.length > labelCounts.size) {
    const taken = new Set(lanes.map((l) => l.label));
    const numbered = new Map<string, number>();
    for (const l of lanes) {
      if ((labelCounts.get(l.label) ?? 0) < 2) continue;
      let n = (numbered.get(l.label) ?? 0) + 1;
      if (n > 1) {
        while (taken.has(`${l.label} ${n}`)) n++;
        taken.add(`${l.label} ${n}`);
      }
      numbered.set(l.label, n);
      if (n > 1) l.label = `${l.label} ${n}`;
    }
  }
  lanesMemo.set(track, lanes);
  return lanes;
}

/** THE "is this lane row muted" predicate — the single audibility source of truth for the header
 *  button, the canvas dim, playback, AND (future) mixdown export / overall-waveform display. Mute is
 *  per-ROW (`laneMutes[rowKey]`, loose: resets on rename/ungroup) with a legacy fallback to the old
 *  per-laneId `laneControls.muted` (pre-S28 saves), so old projects keep their mutes until toggled. */
export function isLaneRowMuted(track: Track, rowKey: string, laneId: string): boolean {
  return track.laneMutes?.[rowKey] ?? track.laneControls[laneId]?.muted ?? false;
}

/** THE "does this segment play its sub-lanes (instead of the original audio)" predicate — the single
 *  SOURCE-selection truth for playback scheduling, the main-row waveform (sum of lanes vs original),
 *  the lanes-dimmed visual, AND (future) mixdown export. True when the segment has at least one ready
 *  (non-loading) deposited lane and the track's `playOriginal` bypass is off. Deliberately NOT gated
 *  on `track.expanded` — collapse/expand is pure view state and must never change what you hear. */
export function segmentPlaysLanes(track: Track, seg: Segment): boolean {
  return !track.playOriginal && !!seg.processedOutputs?.some((o) => !o.loading);
}

/** The segment's ready (non-loading) lanes that have waveform peaks — the sum's contributors. */
function readyPeakOuts(seg: Segment): ProcessedOutput[] {
  return (seg.processedOutputs ?? []).filter((o) => !o.loading && !!o.waveformPeaks && o.waveformPeaks.length > 0);
}

/** Change-signature of a segment's lane-sum INPUTS (each ready lane's identity + peaks content + row
 *  mute, plus the laneOps recipe) — "" when no ready lane has peaks. ONE definition shared by the
 *  sum memo and the OverviewMap's waveform-cache key, so both invalidate on exactly the same inputs. */
export function laneSumSig(track: Track, seg: Segment): string {
  const outs = readyPeakOuts(seg);
  if (outs.length === 0) return "";
  return (
    outs
      .map((o) => `${o.laneId}:${peaksSignature(o.waveformPeaks!)}:${isLaneRowMuted(track, laneRowKey(o), o.laneId) ? 1 : 0}`)
      .join("|") + `~${laneOpsSig(seg.laneOps)}`
  );
}

/** Memo for segmentLaneSumPeaks — keyed by SEGMENT object identity plus an input signature (row mutes
 *  live on the TRACK, which changes without replacing the segment object; the sig catches that). */
const laneSumMemo = new WeakMap<Segment, { sig: string; peaks: number[] }>();

/** The REAL audible mix envelope of a segment's sub-lanes, in STEM space (the stem spans the whole
 *  source audio, so callers draw this with the SAME window ratios as the original waveform): a
 *  per-bucket sum of every ready lane's peaks, skipping muted rows (isLaneRowMuted — THE predicate)
 *  and zeroing regions outside each 组's kept laneOps intervals (slice/trim = silence), clamped to 1.
 *  All rows muted ⇒ an all-zero envelope (a flat line — correct: playback is silent), NOT null.
 *  Returns null only when NO ready lane has peaks — the caller falls back to the original waveform,
 *  matching segmentPlaysLanes' fall-through. UNITY GAIN by design: like the lane rows' own waveforms,
 *  the sum does not scale with the group faders (scaling would re-bake the static layer per 0.5 dB
 *  drag step, and the per-row displays would disagree with the main row). */
export function segmentLaneSumPeaks(track: Track, seg: Segment): number[] | null {
  if (seg.content.type !== "audioClip" || seg.content.totalDurationMs <= 0) return null;
  const stemDurMs = seg.content.totalDurationMs;
  const outs = readyPeakOuts(seg);
  if (outs.length === 0) return null;
  const sig = laneSumSig(track, seg);
  const memo = laneSumMemo.get(seg);
  if (memo && memo.sig === sig) return memo.peaks;
  const n = Math.max(...outs.map((o) => o.waveformPeaks!.length));
  const sum = new Array<number>(n).fill(0);
  for (const o of outs) {
    if (isLaneRowMuted(track, laneRowKey(o), o.laneId)) continue;
    const peaks = o.waveformPeaks!;
    const pn = peaks.length;
    // Clips are ordered + non-overlapping; `laneEnd` dedups the SHARED boundary bucket of two
    // adjacent clips (floor/ceil of a fractional cut overlap by one bucket — without this every
    // slice point double-added its bucket and drew a 2× spike, review-caught).
    let laneEnd = 0;
    for (const clip of materializeClips(seg.laneOps?.[laneGroupId(o)], stemDurMs)) {
      const b0 = Math.max(laneEnd, Math.floor((clip.start / stemDurMs) * n));
      const b1 = Math.min(n, Math.ceil((clip.end / stemDurMs) * n));
      for (let i = b0; i < b1; i++) {
        sum[i]! += peaks[Math.min(pn - 1, Math.floor(((i + 0.5) / n) * pn))]!;
      }
      laneEnd = Math.max(laneEnd, b1);
    }
  }
  for (let i = 0; i < n; i++) if (sum[i]! > 1) sum[i] = 1;
  laneSumMemo.set(seg, { sig, peaks: sum });
  return sum;
}

/** THE lane volume/pan accessor — group-scoped: keyed by the producing Output node (`groupId` =
 *  laneGroupId), so the mix is "recorded on the Output node" like laneOps and survives renames AND
 *  upstream rewiring (reconnecting the same Output inherits it back). Legacy fallback: pre-S28 saves
 *  keyed these by laneId. Writes always go to the group key (updateLaneControl). */
export function laneControlFor(track: Track, groupId: string, laneId: string): LaneControl | undefined {
  return track.laneControls[groupId] ?? track.laneControls[laneId];
}

/** One visual sub-lane GROUP (a contiguous run of same-组-same-名 rows) + its slim GROUP BAR above the
 *  rows. The bar hosts the group name + the group-level volume/pan controls (drawn ONCE per 组 — P5). */
export interface LaneGroupRun {
  /** Run identity (`laneRunKey`) — keys the header column's group blocks. */
  key: string;
  /** The 组 id (`laneGroupId` = producing Output node) — the laneControls (volume/pan) key. */
  groupId: string;
  /** The run's 轨道组 display name (shown once, on the bar). */
  name: string;
  /** First row's laneId — laneControlFor's legacy pre-S28 fallback key. */
  laneId: string;
  /** Row index range into getLanes(track): rows [start, start + count). */
  start: number;
  count: number;
  /** Unscaled Y of the group BAR, relative to the TRACK TOP (multiply by vZoom). */
  barY: number;
  /** LANE_COLORS index — keyed by the 轨道组 NAME (first-seen among the track's distinct names), NOT
   *  the run: 组 boundaries are already delineated by the bar, so color marks the 轨道组 (same name =
   *  same hue, even across runs/detach; future multi-singer setups get one color per singer). */
  colorIndex: number;
}

/** The expanded track's sub-lane geometry — THE single row-position source for the header column AND
 *  the canvas (draw / hit-test / loading overlay / crossfades). Never compute a row Y as
 *  `headerH + index * LANE_HEIGHT` — the group bars above each run shift the rows. */
export interface LaneLayout {
  /** Unscaled Y of each lane row, index-aligned with getLanes(track), relative to the TRACK TOP. */
  rowY: number[];
  /** Group-run index of each row (index-aligned with getLanes) — e.g. the lane-color key. */
  rowRun: number[];
  runs: LaneGroupRun[];
  /** Unscaled total height of the expanded lanes area (all bars + rows, header excluded). */
  lanesHeight: number;
}

const laneLayoutMemo = new WeakMap<Track, LaneLayout>();

export function getLaneLayout(track: Track): LaneLayout {
  const memo = laneLayoutMemo.get(track);
  if (memo) return memo;
  const lanes = getLanes(track);
  const rowY: number[] = [];
  const rowRun: number[] = [];
  const runs: LaneGroupRun[] = [];
  const nameColor = new Map<string, number>(); // 轨道组 name → first-seen color index
  let y = TRACK_HEADER_HEIGHT;
  for (let i = 0; i < lanes.length; i++) {
    const l = lanes[i]!;
    const key = laneRunKey(l);
    let run = runs[runs.length - 1];
    if (!run || run.key !== key) {
      if (!nameColor.has(l.group)) nameColor.set(l.group, nameColor.size);
      run = { key, groupId: l.groupId, name: l.group, laneId: l.laneId, start: i, count: 0, barY: y, colorIndex: nameColor.get(l.group)! };
      runs.push(run);
      y += LANE_GROUP_BAR_HEIGHT;
    }
    run.count++;
    rowRun.push(runs.length - 1);
    rowY.push(y);
    y += LANE_HEIGHT;
  }
  const layout: LaneLayout = { rowY, rowRun, runs, lanesHeight: y - TRACK_HEADER_HEIGHT };
  laneLayoutMemo.set(track, layout);
  return layout;
}

/** Lane-row index at an unscaled Y relative to the TRACK TOP (divide the scaled offset by vZoom
 *  first), or -1 when the Y falls on the header, a group bar, or below the last row. */
export function laneRowAtY(track: Track, yUnscaled: number): number {
  const { rowY } = getLaneLayout(track);
  for (let i = 0; i < rowY.length; i++) {
    if (yUnscaled >= rowY[i]! && yUnscaled < rowY[i]! + LANE_HEIGHT) return i;
  }
  return -1;
}

/** Track display height. `scale` is the vertical zoom (vZoom) applied to header + lanes.
 *  Expanded lanes = one row per distinct laneRowKey + one GROUP BAR per 组+名 run (getLaneLayout),
 *  matching the header column's rendering, so the column never overflows its box when a track's
 *  segments emit heterogeneous lane sets. */
export function computeTrackHeight(track: Track, scale = 1): number {
  const lanesH = track.expanded ? getLaneLayout(track).lanesHeight : 0;
  return (TRACK_HEADER_HEIGHT + lanesH) * scale;
}

export function computeTrackYOffsets(tracks: Track[], scale = 1): number[] {
  const offsets: number[] = [];
  let y = 0;
  for (const track of tracks) {
    offsets.push(y);
    y += computeTrackHeight(track, scale);
  }
  return offsets;
}

export function computeTotalTracksHeight(tracks: Track[], scale = 1): number {
  let h = 0;
  for (const track of tracks) {
    h += computeTrackHeight(track, scale);
  }
  return h;
}

export function findTrackAtY(offsets: number[], y: number): number {
  for (let i = offsets.length - 1; i >= 0; i--) {
    if (y >= offsets[i]!) return i;
  }
  return -1;
}
