import { TRACK_HEADER_HEIGHT, LANE_HEIGHT, LANE_GROUP_BAR_HEIGHT, LOUDNESS_LANE_HEIGHT } from "./constants";
import { laneRowKey, laneGroupName, laneGroupId, laneOpsSig, materializeClips } from "./audio/laneOps";
import { peaksSignature } from "./waveformCache";
/** Order a segment's deposited lanes by the position of their producing Output node in the workflow, so
 *  lane ROWS stay in a DETERMINISTIC workflow order instead of the async deposit-COMPLETION order (which
 *  let a later-finishing group — e.g. "Main2" — jump above an earlier one's stems). STABLE: the original
 *  index is the tiebreaker, so edge order WITHIN a group (and legacy lanes with no outputNodeId, parked at
 *  the end) is preserved. Keyed by outputNodeId only — never touches laneId/laneOps identity. */
export function orderProcessedOutputs(outputs, workflow) {
    if (!workflow || outputs.length < 2)
        return outputs;
    const order = new Map();
    let i = 0;
    for (const n of workflow.nodes)
        if (n.nodeType === "output")
            order.set(n.id, i++);
    const rank = (o) => o.outputNodeId !== undefined ? order.get(o.outputNodeId) ?? Number.MAX_SAFE_INTEGER : Number.MAX_SAFE_INTEGER;
    return outputs
        .map((o, idx) => [o, idx])
        .sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1])
        .map(([o]) => o);
}
/** Total tick span of the arrangement: at least 32 bars of scroll headroom, or the last segment end.
 *  Shared by DawView (scroll width / HScrollbar) and OverviewMap so the minimap's viewport box and
 *  drag map 1:1 to the real scrollable range. The 32-bar floor is meter-aware (TimeAxis.ticksForBars);
 *  for a 4/4 project it is 480*4*32 = 61440, unchanged. */
export function computeTotalTicks(tracks, axis) {
    return Math.max(axis.ticksForBars(32), ...tracks.flatMap((t) => t.segments.map((s) => s.startTick + s.durationTicks)));
}
/** Tick where the last PLAYABLE (non-loading) segment ends — the real audible content end (no 32-bar
 *  scroll floor). 0 when there's none. Used to detect "playhead at the end" so Play restarts from 0.
 *  Counts audio clips AND baked ② vocal renders (notes segments playing their deposited lane) — the
 *  segments playback actually schedules — so the natural-end snap lands where the audio really stopped. */
export function contentEndTick(tracks) {
    let end = 0;
    for (const t of tracks) {
        for (const s of t.segments) {
            if (s.loading || (s.content.type !== "audioClip" && !segmentPlaysLanes(t, s)))
                continue;
            // The segment BOX (durationTicks) is the content extent playback runs to — for a ② vocal segment
            // whose baked stem is SHORTER than the box (the notes don't fill it), playback continues through the
            // silent tail to the box end rather than stopping the instant the last note's audio finishes (the
            // premature-pause ghost). The transport stops when the PLAYHEAD reaches here (Toolbar rAF), NOT when
            // the audio sources end — see the natural-end guard.
            const e = s.startTick + s.durationTicks;
            if (e > end)
                end = e;
        }
    }
    return end;
}
/** Per-track memo for getLanes: tracks are replaced immutably on every mutation, so the track object
 *  identity IS the cache key — this keeps getLanes effectively free in the per-frame draw/height/hit
 *  paths (it is called per track per frame, and per segment in the loading-overlay pass). */
const lanesMemo = new WeakMap();
/** A row's visual-GROUP membership key: 组 (groupId) + 轨道组 name. One group bar per contiguous run of
 *  this key. Two runs CAN share a groupId (diverged split halves — same node id, renamed group): they
 *  get separate bars (separate names) whose mixer controls mirror ONE laneControls entry, by design. */
function laneRunKey(l) {
    return `${l.groupId}\u0000${l.group}`;
}
/** The track's distinct lanes in first-seen order, keyed by laneRowKey (NOT laneLabel — two Output
 *  nodes can share a label yet be different lanes). Drives row layout, the header column, and
 *  hit-testing. Two rows can still end up with the SAME display label (two same-group Output nodes
 *  fed by the same stem name, across segments/workflows) — those get numbered, display-only. */
export function getLanes(track) {
    const memo = lanesMemo.get(track);
    if (memo)
        return memo;
    const seen = new Map();
    for (const seg of track.segments) {
        for (const out of seg.processedOutputs ?? []) {
            const key = laneRowKey(out);
            let r = seen.get(key);
            if (!r) {
                const gid = laneGroupId(out);
                // 解组's independence marker lives on the Output NODE (params.detached — the graph is the truth
                // source; split copies it, graph-undo of the detach removes it). A detached 组 is otherwise
                // structurally identical to a re-created node, but must NEVER merge back onto the original
                // rows still living on sibling pieces — that would re-couple the volume/pan 解组 just split.
                const detached = seg.workflow?.nodes.some((n) => n.id === gid && n.params?.detached === true) ?? false;
                r = { rowKey: key, label: out.laneLabel, group: laneGroupName(out), laneId: out.laneId, groupId: gid, segIds: new Set(), detached };
                seen.set(key, r);
            }
            r.segIds.add(seg.id);
        }
    }
    const visual = [];
    const byBucket = new Map();
    for (const r of seen.values()) {
        // A DETACHED row always stands alone: it neither joins an existing visual row nor registers as a
        // merge candidate for later rows (its bar stays its own — see the RawRow.detached note).
        if (r.detached) {
            visual.push({ rows: [r], segIds: new Set(r.segIds) });
            continue;
        }
        const bucket = JSON.stringify([r.group, r.label]);
        const cands = byBucket.get(bucket) ?? [];
        let placed;
        for (const v of cands) {
            let ok = true;
            for (const id of r.segIds) {
                if (v.segIds.has(id)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                placed = v;
                break;
            }
        }
        if (!placed) {
            placed = { rows: [], segIds: new Set() };
            cands.push(placed);
            byBucket.set(bucket, cands);
            visual.push(placed);
        }
        placed.rows.push(r);
        for (const id of r.segIds)
            placed.segIds.add(id);
    }
    // 3) BAR clusters: 组-runs LINKED by sharing a visual row render under ONE bar (union-find connected
    //    components — e.g. a full 组 and a partial re-created 组 sharing just the "instr" row). 解组
    //    outputs can never link (same segments / distinct labels ⇒ no shared row), keeping their bars —
    //    and thus their volume/pan — independent.
    const parent = new Map();
    const find = (k) => {
        let root = k;
        while (parent.has(root) && parent.get(root) !== root)
            root = parent.get(root);
        parent.set(k, root);
        return root;
    };
    for (const v of visual) {
        const first = find(laneRunKey(v.rows[0]));
        for (const r of v.rows) {
            const rk = find(laneRunKey(r));
            if (rk !== first)
                parent.set(rk, first);
        }
    }
    // 4) Emit: bar clusters in first-seen order, each cluster's visual rows in first-seen order (keeps a
    //    bar's rows contiguous — getLaneLayout scans contiguous runKey). Primary member = first-seen.
    const clusterOrder = new Map();
    const rooted = visual.map((v, idx) => {
        const root = find(laneRunKey(v.rows[0]));
        if (!clusterOrder.has(root))
            clusterOrder.set(root, clusterOrder.size);
        return { v, idx, root };
    });
    rooted.sort((a, b) => clusterOrder.get(a.root) - clusterOrder.get(b.root) || a.idx - b.idx);
    const lanes = rooted.map(({ v, root }) => {
        const p = v.rows[0];
        return {
            id: p.rowKey,
            label: p.label,
            group: p.group,
            laneId: p.laneId,
            groupId: p.groupId,
            members: v.rows.map((r) => ({ rowKey: r.rowKey, laneId: r.laneId, groupId: r.groupId })),
            runKey: root,
        };
    });
    // Display de-collision: number duplicate labels ("instr", "instr 2", …) in first-seen order.
    // Purely cosmetic — identity is `id`; numbers may shift when rows come/go (like file managers).
    // Generated names skip values that ALREADY exist as literal labels (a row the user named "… 2").
    const labelCounts = new Map();
    for (const l of lanes)
        labelCounts.set(l.label, (labelCounts.get(l.label) ?? 0) + 1);
    if (lanes.length > labelCounts.size) {
        const taken = new Set(lanes.map((l) => l.label));
        const numbered = new Map();
        for (const l of lanes) {
            if ((labelCounts.get(l.label) ?? 0) < 2)
                continue;
            let n = (numbered.get(l.label) ?? 0) + 1;
            if (n > 1) {
                while (taken.has(`${l.label} ${n}`))
                    n++;
                taken.add(`${l.label} ${n}`);
            }
            numbered.set(l.label, n);
            if (n > 1)
                l.label = `${l.label} ${n}`;
        }
    }
    lanesMemo.set(track, lanes);
    return lanes;
}
/** THE "is this lane row muted" predicate — the single audibility source of truth for the header
 *  button, the canvas dim, playback, AND (future) mixdown export / overall-waveform display. Mute is
 *  per-ROW (`laneMutes[rowKey]`, loose: resets on rename/ungroup) with a legacy fallback to the old
 *  per-laneId `laneControls.muted` (pre-S28 saves), so old projects keep their mutes until toggled. */
export function isLaneRowMuted(track, rowKey, laneId) {
    return track.laneMutes?.[rowKey] ?? track.laneControls[laneId]?.muted ?? false;
}
/** THE "does this segment play its sub-lanes (instead of the original audio)" predicate — the single
 *  SOURCE-selection truth for playback scheduling, the main-row waveform (sum of lanes vs original),
 *  the lanes-dimmed visual, AND (future) mixdown export. True when the segment has at least one ready
 *  (non-loading) deposited lane and the track's `playOriginal` bypass is off. Deliberately NOT gated
 *  on `track.expanded` — collapse/expand is pure view state and must never change what you hear. */
export function segmentPlaysLanes(track, seg) {
    return !track.playOriginal && !!seg.processedOutputs?.some((o) => !o.loading);
}
/** The segment's ready (non-loading) lanes that have waveform peaks — the sum's contributors. */
function readyPeakOuts(seg) {
    return (seg.processedOutputs ?? []).filter((o) => !o.loading && !!o.waveformPeaks && o.waveformPeaks.length > 0);
}
/** Change-signature of a segment's lane-sum INPUTS (each ready lane's identity + peaks content + row
 *  mute, plus the laneOps recipe) — "" when no ready lane has peaks. ONE definition shared by the
 *  sum memo and the OverviewMap's waveform-cache key, so both invalidate on exactly the same inputs. */
export function laneSumSig(track, seg) {
    const outs = readyPeakOuts(seg);
    if (outs.length === 0)
        return "";
    return (outs
        .map((o) => `${o.laneId}:${peaksSignature(o.waveformPeaks)}:${isLaneRowMuted(track, laneRowKey(o), o.laneId) ? 1 : 0}`)
        .join("|") + `~${laneOpsSig(seg.laneOps)}`);
}
/** Memo for segmentLaneSumPeaks — keyed by SEGMENT object identity plus an input signature (row mutes
 *  live on the TRACK, which changes without replacing the segment object; the sig catches that). */
const laneSumMemo = new WeakMap();
/** The REAL audible mix envelope of a segment's sub-lanes, in STEM space (the stem spans the whole
 *  source audio, so callers draw this with the SAME window ratios as the original waveform): a
 *  per-bucket sum of every ready lane's peaks, skipping muted rows (isLaneRowMuted — THE predicate)
 *  and zeroing regions outside each 组's kept laneOps intervals (slice/trim = silence), clamped to 1.
 *  All rows muted ⇒ an all-zero envelope (a flat line — correct: playback is silent), NOT null.
 *  Returns null only when NO ready lane has peaks — the caller falls back to the original waveform,
 *  matching segmentPlaysLanes' fall-through. UNITY GAIN by design: like the lane rows' own waveforms,
 *  the sum does not scale with the group faders (scaling would re-bake the static layer per 0.5 dB
 *  drag step, and the per-row displays would disagree with the main row). */
export function segmentLaneSumPeaks(track, seg) {
    if (seg.content.type !== "audioClip" || seg.content.totalDurationMs <= 0)
        return null;
    const stemDurMs = seg.content.totalDurationMs;
    const outs = readyPeakOuts(seg);
    if (outs.length === 0)
        return null;
    const sig = laneSumSig(track, seg);
    const memo = laneSumMemo.get(seg);
    if (memo && memo.sig === sig)
        return memo.peaks;
    const n = Math.max(...outs.map((o) => o.waveformPeaks.length));
    const sum = new Array(n).fill(0);
    for (const o of outs) {
        if (isLaneRowMuted(track, laneRowKey(o), o.laneId))
            continue;
        const peaks = o.waveformPeaks;
        const pn = peaks.length;
        // Clips are ordered + non-overlapping; `laneEnd` dedups the SHARED boundary bucket of two
        // adjacent clips (floor/ceil of a fractional cut overlap by one bucket — without this every
        // slice point double-added its bucket and drew a 2× spike, review-caught).
        let laneEnd = 0;
        for (const clip of materializeClips(seg.laneOps?.[laneGroupId(o)], stemDurMs)) {
            const b0 = Math.max(laneEnd, Math.floor((clip.start / stemDurMs) * n));
            const b1 = Math.min(n, Math.ceil((clip.end / stemDurMs) * n));
            for (let i = b0; i < b1; i++) {
                sum[i] += peaks[Math.min(pn - 1, Math.floor(((i + 0.5) / n) * pn))];
            }
            laneEnd = Math.max(laneEnd, b1);
        }
    }
    for (let i = 0; i < n; i++)
        if (sum[i] > 1)
            sum[i] = 1;
    laneSumMemo.set(seg, { sig, peaks: sum });
    return sum;
}
/** THE lane volume/pan accessor — group-scoped: keyed by the producing Output node (`groupId` =
 *  laneGroupId), so the mix is "recorded on the Output node" like laneOps and survives renames AND
 *  upstream rewiring (reconnecting the same Output inherits it back). Legacy fallback: pre-S28 saves
 *  keyed these by laneId. Writes always go to the group key (updateLaneControl). */
export function laneControlFor(track, groupId, laneId) {
    return track.laneControls[groupId] ?? track.laneControls[laneId];
}
/** Would a clip spanning [startTick, startTick+durTicks) overlap any existing (non-loading) segment on
 *  `track`? THE shared collision test for placing new content (drag-import placement + S61 paste). */
export function clipCollides(track, startTick, durTicks) {
    const end = startTick + durTicks;
    return track.segments.some((s) => !s.loading && startTick < s.startTick + s.durationTicks && s.startTick < end);
}
const laneLayoutMemo = new WeakMap();
export function getLaneLayout(track) {
    const memo = laneLayoutMemo.get(track);
    if (memo)
        return memo;
    const lanes = getLanes(track);
    const rowY = [];
    const rowRun = [];
    const runs = [];
    const rowByKey = new Map();
    const nameColor = new Map(); // 轨道组 name → first-seen color index
    let y = TRACK_HEADER_HEIGHT;
    for (let i = 0; i < lanes.length; i++) {
        const l = lanes[i];
        const key = l.runKey; // bar-cluster key (union-find component) — NOT laneRunKey (a bar can span 组s)
        let run = runs[runs.length - 1];
        if (!run || run.key !== key) {
            if (!nameColor.has(l.group))
                nameColor.set(l.group, nameColor.size);
            run = { key, groupId: l.groupId, groupIds: [], name: l.group, laneId: l.laneId, start: i, count: 0, barY: y, colorIndex: nameColor.get(l.group) };
            runs.push(run);
            y += LANE_GROUP_BAR_HEIGHT;
        }
        run.count++;
        for (const m of l.members) {
            rowByKey.set(m.rowKey, i);
            if (!run.groupIds.includes(m.groupId))
                run.groupIds.push(m.groupId);
        }
        rowRun.push(runs.length - 1);
        rowY.push(y);
        y += LANE_HEIGHT;
    }
    const layout = { rowY, rowRun, runs, rowByKey, lanesHeight: y - TRACK_HEADER_HEIGHT };
    laneLayoutMemo.set(track, layout);
    return layout;
}
/** Lane-row index at an unscaled Y relative to the TRACK TOP (divide the scaled offset by vZoom
 *  first), or -1 when the Y falls on the header, a group bar, or below the last row. */
export function laneRowAtY(track, yUnscaled) {
    const { rowY } = getLaneLayout(track);
    for (let i = 0; i < rowY.length; i++) {
        if (yUnscaled >= rowY[i] && yUnscaled < rowY[i] + LANE_HEIGHT)
            return i;
    }
    return -1;
}
/** S59: the UNSCALED height of the track's loudness-lane band (0 when closed / not an audio
 *  track). Deliberately OUTSIDE getLaneLayout's rowY — the band is not a lane row, so
 *  laneRowAtY/rowByKey consumers (hit-test, loading overlay, crossfade row positioning) never
 *  see it; it just extends computeTrackHeight below the rows. */
export function loudnessBandH(track) {
    return track.loudnessLaneOpen && track.trackType === "audio" ? LOUDNESS_LANE_HEIGHT : 0;
}
/** Track display height. `scale` is the vertical zoom (vZoom) applied to header + lanes.
 *  Expanded lanes = one row per distinct laneRowKey + one GROUP BAR per 组+名 run (getLaneLayout),
 *  matching the header column's rendering, so the column never overflows its box when a track's
 *  segments emit heterogeneous lane sets. */
export function computeTrackHeight(track, scale = 1) {
    const lanesH = track.expanded ? getLaneLayout(track).lanesHeight : 0;
    return (TRACK_HEADER_HEIGHT + lanesH + loudnessBandH(track)) * scale;
}
/** 被折叠文件夹隐藏的子轨 id 集合: folderCollapsed 的文件夹轨道, 其 folderId 指向它的子轨全部隐藏。
 *  折叠是纯视图状态 — 隐藏只影响布局/绘制, 不影响播放与混音。 */
export function hiddenTrackIds(tracks) {
    const collapsed = new Set(tracks.filter((t) => t.isFolder && t.folderCollapsed).map((t) => t.id));
    if (collapsed.size === 0)
        return new Set();
    const hidden = new Set();
    for (const t of tracks) {
        if (t.folderId && collapsed.has(t.folderId))
            hidden.add(t.id);
    }
    return hidden;
}
export function computeTrackYOffsets(tracks, scale = 1) {
    const hidden = hiddenTrackIds(tracks);
    const offsets = [];
    let y = 0;
    for (const track of tracks) {
        // 隐藏子轨保留占位偏移(复用当前 y, 不推进) — offsets 必须与 tracks 索引对齐
        // (Arrangement 到处按 offsets[i] 取), 由绘制/渲染层跳过这些条目。
        if (hidden.has(track.id)) {
            offsets.push(y);
            continue;
        }
        offsets.push(y);
        y += computeTrackHeight(track, scale);
    }
    return offsets;
}
export function computeTotalTracksHeight(tracks, scale = 1) {
    const hidden = hiddenTrackIds(tracks);
    let h = 0;
    for (const track of tracks) {
        if (hidden.has(track.id))
            continue;
        h += computeTrackHeight(track, scale);
    }
    return h;
}
export function findTrackAtY(offsets, y, tracks) {
    const hidden = tracks ? hiddenTrackIds(tracks) : null;
    for (let i = offsets.length - 1; i >= 0; i--) {
        if (y >= offsets[i]) {
            if (hidden && tracks && hidden.has(tracks[i]?.id ?? ""))
                continue;
            return i;
        }
    }
    return -1;
}
