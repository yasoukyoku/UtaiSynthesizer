import type { Track, Segment, ProcessedOutput } from "../../types/project";
import { laneGroupName, rowKeyLaneId } from "../audio/laneOps";

/**
 * `.usp` project-bundle (de)serialization. A `.usp` is a self-contained FOLDER:
 *   <name>.usp/
 *     project.json   — the document (this shape), with media/render paths bundle-RELATIVE
 *     media/         — copies of the original source audio (so the project is portable)
 *     renders/       — copies of the processed-output (render) audio (survive the cache sweep)
 *
 * The TS store is the authoritative document model; Rust only does the file copying + JSON I/O. On
 * save we rewrite absolute media/render paths to relative + emit the copy list; on open we resolve
 * them back to absolute (playback/decode need absolute paths). Voice models/avatars stay as external
 * absolute references (they're shared assets, not project content).
 */

const USP_FORMAT = "usp";
const USP_VERSION = 1;

export interface SaveBundle {
  projectJson: string;
  copies: { from: string; to: string }[];
}

export interface LoadedProject {
  name: string;
  tracks: Track[];
  tempo: number;
  timeSignature: [number, number];
}

function basename(p: string): string {
  return p.split(/[/\\]/).pop() || "file";
}

function splitExt(name: string): [string, string] {
  const i = name.lastIndexOf(".");
  return i > 0 ? [name.slice(0, i), name.slice(i)] : [name, ""];
}

/**
 * Serialize the document to the `project.json` shape, routing every media/render ABSOLUTE path through
 * `mapPath`. This is the ONE definition of what goes into project.json (which fields, strip-loading,
 * processedOutputs) so full-save and autosave can never drift as the document grows:
 *   - `buildSaveBundle`  → mapPath consolidates into the bundle (relative dest + copy list)
 *   - `buildAutosaveJson` → mapPath = identity (keep absolute paths, copy nothing)
 */
function serializeProject(
  name: string,
  tracks: Track[],
  tempo: number,
  timeSignature: [number, number],
  mapPath: (absPath: string, subdir: "media" | "renders") => string,
  stripView = false,
): string {
  const outTracks: Track[] = tracks.map((t) => {
    const segments = t.segments.map((s): Segment => {
      const rest: Segment = { ...s };
      delete rest.loading; // transient runtime flag — never persisted
      let content = rest.content;
      if (content.type === "audioClip") {
        content = { ...content, sourcePath: mapPath(content.sourcePath, "media") };
      }
      let processedOutputs: ProcessedOutput[] | undefined = rest.processedOutputs;
      if (processedOutputs) {
        // Drop transient deposit loading placeholders (loading=true, no real waveform yet) — persisting
        // one would reload as a permanently-spinning lane. Strip the runtime `loading` flag from the rest.
        processedOutputs = processedOutputs
          .filter((o) => !o.loading)
          .map(({ loading: _l, ...o }) => ({ ...o, audioPath: mapPath(o.audioPath, "renders") }));
      }
      return { ...rest, content, processedOutputs };
    });
    const track: Track = { ...t, segments };
    // Normalize `playOriginal` so absent == false in EVERY serialized document (an O on→off cycle
    // stores an explicit false; the raw-JSON content-compare in autosave/hasUnsavedWork would read
    // that as unsaved work vs a pre-toggle baseline where the key is absent). Unconditional — full
    // saves, autosaves, and the savedJson baseline must all agree.
    if (!track.playOriginal) delete (track as { playOriginal?: boolean }).playOriginal;
    // Same normalization for `laneMutes` — drop false entries (and the whole key when empty), or a
    // row-mute on→off cycle never returns to the byte-identical baseline (spurious "unsaved changes"
    // close prompt + endless autosave churn). EXCEPTION: keep an explicit false that MASKS a legacy
    // per-laneId muted flag — isLaneRowMuted is `laneMutes[rowKey] ?? laneControls[laneId].muted ?? false`,
    // so for THOSE rows absent ≠ false (dropping the mask would resurrect the legacy mute on reload).
    if (track.laneMutes) {
      const kept = Object.fromEntries(
        Object.entries(track.laneMutes).filter(
          ([k, v]) => v || track.laneControls[rowKeyLaneId(k)]?.muted === true,
        ),
      );
      if (Object.keys(kept).length > 0) track.laneMutes = kept;
      else delete (track as { laneMutes?: Record<string, boolean> }).laneMutes;
    }
    // `expanded` is pure view state (excluded from undo + dirty). In the AUTOSAVE path strip it so a
    // collapse/expand toggle isn't seen as "unsaved content" by the content-compare. Saved bundles keep it.
    if (stripView) delete (track as { expanded?: boolean }).expanded;
    return track;
  });

  const project = { format: USP_FORMAT, version: USP_VERSION, name, tempo, timeSignature, tracks: outTracks };
  return JSON.stringify(project, null, 2);
}

/** Serialize the document and compute the media/render files to copy into the bundle. */
export function buildSaveBundle(
  name: string,
  tracks: Track[],
  tempo: number,
  timeSignature: [number, number],
): SaveBundle {
  const copies: { from: string; to: string }[] = [];
  const absToRel = new Map<string, string>();
  const usedRel = new Set<string>();

  // Map an absolute media/render source path to a unique bundle-relative dest, recording the copy.
  // Same absolute path → same dest (dedup); basename collisions get a numeric suffix. The collision
  // check is CASE-INSENSITIVE: NTFS extraction is, so "Vocal.wav" and "vocal.wav" would alias one
  // extracted file on open (the second overwrite wins → a segment plays the wrong audio).
  const consolidate = (absPath: string, subdir: "media" | "renders"): string => {
    const existing = absToRel.get(absPath);
    if (existing) return existing;
    const [stem, ext] = splitExt(basename(absPath));
    let rel = `${subdir}/${stem}${ext}`;
    let i = 1;
    while (usedRel.has(rel.toLowerCase())) {
      rel = `${subdir}/${stem}_${i}${ext}`;
      i++;
    }
    usedRel.add(rel.toLowerCase());
    absToRel.set(absPath, rel);
    copies.push({ from: absPath, to: rel });
    return rel;
  };

  const projectJson = serializeProject(name, tracks, tempo, timeSignature, consolidate);
  return { projectJson, copies };
}

/** Serialize for AUTOSAVE — the SAME field set as `buildSaveBundle` (shared `serializeProject`, so
 *  autosave can never drift from save as features are added), but media/render paths stay ABSOLUTE and
 *  nothing is copied: autosave is a fast, frequent crash-recovery snapshot that references your original
 *  files in place. On restore, `parseLoadedBundle` leaves these absolute paths untouched (only
 *  `media/`-/`renders/`-prefixed bundle-relative paths get resolved against a dir). */
export function buildAutosaveJson(
  name: string,
  tracks: Track[],
  tempo: number,
  timeSignature: [number, number],
): string {
  return serializeProject(name, tracks, tempo, timeSignature, (abs) => abs, true);
}

/** Parse a loaded project.json, resolving bundle-relative media/render paths to absolute against `dir`. */
export function parseLoadedBundle(projectJson: string, dir: string): LoadedProject {
  const data = JSON.parse(projectJson) as {
    name?: string;
    tempo?: number;
    timeSignature?: [number, number];
    tracks?: Track[];
  };
  const base = dir.replace(/\\/g, "/").replace(/\/+$/, "");
  const resolve = (p: string): string =>
    p.startsWith("media/") || p.startsWith("renders/") ? `${base}/${p}` : p; // else external/absolute

  const tracks: Track[] = (data.tracks ?? []).map((t) => {
    const segments = (t.segments ?? []).map((s): Segment => {
      const rest: Segment = { ...s };
      delete rest.loading;
      let content = rest.content;
      if (content.type === "audioClip") {
        content = { ...content, sourcePath: resolve(content.sourcePath) };
      }
      let processedOutputs = rest.processedOutputs;
      if (processedOutputs) {
        // Mirror the write-side invariant: a persisted loading placeholder must never reload as a
        // permanently-spinning lane (it would drive an endless rAF redraw). Drop loading=true outputs +
        // strip the runtime flag, so a stale placeholder from ANY source (old file / partial autosave)
        // can't resurrect a spinning lane.
        // Backfill laneId for any pre-laneId render (lanes are keyed by laneId now; fall back to the
        // producing node id, then the label) and DE-COLLIDE duplicates within the segment so legacy
        // shapes (a multi-stem single Output node, or same-label lanes with no outputNodeId) keep
        // separate rows. Replaces the old same-label rename migration — distinct laneIds do the job.
        const usedLaneIds = new Set<string>();
        // The segment's own Output nodes — the authoritative source for group backfill (the node's
        // laneLabel param IS the group; the label-base fallback truncates a group name that itself
        // contains " · ") and the guard for outputNodeId synthesis (a fabricated id that matches no
        // node would get the lane orphan-cleaned = silent loss of a baked render).
        const wfNodeGroup = new Map<string, string>();
        for (const n of rest.workflow?.nodes ?? []) {
          if (n.nodeType === "output" && typeof n.params?.laneLabel === "string") {
            wfNodeGroup.set(n.id, n.params.laneLabel as string);
          }
        }
        // A lane with outputNodeId but NO laneId (the interim multi-stem save shape) must get the
        // CURRENT edge-based id — a bare node-id laneId can never match laneIdFor's
        // `${out}::${fromNode}:${fromPort}`, so rehydrate leaves the render cache cold AND the
        // reconciler's KEEP branch never matches → mounting that segment's editor DELETES the
        // persisted lanes ("uncached + idle → no lane"). Pair each node's laneless lanes, in order,
        // with the connections into it (the same array order parseWorkflowGraph builds inEdges from);
        // the bare-node-id fallback below remains for lanes whose connections ran out / are missing.
        const wfEdges = new Map<string, { fromNode: string; fromPort: number }[]>();
        for (const c of rest.workflow?.connections ?? []) {
          if (!wfNodeGroup.has(c.toNode)) continue;
          const list = wfEdges.get(c.toNode) ?? [];
          list.push({ fromNode: c.fromNode, fromPort: c.fromPort });
          wfEdges.set(c.toNode, list);
        }
        // Synthesize ONLY when the pairing is unambiguous: the node's laneless-lane count equals its
        // edge count. A deleted-edge save (more lanes than edges) or a mixed laneless/laneId shape
        // would otherwise pair a lane with the WRONG surviving edge — the reconciler would then
        // relabel it to that edge's stem and rehydrate would warm the wrong port. Ambiguous nodes
        // keep the bare-node-id fallback (de-collided) for ALL their laneless lanes.
        const lanelessCount = new Map<string, number>();
        for (const o of processedOutputs) {
          if (!o.loading && o.laneId === undefined && o.outputNodeId) {
            lanelessCount.set(o.outputNodeId, (lanelessCount.get(o.outputNodeId) ?? 0) + 1);
          }
        }
        const edgeCursor = new Map<string, number>();
        processedOutputs = processedOutputs
          .filter((o) => !o.loading)
          .map(({ loading: _l, ...o }) => {
            let laneId = o.laneId;
            if (laneId === undefined && o.outputNodeId) {
              const edges = wfEdges.get(o.outputNodeId);
              if (edges && edges.length === lanelessCount.get(o.outputNodeId)) {
                const n = edgeCursor.get(o.outputNodeId) ?? 0;
                edgeCursor.set(o.outputNodeId, n + 1);
                const e = edges[n];
                if (e) laneId = `${o.outputNodeId}::${e.fromNode}:${e.fromPort}`;
              }
            }
            laneId = laneId ?? o.outputNodeId ?? o.laneLabel;
            if (usedLaneIds.has(laneId)) {
              let i = 2;
              while (usedLaneIds.has(`${laneId}#${i}`)) i++;
              laneId = `${laneId}#${i}`;
            }
            usedLaneIds.add(laneId);
            // Backfill for pre-`group` saves: prefer the producing Output node's laneLabel param, then
            // the label base. Synthesize outputNodeId ONLY when the laneId prefix matches a REAL node
            // in this segment's workflow (a label-shaped laneId that merely contains "::" must stay
            // outputNodeId-less to keep its legacy-safe orphan-cleanup exemption).
            const prefix = laneId.includes("::") ? laneId.split("::")[0]! : undefined;
            const outputNodeId = o.outputNodeId ?? (prefix && wfNodeGroup.has(prefix) ? prefix : undefined);
            const group = o.group ?? (outputNodeId ? wfNodeGroup.get(outputNodeId) : undefined) ?? laneGroupName(o);
            return { ...o, laneId, group, outputNodeId, audioPath: resolve(o.audioPath) };
          });
      }
      return { ...rest, content, processedOutputs };
    });
    // laneControls key by laneId — same as every pre-P4 save, so NO migration is needed. The one
    // exception: a brief interim build keyed them by the ROW key (`group\u0000laneId`); strip that
    // prefix back to the laneId (a literal laneId entry, if both exist, wins).
    let laneControls = t.laneControls ?? {};
    if (Object.keys(laneControls).some((k) => k.includes("\u0000"))) {
      const fixed: typeof laneControls = {};
      for (const [key, ctrl] of Object.entries(laneControls)) {
        if (!key.includes("\u0000")) fixed[key] = ctrl;
      }
      for (const [key, ctrl] of Object.entries(laneControls)) {
        const laneId = key.includes("\u0000") ? key.slice(key.indexOf("\u0000") + 1) : key;
        if (!(laneId in fixed)) fixed[laneId] = ctrl;
      }
      laneControls = fixed;
    }
    // P5 group-bar promotion: the mixer now shows ONE control per 组 (laneControls[groupId]); pre-S28
    // saves key these per-laneId ("nodeId::stem"), possibly with DIVERGENT values across one group's
    // rows — which the group model can't represent, and which made the bar display row 1's value while
    // other rows audibly played their own. Promote the FIRST-seen "::" entry of each group to the bare
    // group key when none exists, so display == playback for every row. Original entries are KEPT
    // (isLaneRowMuted still reads the per-laneId legacy muted flag; laneControlFor prefers the group key).
    {
      const promoted = { ...laneControls };
      let changed = false;
      for (const [key, ctrl] of Object.entries(laneControls)) {
        const idx = key.indexOf("::");
        if (idx <= 0) continue;
        const groupKey = key.slice(0, idx);
        if (promoted[groupKey]) continue; // a real group entry / an earlier promotion wins (first-seen)
        // volume/pan ONLY — never the legacy per-ROW mute flag: isLaneRowMuted falls back to
        // laneControls[laneId].muted, and a pre-laneId row whose laneId EQUALS the node id would
        // read the promoted entry and come up wrongly muted (review-caught). The group entry's
        // muted is documented inert; per-row mute state stays on the kept per-laneId entries.
        promoted[groupKey] = { ...ctrl, muted: false };
        changed = true;
      }
      if (changed) laneControls = promoted;
    }
    return { ...t, segments, laneControls };
  });

  return {
    name: data.name ?? "Untitled",
    tracks,
    tempo: data.tempo ?? 120,
    timeSignature: data.timeSignature ?? [4, 4],
  };
}
