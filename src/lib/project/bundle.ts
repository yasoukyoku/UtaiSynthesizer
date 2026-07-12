import type { Track, Segment, ProcessedOutput, PitchCurve } from "../../types/project";
import { laneGroupName, rowKeyLaneId } from "../audio/laneOps";
import { normalizeNotesArray, normalizeCurve, sanitizeVocalParams, sanitizeText } from "../vocalNotes";
import { MAX_DOWNBEAT } from "../constants";
import { LOUDNESS_DB_RANGE } from "../vocalGeometry";

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
    // S59 loudness-lane band: same view-state posture as expanded, plus delete-when-falsy in SAVED
    // bundles too (optional field — an open→close cycle must not leave `false` in the JSON, else a
    // never-touched project and a touched-then-restored one differ byte-wise; playOriginal precedent).
    if (stripView || !track.loudnessLaneOpen) delete (track as { loudnessLaneOpen?: boolean }).loudnessLaneOpen;
    // S59b per-group envelope visibility — identical posture (the store toggle keeps only true
    // entries with sorted keys, so a kept record is already canonical).
    if (stripView || !track.laneLoudnessOpen || Object.keys(track.laneLoudnessOpen).length === 0) {
      delete (track as { laneLoudnessOpen?: Record<string, boolean> }).laneLoudnessOpen;
    }
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
      // S61 MIGRATION: the retired Effects node family (persisted as pitchShift / formantShift /
      // audioEnhance) → the Transpose node, HERE at the single load choke point so neither the
      // editor nor the engine ever sees the legacy types. Node id/position/edges are kept (lane
      // identity embeds node ids — re-keying would orphan deposited lanes). Carried: the summed
      // pitchShift semitones. Dropped by user decision (S61 效果器退役): formantShift (the World
      // path did work — its capability now lives on the voice nodes' formant param) and enhance
      // (never implemented, Err stub). Deposited lanes rendered by the old node keep playing;
      // only a RE-render goes through the new transpose semantics.
      if (rest.workflow?.nodes.some((n) => ["pitchShift", "formantShift", "audioEnhance"].includes(n.nodeType as string))) {
        rest.workflow = {
          ...rest.workflow,
          nodes: rest.workflow.nodes.map((n) => {
            if (!["pitchShift", "formantShift", "audioEnhance"].includes(n.nodeType as string)) return n;
            const fx = Array.isArray(n.params?.effects)
              ? (n.params.effects as Array<{ type?: string; params?: Record<string, unknown> }>)
              : [];
            // SUM every stacked pitchShift entry (the legacy chain applied them sequentially =
            // additive semitones; taking only the first silently changed old projects' total
            // transposition — audit S61). Fractional values are preserved (the Rust command takes
            // f64); formantShift/enhance entries have no transpose equivalent and drop to 0.
            const legacySum = fx
              .filter((e) => e?.type === "pitchShift")
              .reduce((acc, e) => acc + (Number(e?.params?.semitones) || 0), 0);
            const rawSemis = fx.length > 0 ? legacySum : Number(n.params?.semitones ?? 0);
            const semitones = Number.isFinite(rawSemis) ? Math.max(-24, Math.min(24, rawSemis)) : 0;
            return { ...n, nodeType: "transpose" as const, params: { semitones } };
          }),
        };
      }
      let content = rest.content;
      if (content.type === "audioClip") {
        content = { ...content, sourcePath: resolve(content.sourcePath) };
        // UNTRUSTED LOAD BOUNDARY for the S59 clip fields (same posture as the notes funnel below):
        // a corrupt stretch/tempoDetect must not reach tick math (NaN durations) or the canvas.
        if (content.stretch !== undefined) {
          const r = Number(content.stretch);
          if (!Number.isFinite(r) || r <= 0 || Math.abs(r - 1) < 1e-9) delete content.stretch;
          else content.stretch = Math.min(4, Math.max(0.25, r));
        }
        if (content.tempoDetect !== undefined) {
          const td = content.tempoDetect as unknown as Record<string, unknown>;
          const bpm = Number(td?.bpm);
          const anchorMs = Number(td?.anchorMs);
          const downbeat = Number(td?.downbeat);
          const conf = Number(td?.conf);
          if (Number.isFinite(bpm) && bpm >= 20 && bpm <= 400 && Number.isFinite(anchorMs) && anchorMs >= 0) {
            content.tempoDetect = {
              bpm,
              anchorMs,
              downbeat: Number.isFinite(downbeat) ? Math.min(MAX_DOWNBEAT, Math.max(0, Math.round(downbeat))) : 0,
              conf: Number.isFinite(conf) ? Math.min(1, Math.max(0, conf)) : 0,
              ...(td?.notConstant === true ? { notConstant: true as const } : {}),
            };
          } else {
            delete content.tempoDetect;
          }
        }
        if (content.paramCurves !== undefined) {
          let paramCurves: Record<string, PitchCurve> | undefined;
          if (content.paramCurves && typeof content.paramCurves === "object") {
            const out: Record<string, PitchCurve> = {};
            for (const k of Object.keys(content.paramCurves).sort()) {
              const key = sanitizeText(k, 32);
              let nc = normalizeCurve(content.paramCurves[k], "param");
              // the loudness curve feeds a playback GAIN (10^(dB/20)) — clamp a hostile file's
              // ys to the lane's legal dB range so no absurd gain can reach the AudioParam
              if (nc && key === "loudness") {
                nc = {
                  xs: nc.xs,
                  ys: nc.ys.map((y) => Math.max(-LOUDNESS_DB_RANGE, Math.min(LOUDNESS_DB_RANGE, y))),
                };
              }
              if (key && nc) out[key] = nc;
            }
            if (Object.keys(out).length > 0) paramCurves = out;
          }
          if (paramCurves) content.paramCurves = paramCurves;
          else delete content.paramCurves;
        }
      } else if (content.type === "notes") {
        // UNTRUSTED LOAD BOUNDARY (§9.8.1): a hand-edited / corrupt .usp must not slip NaN ticks, out-of-
        // range pitch, control-char lyrics, or unbounded arrays past the editor's clamps straight into the
        // store (→ permanent false-dirty + canvas NaN + Phase-6 out-of-range index). The SAME canonical
        // funnel the editor writes with (normalizeNotesArray / normalizeCurve) runs here too.
        const pitchDev = normalizeCurve(content.pitchDev, "cents");
        let paramCurves: Record<string, PitchCurve> | undefined;
        if (content.paramCurves && typeof content.paramCurves === "object") {
          const out: Record<string, PitchCurve> = {};
          for (const k of Object.keys(content.paramCurves).sort()) {
            const key = sanitizeText(k, 32);
            const nc = normalizeCurve(content.paramCurves[k], "param");
            if (key && nc) out[key] = nc;
          }
          if (Object.keys(out).length > 0) paramCurves = out;
        }
        content = {
          type: "notes",
          notes: normalizeNotesArray(content.notes ?? []),
          ...(pitchDev ? { pitchDev } : {}),
          ...(paramCurves ? { paramCurves } : {}),
        };
      }
      // S59b UNTRUSTED LOAD BOUNDARY for the per-group loudness envelopes (same funnel as the
      // clip curves: normalizeCurve + dB clamp; hostile group keys sanitized, empty bag dropped).
      // audioClip-only: a hand-edited file must not smuggle a playback-domain envelope onto a
      // vocal bake (vocal loudness is render-domain by design).
      if (rest.laneLoudness !== undefined) {
        let laneLoudness: Record<string, PitchCurve> | undefined;
        if (content.type === "audioClip" && rest.laneLoudness && typeof rest.laneLoudness === "object") {
          const out: Record<string, PitchCurve> = {};
          for (const k of Object.keys(rest.laneLoudness).sort()) {
            const key = sanitizeText(k, 64);
            const nc = normalizeCurve(rest.laneLoudness[k], "param");
            if (key && nc) {
              out[key] = {
                xs: nc.xs,
                ys: nc.ys.map((y) => Math.max(-LOUDNESS_DB_RANGE, Math.min(LOUDNESS_DB_RANGE, y))),
              };
            }
          }
          if (Object.keys(out).length > 0) laneLoudness = out;
        }
        if (laneLoudness) rest.laneLoudness = laneLoudness;
        else delete rest.laneLoudness;
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
    // Untrusted load boundary for the track's vocal params (§9.8.1): coerce backend enum, clamp
    // speaker/lang/transpose. Absent (non-vocal track) → stays absent.
    const track: Track = { ...t, segments, laneControls };
    const vp = sanitizeVocalParams(t.vocalParams);
    if (vp) track.vocalParams = vp;
    else delete (track as { vocalParams?: unknown }).vocalParams;
    return track;
  });

  return {
    name: data.name ?? "Untitled",
    tracks,
    tempo: data.tempo ?? 120,
    timeSignature: data.timeSignature ?? [4, 4],
  };
}
