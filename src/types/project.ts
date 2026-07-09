export interface LaneControl {
  volumeDb: number;
  pan: number;
  muted: boolean;
}

export interface Track {
  id: string;
  name: string;
  trackType: "vocal" | "audio" | "instrument";
  segments: Segment[];
  volumeDb: number;
  pan: number;
  muted: boolean;
  solo: boolean;
  voiceModel?: string;
  voiceModelAvatar?: string;
  /** ② Vocal-track (自己唱) settings — backend + the ScoreToCV speaker/lang + a track-level transpose.
   *  Present only on vocal tracks that have been configured; absent = defaults. Persisted + UNDOABLE
   *  (in meaningfulSig, like voiceModel). The SVC voice itself stays in `voiceModel`; the render-time SVC
   *  inference knobs (noise_scale…) join here in Phase 6 when the vocal render is wired. (S48 Phase 3) */
  vocalParams?: VocalTrackParams;
  expanded: boolean;
  /** Per-GROUP mix (volume/pan), keyed by the producing Output node id (`laneGroupId`) — "recorded ON
   *  the Output node", exactly like laneOps: all lanes of one 组 share the setting (解组 to control
   *  independently), a 轨道组 rename OR any upstream rewiring (insert an effects node, reconnect to the
   *  same Output) never re-keys it, and an ungroup inherits it per new node. Future loudness envelopes
   *  live at this same identity. Read through `laneControlFor` (legacy pre-S28 saves keyed by laneId —
   *  the fallback). `muted` inside is LEGACY too — mute lives in `laneMutes` via `isLaneRowMuted`. */
  laneControls: Record<string, LaneControl>;
  /** Per-ROW mute, keyed by `laneRowKey` (轨道组 name + laneId). Deliberately LOOSER than laneControls:
   *  mute is a view/audibility toggle on the ROW you see — resets on rename/ungroup (one click to
   *  redo, one predicate or display/export disagrees with playback), and diverged split-half rows mute
   *  independently. THE "audible or not" source of truth (via isLaneRowMuted) — the future mixdown
   *  export + overall-waveform display MUST consult the same predicate, never laneControls.muted
   *  directly. Absent on old saves. */
  laneMutes?: Record<string, boolean>;
  /** SOURCE selector: true = this track plays its ORIGINAL audio, bypassing the deposited sub-lanes
   *  (they leave the output entirely — playback AND the future mixdown export; a Mute/Solo-class
   *  state, persisted + undoable). Default false = sub-lanes play whenever a segment has ready ones.
   *  NEVER read this (or processedOutputs presence) directly to decide the source — go through
   *  `segmentPlaysLanes` (trackLayout), THE one predicate shared by playback, the main-row waveform,
   *  and (future) mixdown, so what you see is always what you hear. */
  playOriginal?: boolean;
}

/** One kept audio piece of a sub-lane GROUP within a segment, in STEM MILLISECONDS (absolute position
 *  in the rendered stem, 0 = stem start). Non-destructive: the recipe of which portions of the rendered
 *  audio play — the stem file itself is untouched (D2). Stem-ms is INVARIANT under the parent segment's
 *  move / split / resize / tempo change (those only shift the visible window [offsetMs, offsetMs+durMs]
 *  into the stem), so ops never need re-basing — read-time they're intersected with the window. A missing
 *  `laneOps[outputNodeId]` entry = the whole lane plays (implicit); an empty `[]` = explicitly silenced. */
export interface LaneClip {
  /** Start position in the stem, milliseconds. */
  start: number;
  /** End position in the stem, milliseconds. */
  end: number;
}

export interface ProcessedOutput {
  /** Stable per-lane IDENTITY = the producing Output node id (+ `::stem` when that node fans out
   *  multiple stems). The key for rendering rows / selection / laneControls — distinct even when two
   *  Output nodes share a display `laneLabel`, so same-named lanes never collapse onto one row. */
  laneId: string;
  /** Human DISPLAY name ("Group" or "Group · stem"). NOT an identity — may collide across nodes
   *  (the header row de-collides visually by numbering, see getLanes). */
  laneLabel: string;
  /** The producing Output node's GROUP name at deposit time (laneLabel's base, no stem suffix).
   *  Part of the ROW identity (`laneRowKey` = group + laneId) so two split halves that share a laneId
   *  but DIVERGE their group (rename one half's Output node) get separate rows instead of the
   *  first-seen label swallowing the sibling. Backfilled from laneLabel on load for older saves. */
  group?: string;
  audioPath: string;
  totalDurationMs: number;
  waveformPeaks?: number[];
  /** Which Output node produced this lane. Lets a per-node deposit replace only that node's OWN prior
   *  contribution (merge by node identity, not by laneLabel) so two Output nodes sharing a lane name
   *  don't clobber each other. Optional/undefined on legacy projects (merge falls back to laneLabel). */
  outputNodeId?: string;
  /** True while an Output-node deposit is decoding this lane's audio — the track renders a loading
   *  placeholder (same look as an audio import) until the real waveform is merged in. */
  loading?: boolean;
}

export interface Segment {
  id: string;
  startTick: number;
  durationTicks: number;
  content: SegmentContent;
  workflow?: Workflow;
  processedOutputs?: ProcessedOutput[];
  /** Non-destructive sub-lane edits (slice / edge-stretch / delete), keyed by the producing Output
   *  node id (the GROUP — all lanes fanned into one Output node share one recipe: "group-operate").
   *  Each value is the list of kept audio pieces in STEM MS (see LaneClip). UNLIKE processedOutputs
   *  (the baked render = a non-undoable overlay), laneOps is an ARRANGEMENT edit: it IS in the history
   *  meaningfulSig (undoable) and survives a re-render (keyed by node id, not baked into the audio). */
  laneOps?: Record<string, LaneClip[]>;
  /** True while the audio file backing this segment is still being decoded after a drag/import.
   *  A loading segment renders as a striped placeholder and is skipped during playback;
   *  `content.totalDurationMs` holds the probed (approximate) duration until decode finishes. */
  loading?: boolean;
}

export type SegmentContent =
  | {
      type: "notes";
      notes: Note[];
      /** ② Hand-drawn ADDITIVE f0 offset over the whole part (SynthV "Pitch Deviation"), in cents,
       *  X = ticks relative to the segment start. Adds ON TOP of the note-derived baseline (§3.2 layer ③);
       *  a paint gesture REPLACES the covered x-interval. Absent = no manual deviation. (S48 Phase 3) */
      pitchDev?: PitchCurve;
      /** ② Per-parameter automation lanes (loudness / tension / breath / gender …), keyed by param name.
       *  Same PitchCurve shape (X = ticks rel. segment start, Y = param value). Absent = all defaults. */
      paramCurves?: Record<string, PitchCurve>;
    }
  | { type: "audioClip"; sourcePath: string; offsetMs: number; totalDurationMs: number };

/** One vocal note (§3.1 "VocalNote"). A SUPERSET of the original 7-field Note: the base fields are the
 *  musical note; the optional fields (all absent = a plain note at its谱-derived pitch) carry the pitch/
 *  expression edits SynthV/OpenUTAU expose. UNITS ARE FIXED: X = ticks (480 PPQ), Y = cents — end to end.
 *  Every optional is written ONLY when non-default (the store omits defaults) so the raw-JSON
 *  save/autosave compare stays byte-stable (§5 false-dirty rule). All fields are UNDOABLE (contentSig). */
export interface Note {
  id: string;
  tick: number;
  duration: number;
  pitch: number;
  lyric: string;
  phoneme?: string;
  velocity: number;
  /** Fine pitch offset in cents (± ), added to `pitch`. Absent = 0. */
  detune?: number;
  /** ② Per-note pitch-TRANSITION override (SynthV Pitch Transition, §10.3). Shapes how this note connects
   *  to its neighbours (glide in from prev / out to next). Every field optional → absent fields fall back
   *  to the track default (VocalTrackParams.transition). Absent whole = pure track default. */
  transition?: NoteTransition;
  /** ④ Tail/mid vibrato (SynthV model). All fields present when on; absent = none. */
  vibrato?: VibratoSpec;
  /** false = the note's pitch baseline is FROZEN to the user's manual edits (v1 "Path B"); absent/true =
   *  re-derived from the score (Path A). Stored ONLY when false. */
  pitchAuto?: boolean;
  /** Explicit tie / sustain to the previous note (承前元音 legato). Stored ONLY when true. */
  tie?: boolean;
  /** Per-note language override (zh/ja/en/de/fr/es/it). Absent = follow the track default (§3.7 ACE-style). */
  lang?: string;
  /** User override at the TRADITIONAL-phoneme layer (拼音/假名/ARPABET — NOT raw IPA); stage2 converts it
   *  to IPA at render (§3.7). Absent = derive from `lyric`. */
  phonemeInput?: string;
}

/** ② SynthV Pitch Transition — how a note connects to its neighbours (§10.3). ALL times are ABSOLUTE ms
 *  (NOT ticks) so a glide sounds the same at any tempo; overshoot depths are cents. As a per-note override
 *  every field is optional (absent → the concrete track default in VocalTrackParams.transition). */
export interface NoteTransition {
  /** Shift the whole cross-note transition earlier(−)/later(+), ms (SynthV Offset). */
  offsetMs?: number;
  /** How long AFTER this note's onset the pitch arrives from the previous note (arrive-late), ms ≥ 0. */
  durLeftMs?: number;
  /** How long BEFORE this note's end the pitch begins leaving toward the next note (leave-early), ms ≥ 0. */
  durRightMs?: number;
  /** Arrival overshoot at the left transition, signed cents (SynthV Depth Left; ~15¢ default = human feel). */
  depthLeftCents?: number;
  /** Departure overshoot at the right transition, signed cents (SynthV Depth Right). */
  depthRightCents?: number;
}

/** An ordered polyline (X = ticks, Y = cents/param-value). Parallel arrays keep it compact + JSON-stable;
 *  painting replaces the covered x-interval. `xs` is strictly increasing; `xs.length === ys.length`. */
export interface PitchCurve {
  xs: number[];
  ys: number[];
}

/** ④ Vibrato (SynthV model, §10.3). Times ABSOLUTE ms; frequency Hz; amplitude cents. The onset delay
 *  (startMs) is why short notes don't visibly vibrate. (jitter = natural pitch flutter — deferred Phase 6.) */
export interface VibratoSpec {
  /** Amplitude in cents (peak deviation, ± around the base). SynthV default ≈ 100¢ (1 semitone). */
  depthCents: number;
  /** Oscillation rate in Hz (SynthV 1–10, default 5.5). */
  freqHz: number;
  /** Start phase, −1…+1 (fraction of a cycle). */
  phase: number;
  /** Onset delay after the note's start, ms (short notes stay flat). */
  startMs: number;
  /** Linear fade-in / fade-out durations, ms. */
  easeInMs: number;
  easeOutMs: number;
}

/** ② Vocal-track (自己唱) parameters (§3.1). The SVC voice/singer stays in `Track.voiceModel`; this holds
 *  the backend choice + the ScoreToCV conditioning (speaker/lang) + a track-level transpose. */
export interface VocalTrackParams {
  backend: "rvc" | "sovits";
  /** ScoreToCV speaker id (0–76; near speaker-invariant, default 49 = kiritan). NOT the SVC voice. */
  speakerId: number;
  /** ScoreToCV language id (zh0 ja2 en1 de3 fr4 es5 it6). */
  langId: number;
  /** Track-level transpose in semitones, applied to every note's pitch → f0. */
  transpose: number;
  /** Track-level DEFAULT note transition — every field concrete. A note's NoteTransition overrides it
   *  per-field, so every note has a smooth SynthV-style glide by default (§10.3). */
  transition: Required<NoteTransition>;
}

export interface Workflow {
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
}

export interface WorkflowNode {
  id: string;
  nodeType: WorkflowNodeType;
  position: { x: number; y: number };
  params: Record<string, unknown>;
}

export type WorkflowNodeType =
  | "input"
  | "output"
  | "rvc"
  | "sovits"
  | "pitchShift"
  | "formantShift"
  | "audioEnhance"
  | "msstSeparation"
  | "split";

export interface WorkflowConnection {
  fromNode: string;
  fromPort: number;
  toNode: string;
  toPort: number;
}
