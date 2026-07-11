export const TICKS_PER_BEAT = 480;
export const PIXELS_PER_TICK = 0.15;
export const TRACK_HEADER_HEIGHT = 48;
export const LANE_HEIGHT = 32;
/** Height of the slim per-组 GROUP BAR above each run of sub-lane rows (group name + the group-level
 *  volume/pan controls, drawn ONCE per 组 — P5). Part of the shared track-height math: the header
 *  column and the canvas both position rows through getLaneLayout, never by `index * LANE_HEIGHT`. */
export const LANE_GROUP_BAR_HEIGHT = 18;
/** S59: height of the audio track's LOUDNESS LANE band (playback clip-gain envelope editor),
 *  appended BELOW the lane rows when track.loudnessLaneOpen — part of computeTrackHeight, never
 *  of getLaneLayout's rowY (it is NOT a lane row; laneRowAtY must keep returning -1 inside it). */
export const LOUDNESS_LANE_HEIGHT = 44;
/** Height of the "+ add track" footer below the track stack — vertical scroll must leave room
 *  for it (it lives inside the scrolled track-header column), so it can't be clipped off. */
export const TRACK_ADD_FOOTER = 36;

/** The default Output-node lane group name. SINGLE source of truth — was duplicated as "Main" (seed
 *  node + engine fallbacks) vs "Output" (newly-added nodes), a drift that produced inconsistent
 *  defaults. P4 replaces the free-text field with a group dropdown; lanes are disambiguated by
 *  `laneId` (the Output node id), so a shared display name no longer collapses rows. */
export const DEFAULT_OUTPUT_GROUP = "Main";

/** The Output node's accent color — ONE constant for the node chrome AND its palette entry (they had
 *  drifted: palette green vs node amber). Amber = the established Output/deposit status color. */
export const OUTPUT_NODE_COLOR = "#f59e0b";

/** Audio file extensions the app accepts — ONE source of truth for every entry
 *  point (timeline drop, training-page drop, file-picker dialog filters). The
 *  dialog list and the drop regex MUST stay the same set: a format pickable in
 *  the dialog but silently ignored on drop (or vice versa) is drift. */
export const AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "aac", "m4a", "webm", "opus", "wma"];
export const AUDIO_EXT_RE = new RegExp(`\\.(${AUDIO_EXTENSIONS.join("|")})$`, "i");

/** Score/notation formats the "导入 / Import" File-menu action accepts — ONE source of truth for the
 *  native dialog filter (src/lib/vocal/import.ts). ust/ustx are 480-ppq (1:1 with our resolution); midi
 *  is scaled from its header PPQ. The Rust `import_score_file` command dispatches on this same set. */
export const SCORE_EXTENSIONS = ["ustx", "ust", "mid", "midi"];

/** S59: upper bound for a stored TempoDetect.downbeat phase. The UI meter numerator maxes at 16,
 *  and the ×2 grid correction stores the phase mod (2·bpb) to keep ÷2 a perfect round-trip — so
 *  the legal range is [0, 31]. ONE constant for the store's canonical write clamp AND the .usp
 *  load sanitizer (they must agree or a saved value bounces on reload). */
export const MAX_DOWNBEAT = 31;

/** Volume-fader floor (dB). The BOTTOM of a volume fader means −∞/MUTE (the universal DAW convention:
 *  bounded upward, unbounded downward) — dbToLinear maps any value at/below this to gain 0, and the
 *  fader tooltip shows "-∞ dB" there. The STORED volumeDb stays at this finite floor (−Infinity does
 *  not survive JSON). ONE constant for the faders (TrackList), the gain math (playback), and the
 *  future mixdown export. */
export const FADER_MIN_DB = -24;

/** Volume-fader ceiling (dB). Bounded upward (unlike the floor); +24 dB gives real make-up headroom
 *  (the old +6 couldn't push a quiet stem). ONE constant for the faders (TrackList) — the gain math
 *  (dbToLinear) has no upper clamp, so this is purely the UI travel limit. */
export const FADER_MAX_DB = 24;
