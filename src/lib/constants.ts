export const TICKS_PER_BEAT = 480;
export const PIXELS_PER_TICK = 0.15;
export const TRACK_HEADER_HEIGHT = 48;
export const LANE_HEIGHT = 32;
/** Height of the slim per-组 GROUP BAR above each run of sub-lane rows (group name + the group-level
 *  volume/pan controls, drawn ONCE per 组 — P5). Part of the shared track-height math: the header
 *  column and the canvas both position rows through getLaneLayout, never by `index * LANE_HEIGHT`. */
export const LANE_GROUP_BAR_HEIGHT = 18;
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

/** Volume-fader floor (dB). The BOTTOM of a volume fader means −∞/MUTE (the universal DAW convention:
 *  bounded upward, unbounded downward) — dbToLinear maps any value at/below this to gain 0, and the
 *  fader tooltip shows "-∞ dB" there. The STORED volumeDb stays at this finite floor (−Infinity does
 *  not survive JSON). ONE constant for the faders (TrackList), the gain math (playback), and the
 *  future mixdown export. */
export const FADER_MIN_DB = -24;
