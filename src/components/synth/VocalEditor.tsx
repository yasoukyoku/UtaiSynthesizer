// ② Vocal (piano-roll) editor — S48 Phase 4 (§9). Docked at the bottom (mirrors WorkflowEditor), the
// canvas is OFF-React (store.subscribe + rAF; drag paints from off-store refs and commits ONCE on mouseup
// via applyNoteEdits — §9.4), geometry is the single lib/vocalGeometry module (absolute tick space), and
// undo is Route-A timeline-native (vocal fields are already in meaningfulSig; the editor just claims the
// pane so Ctrl+Z/Delete route correctly). Tools: Arrow / Pen / Delete functional this phase; Pitch shows
// the baseline f0 line (full pitch editing = Phase 5). One-position-one-note truncation runs at commit.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import { useProjectStore } from "../../store/project";
import { useAppStore } from "../../store/app";
import { useAudioStore } from "../../store/audio";
import { useTranslation } from "react-i18next";
import { PIXELS_PER_TICK, TICKS_PER_BEAT } from "../../lib/constants";
import { msToTicks } from "../../lib/audio/laneOps";
import { TimeAxis, formatBarBeat } from "../../lib/timeAxis";
import * as playback from "../../lib/audio/playback";
import { resolveOverlaps, DEFAULT_TRANSITION, isBreathLyric } from "../../lib/vocalNotes";
import { DEFAULT_VOCAL_PARAMS } from "../../store/project";
import { useVoiceModelStore } from "../../store/voice-models";
import { renderVocalPart, vocalRenderErrorMessage } from "../../lib/vocal/vocalRender";
import { evalF0CentsAt, paintedDev, evalCurveAt } from "../../lib/f0eval";
import { PLAYHEAD } from "../../lib/canvasDraw";
import {
  type VocalView, V_PITCH_MIN, V_PITCH_MAX, V_ROW_H_MIN, V_ROW_H_MAX,
  tickToX, xToTick, noteTickToX, xToNoteTick, pitchToY, yToPitch, centsToY, yToCents,
  rowsContentHeight, snapFloor, snapRound, isBlackKey, pitchName, pitchToHz, centsToHz,
  paramToY, yToParam, LOUDNESS_DB_RANGE,
} from "../../lib/vocalGeometry";
import type { Note } from "../../types/project";
import { langById, DEFAULT_LANG_ID } from "../../lib/vocal/languages";
import { VocalSidebar } from "./VocalSidebar";
import "./VocalEditor.css";

type Tool = "arrow" | "pen" | "pitch" | "delete";

/** Grid / snap divisions per beat (§9.4 — all land on the constant 12/beat six-based grid; triplets 3/6). */
const GRID_DIVS = [
  { div: 1, key: "1/4" }, { div: 2, key: "1/8" }, { div: 4, key: "1/16" },
  { div: 3, key: "1/8T" }, { div: 6, key: "1/16T" }, { div: 12, key: "1/12" },
];
const KEY_COL_W = 56; // fixed piano-key column at the canvas left edge
const RULER_H = 18; // bar-number ruler strip along the top of the note area
const LANE_H = 88; // ② bottom automation-lane band height (only reserved when the lane is OPEN — §M-defer)
const EDGE_PX = 6; // note right-edge resize hotzone (screen px)

// ② The bottom automation lane shows ONE param at a time (SynthV-style selector). Keys MUST match the
// paramCurves keys the render feed reads (vocalRender.ts: "loudness" / "formant"). Ranges = the 稳健 defaults
// (user pick); values are RELATIVE offsets, neutral 0 = "no change" at the band midline.
type LaneParam = "loudness" | "formant";
const LANE_PARAMS: { id: LaneParam; min: number; max: number; unit: string; labelKey: string }[] = [
  // loudness range = the shared LOUDNESS_DB_RANGE (vocalGeometry) — the S59 audio-track loudness
  // band uses the SAME constant, so the two lanes' dB scales cannot drift apart.
  { id: "loudness", min: -LOUDNESS_DB_RANGE, max: LOUDNESS_DB_RANGE, unit: "dB", labelKey: "vocalEditor.lane.loudness" },
  { id: "formant", min: -12, max: 12, unit: "st", labelKey: "vocalEditor.lane.formant" },
];
const laneCfg = (p: LaneParam) => LANE_PARAMS.find((x) => x.id === p)!;
// S58: the default lyric for a newly drawn note follows the TRACK's language (a ja "あ" on a zh/en
// track would be instant OOV — audit MAJOR). langById falls back to ja for an out-of-range id.
const defaultLyricFor = (langId: number | undefined) => langById(langId ?? DEFAULT_LANG_ID).defaultLyric;
const MIN_LEN_TICKS = TICKS_PER_BEAT / 12; // shortest note the UI allows = 1/12 (40t), the finest grid — so you
// can always drag down to it WITHOUT switching grid; the 60ms singability floor is a Phase-6 render concern (§user)

type DragKind = "marquee" | "move" | "resize" | "create" | "marquee-delete" | "pitch-paint" | "param-point" | "ruler-seek";
interface DragState {
  kind: DragKind;
  clientX0: number; clientY0: number;
  curX: number; curY: number;
  activeIds: string[];
  orig: Map<string, Note>; // snapshot of the dragged notes
  newNote: Note | null; // create: the note being drawn
  anchorRelTick: number; // create/move reference
  moved: boolean;
  additive: boolean; // marquee: keep existing selection
  activeX?: boolean; // move: X/Y axes activate INDEPENDENTLY past a threshold, each measured from the ORIGIN,
  activeY?: boolean; // so switching axis mid-drag keeps the delta already applied — no jump-back (§user bug)
  startRel?: number; // move: CONTENT tick/pitch captured at pointerdown — dTick/dPitch measure from THESE
  startPitch?: number; // (not the screen origin), so scrolling mid-drag re-anchors to the cursor, no drift
  paint?: { xs: number[]; ys: number[] }; // pitch-paint: the drawn (relTick, absCents) path
  param?: LaneParam; // param-point: which lane is being edited
  pointCurve?: { xs: number[]; ys: number[] }; // param-point: the WORKING curve (points) being dragged
  pointIdx?: number; // param-point: index of the point under the cursor
  previewNotes?: () => Note[]; // off-ref draw source during the gesture (attached by withPreview)
}

interface Props {
  segmentId: string;
  onClose: () => void;
  style?: React.CSSProperties;
}

export function VocalEditor({ segmentId, onClose, style }: Props) {
  const { t } = useTranslation();
  const tracks = useProjectStore((s) => s.tracks);
  const tempo = useProjectStore((s) => s.tempo);
  const timeSignature = useProjectStore((s) => s.timeSignature);
  const selectedNotes = useProjectStore((s) => s.selectedNotes);
  const applyNoteEdits = useProjectStore((s) => s.applyNoteEdits);
  const selectNotes = useProjectStore((s) => s.selectNotes);
  const setSegmentPitchDev = useProjectStore((s) => s.setSegmentPitchDev);
  const setSegmentParamCurve = useProjectStore((s) => s.setSegmentParamCurve);
  const setActivePane = useAppStore((s) => s.setActivePane);

  // Resolve the notes part by the STABLE segment id (a track reorder / rename must not lose it).
  const part = useMemo(() => {
    for (const tr of tracks) {
      const sg = tr.segments.find((s) => s.id === segmentId);
      if (sg && sg.content.type === "notes") {
        return {
          trackId: tr.id, trackName: tr.name, seg: sg, notes: sg.content.notes, start: sg.startTick, dur: sg.durationTicks,
          pitchDev: sg.content.pitchDev, paramCurves: sg.content.paramCurves,
          transition: tr.vocalParams?.transition ?? DEFAULT_TRANSITION,
          vocalParams: tr.vocalParams ?? DEFAULT_VOCAL_PARAMS, voiceModel: tr.voiceModel,
        };
      }
    }
    return null;
  }, [tracks, segmentId]);

  const [tool, setTool] = useState<Tool>("arrow");
  const [gridDiv, setGridDiv] = useState(2); // 1/8 default
  const [laneOpen, setLaneOpen] = useState(false); // ② bottom automation lane — collapsed by default (§user)
  const [laneParam, setLaneParam] = useState<LaneParam>("loudness");
  const [maximized, setMaximized] = useState(false);
  const [playing, setPlaying] = useState(false);
  // GLOBAL render flag (one vocal render at a time) — reactive so every editor's button disables together.
  const vocalRenderActive = useAppStore((s) => s.vocalRenderActive);
  const [lyricEdit, setLyricEdit] = useState<{ id: string; x: number; y: number; w: number; value: string } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const drawRef = useRef(() => {});
  const dragRef = useRef<DragState | null>(null);
  const mouseRef = useRef<{ x: number; y: number } | null>(null);
  const clipboardRef = useRef<Note[]>([]);
  const lyricCancelRef = useRef(false); // Escape in the lyric input → suppress the unmount-blur commit
  const lyricNavRef = useRef(false); // Tab/Shift+Tab navigating between notes → suppress the old input's unmount-blur (it already committed)

  // Local view state (NOT the arrangement's global scroll/zoom, §9.4).
  const viewRef = useRef<VocalView>({ scrollX: 0, scrollY: 0, ppt: PIXELS_PER_TICK * 2, rowH: 16, top: RULER_H });
  const sizeRef = useRef({ w: 800, h: 300, dpr: 1 });

  // Content refs (synced each render) so the imperative draw/pointer code reads fresh values.
  const notesRef = useRef<Note[]>(part?.notes ?? []);
  notesRef.current = part?.notes ?? [];
  // Pitch-line inputs (the always-shown real f0 = evalF0Cents): the segment's hand-drawn deviation curve +
  // the track's default transition (per-note transitions live on each note).
  const pitchDevRef = useRef(part?.pitchDev);
  pitchDevRef.current = part?.pitchDev;
  // ② automation-lane curves (loudness/formant) — synced every render so the imperative draw/pointer read fresh.
  const paramCurvesRef = useRef(part?.paramCurves);
  paramCurvesRef.current = part?.paramCurves;
  const transitionRef = useRef(part?.transition ?? DEFAULT_TRANSITION);
  transitionRef.current = part?.transition ?? DEFAULT_TRANSITION;
  // M3 breath: the pitch line skips breath notes (unvoiced — they break the line so the prev note releases /
  // the next scoops, §10.5). Dynamic: editing the token re-connects the OLD token's notes.
  const breathTokenRef = useRef(part?.vocalParams?.breathToken ?? "AP");
  breathTokenRef.current = part?.vocalParams?.breathToken ?? "AP";
  // S58: the track's default lang drives the default lyric of newly drawn notes (must be singable).
  const defaultLyricRef = useRef(defaultLyricFor(part?.vocalParams?.langId));
  defaultLyricRef.current = defaultLyricFor(part?.vocalParams?.langId);
  // ② S58 OOV verdicts for THIS segment (async, from the oovWatch watcher) — ref-synced for the draw
  // closure; the dedicated redraw effect below re-invokes it when the verdict changes.
  const oovIds = useAppStore((s) => s.vocalOov[segmentId]);
  const oovRef = useRef<Set<string>>(new Set());
  oovRef.current = new Set(oovIds ?? []);
  const startRef = useRef(part?.start ?? 0);
  startRef.current = part?.start ?? 0;
  const durRef = useRef(part?.dur ?? 0);
  durRef.current = part?.dur ?? 0;

  // ② Render (Phase 6): build the score triples + Option-A f0 from the edited notes and invoke the Rust
  // score→singing render; the baked wav deposits as a processedOutputs overlay (plays back via the lane
  // path). The SVC voice/backend/transpose/speaker come from the track's vocalParams + voiceModel (sidebar).
  const render = useCallback(async () => {
    if (!part) return;
    // Resolve the live Track + Segment and delegate to the ONE shared render path (renderVocalPart) — the
    // SAME code the Play-time auto-render batch runs, so the manual button and auto-render can never drift.
    const track = useProjectStore.getState().tracks.find((tr) => tr.id === part.trackId);
    const seg = track?.segments.find((s) => s.id === segmentId);
    if (!track || !seg) return;
    try {
      await renderVocalPart(track, seg, tempoRef.current, t("vocalEditor.render.laneLabel"));
    } catch (e) {
      // Shared error→message mapping (vocalRenderErrorMessage) — the SAME one the Play-time auto-render
      // batch uses, so the two paths can never drift (§user: they must report identically).
      useAppStore.getState().showToast(vocalRenderErrorMessage(e), "error");
    }
  }, [part, segmentId, t]);

  // Load the installed voice models once when the editor opens — the sidebar's singer picker needs them
  // (otherwise the list stays empty until the Resource Manager is opened, which triggers the scan; §user).
  useEffect(() => {
    void useVoiceModelStore.getState().fetchModels();
  }, []);
  const selRef = useRef<Set<string>>(new Set(selectedNotes));
  selRef.current = new Set(selectedNotes);
  const toolRef = useRef(tool);
  toolRef.current = tool;
  const gridDivRef = useRef(gridDiv);
  gridDivRef.current = gridDiv;
  const laneOpenRef = useRef(laneOpen);
  laneOpenRef.current = laneOpen;
  const laneParamRef = useRef(laneParam);
  laneParamRef.current = laneParam;
  // ② global transport playhead (project store, ABSOLUTE tick) — read via ref inside the draw closure and
  // driven by a dedicated store.subscribe (below), NOT a reactive selector (that would re-render 60×/s).
  const playheadTickRef = useRef(useProjectStore.getState().playheadTick);
  const tempoRef = useRef(tempo);
  tempoRef.current = tempo;
  // Same meter authority as the arrangement — for the sub-beat grid (built from the global time signature).
  const timeAxis = useMemo(() => TimeAxis.global(timeSignature[0], timeSignature[1]), [timeSignature]);
  const axisRef = useRef(timeAxis);
  axisRef.current = timeAxis;

  const snapTicks = () => TICKS_PER_BEAT / gridDivRef.current; // 480/div: 1/4=480,1/8=240,1/16=120,1/8T=160,1/16T=80,1/12=40

  // ② lane geometry (live, from refs): when the bottom automation lane is OPEN it reserves LANE_H at the
  // canvas bottom, so the note-row area shrinks — EVERY visible-note-height / scroll clamp must subtract it
  // (else the lowest rows sit permanently behind the lane at max scroll). laneOpen=false ⇒ 0 = exact old behavior.
  const laneBandH = () => (laneOpenRef.current ? LANE_H : 0);
  const noteBottom = () => sizeRef.current.h - laneBandH(); // y where the note rows end (lane band below it)
  const visNoteH = () => Math.max(1, sizeRef.current.h - RULER_H - laneBandH()); // visible note-row height

  // rAF-coalesced redraw + preview-playback rAF (scrubs the segment following evalF0Cents so you can HEAR
  // the smooth transitions / vibrato — placeholder tone, not the SVC voice, §9.7).
  const rafRef = useRef(0);
  const playRafRef = useRef(0);
  const edgeRafRef = useRef(0); // marquee edge auto-scroll rAF
  const edgeScrollRef = useRef<() => void>(() => {});
  const requestRedraw = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => { rafRef.current = 0; drawRef.current(); });
  }, []);

  // ── preview playback: scrub the segment following the single evalF0Cents so you HEAR the smooth
  //    transitions / vibrato / drawn pitchDev (placeholder tone, not the SVC voice, §9.7). ──
  const stopPreviewPlay = useCallback(() => {
    if (playRafRef.current) { cancelAnimationFrame(playRafRef.current); playRafRef.current = 0; }
    playback.stopPreviewTone();
    setPlaying(false);
  }, []);
  const startPreviewPlay = useCallback(() => {
    if (!part) return;
    if (playRafRef.current) { cancelAnimationFrame(playRafRef.current); playRafRef.current = 0; } // clear any orphan
    const startMs = performance.now();
    playback.playPreviewTone(centsToHz(6000), 0); // seed a sustained tone; retuned each frame
    setPlaying(true);
    const tick = () => {
      const rel = msToTicks(performance.now() - startMs, tempoRef.current);
      if (rel > durRef.current) { stopPreviewPlay(); return; } // reached the segment end
      // skip breath notes — unvoiced, they break the pitch chain (the preview tone silences over them).
      const sorted = notesRef.current.filter((n) => !isBreathLyric(n.lyric, breathTokenRef.current)).sort((a, b) => a.tick - b.tick);
      // Build opts PER-FRAME from the live refs so a sidebar edit to the TRACK-DEFAULT transition (or tempo)
      // retunes the RUNNING preview immediately — else only the overlay updates and the audio stays stale
      // until stop+replay (review finding E).
      const opts = { tempo: tempoRef.current, defaultTransition: transitionRef.current };
      const r = evalF0CentsAt(sorted, pitchDevRef.current, rel, opts);
      playback.setPreviewToneHz(r.voiced ? centsToHz(r.cents) : 20); // rest → near-silent low freq
      playRafRef.current = requestAnimationFrame(tick);
    };
    playRafRef.current = requestAnimationFrame(tick);
  }, [part, stopPreviewPlay]);

  // marquee edge auto-scroll (§9.4): while the cursor is held at a border during a marquee, scroll the view
  // and PIN the box's anchor to the content (its screen origin shifts opposite the scroll) so the box grows
  // over the newly-revealed area. Reassigned each render so it reads fresh refs; recurses via the ref.
  edgeScrollRef.current = () => {
    const d = dragRef.current, m = mouseRef.current;
    if (!d || (d.kind !== "marquee" && d.kind !== "marquee-delete") || !m) { edgeRafRef.current = 0; return; }
    const { w } = sizeRef.current, v = viewRef.current;
    const EDGE = 28, SPEED = 10;
    let dx = 0, dy = 0;
    if (m.x < KEY_COL_W + EDGE) dx = -SPEED; else if (m.x > w - EDGE) dx = SPEED;
    if (m.y < RULER_H + EDGE) dy = -SPEED; else if (m.y > noteBottom() - EDGE) dy = SPEED; // note-area bottom (above the lane)
    if (!dx && !dy) { edgeRafRef.current = 0; return; } // cursor left the border → stop
    const maxSX = Math.max(0, (startRef.current + durRef.current) * v.ppt + 400 - Math.max(1, w - KEY_COL_W));
    const maxSY = Math.max(0, rowsContentHeight(v.rowH) - visNoteH());
    const nSX = Math.max(0, Math.min(maxSX, v.scrollX + dx)), nSY = Math.max(0, Math.min(maxSY, v.scrollY + dy));
    d.clientX0 -= nSX - v.scrollX; d.clientY0 -= nSY - v.scrollY; // pin the anchor to content
    v.scrollX = nSX; v.scrollY = nSY;
    requestRedraw();
    edgeRafRef.current = requestAnimationFrame(edgeScrollRef.current);
  };

  // ── size / DPR ──
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const measure = () => {
      const r = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      sizeRef.current = { w: Math.max(1, r.width), h: Math.max(1, r.height), dpr };
      const cv = canvasRef.current;
      if (cv) {
        cv.width = Math.round(sizeRef.current.w * dpr);
        cv.height = Math.round(sizeRef.current.h * dpr);
        cv.style.width = `${sizeRef.current.w}px`;
        cv.style.height = `${sizeRef.current.h}px`;
      }
      requestRedraw();
    };
    measure();
    const ob = new ResizeObserver(measure);
    ob.observe(wrap);
    return () => ob.disconnect();
  }, [requestRedraw]);

  // Center the vertical view on the notes' pitch range (or C4) once, on first mount for this segment.
  useEffect(() => {
    const ns = notesRef.current;
    const avg = ns.length > 0 ? ns.reduce((a, n) => a + n.pitch, 0) / ns.length : 60;
    const v = viewRef.current;
    const visH = visNoteH(); // visible note-row height (below the ruler, above the lane band if open)
    // center avg at the MIDDLE of the visible note area [RULER_H, noteBottom()] — not the full canvas, else
    // an open lane pushes the average pitch ~LANE_H/2 too low (into/behind the band). noteBottom()==h when closed.
    v.scrollY = Math.max(0, Math.min(Math.max(0, rowsContentHeight(v.rowH) - visH), pitchToY(avg, { ...v, scrollY: 0 }) - (RULER_H + noteBottom()) / 2));
    v.scrollX = Math.max(0, startRef.current * v.ppt - 24);
    requestRedraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segmentId]);

  // ② Re-clamp scrollY when the lane opens/closes. Toggling the lane reserves/frees LANE_H, changing the
  // visible note-row height, so a scroll position valid for the old height can overshoot the new max — leaving
  // the lowest rows behind the band (open) or a blank strip below the last row (close) until the next wheel
  // event. Clamp to the new max here (band from the `laneOpen` STATE, not the ref, so timing is unambiguous).
  useEffect(() => {
    const v = viewRef.current;
    const vis = Math.max(1, sizeRef.current.h - RULER_H - (laneOpen ? LANE_H : 0));
    const max = Math.max(0, rowsContentHeight(v.rowH) - vis);
    if (v.scrollY > max) { v.scrollY = max; requestRedraw(); }
  }, [laneOpen, requestRedraw]);

  // ② Drive a redraw when the global transport playhead moves (playback / seek), OFF-React (a reactive
  // selector would re-render the whole editor 60×/s). Mirrors TimelineRuler's imperative subscribe — the
  // draw closure reads playheadTickRef.current, so it just needs re-invoking each time the tick changes.
  useEffect(() => {
    const unsub = useProjectStore.subscribe((s) => {
      if (s.playheadTick !== playheadTickRef.current) {
        playheadTickRef.current = s.playheadTick;
        requestRedraw();
      }
    });
    return unsub;
  }, [requestRedraw]);

  // ── the draw closure (rebuilt when content/selection/tool change) ──
  useEffect(() => {
    drawRef.current = () => {
      const cv = canvasRef.current;
      if (!cv) return;
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      const { w, h, dpr } = sizeRef.current;
      const v = viewRef.current;
      const notes = notesRef.current;
      const start = startRef.current;
      const sel = selRef.current;
      const css = getComputedStyle(document.documentElement);
      const col = (n: string) => css.getPropertyValue(n).trim();

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      // background
      ctx.fillStyle = col("--bg-base") || "#0d1220";
      ctx.fillRect(0, 0, w, h);

      const noteAreaX = KEY_COL_W;
      const noteAreaW = w - KEY_COL_W;

      // ── pitch-class row striping + per-semitone row lines ──
      const topPitch = yToPitch(0, v);
      const botPitch = yToPitch(h, v);
      for (let p = botPitch; p <= topPitch; p++) {
        const y = pitchToY(p, v);
        // black-key rows DARKER than white-key rows (visual bands). ⚠ theme's --piano-black is LIGHTER than
        // --bg-base, so tinting with it inverts the shading — use an explicit dark overlay instead.
        if (isBlackKey(p)) {
          ctx.fillStyle = "#000000"; ctx.globalAlpha = 0.22;
          ctx.fillRect(noteAreaX, y, noteAreaW, v.rowH);
          ctx.globalAlpha = 1;
        }
        // a faint line at EVERY semitone boundary so adjacent white keys (E|F, B|C) read as separate rows,
        ctx.strokeStyle = "rgba(130,150,185,0.13)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(noteAreaX, Math.round(y) + 0.5); ctx.lineTo(w, Math.round(y) + 0.5); ctx.stroke();
        // and a STRONGER line at the bottom of each C (octave separation) — white, like the meter lines.
        if (p % 12 === 0) {
          ctx.strokeStyle = "rgba(226,232,244,0.2)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(noteAreaX, Math.round(y + v.rowH) + 0.5); ctx.lineTo(w, Math.round(y + v.rowH) + 0.5); ctx.stroke();
        }
      }

      // ── vertical grid (bar / beat / 1-6 / 1-12) via the single TimeAxis sub-grid (triplet-capable). A
      //    CLEAR prominence hierarchy so the meter reads at a glance — bar ≫ beat ≫ 1/6 ≫ 1/12 (the alphas
      //    were too close before → the sub-grid competed with bar/beat and it all read as clutter). ──
      const absFrom = xToTick(0, v);
      const absTo = xToTick(noteAreaW, v);
      for (const g of axisRef.current.subGridLinesInRange(Math.max(0, Math.floor(absFrom)), Math.ceil(absTo), 12)) {
        const x = noteAreaX + tickToX(g.tick, v);
        if (x < noteAreaX - 1) continue;
        // WHITE/gray meter lines (cyan is reserved for the "current position" guide, so they don't clash).
        let a: number, lw: number;
        if (g.level === "bar") { a = 0.4; lw = 1.6; }        // bar = STRONGEST
        else if (g.level === "beat") { a = 0.19; lw = 1.1; } // beat = clearly second
        else if (g.sub % 2 === 0) { a = 0.06; lw = 1; }      // 1/6-beat = faint
        else { a = 0.03; lw = 1; }                           // 1/12-beat = barely there
        ctx.strokeStyle = `rgba(226,232,244,${a})`; // = --text-primary-ish white
        ctx.lineWidth = lw;
        ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, h); ctx.stroke();
      }

      // ── out-of-part dimming: anything outside [partStart, partStart+dur] is NOT part of THIS segment and
      //    very likely won't be rendered — dim it so notes drawn there read as "outside the part" (§ user). ──
      const partX0 = noteAreaX + tickToX(start, v);
      const partX1 = noteAreaX + tickToX(start + durRef.current, v);
      ctx.fillStyle = col("--bg-deep") || "#080b14";
      ctx.globalAlpha = 0.55;
      if (partX0 > noteAreaX) ctx.fillRect(noteAreaX, 0, partX0 - noteAreaX, h);
      if (partX1 < w) { const rx = Math.max(noteAreaX, partX1); ctx.fillRect(rx, 0, w - rx, h); }
      ctx.globalAlpha = 1;

      // ── part bounds (the editable window) — crisp accent lines on top of the dimming ──
      ctx.strokeStyle = col("--accent-secondary") || "#8b5cf6";
      ctx.globalAlpha = 0.7; ctx.lineWidth = 1.5;
      for (const px of [partX0, partX1]) {
        if (px >= noteAreaX && px <= w) { ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, 0); ctx.lineTo(Math.round(px) + 0.5, h); ctx.stroke(); }
      }
      ctx.globalAlpha = 1;

      // ── notes (from off-ref drag preview if a gesture is live, else the store) ──
      const drawNotes = dragRef.current?.previewNotes?.() ?? notes;
      ctx.font = `${Math.min(13, Math.max(9, v.rowH - 4))}px system-ui, sans-serif`;
      ctx.textBaseline = "middle";
      for (const n of drawNotes) {
        const x0 = noteAreaX + noteTickToX(n.tick, start, v);
        const x1 = noteAreaX + noteTickToX(n.tick + n.duration, start, v);
        if (x1 < noteAreaX || x0 > w) continue;
        const y = pitchToY(n.pitch, v);
        const selected = sel.has(n.id);
        // ② S58 OOV: an unsingable lyric fills RED (LOUD — never silent, §3.7 ACE-style marking). A
        // selected OOV note keeps the selection fill but takes the red stroke, so both states read.
        const oov = oovRef.current.has(n.id);
        const cx0 = Math.max(noteAreaX, x0);
        ctx.fillStyle = selected
          ? (col("--note-selected") || "#8b5cf6")
          : oov
            ? (col("--color-error") || "#f87171")
            : (col("--note-fill") || "#39c5bb");
        ctx.globalAlpha = 0.9;
        ctx.fillRect(cx0, y + 1, Math.max(2, x1 - cx0), Math.max(2, v.rowH - 2));
        ctx.globalAlpha = 1;
        ctx.strokeStyle = oov
          ? (col("--color-error") || "#f87171")
          : selected
            ? (col("--accent-secondary") || "#8b5cf6")
            : "rgba(0,0,0,0.4)";
        ctx.lineWidth = selected ? 1.5 : 1;
        ctx.strokeRect(Math.round(cx0) + 0.5, Math.round(y + 1) + 0.5, Math.max(2, x1 - cx0) - 1, Math.max(2, v.rowH - 2) - 1);
        // lyric
        if (x1 - cx0 > 14 && v.rowH >= 11) {
          ctx.fillStyle = "#0a0f18";
          ctx.save();
          ctx.beginPath(); ctx.rect(cx0 + 2, y, x1 - cx0 - 3, v.rowH); ctx.clip();
          ctx.fillText(n.lyric, cx0 + 4, y + v.rowH / 2 + 0.5);
          ctx.restore();
        }
      }

      // ── PITCH LINE: the always-shown REAL f0 (SynthV — the sounding pitch = base ⊕ transition + pitchDev
      //    + vibrato, all summed by the single evalF0Cents). Sampled across the note area every few px; an
      //    unvoiced (rest) sample BREAKS the line (carrier-aware, §9.3). Emphasized under the Pitch tool. ──
      {
        // the pitch LINE skips breath notes (unvoiced) — the line breaks over a breath, and its neighbours
        // become phrase edges (prev note release-drift 段中尾音, next note onset-scoop). The note RECTANGLES
        // above (drawNotes) still show the breath note; only the sung-pitch chain drops it.
        const sorted = drawNotes.filter((n) => !isBreathLyric(n.lyric, breathTokenRef.current)).sort((a, b) => a.tick - b.tick);
        const opts = { tempo: tempoRef.current, defaultTransition: transitionRef.current };
        const dr0 = dragRef.current; // live pitch-paint → preview the drawn curve on the line
        const dev = dr0?.kind === "pitch-paint" && dr0.paint ? paintedDev(sorted, dr0.paint, pitchDevRef.current, opts) : pitchDevRef.current;
        ctx.strokeStyle = col("--accent-tertiary") || "#ff6b9d";
        ctx.lineWidth = toolRef.current === "pitch" ? 2 : 1.5;
        ctx.globalAlpha = toolRef.current === "pitch" ? 0.95 : 0.7;
        ctx.beginPath();
        let pen = false; // whether a voiced sub-path is open
        for (let px = noteAreaX; px <= w; px += 2) {
          const rel = xToNoteTick(px - KEY_COL_W, start, v);
          const r = evalF0CentsAt(sorted, dev, rel, opts);
          if (!r.voiced) { pen = false; continue; } // rest → break the line
          const y = centsToY(r.cents, v);
          if (!pen) { ctx.moveTo(px, y); pen = true; } else ctx.lineTo(px, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
      }

      // ── live gesture overlays ──
      const d = dragRef.current;
      if (d && (d.kind === "marquee" || d.kind === "marquee-delete")) {
        const x = Math.min(d.clientX0, d.curX), yy = Math.min(d.clientY0, d.curY);
        const ww = Math.abs(d.curX - d.clientX0), hh = Math.abs(d.curY - d.clientY0);
        const del = d.kind === "marquee-delete";
        ctx.fillStyle = del ? (col("--color-error") || "#f87171") : (col("--accent-primary") || "#39c5bb");
        ctx.globalAlpha = 0.12; ctx.fillRect(x, yy, ww, hh);
        ctx.globalAlpha = 0.8; ctx.strokeStyle = del ? (col("--color-error") || "#f87171") : (col("--accent-primary") || "#39c5bb");
        ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.strokeRect(x + 0.5, yy + 0.5, ww, hh); ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // ── piano key column (fixed left) — real-keyboard look: light WHITE keys full width, dark BLACK keys
      //    inset from the right (so they read as shorter, between the whites), thin white-key separators. ──
      // The whole column is WHITE-key surface first; on a real keyboard the black keys are SHORT and sit at
      // the back, with the white keys extending past them to the FRONT. Here the front = the right side, so
      // black keys are dark rects inset from the right and the WHITE shows to their right (that white — not
      // a dark gap — is what makes the inset read).
      ctx.fillStyle = col("--piano-white") || "#c8d0e0";
      ctx.fillRect(0, 0, KEY_COL_W, h);
      const blackW = Math.round(KEY_COL_W * 0.6);
      // black keys: dark, short (inset from the right) → white front stays visible to their right
      for (let p = botPitch; p <= topPitch; p++) {
        if (!isBlackKey(p)) continue;
        const y = pitchToY(p, v);
        ctx.fillStyle = "#10192c";
        ctx.fillRect(0, Math.round(y), blackW, Math.max(1, Math.round(v.rowH)));
      }
      // thin separator at EVERY row boundary so each key (white or black) is framed
      ctx.strokeStyle = "rgba(10,15,24,0.4)"; ctx.lineWidth = 1;
      for (let p = botPitch; p <= topPitch; p++) {
        const y = Math.round(pitchToY(p, v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(KEY_COL_W, y); ctx.stroke();
      }
      // C labels (dark text on the white front, right side — always visible even on a black-key neighbour)
      if (v.rowH >= 10) {
        ctx.font = "10px system-ui, sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
        ctx.fillStyle = "#0a0f18";
        for (let p = botPitch; p <= topPitch; p++) {
          if (p % 12 !== 0) continue;
          ctx.fillText(pitchName(p), KEY_COL_W - 20, pitchToY(p, v) + v.rowH / 2 + 0.5);
        }
      }
      ctx.strokeStyle = col("--border-default") || "#2a3a5c";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(KEY_COL_W + 0.5, 0); ctx.lineTo(KEY_COL_W + 0.5, h); ctx.stroke();

      // ── bar-number ruler (top strip) — always know which bar you're in (mirrors the arrangement ruler) ──
      ctx.fillStyle = col("--bg-panel") || "#1a2236";
      ctx.fillRect(0, 0, w, RULER_H);
      ctx.strokeStyle = col("--border-default") || "#2a3a5c";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, RULER_H + 0.5); ctx.lineTo(w, RULER_H + 0.5); ctx.stroke();
      ctx.font = "10px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      for (const g of axisRef.current.gridLinesInRange(Math.max(0, Math.floor(absFrom)), Math.ceil(absTo))) {
        const gx = noteAreaX + tickToX(g.tick, v);
        if (gx < noteAreaX) continue;
        ctx.strokeStyle = g.isBar ? "rgba(226,232,244,0.5)" : "rgba(226,232,244,0.26)"; // white, match the note-grid
        ctx.lineWidth = g.isBar ? 1.4 : 1;
        ctx.beginPath(); ctx.moveTo(Math.round(gx) + 0.5, g.isBar ? 3 : RULER_H - 5); ctx.lineTo(Math.round(gx) + 0.5, RULER_H); ctx.stroke();
        if (g.isBar) {
          ctx.fillStyle = col("--text-muted") || "#556b94";
          ctx.fillText(String(axisRef.current.tickToBarBeat(g.tick).bar), gx + 3, RULER_H / 2);
        }
      }

      // ── ② bottom automation lane (loudness / formant) — a FIXED band drawn OVER the bottom LANE_H of the
      //    note area (its opaque bg covers the note rows/pitch-line that painted into this region). Shows ONE
      //    param at a time (selector in the header); the curve is a RELATIVE offset, neutral 0 at the midline. ──
      if (laneOpenRef.current) {
        const cfg = laneCfg(laneParamRef.current);
        const laneTop = h - LANE_H;
        ctx.fillStyle = col("--bg-panel") || "#1a2236";
        ctx.fillRect(0, laneTop, w, LANE_H);
        ctx.strokeStyle = col("--border-default") || "#2a3a5c"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, laneTop + 0.5); ctx.lineTo(w, laneTop + 0.5); ctx.stroke();
        // vertical meter grid within the band (time alignment with the notes above)
        for (const g of axisRef.current.subGridLinesInRange(Math.max(0, Math.floor(absFrom)), Math.ceil(absTo), 12)) {
          const gx = noteAreaX + tickToX(g.tick, v);
          if (gx < noteAreaX) continue;
          const a = g.level === "bar" ? 0.28 : g.level === "beat" ? 0.13 : g.sub % 2 === 0 ? 0.05 : 0.025;
          ctx.strokeStyle = `rgba(226,232,244,${a})`; ctx.lineWidth = g.level === "bar" ? 1.4 : 1;
          ctx.beginPath(); ctx.moveTo(Math.round(gx) + 0.5, laneTop + 1); ctx.lineTo(Math.round(gx) + 0.5, h); ctx.stroke();
        }
        // neutral (0) midline
        const midY = paramToY(0, cfg.min, cfg.max, laneTop, LANE_H);
        ctx.strokeStyle = "rgba(226,232,244,0.22)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(noteAreaX, Math.round(midY) + 0.5); ctx.lineTo(w, Math.round(midY) + 0.5); ctx.stroke(); ctx.setLineDash([]);
        // the param ENVELOPE = a piecewise-linear curve through the user's control POINTS (§user: insert +
        // drag points, not freehand). Live = the working curve during a point drag, else the stored curve.
        const stored = paramCurvesRef.current?.[cfg.id];
        const dr0 = dragRef.current;
        const live = dr0?.kind === "param-point" && dr0.pointCurve && dr0.param === cfg.id ? dr0.pointCurve : stored;
        ctx.strokeStyle = col("--accent-primary") || "#39c5bb"; ctx.lineWidth = 1.8; ctx.globalAlpha = 0.95;
        ctx.beginPath();
        for (let px = noteAreaX; px <= w; px += 2) {
          const rel = xToNoteTick(px - KEY_COL_W, start, v);
          const y = paramToY(evalCurveAt(live, rel), cfg.min, cfg.max, laneTop, LANE_H);
          if (px === noteAreaX) ctx.moveTo(px, y); else ctx.lineTo(px, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
        // control-point HANDLES (grabbable squares) — the dragged one highlighted.
        if (live && live.xs.length) {
          for (let i = 0; i < live.xs.length; i++) {
            const hx = noteAreaX + noteTickToX(live.xs[i]!, start, v);
            if (hx < noteAreaX - 4 || hx > w + 4) continue;
            const hy = paramToY(live.ys[i]!, cfg.min, cfg.max, laneTop, LANE_H);
            const activePt = dr0?.kind === "param-point" && dr0.pointIdx === i && dr0.param === cfg.id;
            ctx.fillStyle = activePt ? (col("--note-selected") || "#8b5cf6") : (col("--accent-primary") || "#39c5bb");
            ctx.fillRect(Math.round(hx) - 3, Math.round(hy) - 3, 6, 6);
          }
        }
        // left scale column (over the key-column width): +max / 0 / min + unit — numbers + symbol, no i18n.
        ctx.fillStyle = col("--bg-deep") || "#080b14"; ctx.fillRect(0, laneTop, KEY_COL_W, LANE_H);
        ctx.strokeStyle = col("--border-default") || "#2a3a5c";
        ctx.beginPath(); ctx.moveTo(KEY_COL_W + 0.5, laneTop); ctx.lineTo(KEY_COL_W + 0.5, h); ctx.stroke();
        ctx.fillStyle = col("--text-muted") || "#556b94"; ctx.font = "9px system-ui, sans-serif";
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText(`+${cfg.max}`, KEY_COL_W - 4, laneTop + 9);
        ctx.fillText("0", KEY_COL_W - 4, midY);
        ctx.fillText(`${cfg.min}`, KEY_COL_W - 4, h - 9);
        ctx.textAlign = "left"; ctx.textBaseline = "top";
        ctx.fillText(cfg.unit, 4, laneTop + 3);
      }

      // ── mouse-position guide (ALL tools): a vertical line at the snapped tick under the cursor, drawn
      //    LAST (over the key column + ruler) so it is NEVER covered — fixes "disappears at the leftmost".
      const mp = mouseRef.current;
      if (mp && !dragRef.current && mp.x >= KEY_COL_W && mp.y >= RULER_H) {
        const gx = noteAreaX + noteTickToX(snapFloor(xToNoteTick(mp.x - KEY_COL_W, start, v), snapTicks()), start, v);
        if (gx <= w) {
          const gxr = Math.max(KEY_COL_W, Math.round(gx)) + 0.5;
          ctx.strokeStyle = col("--accent-primary") || "#39c5bb";
          ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(gxr, RULER_H); ctx.lineTo(gxr, h); ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      // ── ② pink transport playhead (absolute tick → note-area x), drawn LAST so it's never covered; spans
      //    the note rows AND the lane band. Solid #ff6b9d (PLAYHEAD, shared) + a triangle cap in the ruler
      //    disambiguates it from the same-pink f0 line. Its bar:beat readout sits top-right of the ruler. ──
      {
        const phx = noteAreaX + tickToX(playheadTickRef.current, v);
        if (phx >= noteAreaX && phx <= w) {
          const xr = Math.round(phx) + 0.5;
          ctx.strokeStyle = PLAYHEAD; ctx.lineWidth = 2; ctx.globalAlpha = 1;
          ctx.beginPath(); ctx.moveTo(xr, RULER_H); ctx.lineTo(xr, h); ctx.stroke();
          ctx.fillStyle = PLAYHEAD;
          ctx.beginPath(); ctx.moveTo(phx - 4, RULER_H - 7); ctx.lineTo(phx + 4, RULER_H - 7); ctx.lineTo(phx, RULER_H); ctx.closePath(); ctx.fill();
        }
        // bar:beat:sub readout (transport position) — top-right of the ruler, reuses formatBarBeat (no drift).
        const txt = formatBarBeat(axisRef.current, Math.max(0, playheadTickRef.current));
        ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
        const tw = ctx.measureText(txt).width;
        ctx.fillStyle = col("--bg-panel") || "#1a2236"; ctx.fillRect(w - tw - 10, 0, tw + 10, RULER_H);
        ctx.fillStyle = PLAYHEAD; ctx.fillText(txt, w - 5, RULER_H / 2);
      }

      ctx.restore();
    };
    requestRedraw();
  }, [part?.notes, selectedNotes, tool, gridDiv, laneOpen, laneParam, requestRedraw]);

  // Warm the shared AudioContext on mount so the first key-preview isn't delayed by an on-gesture resume;
  // and make focus-loss / unmount RELEASE a sustained preview tone (belt-and-suspenders with pointerup/
  // cancel/Esc): if the window loses focus or is hidden mid-drag, the terminal pointerup may never arrive
  // (release over another window), so cancel the gesture + stop the tone here (verify: stuck-tone race).
  useEffect(() => {
    playback.getPreviewContext();
    const release = () => {
      if (dragRef.current) { dragRef.current = null; requestRedraw(); }
      if (edgeRafRef.current) { cancelAnimationFrame(edgeRafRef.current); edgeRafRef.current = 0; }
      if (playRafRef.current) { cancelAnimationFrame(playRafRef.current); playRafRef.current = 0; setPlaying(false); }
      playback.stopPreviewTone();
    };
    window.addEventListener("blur", release);
    document.addEventListener("visibilitychange", release);
    return () => {
      window.removeEventListener("blur", release);
      document.removeEventListener("visibilitychange", release);
      if (playRafRef.current) cancelAnimationFrame(playRafRef.current);
      if (edgeRafRef.current) cancelAnimationFrame(edgeRafRef.current);
      playback.stopPreviewTone();
    };
  }, [requestRedraw]);

  // Redraw when the part boundary (moved / resized on the ARRANGEMENT) or the meter changes. The boundary
  // edit keeps the notes array ref (only startTick/durationTicks move), so the draw effect above does NOT
  // re-run — without this the boundary line only "happens" to refresh when some other redraw fires (hover/
  // scroll), which is the reported intermittent staleness. The draw closure reads start/dur/axis via refs,
  // so re-invoking it is enough (no rebuild).
  useEffect(() => { requestRedraw(); }, [part?.start, part?.dur, part?.pitchDev, part?.paramCurves, part?.transition, part?.vocalParams?.breathToken, timeSignature, tempo, requestRedraw]);

  // ② S58 OOV marking: async verdicts from the oovWatch watcher (app store) → ref + redraw (the draw
  // closure reads the ref — the standard三处同步: ref sync here + this dedicated redraw effect).
  useEffect(() => { requestRedraw(); }, [oovIds, requestRedraw]);

  // Attach the live preview-notes closure to the drag state (draw reads it).
  const withPreview = (d: DragState): DragState & { previewNotes: () => Note[] } => {
    const dd = d as DragState & { previewNotes: () => Note[] };
    dd.previewNotes = () => computePreview(dd);
    return dd;
  };

  // ── geometry helpers on the live view ──
  const relTickAt = (clientX: number) => {
    const cv = canvasRef.current; if (!cv) return 0;
    const r = cv.getBoundingClientRect();
    return xToNoteTick(clientX - r.left - KEY_COL_W, startRef.current, viewRef.current);
  };
  const pitchAt = (clientY: number) => {
    const cv = canvasRef.current; if (!cv) return 60;
    const r = cv.getBoundingClientRect();
    return yToPitch(clientY - r.top, viewRef.current);
  };
  const centsAt = (clientY: number) => {
    const cv = canvasRef.current; if (!cv) return 6000;
    const r = cv.getBoundingClientRect();
    return yToCents(clientY - r.top, viewRef.current);
  };
  const localXY = (clientX: number, clientY: number) => {
    const cv = canvasRef.current; if (!cv) return { x: 0, y: 0 };
    const r = cv.getBoundingClientRect();
    return { x: clientX - r.left, y: clientY - r.top };
  };

  // note under a point (reverse z so the topmost/last wins); returns {note, onEdge}
  const noteAt = (clientX: number, clientY: number): { note: Note; onEdge: boolean } | null => {
    const { x } = localXY(clientX, clientY);
    if (x < KEY_COL_W) return null;
    const p = pitchAt(clientY);
    const rel = relTickAt(clientX);
    const v = viewRef.current;
    for (let i = notesRef.current.length - 1; i >= 0; i--) {
      const n = notesRef.current[i]!;
      if (n.pitch !== p) continue;
      if (rel >= n.tick && rel < n.tick + n.duration) {
        const x1 = noteTickToX(n.tick + n.duration, startRef.current, v) + KEY_COL_W;
        const onEdge = (x1 - x) <= EDGE_PX && n.duration * v.ppt > EDGE_PX * 2;
        return { note: n, onEdge };
      }
    }
    return null;
  };

  // ② index of the lane control-point under the cursor (within LANE_PT_HIT px), or -1. Uses the CURRENTLY
  // selected lane's stored curve; the caller has already confirmed the cursor is in the lane band.
  const LANE_PT_HIT = 8;
  const laneParamPointAt = (clientX: number, clientY: number): number => {
    const cfg = laneCfg(laneParamRef.current);
    const curve = paramCurvesRef.current?.[cfg.id];
    if (!curve || curve.xs.length === 0) return -1;
    const { x: cx, y: cy } = localXY(clientX, clientY);
    const v = viewRef.current, laneTop = noteBottom();
    let best = -1, bestD = LANE_PT_HIT;
    for (let i = 0; i < curve.xs.length; i++) {
      const px = KEY_COL_W + noteTickToX(curve.xs[i]!, startRef.current, v);
      const py = paramToY(curve.ys[i]!, cfg.min, cfg.max, laneTop, LANE_H);
      const d = Math.hypot(px - cx, py - cy);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  };

  // ── compute the off-ref preview note array for a live gesture (no store writes) ──
  const computePreview = (d: DragState): Note[] => {
    const base = notesRef.current;
    if (d.kind === "create" && d.newNote) {
      return [...base, d.newNote];
    }
    if (d.kind === "resize") {
      return base.map((n) => {
        const o = d.orig.get(n.id);
        if (!o) return n;
        const newEnd = Math.max(o.tick + MIN_LEN_TICKS, snapRound(relTickAt(d.curX), MIN_LEN_TICKS)); // step by 1/12 (40t)
        return { ...o, duration: newEnd - o.tick };
      });
    }
    if (d.kind === "move") {
      const origs = [...d.orig.values()];
      // AXIS LOCK: a mostly-vertical drag is PURE pitch (dTick=0); a mostly-horizontal drag is PURE timing
      // (dPitch=0). Fixes "multi-note vertical drag drifts sideways" — a tiny x-jitter used to jump a whole
      // snap cell (felt like huge sideways sensitivity), amplified across a multi-selection. Free move (§user).
      // Adjustment drags step by 1/12 (40t), NOT the (possibly coarse) grid cell (§user: only CREATION uses
      // the grid). Each axis is INDEPENDENT (activeX/activeY set in onPointerMove) and measured from the
      // origin — so switching direction mid-drag keeps what the other axis already moved (no jump-back).
      const rawDTick = d.activeX ? snapRound(relTickAt(d.curX), MIN_LEN_TICKS) - snapRound(d.startRel ?? 0, MIN_LEN_TICKS) : 0;
      const rawDPitch = d.activeY ? pitchAt(d.curY) - (d.startPitch ?? 0) : 0;
      // GROUP clamp (§9.2): clamp the SHARED delta by the group's headroom so spacing is preserved and two
      // notes can't collapse onto one tick/pitch at a wall. Translate whole notes — transition/vibrato are
      // in ABSOLUTE ms so they ride along unchanged; NO retimeNote rebase (that's only the truncation head-move).
      const dTick = Math.max(rawDTick, -Math.min(...origs.map((o) => o.tick)));
      const loP = Math.min(...origs.map((o) => o.pitch));
      const hiP = Math.max(...origs.map((o) => o.pitch));
      const dPitch = Math.max(V_PITCH_MIN - loP, Math.min(V_PITCH_MAX - hiP, rawDPitch));
      return base.map((n) => {
        const o = d.orig.get(n.id);
        return o ? { ...o, tick: o.tick + dTick, pitch: o.pitch + dPitch } : n;
      });
    }
    // pitch-paint doesn't touch notes — the drawn pitchDev is previewed via paintedDev() in draw/commit.
    return base;
  };

  const clampPitch = (p: number) => Math.min(V_PITCH_MAX, Math.max(V_PITCH_MIN, p));

  // ── commit: diff the resolved next-array vs current → applyNoteEdits (ONE step); skip a no-op ──
  const commitNotes = (nextNotes: Note[], activeIds: string[]) => {
    if (!part) return;
    // min 1 tick = zero-length guard only. The 60ms singability floor is a Phase-6 RENDER concern (Rust
    // min_frames), NOT a UI clamp — clamping here to 60ms would conflict with a 1/12 grid cell (§user).
    const resolved = resolveOverlaps(nextNotes, new Set(activeIds), 1);
    const cur = new Map(part.notes.map((n) => [n.id, n]));
    const next = new Map(resolved.map((n) => [n.id, n]));
    const add: Note[] = [];
    const update: Record<string, Partial<Note>> = {};
    const remove: string[] = [];
    for (const [id, n] of next) {
      const c = cur.get(id);
      if (!c) add.push(n);
      else if (noteSig(c) !== noteSig(n)) update[id] = n;
    }
    for (const id of cur.keys()) if (!next.has(id)) remove.push(id);
    if (add.length === 0 && remove.length === 0 && Object.keys(update).length === 0) return; // no-op → no dirty
    applyNoteEdits(part.trackId, segmentId, { add, update, remove });
  };

  // ── pointer handlers ──
  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;
    setActivePane("vocal");
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    const { x, y } = localXY(e.clientX, e.clientY);
    if (x < KEY_COL_W) { // left column: a piano key (note area) → preview tone; the lane's scale column → INERT
      if (y < noteBottom()) playback.playPreviewTone(pitchToHz(pitchAt(e.clientY)), 220);
      return; // never fall through to the note-area marquee (which would clear the selection)
    }
    // ② TOP RULER → seek the global transport playhead (§user: re-listen after an edit WITHOUT going to the main
    // arrangement to find the spot). Drag scrubs; during playback set `seeking` so the transport reschedules from
    // the new tick on release (mirrors TimelineRuler). Absolute tick space (playhead is absolute; x already ≥ KEY_COL_W).
    if (y < RULER_H) {
      useProjectStore.getState().setPlayhead(Math.max(0, Math.round(xToTick(x - KEY_COL_W, viewRef.current))));
      if (useAudioStore.getState().isPlaying) useAudioStore.getState().setSeeking(true);
      dragRef.current = withPreview({
        kind: "ruler-seek", clientX0: e.clientX, clientY0: e.clientY, curX: e.clientX, curY: e.clientY,
        activeIds: [], orig: new Map(), newNote: null, anchorRelTick: 0, moved: false, additive: false,
      });
      requestRedraw();
      return;
    }
    // ② bottom automation lane: INSERT / DRAG control points (§user: point-based, not freehand). Guarded FIRST so
    // a lane gesture never touches notes. Click on empty → insert a point + drag it; click ON a point → drag it;
    // right-click a point → delete (onContextMenu). Commits ONCE on pointerup (one undo step).
    if (laneOpenRef.current && y >= noteBottom() && x >= KEY_COL_W) {
      const cfg = laneCfg(laneParamRef.current);
      const laneTop = noteBottom();
      const stored = paramCurvesRef.current?.[cfg.id];
      const rel = Math.max(0, Math.round(relTickAt(e.clientX)));
      const val = Math.round(yToParam(y, cfg.min, cfg.max, laneTop, LANE_H) * 10) / 10; // 0.1-unit quantize
      const hitIdx = laneParamPointAt(e.clientX, e.clientY);
      let curve: { xs: number[]; ys: number[] }, idx: number;
      if (hitIdx >= 0 && stored) {
        curve = { xs: [...stored.xs], ys: [...stored.ys] }; idx = hitIdx; // grab the existing point
      } else {
        const xs = stored ? [...stored.xs] : [], ys = stored ? [...stored.ys] : [];
        const exact = xs.indexOf(rel);
        if (exact >= 0) { curve = { xs, ys }; idx = exact; } // a point already sits at this tick → move it
        else { let pos = xs.findIndex((xx) => xx > rel); if (pos < 0) pos = xs.length; xs.splice(pos, 0, rel); ys.splice(pos, 0, val); curve = { xs, ys }; idx = pos; }
      }
      dragRef.current = withPreview({
        kind: "param-point", clientX0: e.clientX, clientY0: e.clientY, curX: e.clientX, curY: e.clientY,
        activeIds: [], orig: new Map(), newNote: null, anchorRelTick: rel, moved: false, additive: false,
        param: cfg.id, pointCurve: curve, pointIdx: idx,
      });
      requestRedraw();
      return;
    }
    const hit = noteAt(e.clientX, e.clientY);
    const tl = toolRef.current;

    if (tl === "pitch") {
      // Pitch tool = paint Pitch Deviation (SynthV Pencil): drag across the pitch area to draw the TARGET f0
      // line; on commit the delta vs the automatic line (base ⊕ transition + vibrato) is stored into the
      // segment's pitchDev (interval-replace). A single click seeds one point; dragging appends the path.
      // QUANTIZE to whole ticks / whole cents so sub-pixel mouse jitter (the pitch line is ~6¢/px) can't make
      // the drawn line shiver; 1¢ is far below the audible/visible floor and matches normalizeCurve on commit.
      const rel = Math.max(0, Math.round(relTickAt(e.clientX)));
      const c0 = Math.round(centsAt(e.clientY));
      playback.playPreviewTone(centsToHz(c0), 0); // sustained; follows the drawn pitch
      dragRef.current = withPreview({
        kind: "pitch-paint", clientX0: e.clientX, clientY0: e.clientY, curX: e.clientX, curY: e.clientY,
        activeIds: [], orig: new Map(), newNote: null, anchorRelTick: rel, moved: false, additive: false,
        paint: { xs: [rel], ys: [c0] },
      });
      requestRedraw();
      return;
    }

    if (tl === "delete") {
      if (hit) { commitNotes(notesRef.current.filter((n) => n.id !== hit.note.id), []); }
      else { dragRef.current = withPreview({ kind: "marquee-delete", clientX0: x, clientY0: localXY(e.clientX, e.clientY).y, curX: x, curY: localXY(e.clientX, e.clientY).y, activeIds: [], orig: new Map(), newNote: null, anchorRelTick: 0, moved: false, additive: false }); }
      requestRedraw();
      return;
    }

    if (hit) {
      // Pen tool: grabbing a note EXTENDS its length (resize its tail), never moves it (§user — a click on a
      // note in draw mode most likely means "make it longer", and there was no other easy way to resize).
      if (toolRef.current === "pen") {
        selectNotes([hit.note.id]);
        const o = notesRef.current.find((n) => n.id === hit.note.id)!;
        playback.playPreviewTone(pitchToHz(o.pitch), 160);
        dragRef.current = withPreview({
          kind: "resize", clientX0: e.clientX, clientY0: e.clientY, curX: e.clientX, curY: e.clientY,
          activeIds: [o.id], orig: new Map([[o.id, o]]), newNote: null, anchorRelTick: o.tick, moved: false, additive: false,
        });
        return;
      }
      // Arrow tool: select (respect shift/ctrl multi-select), then move or resize by edge
      let nextSel: string[];
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        nextSel = selRef.current.has(hit.note.id) ? [...selRef.current].filter((i) => i !== hit.note.id) : [...selRef.current, hit.note.id];
      } else {
        nextSel = selRef.current.has(hit.note.id) ? [...selRef.current] : [hit.note.id];
      }
      selectNotes(nextSel);
      const ids = nextSel.includes(hit.note.id) ? nextSel : [hit.note.id];
      const orig = new Map(notesRef.current.filter((n) => ids.includes(n.id)).map((n) => [n.id, n]));
      // A MOVE gets a SUSTAINED tone that retunes along the drag (audition the pitch the whole way, §user);
      // a resize (edge) just gets a brief click. Stopped on pointerup/cancel/Esc/unmount (all wired).
      playback.playPreviewTone(pitchToHz(hit.note.pitch), hit.onEdge ? 160 : 0);
      dragRef.current = withPreview({
        kind: hit.onEdge ? "resize" : "move",
        clientX0: e.clientX, clientY0: e.clientY, curX: e.clientX, curY: e.clientY,
        activeIds: ids, orig, newNote: null, anchorRelTick: hit.note.tick, moved: false, additive: false,
        startRel: relTickAt(e.clientX), startPitch: pitchAt(e.clientY),
      });
      return;
    }

    // empty space
    const drawNote = tl === "pen" || ((tl === "arrow") && (e.ctrlKey || e.metaKey));
    if (drawNote) {
      const snap = snapTicks();
      const relStart = Math.max(0, snapFloor(relTickAt(e.clientX), snap));
      const p = clampPitch(pitchAt(e.clientY));
      // Created note honors the 60ms floor like resize/truncation do (§9.2) — a fine grid can't make an
      // inaudible sub-floor note.
      const newNote: Note = { id: crypto.randomUUID(), tick: relStart, duration: Math.max(1, snap), pitch: p, lyric: defaultLyricRef.current, velocity: 100 };
      playback.playPreviewTone(pitchToHz(p), 0); // sustained while drawing; stops on pointerup
      dragRef.current = withPreview({
        kind: "create", clientX0: e.clientX, clientY0: e.clientY, curX: e.clientX, curY: e.clientY,
        activeIds: [newNote.id], orig: new Map(), newNote, anchorRelTick: relStart, moved: false, additive: false,
      });
      requestRedraw();
      return;
    }

    // arrow on empty → marquee select
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    if (!additive) selectNotes([]);
    const p0 = localXY(e.clientX, e.clientY);
    dragRef.current = withPreview({ kind: "marquee", clientX0: p0.x, clientY0: p0.y, curX: p0.x, curY: p0.y, activeIds: [], orig: new Map(), newNote: null, anchorRelTick: 0, moved: false, additive });
    requestRedraw();
  }, [setActivePane, selectNotes, applyNoteEdits, part, segmentId]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const p = localXY(e.clientX, e.clientY);
    mouseRef.current = p;
    const d = dragRef.current;
    if (!d) {
      // hover cursor: a left-right resize affordance where grabbing would change a note's LENGTH (Arrow near
      // the tail / Pen anywhere on a note, §user). Delete keeps its CSS crosshair.
      const cv = canvasRef.current;
      if (cv) {
        if (p.y < RULER_H && p.x >= KEY_COL_W) cv.style.cursor = "col-resize"; // ② ruler = seek the playhead
        else if (laneOpenRef.current && p.y >= noteBottom() && p.x >= KEY_COL_W)
          cv.style.cursor = laneParamPointAt(e.clientX, e.clientY) >= 0 ? "grab" : "crosshair"; // ② over a point vs insert
        else if (toolRef.current === "delete") cv.style.cursor = "";
        else { const hov = noteAt(e.clientX, e.clientY); cv.style.cursor = hov && (toolRef.current === "pen" || hov.onEdge) ? "ew-resize" : ""; }
      }
      requestRedraw(); return; // hover → redraw so the mouse-position guide follows (all tools)
    }
    d.moved = true;
    if (d.kind === "marquee" || d.kind === "marquee-delete") {
      d.curX = p.x; d.curY = p.y;
      if (!edgeRafRef.current) { // kick off edge auto-scroll when the cursor reaches a border
        const { w } = sizeRef.current, EDGE = 28;
        // bottom trigger = the note-area bottom (above the lane band), MATCHING edgeScrollRef's condition —
        // else the lane's LANE_H creates a dead zone where the loop wants to scroll but was never kicked off.
        if (p.x < KEY_COL_W + EDGE || p.x > w - EDGE || p.y < RULER_H + EDGE || p.y > noteBottom() - EDGE)
          edgeRafRef.current = requestAnimationFrame(edgeScrollRef.current);
      }
    }
    else {
      d.curX = e.clientX; d.curY = e.clientY;
      // live retune preview for move (only when pitch is unlocked; else the tone stays put — pitch is locked)
      if (d.kind === "move") {
        // Each axis ACTIVATES independently once its OWN motion passes a threshold; both are then measured
        // from the origin. A pure-vertical drag never nudges timing (X stays inactive); switching to
        // horizontal mid-drag KEEPS the pitch already moved (no jump back to the start height — §user bug).
        if (!d.activeX && Math.abs(e.clientX - d.clientX0) > 4) d.activeX = true;
        if (!d.activeY && Math.abs(e.clientY - d.clientY0) > 4) d.activeY = true;
        const dPitch = d.activeY ? pitchAt(e.clientY) - (d.startPitch ?? 0) : 0; // CONTENT origin (scroll-safe, matches computePreview)
        const anyId = d.activeIds[0];
        const o = anyId ? d.orig.get(anyId) : undefined;
        if (o) playback.setPreviewToneHz(pitchToHz(clampPitch(o.pitch + dPitch)));
      } else if (d.kind === "pitch-paint" && d.paint) {
        const cy = Math.round(centsAt(e.clientY)); // quantize (see pointerdown) — kills sub-pixel line shiver
        d.paint.xs.push(Math.max(0, Math.round(relTickAt(e.clientX))));
        d.paint.ys.push(cy);
        playback.setPreviewToneHz(centsToHz(cy)); // hear the pitch being drawn
      } else if (d.kind === "param-point" && d.pointCurve && d.pointIdx !== undefined && d.param) {
        const cfg = laneCfg(d.param); // ② drag the grabbed point: x clamped strictly between neighbors (xs stays sorted), y in range
        const c = d.pointCurve, i = d.pointIdx;
        const rel = Math.max(0, Math.round(relTickAt(e.clientX)));
        const lo = i > 0 ? c.xs[i - 1]! + 1 : 0;
        const hi = i < c.xs.length - 1 ? c.xs[i + 1]! - 1 : Number.MAX_SAFE_INTEGER;
        c.xs[i] = Math.min(hi, Math.max(lo, rel));
        c.ys[i] = Math.round(yToParam(localXY(e.clientX, e.clientY).y, cfg.min, cfg.max, noteBottom(), LANE_H) * 10) / 10;
      } else if (d.kind === "ruler-seek") {
        // playback may have STARTED mid-drag (Space is a global key) → pin `seeking` here too, not just at
        // pointerdown, so the transport reschedules from the drop tick on release (mirrors TimelineRuler:124).
        const a = useAudioStore.getState();
        if (a.isPlaying && !a.seeking) a.setSeeking(true);
        useProjectStore.getState().setPlayhead(Math.max(0, Math.round(xToTick(localXY(e.clientX, e.clientY).x - KEY_COL_W, viewRef.current))));
      } else if (d.kind === "create" && d.newNote) {
        const snap = snapTicks();
        const end = Math.max(d.anchorRelTick + snap, snapRound(relTickAt(e.clientX), snap));
        d.newNote = { ...d.newNote, duration: Math.max(snap, end - d.anchorRelTick) };
      }
    }
    requestRedraw();
  }, [requestRedraw]);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (edgeRafRef.current) { cancelAnimationFrame(edgeRafRef.current); edgeRafRef.current = 0; }
    if (!d) return; // a bare key/note click (no drag) — let its short audition tone ring out
    playback.stopPreviewTone();
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);

    if (d.kind === "marquee" || d.kind === "marquee-delete") {
      const ids = notesInMarquee(d);
      if (d.kind === "marquee-delete") {
        if (ids.length) commitNotes(notesRef.current.filter((n) => !ids.includes(n.id)), []);
      } else {
        const base = d.additive ? [...selRef.current] : [];
        selectNotes([...new Set([...base, ...ids])]);
      }
      requestRedraw();
      return;
    }
    if (d.kind === "create" && d.newNote) {
      const nn = d.newNote;
      commitNotes([...notesRef.current, nn], [nn.id]);
      selectNotes([nn.id]);
      return;
    }
    if (d.kind === "pitch-paint" && d.paint && part) {
      const sorted = [...notesRef.current].sort((a, b) => a.tick - b.tick);
      const opts = { tempo: tempoRef.current, defaultTransition: transitionRef.current };
      setSegmentPitchDev(part.trackId, segmentId, paintedDev(sorted, d.paint, pitchDevRef.current, opts)); // normalizeCurve inside
      requestRedraw();
      return;
    }
    if (d.kind === "ruler-seek") {
      // release the seek flag so the transport reschedules from the new playhead (if it was playing).
      if (useAudioStore.getState().seeking) useAudioStore.getState().setSeeking(false);
      requestRedraw();
      return;
    }
    if (d.kind === "param-point" && d.pointCurve && d.param && part) {
      // ② one set() → one undo step. An empty curve (last point dragged out of use / deleted) clears the lane.
      // normalizeCurve(...,"param") rounds/dedups + canonical key order (sig↔serialize consistent).
      setSegmentParamCurve(part.trackId, segmentId, d.param, d.pointCurve.xs.length ? d.pointCurve : undefined);
      requestRedraw();
      return;
    }
    // move / resize
    commitNotes(computePreview(d), d.activeIds);
  }, [selectNotes, part, segmentId, setSegmentPitchDev, setSegmentParamCurve, requestRedraw]);

  // ② right-click a lane control point → delete it (onPointerDown bails on non-left buttons). Suppress the
  // native context menu inside the editor either way. One set() = one undo step; empty curve clears the lane.
  const onContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    if (!part || !laneOpenRef.current) return;
    const { x, y } = localXY(e.clientX, e.clientY);
    if (y < noteBottom() || x < KEY_COL_W) return;
    const idx = laneParamPointAt(e.clientX, e.clientY);
    if (idx < 0) return;
    const cfg = laneCfg(laneParamRef.current);
    const stored = paramCurvesRef.current?.[cfg.id];
    if (!stored) return;
    const xs = stored.xs.filter((_, i) => i !== idx);
    const ys = stored.ys.filter((_, i) => i !== idx);
    setSegmentParamCurve(part.trackId, segmentId, cfg.id, xs.length ? { xs, ys } : undefined);
    requestRedraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [part, segmentId, setSegmentParamCurve, requestRedraw]);

  const notesInMarquee = (d: DragState): string[] => {
    const v = viewRef.current;
    const start = startRef.current;
    const x0 = Math.min(d.clientX0, d.curX), x1 = Math.max(d.clientX0, d.curX);
    const y0 = Math.min(d.clientY0, d.curY), y1 = Math.max(d.clientY0, d.curY);
    const out: string[] = [];
    for (const n of notesRef.current) {
      const nx0 = KEY_COL_W + noteTickToX(n.tick, start, v);
      const nx1 = KEY_COL_W + noteTickToX(n.tick + n.duration, start, v);
      const ny0 = pitchToY(n.pitch, v);
      const ny1 = ny0 + v.rowH;
      if (nx1 >= x0 && nx0 <= x1 && ny1 >= y0 && ny0 <= y1) out.push(n.id);
    }
    return out;
  };

  // Notes in canonical next/prev order — tick asc, id tiebreak. ONE comparator for lyric-commit
  // distribution AND Tab/Shift+Tab navigation (so "who's next" never drifts between the two paths).
  const orderedNotes = () => [...(part?.notes ?? [])].sort((a, b) => a.tick - b.tick || (a.id < b.id ? -1 : 1));

  // The lyric-input overlay geometry (x/y/w) for a note — shared by double-click open and Tab nav.
  const lyricEditFor = (note: Note) => {
    const v = viewRef.current;
    const x0 = KEY_COL_W + noteTickToX(note.tick, part!.start, v);
    const y = pitchToY(note.pitch, v);
    const w = Math.max(40, noteTickToX(note.tick + note.duration, part!.start, v) - noteTickToX(note.tick, part!.start, v));
    return { id: note.id, x: Math.max(KEY_COL_W, x0), y, w, value: note.lyric };
  };

  const onDoubleClick = useCallback((e: React.MouseEvent) => {
    if (toolRef.current === "pitch" || toolRef.current === "delete") return; // lyric editing lives in Arrow/Pen only (§user)
    // ② lane guard (mirror onPointerDown:818 / onContextMenu:1055): never open a lyric editor inside the bottom
    // automation lane — a note can scroll behind the band, and the <input> would render overlaying the lane.
    if (laneOpenRef.current && localXY(e.clientX, e.clientY).y >= noteBottom()) return;
    const hit = noteAt(e.clientX, e.clientY);
    if (!hit || !part) return;
    setLyricEdit(lyricEditFor(hit.note));
  }, [part]);

  const commitLyric = (id: string, value: string) => {
    if (!part) { setLyricEdit(null); return; }
    const tokens = splitLyricTokens(value.trim(), defaultLyricRef.current);
    const ordered = orderedNotes();
    const startIdx = ordered.findIndex((n) => n.id === id);
    if (tokens.length <= 1 || startIdx < 0) {
      applyNoteEdits(part.trackId, segmentId, { update: { [id]: { lyric: tokens[0] ?? defaultLyricRef.current } } });
    } else {
      const update: Record<string, Partial<Note>> = {};
      for (let i = 0; i < tokens.length && startIdx + i < ordered.length; i++) {
        update[ordered[startIdx + i]!.id] = { lyric: tokens[i]! };
      }
      applyNoteEdits(part.trackId, segmentId, { update });
    }
    setLyricEdit(null);
  };

  // Tab (dir +1) / Shift+Tab (dir −1) while editing a lyric: commit the current note, then open the
  // adjacent note's lyric input (SynthV/OpenUTAU convention). At the ends, just commit + close.
  const navLyric = (fromId: string, value: string, dir: 1 | -1) => {
    if (!part) { setLyricEdit(null); return; }
    lyricNavRef.current = true; // the old input unmounts (key change) → its blur must NOT re-commit
    queueMicrotask(() => { lyricNavRef.current = false; });
    const ordered = orderedNotes(); // ticks don't change on a lyric commit → pre-commit order == post
    const idx = ordered.findIndex((n) => n.id === fromId);
    commitLyric(fromId, value); // applyNoteEdits + setLyricEdit(null)
    const next = idx < 0 ? undefined : ordered[idx + dir];
    if (next) setLyricEdit(lyricEditFor(next)); // overrides the null → opens the neighbour (remounts via key)
  };

  // ── keyboard (own handler; MUST bail on editable targets — §9.6) ──
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (useAppStore.getState().activePane !== "vocal") return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable)) return;
      const p = part;
      if (!p) return;
      const sel = [...selRef.current];
      if (e.key === "Delete" || e.key === "Backspace") {
        if (sel.length) { e.preventDefault(); commitNotes(p.notes.filter((n) => !selRef.current.has(n.id)), []); selectNotes([]); }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a") {
        e.preventDefault(); selectNotes(p.notes.map((n) => n.id));
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") {
        clipboardRef.current = p.notes.filter((n) => selRef.current.has(n.id));
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "x") {
        clipboardRef.current = p.notes.filter((n) => selRef.current.has(n.id));
        if (sel.length) { commitNotes(p.notes.filter((n) => !selRef.current.has(n.id)), []); selectNotes([]); }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") {
        pasteAt();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "d") {
        e.preventDefault(); clipboardRef.current = p.notes.filter((n) => selRef.current.has(n.id)); pasteAt(true);
      } else if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key) && sel.length) {
        e.preventDefault(); nudge(e.key);
      } else if (e.key === "Escape") {
        // Cancel a live gesture / stop preview playback — MUST also stop the sustained tone (else it rings).
        if (playRafRef.current) { cancelAnimationFrame(playRafRef.current); playRafRef.current = 0; playback.stopPreviewTone(); setPlaying(false); }
        else if (dragRef.current) { dragRef.current = null; playback.stopPreviewTone(); requestRedraw(); } else selectNotes([]);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [part, segmentId]);

  const pasteAt = (dupInPlace = false) => {
    if (!part || clipboardRef.current.length === 0) return;
    const clip = clipboardRef.current;
    const minTick = Math.min(...clip.map((n) => n.tick));
    const anchor = dupInPlace ? minTick + snapTicks() : Math.max(0, relTickAt((mouseRef.current?.x ?? KEY_COL_W + 20) + (canvasRef.current?.getBoundingClientRect().left ?? 0)));
    const shift = dupInPlace ? snapTicks() : anchor - minTick;
    const pasted = clip.map((n): Note => ({ ...n, id: crypto.randomUUID(), tick: Math.max(0, n.tick + shift) }));
    commitNotes([...part.notes, ...pasted], pasted.map((n) => n.id));
    selectNotes(pasted.map((n) => n.id));
  };

  const nudge = (key: string) => {
    if (!part) return;
    const snap = snapTicks();
    const dTick = key === "ArrowLeft" ? -snap : key === "ArrowRight" ? snap : 0;
    const dPitch = key === "ArrowUp" ? 1 : key === "ArrowDown" ? -1 : 0;
    const sel = part.notes.filter((n) => selRef.current.has(n.id));
    if (sel.length === 0) return;
    // GROUP clamp at the tick=0 / pitch 0-127 walls so the selection keeps its shape (no collapse/overlap).
    const dT = dTick < 0 ? Math.max(dTick, -Math.min(...sel.map((n) => n.tick))) : dTick;
    const loP = Math.min(...sel.map((n) => n.pitch));
    const hiP = Math.max(...sel.map((n) => n.pitch));
    const dP = Math.max(V_PITCH_MIN - loP, Math.min(V_PITCH_MAX - hiP, dPitch));
    const next = part.notes.map((n) => (selRef.current.has(n.id) ? { ...n, tick: n.tick + dT, pitch: n.pitch + dP } : n));
    commitNotes(next, [...selRef.current]);
  };

  // ── wheel (Ctrl=h-zoom cursor-anchored / Alt=row-height / plain=v-scroll / Shift=h-scroll) ──
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.stopPropagation();
    const v = viewRef.current;
    const sx0 = v.scrollX, sy0 = v.scrollY; // capture pre-scroll → re-anchor an in-progress marquee below
    // Upper scroll bound: the part's right edge (+ a margin) — can't scroll off into empty space.
    const maxScrollX = () => Math.max(0, (startRef.current + durRef.current) * v.ppt + 400 - Math.max(1, sizeRef.current.w - KEY_COL_W));
    if (e.ctrlKey) {
      e.preventDefault();
      const cv = canvasRef.current; const r = cv?.getBoundingClientRect();
      const cx = (r ? e.clientX - r.left : sizeRef.current.w / 2) - KEY_COL_W;
      const tickAt = (cx + v.scrollX) / v.ppt;
      v.ppt = Math.max(0.02, Math.min(4, v.ppt * (e.deltaY > 0 ? 0.9 : 1.1)));
      v.scrollX = Math.max(0, Math.min(maxScrollX(), tickAt * v.ppt - cx));
    } else if (e.altKey) {
      // Vertical (row-height) zoom, ANCHORED at the cursor — the pitch under the mouse stays at the same
      // screen y (mirrors the horizontal ctrl-zoom; unified with the arrangement's Alt+wheel). This is the
      // fiddly coordinate part: solve scrollY so the cursor's continuous cents map back to its screen y.
      e.preventDefault();
      const cvR = canvasRef.current?.getBoundingClientRect();
      const cy = Math.min(noteBottom(), Math.max(RULER_H, cvR ? e.clientY - cvR.top : (sizeRef.current.h + RULER_H) / 2)); // cursor y clamped to the note area (not the lane)
      const anchorCents = yToCents(cy, v);
      v.rowH = Math.max(V_ROW_H_MIN, Math.min(V_ROW_H_MAX, v.rowH * (e.deltaY > 0 ? 0.9 : 1.1)));
      const maxSY = Math.max(0, rowsContentHeight(v.rowH) - visNoteH());
      v.scrollY = Math.max(0, Math.min(maxSY, centsToY(anchorCents, { ...v, scrollY: 0 }) - cy)); // keep anchorCents under the cursor
    } else if (e.shiftKey) {
      v.scrollX = Math.max(0, Math.min(maxScrollX(), v.scrollX + e.deltaY));
    } else {
      v.scrollY = Math.max(0, Math.min(Math.max(0, rowsContentHeight(v.rowH) - visNoteH()), v.scrollY + e.deltaY));
    }
    // A wheel-scroll during a note-move changes the cursor's CONTENT pitch, but onPointerMove doesn't fire
    // (the mouse didn't move) — so refresh the sustained preview tone here, else it keeps sounding the old
    // pitch while the note visibly follows the scroll (§user preview race).
    const d = dragRef.current;
    if (d?.kind === "move" && d.activeY) {
      const o = d.activeIds[0] ? d.orig.get(d.activeIds[0]) : undefined;
      if (o) playback.setPreviewToneHz(pitchToHz(clampPitch(o.pitch + (pitchAt(d.curY) - (d.startPitch ?? 0)))));
    } else if (d && (d.kind === "marquee" || d.kind === "marquee-delete")) {
      // pin the marquee's anchor to content while scrolling — else the box stays at its screen position and
      // selects the wrong notes as content scrolls under it (§user bug; same fix the edge auto-scroll uses).
      d.clientX0 -= v.scrollX - sx0;
      d.clientY0 -= v.scrollY - sy0;
    }
    requestRedraw();
  }, [requestRedraw]);

  if (!part) return null;

  const TOOLS: { id: Tool; label: string; icon: ReactElement }[] = [
    { id: "arrow", label: t("vocalEditor.toolArrow"), icon: <path d="M5 3l14 8-6 1.5L10 19 8 12 5 3z" /> },
    { id: "pen", label: t("vocalEditor.toolPen"), icon: <path d="M4 20l3-1 10-10-2-2L5 17l-1 3zM15 5l2 2 2-2-2-2-2 2z" /> },
    { id: "pitch", label: t("vocalEditor.toolPitch"), icon: <path d="M3 17c4 0 4-10 8-10s4 10 9 4" fill="none" stroke="currentColor" strokeWidth="2" /> },
    { id: "delete", label: t("vocalEditor.toolDelete"), icon: <path d="M6 7h12l-1 13H7L6 7zm3-3h6l1 2H8l1-2z" /> },
  ];

  return (
    <div
      className={`vocal-editor${maximized ? " vocal-editor--max" : ""}`}
      style={maximized ? undefined : style}
      onPointerDownCapture={() => { setActivePane("vocal"); playback.getPreviewContext(); }}
      onFocusCapture={() => setActivePane("vocal")}
    >
      <div className="vocal-editor-header">
        <span className="vocal-editor-title">{part.trackName || t("vocalEditor.title")}</span>
        {/* ② loudness / formant automation — two labels sitting DIRECTLY next to the track title (§user: no
            "lane" jargon, no extra open step). Each is a self-toggle: click opens + selects that param's bottom
            editor; click the active one again closes it; clicking the other switches param with it staying open. */}
        <div className="vocal-lane-ctl">
          {LANE_PARAMS.map((lp) => (
            <button
              key={lp.id}
              className={`snap-toggle vocal-grid-btn${laneOpen && laneParam === lp.id ? " active" : ""}`}
              onClick={() => {
                if (laneOpen && laneParam === lp.id) setLaneOpen(false);
                else { setLaneParam(lp.id); setLaneOpen(true); }
              }}
            >{t(lp.labelKey)}</button>
          ))}
        </div>
        <div className="vocal-editor-header-spacer" />
        <label className="vocal-grid-label">{t("vocalEditor.grid")}</label>
        <div className="vocal-grid-select">
          {GRID_DIVS.map((g) => (
            <button key={g.div} className={`snap-toggle vocal-grid-btn${gridDiv === g.div ? " active" : ""}`} onClick={() => setGridDiv(g.div)}>{g.key}</button>
          ))}
        </div>
        <button className="vocal-icon-btn" title={playing ? t("vocalEditor.stop") : t("vocalEditor.preview")} onClick={() => (playing ? stopPreviewPlay() : startPreviewPlay())}>
          <svg viewBox="0 0 24 24" width="13" height="13"><path fill="currentColor" d={playing ? "M6 5h4v14H6zM14 5h4v14h-4z" : "M7 5l12 7-12 7z"} /></svg>
        </button>
        <button className="vocal-icon-btn" title={maximized ? t("vocalEditor.restore") : t("vocalEditor.maximize")} onClick={() => setMaximized((m) => !m)}>
          <svg viewBox="0 0 24 24" width="15" height="15"><path fill="currentColor" d={maximized ? "M8 8h8v8H8V8zM4 4h6v2H6v4H4V4zm10 0h6v6h-2V6h-4V4z" : "M4 4h6v2H6v4H4V4zm10 0h6v6h-2V6h-4V4zM6 14v4h4v2H4v-6h2zm12 0h2v6h-6v-2h4v-4z"} /></svg>
        </button>
        <button className="vocal-icon-btn" title={t("vocalEditor.close")} onClick={onClose}>
          <svg viewBox="0 0 24 24" width="15" height="15"><path fill="none" d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" /></svg>
        </button>
      </div>
      <div className="vocal-editor-body">
        <div className={`vocal-tools${tool === "delete" ? " danger" : ""}`}>
          {TOOLS.map((tt) => (
            <button
              key={tt.id}
              className={`snap-toggle vocal-tool${tool === tt.id ? " active" : ""}${tt.id === "delete" ? " vocal-tool-delete" : ""}`}
              title={tt.label}
              onClick={() => setTool(tt.id)}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">{tt.icon}</svg>
            </button>
          ))}
        </div>
        <div className={`vocal-canvas-wrap${tool === "delete" ? " delete-mode" : ""}`} ref={wrapRef}>
          <canvas
            ref={canvasRef}
            className="vocal-canvas"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onLostPointerCapture={onPointerUp}
            onDoubleClick={onDoubleClick}
            onContextMenu={onContextMenu}
            onWheel={onWheel}
          />
          {tool === "delete" && <div className="vocal-delete-overlay" />}
          {lyricEdit && (
            <input
              key={lyricEdit.id}
              className="vocal-lyric-input"
              autoFocus
              style={{ left: lyricEdit.x, top: lyricEdit.y, width: Math.max(40, lyricEdit.w) }}
              defaultValue={lyricEdit.value}
              onFocus={() => { lyricCancelRef.current = false; }} // a freshly-focused input never suppresses its OWN commit
              onKeyDown={(e) => {
                if (e.key === "Enter") commitLyric(lyricEdit.id, (e.target as HTMLInputElement).value);
                else if (e.key === "Escape") { lyricCancelRef.current = true; setLyricEdit(null); }
                else if (e.key === "Tab") { e.preventDefault(); navLyric(lyricEdit.id, (e.target as HTMLInputElement).value, e.shiftKey ? -1 : 1); } // preventDefault: stop the native focus-move off the input
                e.stopPropagation();
              }}
              onBlur={(e) => {
                if (lyricNavRef.current || lyricCancelRef.current) { lyricCancelRef.current = false; return; } // Tab already committed / Escape cancelled
                commitLyric(lyricEdit.id, e.target.value);
              }}
            />
          )}
        </div>
        <VocalSidebar
          trackId={part.trackId}
          segmentId={segmentId}
          notes={part.notes}
          selectedIds={selectedNotes}
          trackTransition={part.transition}
          vocalParams={part.vocalParams}
          voiceModel={part.voiceModel}
          onRender={render}
          rendering={vocalRenderActive}
        />
      </div>
    </div>
  );
}

// ── module helpers ──

/** Full per-note content signature for the commit diff (mirrors history.ts noteSig, minus id which is the
 *  diff key). ⚠ MUST cover EVERY editable field: commitNotes drops an edit whose sig is unchanged, so a
 *  transition/vibrato/pitchAuto-only edit would be SILENTLY lost if the sig omitted them (silent-regression
 *  class). Keep in lockstep with history.ts noteSig. */
function noteSig(n: Note): string {
  const t = n.transition;
  const tr = t ? `${t.offsetMs ?? ""}|${t.durLeftMs ?? ""}|${t.durRightMs ?? ""}|${t.depthLeftCents ?? ""}|${t.depthRightCents ?? ""}|${t.openEdgeCents ?? ""}` : "";
  const v = n.vibrato;
  const vib = v ? `${v.depthCents},${v.freqHz},${v.phase},${v.startMs},${v.easeInMs},${v.easeOutMs}` : "";
  return (
    `${n.tick}.${n.duration}.${n.pitch}.${n.lyric}.${n.phoneme ?? ""}.${n.velocity}` +
    `.${n.detune ?? 0}.${n.tie ? 1 : 0}.${n.pitchAuto === false ? 0 : 1}.${n.lang ?? ""}.${n.phonemeInput ?? ""}.${tr}.${vib}`
  );
}

const SMALL_KANA = new Set([..."ぁぃぅぇぉゃゅょゎっゕゖァィゥェォャュョヮッ"]);
/** Split a typed lyric phrase into per-note tokens (§9.2 auto-distribute). Whitespace-separated first;
 *  else an all-kana run splits per mora (a base kana + trailing small kana); an all-Han run splits per
 *  character (S58 — one hanzi per note; the Rust zh G2P reads phrase context from the NOTE SEQUENCE, so
 *  polyphones still resolve after the split); otherwise one token (latin needs explicit spaces).
 *  Minimal + JS-side (a splitter, NOT a dictionary — the Rust classifier owns phoneme validation). */
function splitLyricTokens(s: string, emptyFallback: string): string[] {
  if (!s) return [emptyFallback];
  if (/\s/.test(s)) return s.split(/\s+/).filter(Boolean);
  if (/^[\p{Script=Han}]+$/u.test(s)) return [...s];
  const isKana = /^[぀-ヿ゠-ヿー]+$/.test(s);
  if (!isKana) return [s];
  const out: string[] = [];
  for (const ch of s) {
    if (out.length > 0 && SMALL_KANA.has(ch)) out[out.length - 1] += ch;
    else out.push(ch);
  }
  return out;
}
