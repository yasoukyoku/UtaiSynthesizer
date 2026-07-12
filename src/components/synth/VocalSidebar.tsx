// ② Vocal editor — PROPERTY SIDEBAR (S48 Phase 5 step 6, §9.6/§10.3/§10.4). The editor body's THIRD flex
// child (mirrors NodePalette|canvas): self-contained, hides/shows with the editor, no App.tsx change. A
// PASSIVE numeric mirror (§9.3) — it never steals a canvas click; it just exposes the pitch model the
// overlay/preview already evaluate:
//   ① selected note(s) · Pitch TRANSITION override (SynthV glide/portamento between notes + the open-edge
//      scoop-in/drift-out §10.5). Each field is optional: absent = inherit the track default; a slider edit
//      writes an explicit override; ↺ resets it back to inherit. Editing applies to the WHOLE selection in ONE
//      undo step.
//   ② selected note(s) · VIBRATO (add/remove + the 6 SynthV fields).
//   ③ track · default TRANSITION (VocalTrackParams.transition — the concrete base every note inherits).
// Sliders reuse VolumeFader (the one gesture-bracketed fader — TrackList uses the same begin/commitTransaction
// pattern); a drag = ONE undo step. effTransition is imported (not re-derived) so the shown effective value ==
// what f0eval evaluates. All strings go through i18n.
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { VolumeFader } from "../common/VolumeFader";
import { useProjectStore } from "../../store/project";
import { useHistoryStore } from "../../store/history";
import { useAppStore } from "../../store/app";
import { useVoiceModelStore, voiceHasDiffusion, type VoiceModelEntry } from "../../store/voice-models";
import { effTransition } from "../../lib/f0eval";
import { DEFAULT_TRANSITION } from "../../lib/vocalNotes";
import { VOCAL_LANGUAGES, langById } from "../../lib/vocal/languages";
import { backendOf, backendLabel, pickVoiceForTrack } from "../../lib/vocal/voicePick";
import { DIFFUSION_METHODS, RVC_DEFAULTS, SOVITS_DEFAULTS, type RvcOptions, type SovitsOptions } from "../../lib/workflow/voiceDefaults";
import { VocoderSelect } from "../workflow/nodes/VoiceModelPicker";
import type { Note, NoteTransition, VibratoSpec, VocalTrackParams } from "../../types/project";
import "./VocalSidebar.css";

/** Default vibrato seeded by "Add vibrato" (depthCents>0 so normalizeNote keeps it). ⚠ startMs/ease are SMALL
 *  on purpose so vibrato is VISIBLE the instant it's added, even on a shorter (tail) note — the old SynthV-ish
 *  250 ms onset + 200 ms fades suppressed it entirely below ~2 beats (§user "尾音加不了颤音"). The onset delay
 *  is still a slider, so a swell-in can be dialed back in per note. */
const DEFAULT_VIBRATO: VibratoSpec = { depthCents: 100, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 };

// UI ranges are deliberately GENEROUS (§user: bold freedom is the real expressiveness lever). They stay within
// the data-layer clamps (normalizeTransition: dur 0–2000, depth ±1200; vibrato: depth 0–2400, freq 0.1–40,
// start 0–60000, ease 0–10000) so a slider value never gets silently re-clamped on write.
interface FieldCfg { key: keyof Required<NoteTransition>; min: number; max: number; step: number; unit: string; bipolar: boolean; }
const TRANSITION_FIELDS: FieldCfg[] = [
  { key: "durLeftMs", min: 0, max: 1000, step: 1, unit: "ms", bipolar: false },
  { key: "durRightMs", min: 0, max: 1000, step: 1, unit: "ms", bipolar: false },
  { key: "depthLeftCents", min: -600, max: 600, step: 1, unit: "¢", bipolar: true },
  { key: "depthRightCents", min: -600, max: 600, step: 1, unit: "¢", bipolar: true },
  { key: "offsetMs", min: -500, max: 500, step: 1, unit: "ms", bipolar: true },
  { key: "openEdgeCents", min: 0, max: 600, step: 1, unit: "¢", bipolar: false },
];
interface VibCfg { key: keyof VibratoSpec; min: number; max: number; step: number; unit: string; bipolar: boolean; }
const VIBRATO_FIELDS: VibCfg[] = [
  { key: "depthCents", min: 0, max: 1200, step: 1, unit: "¢", bipolar: false },
  { key: "freqHz", min: 0.5, max: 40, step: 0.1, unit: "Hz", bipolar: false },
  { key: "phase", min: -1, max: 1, step: 0.01, unit: "", bipolar: true },
  { key: "startMs", min: 0, max: 2000, step: 5, unit: "ms", bipolar: false },
  { key: "easeInMs", min: 0, max: 2000, step: 5, unit: "ms", bipolar: false },
  { key: "easeOutMs", min: 0, max: 2000, step: 5, unit: "ms", bipolar: false },
];

const fmt = (v: number, step: number): string => (step >= 1 ? String(Math.round(v)) : step >= 0.1 ? v.toFixed(1) : v.toFixed(2));

interface Props {
  trackId: string;
  segmentId: string;
  notes: Note[];
  selectedIds: string[];
  trackTransition: Required<NoteTransition>;
  /** Track-level render config (backend / ScoreToCV speaker+lang / transpose). */
  vocalParams: VocalTrackParams;
  /** Selected SVC voice (singer) name, from Track.voiceModel. */
  voiceModel?: string;
  onRender: () => void;
  rendering: boolean;
}

export function VocalSidebar({ trackId, segmentId, notes, selectedIds, trackTransition, vocalParams, voiceModel, onRender, rendering }: Props) {
  const { t, i18n } = useTranslation();
  // SynthV-style tabs: "singer" = voice + tone/quality; "pitch" = pitch tuning. Splits the (previously long)
  // single scroll into two focused panels; the Render action is a pinned footer visible on both.
  const [tab, setTab] = useState<"singer" | "pitch">("singer");
  const applyNoteEdits = useProjectStore((s) => s.applyNoteEdits);
  const setVocalParams = useProjectStore((s) => s.setVocalParams);
  const toggleModelManager = useAppStore((s) => s.toggleModelManager);
  const models = useVoiceModelStore((s) => s.models);

  // ONE unified singer list (SoVITS + RVC). Identity is (model_type, name) — same-name rvc/sovits pairs are
  // a standard workflow, so never key by name alone; the picked model's type auto-sets the backend.
  const allVoices = useMemo(() => [...models.sovits, ...models.rvc], [models]);
  const selectedVoice = useMemo(
    () => allVoices.find((m) => m.name === voiceModel && backendOf(m) === vocalParams.backend),
    [allVoices, voiceModel, vocalParams.backend],
  );
  // Pick a singer → the SHARED single pick path (voiceModel + backend, one undo step) — the track-header
  // singer popup uses the same function (S58, NO-dup).
  const pickVoice = (m: VoiceModelEntry) => pickVoiceForTrack(trackId, m);

  // The selection = the notes to edit; the FIRST is the display anchor (its values fill the sliders; edits
  // apply to ALL selected). Recomputed only when the ids/notes change.
  const selected = useMemo(() => {
    const set = new Set(selectedIds);
    return notes.filter((n) => set.has(n.id));
  }, [notes, selectedIds]);
  const anchor = selected[0];
  const hasSel = selected.length > 0;

  // ── batch helpers: build a per-id update map → ONE applyNoteEdits (one undo step; the drag transaction
  //    coalesces the per-frame calls). Each note keeps its OTHER overrides (merge onto its own current). ──
  const editTransition = (key: keyof Required<NoteTransition>, value: number | undefined) => {
    const update: Record<string, Partial<Note>> = {};
    for (const n of selected) update[n.id] = { transition: { ...(n.transition ?? {}), [key]: value } };
    applyNoteEdits(trackId, segmentId, { update });
  };
  const editVibratoField = (key: keyof VibratoSpec, value: number) => {
    const update: Record<string, Partial<Note>> = {};
    // Only retune notes that ALREADY vibrate — a slider tweak must NOT silently seed a full audible vibrato on
    // a selected note that had none (use "Add vibrato" for that). The anchor has one (that's why we render).
    for (const n of selected) if (n.vibrato) update[n.id] = { vibrato: { ...n.vibrato, [key]: value } };
    applyNoteEdits(trackId, segmentId, { update });
  };
  const setVibrato = (spec: VibratoSpec | undefined) => {
    const update: Record<string, Partial<Note>> = {};
    // ADD (spec) seeds the default ONLY where a note lacks a vibrato — a note that already has a tuned vibrato
    // KEEPS it (never clobber another selected note's data). REMOVE (spec===undefined) clears all selected.
    for (const n of selected) update[n.id] = { vibrato: spec ? (n.vibrato ?? { ...spec }) : undefined };
    applyNoteEdits(trackId, segmentId, { update }); // depthCents≤0 → normalizeNote strips → absent (= remove)
  };
  // S58 per-note language override for the whole selection (ONE undo step); undefined = follow the track.
  const editNoteLang = (code: string | undefined) => {
    const update: Record<string, Partial<Note>> = {};
    for (const n of selected) update[n.id] = { lang: code };
    applyNoteEdits(trackId, segmentId, { update });
  };

  const eff = anchor ? effTransition(anchor, trackTransition) : trackTransition;
  const vib = anchor?.vibrato; // the anchor's vibrato (fills the sliders; edits apply to all selected)

  // ── Item-1 quality knobs: only the CHANGED keys live on vocalParams.sovits/.rvc (absent = contract
  //    default). A drag/toggle = one setVocalParams (one undo step). Asset-gated rows hide when the singer
  //    lacks the asset (mirrors the 翻唱 SoVITS/RVC node). ──
  const sv = (vocalParams.sovits ?? {}) as Partial<SovitsOptions>;
  const rv = (vocalParams.rvc ?? {}) as Partial<RvcOptions>;
  const svGet = <K extends keyof SovitsOptions>(k: K): SovitsOptions[K] => (sv[k] ?? SOVITS_DEFAULTS[k]) as SovitsOptions[K];
  const rvGet = <K extends keyof RvcOptions>(k: K): RvcOptions[K] => (rv[k] ?? RVC_DEFAULTS[k]) as RvcOptions[K];
  const setSv = (patch: Partial<SovitsOptions>) => setVocalParams(trackId, { sovits: { ...sv, ...patch } });
  const setRv = (patch: Partial<RvcOptions>) => setVocalParams(trackId, { rvc: { ...rv, ...patch } });
  const hasRetrieval = !!selectedVoice?.index_path; // cluster (SoVITS) / feature index (RVC) sibling asset
  const hasDiffusion = voiceHasDiffusion(selectedVoice);
  const diffusionOn = !!svGet("shallow_diffusion") && hasDiffusion;
  const ratioCfg = { min: 0, max: 1, step: 0.01, unit: "", bipolar: false };

  return (
    <div className="vocal-sidebar">
      {/* Two tabs (SynthV-style): 歌手 = singer + tone/quality; 调教 = pitch tuning. The Render action is a
          pinned footer, visible on both tabs. */}
      <div className="vsb-tabs">
        <button className={tab === "singer" ? "active" : ""} onClick={() => setTab("singer")}>{t("vocalEditor.sidebar.tabSinger")}</button>
        <button className={tab === "pitch" ? "active" : ""} onClick={() => setTab("pitch")}>{t("vocalEditor.sidebar.tabPitch")}</button>
      </div>
      <div className="vsb-body">
      {tab === "singer" ? (
      <>
      {/* ⓪ track · VOICE — ONE unified singer list (SoVITS + RVC); the picked model's TYPE drives the
          backend automatically. */}
      <div className="vsb-section">
        <div className="vsb-head">
          <span>{t("vocalEditor.sidebar.voice")}</span>
          {selectedVoice && <span className="vsb-backend-tag">{backendLabel(selectedVoice)}</span>}
        </div>
        {allVoices.length === 0 ? (
          <div className="voice-no-model">
            <span className="sep-no-model">{t("vocalEditor.sidebar.noVoiceModel")}</span>
            <button className="voice-manage-btn" onClick={() => toggleModelManager()}>
              {t("vocalEditor.sidebar.goImport")}
            </button>
          </div>
        ) : (
          <select
            className="sep-model-select"
            value={selectedVoice ? String(allVoices.indexOf(selectedVoice)) : ""}
            onChange={(e) => {
              const m = allVoices[Number(e.target.value)];
              if (m) pickVoice(m);
            }}
          >
            <option value="" disabled>{t("vocalEditor.sidebar.pickVoice")}</option>
            {allVoices.map((m, i) => (
              <option key={`${m.model_type}:${m.name}:${i}`} value={i}>
                {m.name} · {backendLabel(m)}
              </option>
            ))}
          </select>
        )}
        <Slider
          label={t("vocalEditor.sidebar.transpose")}
          value={vocalParams.transpose}
          cfg={{ min: -24, max: 24, step: 1, unit: "st", bipolar: true }}
          onChange={(v) => setVocalParams(trackId, { transpose: v })}
        />
        {/* ② 共振腔/formant track-level SCALAR (semitones), ADDED to the bottom formant lane at render. */}
        <Slider
          label={t("vocalEditor.sidebar.formant")}
          value={vocalParams.formant ?? 0}
          cfg={{ min: -12, max: 12, step: 1, unit: "st", bipolar: true }}
          overridden={(vocalParams.formant ?? 0) !== 0}
          resetTitle={t("vocalEditor.sidebar.formantTip")}
          onReset={() => setVocalParams(trackId, { formant: 0 })}
          onChange={(v) => setVocalParams(trackId, { formant: v })}
        />
        {/* M3 breath token: the lyric that means "audible inhale" (mapped to AP at render). Editable so a
            custom trigger never steals a glyph the user needs as a real lyric. */}
        <div className="vsb-inline">
          <label className="vsb-label" title={t("vocalEditor.sidebar.breathTokenTip")}>{t("vocalEditor.sidebar.breathToken")}</label>
          <input
            className="vsb-text"
            type="text"
            spellCheck={false}
            value={vocalParams.breathToken ?? "AP"}
            onChange={(e) => setVocalParams(trackId, { breathToken: e.target.value })}
          />
        </div>
        {/* S60-2 音域扩展: v1 recipe (shift into the singer's tested comfort zone, TD-PSOLA back).
            A no-op until the model carries a vocal_range record (资源管理器 → 测音域). Default ON. */}
        <div title={t("vocalEditor.sidebar.rangeExtendTip")}>
          <ToggleRow
            label={t("vocalEditor.sidebar.rangeExtend")}
            checked={vocalParams.rangeExtend !== false}
            onChange={(c) => setVocalParams(trackId, { rangeExtend: c })}
          />
        </div>
      </div>

      {/* ⓪.3 语言 (S58 §3.7) — lives on the SINGER tab (it configures WHAT is sung, not the pitch; §user).
          Track default (VocalTrackParams.langId — the whole track's G2P language) + a per-note override
          for the selection (Note.lang; absent = follow the track). Both feed the render per note. */}
      <div className="vsb-section">
        <div className="vsb-head"><span>{t("vocalEditor.sidebar.language")}</span></div>
        <div className="vsb-inline">
          <label className="vsb-label" title={t("vocalEditor.sidebar.trackLangTip")}>{t("vocalEditor.sidebar.trackLang")}</label>
          <select
            className="sep-model-select vsb-inline-select"
            value={vocalParams.langId}
            onChange={(e) => setVocalParams(trackId, { langId: Number(e.target.value) })}
          >
            {VOCAL_LANGUAGES.map((l) => (
              <option key={l.code} value={l.id}>{t(`langs.${l.code}`)} ({l.short})</option>
            ))}
          </select>
        </div>
        <div className="vsb-inline">
          <label className="vsb-label" title={t("vocalEditor.sidebar.noteLangTip")}>
            {t("vocalEditor.sidebar.noteLang")}
            {hasSel && selected.length > 1 ? ` ×${selected.length}` : ""}
          </label>
          {!hasSel ? (
            <span className="vsb-hint-inline">{t("vocalEditor.sidebar.selectNoteHint")}</span>
          ) : (
            <select
              className="sep-model-select vsb-inline-select"
              value={anchor?.lang ?? ""}
              onChange={(e) => editNoteLang(e.target.value || undefined)}
            >
              <option value="">
                {t("vocalEditor.sidebar.langFollow")} ({langById(vocalParams.langId).short})
              </option>
              {VOCAL_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>{t(`langs.${l.code}`)} ({l.short})</option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* ⓪.5 音质 (Item-1): the singer's quality knobs — SoVITS = cluster + shallow diffusion + NSF
          enhancer; RVC = feature index + protect (+ noise for both). Only changed keys are stored; asset-
          gated rows hide when the singer lacks the asset. auto_f0/loudness/f0_shift are force-off Rust-side. */}
      <div className="vsb-section">
        <div className="vsb-head">
          <span>{t("vocalEditor.sidebar.quality")}</span>
          {selectedVoice && <span className="vsb-backend-tag">{backendLabel(selectedVoice)}</span>}
        </div>
        {!selectedVoice ? (
          <div className="vsb-hint">{t("vocalEditor.sidebar.selectVoiceHint")}</div>
        ) : vocalParams.backend === "sovits" ? (
          <>
            <Slider label={t("vocalEditor.sidebar.q_noise")} value={svGet("noise_scale")} cfg={ratioCfg} onChange={(v) => setSv({ noise_scale: v })} />
            {hasRetrieval && (
              <Slider label={t("vocalEditor.sidebar.q_cluster")} value={svGet("cluster_ratio")} cfg={ratioCfg} onChange={(v) => setSv({ cluster_ratio: v })} />
            )}
            {hasDiffusion && (
              <ToggleRow
                label={t("vocalEditor.sidebar.q_diffusion")}
                checked={diffusionOn}
                onChange={(c) => setSv(c ? { shallow_diffusion: true } : { shallow_diffusion: false, only_diffusion: false })}
              />
            )}
            {diffusionOn && (
              <>
                <SelectRow label={t("vocalEditor.sidebar.q_sampler")} value={svGet("diffusion_method")} options={DIFFUSION_METHODS} onChange={(v) => setSv({ diffusion_method: v })} />
                <Slider label={t("vocalEditor.sidebar.q_kstep")} value={svGet("k_step")} cfg={{ min: 10, max: 1000, step: 10, unit: "", bipolar: false }} onChange={(v) => setSv({ k_step: v })} />
                <Slider label={t("vocalEditor.sidebar.q_speedup")} value={svGet("diffusion_speedup")} cfg={{ min: 1, max: 100, step: 1, unit: "×", bipolar: false }} onChange={(v) => setSv({ diffusion_speedup: v })} />
              </>
            )}
            {!diffusionOn && (
              <ToggleRow label={t("vocalEditor.sidebar.q_enhancer")} checked={!!svGet("nsf_enhance")} onChange={(c) => setSv({ nsf_enhance: c })} />
            )}
            {/* fine-tuned NSF-HiFiGAN vocoder — meaningful only on the mel→audio paths (shallow diffusion /
                enhancer). Backend already honors vocoder_name for the ② render (resolve_sovits_quality). */}
            {(diffusionOn || !!svGet("nsf_enhance")) && (
              <VocoderSelect
                value={svGet("vocoder_name") ?? null}
                lang={i18n.language}
                onChange={(v) => setSv({ vocoder_name: v })}
                rowClass="vsb-inline"
                selectClass="sep-model-select vsb-inline-select"
                labelClass="vsb-label"
              />
            )}
          </>
        ) : (
          <>
            <Slider label={t("vocalEditor.sidebar.q_noise")} value={rvGet("noise_scale")} cfg={ratioCfg} onChange={(v) => setRv({ noise_scale: v })} />
            {hasRetrieval && (
              <Slider label={t("vocalEditor.sidebar.q_index")} value={rvGet("index_ratio")} cfg={ratioCfg} onChange={(v) => setRv({ index_ratio: v })} />
            )}
            <Slider label={t("vocalEditor.sidebar.q_protect")} value={rvGet("protect")} cfg={{ min: 0, max: 0.5, step: 0.01, unit: "", bipolar: false }} onChange={(v) => setRv({ protect: v })} />
          </>
        )}
      </div>
      </>
      ) : (
      <>
      {/* ① selected-note transition override (glide / portamento between notes) */}
      <div className="vsb-section">
        <div className="vsb-head">
          <span>{t("vocalEditor.sidebar.noteTransition")}</span>
          {hasSel && selected.length > 1 && <span className="vsb-count">×{selected.length}</span>}
        </div>
        {!hasSel ? (
          <div className="vsb-hint">{t("vocalEditor.sidebar.selectNoteHint")}</div>
        ) : (
          TRANSITION_FIELDS.map((f) => (
            <Slider
              key={f.key}
              label={t(`vocalEditor.sidebar.tr_${f.key}`)}
              value={eff[f.key]}
              cfg={f}
              overridden={anchor?.transition?.[f.key] !== undefined}
              onReset={() => editTransition(f.key, undefined)}
              resetTitle={t("vocalEditor.sidebar.resetInherit")}
              onChange={(v) => editTransition(f.key, v)}
            />
          ))
        )}
      </div>

      {/* ② selected-note vibrato */}
      <div className="vsb-section">
        <div className="vsb-head">
          <span>{t("vocalEditor.sidebar.vibrato")}</span>
          {hasSel && selected.length > 1 && <span className="vsb-count">×{selected.length}</span>}
        </div>
        {!hasSel ? (
          <div className="vsb-hint">{t("vocalEditor.sidebar.selectNoteHint")}</div>
        ) : !vib ? (
          <button className="snap-toggle vsb-btn" onClick={() => setVibrato(DEFAULT_VIBRATO)}>
            + {t("vocalEditor.sidebar.addVibrato")}
          </button>
        ) : (
          <>
            <button className="snap-toggle vsb-btn vsb-btn-danger" onClick={() => setVibrato(undefined)}>
              {t("vocalEditor.sidebar.removeVibrato")}
            </button>
            {VIBRATO_FIELDS.map((f) => (
              <Slider
                key={f.key}
                label={t(`vocalEditor.sidebar.vib_${f.key}`)}
                value={vib[f.key]}
                cfg={f}
                onChange={(v) => editVibratoField(f.key, v)}
              />
            ))}
          </>
        )}
      </div>

      {/* ③ track default transition (the concrete base every note inherits). Each field has its OWN ↺ back to
          the factory default (DEFAULT_TRANSITION) — a single reset-all would force re-dragging the fields you
          wanted to keep (§user). ↺ lights only when the field differs from the factory default. */}
      <div className="vsb-section">
        <div className="vsb-head"><span>{t("vocalEditor.sidebar.trackTransition")}</span></div>
        {TRANSITION_FIELDS.map((f) => (
          <Slider
            key={f.key}
            label={t(`vocalEditor.sidebar.tr_${f.key}`)}
            value={trackTransition[f.key]}
            cfg={f}
            overridden={trackTransition[f.key] !== DEFAULT_TRANSITION[f.key]}
            onReset={() => setVocalParams(trackId, { transition: { ...trackTransition, [f.key]: DEFAULT_TRANSITION[f.key] } })}
            resetTitle={t("vocalEditor.sidebar.resetTrack")}
            onChange={(v) => setVocalParams(trackId, { transition: { ...trackTransition, [f.key]: v } })}
          />
        ))}
      </div>
      </>
      )}
      </div>
      {/* pinned footer — the Render action is needed regardless of tab (bake without playing / re-bake). */}
      <div className="vsb-foot">
        <button
          className="snap-toggle vsb-render"
          disabled={rendering || !selectedVoice || notes.length === 0}
          onClick={onRender}
        >
          {rendering ? t("vocalEditor.render.rendering") : t("vocalEditor.render.render")}
        </button>
      </div>
    </div>
  );
}

// ── one labeled fader row (VolumeFader = the gesture-bracketed fader; a drag = one undo step) ──
function Slider({ label, value, cfg, overridden, onReset, resetTitle, onChange }: {
  label: string;
  value: number;
  cfg: { min: number; max: number; step: number; unit: string; bipolar: boolean };
  overridden?: boolean;
  onReset?: () => void;
  resetTitle?: string;
  onChange: (v: number) => void;
}) {
  const showReset = onReset !== undefined;
  return (
    <div className="vsb-row">
      <div className="vsb-row-top">
        <label className={`vsb-label${overridden ? " ovr" : ""}`} title={overridden ? resetTitle : undefined}>{label}</label>
        <span className="vsb-val">{fmt(value, cfg.step)}{cfg.unit}</span>
      </div>
      <div className="vsb-row-bot">
        <VolumeFader
          value={value}
          min={cfg.min}
          max={cfg.max}
          step={cfg.step}
          width={showReset ? 176 : 200}
          fillFrom={cfg.bipolar ? "center" : "left"}
          onChange={onChange}
          onGestureStart={() => useHistoryStore.getState().beginTransaction()}
          onGestureEnd={() => useHistoryStore.getState().commitTransaction()}
          format={(v) => `${fmt(v, cfg.step)}${cfg.unit}`}
        />
        {showReset && (
          <button className="vsb-reset" disabled={!overridden} title={resetTitle} onClick={onReset}>↺</button>
        )}
      </div>
    </div>
  );
}

// ── a labeled checkbox row (a toggle = one setVocalParams = one undo step; no fader gesture) ──
function ToggleRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (c: boolean) => void }) {
  return (
    <div className="vsb-inline">
      <label className="vsb-label">{label}</label>
      <input type="checkbox" className="vsb-check" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </div>
  );
}

// ── a labeled <select> row (diffusion sampler pick) ──
function SelectRow({ label, value, options, onChange }: { label: string; value: string; options: readonly string[]; onChange: (v: string) => void }) {
  return (
    <div className="vsb-inline">
      <label className="vsb-label">{label}</label>
      <select className="sep-model-select vsb-inline-select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}
