import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
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
import { useVoiceModelStore, voiceHasDiffusion, voiceHasRangeRecord, vocalTrackSpeakerId, voiceSpeakerOptions } from "../../store/voice-models";
import { speakerRecordOf } from "../../lib/vocal/rangeBounds";
import { RangeBoundsEditor } from "../vocal/RangeBoundsEditor";
import { effTransition } from "../../lib/f0eval";
import { DEFAULT_CONSONANT_EMPHASIS_DB, DEFAULT_CONSONANT_VALLEY, DEFAULT_BREATH_TOKEN, DEFAULT_REST_TOKEN } from "../../lib/vocalNotes";
import { VOCAL_LANGUAGES, langById } from "../../lib/vocal/languages";
import { backendOf, backendLabel, pickVoiceForTrack } from "../../lib/vocal/voicePick";
import { DIFFUSION_METHODS, RVC_DEFAULTS, SOVITS_DEFAULTS } from "../../lib/workflow/voiceDefaults";
import { VocoderSelect } from "../workflow/nodes/VoiceModelPicker";
import "./VocalSidebar.css";
/** Default vibrato seeded by "Add vibrato" (depthCents>0 so normalizeNote keeps it). ⚠ startMs/ease are SMALL
 *  on purpose so vibrato is VISIBLE the instant it's added, even on a shorter (tail) note — the old SynthV-ish
 *  250 ms onset + 200 ms fades suppressed it entirely below ~2 beats (§user "尾音加不了颤音"). The onset delay
 *  is still a slider, so a swell-in can be dialed back in per note. */
const DEFAULT_VIBRATO = { depthCents: 100, freqHz: 5.5, phase: 0, startMs: 0, easeInMs: 80, easeOutMs: 120 };
const TRANSITION_FIELDS = [
    { key: "durLeftMs", min: 0, max: 1000, step: 1, unit: "ms", bipolar: false },
    { key: "durRightMs", min: 0, max: 1000, step: 1, unit: "ms", bipolar: false },
    { key: "depthLeftCents", min: -600, max: 600, step: 1, unit: "¢", bipolar: true },
    { key: "depthRightCents", min: -600, max: 600, step: 1, unit: "¢", bipolar: true },
    { key: "offsetMs", min: -500, max: 500, step: 1, unit: "ms", bipolar: true },
    { key: "openEdgeCents", min: 0, max: 600, step: 1, unit: "¢", bipolar: false },
];
const VIBRATO_FIELDS = [
    { key: "depthCents", min: 0, max: 1200, step: 1, unit: "¢", bipolar: false },
    { key: "freqHz", min: 0.5, max: 40, step: 0.1, unit: "Hz", bipolar: false },
    { key: "phase", min: -1, max: 1, step: 0.01, unit: "", bipolar: true },
    { key: "startMs", min: 0, max: 2000, step: 5, unit: "ms", bipolar: false },
    { key: "easeInMs", min: 0, max: 2000, step: 5, unit: "ms", bipolar: false },
    { key: "easeOutMs", min: 0, max: 2000, step: 5, unit: "ms", bipolar: false },
];
const fmt = (v, step) => (step >= 1 ? String(Math.round(v)) : step >= 0.1 ? v.toFixed(1) : v.toFixed(2));
export function VocalSidebar({ trackId, segmentId, notes, selectedIds, trackTransition, vocalParams, voiceModel, onRender, rendering }) {
    const { t, i18n } = useTranslation();
    // SynthV-style tabs: "singer" = voice + tone/quality; "pitch" = pitch tuning. Splits the (previously long)
    // single scroll into two focused panels; the Render action is a pinned footer visible on both.
    const [tab, setTab] = useState("singer");
    // ── S73b/c/d 自动调教:全部旋钮持久在 vocalParams,改动 → AutoTuneWatcher 静默重调教 →
    //    重渲染(一步 undo=Slider 手势事务)。S73d 起区块零按钮零异步:Retake 抽奖被确定性
    //    Take 旋钮替代;失败 UX(冷却自愈+缺模型对话框)全在 watcher。 ──
    const applyNoteEdits = useProjectStore((s) => s.applyNoteEdits);
    const setVocalParams = useProjectStore((s) => s.setVocalParams);
    const toggleModelManager = useAppStore((s) => s.toggleModelManager);
    const models = useVoiceModelStore((s) => s.models);
    // ONE unified singer list (SoVITS + RVC). Identity is (model_type, name) — same-name rvc/sovits pairs are
    // a standard workflow, so never key by name alone; the picked model's type auto-sets the backend.
    const allVoices = useMemo(() => [...models.sovits, ...models.rvc], [models]);
    const selectedVoice = useMemo(() => allVoices.find((m) => m.name === voiceModel && backendOf(m) === vocalParams.backend), [allVoices, voiceModel, vocalParams.backend]);
    // Pick a singer → the SHARED single pick path (voiceModel + backend, one undo step) — the track-header
    // singer popup uses the same function (S58, NO-dup).
    const pickVoice = (m) => pickVoiceForTrack(trackId, m);
    // S146e — which (model, speaker) record the 音域扩展栏's bound knobs edit.
    // ⛔ The speaker here is `vocalTrackSpeakerId` = the max-weight blend entry, which the user
    // never explicitly chose; `speakerLabel` is therefore mandatory, not decoration (S146e recon).
    const [boundsOpen, setBoundsOpen] = useState(false);
    const rangeRecordForTrack = useMemo(() => {
        if (!selectedVoice)
            return null;
        const speakerId = vocalTrackSpeakerId(vocalParams);
        const sp = speakerRecordOf(selectedVoice.config, speakerId);
        if (!sp)
            return null;
        const opts = voiceSpeakerOptions(selectedVoice);
        return {
            sp,
            name: selectedVoice.name,
            backend: backendOf(selectedVoice),
            speakerId,
            speakerLabel: opts.length > 1 ? opts.find((s) => s.id === speakerId)?.label : undefined,
        };
    }, [selectedVoice, vocalParams]);
    // The selection = the notes to edit; the FIRST is the display anchor (its values fill the sliders; edits
    // apply to ALL selected). Recomputed only when the ids/notes change.
    const selected = useMemo(() => {
        const set = new Set(selectedIds);
        return notes.filter((n) => set.has(n.id));
    }, [notes, selectedIds]);
    const anchor = selected[0];
    const hasSel = selected.length > 0;
    // ── batch helpers: build a per-id update map → ONE applyNoteEdits (one undo step; the drag transaction
    //    coalesces the per-frame calls). Each note keeps its OTHER overrides (merge onto its own current).
    //    ★S73 所有权:手动改 transition/vibrato = 用户接管调教 → 剥 autoTuned 标记(此后自动调教
    //    绕行这个音符;SynthV「用户设过值则自动过程跳过」同构)。
    //    ★S73d 多选=DELTA 编辑:滑杆位移作为增量加到【每个音符自己的当前值】上(各自钳位)——
    //    旧「锚值写给所有人」会把短音符塞进长音符的绝对值(轻推一下,某音符音高线爆炸,§user)。
    //    每帧 delta = 新值 − 锚当前值,写入后锚同步前进 → 逐帧增量在全员上等量累积;单选时
    //    锚+delta ≡ 绝对设值,行为不变。 ──
    const editTransition = (key, value) => {
        const update = {};
        if (value === undefined) {
            // 重置回继承:保持绝对语义(全员剥该字段)
            for (const n of selected)
                update[n.id] = { transition: { ...(n.transition ?? {}), [key]: undefined }, autoTuned: undefined };
        }
        else {
            const delta = value - effTransition(anchor, trackTransition)[key];
            for (const n of selected) {
                const cur = effTransition(n, trackTransition)[key];
                update[n.id] = { transition: { ...(n.transition ?? {}), [key]: cur + delta }, autoTuned: undefined };
            }
        }
        applyNoteEdits(trackId, segmentId, { update });
    };
    const editVibratoField = (key, value) => {
        const update = {};
        // Only retune notes that ALREADY vibrate — a slider tweak must NOT silently seed a full audible vibrato on
        // a selected note that had none (use "Add vibrato" for that). The anchor has one (that's why we render).
        const anchorCur = anchor?.vibrato?.[key];
        const delta = anchorCur !== undefined ? value - anchorCur : 0;
        for (const n of selected) {
            if (!n.vibrato)
                continue;
            let next = n.vibrato[key] + delta;
            // ★depth 特判(S73d 复核 CONFIRMED):多选 delta 把浅颤音压过 0 → normalizeNote 连整个
            //   vibrato 容器(freq/phase/…)一起剥成 absent,后续帧 `!n.vibrato` 跳过=拖回也救不回。
            //   垫 0.01¢ 保容器;只有滑杆本身拖到 0(全员归零=删除语义)才放行剥除。
            if (key === "depthCents" && value > 0)
                next = Math.max(0.01, next);
            update[n.id] = { vibrato: { ...n.vibrato, [key]: next }, autoTuned: undefined };
        }
        applyNoteEdits(trackId, segmentId, { update });
    };
    const setVibrato = (spec) => {
        const update = {};
        // ADD (spec) seeds the default ONLY where a note lacks a vibrato — a note that already has a tuned vibrato
        // KEEPS it (never clobber another selected note's data). REMOVE (spec===undefined) clears all selected.
        for (const n of selected)
            update[n.id] = { vibrato: spec ? (n.vibrato ?? { ...spec }) : undefined, autoTuned: undefined };
        applyNoteEdits(trackId, segmentId, { update }); // depthCents≤0 → normalizeNote strips → absent (= remove)
    };
    // S58 per-note language override for the whole selection (ONE undo step); undefined = follow the track.
    const editNoteLang = (code) => {
        const update = {};
        for (const n of selected)
            update[n.id] = { lang: code };
        applyNoteEdits(trackId, segmentId, { update });
    };
    const eff = anchor ? effTransition(anchor, trackTransition) : trackTransition;
    const vib = anchor?.vibrato; // the anchor's vibrato (fills the sliders; edits apply to all selected)
    // ── Item-1 quality knobs: only the CHANGED keys live on vocalParams.sovits/.rvc (absent = contract
    //    default). A drag/toggle = one setVocalParams (one undo step). Asset-gated rows hide when the singer
    //    lacks the asset (mirrors the 翻唱 SoVITS/RVC node). ──
    const sv = (vocalParams.sovits ?? {});
    const rv = (vocalParams.rvc ?? {});
    const svGet = (k) => (sv[k] ?? SOVITS_DEFAULTS[k]);
    const rvGet = (k) => (rv[k] ?? RVC_DEFAULTS[k]);
    const setSv = (patch) => setVocalParams(trackId, { sovits: { ...sv, ...patch } });
    const setRv = (patch) => setVocalParams(trackId, { rvc: { ...rv, ...patch } });
    const hasRetrieval = !!selectedVoice?.index_path; // cluster (SoVITS) / feature index (RVC) sibling asset
    const hasDiffusion = voiceHasDiffusion(selectedVoice);
    const diffusionOn = !!svGet("shallow_diffusion") && hasDiffusion;
    const ratioCfg = { min: 0, max: 1, step: 0.01, unit: "", bipolar: false };
    return (_jsxs("div", { className: "vocal-sidebar", children: [_jsxs("div", { className: "vsb-tabs", children: [_jsx("button", { className: tab === "singer" ? "active" : "", onClick: () => setTab("singer"), children: t("vocalEditor.sidebar.tabSinger") }), _jsx("button", { className: tab === "pitch" ? "active" : "", onClick: () => setTab("pitch"), children: t("vocalEditor.sidebar.tabPitch") })] }), _jsx("div", { className: "vsb-body", children: tab === "singer" ? (_jsxs(_Fragment, { children: [_jsxs("div", { className: "vsb-section", children: [_jsxs("div", { className: "vsb-head", children: [_jsx("span", { children: t("vocalEditor.sidebar.voice") }), selectedVoice && _jsx("span", { className: "vsb-backend-tag", children: backendLabel(selectedVoice) })] }), allVoices.length === 0 ? (_jsxs("div", { className: "voice-no-model", children: [_jsx("span", { className: "sep-no-model", children: t("vocalEditor.sidebar.noVoiceModel") }), _jsx("button", { className: "voice-manage-btn", onClick: () => toggleModelManager(), children: t("vocalEditor.sidebar.goImport") })] })) : (_jsxs("select", { className: "sep-model-select", value: selectedVoice ? String(allVoices.indexOf(selectedVoice)) : "", onChange: (e) => {
                                        const m = allVoices[Number(e.target.value)];
                                        if (m)
                                            pickVoice(m);
                                    }, children: [_jsx("option", { value: "", disabled: true, children: t("vocalEditor.sidebar.pickVoice") }), allVoices.map((m, i) => (_jsxs("option", { value: i, children: [m.name, " \u00B7 ", backendLabel(m)] }, `${m.model_type}:${m.name}:${i}`)))] })), _jsx(Slider, { label: t("vocalEditor.sidebar.transpose"), value: vocalParams.transpose, cfg: { min: -24, max: 24, step: 1, unit: "st", bipolar: true }, onChange: (v) => setVocalParams(trackId, { transpose: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.formant"), value: vocalParams.formant ?? 0, cfg: { min: -12, max: 12, step: 1, unit: "st", bipolar: true }, overridden: (vocalParams.formant ?? 0) !== 0, resetTitle: t("vocalEditor.sidebar.formantTip"), onReset: () => setVocalParams(trackId, { formant: 0 }), onChange: (v) => setVocalParams(trackId, { formant: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.consonant"), value: vocalParams.consonantEmphasis ?? DEFAULT_CONSONANT_EMPHASIS_DB, cfg: { min: 0, max: 6, step: 0.5, unit: "dB", bipolar: false }, overridden: (vocalParams.consonantEmphasis ?? DEFAULT_CONSONANT_EMPHASIS_DB) !== DEFAULT_CONSONANT_EMPHASIS_DB, resetTitle: t("vocalEditor.sidebar.consonantTip"), onReset: () => setVocalParams(trackId, { consonantEmphasis: DEFAULT_CONSONANT_EMPHASIS_DB }), onChange: (v) => setVocalParams(trackId, { consonantEmphasis: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.consonantValley"), value: vocalParams.consonantValley ?? DEFAULT_CONSONANT_VALLEY, cfg: { min: 0, max: 2, step: 0.1, unit: "×", bipolar: false }, overridden: (vocalParams.consonantValley ?? DEFAULT_CONSONANT_VALLEY) !== DEFAULT_CONSONANT_VALLEY, resetTitle: t("vocalEditor.sidebar.consonantValleyTip"), onReset: () => setVocalParams(trackId, { consonantValley: DEFAULT_CONSONANT_VALLEY }), onChange: (v) => setVocalParams(trackId, { consonantValley: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.voiceRealism"), value: vocalParams.voiceRealism ?? 0, cfg: { min: 0, max: 15, step: 0.5, unit: "%", bipolar: false }, overridden: (vocalParams.voiceRealism ?? 0) !== 0, resetTitle: t("vocalEditor.sidebar.voiceRealismTip"), onReset: () => setVocalParams(trackId, { voiceRealism: 0 }), onChange: (v) => setVocalParams(trackId, { voiceRealism: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.formantJitter"), value: vocalParams.formantJitter ?? 0, cfg: { min: 0, max: 8, step: 0.5, unit: "%", bipolar: false }, overridden: (vocalParams.formantJitter ?? 0) !== 0, resetTitle: t("vocalEditor.sidebar.formantJitterTip"), onReset: () => setVocalParams(trackId, { formantJitter: 0 }), onChange: (v) => setVocalParams(trackId, { formantJitter: v }) }), _jsx("div", { title: t("vocalEditor.sidebar.vowelClarityTip"), children: _jsx(ToggleRow, { label: t("vocalEditor.sidebar.vowelClarity"), checked: vocalParams.vowelClarity !== false, onChange: (c) => setVocalParams(trackId, { vowelClarity: c }) }) }), _jsx("div", { title: t("vocalEditor.sidebar.consonantPrerollTip"), children: _jsx(ToggleRow, { label: t("vocalEditor.sidebar.consonantPreroll"), checked: vocalParams.consonantPreroll !== false, onChange: (c) => setVocalParams(trackId, { consonantPreroll: c }) }) }), _jsx("div", { title: t("vocalEditor.sidebar.breathLayerTip"), children: _jsx(ToggleRow, { label: t("vocalEditor.sidebar.breathLayer"), checked: vocalParams.breathLayer !== false, onChange: (c) => setVocalParams(trackId, { breathLayer: c }) }) }), _jsxs("div", { className: "vsb-inline", children: [_jsx("label", { className: "vsb-label", title: t("vocalEditor.sidebar.breathTokenTip"), children: t("vocalEditor.sidebar.breathToken") }), _jsx("input", { className: "vsb-text", type: "text", spellCheck: false, value: vocalParams.breathToken ?? DEFAULT_BREATH_TOKEN, onChange: (e) => setVocalParams(trackId, { breathToken: e.target.value }) })] }), _jsxs("div", { className: "vsb-inline", children: [_jsx("label", { className: "vsb-label", title: t("vocalEditor.sidebar.restTokenTip"), children: t("vocalEditor.sidebar.restToken") }), _jsx("input", { className: "vsb-text", type: "text", spellCheck: false, value: vocalParams.restToken ?? DEFAULT_REST_TOKEN, onChange: (e) => setVocalParams(trackId, { restToken: e.target.value }) })] }), voiceHasRangeRecord(selectedVoice, vocalTrackSpeakerId(vocalParams)) && (_jsxs(_Fragment, { children: [_jsx("div", { title: t("vocalEditor.sidebar.rangeExtendTip"), children: _jsx(ToggleRow, { label: t("vocalEditor.sidebar.rangeExtend"), checked: vocalParams.rangeExtend !== false, onChange: (c) => setVocalParams(trackId, { rangeExtend: c }) }) }), vocalParams.rangeExtend !== false && (_jsx(Slider, { label: t("vocalEditor.sidebar.rangeFormant"), tip: t("vocalEditor.sidebar.rangeFormantTip"), value: (vocalParams.backend === "rvc"
                                                ? vocalParams.rvc?.range_formant_follow
                                                : vocalParams.sovits?.range_formant_follow) ?? 0, cfg: { min: 0, max: 1, step: 0.01, unit: "", bipolar: false }, onChange: (v) => setVocalParams(trackId, {
                                                sovits: { ...(vocalParams.sovits ?? {}), range_formant_follow: v },
                                                rvc: { ...(vocalParams.rvc ?? {}), range_formant_follow: v },
                                            }) })), vocalParams.rangeExtend !== false && rangeRecordForTrack && (_jsx("div", { className: "vsb-inline vsb-range-bounds", children: !boundsOpen ? (_jsx("button", { className: "rm-range-btn", title: t("vocalEditor.sidebar.rangeBoundsTip"), onClick: () => setBoundsOpen(true), children: t("vocalEditor.sidebar.rangeBounds") })) : (_jsx(RangeBoundsEditor, { sp: rangeRecordForTrack.sp, modelName: rangeRecordForTrack.name, backend: rangeRecordForTrack.backend, speakerId: rangeRecordForTrack.speakerId, speakerLabel: rangeRecordForTrack.speakerLabel, lang: i18n.language, onClose: () => setBoundsOpen(false) })) }))] }))] }), _jsxs("div", { className: "vsb-section", children: [_jsx("div", { className: "vsb-head", children: _jsx("span", { children: t("vocalEditor.sidebar.language") }) }), _jsxs("div", { className: "vsb-inline", children: [_jsx("label", { className: "vsb-label", title: t("vocalEditor.sidebar.trackLangTip"), children: t("vocalEditor.sidebar.trackLang") }), _jsx("select", { className: "sep-model-select vsb-inline-select", value: vocalParams.langId, onChange: (e) => setVocalParams(trackId, { langId: Number(e.target.value) }), children: VOCAL_LANGUAGES.map((l) => (_jsxs("option", { value: l.id, children: [t(`langs.${l.code}`), " (", l.short, ")"] }, l.code))) })] }), _jsxs("div", { className: "vsb-inline", children: [_jsxs("label", { className: "vsb-label", title: t("vocalEditor.sidebar.noteLangTip"), children: [t("vocalEditor.sidebar.noteLang"), hasSel && selected.length > 1 ? ` ×${selected.length}` : ""] }), !hasSel ? (_jsx("span", { className: "vsb-hint-inline", children: t("vocalEditor.sidebar.selectNoteHint") })) : (_jsxs("select", { className: "sep-model-select vsb-inline-select", value: anchor?.lang ?? "", onChange: (e) => editNoteLang(e.target.value || undefined), children: [_jsxs("option", { value: "", children: [t("vocalEditor.sidebar.langFollow"), " (", langById(vocalParams.langId).short, ")"] }), VOCAL_LANGUAGES.map((l) => (_jsxs("option", { value: l.code, children: [t(`langs.${l.code}`), " (", l.short, ")"] }, l.code)))] }))] }), (langById(vocalParams.langId).code === "en" ||
                                    !!vocalParams.phonemeSet ||
                                    selected.some((n) => (n.lang ?? langById(vocalParams.langId).code) === "en")) && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "vsb-inline", children: [_jsx("label", { className: "vsb-label", title: t("vocalEditor.sidebar.phonemeSetTip"), children: t("vocalEditor.sidebar.phonemeSet") }), _jsx("select", { className: "sep-model-select vsb-inline-select", value: vocalParams.phonemeSet ?? "words", onChange: (e) => setVocalParams(trackId, {
                                                        phonemeSet: e.target.value === "words" ? undefined : e.target.value,
                                                    }), children: ["words", "arpasing", "xsampa", "vccv"].map((k) => (_jsx("option", { value: k, children: t(`vocalEditor.sidebar.phonemeSet_${k}`) }, k))) })] }), _jsx("div", { className: "vsb-hint", title: t("vocalEditor.sidebar.phonemeHintTip"), children: vocalParams.phonemeSet
                                                ? t("vocalEditor.sidebar.phonemeSetHint")
                                                : t("vocalEditor.sidebar.phonemeHint") })] })), (langById(vocalParams.langId).code === "es" ||
                                    !!vocalParams.esDialect ||
                                    selected.some((n) => (n.lang ?? langById(vocalParams.langId).code) === "es")) && (_jsxs("div", { className: "vsb-inline", children: [_jsx("label", { className: "vsb-label", title: t("vocalEditor.sidebar.esDialectTip"), children: t("vocalEditor.sidebar.esDialect") }), _jsx("select", { className: "sep-model-select vsb-inline-select", value: vocalParams.esDialect ?? "dictionary", onChange: (e) => setVocalParams(trackId, {
                                                esDialect: e.target.value === "dictionary" ? undefined : e.target.value,
                                            }), children: ["dictionary", "castilian", "castilian_yeista", "latam", "andean"].map((k) => (_jsx("option", { value: k, children: t(`vocalEditor.sidebar.esDialect_${k}`) }, k))) })] }))] }), _jsxs("div", { className: "vsb-section", children: [_jsxs("div", { className: "vsb-head", children: [_jsx("span", { children: t("vocalEditor.sidebar.quality") }), selectedVoice && _jsx("span", { className: "vsb-backend-tag", children: backendLabel(selectedVoice) })] }), !selectedVoice ? (_jsx("div", { className: "vsb-hint", children: t("vocalEditor.sidebar.selectVoiceHint") })) : vocalParams.backend === "sovits" ? (_jsxs(_Fragment, { children: [_jsx(Slider, { label: t("vocalEditor.sidebar.q_noise"), value: svGet("noise_scale"), cfg: ratioCfg, onChange: (v) => setSv({ noise_scale: v }) }), hasRetrieval && (_jsx(Slider, { label: t("vocalEditor.sidebar.q_cluster"), value: svGet("cluster_ratio"), cfg: ratioCfg, onChange: (v) => setSv({ cluster_ratio: v }) })), hasDiffusion && (_jsx(ToggleRow, { label: t("vocalEditor.sidebar.q_diffusion"), checked: diffusionOn, onChange: (c) => setSv(c ? { shallow_diffusion: true } : { shallow_diffusion: false, only_diffusion: false }) })), diffusionOn && (_jsxs(_Fragment, { children: [_jsx(SelectRow, { label: t("vocalEditor.sidebar.q_sampler"), value: svGet("diffusion_method"), options: DIFFUSION_METHODS, onChange: (v) => setSv({ diffusion_method: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.q_kstep"), value: svGet("k_step"), cfg: { min: 10, max: 1000, step: 10, unit: "", bipolar: false }, onChange: (v) => setSv({ k_step: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.q_speedup"), value: svGet("diffusion_speedup"), cfg: { min: 1, max: 100, step: 1, unit: "×", bipolar: false }, onChange: (v) => setSv({ diffusion_speedup: v }) })] })), !diffusionOn && (_jsxs("div", { title: t("vocalEditor.sidebar.q_enhancerTip"), children: [_jsx(ToggleRow, { label: t("vocalEditor.sidebar.q_enhancer"), checked: !!svGet("nsf_enhance"), onChange: (c) => setSv({ nsf_enhance: c }) }), !!svGet("nsf_enhance") && (_jsx("div", { className: "vsb-note", children: t("vocalEditor.sidebar.q_enhancerCost") }))] })), (diffusionOn || !!svGet("nsf_enhance")) && (_jsx(VocoderSelect, { value: svGet("vocoder_name") ?? null, lang: i18n.language, onChange: (v) => setSv({ vocoder_name: v }), rowClass: "vsb-inline", selectClass: "sep-model-select vsb-inline-select", labelClass: "vsb-label" }))] })) : (_jsxs(_Fragment, { children: [_jsx(Slider, { label: t("vocalEditor.sidebar.q_noise"), value: rvGet("noise_scale"), cfg: ratioCfg, onChange: (v) => setRv({ noise_scale: v }) }), hasRetrieval && (_jsx(Slider, { label: t("vocalEditor.sidebar.q_index"), value: rvGet("index_ratio"), cfg: ratioCfg, onChange: (v) => setRv({ index_ratio: v }) })), _jsx(Slider, { label: t("vocalEditor.sidebar.q_protect"), value: rvGet("protect"), cfg: { min: 0, max: 0.5, step: 0.01, unit: "", bipolar: false }, onChange: (v) => setRv({ protect: v }) })] }))] })] })) : (_jsxs(_Fragment, { children: [_jsxs("div", { className: "vsb-section", children: [_jsxs("div", { className: "vsb-head", children: [_jsx("span", { children: t("vocalEditor.sidebar.autotune") }), hasSel && selected.length > 1 && _jsxs("span", { className: "vsb-count", children: ["\u00D7", selected.length] })] }), _jsx("div", { title: t("vocalEditor.sidebar.autotuneFollowTip"), children: _jsx(ToggleRow, { label: t("vocalEditor.sidebar.autotuneFollow"), checked: vocalParams.autoTuneFollow !== false, onChange: (c) => setVocalParams(trackId, { autoTuneFollow: c }) }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.autotuneRigid"), tip: t("vocalEditor.sidebar.autotuneRigidTip"), value: Math.round((1 - (vocalParams.autoTuneVib ?? 1)) * 100), cfg: { min: -100, max: 100, step: 1, unit: "%", bipolar: true }, onChange: (r) => setVocalParams(trackId, { autoTuneVib: 1 - r / 100 }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.autotuneExpr"), tip: t("vocalEditor.sidebar.autotuneExprTip"), value: vocalParams.autoTuneExpr ?? 2, cfg: { min: 0, max: 4, step: 0.01, unit: "×", bipolar: false }, onChange: (v) => setVocalParams(trackId, { autoTuneExpr: v }) }), _jsx(Slider, { label: t("vocalEditor.sidebar.autotuneTake"), tip: t("vocalEditor.sidebar.autotuneTakeTip"), value: vocalParams.autoTuneTake ?? 0, cfg: { min: 0, max: 99, step: 1, unit: "", bipolar: false }, onChange: (v) => setVocalParams(trackId, { autoTuneTake: Math.round(v) }) })] }), _jsxs("div", { className: "vsb-section", children: [_jsxs("div", { className: "vsb-head", children: [_jsx("span", { children: t("vocalEditor.sidebar.noteTransition") }), hasSel && selected.length > 1 && _jsxs("span", { className: "vsb-count", children: ["\u00D7", selected.length] })] }), !hasSel ? (_jsx("div", { className: "vsb-hint", children: t("vocalEditor.sidebar.selectNoteHint") })) : (TRANSITION_FIELDS.map((f) => (_jsx(Slider, { label: t(`vocalEditor.sidebar.tr_${f.key}`), value: eff[f.key], cfg: f, overridden: anchor?.transition?.[f.key] !== undefined, onReset: () => editTransition(f.key, undefined), resetTitle: t("vocalEditor.sidebar.resetInherit"), onChange: (v) => editTransition(f.key, v) }, f.key))))] }), _jsxs("div", { className: "vsb-section", children: [_jsxs("div", { className: "vsb-head", children: [_jsx("span", { children: t("vocalEditor.sidebar.vibrato") }), hasSel && selected.length > 1 && _jsxs("span", { className: "vsb-count", children: ["\u00D7", selected.length] })] }), !hasSel ? (_jsx("div", { className: "vsb-hint", children: t("vocalEditor.sidebar.selectNoteHint") })) : !vib ? (_jsxs("button", { className: "snap-toggle vsb-btn", onClick: () => setVibrato(DEFAULT_VIBRATO), children: ["+ ", t("vocalEditor.sidebar.addVibrato")] })) : (_jsxs(_Fragment, { children: [_jsx("button", { className: "snap-toggle vsb-btn vsb-btn-danger", onClick: () => setVibrato(undefined), children: t("vocalEditor.sidebar.removeVibrato") }), VIBRATO_FIELDS.map((f) => (_jsx(Slider, { label: t(`vocalEditor.sidebar.vib_${f.key}`), value: vib[f.key], cfg: f, onChange: (v) => editVibratoField(f.key, v) }, f.key)))] }))] })] })) }), _jsx("div", { className: "vsb-foot", children: _jsx("button", { className: "snap-toggle vsb-render", disabled: rendering || !selectedVoice || notes.length === 0, onClick: onRender, children: rendering ? t("vocalEditor.render.rendering") : t("vocalEditor.render.render") }) })] }));
}
// ── one labeled fader row (VolumeFader = the gesture-bracketed fader; a drag = one undo step) ──
function Slider({ label, tip, value, cfg, overridden, onReset, resetTitle, onChange }) {
    const showReset = onReset !== undefined;
    return (_jsxs("div", { className: "vsb-row", children: [_jsxs("div", { className: "vsb-row-top", children: [_jsx("label", { className: `vsb-label${overridden ? " ovr" : ""}`, title: overridden ? resetTitle : tip, children: label }), _jsxs("span", { className: "vsb-val", children: [fmt(value, cfg.step), cfg.unit] })] }), _jsxs("div", { className: "vsb-row-bot", children: [_jsx(VolumeFader, { value: value, min: cfg.min, max: cfg.max, step: cfg.step, width: showReset ? 176 : 200, fillFrom: cfg.bipolar ? "center" : "left", onChange: onChange, onGestureStart: () => useHistoryStore.getState().beginTransaction(), onGestureEnd: () => useHistoryStore.getState().commitTransaction(), format: (v) => `${fmt(v, cfg.step)}${cfg.unit}`, tip: tip }), showReset && (_jsx("button", { className: "vsb-reset", disabled: !overridden, title: resetTitle, onClick: onReset, children: "\u21BA" }))] })] }));
}
// ── a labeled checkbox row (a toggle = one setVocalParams = one undo step; no fader gesture) ──
function ToggleRow({ label, checked, onChange }) {
    return (_jsxs("div", { className: "vsb-inline", children: [_jsx("label", { className: "vsb-label", children: label }), _jsx("input", { type: "checkbox", className: "vsb-check", checked: checked, onChange: (e) => onChange(e.target.checked) })] }));
}
// ── a labeled <select> row (diffusion sampler pick) ──
function SelectRow({ label, value, options, onChange }) {
    return (_jsxs("div", { className: "vsb-inline", children: [_jsx("label", { className: "vsb-label", children: label }), _jsx("select", { className: "sep-model-select vsb-inline-select", value: value, onChange: (e) => onChange(e.target.value), children: options.map((o) => _jsx("option", { value: o, children: o }, o)) })] }));
}
