/**
 * Full-screen training page (S37) — four stages: 数据 → 对象 → 参数 → 运行.
 * Covers the DAW (which stays mounted) as an absolute overlay inside app-content.
 * Training itself is fully backend-driven; this page is a projection of the
 * training store (event-fed) and may be closed/reopened at any time mid-run.
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { exists, readFile } from "@tauri-apps/plugin-fs";
import { useAppStore } from "../../store/app";
import {
  diffPoolReady,
  trainingDataOk,
  backendSupportsMultiSpeaker,
  setupTrainingListeners,
  useTrainingStore,
  type CkptInfo,
  type DatasetFile,
  type WorkspaceInfo,
} from "../../store/training";
import {
  useVoiceModelStore,
  voiceFeatureDim,
  voiceVersionBadge,
  type VoiceModelEntry,
} from "../../store/voice-models";
import { AUDIO_EXT_RE, AUDIO_EXTENSIONS } from "../../lib/constants";
import { backendErrorMessage, isBusyError, isCancelError } from "../../lib/backendError";
import { runCandidateRangeTest, midiName } from "../../lib/vocal/rangeTest";
import { Dropdown } from "../common/Dropdown";
import { t18 } from "../../lib/models/msst-catalog";
import { preview } from "../common/previewPlayer";
import { Scrubber } from "../common/Scrubber";
import { LossChart, type LossChartHandle } from "./LossChart";
import "./TrainingPage.css";

/** Preprocessing stage sequence per backend (stage names come from the sidecar
 *  protocol; these arrays only order/tick the checklist display). */
const STAGE_ORDERS: Record<string, string[]> = {
  // S41: augment + aug_check always emit (an instant "skipped" tick when
  // copies=0), and the sovits/diff filelist stage moved AFTER extract/gate
  // (the aug quality gate must finish before the filelists are written)
  rvc: ["import", "slice", "augment", "f0", "feature", "aug_check", "index", "filelist", "train_prep"],
  sovits: ["import", "slice", "augment", "extract", "aug_check", "filelist", "index", "train_prep"],
  sovits_diff: ["import", "slice", "augment", "extract", "aug_check", "filelist", "diff_prep", "train_prep"],
  vocoder: ["import", "slice", "augment", "process", "aug_check", "filelist", "train_prep"],
};

function fmtDur(totalSecs: number): string {
  const s = Math.max(0, Math.floor(totalSecs));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}

export function TrainingPage() {
  const { t } = useTranslation();
  const closePage = useAppStore((s) => s.toggleTrainingPage);
  const { wizard, setWizard, dataset, speakerGroups, snapshot, refresh, config, diffWsInfo } =
    useTrainingStore();
  const [dropActive, setDropActive] = useState(false);

  useEffect(() => {
    void setupTrainingListeners();
    void refresh();
  }, [refresh]);

  // single fetch site for the diff host's workspace info (S41 共享池模式):
  // DataStep/ParamsStep/RunStep all consume the store copy via diffPoolReady
  useEffect(() => {
    const name = config.modelName.trim();
    if (config.backend !== "sovits_diff" || !name) {
      useTrainingStore.getState().setDiffWsInfo(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const info = await invoke<WorkspaceInfo>("get_training_workspace_info", { name });
        if (!cancelled) useTrainingStore.getState().setDiffWsInfo(info);
      } catch {
        if (!cancelled) useTrainingStore.getState().setDiffWsInfo(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [config.backend, config.modelName]);

  const running = snapshot.state === "starting" || snapshot.state === "running";

  // OS drag-drop: the webview event is global, so the Arrangement timeline (which
  // stays mounted under this page) short-circuits while this page is open and we
  // take the drop here as dataset import. Registered ONCE (reads live state via
  // getState, like Arrangement) — addFiles dedupes, so a StrictMode double-mount
  // is harmless. NB Tauri's "over" payload has NO `paths` (only enter/drop do).
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    // enter-time decision (is this an audio drag we accept?) — reused on `over`
    // and `drop`, which is why it lives in the effect closure, not React state.
    let dragAccept = false;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const p = event.payload;
        const liveNow = () => {
          const s = useTrainingStore.getState().snapshot.state;
          return s === "starting" || s === "running";
        };
        // ①c: with ≥2 singers a drop lands on the singer CARD under the cursor,
        // else the FIRST singer (the fallback) — returning that id means the
        // hover highlight always shows exactly where the drop will land, so a
        // fallback-to-first is never a surprise. Hit-test by card GEOMETRY
        // (getBoundingClientRect), NOT elementFromPoint — the full-screen drop
        // overlay sits on top of the cards, so elementFromPoint would always
        // return the overlay (and it isn't torn down synchronously by
        // setDropActive(false)). Tauri gives a PHYSICAL-pixel position; rects
        // are CSS px, so divide by DPR.
        const hitTestSpeaker = (position?: { x: number; y: number }): string | null => {
          const st = useTrainingStore.getState();
          if (!backendSupportsMultiSpeaker(st.config.backend) || st.speakerGroups.length <= 1) {
            return null;
          }
          if (position) {
            const dpr = window.devicePixelRatio || 1;
            const x = position.x / dpr;
            const y = position.y / dpr;
            const cards = document.querySelectorAll<HTMLElement>("[data-spk-id]");
            for (const card of cards) {
              const r = card.getBoundingClientRect();
              if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) {
                return card.getAttribute("data-spk-id");
              }
            }
          }
          return st.speakerGroups[0]?.id ?? null; // fallback: the first singer
        };
        const setHover = (id: string | null) => {
          const st = useTrainingStore.getState();
          if (st.dragOverSpeakerId !== id) st.setDragOverSpeakerId(id);
        };
        if (p.type === "enter") {
          // don't invite a drop we'll refuse: adding to the dataset only affects
          // the NEXT run, so while one is live we accept nothing (matches the
          // Arrangement convention: no affordance for a drop that won't import)
          dragAccept = !liveNow() && p.paths.some((pp) => AUDIO_EXT_RE.test(pp));
          setDropActive(dragAccept);
          setHover(dragAccept ? hitTestSpeaker(p.position) : null);
        } else if (p.type === "over") {
          // 'over' carries no paths — reuse the enter-time accept decision
          setHover(dragAccept ? hitTestSpeaker(p.position) : null);
        } else if (p.type === "leave") {
          dragAccept = false;
          setDropActive(false);
          setHover(null);
        } else if (p.type === "drop") {
          setDropActive(false);
          const target = dragAccept ? hitTestSpeaker(p.position) : null;
          dragAccept = false;
          setHover(null);
          if (liveNow()) return;
          const audio = p.paths.filter((pp) => AUDIO_EXT_RE.test(pp));
          if (audio.length === 0) return;
          const st = useTrainingStore.getState();
          // ①c: SoVITS/RVC data is the singer list — a drop lands on the card under
          // the cursor (or the first singer if not over a card, so files are
          // never lost); diff/vocoder use the flat dataset.
          if (backendSupportsMultiSpeaker(st.config.backend)) {
            const gid = target ?? st.speakerGroups[0]?.id;
            if (gid) void st.addSpeakerFiles(gid, audio);
          } else {
            void st.addFiles(audio);
          }
          st.setWizard(2); // the data page (step 2 since the S41 order swap)
        }
      })
      .then((u) => {
        if (cancelled) u();
        else unlisten = u;
      });
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, []);

  // ①c: when the training object changes, move the imported file list between
  // the flat dataset (rvc/vocoder/diff) and the first singer (sovits) so a
  // switch never leaves already-imported files stranded/invisible.
  const prevBackendRef = useRef(config.backend);
  useEffect(() => {
    const prev = prevBackendRef.current;
    if (prev !== config.backend) {
      prevBackendRef.current = config.backend;
      useTrainingStore.getState().migrateOnBackendSwitch(prev, config.backend);
    }
  }, [config.backend]);

  // S41 order: 1=训练对象 2=数据 3=参数 4=运行 (the object decides whether
  // data is even needed, so it comes first; diff with a reusable shared pool
  // skips step 2). Steps 1/2 are always reachable — every ACTION is guarded
  // downstream (step gating here, start button, Rust validation).
  const diffPool = diffPoolReady(config.backend, diffWsInfo);
  const step3Ok = trainingDataOk(config.backend, dataset, speakerGroups, diffPool);
  const step4Ok = step3Ok || running || snapshot.state !== "idle";
  const stepOk = [true, true, step3Ok, step4Ok];

  // 防逃课 invariant (S41 user report): whatever path INVALIDATES the current
  // step (清空结果 with no data, backend switched away from diff, ...) bounces
  // back to step 1 instead of stranding the user on a locked stage.
  useEffect(() => {
    if ((wizard === 3 && !step3Ok) || (wizard === 4 && !step4Ok)) setWizard(1);
  }, [wizard, step3Ok, step4Ok, setWizard]);
  const steps = [
    t("training.step1"),
    t("training.step2"),
    t("training.step3"),
    t("training.step4"),
  ];

  return (
    <div className="training-page">
      <div className="training-page-header">
        <span className="panel-title">{t("training.title")}</span>
        {running && (
          <span className="training-live">
            <span className="pulse-dot" />
            {t("training.active")}
          </span>
        )}
        <div className="training-header-spacer" />
        <button className="panel-close" onClick={closePage} title={t("training.close")}>
          X
        </button>
      </div>
      <nav className="training-steps">
        {steps.map((label, i) => {
          const n = (i + 1) as 1 | 2 | 3 | 4;
          const enabled = stepOk[i];
          return (
            <button
              key={n}
              className={`training-step-tab ${wizard === n ? "active" : ""}`}
              disabled={!enabled}
              onClick={() => setWizard(n)}
            >
              <span className="training-step-num">{n}</span>
              {label}
            </button>
          );
        })}
      </nav>
      <div className="training-step-body">
        {/* S41 order swap (user ruling): the training object decides whether
            data is even needed, so it comes FIRST — diff with a reusable pool
            skips the data page entirely (TargetStep's next jumps to 3) */}
        {wizard === 1 && <TargetStep />}
        {wizard === 2 && <DataStep />}
        {wizard === 3 && <ParamsStep />}
        {wizard === 4 && <RunStep />}
      </div>
      {/* ①c: on the DATA step (2) with ≥2 singers the per-card highlight IS the
          drop affordance, so suppress the full-screen overlay there; on other
          steps the cards aren't mounted, so keep the overlay (an off-step drop
          routes to singer #1 by design) */}
      {dropActive &&
        !(backendSupportsMultiSpeaker(config.backend) && speakerGroups.length > 1 && wizard === 2) && (
          <div className="training-drop-overlay">{t("training.dropHint")}</div>
        )}
    </div>
  );
}

/* -------------------- step 2 (since the S41 order swap): data -------------------- */

// PreviewPlayer extracted to components/common/previewPlayer.ts in S41 (the
// audition rows share it; singleton = data-step preview and audition playback
// preempt each other, which is the intended behavior)

// Scrubber extracted to components/common/Scrubber.tsx (S66) — the workflow node output
// preview shares it. The data-list margin moved to .training-scrubber-slot (caller-owned).

function DataStep() {
  const { t } = useTranslation();
  const {
    dataset,
    speakerGroups,
    addFiles,
    removeFile,
    addSpeaker,
    removeSpeaker,
    setSpeakerName,
    addSpeakerFiles,
    removeSpeakerFile,
    dragOverSpeakerId,
    flashSpeaker,
    clearFlashSpeaker,
    setWizard,
    config,
    diffWsInfo,
  } = useTrainingStore();
  const diffPool = diffPoolReady(config.backend, diffWsInfo);
  // ①c: SoVITS (α) + RVC (α′) data is a SINGER LIST (default 1 singer = single-speaker);
  // diff/vocoder keep the flat file list.
  const singerList = backendSupportsMultiSpeaker(config.backend);
  const [playingPath, setPlayingPath] = useState<string | null>(null);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [pos, setPos] = useState(0); // seconds into the active file
  const rafRef = useRef<number | null>(null);
  const playTokenRef = useRef(0); // guards a superseded decode from starting playback

  const stopTicker = () => {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  };
  const runTicker = () => {
    stopTicker();
    const tick = () => {
      setPos(preview.position);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };
  // reset local preview state — preview.stop() does NOT fire onEnd, so callers
  // that stop playback explicitly (remove-file / remove-singer) must reset here
  // or playingPath + the rAF ticker leak (and a same-path file in another group
  // would show a false "playing" indicator)
  const resetPreviewState = () => {
    stopTicker();
    setPlayingPath(null);
    setPaused(false);
    setPos(0);
  };

  useEffect(() => {
    preview.onEnd = resetPreviewState;
    return () => {
      preview.onEnd = null;
      stopTicker();
      preview.stop();
    };
  }, []);

  const pickFiles = async () => {
    const picked = await open({
      multiple: true,
      filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
      title: t("training.addFiles"),
    });
    if (!picked) return;
    await addFiles(Array.isArray(picked) ? picked : [picked]);
  };

  const togglePlay = async (path: string) => {
    if (playingPath === path) {
      if (paused) {
        preview.resume();
        setPaused(false);
        runTicker();
      } else {
        preview.pause();
        setPaused(true);
        stopTicker();
      }
      return;
    }
    preview.stop();
    stopTicker();
    const token = ++playTokenRef.current;
    setPlayingPath(path);
    setLoadingPath(path);
    setPaused(false);
    setPos(0);
    try {
      // Read the ORIGINAL file directly (fs scope allows **) and decode on the
      // player's own context. This deliberately skips load_audio_file — that
      // command fully re-decodes + extracts waveform peaks (unused here) + writes
      // a cache-WAV copy, which for a 35-min file is the multi-second stall; a
      // preview only needs the decoded samples.
      const bytes = await readFile(path);
      const buffer = await preview.decode(bytes);
      // superseded by a newer play gesture (or the file was removed) while decoding
      if (token !== playTokenRef.current) return;
      // ①c: the file may live in the flat dataset OR a speaker group
      const st = useTrainingStore.getState();
      const stillPresent =
        st.dataset.some((f) => f.path === path) ||
        st.speakerGroups.some((g) => g.files.some((f) => f.path === path));
      if (!stillPresent) {
        setPlayingPath(null);
        setLoadingPath(null);
        return;
      }
      await preview.play(path, buffer);
      setLoadingPath(null);
      runTicker();
    } catch (e) {
      if (token !== playTokenRef.current) return;
      preview.stop();
      setPlayingPath(null);
      setLoadingPath(null);
      useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
    }
  };

  const totalMs = dataset.reduce((acc, f) => acc + (f.durationMs ?? 0), 0);
  const activeDur = preview.duration || 0;

  const pickSpeakerFiles = async (id: string) => {
    const picked = await open({
      multiple: true,
      filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
      title: t("training.addFiles"),
    });
    if (!picked) return;
    await addSpeakerFiles(id, Array.isArray(picked) ? picked : [picked]);
  };

  // one file row (play / scrub / remove) — shared by the flat list AND each
  // speaker group so the preview logic stays single-source. onRemove differs
  // (removeFile vs removeSpeakerFile) but the play/scrub state is by path.
  const renderFileRow = (f: DatasetFile, onRemove: () => void) => {
    const isActive = playingPath === f.path;
    const isLoading = loadingPath === f.path;
    const isPlaying = isActive && !paused && !isLoading;
    return (
      <div key={f.path} className="training-file-row" title={f.path}>
        <div className="training-file-main">
          <button
            className={`training-file-play ${isPlaying ? "on" : ""} ${isLoading ? "loading" : ""}`}
            onClick={() => void togglePlay(f.path)}
            disabled={isLoading}
            title={
              isLoading
                ? t("training.loadingPreview")
                : isPlaying
                  ? t("training.pausePreview")
                  : t("training.preview")
            }
          >
            {isLoading ? "◌" : isPlaying ? "❚❚" : "▶"}
          </button>
          <span className="training-file-name">{f.name}</span>
          <span className="training-file-dur">
            {isActive && activeDur > 0
              ? `${fmtDur(pos)} / ${fmtDur(activeDur)}`
              : f.durationMs != null
                ? fmtDur(f.durationMs / 1000)
                : "--:--"}
          </span>
          <button
            className="training-file-remove"
            onClick={() => {
              if (isActive) {
                preview.stop();
                resetPreviewState();
              }
              onRemove();
            }}
            title={t("training.remove")}
          >
            X
          </button>
        </div>
        {isActive && activeDur > 0 && (
          <Scrubber
            className="training-scrubber-slot"
            value={pos / activeDur}
            onSeek={(frac) => {
              preview.seek(frac);
              setPos(preview.position);
            }}
          />
        )}
      </div>
    );
  };

  return (
    <div className="training-data-step">
      <div className="training-hint">{t("training.dataHint")}</div>

      {singerList ? (
        <div className="training-spk-stack">
          {speakerGroups.map((g, i) => {
            const gMs = g.files.reduce((a, f) => a + (f.durationMs ?? 0), 0);
            // with a single singer this is just the flat list (no card chrome /
            // name / remove); the name + remove appear once there are ≥2 singers
            const multiSinger = speakerGroups.length > 1;
            return (
              <div
                key={g.id}
                // data-spk-id = the drag hit-test anchor (only meaningful with
                // ≥2 singers; a lone singer takes any drop)
                data-spk-id={multiSinger ? g.id : undefined}
                className={
                  multiSinger
                    ? `training-spk-group${dragOverSpeakerId === g.id ? " drop-target" : ""}`
                    : undefined
                }
              >
                {multiSinger && flashSpeaker?.id === g.id && (
                  // one-shot pulse confirming files landed on THIS singer; the
                  // nonce key remounts it so repeat adds re-trigger the animation
                  <span
                    key={flashSpeaker.nonce}
                    className="training-spk-flash"
                    onAnimationEnd={() => clearFlashSpeaker(flashSpeaker.nonce)}
                  />
                )}
                {multiSinger && (
                  <div className="training-spk-header">
                    <span className="training-spk-idx">{i + 1}</span>
                    <input
                      className="training-spk-name"
                      value={g.name}
                      placeholder={t("training.speakerName")}
                      onChange={(e) => setSpeakerName(g.id, e.target.value)}
                    />
                    <span className="training-spk-count">
                      {t("training.files", { count: g.files.length })}
                      {gMs > 0 && <> · {fmtDur(gMs / 1000)}</>}
                    </span>
                    <button
                      className="training-file-remove"
                      onClick={() => {
                        // stop an in-progress preview of a file in THIS singer
                        // before the row (and its scrubber) unmounts
                        if (playingPath && g.files.some((f) => f.path === playingPath)) {
                          preview.stop();
                          resetPreviewState();
                        }
                        removeSpeaker(g.id);
                      }}
                      title={t("training.removeSpeaker")}
                    >
                      X
                    </button>
                  </div>
                )}
                <div className="training-data-actions">
                  <button
                    className="training-btn"
                    onClick={() => void pickSpeakerFiles(g.id)}
                  >
                    {t("training.addFiles")}
                  </button>
                  {!multiSinger && (
                    <span className="training-data-total">
                      {t("training.files", { count: g.files.length })}
                      {gMs > 0 && (
                        <> · {t("training.totalDur", { dur: fmtDur(gMs / 1000) })}</>
                      )}
                    </span>
                  )}
                </div>
                {g.files.length > 0 ? (
                  <div
                    className={
                      multiSinger
                        ? "training-file-list training-spk-files"
                        : "training-file-list"
                    }
                  >
                    {g.files.map((f) =>
                      renderFileRow(f, () => removeSpeakerFile(g.id, f.path)),
                    )}
                  </div>
                ) : (
                  !multiSinger && (
                    <div className="training-empty">{t("training.empty")}</div>
                  )
                )}
              </div>
            );
          })}
          <button className="training-btn training-spk-add" onClick={addSpeaker}>
            {t("training.addSpeaker")}
          </button>
          <div className="training-hint training-spk-hint">
            {t("training.multiSpeakerHint")}
          </div>
        </div>
      ) : (
        <>
          <div className="training-data-actions">
            <button className="training-btn" onClick={() => void pickFiles()}>
              {t("training.addFiles")}
            </button>
            <span className="training-data-total">
              {t("training.files", { count: dataset.length })}
              {totalMs > 0 && (
                <> · {t("training.totalDur", { dur: fmtDur(totalMs / 1000) })}</>
              )}
            </span>
          </div>
          {dataset.length === 0 ? (
            <div className="training-empty">
              {t("training.empty")}
              {diffPool && (
                <div className="training-fixed-note">{t("training.diffPoolHint")}</div>
              )}
            </div>
          ) : (
            <div className="training-file-list">
              {dataset.map((f) => renderFileRow(f, () => removeFile(f.path)))}
            </div>
          )}
        </>
      )}

      <div className="training-step-nav">
        <button
          className="training-btn primary"
          disabled={!trainingDataOk(config.backend, dataset, speakerGroups, diffPool)}
          onClick={() => setWizard(3)}
        >
          {t("training.next")}
        </button>
      </div>
    </div>
  );
}

/* ------------------- step 1 (since the S41 order swap): target ------------------- */

function TargetStep() {
  const { t } = useTranslation();
  const { config, updateConfig, setWizard, diffWsInfo } = useTrainingStore();
  const [exists, setExists] = useState(false);
  const [wsInfo, setWsInfo] = useState<WorkspaceInfo | null>(null);
  // the diffusion companion is BOUND to a SoVITS model — the diff card picks
  // one from the installed registry instead of free-typing a name (its
  // version/dim derive from the pick; a same-named training workspace reuses
  // its preprocessing caches)
  const sovitsModels = useVoiceModelStore((s) => s.models.sovits);
  useEffect(() => {
    void useVoiceModelStore.getState().fetchModels();
  }, []);
  const isDiffCard = config.backend === "sovits_diff";
  const diffModelPicked =
    isDiffCard && sovitsModels.some((m) => m.name === config.modelName);

  const pickDiffModel = (name: string) => {
    const m = sovitsModels.find((x) => x.name === name);
    if (!m) return;
    updateConfig({
      modelName: m.name,
      diffVersion: voiceFeatureDim(m) === 256 ? ("4.0" as const) : ("4.1" as const),
    });
    void checkName(m.name);
  };

  // keep the derived version in lock-step with the picked model on EVERY path
  // (card switch with a pre-filled matching name, registry refresh, dropdown)
  useEffect(() => {
    if (!isDiffCard) return;
    const m = sovitsModels.find((x) => x.name === config.modelName);
    if (!m) return;
    const v = voiceFeatureDim(m) === 256 ? ("4.0" as const) : ("4.1" as const);
    if (v !== config.diffVersion) updateConfig({ diffVersion: v });
  }, [isDiffCard, config.modelName, config.diffVersion, sovitsModels, updateConfig]);
  // the name the current hints were computed FOR — typing a different name
  // must hide them instead of showing stale facts until blur (review F19)
  const [checkedName, setCheckedName] = useState("");
  // card switches re-check with the new backend — a slower older invoke must
  // not overwrite the newer answer
  const checkSeq = useRef(0);

  // re-check on mount: the page may be reopened with a name already filled in
  useEffect(() => {
    void checkName(config.modelName);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const checkName = async (name: string) => {
    const mySeq = ++checkSeq.current;
    if (!name.trim()) {
      setExists(false);
      setWsInfo(null);
      setCheckedName("");
      return;
    }
    const backend = useTrainingStore.getState().config.backend;
    if (backend === "sovits_diff") {
      // the diff card's same-name semantics are INVERTED: a same-named sovits
      // workspace is the good case (its preprocessing caches get reused) —
      // check_model_exists doesn't apply (the product is an attachment)
      let info: WorkspaceInfo | null = null;
      try {
        info = await invoke<WorkspaceInfo>("get_training_workspace_info", {
          name: name.trim(),
        });
      } catch {
        info = null;
      }
      if (mySeq === checkSeq.current) {
        setWsInfo(info);
        setExists(false);
        setCheckedName(name.trim());
      }
      return;
    }
    let found = false;
    try {
      found = await invoke<boolean>("check_model_exists", {
        name: name.trim(),
        modelType: backend,
      });
    } catch {
      found = false;
    }
    if (mySeq === checkSeq.current) {
      setExists(found);
      setWsInfo(null);
      setCheckedName(name.trim());
    }
  };

  const cards = [
    {
      key: "rvc",
      active: config.backend === "rvc",
      pick: () => updateConfig({ backend: "rvc" as const }),
      name: t("training.backendRvc"),
      desc: t("training.backendRvcDesc"),
    },
    {
      key: "sovits41",
      active: config.backend === "sovits" && config.sovitsVersion === "4.1",
      pick: () => updateConfig({ backend: "sovits" as const, sovitsVersion: "4.1" as const }),
      name: t("training.backendSovits41"),
      desc: t("training.backendSovits41Desc"),
    },
    {
      key: "sovits40",
      active: config.backend === "sovits" && config.sovitsVersion === "4.0",
      pick: () => updateConfig({ backend: "sovits" as const, sovitsVersion: "4.0" as const }),
      name: t("training.backendSovits40"),
      desc: t("training.backendSovits40Desc"),
    },
    {
      key: "sovits_diff",
      active: config.backend === "sovits_diff",
      pick: () => updateConfig({ backend: "sovits_diff" as const }),
      name: t("training.backendDiff"),
      desc: t("training.backendDiffDesc"),
    },
    {
      key: "vocoder",
      active: config.backend === "vocoder",
      pick: () => updateConfig({ backend: "vocoder" as const }),
      name: t("training.backendVocoder"),
      desc: t("training.backendVocoderDesc"),
    },
  ];

  const diffHint =
    config.backend === "sovits_diff" &&
    wsInfo?.exists &&
    checkedName === config.modelName.trim()
      ? wsInfo.family && wsInfo.family !== "sovits"
        ? { kind: "warn" as const, text: t("training.diffWorkspaceForeign", { family: wsInfo.family }) }
        : wsInfo.diff_steps > 0
          ? { kind: "info" as const, text: t("training.diffWorkspaceProgress", { steps: wsInfo.diff_steps }) }
          : { kind: "info" as const, text: t("training.diffWorkspaceReuse") }
      : null;

  return (
    <div className="training-target-step">
      <div className="training-backend-cards">
        {cards.map((c) => (
          <button
            key={c.key}
            className={`training-backend-card ${c.active ? "active" : ""}`}
            onClick={() => {
              c.pick();
              void checkName(config.modelName);
            }}
          >
            <span className="training-backend-name">{c.name}</span>
            <span className="training-backend-desc">{c.desc}</span>
          </button>
        ))}
      </div>
      {isDiffCard ? (
        sovitsModels.length > 0 ? (
          <div className="training-form-row">
            <label>{t("training.diffPickModel")}</label>
            <Dropdown
              value={diffModelPicked ? config.modelName : ""}
              options={sovitsModels.map((m) => ({
                value: m.name,
                label: `${m.name} · ${voiceFeatureDim(m) === 256 ? "4.0" : "4.1"}`,
              }))}
              onChange={(v) => pickDiffModel(v)}
            />
          </div>
        ) : (
          <div className="training-name-exists">{t("training.diffNoModels")}</div>
        )
      ) : (
        <div className="training-form-row">
          <label>{t("training.modelName")}</label>
          <input
            type="text"
            value={config.modelName}
            placeholder={t("training.modelNamePlaceholder")}
            onChange={(e) => updateConfig({ modelName: e.target.value })}
            onBlur={(e) => void checkName(e.target.value)}
          />
        </div>
      )}
      {exists && !isDiffCard && (
        <div className="training-name-exists">{t("training.nameExists")}</div>
      )}
      {diffHint && (
        <div className={diffHint.kind === "warn" ? "training-name-exists" : "training-hint"}>
          {diffHint.text}
        </div>
      )}

      <div className="training-step-nav">
        <button
          className="training-btn primary"
          disabled={!config.modelName.trim() || (isDiffCard && !diffModelPicked)}
          onClick={() =>
            // diff with a reusable shared pool skips the data page (S41 order
            // swap rationale); the data tab stays clickable for a manual
            // dataset update
            setWizard(diffPoolReady(config.backend, diffWsInfo) ? 3 : 2)
          }
        >
          {t("training.next")}
        </button>
      </div>
    </div>
  );
}

/* ---------------------------------- step 3: params ---------------------------------- */

/** Number field with themed square ▲/▼ steppers (native spinner hidden in CSS).
 *  Typing goes through a DRAFT: clamping only on blur/steppers — a clamp on
 *  every keystroke makes values below `min` untypeable (typing "100" with
 *  min=50 would clamp the leading "1" to 50). In-range keystrokes commit live. */
function NumberField({
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const clamp = (v: number) => Math.max(min, Math.min(max, v));
  const commitDraft = () => {
    if (draft !== null) {
      const n = parseInt(draft, 10);
      if (Number.isFinite(n)) onChange(clamp(n));
    }
    setDraft(null);
  };
  const stepBy = (d: number) => {
    setDraft(null);
    onChange(clamp(value + d));
  };
  return (
    <div className="training-number">
      <input
        type="number"
        min={min}
        max={max}
        value={draft ?? value}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = parseInt(e.target.value, 10);
          if (Number.isFinite(n) && n >= min && n <= max) onChange(n);
        }}
        onBlur={commitDraft}
        onKeyDown={(e) => {
          if (e.key === "Enter") commitDraft();
        }}
      />
      <div className="training-number-steps">
        <button type="button" tabIndex={-1} onClick={() => stepBy(step)}>
          ▲
        </button>
        <button type="button" tabIndex={-1} onClick={() => stepBy(-step)}>
          ▼
        </button>
      </div>
    </div>
  );
}

function ParamsStep() {
  const { t } = useTranslation();
  const { config, updateConfig, setWizard, diffWsInfo } = useTrainingStore();
  const [gpus, setGpus] = useState<string[]>([]);
  const [cudaOk, setCudaOk] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  // diff inherits 数据增强份数 from the host workspace manifest — show the
  // REAL inherited value (store diffWsInfo, fetched by the root effect)
  const diffAugInherit =
    config.backend === "sovits_diff" && diffWsInfo?.exists ? diffWsInfo.aug_copies : null;

  useEffect(() => {
    void (async () => {
      try {
        const hw = await invoke<{ gpu_name: string; cuda_available: boolean }>(
          "get_hardware_info",
        );
        setGpus(
          hw.gpu_name
            .split(",")
            .map((g) => g.trim())
            .filter(Boolean),
        );
        setCudaOk(hw.cuda_available);
        if (!hw.cuda_available) {
          // keep the PAYLOAD truthful, not just the checkbox display
          useTrainingStore.getState().updateConfig({ forceCpu: true });
        }
      } catch {
        setGpus([]);
      }
    })();
  }, []);

  const sovits = config.backend === "sovits";
  const diff = config.backend === "sovits_diff";
  const voc = config.backend === "vocoder";

  const gpuRow = gpus.length > 0 && !config.forceCpu && (
    <div className="training-form-row">
      <label>{t("training.gpu")}</label>
      <Dropdown
        value={config.gpu}
        options={gpus.map((g, i) => ({ value: i, label: g }))}
        onChange={(v) => updateConfig({ gpu: v })}
      />
    </div>
  );

  const forceCpuRow = cudaOk && (
    <label className="training-check-row">
      <input
        type="checkbox"
        checked={config.forceCpu}
        onChange={(e) => updateConfig({ forceCpu: e.target.checked })}
      />
      {t("training.forceCpu")}
    </label>
  );

  return (
    <div className="training-params-step">
      {diff ? (
        <>
          <div className="training-form-grid">
            <div className="training-form-row">
              <label>{t("training.version")}</label>
              {/* bound to the SoVITS model picked in step 2 — not a choice */}
              <span className="training-fixed-value">
                SoVITS {config.diffVersion} · {t("training.versionFollowsModel")}
              </span>
            </div>
            <div className="training-form-row">
              <label>{t("training.totalSteps")}</label>
              <NumberField
                min={1000}
                max={1000000}
                step={1000}
                value={config.diffTotalSteps}
                onChange={(v) => updateConfig({ diffTotalSteps: v })}
              />
            </div>
            <div className="training-form-row">
              <label>{t("training.batchSize")}</label>
              <NumberField
                min={1}
                max={128}
                value={config.diffBatchSize}
                onChange={(v) => updateConfig({ diffBatchSize: v })}
              />
            </div>
            <div className="training-form-row">
              <label>{t("training.saveEverySteps")}</label>
              <NumberField
                min={100}
                max={20000}
                step={100}
                value={config.diffSaveEverySteps}
                onChange={(v) => updateConfig({ diffSaveEverySteps: v })}
              />
            </div>
            <div className="training-form-row">
              <label>{t("training.kStepMax")}</label>
              <Dropdown
                value={config.diffKStepMax}
                options={[
                  { value: 0, label: t("training.kStepFull") },
                  { value: 100, label: "100" },
                  { value: 200, label: "200" },
                  { value: 300, label: "300" },
                ]}
                onChange={(v) => updateConfig({ diffKStepMax: v })}
              />
            </div>
            {gpuRow}
          </div>
          {config.diffVersion === "4.0" && (
            <div className="training-hint">{t("training.diffNoBase40")}</div>
          )}
        </>
      ) : voc ? (
        <>
          <div className="training-form-grid">
            <div className="training-form-row">
              <label>{t("training.vocScope")}</label>
              {/* 一期单格式类 — an informational row, not a choice (不能选隐藏) */}
              <span className="training-fixed-value">{t("training.vocScopeValue")}</span>
            </div>
            <div className="training-form-row">
              <label title={t("training.vocTotalStepsTip")}>{t("training.totalSteps")}</label>
              <NumberField
                min={100}
                max={100000}
                step={100}
                value={config.vocTotalSteps}
                onChange={(v) => updateConfig({ vocTotalSteps: v })}
              />
            </div>
            <div className="training-form-row">
              <label title={t("training.vocBatchTip")}>{t("training.batchSize")}</label>
              <NumberField
                min={1}
                max={64}
                value={config.vocBatchSize}
                onChange={(v) => updateConfig({ vocBatchSize: v })}
              />
            </div>
            <div className="training-form-row">
              <label>{t("training.saveEverySteps")}</label>
              <NumberField
                min={50}
                max={10000}
                step={50}
                value={config.vocSaveEverySteps}
                onChange={(v) => updateConfig({ vocSaveEverySteps: v })}
              />
            </div>
            {gpuRow}
          </div>
          <div className="training-hint">{t("training.vocLicenseNote")}</div>
        </>
      ) : !sovits ? (
        <div className="training-form-grid">
          <div className="training-form-row">
            <label>{t("training.version")}</label>
            <Dropdown
              value={config.version}
              options={[
                { value: "v2", label: "v2" },
                { value: "v1", label: "v1" },
              ]}
              onChange={(v) => updateConfig({ version: v })}
            />
          </div>
          <div className="training-form-row">
            <label>{t("training.sampleRate")}</label>
            <Dropdown
              value={config.sampleRate}
              options={[
                { value: "48k", label: "48k" },
                { value: "40k", label: "40k" },
                { value: "32k", label: "32k" },
              ]}
              onChange={(v) => updateConfig({ sampleRate: v })}
            />
          </div>
          <div className="training-form-row">
            <label>{t("training.totalEpoch")}</label>
            <NumberField
              min={1}
              max={10000}
              value={config.totalEpoch}
              onChange={(v) => updateConfig({ totalEpoch: v })}
            />
          </div>
          <div className="training-form-row">
            <label>{t("training.batchSize")}</label>
            <NumberField
              min={1}
              max={64}
              value={config.batchSize}
              onChange={(v) => updateConfig({ batchSize: v })}
            />
          </div>
          {gpuRow}
        </div>
      ) : (
        <div className="training-form-grid">
          <div className="training-form-row">
            <label>{t("training.totalEpoch")}</label>
            <NumberField
              min={1}
              max={100000}
              value={config.sovitsTotalEpoch}
              onChange={(v) => updateConfig({ sovitsTotalEpoch: v })}
            />
          </div>
          <div className="training-form-row">
            <label>{t("training.batchSize")}</label>
            <NumberField
              min={1}
              max={64}
              value={config.sovitsBatchSize}
              onChange={(v) => updateConfig({ sovitsBatchSize: v })}
            />
          </div>
          <div className="training-form-row">
            <label>{t("training.saveEverySteps")}</label>
            <NumberField
              min={50}
              max={20000}
              step={50}
              value={config.sovitsSaveEverySteps}
              onChange={(v) => updateConfig({ sovitsSaveEverySteps: v })}
            />
          </div>
          <div className="training-form-row">
            <label>{t("training.keepCkpts")}</label>
            <NumberField
              min={1}
              max={50}
              value={config.sovitsKeepCkpts}
              onChange={(v) => updateConfig({ sovitsKeepCkpts: v })}
            />
          </div>
          {gpuRow}
        </div>
      )}

      <div className="training-fixed-note">
        {diff
          ? t("training.diffFixedNote")
          : voc
            ? t("training.vocFixedNote")
            : sovits
              ? t("training.sovitsFixedNote")
              : t("training.fixedNote")}
      </div>

      <button
        className="training-advanced-toggle"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? "▼" : "▶"} {t("training.advanced")}
      </button>
      {showAdvanced &&
        (diff ? (
          <div className="training-form-grid">
            <div className="training-form-row">
              <label>{t("training.forceSaveSteps")}</label>
              <NumberField
                min={1000}
                max={200000}
                step={1000}
                value={config.diffForceSaveSteps}
                onChange={(v) => updateConfig({ diffForceSaveSteps: v })}
              />
            </div>
            <div className="training-form-row">
              <label title={t("training.augCopiesTip")}>{t("training.augCopies")}</label>
              {/* inherited from the workspace manifest (shared dataset_44k
                  slice pool) — not a diff-run choice, like loudnorm */}
              <span className="training-fixed-value">
                {t("training.augFollowWorkspace")}
                {diffAugInherit !== null
                  ? ` · ${t("training.augInheritCount", { count: diffAugInherit })}`
                  : ""}
              </span>
            </div>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.diffFp16}
                onChange={(e) => updateConfig({ diffFp16: e.target.checked })}
              />
              {t("training.fp16")}
            </label>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.diffCacheAllData}
                onChange={(e) => updateConfig({ diffCacheAllData: e.target.checked })}
              />
              {t("training.cacheAllData")}
            </label>
            {forceCpuRow}
          </div>
        ) : voc ? (
          <div className="training-form-grid">
            <div className="training-form-row">
              <label title={t("training.vocCropTip")}>{t("training.vocCrop")}</label>
              <NumberField
                min={16}
                max={128}
                step={8}
                value={config.vocCropMelFrames}
                onChange={(v) => updateConfig({ vocCropMelFrames: v })}
              />
            </div>
            <div className="training-form-row">
              <label>{t("training.keepCkpts")}</label>
              <NumberField
                min={1}
                max={50}
                value={config.vocKeepCkpts}
                onChange={(v) => updateConfig({ vocKeepCkpts: v })}
              />
            </div>
            <label className="training-check-row" title={t("training.vocFreezeMpdTip")}>
              <input
                type="checkbox"
                checked={config.vocFreezeMpd}
                onChange={(e) => updateConfig({ vocFreezeMpd: e.target.checked })}
              />
              {t("training.vocFreezeMpd")}
            </label>
            <div className="training-form-row">
              <label title={t("training.augCopiesTip")}>{t("training.augCopies")}</label>
              <NumberField
                min={0}
                max={3}
                value={config.vocAugCopies}
                onChange={(v) => updateConfig({ vocAugCopies: v })}
              />
            </div>
            {forceCpuRow}
          </div>
        ) : !sovits ? (
          <div className="training-form-grid">
            <div className="training-form-row">
              <label>{t("training.saveEvery")}</label>
              <NumberField
                min={1}
                max={1000}
                value={config.saveEveryEpoch}
                onChange={(v) => updateConfig({ saveEveryEpoch: v })}
              />
            </div>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.saveEveryWeights}
                onChange={(e) => updateConfig({ saveEveryWeights: e.target.checked })}
              />
              {t("training.saveWeights")}
            </label>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.keepOnlyLatest}
                onChange={(e) => updateConfig({ keepOnlyLatest: e.target.checked })}
              />
              {t("training.keepLatest")}
            </label>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.cacheGpu}
                onChange={(e) => updateConfig({ cacheGpu: e.target.checked })}
              />
              {t("training.cacheGpu")}
            </label>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.fp16}
                onChange={(e) => updateConfig({ fp16: e.target.checked })}
              />
              {t("training.fp16")}
            </label>
            <div className="training-form-row">
              <label title={t("training.augCopiesTip")}>{t("training.augCopies")}</label>
              <NumberField
                min={0}
                max={3}
                value={config.augCopies}
                onChange={(v) => updateConfig({ augCopies: v })}
              />
            </div>
            {forceCpuRow}
          </div>
        ) : (
          <div className="training-form-grid">
            {config.sovitsVersion === "4.1" && (
              <label className="training-check-row">
                <input
                  type="checkbox"
                  checked={config.sovitsVolEmbedding}
                  onChange={(e) => updateConfig({ sovitsVolEmbedding: e.target.checked })}
                />
                {t("training.volEmbedding")}
              </label>
            )}
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.sovitsKmeans}
                onChange={(e) => updateConfig({ sovitsKmeans: e.target.checked })}
              />
              {t("training.kmeansOpt")}
            </label>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.sovitsLoudnorm}
                onChange={(e) => updateConfig({ sovitsLoudnorm: e.target.checked })}
              />
              {t("training.loudnorm")}
            </label>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.sovitsFp16}
                onChange={(e) => updateConfig({ sovitsFp16: e.target.checked })}
              />
              {t("training.fp16")}
            </label>
            <label className="training-check-row">
              <input
                type="checkbox"
                checked={config.sovitsAllInMem}
                onChange={(e) => updateConfig({ sovitsAllInMem: e.target.checked })}
              />
              {t("training.allInMem")}
            </label>
            <div className="training-form-row">
              <label title={t("training.augCopiesTip")}>{t("training.augCopies")}</label>
              <NumberField
                min={0}
                max={3}
                value={config.sovitsAugCopies}
                onChange={(v) => updateConfig({ sovitsAugCopies: v })}
              />
            </div>
            {forceCpuRow}
          </div>
        ))}

      <div className="training-step-nav">
        <button className="training-btn primary" onClick={() => setWizard(4)}>
          {t("training.next")}
        </button>
      </div>
    </div>
  );
}

/* ---------------------------------- step 4: run ---------------------------------- */

function RunStep() {
  const { t, i18n } = useTranslation();
  const rlang = i18n.language;
  const showConfirm = useAppStore((s) => s.showConfirm);
  const showToast = useAppStore((s) => s.showToast);
  const {
    snapshot,
    snapshotAt,
    history,
    dataset,
    speakerGroups,
    config,
    starting,
    start,
    stop,
    forceStop,
    resetRun,
    setWizard,
    diffWsInfo,
  } = useTrainingStore();
  const chartRef = useRef<LossChartHandle>(null);
  const [, forceTick] = useState(0);

  const running = snapshot.state === "starting" || snapshot.state === "running";
  const finished = snapshot.state === "completed" || snapshot.state === "stopped";
  const isDiff = snapshot.backend === "sovits_diff";
  // vocoder shares the "best = true validation loss" semantics with diff
  // (labels only; its checkpoints go through importCkpt, not the attach flow)
  const isVocoderRun = snapshot.backend === "vocoder";

  // ---- diffusion attach flow (S39): a trained diffusion ckpt is not a
  // standalone model — it converts into `<stem>.diffusion/` of an INSTALLED
  // SoVITS model whose ContentVec dim matches; the rvc list feeds the
  // installed-model version check in onStart ----
  const sovitsModels = useVoiceModelStore((s) => s.models.sovits);
  const rvcModels = useVoiceModelStore((s) => s.models.rvc);
  const vocoderModels = useVoiceModelStore((s) => s.models.vocoder);
  const [attachTarget, setAttachTarget] = useState("");
  const [attaching, setAttaching] = useState<string | null>(null);
  const summaryDim = (snapshot.summary as { encoder_dim?: number } | null)?.encoder_dim;
  const attachCandidates = isDiff
    ? sovitsModels.filter((m: VoiceModelEntry) => {
        if (!summaryDim) return true; // dim unknown (e.g. force-stopped run) — Rust re-validates
        const dim = voiceFeatureDim(m);
        return dim === null || dim === summaryDim;
      })
    : [];

  useEffect(() => {
    if (isDiff && finished) void useVoiceModelStore.getState().fetchModels();
  }, [isDiff, finished]);

  // a NEW run invalidates any previously chosen target — without this reset a
  // still-valid selection from the last run survives and the default-target
  // effect below early-returns, silently pointing this run's checkpoints at
  // the previous run's model (review F16)
  useEffect(() => {
    setAttachTarget("");
  }, [snapshot.model_name, snapshot.workspace]);

  // default the target to the same-named model (the intended pairing)
  useEffect(() => {
    if (!isDiff) return;
    if (attachTarget && attachCandidates.some((m) => m.name === attachTarget)) return;
    const sameName = attachCandidates.find((m) => m.name === snapshot.model_name);
    setAttachTarget(sameName?.name ?? attachCandidates[0]?.name ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDiff, sovitsModels, summaryDim, snapshot.model_name]);

  const attachCkpt = async (ckpt: CkptInfo) => {
    if (!attachTarget || attaching) return;
    setAttaching(ckpt.path);
    try {
      await invoke("attach_diffusion", { name: attachTarget, ckptPath: ckpt.path });
      await useVoiceModelStore.getState().fetchModels();
      showToast(t("training.diffAttached", { name: attachTarget }), "success");
    } catch (e) {
      showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
    } finally {
      setAttaching(null);
    }
  };

  // ---- S41 audition (试听多选保留): candidates render the bundled 10s clip
  // through the app inference chain; rvc/sovits/vocoder = multi-select keep,
  // diff = listen → pick one → attach (existing attach flow) ----
  type AuditionPhase = "converting" | "rendering" | "ready" | "playing";
  const [auditionState, setAuditionState] = useState<Record<string, AuditionPhase>>({});
  // S60c: per-checkpoint tested ranges (this run's auto-test results; the record itself
  // persists in each candidate's audition sidecar — this map only feeds the row label).
  const [candRanges, setCandRanges] = useState<Record<string, { usable: [number, number]; comfort: [number, number] }>>({});
  const candRangeRunRef = useRef<string | null>(null);
  const [auditionWavs, setAuditionWavs] = useState<Record<string, string>>({});
  const [selectedCkpts, setSelectedCkpts] = useState<Record<string, boolean>>({});
  const [missingCkpts, setMissingCkpts] = useState<Record<string, boolean>>({});
  const [importingAll, setImportingAll] = useState(false);
  // ①c: audition a chosen speaker of a multi-speaker rvc/sovits run. Names come from the RUN's
  // frozen speaker list (snapshot.speakers, index = emb_g id = the converter's speaker-map id) —
  // NOT the editable DataStep state, so it survives a DataStep edit and reflects what was trained.
  // Empty for single-speaker / diff / vocoder → the render falls back to speaker 0 (unchanged).
  const [auditionSpeaker, setAuditionSpeaker] = useState(0);
  const auditionSpeakers =
    backendSupportsMultiSpeaker(snapshot.backend) && (snapshot.speakers?.length ?? 0) > 1
      ? snapshot.speakers!.map((name, i) => ({ id: i, name: name.trim() || `#${i}` }))
      : [];
  const auditionBusy = Object.values(auditionState).some(
    (s) => s === "converting" || s === "rendering",
  );

  // stale-resolution fence (审查修复 FE-3/FE-5/AUD-HOST-SWITCH-STALE): every
  // context change that invalidates in-flight results bumps the epoch; a
  // resolving invoke compares its captured epoch and discards itself
  const auditionEpochRef = useRef(0);

  // new-run reset (red-team R9): best/final snapshot PATHS are identical across
  // runs of the same model — a stale ready-state would replay the previous
  // run's render as this run's voice
  useEffect(() => {
    auditionEpochRef.current += 1;
    setAuditionState({});
    setAuditionWavs({});
    setSelectedCkpts({});
    setMissingCkpts({});
    setAuditionSpeaker(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot.model_name, snapshot.workspace, running]);

  // the diff audition cache is host-specific — switching the host must
  // invalidate every rendered result INCLUDING one still in flight
  useEffect(() => {
    if (!isDiff) return;
    auditionEpochRef.current += 1;
    setAuditionState({});
    setAuditionWavs({});
  }, [attachTarget, isDiff]);

  // remount reconciliation (审查修复 FE-1/AUD-DONE-DROPPED): transient
  // converting/rendering phases die with the page — if Rust says nothing is
  // in flight, drop any stranded busy phase so auditionBusy can't deadlock
  // the whole finished area
  useEffect(() => {
    if (!finished) return;
    void (async () => {
      try {
        const active = await invoke<boolean>("audition_active");
        if (!active) {
          setAuditionState((s) => {
            const n: typeof s = {};
            for (const [k, v] of Object.entries(s)) {
              if (v === "ready" || v === "playing") n[k] = v;
            }
            return n;
          });
        }
      } catch {
        /* reconciliation is best-effort */
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished]);

  // ── S60c: post-training auto range test (§user) — every rvc/sovits checkpoint gets the
  // C2–C7 scale test (~1-2 s each) so ① its audition pre-shifts a low-range singer into
  // comfort (the bundled clip skews high — an untested low singer sounds "training failed")
  // and ② the row shows the singer's range. Once per finished run (ref-guarded), sequential
  // (the Rust audition FlightGuard is single-flight anyway); failures skip silently — the
  // audition still works without a record.
  useEffect(() => {
    if (!finished || snapshot.ckpts.length === 0) return;
    if (snapshot.backend !== "rvc" && snapshot.backend !== "sovits") return;
    const runKey = `${snapshot.workspace}|${snapshot.ckpts.map((c) => c.path).join(",")}`;
    if (candRangeRunRef.current === runKey) return;
    candRangeRunRef.current = runKey;
    let alive = true;
    void (async () => {
      for (const c of snapshot.ckpts) {
        if (!alive) return;
        try {
          const r = await runCandidateRangeTest(
            snapshot.workspace,
            snapshot.backend as "rvc" | "sovits",
            c.path,
            c.path,
          );
          if (alive && r) setCandRanges((s) => ({ ...s, [c.path]: r }));
        } catch {
          /* busy (user auditioning) or a broken ckpt — skip; retest happens on the next run */
        }
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished, snapshot.ckpts, snapshot.backend, snapshot.workspace]);

  // archive policies prune files the snapshot still lists (diff periodics,
  // red-team F16) — grey those rows instead of offering dead buttons
  useEffect(() => {
    if (!finished || snapshot.ckpts.length === 0) return;
    let alive = true;
    void (async () => {
      const gone: Record<string, boolean> = {};
      for (const c of snapshot.ckpts) {
        try {
          if (!(await exists(c.path))) gone[c.path] = true;
        } catch {
          /* treat unprobeable as present — Rust errors loudly on use */
        }
      }
      if (alive) setMissingCkpts(gone);
    })();
    return () => {
      alive = false;
    };
  }, [finished, snapshot.ckpts]);

  // conversion/render phases arrive as events (the invoke itself resolves with
  // the wav path; a closed page loses transient phases — the wav cache makes a
  // re-click instant, design A19)
  useEffect(() => {
    const un = listen<{ candidate_id: string; phase: string; wav?: string | null }>(
      "audition-progress",
      (e) => {
        const { candidate_id, phase, wav } = e.payload;
        if (phase === "converting" || phase === "rendering") {
          setAuditionState((s) => ({ ...s, [candidate_id]: phase as AuditionPhase }));
        } else if (phase === "done") {
          // terminal events are the busy-state ground truth (审查修复 FE-1):
          // the invoke resolution may belong to a dead component instance
          if (wav) setAuditionWavs((s) => ({ ...s, [candidate_id]: wav }));
          setAuditionState((s) =>
            s[candidate_id] === "playing" ? s : { ...s, [candidate_id]: "ready" },
          );
        } else if (phase === "error") {
          setAuditionState((s) => {
            const n = { ...s };
            delete n[candidate_id];
            return n;
          });
        }
      },
    );
    return () => {
      void un.then((f) => f());
    };
  }, []);

  // shared preview singleton — consumer contract (previewPlayer.ts): stop +
  // release onEnd on unmount so a stale callback can't drive dead state
  useEffect(() => {
    return () => {
      auditionEpochRef.current += 1; // in-flight resolutions become no-ops
      preview.stop();
      preview.onEnd = null;
    };
  }, []);

  const playAuditionWav = async (id: string, wavPath: string) => {
    preview.stop();
    const bytes = await readFile(wavPath);
    const buf = await preview.decode(bytes);
    preview.onEnd = () => setAuditionState((s) => ({ ...s, [id]: "ready" }));
    await preview.play(wavPath, buf);
    // single-playing invariant (审查修复 FE-4): the preview singleton can only
    // play one thing — every other 'playing' marker demotes to 'ready'
    setAuditionState((s) => {
      const n: typeof s = {};
      for (const [k, v] of Object.entries(s)) n[k] = v === "playing" ? "ready" : v;
      n[id] = "playing";
      return n;
    });
  };

  /** c === null → the built-in default vocoder A/B reference row. */
  const auditionCandidate = async (c: CkptInfo | null) => {
    const id = c ? c.path : "__default__";
    const phase = auditionState[id];
    // pause only when this row REALLY owns the playback — a stale 'playing'
    // marker (superseded by another row) falls through to replay (FE-4)
    if (phase === "playing" && preview.path === auditionWavs[id]) {
      preview.pause();
      setAuditionState((s) => ({ ...s, [id]: "ready" }));
      return;
    }
    if ((phase === "ready" || phase === "playing") && auditionWavs[id]) {
      try {
        await playAuditionWav(id, auditionWavs[id]);
      } catch {
        // cache swept underneath us (清空结果/new run) — drop back to idle
        setAuditionState((s) => {
          const n = { ...s };
          delete n[id];
          return n;
        });
        setAuditionWavs((s) => {
          const n = { ...s };
          delete n[id];
          return n;
        });
      }
      return;
    }
    if (phase === "converting" || phase === "rendering" || auditionBusy || importingAll) return;
    if (isDiff && !attachTarget) {
      showToast(t("training.auditionNeedHost"), "error");
      return;
    }
    // stale fence: unmount / new run / host switch bump the epoch — a late
    // resolution must not populate state or start playback (FE-3/FE-5)
    const epoch = auditionEpochRef.current;
    setAuditionState((s) => ({ ...s, [id]: "converting" }));
    try {
      let wav: string;
      if (isVocoderRun || c === null) {
        wav = await invoke<string>("render_audition_vocoder", {
          ckptPath: c?.path ?? null,
          workspace: snapshot.workspace,
          candidateId: id,
        });
      } else if (isDiff) {
        wav = await invoke<string>("render_audition_diffusion", {
          hostName: attachTarget,
          ckptPath: c.path,
          workspace: snapshot.workspace,
          candidateId: id,
        });
      } else {
        wav = await invoke<string>("render_audition_voice", {
          backend: snapshot.backend,
          ckptPath: c.path,
          workspace: snapshot.workspace,
          candidateId: id,
          // ①c: null for single-speaker (→ speaker 0, byte-identical); the chosen speaker otherwise
          speakerId: auditionSpeakers.length > 0 ? auditionSpeaker : null,
        });
      }
      if (epoch !== auditionEpochRef.current) return; // superseded — discard
      setAuditionWavs((s) => ({ ...s, [id]: wav }));
      await playAuditionWav(id, wav);
    } catch (e) {
      if (epoch !== auditionEpochRef.current) return;
      setAuditionState((s) => {
        const n = { ...s };
        delete n[id];
        return n;
      });
      if (isCancelError(e)) return; // user cancelled the audition — not an error, no toast
      // APP_BUSY: another audition/render holds the FlightGuard → info; real failures stay errors.
      showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
    }
  };

  const auditionLabel = (id: string) => {
    switch (auditionState[id]) {
      case "converting":
        return t("training.auditionConverting");
      case "rendering":
        return t("training.auditionRendering");
      case "playing":
        return "❚❚";
      case "ready":
        return "▶";
      default:
        return t("training.audition");
    }
  };

  // ①c: switching the audition speaker invalidates every rendered clip (they were the OLD
  // speaker's voice) — bump the epoch (discard in-flight) + clear the display caches so the
  // next play re-renders with the new speaker (the Rust side caches per speaker, so re-picking
  // a previously-heard speaker is an instant cache hit).
  const changeAuditionSpeaker = (id: number) => {
    if (id === auditionSpeaker) return;
    preview.stop();
    auditionEpochRef.current += 1;
    setAuditionState({});
    setAuditionWavs({});
    setAuditionSpeaker(id);
  };

  // elapsed ticker (1 Hz) while running
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => forceTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [running]);

  const elapsed = running
    ? snapshot.elapsed_secs + Math.max(0, (Date.now() - snapshotAt) / 1000)
    : snapshot.elapsed_secs;

  const bestCkpt = snapshot.ckpts.find((c) => c.kind === "best");

  const onStart = async () => {
    // S41 共享池模式: diff may start with no fresh import when the host
    // workspace has a reusable pool (Rust re-verifies authoritatively).
    // ①c: multi-speaker needs ≥2 named non-empty groups (trainingDataOk).
    if (
      !trainingDataOk(
        config.backend,
        dataset,
        speakerGroups,
        diffPoolReady(config.backend, diffWsInfo),
      )
    ) {
      showToast(t("training.needData"), "error");
      return;
    }
    const name = config.modelName.trim();
    if (!name) {
      showToast(t("training.needName"), "error");
      return;
    }

    if (config.backend === "sovits_diff") {
      // diff semantics: a same-named workspace is the EXPECTED case (cache
      // reuse); the dialog fires whenever one exists so the user always gets
      // the 重训-only-diffusion escape hatch (a half-baked diff-first
      // workspace version-locks the manifest — retrain is the way out)
      let info: WorkspaceInfo | null = null;
      try {
        info = await invoke<WorkspaceInfo>("get_training_workspace_info", { name });
      } catch {
        info = null;
      }
      if (info?.exists && info.family && info.family !== "sovits") {
        showToast(t("training.diffWorkspaceForeign", { family: info.family }), "error");
        return;
      }
      let fresh = false;
      if (info?.exists) {
        const hasProgress = info.diff_steps > 0;
        // 重训 only spares the workspace when a main model lives in it — a
        // diff-only workspace gets fully wiped (that is what unlocks a version
        // change); the dialog must not promise otherwise (review F17)
        const wipeNote = info.has_main_progress
          ? ""
          : " " + t("training.diffRetrainFullWipeNote");
        const choice = await showConfirm({
          title: t("training.diffConfirmTitle"),
          body:
            (hasProgress
              ? t("training.diffConfirmResumeBody", { name, steps: info.diff_steps })
              : t("training.diffConfirmReuseBody", { name })) + wipeNote,
          buttons: [
            {
              id: "resume",
              label: hasProgress ? t("training.resume") : t("training.continueTrain"),
              kind: "primary",
            },
            { id: "retrain", label: t("training.retrainDiff"), kind: "danger" },
            { id: "cancel", label: t("training.cancel") },
          ],
        });
        if (choice !== "resume" && choice !== "retrain") return;
        fresh = choice === "retrain";
      }
      await start(fresh).catch(() => undefined);
      return;
    }

    let fresh = true;
    let modelExists = false;
    try {
      modelExists = await invoke<boolean>("check_model_exists", {
        name,
        modelType: config.backend,
      });
    } catch {
      modelExists = false;
    }
    let info: WorkspaceInfo | null = null;
    try {
      info = await invoke<WorkspaceInfo>("get_training_workspace_info", { name });
    } catch {
      info = null;
    }
    let wsExists = info?.exists ?? false;
    if (!info) {
      // legacy fallback: unknown version → the generic dialog below (whose
      // 续训 the Rust manifest guard still backstops)
      try {
        wsExists = await invoke<boolean>("check_training_workspace", { name });
      } catch {
        /* keep false */
      }
    }

    // vocoder's "version" is the fixed manifest marker — hitting the sovits
    // fallback here would compare "nsf_hifigan" vs "4.1" and lock every
    // vocoder workspace into the retrain-only mismatch dialog (红队 A16)
    const selectedVersion =
      config.backend === "rvc"
        ? config.version
        : config.backend === "vocoder"
          ? "nsf_hifigan"
          : config.sovitsVersion;
    const selectedSr = config.backend === "rvc" ? config.sampleRate : "44k";
    // the wipe would also destroy any diffusion training progress living in
    // this workspace — the user must see that before choosing 重训
    const diffWarn =
      info && info.diff_steps > 0
        ? " " + t("training.retrainWipesDiff", { steps: info.diff_steps })
        : "";

    if (wsExists && info) {
      // ①c: item-by-item diff of the new form vs the stored workspace. Resume needs EVERY
      // resume-guarded param to match (the Rust guard rejects otherwise) — surface the mismatches
      // HERE so the user can调回一致 and resume, instead of a red error toast at start time. (Was
      // just the 2-field version/sr check; now generalized to every guarded param.) vol_embedding
      // mirrors start()'s expression; speakers compares the RUN's names (info.speakers, from
      // run.json) BY ORDER = emb_g id — the exact thing the slug guard checks, shown as names.
      const onOff = (b: boolean) =>
        t18({ zh: b ? "开" : "关", en: b ? "On" : "Off", ja: b ? "オン" : "オフ" }, rlang);
      const singleSpk = t18({ zh: "单歌手", en: "single speaker", ja: "単一話者" }, rlang);
      const diffRows: { label: string; oldv: string; newv: string }[] = [];
      if (info.version && info.version !== selectedVersion)
        diffRows.push({ label: t18({ zh: "版本", en: "Version", ja: "バージョン" }, rlang), oldv: info.version, newv: selectedVersion });
      if (info.sample_rate && info.sample_rate !== selectedSr)
        diffRows.push({ label: t18({ zh: "采样率", en: "Sample rate", ja: "サンプルレート" }, rlang), oldv: info.sample_rate, newv: selectedSr });
      if (config.backend === "sovits" && info.vol_embedding != null) {
        const curVol = config.sovitsVersion === "4.1" ? config.sovitsVolEmbedding : false;
        if (info.vol_embedding !== curVol)
          diffRows.push({ label: t18({ zh: "响度嵌入", en: "Vol embedding", ja: "音量埋め込み" }, rlang), oldv: onOff(info.vol_embedding), newv: onOff(curVol) });
      }
      if (backendSupportsMultiSpeaker(config.backend)) {
        const oldNames = info.speakers ?? [];
        const curNames = speakerGroups.length > 1 ? speakerGroups.map((g) => g.name.trim()) : [];
        const same = oldNames.length === curNames.length && oldNames.every((n, i) => n === curNames[i]);
        if (!same)
          diffRows.push({
            label: t18({ zh: "歌手（含顺序）", en: "Speakers (order)", ja: "話者（順序）" }, rlang),
            oldv: oldNames.length ? oldNames.join("、") : singleSpk,
            newv: curNames.length ? curNames.join("、") : singleSpk,
          });
      }
      // (扩散深度 k_step_max resume-diff belongs to the sovits_diff variant handled in its own
      // earlier branch — config.backend is already narrowed to rvc/sovits/vocoder here.)

      if (diffRows.length > 0) {
        // any mismatch → resume impossible (guard would reject). Show WHAT differs (old → new) so
        // the user can fix the form and resume; else retrain (wipe). Retrain-only here, exactly as
        // the old version/sr branch did — but now itemized for every guarded param.
        const list = diffRows.map((r) => `· ${r.label}：${r.oldv} → ${r.newv}`).join("\n");
        const choice = await showConfirm({
          title: t18({ zh: "配置与原工作区不一致", en: "Config differs from workspace", ja: "設定が元と不一致" }, rlang),
          body:
            t18(
              {
                zh: `「${name}」续训要求新配置与原工作区完全一致。以下不同——调回一致即可正常续训，或重训（清空重来）：`,
                en: `Resuming "${name}" needs the config to match the workspace exactly. These differ — set them back to resume, or retrain (wipe):`,
                ja: `「${name}」の続行には設定が元と完全一致が必要です。以下が相違——一致させれば続行可、または再学習（消去）：`,
              },
              rlang,
            ) +
            "\n\n" +
            list +
            diffWarn,
          buttons: [
            { id: "retrain", label: t("training.retrain"), kind: "danger" },
            { id: "cancel", label: t("training.cancel") },
          ],
        });
        if (choice !== "retrain") return;
        fresh = true;
      } else {
        // everything matches → the classic resume/retrain choice
        const choice = await showConfirm({
          title: t("training.confirmExistTitle"),
          body: t("training.confirmExistBody", { name }) + diffWarn,
          buttons: [
            { id: "resume", label: t("training.resume"), kind: "primary" },
            { id: "retrain", label: t("training.retrain"), kind: "danger" },
            { id: "cancel", label: t("training.cancel") },
          ],
        });
        if (choice !== "resume" && choice !== "retrain") return;
        fresh = choice === "retrain";
      }
    } else if (wsExists) {
      // info unreadable (manifest missing/corrupt): classic resume/retrain — the Rust guard is the
      // authoritative backstop for any param mismatch we couldn't diff here.
      const choice = await showConfirm({
        title: t("training.confirmExistTitle"),
        body: t("training.confirmExistBody", { name }) + diffWarn,
        buttons: [
          { id: "resume", label: t("training.resume"), kind: "primary" },
          { id: "retrain", label: t("training.retrain"), kind: "danger" },
          { id: "cancel", label: t("training.cancel") },
        ],
      });
      if (choice !== "resume" && choice !== "retrain") return;
      fresh = choice === "retrain";
    } else if (modelExists) {
      // installed model, NO workspace: there is nothing to resume —「续训」
      // would silently train from scratch; say what actually happens (and
      // call out a version mismatch when the registry knows the version)
      const installed = (
        config.backend === "rvc"
          ? rvcModels
          : config.backend === "vocoder"
            ? vocoderModels
            : sovitsModels
      ).find((m) => m.name === name);
      const installedVersion = installed ? voiceVersionBadge(installed) : null;
      const mismatch = installedVersion && installedVersion !== selectedVersion;
      const choice = await showConfirm({
        title: t("training.confirmExistTitle"),
        body: mismatch
          ? t("training.modelVersionMismatchBody", {
              name,
              old: installedVersion,
              new: selectedVersion,
            })
          : t("training.noWorkspaceBody", { name }),
        buttons: [
          { id: "go", label: t("training.continueTrain"), kind: "primary" },
          { id: "cancel", label: t("training.cancel") },
        ],
      });
      if (choice !== "go") return;
      fresh = true;
    }
    await start(fresh).catch(() => undefined);
  };

  const onStop = async () => {
    await stop();
  };

  // confirm before clearing: the ckpt list (with its import/attach buttons)
  // is the LAST surface for this run's artifacts — a confirmed clear means
  // the user is done with them, which is why there is deliberately no
  // "re-attach later" entry elsewhere (user decision 2026-07-06)
  const onClearResult = async () => {
    const choice = await showConfirm({
      title: t("training.clearResult"),
      body: t("training.clearResultConfirmBody"),
      buttons: [
        { id: "clear", label: t("training.clearResult"), kind: "primary" },
        { id: "cancel", label: t("training.cancel") },
      ],
    });
    if (choice !== "clear") return;
    // anti-escape (user report, S41 live test): after a page refresh the
    // dataset list is gone (in-memory) while the snapshot survives (backend);
    // clearing from that state used to leave the wizard parked on step 4 with
    // zero data. 清空 semantically ends the round — jump back to step 1
    // (only on an ACCEPTED clear; a refused one keeps the results visible).
    if (await resetRun()) setWizard(1);
  };

  const onForceStop = async () => {
    const choice = await showConfirm({
      title: t("training.forceStopConfirmTitle"),
      body: t("training.forceStopConfirmBody"),
      buttons: [
        { id: "kill", label: t("training.forceStop"), kind: "danger" },
        { id: "cancel", label: t("training.cancel") },
      ],
    });
    if (choice === "kill") await forceStop();
  };

  const exportChart = async () => {
    const blob = await chartRef.current?.toPngBlob();
    if (!blob) return;
    const path = await save({
      defaultPath: `${snapshot.model_name || "training"}_loss.png`,
      filters: [{ name: "PNG", extensions: ["png"] }],
    });
    if (!path) return;
    const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
    try {
      await invoke("save_binary_file", { path, data: bytes });
      showToast(t("training.chartSaved"), "success");
    } catch (e) {
      showToast(String(e), "error");
    }
  };

  // sovits/vocoder periodics are step-cadenced (several per epoch) — an
  // epoch-keyed suggestion would collide and silently replace the previous
  // import (rvc keeps its historical epoch tag)
  const suggestedName = (ckpt: CkptInfo) => {
    const tag =
      ckpt.kind === "best"
        ? "best"
        : snapshot.backend === "rvc"
          ? `e${ckpt.epoch}`
          : `s${ckpt.step}`;
    return ckpt.kind === "final" ? snapshot.model_name : `${snapshot.model_name}_${tag}`;
  };

  // fallbacks for runs without a summary (e.g. force-stopped): rvc keeps its
  // historical total_fea.npy; sovits probes the workspace cluster assets
  // (built before training, so they exist even for early stops). Shared by the
  // single-import prompt and the S41 batch import (single source).
  const resolveIndexPath = async (): Promise<string | undefined> => {
    const summaryIndex = (snapshot.summary as { index?: string } | null)?.index;
    let indexPath = summaryIndex;
    if (!indexPath && snapshot.backend !== "vocoder") {
      // vocoders have no index/cluster companion — probing would only find
      // another backend's leftovers (红队 A16 fallback-site sweep)
      if (snapshot.backend === "rvc") {
        indexPath = `${snapshot.workspace}\\total_fea.npy`;
      } else {
        for (const cand of [
          `${snapshot.workspace}\\cluster\\kmeans_10000.pt`,
          `${snapshot.workspace}\\cluster\\0.index_vectors.npy`,
        ]) {
          if (await exists(cand)) {
            indexPath = cand;
            break;
          }
        }
      }
    }
    return indexPath;
  };

  const importCkpt = async (ckpt: CkptInfo) => {
    const name = await showConfirm({
      title: t("training.import"),
      body: t("training.importName"),
      buttons: [
        { id: "ok", label: t("training.import"), kind: "primary" },
        // "__cancel": with input mode the PRIMARY resolves the typed VALUE, other
        // buttons resolve their id — a plain "cancel" id would collide with a
        // model literally named "cancel"
        { id: "__cancel", label: t("training.cancel") },
      ],
      input: { initial: suggestedName(ckpt) },
    });
    if (!name || name === "__cancel") return;
    const indexPath = await resolveIndexPath();
    try {
      await invoke("import_model", {
        name,
        path: ckpt.path,
        modelType: snapshot.backend,
        indexPath,
      });
      await useVoiceModelStore.getState().fetchModels();
      showToast(t("training.imported", { name }), "success");
    } catch (e) {
      // MODEL_BUSY_AUDITION / APP_BUSY land here raw without the shared mapper (audit gap).
      showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
    }
  };

  /** S41 batch import of the checked candidates, auto-named by the single-
   *  import suggestion rules with in-batch dedupe (red-team A9: a stop archive
   *  can share its step/epoch with a periodic — REPLACE would silently eat
   *  one). Prefers the audition-converted onnx when present (instant copy). */
  const importSelected = async () => {
    const chosen = snapshot.ckpts.filter(
      (c) => !missingCkpts[c.path] && (selectedCkpts[c.path] ?? true),
    );
    if (chosen.length === 0 || importingAll) return;
    const names = new Map<string, string>();
    const used = new Set<string>();
    for (const c of chosen) {
      let n = suggestedName(c);
      if (used.has(n)) n = `${n}_${c.kind}`;
      let i = 2;
      while (used.has(n)) {
        n = `${suggestedName(c)}_${c.kind}${i}`;
        i += 1;
      }
      used.add(n);
      names.set(c.path, n);
    }
    const lines = chosen.map(
      (c) => `${names.get(c.path)}  ←  ${c.path.split(/[\\/]/).pop()}`,
    );
    const okId = await showConfirm({
      title: t("training.importSelectedTitle"),
      body: `${t("training.importSelectedBody")}\n\n${lines.join("\n")}`,
      buttons: [
        { id: "ok", label: t("training.import"), kind: "primary" },
        { id: "cancel", label: t("training.cancel") },
      ],
    });
    if (okId !== "ok") return;
    setImportingAll(true);
    try {
      const indexPath = await resolveIndexPath();
      const audName = isVocoderRun ? "vocoder" : "model";
      let ok = 0;
      const failed: string[] = [];
      const warns: string[] = [];
      for (const c of chosen) {
        let path = c.path;
        try {
          const stem = c.path
            .split(/[\\/]/)
            .pop()!
            .replace(/\.[^.]+$/, "");
          const dir = `${snapshot.workspace}\\audition\\${stem}`;
          // the sidecar json is the conversion's COMPLETION marker (exporters
          // write it last, 审查修复 S41-RUST-1/2) — a bare onnx is an
          // interrupted/rejected conversion and must fall back to the raw ckpt
          if (
            (await exists(`${dir}\\${audName}.onnx`)) &&
            (await exists(`${dir}\\${audName}.json`))
          ) {
            path = `${dir}\\${audName}.onnx`;
          }
        } catch {
          /* fall back to the raw ckpt (import converts it itself) */
        }
        try {
          const outcome = await invoke<{ warnings?: string[] }>("import_model", {
            name: names.get(c.path),
            path,
            modelType: snapshot.backend,
            indexPath,
          });
          ok += 1;
          for (const w of outcome?.warnings ?? []) {
            warns.push(`${names.get(c.path)}: ${backendErrorMessage(w) ?? w}`);
          }
        } catch (e) {
          failed.push(`${names.get(c.path)}: ${backendErrorMessage(e) ?? e}`);
        }
      }
      await useVoiceModelStore.getState().fetchModels();
      if (failed.length > 0) {
        showToast(
          `${t("training.importSelectedPartial", { ok, total: chosen.length })}\n${[...failed, ...warns].join("\n")}`,
          "error",
        );
      } else if (warns.length > 0) {
        showToast(
          `${t("training.importSelectedDone", { count: ok })}\n${warns.join("\n")}`,
          "info",
        );
      } else {
        showToast(t("training.importSelectedDone", { count: ok }), "success");
      }
    } finally {
      setImportingAll(false);
    }
  };

  /* -------- idle -------- */
  if (snapshot.state === "idle") {
    return (
      <div className="training-run-step">
        <div className="training-run-summary-line">
          {config.backend === "rvc" ? (
            <>
              {config.modelName || "—"} · RVC {config.version} · {config.sampleRate} ·{" "}
              {t("training.totalEpoch")} {config.totalEpoch} · batch {config.batchSize}
            </>
          ) : config.backend === "vocoder" ? (
            <>
              {config.modelName || "—"} · {t("training.backendVocoder")} · 44.1k ·{" "}
              {t("training.totalSteps")} {config.vocTotalSteps} · batch {config.vocBatchSize}
            </>
          ) : config.backend === "sovits_diff" ? (
            <>
              {config.modelName || "—"} · {t("training.backendDiff")} · SoVITS{" "}
              {config.diffVersion} · {t("training.totalSteps")} {config.diffTotalSteps} ·
              batch {config.diffBatchSize}
            </>
          ) : (
            <>
              {config.modelName || "—"} · SoVITS {config.sovitsVersion} · 44.1k ·{" "}
              {t("training.totalEpoch")} {config.sovitsTotalEpoch} · batch{" "}
              {config.sovitsBatchSize}
            </>
          )}
        </div>
        <button
          className="training-btn primary training-start-btn"
          disabled={starting}
          onClick={() => void onStart()}
        >
          {t("training.start")}
        </button>
      </div>
    );
  }

  const trainingStarted = snapshot.step != null || history.length > 0;

  return (
    <div className="training-run-step">
      {/* preprocessing stages (ordered by the LIVE run's backend) */}
      {!trainingStarted && running && (
        <div className="training-stages">
          {(STAGE_ORDERS[snapshot.backend] ?? STAGE_ORDERS.rvc!).map((stage, idx, order) => {
            const cur = snapshot.stage;
            const curIdx = cur ? order.indexOf(cur.stage) : -1;
            const state = idx < curIdx ? "done" : idx === curIdx ? "active" : "pending";
            return (
              <div key={stage} className={`training-stage-row ${state}`}>
                <span className="training-stage-mark">
                  {state === "done" ? "✓" : state === "active" ? "▸" : "·"}
                </span>
                <span className="training-stage-label">{t(`training.stage_${stage}`)}</span>
                {state === "active" && cur?.progress != null && (
                  <div className="training-stage-bar">
                    <div
                      className="training-stage-bar-fill"
                      style={{ width: `${Math.round((cur.progress ?? 0) * 100)}%` }}
                    />
                  </div>
                )}
                {state === "active" && cur?.message && (
                  // Stage messages are mostly file names (pass through raw); the odd status CODE
                  // (SHARED_POOL_REUSED) localizes via the shared mapper.
                  <span className="training-stage-msg">{backendErrorMessage(cur.message) ?? cur.message}</span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* training monitor */}
      {trainingStarted && (
        <>
          <div className="training-monitor-row">
            <span>
              {t("training.step")} {snapshot.step?.step ?? 0}/{snapshot.step?.total_steps ?? 0}
            </span>
            {/* diffusion runs are step-based — total_epochs 0 is a sentinel,
                a meaningless "epoch 3/0" line is hidden (house rule) */}
            {(snapshot.step?.total_epochs ?? snapshot.total_epochs) > 0 && (
              <span>
                epoch {snapshot.step?.epoch ?? 0}/{snapshot.step?.total_epochs ?? snapshot.total_epochs}
              </span>
            )}
            <span>
              {t("training.elapsed")} {fmtDur(elapsed)}
            </span>
            {running && snapshot.step?.eta_secs != null && (
              <span>
                {t("training.eta")} {fmtDur(snapshot.step.eta_secs)}
              </span>
            )}
            <span>
              {isDiff || isVocoderRun ? t("training.bestVal") : t("training.best")}:{" "}
              {bestCkpt
                ? `${bestCkpt.metric?.toFixed(3) ?? "?"} @ ${bestCkpt.step}`
                : t("training.bestNone")}
            </span>
          </div>
          <LossChart ref={chartRef} history={history} bestStep={bestCkpt?.step ?? null} />
          <div className="training-chart-actions">
            <button className="training-btn" onClick={() => void exportChart()}>
              {t("training.exportChart")}
            </button>
          </div>
        </>
      )}

      {/* controls */}
      {running && (
        <div className="training-run-controls">
          {!snapshot.stop_requested ? (
            <button className="training-btn danger" onClick={() => void onStop()}>
              {t("training.stop")}
            </button>
          ) : (
            <>
              <span className="training-stopping">{t("training.stopping")}</span>
              <button className="training-btn danger" onClick={() => void onForceStop()}>
                {t("training.forceStop")}
              </button>
            </>
          )}
        </div>
      )}

      {/* finished summary */}
      {(snapshot.state === "completed" || snapshot.state === "stopped") && (
        <div className="training-summary-card">
          <div className="training-summary-title">
            {snapshot.state === "completed"
              ? t("training.doneCompleted")
              : t("training.doneStopped")}
          </div>
          <div className="training-summary-facts">
            <span>
              {t("training.sumSteps")}: {snapshot.step?.step ?? 0}
            </span>
            <span>
              {t("training.sumTime")}: {fmtDur(snapshot.elapsed_secs)}
            </span>
            {bestCkpt && (
              <span>
                {isDiff || isVocoderRun ? t("training.sumBestVal") : t("training.sumBest")}:{" "}
                {bestCkpt.metric?.toFixed(3)} @ {bestCkpt.step}
              </span>
            )}
          </div>
          {/* diffusion products attach to an INSTALLED SoVITS model (dim-matched);
              no candidates -> hide the buttons, show guidance (house rule) */}
          {isDiff &&
            (attachCandidates.length > 0 ? (
              <div className="training-attach-row">
                <label>{t("training.attachTarget")}</label>
                <Dropdown
                  value={attachTarget}
                  options={attachCandidates.map((m) => ({ value: m.name, label: m.name }))}
                  onChange={(v) => setAttachTarget(v)}
                />
              </div>
            ) : (
              <div className="training-hint">{t("training.noAttachTarget")}</div>
            ))}
          {/* ①c: pick which speaker of a multi-speaker run to audition (names from the run's
              singer list). Hidden for single-speaker / diff / vocoder. */}
          {auditionSpeakers.length > 0 && (
            <div className="training-attach-row">
              <label title={t("training.auditionSpeakerTip")}>{t("training.auditionSpeaker")}</label>
              <Dropdown
                value={String(auditionSpeaker)}
                options={auditionSpeakers.map((s) => ({ value: String(s.id), label: s.name }))}
                onChange={(v) => changeAuditionSpeaker(parseInt(v, 10))}
              />
            </div>
          )}
          <div className="training-ckpt-list">
            {/* S41: the vocoder run gets a pinned A/B reference row — the
                built-in default vocoder rendering the SAME clip */}
            {isVocoderRun && (
              <div className="training-ckpt-row reference">
                <span className="training-ckpt-kind reference">A/B</span>
                <span className="training-ckpt-name">{t("training.auditionRef")}</span>
                <button
                  className="training-btn small"
                  disabled={(auditionBusy && !auditionState["__default__"]) || importingAll}
                  onClick={() => void auditionCandidate(null)}
                >
                  {auditionLabel("__default__")}
                </button>
              </div>
            )}
            {snapshot.ckpts.map((c) => {
              const gone = missingCkpts[c.path] === true;
              const phase = auditionState[c.path];
              return (
                <div
                  key={`${c.kind}-${c.step}-${c.path}`}
                  className={`training-ckpt-row${gone ? " missing" : ""}`}
                  title={gone ? t("training.ckptMissing") : undefined}
                >
                  {/* multi-select keep (rvc/sovits/vocoder; diff keeps its
                      listen→pick-one→attach semantics — no checkbox) */}
                  {!isDiff && (
                    <input
                      type="checkbox"
                      className="training-ckpt-check"
                      disabled={gone}
                      checked={!gone && (selectedCkpts[c.path] ?? true)}
                      onChange={(e) =>
                        setSelectedCkpts((s) => ({ ...s, [c.path]: e.target.checked }))
                      }
                    />
                  )}
                  <span className={`training-ckpt-kind ${c.kind}`}>
                    {t(`training.kind_${c.kind}`)}
                  </span>
                  <span className="training-ckpt-name" title={c.path}>
                    {c.path.replace(/\\/g, "/").split("/").pop()}
                  </span>
                  <span className="training-ckpt-meta">
                    {/* diffusion epochs are sentinel units — steps only */}
                    {isDiff ? <>s{c.step}</> : <>e{c.epoch} · s{c.step}</>}
                    {c.metric != null && <> · {c.metric.toFixed(3)}</>}
                    {/* S60c: auto-tested comfort zone (note names — the audience reads F#2,
                        not MIDI numbers §user); doubles as convergence feedback per ckpt */}
                    {candRanges[c.path] && (
                      <span className="training-ckpt-range" title={t("training.ckptRangeTip")}>
                        {" · "}
                        {midiName(candRanges[c.path]!.comfort[0])}–{midiName(candRanges[c.path]!.comfort[1])}
                      </span>
                    )}
                  </span>
                  {gone ? (
                    <span className="training-ckpt-missing">{t("training.ckptMissing")}</span>
                  ) : (
                    <>
                      <button
                        className="training-btn small"
                        disabled={
                          (auditionBusy && phase !== "converting" && phase !== "rendering") ||
                          (isDiff && !attachTarget) ||
                          // batch import copies audition onnx files — a render
                          // writing one concurrently would be a TOCTOU (FE-2)
                          importingAll
                        }
                        onClick={() => void auditionCandidate(c)}
                      >
                        {auditionLabel(c.path)}
                      </button>
                      {isDiff ? (
                        attachCandidates.length > 0 && (
                          <button
                            className="training-btn small"
                            disabled={!attachTarget || attaching != null}
                            onClick={() => void attachCkpt(c)}
                          >
                            {attaching === c.path
                              ? t("training.attaching")
                              : t("training.attach")}
                          </button>
                        )
                      ) : (
                        <button
                          className="training-btn small"
                          onClick={() => void importCkpt(c)}
                        >
                          {t("training.import")}
                        </button>
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>
          {/* S41 batch keep — default all-checked (user spec) */}
          {!isDiff && snapshot.ckpts.some((c) => !missingCkpts[c.path]) && (
            <div className="training-audition-bar">
              <button
                className="training-btn primary small"
                disabled={
                  importingAll ||
                  auditionBusy ||
                  snapshot.ckpts.filter(
                    (c) => !missingCkpts[c.path] && (selectedCkpts[c.path] ?? true),
                  ).length === 0
                }
                onClick={() => void importSelected()}
              >
                {importingAll
                  ? t("training.importingSelected")
                  : t("training.importSelected", {
                      count: snapshot.ckpts.filter(
                        (c) => !missingCkpts[c.path] && (selectedCkpts[c.path] ?? true),
                      ).length,
                    })}
              </button>
            </div>
          )}
        </div>
      )}

      {/* error */}
      {snapshot.state === "error" && (
        <div className="training-error-card">
          <div className="training-error-title">{t("training.doneError")}</div>
          <div className="training-error-msg">{backendErrorMessage(snapshot.error) ?? snapshot.error}</div>
          {snapshot.stderr_tail.length > 0 && (
            <pre className="training-error-tail">{snapshot.stderr_tail.join("\n")}</pre>
          )}
          <div className="training-error-hint">
            {t("training.errorHint")} ({snapshot.workspace})
          </div>
        </div>
      )}

      {/* a finished run must not be a dead end — start the next one right here.
          清空结果 clears only the DISPLAY (snapshot + curve); the workspace and
          its checkpoints stay resumable */}
      {(snapshot.state === "completed" ||
        snapshot.state === "stopped" ||
        snapshot.state === "error") && (
        <div className="training-run-controls">
          {/* auditionBusy: a conversion subprocess is writing into the
              audition dir — starting/clearing would race it (Rust enforces
              this too; the disable is the friendly first line) */}
          <button
            className="training-btn primary"
            disabled={starting || auditionBusy || importingAll}
            onClick={() => void onStart()}
          >
            {t("training.start")}
          </button>
          <button
            className="training-btn"
            disabled={starting || auditionBusy || importingAll}
            title={t("training.clearResultTip")}
            onClick={() => void onClearResult()}
          >
            {t("training.clearResult")}
          </button>
        </div>
      )}
    </div>
  );
}
