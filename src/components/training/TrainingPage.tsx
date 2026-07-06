/**
 * Full-screen training page (S37) — four stages: 数据 → 对象 → 参数 → 运行.
 * Covers the DAW (which stays mounted) as an absolute overlay inside app-content.
 * Training itself is fully backend-driven; this page is a projection of the
 * training store (event-fed) and may be closed/reopened at any time mid-run.
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { exists, readFile } from "@tauri-apps/plugin-fs";
import { useAppStore } from "../../store/app";
import {
  setupTrainingListeners,
  useTrainingStore,
  type CkptInfo,
  type WorkspaceInfo,
} from "../../store/training";
import {
  useVoiceModelStore,
  voiceFeatureDim,
  voiceVersionBadge,
  type VoiceModelEntry,
} from "../../store/voice-models";
import { AUDIO_EXT_RE, AUDIO_EXTENSIONS } from "../../lib/constants";
import { Dropdown } from "../common/Dropdown";
import { LossChart, type LossChartHandle } from "./LossChart";
import "./TrainingPage.css";

/** Preprocessing stage sequence per backend (stage names come from the sidecar
 *  protocol; these arrays only order/tick the checklist display). */
const STAGE_ORDERS: Record<string, string[]> = {
  rvc: ["import", "slice", "f0", "feature", "index", "filelist", "train_prep"],
  sovits: ["import", "slice", "filelist", "extract", "index", "train_prep"],
  sovits_diff: ["import", "slice", "filelist", "extract", "diff_prep", "train_prep"],
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
  const { wizard, setWizard, dataset, snapshot, refresh } = useTrainingStore();
  const [dropActive, setDropActive] = useState(false);

  useEffect(() => {
    void setupTrainingListeners();
    void refresh();
  }, [refresh]);

  const running = snapshot.state === "starting" || snapshot.state === "running";

  // OS drag-drop: the webview event is global, so the Arrangement timeline (which
  // stays mounted under this page) short-circuits while this page is open and we
  // take the drop here as dataset import. Registered ONCE (reads live state via
  // getState, like Arrangement) — addFiles dedupes, so a StrictMode double-mount
  // is harmless. NB Tauri's "over" payload has NO `paths` (only enter/drop do).
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const p = event.payload;
        const liveNow = () => {
          const s = useTrainingStore.getState().snapshot.state;
          return s === "starting" || s === "running";
        };
        if (p.type === "enter") {
          // don't invite a drop we'll refuse: adding to the dataset only affects
          // the NEXT run, so while one is live we accept nothing (matches the
          // Arrangement convention: no affordance for a drop that won't import)
          setDropActive(!liveNow() && p.paths.some((pp) => AUDIO_EXT_RE.test(pp)));
        } else if (p.type === "leave") {
          setDropActive(false);
        } else if (p.type === "drop") {
          setDropActive(false);
          if (liveNow()) return;
          const audio = p.paths.filter((pp) => AUDIO_EXT_RE.test(pp));
          if (audio.length === 0) return;
          const st = useTrainingStore.getState();
          void st.addFiles(audio);
          st.setWizard(1);
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
  const stepOk = [
    true,
    dataset.length > 0,
    dataset.length > 0,
    dataset.length > 0 || running || snapshot.state !== "idle",
  ];
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
        {wizard === 1 && <DataStep />}
        {wizard === 2 && <TargetStep />}
        {wizard === 3 && <ParamsStep />}
        {wizard === 4 && <RunStep />}
      </div>
      {dropActive && (
        <div className="training-drop-overlay">{t("training.dropHint")}</div>
      )}
    </div>
  );
}

/* ---------------------------------- step 1: data ---------------------------------- */

/** Single-file preview player: one AudioContext, decodes into ITS OWN buffer (not
 *  the DAW's shared loadedBuffers cache — a preview must not pin decoded PCM there
 *  for the session). WebAudio sources are one-shot, so pause/seek = stop + restart
 *  at an offset; `seq` guards a stop's onended from a superseded gesture. */
class PreviewPlayer {
  private ctx: AudioContext | null = null;
  private src: AudioBufferSourceNode | null = null;
  private buffer: AudioBuffer | null = null;
  private startedAt = 0; // ctx time when the current source started
  private offset = 0; // seconds into the buffer at startedAt
  private seq = 0;
  path: string | null = null;
  paused = false;
  onEnd: (() => void) | null = null;

  private ensureCtx(): AudioContext {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  /** Decode on the player's OWN context (one per session) — a fresh AudioContext
   *  per decode would hit the browser's ~6-context cap after a few previews. */
  decode(bytes: Uint8Array): Promise<AudioBuffer> {
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return this.ensureCtx().decodeAudioData(ab as ArrayBuffer);
  }

  get duration(): number {
    return this.buffer?.duration ?? 0;
  }

  get position(): number {
    if (!this.ctx || this.paused || !this.src) return this.offset;
    return Math.min(this.duration, this.offset + (this.ctx.currentTime - this.startedAt));
  }

  private startSource(offsetSec: number) {
    const ctx = this.ensureCtx();
    this.stopSource();
    const src = ctx.createBufferSource();
    src.buffer = this.buffer;
    src.connect(ctx.destination);
    const mySeq = ++this.seq;
    src.onended = () => {
      if (mySeq !== this.seq) return; // stopped for seek/pause/switch, not a real end
      this.path = null;
      this.onEnd?.();
    };
    this.offset = offsetSec;
    this.startedAt = ctx.currentTime;
    src.start(0, offsetSec);
    this.src = src;
    this.paused = false;
  }

  private stopSource() {
    this.seq++; // invalidate the outgoing source's onended
    if (this.src) {
      try {
        this.src.stop();
      } catch {
        /* already stopped */
      }
      this.src = null;
    }
  }

  async play(path: string, buffer: AudioBuffer) {
    this.buffer = buffer;
    this.path = path;
    this.startSource(0);
  }

  pause() {
    if (!this.src || this.paused) return;
    this.offset = this.position;
    this.stopSource();
    this.paused = true;
  }

  resume() {
    if (!this.paused || !this.buffer) return;
    this.startSource(this.offset);
  }

  seek(frac: number) {
    if (!this.buffer) return;
    const target = Math.max(0, Math.min(1, frac)) * this.duration;
    if (this.paused) {
      this.offset = target;
    } else {
      this.startSource(target);
    }
  }

  stop() {
    this.stopSource();
    this.path = null;
    this.paused = false;
    this.offset = 0;
    this.buffer = null; // release the decoded PCM (a long file is hundreds of MB)
  }
}

const preview = new PreviewPlayer();

/** Thin div scrubber (house style: square 4×12 head, no solid dot). Click/drag to
 *  seek; parent drives `value` via rAF. */
function Scrubber({ value, onSeek }: { value: number; onSeek: (frac: number) => void }) {
  const trackRef = useRef<HTMLDivElement>(null);
  const seekAt = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    onSeek((clientX - r.left) / Math.max(1, r.width));
  };
  const onDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    seekAt(e.clientX);
  };
  const onMove = (e: React.PointerEvent) => {
    if (e.buttons & 1) seekAt(e.clientX);
  };
  return (
    <div className="training-scrubber" ref={trackRef} onPointerDown={onDown} onPointerMove={onMove}>
      <div className="training-scrubber-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      <div className="training-scrubber-head" style={{ left: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

function DataStep() {
  const { t } = useTranslation();
  const { dataset, addFiles, removeFile, setWizard } = useTrainingStore();
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

  useEffect(() => {
    preview.onEnd = () => {
      stopTicker();
      setPlayingPath(null);
      setPaused(false);
      setPos(0);
    };
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
      if (useTrainingStore.getState().dataset.every((f) => f.path !== path)) {
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
      useAppStore.getState().showToast(String(e), "error");
    }
  };

  const totalMs = dataset.reduce((acc, f) => acc + (f.durationMs ?? 0), 0);
  const activeDur = preview.duration || 0;

  return (
    <div className="training-data-step">
      <div className="training-hint">{t("training.dataHint")}</div>
      <div className="training-data-actions">
        <button className="training-btn" onClick={() => void pickFiles()}>
          {t("training.addFiles")}
        </button>
        <span className="training-data-total">
          {t("training.files", { count: dataset.length })}
          {totalMs > 0 && <> · {t("training.totalDur", { dur: fmtDur(totalMs / 1000) })}</>}
        </span>
      </div>
      {dataset.length === 0 ? (
        <div className="training-empty">{t("training.empty")}</div>
      ) : (
        <div className="training-file-list">
          {dataset.map((f) => {
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
                      if (isActive) preview.stop();
                      removeFile(f.path);
                    }}
                    title={t("training.remove")}
                  >
                    X
                  </button>
                </div>
                {isActive && activeDur > 0 && (
                  <Scrubber
                    value={pos / activeDur}
                    onSeek={(frac) => {
                      preview.seek(frac);
                      setPos(preview.position);
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
      <div className="training-step-nav">
        <button
          className="training-btn primary"
          disabled={dataset.length === 0}
          onClick={() => setWizard(2)}
        >
          {t("training.next")}
        </button>
      </div>
    </div>
  );
}

/* ---------------------------------- step 2: target ---------------------------------- */

function TargetStep() {
  const { t } = useTranslation();
  const { config, updateConfig, setWizard } = useTrainingStore();
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
      <div className="training-hint">{t("training.comingSoon")}</div>

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
          onClick={() => setWizard(3)}
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
  const { config, updateConfig, setWizard } = useTrainingStore();
  const [gpus, setGpus] = useState<string[]>([]);
  const [cudaOk, setCudaOk] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

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
  const { t } = useTranslation();
  const showConfirm = useAppStore((s) => s.showConfirm);
  const showToast = useAppStore((s) => s.showToast);
  const {
    snapshot,
    snapshotAt,
    history,
    dataset,
    config,
    starting,
    start,
    stop,
    forceStop,
    resetRun,
  } = useTrainingStore();
  const chartRef = useRef<LossChartHandle>(null);
  const [, forceTick] = useState(0);

  const running = snapshot.state === "starting" || snapshot.state === "running";
  const finished = snapshot.state === "completed" || snapshot.state === "stopped";
  const isDiff = snapshot.backend === "sovits_diff";

  // ---- diffusion attach flow (S39): a trained diffusion ckpt is not a
  // standalone model — it converts into `<stem>.diffusion/` of an INSTALLED
  // SoVITS model whose ContentVec dim matches; the rvc list feeds the
  // installed-model version check in onStart ----
  const sovitsModels = useVoiceModelStore((s) => s.models.sovits);
  const rvcModels = useVoiceModelStore((s) => s.models.rvc);
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
      showToast(String(e), "error");
    } finally {
      setAttaching(null);
    }
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
    if (dataset.length === 0) {
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

    const selectedVersion = config.backend === "rvc" ? config.version : config.sovitsVersion;
    const selectedSr = config.backend === "rvc" ? config.sampleRate : "44k";
    // the wipe would also destroy any diffusion training progress living in
    // this workspace — the user must see that before choosing 重训
    const diffWarn =
      info && info.diff_steps > 0
        ? " " + t("training.retrainWipesDiff", { steps: info.diff_steps })
        : "";

    if (
      wsExists &&
      info &&
      ((info.version && info.version !== selectedVersion) ||
        (info.sample_rate && info.sample_rate !== selectedSr))
    ) {
      // a version/sample-rate mismatch can NEVER be resumed (the Rust manifest
      // guard refuses it) — offering 续训 here would be a lie; retrain-only
      const choice = await showConfirm({
        title: t("training.versionMismatchTitle"),
        body:
          t("training.versionMismatchBody", {
            name,
            old: `${info.version}/${info.sample_rate}`,
            new: `${selectedVersion}/${selectedSr}`,
          }) + diffWarn,
        buttons: [
          { id: "retrain", label: t("training.retrain"), kind: "danger" },
          { id: "cancel", label: t("training.cancel") },
        ],
      });
      if (choice !== "retrain") return;
      fresh = true;
    } else if (wsExists) {
      // same-version workspace: the classic resume/retrain choice
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
      const installed = (config.backend === "rvc" ? rvcModels : sovitsModels).find(
        (m) => m.name === name,
      );
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

  const importCkpt = async (ckpt: CkptInfo) => {
    // sovits periodics are step-cadenced (several per epoch) — an epoch-keyed
    // suggestion would collide and silently replace the previous import
    const tag =
      ckpt.kind === "best"
        ? "best"
        : snapshot.backend === "sovits"
          ? `s${ckpt.step}`
          : `e${ckpt.epoch}`;
    const suggested =
      ckpt.kind === "final" ? snapshot.model_name : `${snapshot.model_name}_${tag}`;
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
      input: { initial: suggested },
    });
    if (!name || name === "__cancel") return;
    const summaryIndex = (snapshot.summary as { index?: string } | null)?.index;
    // fallbacks for runs without a summary (e.g. force-stopped): rvc keeps its
    // historical total_fea.npy; sovits probes the workspace cluster assets
    // (built before training, so they exist even for early stops)
    let indexPath = summaryIndex;
    if (!indexPath) {
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
      showToast(String(e), "error");
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
                  <span className="training-stage-msg">{cur.message}</span>
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
              {isDiff ? t("training.bestVal") : t("training.best")}:{" "}
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
                {isDiff ? t("training.sumBestVal") : t("training.sumBest")}:{" "}
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
          <div className="training-ckpt-list">
            {snapshot.ckpts.map((c) => (
              <div key={`${c.kind}-${c.step}-${c.path}`} className="training-ckpt-row">
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
                </span>
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
                  <button className="training-btn small" onClick={() => void importCkpt(c)}>
                    {t("training.import")}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* error */}
      {snapshot.state === "error" && (
        <div className="training-error-card">
          <div className="training-error-title">{t("training.doneError")}</div>
          <div className="training-error-msg">{snapshot.error}</div>
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
          <button
            className="training-btn primary"
            disabled={starting}
            onClick={() => void onStart()}
          >
            {t("training.start")}
          </button>
          <button
            className="training-btn"
            disabled={starting}
            title={t("training.clearResultTip")}
            onClick={() => void resetRun()}
          >
            {t("training.clearResult")}
          </button>
        </div>
      )}
    </div>
  );
}
