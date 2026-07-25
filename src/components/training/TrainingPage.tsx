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
import { logToBackend } from "../../lib/log";
import {
  trainingDataOk,
  backendSupportsMultiSpeaker,
  setupTrainingListeners,
  useTrainingStore,
  type CkptRecord,
  type DatasetFileRow,
  type ProjectDetail as ProjectDetailData,
  type TrainingGpu,
  type WorkspaceInfo,
  backendFamily,
  mergeCkptSources,
  segTab,
  asTrainingBackend,
  IDLE_SNAPSHOT,
  type TrainingSeg,
} from "../../store/training";
import { ProjectsStep } from "./ProjectsStep";
import { ProjectDetail } from "./ProjectDetail";
import {
  useVoiceModelStore,
  voiceFeatureDim,
  voiceVersionBadge,
  type VoiceModelEntry,
} from "../../store/voice-models";
import { AUDIO_EXT_RE, AUDIO_EXTENSIONS, fmtDur, fmtSize } from "../../lib/constants";
import { backendErrorMessage, isBusyError, isCancelError } from "../../lib/backendError";
import { maybeShowErrorModal } from "../../lib/errorDisplay";
import { lockedFieldIds, resumeWouldBeGuarded } from "../../lib/resumeLock";
import { runCandidateRangeTest, midiName } from "../../lib/vocal/rangeTest";
import { Dropdown } from "../common/Dropdown";
import { preview } from "../common/previewPlayer";
import { PreviewFileRow, useFilePreview } from "./PreviewFileRow";
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
  sovits_v2: ["import", "slice", "augment", "extract", "aug_check", "filelist", "index", "train_prep"],
  sovits_diff: ["import", "slice", "augment", "extract", "aug_check", "filelist", "diff_prep", "train_prep"],
  vocoder: ["import", "slice", "augment", "process", "aug_check", "filelist", "train_prep"],
};

/** family → its i18n label key, for the archive page header (RunStep sees only a backend id). */
const FAMILY_LABEL_KEY: Record<string, string> = {
  rvc: "backendRvc",
  sovits: "familySovits",
  sovits_v2: "backendSovits40v2",
  vocoder: "backendVocoder",
};

export function TrainingPage() {
  const { t } = useTranslation();
  const closePage = useAppStore((s) => s.toggleTrainingPage);
  const { route, setRoute, goSeg, snapshot, refresh, config, diffWsInfo, projectDataset } =
    useTrainingStore();
  const [dropActive, setDropActive] = useState(false);

  useEffect(() => {
    void setupTrainingListeners();
    void refresh();
  }, [refresh]);

  // single fetch site for the diff host's slot info (S41 共享池模式):
  // DataStep/ParamsStep/RunStep all consume the store copy via diffPoolReady.
  // S76 batch 4: keyed by PROJECT, not by the typed model name. Picking a host that belongs to
  // another project routes there first (ProjectDetail.startDiff), so the current project always
  // IS the host's project — and a rename can no longer make this probe address nothing.
  // S78: fetched for EVERY backend, not just diffusion — the parameters page needs the slot's
  // frozen values to render the resume-locked fields read-only (`resume_lock`). `diffWsInfo`
  // keeps its name and its diff-only meaning (免导入直训 asks it about the SHARED slice pool);
  // `slotInfo` is the same probe for whichever backend is selected.
  useEffect(() => {
    const pid = route.projectId;
    if (!pid) {
      useTrainingStore.getState().setDiffWsInfo(null);
      useTrainingStore.getState().setSlotInfo(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const info = await invoke<WorkspaceInfo>("get_training_slot_info", {
          projectId: pid,
          backend: config.backend,
        });
        if (cancelled) return;
        useTrainingStore.getState().setSlotInfo(info);
        useTrainingStore
          .getState()
          .setDiffWsInfo(config.backend === "sovits_diff" ? info : null);
      } catch {
        if (cancelled) return;
        useTrainingStore.getState().setSlotInfo(null);
        useTrainingStore.getState().setDiffWsInfo(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [config.backend, route.projectId]);

  // Does the CURRENT project hold a reusable flat (single-speaker) dataset? Derived HERE, keyed
  // on route.projectId, so it stays correct on every path — including「训练中直落运行段」, which
  // switches project via setRoute and never mounts ProjectDetail. Consumed via poolReusable so
  // an existing flat project trains without being sent back to re-import.
  useEffect(() => {
    const pid = route.projectId;
    if (!pid) {
      useTrainingStore.getState().setProjectInfo(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const d = await invoke<ProjectDetailData>("get_training_project", { projectId: pid });
        if (!cancelled) useTrainingStore.getState().setProjectInfo(d);
      } catch {
        if (!cancelled) useTrainingStore.getState().setProjectInfo(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [route.projectId]);

  const running = snapshot.state === "starting" || snapshot.state === "running";

  // 训练中打开训练页 = 直接落到运行段。
  //
  // 以前这件事「能用」纯属巧合:向导恰好停在第 4 段,所以刷新一次就回到第 1 段,训练还在跑却
  // 找不到它。做成显式路由之后要小心两点:
  //  · 挂载首帧的 snapshot 是 IDLE_SNAPSHOT(refresh() 是异步的),这一帧判「没在跑」是错的,
  //    所以只在真的看到 starting/running 时才落;
  //  · 只落一次(ref 守卫)。否则用户在训练中主动切到别的段,会被反复拽回来。
  const landedRef = useRef(false);
  /** Where the page was WHEN IT OPENED. `route` survives closing the page (module-level store),
   *  so "did the user open this page while parked on some other project?" can only be asked of
   *  the mount-time value — asking it of the live one made the answer change under us. */
  const openedAtRef = useRef(route.projectId);
  useEffect(() => {
    if (landedRef.current || !running || !snapshot.project_id) return;
    // Never yank a user who was deliberately looking at ANOTHER project when they opened the
    // page. Besides the jump itself, this effect rewrites `config`, so from another project it
    // would re-label that project's half-filled form with this run's name and backend.
    //
    // ★ Anchored on the MOUNT-time route, and latched either way. Keyed on the live
    // `route.projectId` (which is a dependency of this effect) it re-armed on every navigation:
    // the very next thing such a user does is press「← 全部项目」, which sets it to "" — the
    // guard then read as「no project selected」and threw them into the running project's run
    // segment instead of the list, with `updateConfig` undoing the clear `enterProject` had
    // just performed.
    landedRef.current = true;
    if (openedAtRef.current !== "" && openedAtRef.current !== snapshot.project_id) return;
    // 让整页与这次运行一致:存档列表、参数回显、试听都读 config
    const backend = asTrainingBackend(snapshot.backend);
    useTrainingStore.getState().updateConfig({
      ...(backend ? { backend } : {}),
      modelName: snapshot.model_name,
    });
    setRoute({ seg: "run", projectId: snapshot.project_id });
  }, [running, snapshot.project_id, snapshot.backend, snapshot.model_name, setRoute]);

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
          // S78: the singers are the project's on-disk directories; a card's identity is its slug.
          const groups = st.projectDataset?.groups ?? [];
          if (!backendSupportsMultiSpeaker(st.config.backend) || groups.length <= 1) {
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
          return groups[0]?.slug ?? null; // fallback: the first singer
        };
        const setHover = (id: string | null) => {
          const st = useTrainingStore.getState();
          if (st.dragOverSpeakerId !== id) st.setDragOverSpeakerId(id);
        };
        if (p.type === "enter") {
          // don't invite a drop we'll refuse: adding to the dataset only affects
          // the NEXT run, so while one is live we accept nothing (matches the
          // Arrangement convention: no affordance for a drop that won't import).
          // S76 batch 4: the project list is the same case — a drop there has no
          // project to import INTO, so it must not light up either.
          dragAccept =
            !liveNow() &&
            !!useTrainingStore.getState().route.projectId &&
            p.paths.some((pp) => AUDIO_EXT_RE.test(pp));
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
          // S76 batch 4: a drop is「往某个项目里导数据」. On the landing there is no project to
          // import INTO, so an audio drop there has no destination — ignore it rather than
          // route somewhere arbitrary. (`goSeg` refuses a project-less jump too; this keeps the
          // files from being staged into a form nothing will read.)
          if (!st.route.projectId) return;
          // S78: a drop IMPORTS — the files land in `<project>/dataset/` immediately. The target
          // is a singer directory when the project has singers, else the flat dataset. Failures
          // surface here because there is no form left to leave the files sitting in.
          const groups = st.projectDataset?.groups ?? [];
          const slug = backendSupportsMultiSpeaker(st.config.backend)
            ? (target ?? groups[0]?.slug ?? null)
            : null;
          const owner = groups.find((g) => g.slug === slug);
          st.goSeg("data");
          void st.importIntoProject(audio, owner ? owner.name || owner.slug : null).catch((e) => {
            const msg = backendErrorMessage(e) ?? String(e);
            if (!maybeShowErrorModal(e, msg)) useAppStore.getState().showToast(msg, "error");
          });
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

  // 「数据满足了吗」= 项目**盘上**有没有这个架构能吃的数据(S78:导入即落盘,不再有暂存表)。
  const hasProject = route.projectId !== "";
  // 一次运行的身份 = 项目 + 架构槽 + 本次训练名。名字只在详情页点某张槽卡片时才产生
  //(askRunName,已跑过的槽直接沿用冻结的旧名),换项目会清掉它。没有它就不该往参数/运行段走:
  // 那两段的每个动作最终都要把这个名字发给后端当产物身份,而中间没有任何一处能填它。
  const runNameSet = config.modelName.trim() !== "";
  const step3Ok =
    hasProject && runNameSet && trainingDataOk(config.backend, projectDataset, diffWsInfo);
  // 运行段属于「这个项目的这次运行」。以前的判据是「有任何非 idle 的 snapshot」——那时没有项目
  // 概念,所以够用;现在它有两个洞:①没选项目时也为真,tab 亮着但一点就被防逃课弹回;
  // ②在项目 B 上会亮出项目 A 的运行结果。运行中的 run 由 project_id 认领(批 1 起每次运行都带)。
  const runIsHere = hasProject && snapshot.state !== "idle" && snapshot.project_id === route.projectId;
  const step4Ok = step3Ok || runIsHere;
  const stepOk = [true, hasProject, step3Ok, step4Ok];
  const tab = segTab(route.seg);

  // 防逃课 invariant (S41 用户实测报的):任何让当前段失效的路径(清空结果后没有数据、
  // 后端从浅扩散切走……)都要把用户弹回去,而不是留在一个点不动的段上。
  //
  // 落点是**项目详情**,不是项目列表:失效的是「这个项目的下一步」,把人一路踢回卡片墙
  // 等于说「你刚才干的事整个作废了」。没有项目才回列表(那是唯一说得通的落点)。
  useEffect(() => {
    if (!hasProject) {
      if (route.seg !== "projects") setRoute({ seg: "projects", projectId: "" });
      return;
    }
    if ((route.seg === "params" && !step3Ok) || (route.seg === "run" && !step4Ok)) {
      goSeg("detail");
    }
  }, [route.seg, hasProject, step3Ok, step4Ok, goSeg, setRoute]);
  const steps = [
    t("training.step1"),
    t("training.step2"),
    t("training.step3"),
    t("training.step4"),
  ];
  /** Tab 1 覆盖两个段:有项目就回它的详情,没有就是列表本身。 */
  const tabSeg = (n: 1 | 2 | 3 | 4): TrainingSeg =>
    n === 1 ? (hasProject ? "detail" : "projects") : n === 2 ? "data" : n === 3 ? "params" : "run";

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
              className={`training-step-tab ${tab === n ? "active" : ""}`}
              disabled={!enabled}
              onClick={() => setRoute({ seg: tabSeg(n), projectId: route.projectId })}
            >
              <span className="training-step-num">{n}</span>
              {label}
            </button>
          );
        })}
      </nav>
      <div className={`training-step-body ${tab === 1 ? "wide" : ""}`}>
        {/* 第 1 段有两个面:项目卡片墙,和选中项目的详情(架构槽在这里选)。 */}
        {route.seg === "projects" && <ProjectsStep />}
        {route.seg === "detail" && <ProjectDetail />}
        {route.seg === "data" && <DataStep />}
        {route.seg === "params" && <ParamsStep />}
        {route.seg === "run" && <RunStep />}
        {/* 存档中心:从项目详情的槽卡片进来,复用 RunStep 的产物逻辑(试听/导入/挂接),
            只是隐去运行段的开始/进度/结算外壳。零重复——字面同一个组件。 */}
        {route.seg === "archive" && <RunStep archiveOnly />}
      </div>
      {/* ①c: on the DATA segment with ≥2 singers the per-card highlight IS the
          drop affordance, so suppress the full-screen overlay there; on other
          segments the cards aren't mounted, so keep the overlay (an off-segment
          drop routes to singer #1 by design) */}
      {dropActive &&
        !(
          backendSupportsMultiSpeaker(config.backend) &&
          (projectDataset?.groups.length ?? 0) > 1 &&
          route.seg === "data"
        ) && <div className="training-drop-overlay">{t("training.dropHint")}</div>}
    </div>
  );
}

/* -------------------- step 2 (since the S41 order swap): data -------------------- */

// PreviewPlayer extracted to components/common/previewPlayer.ts in S41 (the
// audition rows share it; singleton = data-step preview and audition playback
// preempt each other, which is the intended behavior)

// Scrubber extracted to components/common/Scrubber.tsx (S66) — the workflow node output
// preview shares it. The data-list margin moved to .training-scrubber-slot (caller-owned).

/**
 * 第 2 段 · 数据 —— 项目数据集的编辑器(S78 批 5)。
 *
 * 以前这里是一张**新导入的暂存表**:选好文件,等训练开始时才整体写进 `<project>/dataset/`。
 * 于是「项目上方写着有数据、点进来却是空表」,而且在这里选一个文件会**整体替换**已经攒好的
 * 一整份数据。现在导入是它自己的动作:文件立刻落盘,这一页显示的就是磁盘上的真实内容。
 *
 * 三条不变量:
 *  1. 唯一真源是磁盘。每次增删都以 `refreshProjectDataset()` 收尾,界面从不自己打补丁。
 *  2. 歌手的**顺序就是 emb_g 行号**(0 起,与 config.spk、推理歌手下拉同一套编号)。
 *  3. 歌手集合被任一架构冻结后,增删歌手会被后端硬拒(`DATASET_SPEAKERS_FROZEN`);
 *     给已有歌手增删文件不受限,只是下次续训要重做特征提取。
 */
function DataStep() {
  const { t } = useTranslation();
  const {
    goSeg,
    config,
    diffWsInfo,
    projectDataset,
    importIntoProject,
    deleteFromProject,
    dragOverSpeakerId,
    flashSpeaker,
    clearFlashSpeaker,
  } = useTrainingStore();
  const showConfirm = useAppStore((s) => s.showConfirm);
  const showToast = useAppStore((s) => s.showToast);
  const [busy, setBusy] = useState(false);

  const ds = projectDataset;
  const groups = ds?.groups ?? [];
  const flat = ds?.entries.filter((e) => !e.rel.includes("/")) ?? [];
  // ①c: SoVITS (α) + RVC (α′) + v2 take a SINGER LIST; diff/vocoder are flat-only.
  const singerList = backendSupportsMultiSpeaker(config.backend);
  const dataOk = trainingDataOk(config.backend, ds, diffWsInfo);

  // The preview reads files by absolute path; the presence check re-reads the CURRENT listing
  // because a long decode outlives its gesture and the row may be deleted by then.
  const dsRef = useRef(ds);
  dsRef.current = ds;
  const p = useFilePreview((path) => {
    const d = dsRef.current;
    return !!d && d.entries.some((e) => `${d.datasetDir}/${e.rel}` === path);
  });

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      const msg = backendErrorMessage(e) ?? String(e);
      if (!maybeShowErrorModal(e, msg)) showToast(msg, "error");
    } finally {
      setBusy(false);
    }
  };

  const pickAudio = async () => {
    const picked = await open({
      multiple: true,
      filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
      title: t("training.addFiles"),
    });
    if (!picked) return null;
    return Array.isArray(picked) ? picked : [picked];
  };

  const addTo = async (speaker: string | null) => {
    const files = await pickAudio();
    if (!files) return;
    await run(() => importIntoProject(files, speaker));
  };

  /** A new singer only exists once it has audio — an empty directory is a speaker with no data,
   *  which the start guard refuses anyway. So the name prompt is followed straight by the file
   *  picker, and nothing is created if the user backs out of either. */
  const addSinger = async () => {
    const taken = new Set(groups.map((g) => (g.name || g.slug).trim()));
    const name = await showConfirm({
      title: t("training.addSpeaker"),
      body: t("training.addSpeakerBody"),
      buttons: [
        { id: "__cancel", label: t("training.cancel") },
        { id: "ok", label: t("training.next"), kind: "primary" },
      ],
      input: {
        initial: "",
        invalid: (v) =>
          !v.trim()
            ? t("backend.TRAINING_SPEAKER_NAME_EMPTY")
            : taken.has(v.trim())
              ? t("backend.TRAINING_SPEAKER_NAME_DUP")
              : null,
      },
    });
    if (!name || name === "__cancel") return;
    const files = await pickAudio();
    if (!files) return;
    await run(() => importIntoProject(files, name.trim()));
  };

  /** Deleting is confirmed only when it COSTS something: with a trained slot in the project the
   *  next run has to redo slicing + feature extraction. With nothing trained it is free, and a
   *  dialog per file would just be in the way. */
  const removeFile = async (rel: string, label: string) => {
    if (useTrainingStore.getState().projectHasProgress) {
      const ok = await showConfirm({
        title: t("training.datasetRemoveTitle"),
        body: t("training.datasetRemoveBody", { name: label }),
        buttons: [
          { id: "__cancel", label: t("training.cancel") },
          { id: "go", label: t("training.remove"), kind: "danger" },
        ],
      });
      if (ok !== "go") return;
    }
    if (ds) p.stopIfPlaying(`${ds.datasetDir}/${rel}`);
    await run(() => deleteFromProject([rel]));
  };

  const removeSinger = async (slug: string, label: string, files: number) => {
    const ok = await showConfirm({
      title: t("training.removeSpeaker"),
      body: t("training.datasetRemoveSpeakerBody", { name: label, count: files }),
      buttons: [
        { id: "__cancel", label: t("training.cancel") },
        { id: "go", label: t("training.remove"), kind: "danger" },
      ],
    });
    if (ok !== "go") return;
    const rels = (ds?.entries ?? [])
      .filter((e) => e.rel.startsWith(`${slug}/`))
      .map((e) => e.rel);
    if (ds) rels.forEach((r) => p.stopIfPlaying(`${ds.datasetDir}/${r}`));
    await run(() => deleteFromProject(rels));
  };

  const row = (e: DatasetFileRow, lead?: React.ReactNode) => (
    <PreviewFileRow
      key={e.rel}
      p={p}
      path={`${ds!.datasetDir}/${e.rel}`}
      title={e.rel}
      lead={lead}
      // No original name = imported before the annotation existed; showing the on-disk name is
      // the honest fallback.
      name={e.name || e.rel}
      meta={e.durationMs != null ? fmtDur(e.durationMs / 1000) : fmtSize(e.bytes)}
      onRemove={busy ? undefined : () => void removeFile(e.rel, e.name || e.rel)}
    />
  );

  const total = (entries: DatasetFileRow[]) => {
    const ms = entries.reduce((a, e) => a + (e.durationMs ?? 0), 0);
    return (
      <span className="training-data-total">
        {t("training.files", { count: entries.length })}
        {ms > 0 && <> · {t("training.totalDur", { dur: fmtDur(ms / 1000) })}</>}
      </span>
    );
  };

  return (
    <div className="training-data-step">
      <div className="training-hint">{t("training.dataHint")}</div>

      {singerList && groups.length > 0 ? (
        // ── 多歌手:每位歌手一张卡,顺序 = emb_g 行号 ──────────────────────────
        <div className="training-spk-stack">
          {groups.map((g, i) => {
            const mine = (ds?.entries ?? []).filter((e) => e.rel.startsWith(`${g.slug}/`));
            return (
              <div
                key={g.slug}
                // the drag hit-test anchor — the slug is the identity now, not a draft id
                data-spk-id={g.slug}
                className={`training-spk-group${dragOverSpeakerId === g.slug ? " drop-target" : ""}`}
              >
                {flashSpeaker?.id === g.slug && (
                  <span
                    key={flashSpeaker.nonce}
                    className="training-spk-flash"
                    onAnimationEnd={() => clearFlashSpeaker(flashSpeaker.nonce)}
                  />
                )}
                <div className="training-spk-header">
                  {/* 0-based: this IS the emb_g row, the number a manual rebuild must reproduce */}
                  <span className="training-spk-idx" title={t("training.projectDatasetOrderNote")}>
                    {ds?.orderKnown ? i : "?"}
                  </span>
                  <span className="training-spk-name tproj-ds-spk-name" title={g.slug}>
                    {g.name || g.slug}
                  </span>
                  <span className="training-spk-count">
                    {t("training.files", { count: g.files })} · {fmtSize(g.bytes)}
                  </span>
                  <button
                    className="training-file-remove"
                    disabled={busy}
                    onClick={() => void removeSinger(g.slug, g.name || g.slug, g.files)}
                    title={t("training.removeSpeaker")}
                  >
                    X
                  </button>
                </div>
                <div className="training-data-actions">
                  <button
                    className="training-btn"
                    disabled={busy}
                    onClick={() => void addTo(g.name || g.slug)}
                  >
                    {t("training.addFiles")}
                  </button>
                </div>
                <div className="training-file-list training-spk-files">
                  {mine.map((e) => row(e))}
                </div>
              </div>
            );
          })}
          <button className="training-btn training-spk-add" disabled={busy} onClick={() => void addSinger()}>
            {t("training.addSpeaker")}
          </button>
          <div className="training-hint training-spk-hint">
            {ds?.orderKnown
              ? t("training.multiSpeakerHint")
              : t("training.projectDatasetOrderUnknown")}
          </div>
        </div>
      ) : (
        // ── 平铺(单歌手 / 浅扩散 / 声码器)───────────────────────────────────
        <>
          <div className="training-data-actions">
            <button className="training-btn" disabled={busy} onClick={() => void addTo(null)}>
              {t("training.addFiles")}
            </button>
            {total(flat)}
            {singerList && flat.length > 0 && (
              // Turning a flat project into a co-training one: the existing audio stays where it
              // is, the new singer gets its own directory. Refused by the backend once a slot has
              // frozen the (single-speaker) structure, with a CODE that says why.
              <button className="training-btn" disabled={busy} onClick={() => void addSinger()}>
                {t("training.addSpeaker")}
              </button>
            )}
          </div>
          {flat.length === 0 ? (
            <div className="training-empty">
              {t("training.empty")}
              {config.backend === "sovits_diff" && dataOk && (
                <div className="training-fixed-note">{t("training.diffPoolHint")}</div>
              )}
            </div>
          ) : (
            <div className="training-file-list">{flat.map((e) => row(e))}</div>
          )}
        </>
      )}

      <div className="training-step-nav">
        {/* 数据属于**项目**,训练属于**架构槽**。从项目详情的「导入数据」进来是没有槽的
            ——参数/运行段的每个动作都要一个「本次训练名」,而这里没有任何地方能填。所以这条
            路的下一步是回项目去选架构,而不是一颗点不动的「下一步」。 */}
        {config.modelName.trim() === "" ? (
          <button className="training-btn primary" onClick={() => goSeg("detail")}>
            {t("training.pickSlot")}
          </button>
        ) : (
          <button
            className="training-btn primary"
            disabled={!dataOk || busy}
            onClick={() => goSeg("params")}
          >
            {t("training.next")}
          </button>
        )}
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
  const { config, updateConfig, goSeg, diffWsInfo, slotInfo, retrainIntent } = useTrainingStore();
  const [gpus, setGpus] = useState<TrainingGpu[]>([]);
  /** S75: "usable GPU" = at least one SELECTABLE entry. A list of nothing but greyed-out cards
   *  must land on the force-CPU path, not on a dropdown where every choice is dead. */
  const [gpuOk, setGpuOk] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  // diff inherits 数据增强份数 from the host workspace manifest — show the
  // REAL inherited value (store diffWsInfo, fetched by the root effect)
  // 有没有主模型共用这个槽的切片池 —— 判据与后端 `eff_aug_copies` 同源(has_main_progress)
  const diffHasHost = config.backend === "sovits_diff" && !!diffWsInfo?.has_main_progress;
  const diffAugInherit =
    config.backend === "sovits_diff" && diffWsInfo?.exists ? diffWsInfo.aug_copies : null;

  useEffect(() => {
    void (async () => {
      try {
        // S67: the dropdown consumes training_gpus (accelerator-native identities:
        // NVIDIA UUID / vendor index) — NEVER the display string's WMI positions,
        // which silently CPU'd multi-adapter boxes via a wrong visibility mask.
        // "No trainable GPU" now means no NVIDIA/AMD/Intel adapter at all, instead
        // of the old cuda_available (an INFERENCE-runtime probe that wrongly forced
        // AMD/Intel — and NVIDIA-without-CUDA-download — boxes to CPU training).
        // S75: entries now carry selectable/reason (the S74b shape). Unsupported or
        // pack-less cards are LISTED with their reason but cannot be chosen — before, every
        // adapter of the winning vendor was offered unconditionally, so picking a card our
        // runtime cannot drive trained on the CPU with no visible hint.
        const hw = await invoke<{ training_gpus: TrainingGpu[] }>("get_hardware_info");
        const list = hw.training_gpus ?? [];
        setGpus(list);
        const usable = list.filter((g) => g.selectable);
        setGpuOk(usable.length > 0);
        const cur = useTrainingStore.getState().config.gpu;
        if (list.length === 0) {
          // NO adapter at all: nothing to pick, ever. Keep the PAYLOAD truthful, not just the
          // checkbox display. (Pre-S75 behaviour, unchanged — and deliberately the ONLY case
          // that force-checks CPU for the user.)
          useTrainingStore.getState().updateConfig({ forceCpu: true });
        } else if (usable.length === 0) {
          // Adapters exist but none is usable (no pack installed / card unsupported). We must
          // NOT auto-check force-CPU here: `refuse_cpu_only_runtime` returns Ok immediately when
          // force_cpu is set, so doing so would silently disarm the S68b guard that exists for
          // exactly this machine ("GPU present, only the CPU pack installed") and hand the user
          // a multi-hour CPU run with no warning. Leave the flag alone, show the reasons, and
          // let the backend refuse loudly (S75 review).
        } else if (!usable.some((g) => g.id === cur)) {
          // heal "" (fresh session), stale identities, AND an id that is still listed but no
          // longer selectable (pack deleted since the last run) — first USABLE entry wins
          useTrainingStore.getState().updateConfig({ gpu: usable[0]!.id });
        }
      } catch {
        setGpus([]);
      }
    })();
  }, []);

  // sovits-family: 4.1/4.0 and 4.0-v2 share the same param form; v2-only
  // differences (no vol_embedding / fp16 / all_in_mem) are gated inline below
  const sovits = config.backend === "sovits" || config.backend === "sovits_v2";
  const sovitsV2 = config.backend === "sovits_v2";
  const diff = config.backend === "sovits_diff";
  const voc = config.backend === "vocoder";

  // 续训锁:这些值已经写进了槽里现有的产物(图形状、线上输入、emb_g 行、缓存的 ContentVec
  // 空间),续训改不了 —— 后端会拒。所以这里**原地只读显示**,而不是让人改完、到开始训练时
  // 才被拒绝。表在 `lib/resumeLock.ts`,与 Rust 的 `resume_lock.rs` 有跨语言对拍 gate。
  // 「再训一个」会清空这个架构 ⇒ 什么都没被烧进去了,锁自然解除(与后端 `!fresh` 同义)。
  const guarded = !retrainIntent && resumeWouldBeGuarded(config.backend, slotInfo);
  const locked = guarded ? lockedFieldIds(config.backend, "locked") : new Set<string>();
  /** 锁定项的只读渲染:值 + 悬停解释。要改只能在开始训练时选「重训」。 */
  const fixed = (v: string) => (
    <span className="training-fixed-value" title={t("training.resumeLockedTip")}>
      {v} · {t("training.resumeLocked")}
    </span>
  );

  // S68b: an empty GPU list used to hide ALL device UI (dropdown AND the force-CPU
  // checkbox) while silently forcing CPU — a community RTX 3080 box with a dead GPU
  // probe trained on CPU with zero visual hint. The CPU fact now shows on the form.
  // S75 — three states, not two. The middle one is new and is the one that used to lie:
  //   no adapter at all      → "no trainable GPU" (true)
  //   adapters, none usable  → the LIST, each with its reason. Saying "no trainable GPU" here
  //                            would be false (the card is there; a pack is missing), and the
  //                            only actionable CODE we have would never reach the screen.
  //   at least one usable    → the picker
  const gpuOption = (g: TrainingGpu) => {
    const why = g.selectable ? null : (backendErrorMessage(g.reason) ?? g.reason ?? "");
    return {
      value: g.id,
      label: why ? `${g.label} — ${why}` : g.label,
      title: why ? `${g.label}\n${why}` : g.label,
      disabled: !g.selectable,
    };
  };
  const gpuRow =
    gpus.length === 0 ? (
      <div className="training-form-row">
        <label>{t("training.gpu")}</label>
        <span className="training-cpu-note">{t("training.noGpuCpuNote")}</span>
      </div>
    ) : !config.forceCpu ? (
      <>
        <div className="training-form-row">
          <label>{t("training.gpu")}</label>
          <Dropdown
            value={config.gpu}
            options={gpus.map(gpuOption)}
            onChange={(v) => updateConfig({ gpu: v })}
          />
        </div>
        {!gpuOk && (
          <div className="training-form-row">
            <label />
            <span className="training-cpu-note">{t("training.noUsableGpuNote")}</span>
          </div>
        )}
      </>
    ) : null;

  // S75: gated on "adapters exist", not "a usable one exists". When every card is greyed out,
  // opting into CPU is the user's only way forward — hiding the checkbox there (the old `gpuOk`
  // gate) left them with a dead picker and no exit.
  const forceCpuRow = gpus.length > 0 && (
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
              {locked.has("kStepMax") ? (
                fixed(
                  config.diffKStepMax === 0
                    ? t("training.kStepFull")
                    : String(config.diffKStepMax),
                )
              ) : (
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
              )}
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
            {locked.has("version") ? (
              fixed(config.version)
            ) : (
              <Dropdown
                value={config.version}
                options={[
                  { value: "v2", label: "v2" },
                  { value: "v1", label: "v1" },
                ]}
                onChange={(v) => updateConfig({ version: v })}
              />
            )}
          </div>
          <div className="training-form-row">
            <label>{t("training.sampleRate")}</label>
            {locked.has("sampleRate") ? (
              fixed(config.sampleRate)
            ) : (
              <Dropdown
                value={config.sampleRate}
                options={[
                  { value: "48k", label: "48k" },
                  { value: "40k", label: "40k" },
                  { value: "32k", label: "32k" },
                ]}
                onChange={(v) => updateConfig({ sampleRate: v })}
              />
            )}
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
        {locked.size > 0 && <div>{t("training.resumeLockedNote")}</div>}
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
              {/* Shallow diffusion trains on the SoVITS slot's own slice pool and re-fingerprints
                  it, so with a main model in that slot the count is INHERITED — choosing another
                  one would rebuild the shared slices and silently change the data the main model
                  resumes on (same posture as loudnorm). With no main model there (diff-first)
                  nothing is sharing the pool, so it is an ordinary choice. */}
              {diffHasHost ? (
                <span className="training-fixed-value">
                  {t("training.augFollowWorkspace")}
                  {diffAugInherit !== null
                    ? ` · ${t("training.augInheritCount", { count: diffAugInherit })}`
                    : ""}
                </span>
              ) : (
                <NumberField
                  value={config.diffAugCopies}
                  min={0}
                  max={3}
                  onChange={(v) => updateConfig({ diffAugCopies: v })}
                />
              )}
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
            {!sovitsV2 && config.sovitsVersion === "4.1" && (
              locked.has("volEmbedding") ? (
                <div className="training-form-row">
                  <label>{t("training.volEmbedding")}</label>
                  {fixed(t(config.sovitsVolEmbedding ? "training.on" : "training.off"))}
                </div>
              ) : (
                <label className="training-check-row">
                  <input
                    type="checkbox"
                    checked={config.sovitsVolEmbedding}
                    onChange={(e) => updateConfig({ sovitsVolEmbedding: e.target.checked })}
                  />
                  {t("training.volEmbedding")}
                </label>
              )
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
            {/* v2 (VISinger2) is pure fp32 upstream and its loader has no
                all-in-mem mode — hidden per the "不能选的隐藏" rule */}
            {!sovitsV2 && (
              <label className="training-check-row">
                <input
                  type="checkbox"
                  checked={config.sovitsFp16}
                  onChange={(e) => updateConfig({ sovitsFp16: e.target.checked })}
                />
                {t("training.fp16")}
              </label>
            )}
            {!sovitsV2 && (
              <label className="training-check-row">
                <input
                  type="checkbox"
                  checked={config.sovitsAllInMem}
                  onChange={(e) => updateConfig({ sovitsAllInMem: e.target.checked })}
                />
                {t("training.allInMem")}
              </label>
            )}
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
        <button className="training-btn primary" onClick={() => goSeg("run")}>
          {t("training.next")}
        </button>
      </div>
    </div>
  );
}

/* ---------------------------------- step 4: run ---------------------------------- */

/** `archiveOnly` = the standalone 存档中心 reached from a project's model card: same product
 *  logic (audition / import / attach), none of the run chrome (start / progress / summary). */
function RunStep({ archiveOnly = false }: { archiveOnly?: boolean } = {}) {
  const { t } = useTranslation();
  const showConfirm = useAppStore((s) => s.showConfirm);
  const showToast = useAppStore((s) => s.showToast);
  const {
    snapshot: liveSnapshot,
    snapshotAt,
    history,
    config,
    starting,
    start,
    stop,
    forceStop,
    resetRun,
    goSeg,
    route,
    diffWsInfo,
    slotInfo,
  } = useTrainingStore();
  const chartRef = useRef<LossChartHandle>(null);
  const [, forceTick] = useState(0);

  /** ★ THE run this segment is about — the store holds exactly ONE snapshot, and after S76 the
   *  page can be pointed at a project that snapshot does not belong to.
   *
   *  Without this substitution, opening project B's run segment while project A's finished run
   *  is still in memory rendered A's completion card, A's loss curve, A's候选 checkpoints and
   *  their 试听/导入 buttons — all filed under B. `mergeCkptSources` made it concrete: A's
   *  in-memory ckpts merge into B's disk scan with `mtimeMs = MAX_SAFE_INTEGER`, i.e. pinned to
   *  the TOP of B's archive list. Substituting the idle snapshot gives B exactly what it should
   *  have: the pre-start view.
   *
   *  Liveness stays keyed on the LIVE snapshot (`anyRunning`): a run in another project still
   *  blocks starting one here, and pretending otherwise would just move the refusal to Rust. */
  const snapshot = liveSnapshot.project_id === route.projectId ? liveSnapshot : IDLE_SNAPSHOT;
  const anyRunning = liveSnapshot.state === "starting" || liveSnapshot.state === "running";

  const running = snapshot.state === "starting" || snapshot.state === "running";
  const finished = snapshot.state === "completed" || snapshot.state === "stopped";
  // Which architecture this segment is ACTING on. A displayed run owns the question; with no
  // run to show (idle after「清空结果」/ a restart) it is the form's current pick. S78: the
  // audition/import/attach machine keys on THIS, not on `snapshot.backend`, so it keeps working
  // when there is no live run — during/after a run the two are identical (snapshot.state is not
  // idle then), so run behaviour is unchanged.
  // The standalone 存档中心 always speaks for the family its card chose (`config.backend`), even
  // if some OTHER project's run is live — otherwise its list would follow that run.
  const archiveBackend =
    archiveOnly || snapshot.state === "idle" ? config.backend : snapshot.backend;
  const isDiff = archiveBackend === "sovits_diff";
  // vocoder shares the "best = true validation loss" semantics with diff
  // (labels only; its checkpoints go through importCkpt, not the attach flow)
  const isVocoderRun = archiveBackend === "vocoder";

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
  // Attach is possible for a diffusion RUN (isDiff) AND for the sovits-family 存档中心, whose
  // list can hold diffusion checkpoints from a prior run. Gating on isDiff alone left the archive
  // page with ZERO candidates — the exact death-lock this whole change exists to remove (an
  // archive diffusion row would show「无可挂接的模型」and never attach). Only sovits slots ever
  // hold diffusion, so the family test is the right widening.
  const canAttachHere = isDiff || backendFamily(archiveBackend) === "sovits";
  const attachCandidates = canAttachHere
    ? sovitsModels.filter((m: VoiceModelEntry) => {
        if (!summaryDim) return true; // dim unknown (cold archive / force-stopped) — Rust re-validates
        const dim = voiceFeatureDim(m);
        return dim === null || dim === summaryDim;
      })
    : [];

  useEffect(() => {
    // refresh the installed-model list when attach becomes relevant: a finished diff run, OR the
    // archive page opening (so its host dropdown is current)
    if ((isDiff && finished) || archiveOnly) void useVoiceModelStore.getState().fetchModels();
  }, [isDiff, finished, archiveOnly]);

  // a NEW run invalidates any previously chosen target — without this reset a
  // still-valid selection from the last run survives and the default-target
  // effect below early-returns, silently pointing this run's checkpoints at
  // the previous run's model (review F16)
  useEffect(() => {
    setAttachTarget("");
  }, [snapshot.model_name, snapshot.workspace]);

  // default the target to the same-named model (the intended pairing). Runs for the archive page
  // too (canAttachHere), so a diffusion row there lands on a sensible default host.
  useEffect(() => {
    if (!canAttachHere) return;
    if (attachTarget && attachCandidates.some((m) => m.name === attachTarget)) return;
    // same-name pairing is the just-finished-diff-run nicety (snapshot.model_name is set then);
    // the archive page has no run name, so it lands on the first candidate — fine.
    const sameName = attachCandidates.find((m) => m.name === snapshot.model_name);
    setAttachTarget(sameName?.name ?? attachCandidates[0]?.name ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canAttachHere, sovitsModels, summaryDim, snapshot.model_name]);

  const attachCkpt = async (ckpt: { path: string }) => {
    if (!attachTarget || attaching) return;
    setAttaching(ckpt.path);
    try {
      await invoke("attach_diffusion", { name: attachTarget, ckptPath: ckpt.path });
      await useVoiceModelStore.getState().fetchModels();
      // Attach IS the export path for shallow diffusion — its checkpoints never go through
      // import_model, so `attach_diffusion` writes the ledger row itself (Rust side, batch 3);
      // this only re-reads the list so its「已导入」marks are current.
      await refreshArchive();
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
  const [archiveOpen, setArchiveOpen] = useState(archiveOnly);
  const projectCkpts = useTrainingStore((s) => s.projectCkpts);
  // Re-scan whenever the run's identity or state changes: a finished run just wrote new
  // archives, and「清空结果」clears the in-memory candidates while the files stay on disk.
  // Keyed on the ROUTE's project, never on the snapshot: an app restart or「清空结果」leaves
  // the snapshot idle and its project_id empty — which is exactly when this inventory is the
  // only way left to reach the files on disk. (Pre-batch-4 it resolved the project from the
  // typed model name, which is now an editable per-run label rather than an identity.)
  //
  // `archiveBackend` (defined at the top of RunStep) is THE family this segment acts on — a
  // displayed run owns it, else the form's pick. Keying anything here on `config.backend` alone
  // let the two diverge: leave a finished SoVITS run, click the RVC slot in the detail page,
  // come back to the run segment (still lit, because the run is this project's) and the
  // candidates were SoVITS while the archive list was RVC, `mergeCkptSources` filing one under
  // the other.
  useEffect(() => {
    void useTrainingStore.getState().refreshProjectCkpts(route.projectId, archiveBackend);
  }, [route.projectId, archiveBackend, snapshot.state, snapshot.ckpts.length]);
  // The scan is the truth, but it is only as fresh as its last run — a ckpt the sidecar just
  // announced would blink out of the list for the moment between the event and the re-scan
  // above, which reads exactly like a checkpoint that failed to save.
  /** Export context for the slot this segment is showing, read from DISK.
   *
   *  The三 things an export needs — the artifact name, the workspace, the index companion —
   *  used to come only from `TrainingSnapshot`, i.e. only while a run was displayed. That is
   *  exactly why a finished diffusion checkpoint went unreachable once anything else was
   *  trained. A live run still wins (its summary knows the真 index it produced); this is what
   *  answers when there is none. */
  const [slotCtx, setSlotCtx] = useState<{
    modelName: string;
    workspace: string;
    indexPath: string | null;
  } | null>(null);
  useEffect(() => {
    const pid = route.projectId;
    if (!pid) {
      setSlotCtx(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const c = await invoke<{ modelName: string; workspace: string; indexPath: string | null }>(
          "get_slot_export_context",
          { projectId: pid, backend: archiveBackend },
        );
        if (!cancelled) setSlotCtx(c);
      } catch {
        if (!cancelled) setSlotCtx(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [route.projectId, archiveBackend, snapshot.state]);

  /** 产物身份:有 run(且这一段就在看它)用 run 的,否则用槽里冻结的(再没有就用项目名)。
   *  archiveOnly 一律走槽——那时的 live snapshot 可能是别的项目的运行。 */
  const useLiveIdentity = !archiveOnly;
  const exportName =
    (useLiveIdentity ? snapshot.model_name : "") || slotCtx?.modelName || config.modelName;
  const exportWorkspace = (useLiveIdentity ? snapshot.workspace : "") || slotCtx?.workspace || "";

  const archiveRows = mergeCkptSources(
    // the standalone archive must not fold in a DIFFERENT run's in-memory candidates
    archiveOnly ? [] : snapshot.ckpts,
    projectCkpts,
    backendFamily(archiveBackend),
  );

  // ── what a given archive row can DO (S78) ─────────────────────────────────────────────────
  // The action follows the FILE, not the run. A single sovits slot can hold both a main model
  // (import) and shallow-diffusion checkpoints (attach), so a per-run `isDiff` flag cannot
  // decide it — the diffusion ones live under `.../diffusion/`.
  const rowIsDiffusion = (rel: string) => rel.replace(/\\/g, "/").includes("/diffusion/");
  /** Deployable model → import. Release/best/final weights, but NOT the diffusion ones. */
  const rowConvertible = (r: CkptRecord) =>
    !rowIsDiffusion(r.rel) && (r.kind === "release" || r.kind === "best" || r.kind === "final");
  /** Shallow-diffusion product → attach to an installed SoVITS host. */
  const rowAttachable = (r: CkptRecord) => rowIsDiffusion(r.rel) && r.kind !== "base";
  /** Auditionable = anything that renders through the inference chain. Raw resume state (G_/D_
   *  pairs, model_ckpt_steps) is not a deployable model — it has no audition. */
  const rowAuditionable = (r: CkptRecord) => rowConvertible(r) || rowAttachable(r);
  const anyAttachable = archiveRows.some(rowAttachable);


  // ①c: audition a chosen speaker of a multi-speaker rvc/sovits run. Names come from the RUN's
  // frozen speaker list (snapshot.speakers, index = emb_g id = the converter's speaker-map id) —
  // NOT the editable DataStep state, so it survives a DataStep edit and reflects what was trained.
  // Empty for single-speaker / diff / vocoder → the render falls back to speaker 0 (unchanged).
  const [auditionSpeaker, setAuditionSpeaker] = useState(0);
  // In the 存档中心 there is no run, so the singer names come from the slot's FROZEN list
  // (`slotInfo.speakers`, index = emb_g id — the same order the model was trained with); during
  // or right after a run they come from the run snapshot. Without this the archive could only
  // ever preview emb_g 0 of a multi-speaker model, silently.
  const auditionSpeakerNames = archiveOnly ? (slotInfo?.speakers ?? []) : (snapshot.speakers ?? []);
  const auditionSpeakers =
    backendSupportsMultiSpeaker(archiveBackend) && auditionSpeakerNames.length > 1
      ? auditionSpeakerNames.map((name, i) => ({ id: i, name: name.trim() || `#${i}` }))
      : [];
  // S67: the auto range battery holds this for its WHOLE duration — the old code's
  // stuck-busy accidentally blocked clicks between candidates, and un-sticking it
  // (terminal events) opened interleave windows (a user 试听 in a gap stole the
  // FlightGuard from the probe AND the probe's busy-skip then wiped the user's row).
  const [rangeTesting, setRangeTesting] = useState(false);
  const auditionBusy =
    rangeTesting ||
    Object.values(auditionState).some((s) => s === "converting" || s === "rendering");

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

  // the diffusion audition cache is host-specific — switching the host must invalidate every
  // rendered result INCLUDING one still in flight. Gated on「列表里有可挂接行」(`anyAttachable`),
  // not the per-run `isDiff`: in the 存档中心 isDiff is false but diffusion rows are auditionable,
  // and a host change there must still clear their stale renders. (Over-clearing a main-model
  // row's cache is harmless — it just re-renders on the next play.)
  useEffect(() => {
    if (!anyAttachable) return;
    auditionEpochRef.current += 1;
    setAuditionState({});
    setAuditionWavs({});
  }, [attachTarget, anyAttachable]);

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
  // S67: the dep is the STRING key, not the snapshot.ckpts array — every status
  // refresh rebuilds the snapshot arrays, and an identity-keyed rerun used to
  // alive=false this loop right after candidate #1 (the runKey ref then blocked a
  // restart, so candidates 2+ never got a range record).
  const rangeRunKey = `${snapshot.workspace}|${snapshot.ckpts.map((c) => c.path).join(",")}`;
  useEffect(() => {
    // ★ NOT in the 存档中心: this is a just-FINISHED-RUN feature (auto-test THIS run's candidates
    // for their vocal range). archiveOnly shares RunStep's hooks, and `snapshot` there may be a
    // same-project leftover run of a DIFFERENT family — firing the battery would grey every
    // archive 试听 button (auditionBusy) and burn GPU on candidates the archive never displays.
    if (archiveOnly) return;
    if (!finished || snapshot.ckpts.length === 0) return;
    if (!["rvc", "sovits", "sovits_v2"].includes(snapshot.backend)) return;
    if (candRangeRunRef.current === rangeRunKey) return;
    candRangeRunRef.current = rangeRunKey;
    const ckpts = snapshot.ckpts;
    let alive = true;
    void (async () => {
      // hold auditionBusy for the WHOLE battery — the inter-candidate gaps must not
      // invite clicks (a user render would steal the probe's FlightGuard, and a
      // busy-skipped probe wiping the user's busy row was review finding S67-1)
      setRangeTesting(true);
      try {
        for (const c of ckpts) {
          if (!alive) return;
          // persisted-record short-circuit: a candidate tested in a previous mount
          // keeps its sidecar record — restore the label instead of re-running the
          // whole battery (and re-freezing the buttons) on every page open
          try {
            const rec = await invoke<{
              speakers?: Record<string, { usable: [number, number]; comfort: [number, number] }>;
            } | null>("get_candidate_vocal_range", {
              workspace: snapshot.workspace,
              ckptPath: c.path,
            });
            const sp = rec?.speakers?.["0"];
            if (sp) {
              if (alive) {
                setCandRanges((s) => ({ ...s, [c.path]: { usable: sp.usable, comfort: sp.comfort } }));
              }
              continue;
            }
          } catch {
            /* unreadable sidecar — fall through to a fresh test */
          }
          let busySkip = false;
          try {
            const r = await runCandidateRangeTest(
              snapshot.workspace,
              snapshot.backend as "rvc" | "sovits" | "sovits_v2",
              c.path,
              c.path,
            );
            if (alive && r) setCandRanges((s) => ({ ...s, [c.path]: r }));
          } catch (e) {
            // busy (a voice render elsewhere holds the guard) or a broken ckpt — skip;
            // an untested candidate has no sidecar record, so the next mount retests it
            busySkip = isBusyError(e);
          } finally {
            // belt-and-braces vs a dropped terminal event (S67): the probe render must
            // never leave its row stranded busy — but a BUSY rejection means the probe
            // never emitted anything, so a busy phase on that row belongs to a REAL
            // audition and must survive
            if (!busySkip) {
              setAuditionState((s) => {
                if (s[c.path] !== "converting" && s[c.path] !== "rendering") return s;
                const n = { ...s };
                delete n[c.path];
                return n;
              });
            }
          }
        }
      } finally {
        setRangeTesting(false);
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished, rangeRunKey, snapshot.backend]);

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
          setAuditionState((s) => {
            if (s[candidate_id] === "playing") return s;
            // S67: a wav-less done is the range-test scale finishing — that row has
            // nothing to play, so it returns to idle instead of a cache-less ▶
            if (!wav) {
              const n = { ...s };
              delete n[candidate_id];
              return n;
            }
            return { ...s, [candidate_id]: "ready" };
          });
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
  // S78: the render command is decided PER ROW, not per run — a unified archive list can hold a
  // main model (voice) and a shallow-diffusion checkpoint (diffusion) in the same sovits slot.
  // `null` candidate = the vocoder A/B reference row.
  const auditionCandidate = async (c: { path: string } | null, mode: "voice" | "diffusion" | "vocoder") => {
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
    if (mode === "diffusion" && !attachTarget) {
      showToast(t("training.auditionNeedHost"), "error");
      return;
    }
    // stale fence: unmount / new run / host switch bump the epoch — a late
    // resolution must not populate state or start playback (FE-3/FE-5)
    const epoch = auditionEpochRef.current;
    setAuditionState((s) => ({ ...s, [id]: "converting" }));
    try {
      let wav: string;
      // S78: the workspace comes from the slot-resolved context (`exportWorkspace`), not the live
      // snapshot — so a row auditions the same whether it is a just-finished candidate or a cold
      // archive entry (with a run displayed the two are equal). The render command is `mode`,
      // decided by the CALLER per row.
      if (mode === "vocoder" || c === null) {
        wav = await invoke<string>("render_audition_vocoder", {
          ckptPath: c?.path ?? null,
          workspace: exportWorkspace,
          candidateId: id,
        });
      } else if (mode === "diffusion") {
        wav = await invoke<string>("render_audition_diffusion", {
          hostName: attachTarget,
          ckptPath: c.path,
          workspace: exportWorkspace,
          candidateId: id,
        });
      } else {
        wav = await invoke<string>("render_audition_voice", {
          // main-model render: the FAMILY, never sovits_diff (a stray main row in a diff-run view)
          backend: backendFamily(archiveBackend),
          ckptPath: c.path,
          workspace: exportWorkspace,
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
      // S74: audition runs render_audition_* (blocking inference) — leave a copyable log trace.
      logToBackend(isBusyError(e) ? "warn" : "error", `Training audition failed (${id}): ${e instanceof Error ? e.message : String(e)}`);
      // S67c: fatal modal-class errors (INFERENCE_LOW_MEMORY) open the alert dialog instead.
      const display = backendErrorMessage(e) ?? String(e);
      if (maybeShowErrorModal(e, display)) return;
      // APP_BUSY: another audition/render holds the FlightGuard → info; real failures stay errors.
      showToast(display, isBusyError(e) ? "info" : "error");
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
    // 免导入直训: a run may start with no fresh import when the project holds a reusable
    // pool (flat on-disk dataset, or a diff host's shared pool) — Rust re-verifies
    // authoritatively. ①c: multi-speaker needs ≥2 named non-empty groups (trainingDataOk).
    if (
      !trainingDataOk(config.backend, useTrainingStore.getState().projectDataset, diffWsInfo)
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
      // A failed probe must NOT read as「没有工作区」: that would skip the foreign-family
      // check and the resume/retrain dialog. Refuse loudly instead (see the main branch).
      let info: WorkspaceInfo;
      try {
        info = await invoke<WorkspaceInfo>("get_training_slot_info", {
          projectId: route.projectId,
          backend: config.backend,
        });
      } catch (e) {
        showToast(t("training.probeFailed", { err: String(e) }), "error");
        return;
      }
      if (info.exists && info.family && info.family !== "sovits") {
        showToast(t("training.diffWorkspaceForeign", { family: info.family }), "error");
        return;
      }
      let fresh = false;
      if (info.exists) {
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
      // fresh here can only come from the「重训(仅扩散)」button above = an answered dialog
      await start(fresh, fresh).catch(() => undefined);
      return;
    }

    // ⚠ `fresh` seeds to true (= WIPE) and is only narrowed inside the dialog branches below,
    // every one of which hangs off a probe. So a swallowed probe failure used to mean「没弹任何
    // 对话框就把几小时的进度整目录删了」. Every probe below is therefore fail-closed: it either
    // answers, or we refuse to start. `wipeConfirmed` carries「用户真的按了重训」to the backend,
    // which refuses an unconfirmed wipe of a workspace that holds work.
    let fresh = true;
    let wipeConfirmed = false;
    let modelExists = false;
    try {
      modelExists = await invoke<boolean>("check_model_exists", {
        name,
        modelType: config.backend,
      });
    } catch (e) {
      showToast(t("training.probeFailed", { err: String(e) }), "error");
      return;
    }
    // ⚠ FAIL-CLOSED, like every other probe on this path. `fresh` is seeded to true = WIPE and
    // is only narrowed inside the dialogs below, each of which hangs off this answer — so
    // swallowing a failure here means「没弹任何对话框就把几小时的进度整目录删了」.
    //
    // Until batch 4 this one probe was deliberately fail-OPEN, degrading to a cruder
    // `check_training_workspace(name, backend)`. That fallback made sense while the two
    // commands asked DIFFERENT questions of different code paths; once both became
    // `checked_project_id` + a path join off the same project id, they fail and succeed
    // together — the fallback was answering exactly when it could not.
    let info: WorkspaceInfo;
    try {
      info = await invoke<WorkspaceInfo>("get_training_slot_info", {
        projectId: route.projectId,
        backend: config.backend,
      });
    } catch (e) {
      showToast(t("training.probeFailed", { err: String(e) }), "error");
      return;
    }
    const wsExists = info.exists;

    // vocoder's "version" is the fixed manifest marker — hitting the sovits fallback here would
    // compare "nsf_hifigan" vs "4.1" (红队 A16). Still needed by the diffusion host check below.
    const selectedVersion =
      config.backend === "rvc"
        ? config.version
        : config.backend === "vocoder"
          ? "nsf_hifigan"
          : config.backend === "sovits_v2"
            ? "4.0-v2"
            : config.sovitsVersion;
    // the wipe would also destroy any diffusion training progress living in
    // this workspace — the user must see that before choosing 重训
    const diffWarn =
      info.diff_steps > 0 ? " " + t("training.retrainWipesDiff", { steps: info.diff_steps }) : "";

    if (wsExists) {
      // S78: the resume-guarded params can no longer DIFFER — the parameters page renders them
      // read-only from the slot's own values (`resume_lock` + `lib/resumeLock.ts`), and the
      // speaker set comes from the project's on-disk directories rather than a form. So this is
      // back to the plain resume/retrain choice.
      //
      // What used to be here was an itemized「配置与原工作区不一致」dialog: six inline t18
      // literals (outside the i18n JSON, so the parity gate never saw them) re-deriving the
      // guard's rule a third time. Keeping a copy that can only ever disagree with the source is
      // how a dialog ends up promising 续训 for a start the backend refuses.
      // Arriving via「再训一个」means the user already chose to wipe (and the parameters page
      // unlocked the resume-locked fields on that basis) — so lead with 重训 rather than making
      // them re-decide against a primary-styled 续训 that may now be refused. Both options stay:
      // this dialog is the authoritative wipe consent, and changing one's mind must be possible.
      const wantsRetrain = useTrainingStore.getState().retrainIntent;
      const choice = await showConfirm({
        title: t("training.confirmExistTitle"),
        body: t("training.confirmExistBody", { name }) + diffWarn,
        buttons: wantsRetrain
          ? [
              { id: "retrain", label: t("training.retrain"), kind: "danger" as const },
              { id: "resume", label: t("training.resume") },
              { id: "cancel", label: t("training.cancel") },
            ]
          : [
              { id: "resume", label: t("training.resume"), kind: "primary" as const },
              { id: "retrain", label: t("training.retrain"), kind: "danger" as const },
              { id: "cancel", label: t("training.cancel") },
            ],
      });
      if (choice !== "resume" && choice !== "retrain") return;
      fresh = choice === "retrain";
      wipeConfirmed = fresh;
      // (The old `else if (wsExists)` branch — "the slot exists but its facts are unreadable" —
      // is gone with the fail-open probe that produced it. `get_training_slot_info` either
      // answers or the start is refused above; an unreadable MANIFEST still lands in the branch
      // above with empty version/sample_rate fields, which is what `diffRows` skips on.)
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
      // installed model but NO workspace: nothing on disk to wipe, so this stays unconfirmed
      // (the backend guard is a no-op when the workspace holds nothing).
      fresh = true;
    }
    await start(fresh, wipeConfirmed).catch(() => undefined);
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
    // clearing from that state used to leave the wizard parked on the run
    // segment with zero data. 清空 semantically ends the round — go back to the
    // project (only on an ACCEPTED clear; a refused one keeps the results visible).
    if (await resetRun()) goSeg("detail");
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
  const suggestedName = (ckpt: { kind: string; epoch?: number; step: number | null; rel?: string }) => {
    // an archive row (CkptRecord) carries no epoch — recover it from the release filename.
    // Anchor to the FULL `_e<epoch>_s<step>` tail, never a bare `_e<digits>_`: the model slug
    // ends in an 8-hex hash, and a hash of the form `e1234567` would otherwise be read as the
    // epoch (the Rust step-parser documents this exact ~2% hazard). Fall back to the step tag.
    const epoch =
      ckpt.epoch ?? (ckpt.rel ? Number(/_e(\d+)_s\d+\./.exec(ckpt.rel)?.[1]) : NaN);
    const tag =
      ckpt.kind === "best"
        ? "best"
        : archiveBackend === "rvc" && Number.isFinite(epoch)
          ? `e${epoch}`
          : `s${ckpt.step ?? 0}`;
    return ckpt.kind === "final" ? exportName : `${exportName}_${tag}`;
  };

  // fallbacks for runs without a summary (e.g. force-stopped): rvc keeps its
  // historical total_fea.npy; sovits probes the workspace cluster assets
  // (built before training, so they exist even for early stops). Shared by the
  // single-import prompt and the S41 batch import (single source).
  const resolveIndexPath = async (): Promise<string | undefined> => {
    // ★ The live run's summary index is ONLY trustworthy when this segment is that run
    // (`useLiveIdentity`). In the standalone 存档中心 the snapshot may be a DIFFERENT family's
    // finished run of the same project, and its `summary.index` (e.g. a SoVITS cluster .npy)
    // would then be silently attached to an RVC import. The same `useLiveIdentity` gate that
    // decouples name/workspace must gate the index too — the correct one is `slotCtx.indexPath`.
    const summaryIndex = useLiveIdentity
      ? (snapshot.summary as { index?: string } | null)?.index
      : undefined;
    let indexPath = summaryIndex ?? slotCtx?.indexPath ?? undefined;
    if (!indexPath && archiveBackend !== "vocoder") {
      // vocoders have no index/cluster companion — probing would only find
      // another backend's leftovers (红队 A16 fallback-site sweep)
      if (archiveBackend === "rvc") {
        indexPath = `${exportWorkspace}\\total_fea.npy`;
      } else {
        for (const cand of [
          `${exportWorkspace}\\cluster\\kmeans_10000.pt`,
          `${exportWorkspace}\\cluster\\0.index_vectors.npy`,
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

  /** Re-read the archive list after an export, so its「已导入」marks are current.
   *
   *  S76 batch 4: this used to ALSO write the ledger row itself — a second write beside the
   *  one Rust already does inside `import_model` / `attach_diffusion` (`record_training_export`,
   *  THE single source since batch 3). Two writers was not merely redundant: they disagreed
   *  about `model_type` and the later one won, so a shallow-diffusion attach ended up recorded
   *  as `sovits_diff` — a value the registry's own type table does not know. And the frontend
   *  half was skipped entirely whenever `snapshot.project_id` was empty, i.e. after a reload
   *  or「清空结果」, which is exactly when exporting from the archive list happens. */
  const refreshArchive = async () => {
    await useTrainingStore.getState().refreshProjectCkpts(route.projectId, archiveBackend);
  };

  const importCkpt = async (ckpt: { kind: string; epoch?: number; step: number | null; rel?: string; path: string }) => {
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
        // the family this segment is acting on — `snapshot.backend` is "" with no run displayed
        modelType: backendFamily(archiveBackend),
        indexPath,
      });
      await useVoiceModelStore.getState().fetchModels();
      await refreshArchive();
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
            // family (identity for every backend the batch bar shows — it is !isDiff gated)
            modelType: backendFamily(archiveBackend),
            indexPath,
            // ★ the ledger must record the ORIGINAL checkpoint, not `path` — which may have
            // been swapped to the audition-converted onnx above. Recording the onnx would leave
            // the real snapshot looking un-imported, and batch 3's cleanup would delete it.
            sourceCkpt: path === c.path ? null : c.path,
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
      await refreshArchive();
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

  /** The project's on-disk inventory, made ACTIONABLE (S78) — rendered in EVERY run state,
   *  deliberately.
   *
   *  It first lived inside the finished-run summary card, which meant it was invisible after an
   *  app restart or「清空结果」— exactly the two situations it exists for (the sidecar's in-memory
   *  candidate list is empty then while the files are on disk). That is why a finished shallow-
   *  diffusion checkpoint became a dead end the moment anything else was trained: its attach
   *  button lived only on the summary card. Here every row carries the action its FILE supports,
   *  at any time. */
  const archiveBlock = archiveRows.length > 0 && (
      <div className="training-archive">
        <button
          className="training-archive-toggle"
          onClick={() => setArchiveOpen((v) => !v)}
        >
          {archiveOpen ? "▾" : "▸"} {t("training.archiveTitle", { count: archiveRows.length })}
        </button>
        {archiveOpen && (
          <>
            {/* diffusion rows attach to an INSTALLED SoVITS model (dim-matched) — one shared
                host selector for the whole list, same as the summary card's */}
            {anyAttachable &&
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
            {/* multi-speaker: which singer to preview. Only in the standalone 存档中心 — during a
                run the summary card carries this selector (same `auditionSpeaker` state). */}
            {archiveOnly && auditionSpeakers.length > 0 && (
              <div className="training-attach-row">
                <label title={t("training.auditionSpeakerTip")}>{t("training.auditionSpeaker")}</label>
                <Dropdown
                  value={String(auditionSpeaker)}
                  options={auditionSpeakers.map((s) => ({ value: String(s.id), label: s.name }))}
                  onChange={(v) => changeAuditionSpeaker(parseInt(v, 10))}
                />
              </div>
            )}
            <div className="training-archive-list">
              {archiveRows.map((r) => {
                const gone = missingCkpts[r.path] === true;
                const phase = auditionState[r.path];
                const diffusion = rowIsDiffusion(r.rel);
                const canAudition = rowAuditionable(r);
                const canImport = rowConvertible(r);
                const canAttach = rowAttachable(r);
                return (
                  <div
                    className={`training-archive-row${gone ? " missing" : ""}`}
                    key={r.rel}
                    title={gone ? t("training.ckptMissing") : r.path}
                  >
                    <span className="training-archive-name" title={r.path}>{r.rel}</span>
                    <span className="training-archive-tag">{t(`training.ckptKind.${r.kind}`)}</span>
                    <span className="training-archive-step">
                      {r.step != null
                        ? t("training.ckptStep", { step: r.step })
                        : // A missing step means two different things and only ONE is「最新」:
                          // RVC's「只保留最新」writes the sentinel G_2333333.pth, whereas
                          // `<slug>.pth` / `_best.pth` just carry no step. Labelling the latter
                          //「最新」would be a lie — and on _best actively misleading.
                          r.kind === "resumable"
                          ? t("training.ckptLatest")
                          : "—"}
                    </span>
                    <span className="training-archive-size">{fmtSize(r.bytes)}</span>
                    {r.imported && (
                      <span className="training-archive-tag imported">{t("training.ckptImported")}</span>
                    )}
                    {gone ? (
                      <span className="training-ckpt-missing">{t("training.ckptMissing")}</span>
                    ) : (
                      <span className="training-archive-actions">
                        {canAudition && (
                          <button
                            className="training-btn small"
                            disabled={
                              (auditionBusy && phase !== "converting" && phase !== "rendering") ||
                              (diffusion && !attachTarget) ||
                              importingAll
                            }
                            onClick={() =>
                              void auditionCandidate(
                                r,
                                diffusion ? "diffusion" : archiveBackend === "vocoder" ? "vocoder" : "voice",
                              )
                            }
                          >
                            {auditionLabel(r.path)}
                          </button>
                        )}
                        {canAttach && attachCandidates.length > 0 && (
                          <button
                            className="training-btn small"
                            disabled={!attachTarget || attaching != null}
                            onClick={() => void attachCkpt(r)}
                          >
                            {attaching === r.path ? t("training.attaching") : t("training.attach")}
                          </button>
                        )}
                        {canImport && (
                          <button
                            className="training-btn small"
                            disabled={importingAll}
                            onClick={() => void importCkpt(r)}
                          >
                            {t("training.import")}
                          </button>
                        )}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    );

  /* -------- 存档中心(独立页,从项目详情的槽卡片进入)-------- */
  if (archiveOnly) {
    return (
      <div className="tproj-detail">
        <div className="tproj-detail-head">
          <button
            className="tproj-back"
            onClick={() =>
              useTrainingStore.getState().setRoute({ seg: "detail", projectId: route.projectId })
            }
          >
            ← {t("training.archiveBack")}
          </button>
          <span className="tproj-detail-name">
            {t(`training.${FAMILY_LABEL_KEY[backendFamily(archiveBackend)] ?? "backendRvc"}`)} ·{" "}
            {t("training.archivePageTitle")}
          </span>
        </div>
        {archiveRows.length > 0 ? (
          archiveBlock
        ) : (
          <div className="training-empty">{t("training.archiveEmpty")}</div>
        )}
      </div>
    );
  }

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
              {config.modelName || "—"} · SoVITS{" "}
              {config.backend === "sovits_v2" ? "4.0-v2" : config.sovitsVersion} · 44.1k ·{" "}
              {t("training.totalEpoch")} {config.sovitsTotalEpoch} · batch{" "}
              {config.sovitsBatchSize}
            </>
          )}
        </div>
        {/* `anyRunning` = the LIVE snapshot, not this segment's view of it: a run in ANOTHER
            project still holds the single training slot, and Rust would refuse with
            TRAINING_ALREADY_RUNNING. Disabling here is the friendly first line. */}
        <button
          className="training-btn primary training-start-btn"
          disabled={starting || anyRunning}
          title={anyRunning ? t("training.active") : undefined}
          onClick={() => void onStart()}
        >
          {t("training.start")}
        </button>
        {archiveBlock}
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
                  onClick={() => void auditionCandidate(null, "vocoder")}
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
                        onClick={() =>
                          void auditionCandidate(
                            c,
                            isVocoderRun ? "vocoder" : isDiff ? "diffusion" : "voice",
                          )
                        }
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

      {archiveBlock}

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
