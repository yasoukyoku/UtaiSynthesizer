/**
 * 训练页 第 1 段 · 项目详情(S76 批 4)。
 *
 * 取代了旧的「训练对象」独占一段(TargetStep):选架构不再是一个步骤,而是在项目里点某张槽卡片
 * 的一次选择。一个项目 = 一份共享数据集 + 每种架构一个槽。
 *
 * 三处必须原样搬过来的既有能力(删 TargetStep 时最容易一起删掉的):
 *  1. **diffVersion 的派生**。浅扩散的版本不是用户选的,是宿主 SoVITS 决定的(256 维=4.0)。
 *     旧代码靠 TargetStep 里一个锁步 effect 维持;丢了它,4.0 宿主会**静默按 4.1 训练**
 *     ——不报错,只是练出来的东西对不上。
 *  2. **「附着到任意已安装 SoVITS 模型」**。浅扩散可以挂到任何已装的 SoVITS 上,不限于本项目。
 *     现在这条能力表现为「跳到那个模型所属的项目」(没有项目就建一个),语义与旧的
 *     `resolve_or_create(model_name)` 完全一致,只是变显式了。
 *  3. **「本次训练名」的冻结**。已经跑过的 run 只读显示旧名,只有全新的 run 才让用户起名
 *     (默认=项目名)。
 *     ⚠★§F2⒝ 批 2 ④b 起,这条只读**不再是数据安全的要求**:产物身份(`weights/<slug>*`、
 *     `audition/<slug>_*`、`config.spk`、池里的切片目录)已经**冻结在 run 自己的 `run.json`
 *     里**,不再每次从显示名现算 —— 改名搬不动任何字节。留着只读是因为**改名要有自己的入口
 *     和自己的闸**(运行中不许改、空名不许),而不是因为改名会毁东西。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import {
  useTrainingStore,
  trainingDataOk,
  type ProjectDetail as ProjectDetailData,
  type SlotDetail,
  type RunDetail,
  type TrainingBackend,
  type TrainingSeg,
  type WorkspaceInfo,
  type DeleteReport,
} from "../../store/training";
import { formForSlot } from "../../lib/training/formForSlot";
import {
  foldRunRows,
  newRunNameProblem,
  pickDiffHost,
  prepPoolLine,
  slotStarted,
  sortRunRows,
  startedRun,
  visibleRuns,
} from "../../lib/training/slotRows";
import {
  diffusionIsLive,
  gateReasonKey,
  liveRunIdFor,
  runRowActions,
  trainingIsLive,
  type RowGate,
} from "../../lib/training/liveRun";
import { backendErrorMessage, isBusyError } from "../../lib/backendError";
import { maybeShowErrorModal } from "../../lib/errorDisplay";
import { AUDIO_EXTENSIONS, fmtSize } from "../../lib/constants";
import { PreviewFileRow, useFilePreview } from "./PreviewFileRow";
import "./TrainingProjects.css";

/** The four architecture slots, in the order the old card grid used. `sovits_diff` is NOT one of
 *  them — it lives INSIDE the sovits slot (`diffusion/`), which is why it gets its own card
 *  below rather than a fifth entry here. */
const FAMILIES = ["rvc", "sovits", "sovits_v2", "vocoder"] as const;
type Family = (typeof FAMILIES)[number];

/** family → the EXISTING backend-card texts. Three of the four map straight onto the labels the
 *  old target step already used; only the sovits SLOT needs its own, because it holds both 4.1
 *  and 4.0 (and the shallow-diffusion progress) and no single old card described that. Writing
 *  a fresh `family.*` namespace instead would have duplicated four labels and their three
 *  translations each. */
const FAMILY_TEXT: Record<Family, { label: string; desc: string }> = {
  rvc: { label: "training.backendRvc", desc: "training.backendRvcDesc" },
  sovits: { label: "training.familySovits", desc: "training.familySovitsDesc" },
  sovits_v2: { label: "training.backendSovits40v2", desc: "training.backendSovits40v2Desc" },
  vocoder: { label: "training.backendVocoder", desc: "training.backendVocoderDesc" },
};

export function ProjectDetail() {
  const { t } = useTranslation();
  const showConfirm = useAppStore((s) => s.showConfirm);
  const showToast = useAppStore((s) => s.showToast);
  const projectId = useTrainingStore((s) => s.route.projectId);
  const setRoute = useTrainingStore((s) => s.setRoute);
  const updateConfig = useTrainingStore((s) => s.updateConfig);
  /** ★★§E2E-M25 —— 这一页此前**完全不订阅实时快照**(只有上面那三个 selector),所以训练
   *  开始/结束时它连重渲染都不会发生。
   *
   *  ⛔ 每一条都必须是**标量** selector:`training-step` 每来一步就换一个新的 snapshot 对象,
   *  订 `s.snapshot` 会让整片卡片墙按训练步频率重渲染(而 `pickDiffHost` / `slots` 的 useMemo
   *  与整棵 `FAMILIES.map` 都挂在这次渲染上,它们此前每屏只算一次)。 */
  const liveState = useTrainingStore((s) => s.snapshot.state);
  const liveProject = useTrainingStore((s) => s.snapshot.project_id);
  const liveBackend = useTrainingStore((s) => s.snapshot.backend);
  const liveRunId = useTrainingStore((s) => s.snapshot.run_id);
  const pendingStart = useTrainingStore((s) => s.starting);

  const [detail, setDetail] = useState<ProjectDetailData | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** The file list is collapsed by default: a real dataset is hundreds of rows, and the counts
   *  above already answer「有没有数据」. It is the「当初导入的到底是什么」question that needs it. */
  const [showFiles, setShowFiles] = useState(false);
  /** 试听 for the files the project already holds — same player and same row as the data step.
   *  The predicate re-reads the CURRENT listing: a decode outlives its gesture, and by the time
   *  it resolves the row may have been deleted (批 5b). */
  const detailRef = useRef<ProjectDetailData | null>(null);
  detailRef.current = detail;
  const filePreview = useFilePreview((path) => {
    const d = detailRef.current;
    if (!d) return false;
    const prefix = `${d.dataset.datasetDir}/`;
    return d.dataset.entries.some((e) => prefix + e.rel === path);
  });
  const [busy, setBusy] = useState(false);
  /** ★S143 §E2E-M25 笔 5 —— 一条**互锁**拒绝该显示成 info,不是 error。
   *
   *  ⛔ `backendError.ts` 的约定写得很清楚:`busy: true` 标的是「另一个任务占着闸,待会儿再试」
   *  这一类可重试的拒绝,漏斗把它们显示成 INFO、其余显示成 error;而这一页的四处 catch
   *  **一律写死 `"error"`**,import 块里连 `isBusyError` 都没有 ⇒ 用户在试听/渲染/分离在飞时
   *  删一个 run,吃到的是一条红色「错误」。
   *  ⚠ `maybeShowErrorModal` 不管这件事:它只处理 modal 那一档(`isModalError`),不看 busy。
   *  ⚠ 笔 1 的禁用做完之后这条路会变冷,但它同时是 `import_project_dataset` /
   *  `delete_project_dataset_files` 的档位,而那两条**不**在禁用范围里 —— 所以这一行还得对。
   *  `store/training.ts` 里那两处早就是这个写法;这里是把它补齐,不是新发明。 */
  const toastLevel = (e: unknown): "info" | "error" => (isBusyError(e) ? "info" : "error");
  /** ★S141:每个槽的 run 列表展开状态(family -> 展开?)。默认收起 —— 折叠只在超过阈值时
   *  才有东西可收,所以「默认收起」对少量 run 的槽是 no-op。 */
  const [runsOpen, setRunsOpen] = useState<Record<string, boolean>>({});

  /** Does anything in this project depend on the current data? Deleting or adding then costs a
   *  full re-extraction on the next run — worth one confirmation. With nothing trained yet it is
   *  free, and a dialog per file would just be in the way. */
  const dataHasDependents = (d: ProjectDetailData) =>
    // ★§F2⒝ 批 2 ④ —— 对**每个** run 求或。这是一道「要不要确认」的闸,漏看一个 run 就是
    // fail-open:代价落在那个 run 的几小时预处理上,而对话框根本不会弹。
    // ⛔ S141:这里原本手抄了一份 `r.hasResumePoint || r.info.has_main_progress` —— 与槽卡片
    // 那一份是同一个谓词的第二个副本,而两份会各自漂。改判「练出过东西没有」时只有一处会被
    // 想起来,另一处静默保持旧语义,而这一处的错法是 fail-open(对话框不弹)。
    d.slots.some((s) => s.ckptCount > 0 || s.runs.some(startedRun));

  const addFilesTo = async (speaker?: string) => {
    const picked = await open({
      multiple: true,
      filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
      title: t("training.addFiles"),
    });
    if (!picked) return;
    const files = Array.isArray(picked) ? picked : [picked];
    setBusy(true);
    try {
      await invoke("import_project_dataset", { projectId, files, speaker: speaker ?? null });
      await load();
    } catch (e) {
      const msg = backendErrorMessage(e) ?? String(e);
      if (!maybeShowErrorModal(e, msg)) showToast(msg, toastLevel(e));
    } finally {
      setBusy(false);
    }
  };

  /** ★§F2⒝ 批 2 ④b —— 改**这一个 run 的标签**。
   *
   *  这条能力之所以现在才敢做,是因为产物身份先被冻结在 run 自己的 `run.json` 里
   *  (`training::effective_artifact_slug`)。在那之前「改名」会在这个 run **下一次开始**时
   *  把 `hps.name`、`weights/<slug>*`、`audition/<slug>_*` 以及**共享池里**的
   *  `dataset_44k/<slug>/` 一起改指向:已有产物全变孤儿,池里凭空多一棵完整预处理树,
   *  而没有任何东西会说一句话。
   *
   *  ⚠ §F2⒝ ④d 把最后那一项拿掉了(只对**单说话人**、且池身份已是 v2 的槽):切片目录名
   *  与 `config.spk` 的键是**池**产物,现在是一个常量而不是这个 run 的名字。前三项仍然按
   *  名字派生,所以冻结身份仍然是这条能力安全的理由。
   *
   *  ⚠ 已经导出到资源管理的模型**不跟着改名** —— 导出是一次快照,那边的名字归那边管
   *  (而且按名字改动那边会撞上「同名即替换」)。文案里明写了这一条。 */
  const renameRun = async (family: Family, run: RunDetail) => {
    const typed = await showConfirm({
      title: t("training.runRenameTitle"),
      body: t("training.runRenameBody"),
      buttons: [
        { id: "ok", label: t("training.next"), kind: "primary" },
        { id: "__cancel", label: t("training.cancel") },
      ],
      input: {
        initial: run.modelName ?? "",
        // ★★S143 §E2E-M25 笔 5 —— 同槽两个 run 不许同名。此前这里**只判空**,而「再训一个」
        // 那条路早就走 `newRunNameProblem` ⇒「起两个不同名字,再把其中一个改成另一个的名字」
        // 是一条用户按得出来的路径,而后果是数据级的(同名 ⇒ 同 slug ⇒ `plan_cleanup` 会把
        // 另一个 run 的快照判成 StillInstalled 永久保留)。
        // ⛔ 第三个实参是**排除自己**:不传的话,「改成自己已有的名字」会被自己挡住。
        //    后端 `tproject::rename_run` 有同一道闸(`TRAINING_NAME_TAKEN`)—— 这一份是让用户
        //    在打字时就知道,不是唯一的守卫。
        invalid: (v) => {
          const problem = newRunNameProblem(v, slots.get(family)?.runs ?? [], run.id);
          if (problem === "empty") return t("backend.TRAINING_NAME_EMPTY");
          if (problem === "taken") return t("backend.TRAINING_NAME_TAKEN");
          return null;
        },
      },
    });
    if (!typed || typed === "__cancel") return;
    if (typed.trim() === (run.modelName ?? "")) return;
    setBusy(true);
    try {
      await invoke("rename_training_run", {
        projectId,
        backend: family,
        runId: run.id,
        name: typed,
      });
      await load();
    } catch (e) {
      const msg = backendErrorMessage(e) ?? String(e);
      if (!maybeShowErrorModal(e, msg)) showToast(msg, toastLevel(e));
    } finally {
      setBusy(false);
    }
  };

  /** ★★§F2⒝ 批 2 ④e —— 删掉**这一个 run**。槽、它的预处理池、兄弟 run 与项目数据集都不动。
   *
   *  ⛔ 只在 `run.id` **非空**时才画得出来:空串是「未迁移的槽,槽根就是那个 run」这个**肯定
   *  事实**(`RunDetail.id` 的 doc),而后端对空 id 一律 `RUN_ID_REQUIRED`。对那种槽,诚实的
   *  出口是设置里的「删除这个架构」——因为那时 run 产物与池产物混在同一个槽根上,「只删这个
   *  run」根本不是一个可分离的操作。
   *
   *  ⚠ 爆炸半径要说清楚两件事,而它们都有先例:
   *  ⒜ **浅扩散跟着走** —— 它训练在 `runs/<主 run>/diffusion/` 里,而「sovits」这个词一个字
   *     都没提到扩散(存储页的 `SlotUsage` 为同一件事专门带了一个 `diff_steps` 字段);
   *  ⒝ **已导出的模型不受影响** —— 导入是独立副本,来源没了它照样装着(项目页那一行会标
   *     「来源已删除」)。 */
  const deleteRun = async (family: Family, run: RunDetail) => {
    const steps = run.info.diff_steps ?? 0;
    const choice = await showConfirm({
      title: t("training.runDeleteTitle"),
      body:
        t("training.runDeleteBody", {
          name: run.modelName?.trim() || t("training.runUnnamed"),
          size: fmtSize(run.ckptBytes),
        }) + (steps > 0 ? " " + t("training.runDeleteDiffNote", { steps }) : ""),
      buttons: [
        { id: "cancel", label: t("training.cancel") },
        { id: "go", label: t("training.runDelete"), kind: "danger" },
      ],
    });
    if (choice !== "go") return;
    setBusy(true);
    try {
      const report = await invoke<DeleteReport>("training_delete_run", {
        projectId,
        family,
        runId: run.id,
      });
      showToast(t("training.runDeleteDone", { size: fmtSize(report.freedBytes) }), "info");
      await load();
    } catch (e) {
      const msg = backendErrorMessage(e) ?? String(e);
      if (!maybeShowErrorModal(e, msg)) showToast(msg, toastLevel(e));
    } finally {
      setBusy(false);
    }
  };

  const removeFile = async (rel: string, label: string) => {
    if (!detail) return;
    if (dataHasDependents(detail)) {
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
    filePreview.stopIfPlaying(`${detail.dataset.datasetDir}/${rel}`);
    setBusy(true);
    try {
      await invoke("delete_project_dataset_files", { projectId, rels: [rel] });
      await load();
    } catch (e) {
      const msg = backendErrorMessage(e) ?? String(e);
      if (!maybeShowErrorModal(e, msg)) showToast(msg, toastLevel(e));
    } finally {
      setBusy(false);
    }
  };

  const load = useCallback(async () => {
    if (!projectId) return;
    try {
      const d = await invoke<ProjectDetailData>("get_training_project", { projectId });
      setDetail(d);
      // The page-root effect derives `poolFlat` on route change only, so a dataset edited HERE
      // would leave it stale for the rest of the session — and it is what decides whether the
      // run may skip the data page. Refresh it from the same response.
      useTrainingStore.getState().setProjectInfo(d);
      setError(null);
    } catch (e) {
      setError(backendErrorMessage(e) ?? String(e));
    }
    // NB the TrainingPage-root effect derives the same thing on every route change — that is what
    // keeps it correct on the paths where ProjectDetail never mounts (training-in-progress lands
    // straight on the run segment via setRoute). This call is the ADDITIONAL refresh for the case
    // that effect cannot see: the dataset being edited on this very page.
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load]);

  /** ★★§E2E-M25 ⑶ —— 「此刻有没有**任何**长任务在跑」,后端 `running_tasks_of` 的镜像。
   *
   *  ⛔ 为什么不是「有没有训练在跑」:删除与改名的后端前置(`ensure_safe_to_delete` /
   *  `ensure_idle_for_run_rename`)走的都是 `running_tasks_of`,它把 training / separation /
   *  render / **audition** / 全部 `active_tasks` 一起算(那个函数的 doc 明写
   *  「Deliberately FAIL-CLOSED and coarse」)。只按训练禁,前端就比后端**更宽** ——
   *  用户在同一个训练页里试听一段存档、顺手点删除,照样吃一条拒绝。
   *
   *  ⚠ 它只能轮询:这几个标志没有事件推到前端(`running_tasks` 是一条普通命令,仓里另外两个
   *  消费点 `exitFlow.ts` 与 `Settings.tsx` 都是一次性取)。⇒ **诚实边界**:这个布尔最多晚
   *  一个 tick;晚的方向是「刚闲下来还灰着」(顶多多等两秒)与「刚忙起来还亮着」(点下去被
   *  后端拒,与今天一样)。训练那一支不受这个延迟影响 —— 它由下面的 `trainingLive` 直接短路,
   *  走的是事件驱动的快照。 */
  const [tasksBusy, setTasksBusy] = useState(false);
  useEffect(() => {
    let alive = true;
    const tick = () => {
      invoke<string[]>("running_tasks")
        .then((ts) => {
          if (alive) setTasksBusy(ts.length > 0);
        })
        .catch(() => {
          /* best-effort:探针失败时不谎报「闲着」,保持上一次的读数 */
        });
    };
    tick();
    const h = setInterval(tick, 2000);
    return () => {
      alive = false;
      clearInterval(h);
    };
    // 训练状态一变就立刻重取一次,免得那一档也要等满一个 tick。
  }, [liveState, pendingStart]);

  const slots = useMemo(() => {
    const by = new Map<string, SlotDetail>();
    for (const s of detail?.slots ?? []) by.set(s.family, s);
    return by;
  }, [detail]);

  /** Where a run of this backend goes next: straight to the parameters when the data
   *  requirement is already met, otherwise the data page.
   *
   *  Asked of the SAME predicate the step tabs and the data page's Next button use, so the two
   *  can never disagree — jumping to params on a state those gates read as unsatisfied would
   *  bounce straight back here via the 防逃课 invariant, with nothing on screen to explain why.
   *  The `poolReusable` term is what lets an EXISTING flat project train without re-importing the
   *  data it already holds (the「上方写着有数据、点训练却让我再导入」symptom). */
  const nextSegFor = (backend: TrainingBackend): TrainingSeg => {
    const st = useTrainingStore.getState();
    const ready = trainingDataOk(backend, st.projectDataset, st.diffWsInfo);
    return ready ? "params" : "data";
  };

  /** 这个 run 的**标签**。已经起过名的 run 直接沿用,只有还没起过的才问。
   *
   *  ★§F2⒝ 批 2 ④ —— 参数从槽换成 **run**:槽级取值在两个 run 之后会给**每一个** run 回答
   *  最后那个 run 的名字。
   *  ★§F2⒝ 批 2 ④b —— 它**不再是产物身份**。以前这个字符串是 `slugify` 的输入,而 slug 是
   *  `dataset_44k/<slug>/`、`config.spk` 的键和 `weights/<slug>*` 的前缀;现在身份冻结在
   *  `run.json[model_slug]` 里(`training::effective_artifact_slug`),这里只决定用户看见什么。
   *  ★§F2⒝ 批 2 ④d —— 前两项**连身份都不再是**:单说话人的切片目录与 `config.spk` 键是**池**
   *  产物,身份 v2 之后是一个常量。slug 今天只名了 `weights/<slug>*` 与 `hps.name`。 */
  const askRunName = async (run: RunDetail | undefined): Promise<string | null> => {
    if (run?.modelName) return run.modelName;
    const name = await showConfirm({
      title: t("training.runNameTitle"),
      body: t("training.runNameBody"),
      buttons: [
        { id: "ok", label: t("training.next"), kind: "primary" },
        { id: "__cancel", label: t("training.cancel") },
      ],
      input: {
        initial: detail?.name ?? "",
        invalid: (v) => (v.trim() ? null : t("backend.TRAINING_NAME_EMPTY")),
      },
    });
    if (!name || name === "__cancel") return null;
    return name;
  };

  /** Put the form back where this RUN already is —— 实现在 `lib/training/formForSlot.ts`。
   *
   *  ★§F2⒝ 批 2 ④d —— 它从这个组件里搬出去了,理由有两条,都不是整洁:
   *  ⒜ `resume_lock.rs` 的模块头把「项目页的表单还原」列为必须与守卫一致的四处之一,而这一处
   *     此前**结构上无法有闸**(组件内闭包,vitest 不做组件测试);
   *  ⒝ 它此前**只还原 locked 那一档**,漏掉的 `costly` 那一档恰好是「决定用哪个预处理池」的
   *     那些值 —— 漏还原不会被拒,而是静默换池重跑几小时,并把 manifest 里的记录一起覆盖掉。
   *  两条的完整推导与判据在那个文件的头注释与 `formForSlot.test.ts`。 */
  const formFor = (family: Family, run: RunDetail | undefined, pin?: "4.1" | "4.0") =>
    formForSlot(family, run?.info, pin);

  /** Open this architecture's 存档中心. Sets the family the archive page speaks for (it reads
   *  `config.backend`), then routes — no name prompt, no run, just the product inventory. */
  const openArchive = (family: Family) => {
    updateConfig({ backend: family });
    setRoute({ seg: "archive", projectId });
  };

  /** 继续训练 / 首次开始 —— `run` 是要继续的**那一个** run(未开始的槽只有一条占位 run)。 */
  const startFamily = async (
    family: Family,
    run: RunDetail | undefined,
    sovitsVersion?: "4.1" | "4.0",
  ) => {
    const runName = await askRunName(run);
    if (runName === null) return;
    // 继续训练 / 首次开始:this run's frozen values stand, so the params page locks them.
    useTrainingStore.getState().setRetrainIntent(false);
    updateConfig({
      backend: family,
      modelName: runName,
      // ★§F2⒝ 批 2 ④ —— 把「哪个 run」一路带到 start_training。
      // ⚠ 那句「今天恒为 ""(每槽一个 run)」随 S132 的 flip 死了:「再训一个」现在真的铸第二个
      //   run,所以这个值从此可以是真 id,而 `resolve_run_dir` 对多于一个 run **拒绝作答** ——
      //   漏穿的调用点是响亮的错误,不是悄悄写进别人的 run。(S133 R2:训练页那四处探针
      //   正是漏穿的,后果是那个槽从此开不了训。)
      runId: run?.id ?? "",
      ...formFor(family, run, sovitsVersion),
    });
    setRoute({ seg: nextSegFor(family), projectId });
  };

  /** 再训一个:说清楚代价再走。真正的擦除同意仍然只有一处 —— 运行段「开始训练」那个
   *  续训/重训对话框(后端的 wipe_confirmed 就认它)。这里不重复那道门,只负责让用户在点进去
   *  之前就知道会发生什么。 */
  const retrainFamily = async (family: Family, run: RunDetail | undefined) => {
    const slot = slots.get(family);
    // ★ The SoVITS slot holds both 4.1 and 4.0, and the version can only ever change on a
    // retrain (a resume is version-locked). With the old target step gone, this dialog is the
    // ONLY remaining way to reach 4.0 in a project whose sovits slot is 4.1 — carrying the
    // manifest version over silently would make 4.0 unreachable AND retrain a 4.0 slot as 4.1.
    const versions: ("4.1" | "4.0")[] = family === "sovits" ? ["4.1", "4.0"] : [];
    const choice = await showConfirm({
      title: t("training.slotRetrainTitle"),
      body: t("training.slotRetrainBody", {
        family: t(FAMILY_TEXT[family].label),
        size: fmtSize(slot?.ckptBytes ?? 0),
      }),
      buttons: [
        { id: "cancel", label: t("training.cancel") },
        ...(versions.length > 0
          ? versions.map((v) => ({
              id: v,
              label: t("training.slotRetrainVersion", { version: v }),
              kind: "danger" as const,
            }))
          : [{ id: "go", label: t("training.slotRetrain"), kind: "danger" as const }]),
      ],
    });
    if (choice !== "go" && !versions.includes(choice as "4.1" | "4.0")) return;
    const pin = versions.includes(choice as "4.1" | "4.0") ? (choice as "4.1" | "4.0") : undefined;
    // ★★§F2⒝ ④e —— 「再训一个」铸的是一个**新** run,所以它要一个**新**名字。
    // ⛔ 不能沿用 `askRunName`:那个在 `run.modelName` 已经有值时**直接返回它、不问** ——
    //    对「继续训练」是对的,对这里是灾难:名字是产物前缀(`weights/<slug>*`、`audition/<slug>_*`),
    //    两个 run 同名 ⇒ 同 slug ⇒ 存档页两行同名,而 `plan_cleanup` 的 `installed_stem`
    //    按 file_stem 判「还装着」,于是会把**另一个** run 的快照也判成 StillInstalled 永久保留。
    //    判据在 `lib/training/slotRows.ts` 的 `newRunNameProblem`(S141 §E2E-M24 把它搬出来:
    //    写在这里的校验闭包 vitest 结构上够不着,而 i18n 那两道闸只钉结构、看不见对话框行为)。
    const newName = await showConfirm({
      title: t("training.newRunNameTitle"),
      body: t("training.newRunNameBody"),
      buttons: [
        { id: "ok", label: t("training.next"), kind: "primary" },
        { id: "__cancel", label: t("training.cancel") },
      ],
      input: {
        initial: "",
        invalid: (v) => {
          const problem = newRunNameProblem(v, slot?.runs ?? []);
          if (problem === "empty") return t("backend.TRAINING_NAME_EMPTY");
          if (problem === "taken") return t("training.newRunNameTaken");
          return null;
        },
      },
    });
    if (!newName || newName === "__cancel") return;
    // ★ 再训一个 = **另起一个 run**,所以续训锁全部解除 —— 新 run 里什么都还没被烧进去,
    // 换采样率/换版本重练正是这颗按钮的用途。
    // ⚠ 这句话在 S132 的 flip 之前写的是「这个架构会被清空」,那已经是假的(旧 run 原样留着,
    //   `remove_dir_all_robust(&workspace)` 那一行没有了);结论不变,理由换成真的那个 ——
    //   一条**理由已死的注释**会活过重构,然后在下一个人手里背书一个错的改动。
    // 真正的擦除同意仍然只有一处(运行段那个对话框,后端的 wipe_confirmed 只认它)。
    useTrainingStore.getState().setRetrainIntent(true);
    updateConfig({
      backend: family,
      modelName: newName.trim(),
      // ★§F2⒝ ④e —— 它带的仍然是**用户按下这颗按钮的那一行 run** 的 id,而且必须是:
      // 后端每一道闸都要判**那个** run 的启动前状态(锁、家族、已有产物)。
      // ⛔ 而「写到哪里去」**不**由它决定 —— `trun::run_dir_for_start` 在 mint 时无视它并
      //    铸一个新目录。两者混为一谈正是「再训一个 = 续训并覆盖旧 run」那条静默失败。
      runId: run?.id ?? "",
      ...formFor(family, run, pin),
    });
    setRoute({ seg: nextSegFor(family), projectId });
  };

  /** 浅扩散。**训练进本项目**的 sovits 槽(`<project>/sovits/diffusion/`),用本项目 sovits 槽的
   *  ContentVec 空间 —— 那正是它要复用的预处理缓存所在。
   *
   *  「附着到任意已安装 SoVITS 模型」不在这里:它在**运行段**,练完之后按维度筛出候选再选
   *  (`attachTarget` + `attach_diffusion`),那才是附着真正发生的地方。这一页曾经放过一个
   *  「或挂到其它模型」的下拉当快捷方式,它做了三件不该做的事:把 config.modelName 设成那个
   *  已安装模型的名字后训练,而浅扩散 run 会整体重写 `<project>/sovits/run.json` —— 于是宿主槽
   *  的「本次训练名」被悄悄改掉(正是本文件第 3 条要防的);跳过去的项目若被标记 needsAttention
   *  也照进不误;模型没有项目时还会凭空建一个。 */
  const startDiff = async (version: "4.1" | "4.0", run: RunDetail | undefined) => {
    const runName = await askRunName(run);
    if (runName === null) return;
    // Fetch the slot's REAL facts BEFORE touching the form. Two things depend on them and both
    // fail quietly if guessed —
    //  · 免导入直训 (`diffPoolReady`) decides whether the data segment can be skipped, and
    //    `diffWsInfo` is written by the page's root effect, which has not re-run for this
    //    backend yet: reading it now would always see null and cost the skip;
    //  · `k_step_max` is resume-guarded (`RESUME_KSTEP_MISMATCH`) once there IS diffusion
    //    progress, so leaving the form's default (0 = full diffusion) against a slot trained
    //    shallow turns「继续训练」into a refusal at start time.
    let info: WorkspaceInfo | null = null;
    try {
      info = await invoke<WorkspaceInfo>("get_training_slot_info", {
        projectId,
        backend: "sovits_diff",
        // 浅扩散跑在**主模型那个 run** 的目录里(`runs/<主 run>/diffusion/`),两道 config.json
        // 闸就挂在「相邻」上 —— 所以问的必须是同一个 run。
        runId: run?.id ?? "",
      });
      useTrainingStore.getState().setDiffWsInfo(info);
    } catch {
      /* the data segment is the safe default */
    }
    useTrainingStore.getState().setRetrainIntent(false);
    updateConfig({
      backend: "sovits_diff",
      modelName: runName,
      runId: run?.id ?? "",
      // ★ DERIVED from the slot whenever it has one: the diffusion has to live in the same
      //   ContentVec space as the cached features it trains on (4.1 = vec768 / 4.0 = vec256).
      //   A mismatch does not error — it just produces something that does not fit its host.
      diffVersion: version,
      // ★§F2⒝ 批 2 ④d —— k_step_max 的还原(以及 diff-first 槽的增强份数)与另外四张卡走
      // **同一个**函数。此前这里自己写了一份 `diff_steps > 0` 的判断,而那正是「同一条规则的
      // 第五份副本」—— `resume_lock.rs` 的模块头说这个模块存在就是为了消灭它们。
      ...formForSlot("sovits_diff", info),
    });
    setRoute({ seg: nextSegFor("sovits_diff"), projectId });
  };

  /** ★ The error and loading states MUST keep the「← 全部项目」button on screen.
   *
   *  Returning a bare message instead removed the only way out of this segment: tab 1 maps back
   *  to `detail` whenever a project is selected, the 防逃课 invariant only fires when there is
   *  NO project, and `route` lives in a module-level store that closing the page does not reset —
   *  so a project that stops loading (deleted from Settings → 存储占用, `project.json` locked, a
   *  data drive that went away) locked the whole training page until the app was restarted.
   *  A retry belongs here too: the transient cases fix themselves. */
  const shell = (body: React.ReactNode) => (
    <div className="tproj-detail">
      <div className="tproj-detail-head">
        <button
          className="tproj-back"
          onClick={() => useTrainingStore.getState().enterProject("")}
        >
          ← {t("training.projectBack")}
        </button>
        <span className="tproj-detail-name">{detail?.name ?? projectId}</span>
      </div>
      {body}
    </div>
  );

  if (error) {
    return shell(
      <>
        <div className="training-name-exists">{error}</div>
        <div className="tproj-slot-actions">
          <button className="training-btn small" onClick={() => void load()}>
            {t("training.projectRetry")}
          </button>
        </div>
      </>,
    );
  }
  if (!detail) return shell(<div className="training-empty">{t("training.projectLoading")}</div>);

  // A flagged project is refused by `resolve_or_create` before anything can start, so its
  // actions are dead ends dressed as buttons.
  const blocked = !!detail.needsAttention;
  /** ★★§E2E-M25 —— 决策全部在 `lib/training/liveRun.ts`,这里只搬事实。
   *  ⛔ 两个谓词的作用域是**相反**的(全局 vs 本项目本槽),合成一个必然错一边;
   *  为什么、以及每一格夹具挡的是哪个错法,写在那个模块的头注与它的判据里。 */
  const liveFacts = {
    state: liveState,
    project_id: liveProject,
    backend: liveBackend,
    run_id: liveRunId,
  };
  const trainingLive = trainingIsLive(liveFacts, pendingStart);
  // 训练那一支**短路**轮询:它是事件驱动的,不该跟着 2 秒的 tick 走。
  const anyTaskLive = trainingLive || tasksBusy;
  /** 禁用原因要**上屏**(tooltip),不是给日志看的 —— 一颗禁着而不说为什么的按钮,
   *  与一颗点下去被拒的按钮相比只是把困惑换了个位置。键由 `gateReasonKey` 那张表给。 */
  /** 槽级 / 浅扩散卡上的「开始训练」—— 与 run 行的「继续 / 再训一个」是**同一条规则**
   *  (后端 `TRAINING_ALREADY_RUNNING` 是进程级单槽),只是它不针对某一行。 */
  const slotStart = runRowActions({
    blocked,
    inFlight: busy,
    trainingLive,
    anyTaskLive,
    hasRunId: true,
  }).cont;
  const gateTitle = (g: RowGate, fallback?: string): string | undefined => {
    const k = gateReasonKey(g.reason);
    return k ? t(k) : fallback;
  };
  const sovitsSlot = slots.get("sovits");
  /** ★§F2⒝ 批 2 ④ —— 浅扩散跑在**主模型那个 run** 里(`runs/<主 run>/diffusion/`),所以这张
   *  卡问的是「哪个 run 里有主模型」。决策与它的全部说明在 `lib/training/slotRows.ts`;
   *  ⛔ S141 §E2E-M4 把它搬出组件体,是因为写在这里的表达式 vitest 结构上够不着 ——
   *  那条「两个 run 都有主模型时它挑的是字典序第一个」的已知歧义,此前只有一段注释在守,
   *  现在是 `withMainProgress` 这个可断言的数(行为仍然保持现状,改它是 B2-⑤)。 */
  const diff = pickDiffHost(sovitsSlot);
  const diffHost = diff.host;
  const sovitsVersionPinned = diff.pinnedVersion;
  const diffSteps = diff.steps;

  return shell(
    <>
      {detail.note && <div className="tproj-note">{detail.note}</div>}
      {/* `resolve_or_create` / `try_start` refuse a flagged project outright. Showing live
          buttons that walk the user through picking a name and importing data, only to refuse
          at the very end, is the shape of guidance this codebase keeps removing. */}
      {detail.needsAttention && (
        <div className="training-name-exists">
          {backendErrorMessage(detail.needsAttention) ?? t("training.projectNeedsAttention")}
        </div>
      )}

      {/* ── 数据集 ───────────────────────────────────────────────────────── */}
      <div className="tproj-section">
        <div className="tproj-section-head">
          <span className="tproj-section-title">{t("training.projectDataset")}</span>
          {/* 「重新导入」而不是「管理」:数据段是一张**新导入**的暂存表,不是这个项目已有数据
              的编辑器 —— 导完会整体替换项目的共享数据集(Rust 侧还会因此拦住已有进度的兄弟槽)。
              把它叫「管理数据」会让人以为打开就能看到那 N 个文件,而那张表是空的。 */}
          <div className="tproj-ds-head-actions">
            {/* 追加 lands straight in `dataset/` — no run required, nothing replaced. The wizard's
                data page still exists for「重新导入」(it replaces the whole set), which is a
                different act and keeps its own button. */}
            {detail.dataset.groups.length === 0 && (
              <button
                className="training-btn small"
                disabled={!!detail.needsAttention || busy}
                onClick={() => void addFilesTo()}
              >
                ＋ {t("training.addFiles")}
              </button>
            )}
            <button
              className="training-btn small"
              disabled={!!detail.needsAttention || busy}
              onClick={() => setRoute({ seg: "data", projectId })}
            >
              {detail.dataset.files > 0
                ? t("training.projectReimportData")
                : t("training.projectImportData")}
            </button>
          </div>
        </div>
        {detail.dataset.files > 0 ? (
          <>
            <div className="tproj-dataset">
              <span>{t("training.projectDatasetFiles", { count: detail.dataset.files })}</span>
              <span className="tproj-dot">·</span>
              <span>{fmtSize(detail.dataset.bytes)}</span>
              {detail.dataset.speakers.length > 0 && (
                <>
                  <span className="tproj-dot">·</span>
                  <span>
                    {t("training.projectDatasetSpeakers", { count: detail.dataset.speakers.length })}
                  </span>
                </>
              )}
            </div>

            {/* ── 歌手结构 ─────────────────────────────────────────────────
                The row number IS the emb_g row id, and reproducing it is the whole point:
                rebuilding a multi-singer dataset in a different order re-assigns every
                singer's timbre, silently. So the number is printed ONLY when something on
                disk actually recorded the order (`orderKnown`) — a plausible-looking guess
                is worse here than an honest「顺序未记录」. */}
            {detail.dataset.groups.length > 0 && (
              <div className="tproj-ds-block">
                {/* The「错一位整批错位」rule is a RULE, not a state — it reads as noise pinned
                    under the table, so it lives on hover. What stays on screen is the one thing
                    that IS a state: whether the order is known at all. */}
                <div className="tproj-ds-sub" title={t("training.projectDatasetOrderNote")}>
                  {t("training.projectDatasetStructure")}
                </div>
                {detail.dataset.groups.map((g, i) => (
                  <div key={g.slug} className="tproj-ds-spk-row">
                    {detail.dataset.orderKnown && (
                      <span className="training-spk-idx">{i}</span>
                    )}
                    <span className="tproj-ds-spk-name" title={g.slug}>
                      {g.name || g.slug}
                    </span>
                    <span className="training-file-dur">
                      {t("training.projectDatasetFiles", { count: g.files })} · {fmtSize(g.bytes)}
                    </span>
                    {/* Adding to an EXISTING singer never changes the speaker set, so it is
                        allowed even after that set is frozen — the cost is a re-extraction,
                        which the row below says out loud. Creating a singer is not offered
                        here; that is a data-page act (批 5b 后续). */}
                    <button
                      className="training-btn small"
                      disabled={!!detail.needsAttention || busy}
                      onClick={() => void addFilesTo(g.name || g.slug)}
                      title={t("training.addFiles")}
                    >
                      ＋
                    </button>
                  </div>
                ))}
                {!detail.dataset.orderKnown && (
                  <div className="tproj-ds-note">
                    {t("training.projectDatasetOrderUnknown")}
                  </div>
                )}
              </div>
            )}

            {/* ── 文件列表 ─────────────────────────────────────────────── */}
            <button
              className="training-btn small tproj-ds-toggle"
              onClick={() => setShowFiles((v) => !v)}
              aria-expanded={showFiles}
            >
              {showFiles ? "▼" : "▶"} {t("training.projectDatasetFileList")}
            </button>
            {showFiles && (
              <>
                <div className="training-file-list tproj-ds-files">
                  {detail.dataset.entries.map((e) => {
                    const cut = e.rel.indexOf("/");
                    const slug = cut > 0 ? e.rel.slice(0, cut) : "";
                    const owner = slug
                      ? detail.dataset.groups.find((g) => g.slug === slug)
                      : undefined;
                    return (
                      <PreviewFileRow
                        key={e.rel}
                        p={filePreview}
                        // `rel` is forward-slashed by contract; Windows takes it as-is
                        path={`${detail.dataset.datasetDir}/${e.rel}`}
                        title={e.rel}
                        lead={
                          slug ? (
                            <span className="tproj-ds-file-spk">{owner?.name || slug}</span>
                          ) : undefined
                        }
                        // No original name = imported before the annotation existed. Showing the
                        // on-disk name is the honest fallback; inventing one is not.
                        name={e.name || e.rel}
                        meta={fmtSize(e.bytes)}
                        onRemove={busy ? undefined : () => void removeFile(e.rel, e.name || e.rel)}
                      />
                    );
                  })}
                </div>
                {detail.dataset.entries.some((e) => !e.name) && (
                  <div className="tproj-ds-note">
                    {t("training.projectDatasetNamesMissing", {
                      count: detail.dataset.entries.filter((e) => !e.name).length,
                    })}
                  </div>
                )}
              </>
            )}
          </>
        ) : (
          <div className="tproj-dataset empty">{t("training.projectDatasetEmpty")}</div>
        )}
      </div>

      {/* ── 架构槽 ───────────────────────────────────────────────────────── */}
      <div className="tproj-section">
        <div className="tproj-section-head">
          <span className="tproj-section-title">{t("training.projectSlots")}</span>
        </div>
        <div className="tproj-slots">
          {FAMILIES.map((f) => {
            const slot = slots.get(f);
            // ★§F2⒝ 批 2 ④ —— 这张卡从此是「一个槽 + 它的每个 run」。今天恒一条,所以视觉上
            // 与改动前几乎相同;形状先变复数,是因为回答「这个 run 练到哪了」的解析器在两个
            // run 之后**拒绝作答**,而四个槽是经同一个 `Result` 收上来的。
            const runs = slot?.runs ?? [];
            /** ⛔★★§F2⒝ ④e —— 画哪几行。机理与「为什么真 run 一律画、只有 `id === ""` 的
             *  伪造行按练没练过滤」的全部说明在 `lib/training/slotRows.ts`(S141 §E2E-M3 把它
             *  搬出组件体:写在这里的表达式没有导出,vitest 结构上够不着,变异会存活)。 */
            /** ★★§E2E-M25 ⑵ —— 这个槽里**哪一行**正在训练。`null` = 没有;⛔ 空串是一个
             *  合法答案(未迁移槽的槽根就是那个 run),所以比较必须是 `=== r.id` 而不是真值判断。 */
            const liveRow = liveRunIdFor(liveFacts, projectId, f);
            /** ★★§E2E-M25 ⑷ —— **展示序**:正在跑的置顶,其余按名字。
             *  ⛔ 插在这里(`visibleRuns` 之后、`foldRunRows` 之前)是唯一安全的落点:
             *  `slotStarted` 吃的是**原始 `runs`**(下面那一行),与这条链不相交;而折叠取的是
             *  **头部** `limit` 条,所以置顶必须发生在它**之前**,否则 `slice(0, 2)` 照样能
             *  把正在跑的那条切掉。
             *  ⛔ 而 `pickDiffHost` 吃的是**顶层那个 `SlotDetail`**(`FAMILIES.map` 之外),
             *  **不许**把这里的结果喂给它:那会静默换掉浅扩散训练进哪个 run 目录、用谁的名字、
             *  还原谁的表单 —— 三条都要几小时后才看得出来。判据在 `slotRows.test.ts` 里钉了
             *  「喂进去确实会换宿主」,接线由 `rowIdentityWiring` 的源码闸守。 */
            const allRows = sortRunRows(visibleRuns(runs, liveRow), liveRow);
            // 「尚未开始」与槽级「开始」按钮跟着**看得见的行**走,否则会出现「有一行 run」
            // 同时「尚未开始」的自相矛盾。
            // ⛔ 它跟的是**过滤后**的全部行,不是折叠后剩下的那几行 —— 折叠是纯观感,
            //    不许改变「这个槽开始过没有」这个事实(否则收起来之后卡片会写「尚未开始」)。
            // ⛔★★S144 —— 这两行**必须收到同一个 `liveRow`**。只给上面那半传,新槽第一次训练时
            //    屏幕上会同时出现一行带「训练中」徽章的 run 和一句「未开始」;只给下面那半传,
            //    卡片说开始过了却一行也不画。机理在 `slotRows.ts` 的 `visibleRuns` 头注。
            const started = slotStarted(runs, liveRow);
            const prepPools = prepPoolLine(slot);
            /** ★S141(用户实机提的):run 多了才收。少量 run 时逐条照画,与今天逐像素相同。 */
            const fold = foldRunRows(allRows, !!runsOpen[f]);
            const rows = fold.rows;
            return (
              <div key={f} className={`tproj-slot ${started ? "started" : ""}`}>
                <div className="tproj-slot-head">
                  <span className="tproj-slot-name">{t(FAMILY_TEXT[f].label)}</span>
                  {runs.length > 1 && (
                    <span className="tproj-slot-ver">
                      {t("training.slotRunCount", { count: runs.length })}
                    </span>
                  )}
                </div>
                <div className="tproj-slot-desc">{t(FAMILY_TEXT[f].desc)}</div>
                {!started && (
                  <div className="tproj-slot-facts">
                    <span>{t("training.slotNotStarted")}</span>
                  </div>
                )}
                {/* ── 每个 run 一行 ─────────────────────────────────────────── */}
                {rows.map((r) => {
                  /** ★★§E2E-M25 ⑶ —— 这一行每颗按钮该不该禁、以及**为什么**。
                   *  ⛔ 决策不许写回这里:四颗按钮跟的不是同一个谓词,而把它们塌成一个布尔
                   *  会让判据分不开那两条规则(极性写反不会被任何后端判据抓到)。 */
                  const gates = runRowActions({
                    blocked,
                    inFlight: busy,
                    trainingLive,
                    anyTaskLive,
                    hasRunId: r.id !== "",
                  });
                  return (
                  <div key={r.id} className="tproj-run">
                    <div className="tproj-run-head">
                      <span className="tproj-run-name">
                        {r.modelName
                          ? t("training.runNameFrozen", { name: r.modelName })
                          : t("training.runUnnamed")}
                        {/* ★§F2⒝ 批 2 ④b —— 训练名从此只是标签,所以它可以改。同一个 ✎ 字形
                            与项目改名一致(UI 铁律:方角复古、不用 emoji)。 */}
                        <button
                          className="tproj-run-rename"
                          disabled={gates.rename.disabled}
                          title={gateTitle(gates.rename, t("training.runRename"))}
                          onClick={() => void renameRun(f, r)}
                        >
                          ✎
                        </button>
                      </span>
                      {r.info.version && <span className="tproj-slot-ver">{r.info.version}</span>}
                      {/* ★★§E2E-M25 ⑵ —— 说出正在跑的是**哪一个 run**。此前屏幕上唯一说这件事
                          的地方是页头那盏灯,而它不分项目、也不分 run。 */}
                      {liveRow === r.id && (
                        <span className="tproj-run-live">{t("training.runTraining")}</span>
                      )}
                    </div>
                    <div className="tproj-slot-facts">
                      <span>
                        {r.hasResumePoint
                          ? r.resumeStep != null
                            ? t("training.slotResume", { step: r.resumeStep })
                            : t("training.slotResumeLatest")
                          : t("training.slotNoResume")}
                      </span>
                      <span className="tproj-dot">·</span>
                      {/* 「存档 X GB」→ the 存档中心: audition / import / attach any checkpoint,
                          at any time, without walking the wizard to step 4. The diffusion
                          checkpoints under this sovits slot show up here too (one list). */}
                      <button
                        className="tproj-archive-link"
                        disabled={blocked}
                        onClick={() => openArchive(f)}
                        title={t("training.archiveOpen")}
                      >
                        {t("training.slotArchive", { size: fmtSize(r.ckptBytes) })}
                      </button>
                    </div>
                    {/* ★ The emb_g order THIS RUN froze. This is the authoritative answer to
                        「这个模型的 0 号歌手是谁」, and continuing it demands the exact same order
                        (RESUME_SPEAKER_SET_MISMATCH) — so it belongs on the row that offers 继续训练,
                        never on the slot: two runs may have frozen different orders. */}
                    {r.info.speakers.length > 1 && (
                      <div className="tproj-slot-spk">
                        <span className="tproj-ds-sub" title={t("training.projectDatasetOrderNote")}>
                          {t("training.slotSpeakers")}
                        </span>
                        {r.info.speakers.map((n, i) => (
                          <span key={`${i}:${n}`} className="tproj-slot-spk-item">
                            <span className="training-spk-idx">{i}</span>
                            <span className="tproj-slot-spk-name">{n}</span>
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="tproj-slot-actions">
                      <button
                        className="training-btn small primary"
                        disabled={gates.cont.disabled}
                        title={gateTitle(gates.cont)}
                        onClick={() => void startFamily(f, r)}
                      >
                        {t("training.slotContinue")}
                      </button>
                      <button
                        className="training-btn small"
                        disabled={gates.retrain.disabled}
                        title={gateTitle(gates.retrain)}
                        onClick={() => void retrainFamily(f, r)}
                      >
                        {t("training.slotRetrain")}
                      </button>
                      {/* ★★§F2⒝ ④e —— per-run 删除。⛔ `r.id` 为空 = 未迁移的槽(槽根就是那个
                          run),那时 run 产物与预处理池混在同一个目录里,「只删这个 run」不是一个
                          可分离的操作 ⇒ 不画这颗按钮,诚实的出口是设置里的「删除这个架构」。
                          后端对空 id 一律 `RUN_ID_REQUIRED`,这里只是不让用户白点一次。 */}
                      {r.id !== "" && (
                        <button
                          className="training-btn small danger"
                          disabled={gates.del.disabled}
                          title={gateTitle(gates.del)}
                          onClick={() => void deleteRun(f, r)}
                        >
                          {t("training.runDelete")}
                        </button>
                      )}
                    </div>
                  </div>
                  );
                })}
                {/* ★S141(用户实机提的)—— run 变多之后那一长条的收口。
                    ⛔ 只在真的超过阈值时才出现:用户原话「现在这样直接显示确实很清楚」,
                    所以少量 run 时这一行连出现都不该出现(它自己也占一行)。
                    样式并进 `.training-archive-toggle` 那一组(UI 铁律:先 grep 现有样式再加控件),
                    chevron 用 Unicode 字形而不是 emoji。 */}
                {(fold.hidden > 0 || runsOpen[f]) && (
                  <button
                    className="tproj-runs-toggle"
                    onClick={() => setRunsOpen((m) => ({ ...m, [f]: !m[f] }))}
                  >
                    {runsOpen[f]
                      ? `▾ ${t("training.runsFoldLess")}`
                      : `▸ ${t("training.runsFoldMore", { count: fold.hidden })}`}
                  </button>
                )}
                {/* ★§F2⒝ — the accumulating half of the layout change, made visible where the
                    slot's other sizes already are. A preprocessing parameter change no longer
                    deletes the previous products, so without this line the disk would simply
                    grow with no explanation anywhere in the app. SLOT-level: the pool is shared
                    by every run of it, which is the entire point of layout 2. */}
                {prepPools.show && (
                  <div className="tproj-slot-facts">
                    <span title={t("training.slotPrepPoolsHint")}>
                      {t("training.slotPrepPools", {
                        count: prepPools.count,
                        size: fmtSize(prepPools.bytes),
                      })}
                    </span>
                  </div>
                )}
                {/* 槽级动作 = 「开始」。「继续 / 重训」挂在**每个 run 的那一行**上,因为它们
                    问的是「这一个 run」;放在槽上就必须替用户挑一个,而那正是被判死的规则。
                    ★★§E2E-M25 ⑶ —— 它们与 run 行上那两颗走的是**同一条路**(updateConfig →
                    setRoute → 运行段),所以门禁也必须是同一个:有训练在跑时,预启动卡结构上
                    根本不渲染(它在 `snapshot.state === "idle"` 里面),用户会落在**另一个 run
                    的实时进度**上,没有开始按钮、没有任何解释。 */}
                {!started && (
                  <div className="tproj-slot-actions">
                    {f === "sovits" ? (
                      // The two SoVITS cards of the old target step: 4.1 and 4.0 write into the
                      // same slot, and the choice is only available while it is empty.
                      (["4.1", "4.0"] as const).map((v) => (
                        <button
                          key={v}
                          className="training-btn small"
                          disabled={slotStart.disabled}
                          title={gateTitle(slotStart)}
                          onClick={() => void startFamily("sovits", runs[0], v)}
                        >
                          {t("training.slotStartVersion", { version: v })}
                        </button>
                      ))
                    ) : (
                      <button
                        className="training-btn small primary"
                        disabled={slotStart.disabled}
                        title={gateTitle(slotStart)}
                        onClick={() => void startFamily(f, runs[0])}
                      >
                        {t("training.slotStart")}
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}

          {/* 浅扩散:第 5 张卡,但不是第 5 个槽——它的进度在 sovits 槽的 diffusion/ 里 */}
          <div className={`tproj-slot ${diffSteps > 0 ? "started" : ""}`}>
            <div className="tproj-slot-head">
              <span className="tproj-slot-name">{t("training.backendDiff")}</span>
              {sovitsVersionPinned && (
                <span className="tproj-slot-ver">{sovitsVersionPinned}</span>
              )}
            </div>
            <div className="tproj-slot-desc">{t("training.backendDiffDesc")}</div>
            {/* ⛔ 浅扩散**不是一个 family**,所以它自己问一次:主模型在练时给这张卡贴徽章
                是一句假话,而两者跑在同一个 run 目录里、`liveRunIdFor` 对它们答案相同。 */}
            {diffusionIsLive(liveFacts, projectId) && (
              <div className="tproj-slot-facts">
                <span className="tproj-run-live">{t("training.runTraining")}</span>
              </div>
            )}
            <div className="tproj-slot-facts">
              {diffSteps > 0 ? (
                <span>{t("training.slotResume", { step: diffSteps })}</span>
              ) : (
                <span>{t("training.slotNotStarted")}</span>
              )}
            </div>
            <div className="tproj-slot-actions">
              {sovitsVersionPinned ? (
                <button
                  className="training-btn small primary"
                  disabled={slotStart.disabled}
                  title={gateTitle(slotStart)}
                  onClick={() => void startDiff(sovitsVersionPinned, diffHost)}
                >
                  {diffSteps > 0 ? t("training.slotContinue") : t("training.slotStart")}
                </button>
              ) : (
                // No sovits manifest in this project yet, so nothing pins the ContentVec space
                // — the same choice the empty sovits slot offers, for the same reason.
                (["4.1", "4.0"] as const).map((v) => (
                  <button
                    key={v}
                    className="training-btn small"
                    disabled={slotStart.disabled}
                    title={gateTitle(slotStart)}
                    onClick={() => void startDiff(v, diffHost)}
                  >
                    {t("training.slotStartVersion", { version: v })}
                  </button>
                ))
              )}
            </div>
            <div className="tproj-slot-note">{t("training.diffAttachLater")}</div>
          </div>
        </div>
      </div>

      {/* ── 已导出的模型 ─────────────────────────────────────────────────── */}
      <div className="tproj-section">
        <div className="tproj-section-head">
          <span className="tproj-section-title">{t("training.projectExported")}</span>
        </div>
        {detail.exported.length === 0 ? (
          <div className="tproj-dataset empty">{t("training.projectExportedNone")}</div>
        ) : (
          <div className="tproj-exported">
            {detail.exported.map((e) => (
              <div
                key={`${e.name}:${e.fromCkptRel}`}
                className={`tproj-export-row ${e.installed && !e.sourceDeleted ? "" : "gone"}`}
              >
                <span className="tproj-export-name">{e.name}</span>
                <span className="tproj-export-type">{e.modelType}</span>
                <span className="tproj-export-src" title={e.fromCkptRel}>
                  {e.fromCkptRel}
                </span>
                {/* 「导出过」是历史,不是状态:注册表里已经没有了也照列,只是标灰 */}
                {!e.installed && (
                  <span className="tproj-export-gone">{t("training.exportedDeleted")}</span>
                )}
                {/* ★§F2⒝ ④e —— 同一条口径的第二种「已经不在了」:产它的 run 被删了。
                    模型本身是独立副本,所以它可以**仍然装着**而来源已经没有 —— 两个标记
                    因此是并列的,不是二选一。 */}
                {e.sourceDeleted && (
                  <span className="tproj-export-gone">{t("training.exportedSourceGone")}</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </>,
  );
}
