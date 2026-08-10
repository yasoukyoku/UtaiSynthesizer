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
import { backendErrorMessage } from "../../lib/backendError";
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

  /** Does anything in this project depend on the current data? Deleting or adding then costs a
   *  full re-extraction on the next run — worth one confirmation. With nothing trained yet it is
   *  free, and a dialog per file would just be in the way. */
  const dataHasDependents = (d: ProjectDetailData) =>
    // ★§F2⒝ 批 2 ④ —— 对**每个** run 求或。这是一道「要不要确认」的闸,漏看一个 run 就是
    // fail-open:代价落在那个 run 的几小时预处理上,而对话框根本不会弹。
    d.slots.some((s) => s.ckptCount > 0 || s.runs.some((r) => r.hasResumePoint || r.info.has_main_progress));

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
      if (!maybeShowErrorModal(e, msg)) showToast(msg, "error");
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
        invalid: (v) => (v.trim() ? null : t("backend.TRAINING_NAME_EMPTY")),
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
      if (!maybeShowErrorModal(e, msg)) showToast(msg, "error");
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
      if (!maybeShowErrorModal(e, msg)) showToast(msg, "error");
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
      if (!maybeShowErrorModal(e, msg)) showToast(msg, "error");
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
    const takenNames = new Set(
      (slot?.runs ?? []).map((r) => r.modelName?.trim()).filter((n): n is string => !!n),
    );
    const newName = await showConfirm({
      title: t("training.newRunNameTitle"),
      body: t("training.newRunNameBody"),
      buttons: [
        { id: "ok", label: t("training.next"), kind: "primary" },
        { id: "__cancel", label: t("training.cancel") },
      ],
      input: {
        initial: "",
        invalid: (v) =>
          !v.trim()
            ? t("backend.TRAINING_NAME_EMPTY")
            : takenNames.has(v.trim())
              ? t("training.newRunNameTaken")
              : null,
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
  const sovitsSlot = slots.get("sovits");
  /** ★§F2⒝ 批 2 ④ —— 浅扩散跑在**主模型那个 run** 里(`runs/<主 run>/diffusion/`),所以这张
   *  卡问的是「哪个 run 里有主模型」。今天每槽恒一个 run,所以它就是那一个;两个 run 之后
   *  这条 `find` 是「挂到有主模型的那个」这个**肯定事实**,不是「挑第一个」。 */
  const diffHost =
    sovitsSlot?.runs.find((r) => r.info.has_main_progress) ?? sovitsSlot?.runs[0];
  /** The ContentVec space this project's sovits slot is already committed to, if any. Shallow
   *  diffusion trains on that slot's cached features, so a manifest version PINS it. */
  const sovitsVersionPinned =
    diffHost?.info.version === "4.1" || diffHost?.info.version === "4.0"
      ? diffHost.info.version
      : undefined;
  const diffSteps = diffHost?.info.diff_steps ?? 0;

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
            const startedRun = (r: RunDetail) => r.hasResumePoint || r.info.has_main_progress;
            /** ⛔★★§F2⒝ ④e —— 画哪几行。
             *
             *  此前是 `runs.filter(startedRun)`,而 flip 之后那条过滤会**结构性**地藏起最需要
             *  被删的那一类:`run_manifest.json` 在 spawn **之前**就写好,随后的切片/f0/特征要
             *  跑几小时 —— 中途崩/强停/盘满留下的就是一个**有目录、有 manifest、没有断点**的
             *  run。它两个条件都假 ⇒ 零行零按钮,却
             *  ⑴ 计进 `runs.length` 的徽标(卡片写「2 个 run」而下面只有 1 行);
             *  ⑵ 计进 `list_runs`(那个槽从此每一次探针都要靠 runId);
             *  ⑶ 计进 `SlotDetail.bytes`(盘看得见、删不掉);
             *  ⑷ ★ 它是一个**满强度的冻结源** —— manifest 里有 `speakers` ⇒ 整个项目的数据集
             *     被它锁着,而卡片上写着「尚未开始」。同一屏两句话互相打脸,而且新文案里
             *     「去逐个 run 删掉」的指引对它**执行不下去**。
             *  ⇒ 真正的 run(`id` 非空)一律画出来。只有那条 `id === ""` 的伪造行(后端在**零
             *  run** 时补的,寻址**槽根**)仍然按「练出过东西」过滤 —— 那种槽本来就没有 run 可挑。 */
            const visibleRuns = runs.filter((r) => r.id !== "" || startedRun(r));
            // 「尚未开始」与槽级「开始」按钮跟着**看得见的行**走,否则会出现「有一行 run」
            // 同时「尚未开始」的自相矛盾。
            const started = visibleRuns.length > 0;
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
                {visibleRuns.map((r) => (
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
                          disabled={blocked}
                          title={t("training.runRename")}
                          onClick={() => void renameRun(f, r)}
                        >
                          ✎
                        </button>
                      </span>
                      {r.info.version && <span className="tproj-slot-ver">{r.info.version}</span>}
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
                        disabled={blocked}
                        onClick={() => void startFamily(f, r)}
                      >
                        {t("training.slotContinue")}
                      </button>
                      <button
                        className="training-btn small"
                        disabled={blocked}
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
                          disabled={blocked || busy}
                          onClick={() => void deleteRun(f, r)}
                        >
                          {t("training.runDelete")}
                        </button>
                      )}
                    </div>
                  </div>
                ))}
                {/* ★§F2⒝ — the accumulating half of the layout change, made visible where the
                    slot's other sizes already are. A preprocessing parameter change no longer
                    deletes the previous products, so without this line the disk would simply
                    grow with no explanation anywhere in the app. SLOT-level: the pool is shared
                    by every run of it, which is the entire point of layout 2. */}
                {(slot?.prepPoolCount ?? 0) > 0 && (
                  <div className="tproj-slot-facts">
                    <span title={t("training.slotPrepPoolsHint")}>
                      {t("training.slotPrepPools", {
                        count: slot?.prepPoolCount ?? 0,
                        size: fmtSize(slot?.prepPoolBytes ?? 0),
                      })}
                    </span>
                  </div>
                )}
                {/* 槽级动作 = 「开始」。「继续 / 重训」挂在**每个 run 的那一行**上,因为它们
                    问的是「这一个 run」;放在槽上就必须替用户挑一个,而那正是被判死的规则。 */}
                {!started && (
                  <div className="tproj-slot-actions">
                    {f === "sovits" ? (
                      // The two SoVITS cards of the old target step: 4.1 and 4.0 write into the
                      // same slot, and the choice is only available while it is empty.
                      (["4.1", "4.0"] as const).map((v) => (
                        <button
                          key={v}
                          className="training-btn small"
                          disabled={blocked}
                          onClick={() => void startFamily("sovits", runs[0], v)}
                        >
                          {t("training.slotStartVersion", { version: v })}
                        </button>
                      ))
                    ) : (
                      <button
                        className="training-btn small primary"
                        disabled={blocked}
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
                  disabled={blocked}
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
                    disabled={blocked}
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
