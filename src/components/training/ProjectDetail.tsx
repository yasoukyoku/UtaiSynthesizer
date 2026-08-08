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
 *  3. **「本次训练名」的冻结**。槽里已经跑过的话,产物文件名带的是 `slugify(旧名)`;换个名字
 *     会让 weights/ 里那一堆改名、best_state.json 的携带指标压掉下一次 best 写入。所以已经
 *     跑过的槽只读显示旧名,只有全新的槽才让用户起名(默认=项目名)。
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
  type TrainingBackend,
  type TrainingFormConfig,
  type TrainingSeg,
  type WorkspaceInfo,
} from "../../store/training";
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
    d.slots.some((s) => s.hasResumePoint || s.info.has_main_progress || s.ckptCount > 0);

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

  /** The name this run's ARTIFACTS will carry. Frozen once a slot has produced any, because it
   *  is baked into every file name in `weights/`. */
  const askRunName = async (slot: SlotDetail | undefined): Promise<string | null> => {
    if (slot?.modelName) return slot.modelName;
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

  /** Put the form back where the SLOT already is.
   *
   *  Every field here is resume-guarded Rust-side (`RESUME_PARAMS_MISMATCH` /
   *  `RESUME_VOL_EMBEDDING_MISMATCH`): if the form disagrees with the manifest, 续训 is
   *  impossible and the start dialog can only offer 重训 —— a button labelled「继续训练」whose
   *  one available outcome is wiping the slot. The defaults are NOT harmless here: `sampleRate`
   *  defaults to 48k while an existing RVC slot may be 40k, and `sovitsVersion` defaults to 4.1
   *  while the slot may be 4.0.
   *
   *  `pin` overrides the manifest — used by 重训, which wipes the slot and may therefore
   *  legitimately change the version. */
  const formForSlot = (
    family: Family,
    slot: SlotDetail | undefined,
    pin?: "4.1" | "4.0",
  ): Partial<TrainingFormConfig> => {
    const info = slot?.info;
    if (family === "rvc") {
      const v = info?.version === "v1" || info?.version === "v2" ? info.version : undefined;
      const sr =
        info?.sample_rate === "32k" || info?.sample_rate === "40k" || info?.sample_rate === "48k"
          ? info.sample_rate
          : undefined;
      return { ...(v ? { version: v } : {}), ...(sr ? { sampleRate: sr } : {}) };
    }
    if (family === "sovits") {
      const manifest = info?.version === "4.1" || info?.version === "4.0" ? info.version : undefined;
      const v = pin ?? manifest ?? "4.1";
      return {
        sovitsVersion: v,
        // 响度嵌入 is 4.1-only and is baked into the graph AND the wire inputs.
        ...(v === "4.1" && !pin && info?.vol_embedding != null
          ? { sovitsVolEmbedding: info.vol_embedding }
          : {}),
      };
    }
    // sovits_v2 / vocoder carry fixed manifest markers — nothing to restore.
    return {};
  };

  /** Open this architecture's 存档中心. Sets the family the archive page speaks for (it reads
   *  `config.backend`), then routes — no name prompt, no run, just the product inventory. */
  const openArchive = (family: Family) => {
    updateConfig({ backend: family });
    setRoute({ seg: "archive", projectId });
  };

  const startFamily = async (family: Family, sovitsVersion?: "4.1" | "4.0") => {
    const slot = slots.get(family);
    const runName = await askRunName(slot);
    if (runName === null) return;
    // 继续训练 / 首次开始:the slot's frozen values stand, so the params page locks them.
    useTrainingStore.getState().setRetrainIntent(false);
    updateConfig({
      backend: family,
      modelName: runName,
      ...formForSlot(family, slot, sovitsVersion),
    });
    setRoute({ seg: nextSegFor(family), projectId });
  };

  /** 再训一个:说清楚代价再走。真正的擦除同意仍然只有一处 —— 运行段「开始训练」那个
   *  续训/重训对话框(后端的 wipe_confirmed 就认它)。这里不重复那道门,只负责让用户在点进去
   *  之前就知道会发生什么。 */
  const retrainFamily = async (family: Family) => {
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
    // ★ 再训一个 = 这个架构会被清空,所以续训锁全部解除 —— 换采样率/换版本重练正是这颗按钮
    // 的用途。真正的擦除同意仍然只有一处(运行段那个对话框,后端的 wipe_confirmed 只认它)。
    useTrainingStore.getState().setRetrainIntent(true);
    updateConfig({
      backend: family,
      modelName: slot?.modelName || detail?.name || "",
      ...formForSlot(family, slot, pin),
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
  const startDiff = async (version: "4.1" | "4.0") => {
    const slot = slots.get("sovits");
    const runName = await askRunName(slot);
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
      });
      useTrainingStore.getState().setDiffWsInfo(info);
    } catch {
      /* the data segment is the safe default */
    }
    useTrainingStore.getState().setRetrainIntent(false);
    updateConfig({
      backend: "sovits_diff",
      modelName: runName,
      // ★ DERIVED from the slot whenever it has one: the diffusion has to live in the same
      //   ContentVec space as the cached features it trains on (4.1 = vec768 / 4.0 = vec256).
      //   A mismatch does not error — it just produces something that does not fit its host.
      diffVersion: version,
      ...(info && info.diff_steps > 0 ? { diffKStepMax: Number(info.diff_k_step_max) } : {}),
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
  /** The ContentVec space this project's sovits slot is already committed to, if any. Shallow
   *  diffusion trains on that slot's cached features, so a manifest version PINS it. */
  const sovitsVersionPinned =
    sovitsSlot?.info.version === "4.1" || sovitsSlot?.info.version === "4.0"
      ? sovitsSlot.info.version
      : undefined;
  const diffSteps = sovitsSlot?.info.diff_steps ?? 0;

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
            const started = !!slot && (slot.hasResumePoint || slot.info.has_main_progress);
            return (
              <div key={f} className={`tproj-slot ${started ? "started" : ""}`}>
                <div className="tproj-slot-head">
                  <span className="tproj-slot-name">{t(FAMILY_TEXT[f].label)}</span>
                  {slot?.info.version && (
                    <span className="tproj-slot-ver">{slot.info.version}</span>
                  )}
                </div>
                <div className="tproj-slot-desc">{t(FAMILY_TEXT[f].desc)}</div>
                <div className="tproj-slot-facts">
                  {started ? (
                    <>
                      <span>
                        {slot?.hasResumePoint
                          ? slot.resumeStep != null
                            ? t("training.slotResume", { step: slot.resumeStep })
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
                        {t("training.slotArchive", { size: fmtSize(slot?.ckptBytes ?? 0) })}
                      </button>
                      {/* ★§F2⒝ — the accumulating half of the layout change, made visible where
                          the slot's other sizes already are. A preprocessing parameter change no
                          longer deletes the previous products, so without this line the disk
                          would simply grow with no explanation anywhere in the app. */}
                      {(slot?.prepPoolCount ?? 0) > 0 && (
                        <>
                          <span className="tproj-dot">·</span>
                          <span title={t("training.slotPrepPoolsHint")}>
                            {t("training.slotPrepPools", {
                              count: slot?.prepPoolCount ?? 0,
                              size: fmtSize(slot?.prepPoolBytes ?? 0),
                            })}
                          </span>
                        </>
                      )}
                    </>
                  ) : (
                    <span>{t("training.slotNotStarted")}</span>
                  )}
                </div>
                {slot?.modelName && (
                  <div className="tproj-slot-run">
                    {t("training.runNameFrozen", { name: slot.modelName })}
                  </div>
                )}
                {/* ★ The emb_g order this SLOT froze — per-slot by nature, because each slot
                    trained its own rows. This is the authoritative answer to「这个模型的 0 号
                    歌手是谁」, and it has been available in `info.speakers` since ①c with
                    nothing reading it. Continuing this run demands the exact same order
                    (RESUME_SPEAKER_SET_MISMATCH), so it belongs on the card that offers 继续训练. */}
                {slot && slot.info.speakers.length > 1 && (
                  <div className="tproj-slot-spk">
                    <span
                      className="tproj-ds-sub"
                      title={t("training.projectDatasetOrderNote")}
                    >
                      {t("training.slotSpeakers")}
                    </span>
                    {slot.info.speakers.map((n, i) => (
                      <span key={`${i}:${n}`} className="tproj-slot-spk-item">
                        <span className="training-spk-idx">{i}</span>
                        <span className="tproj-slot-spk-name">{n}</span>
                      </span>
                    ))}
                  </div>
                )}
                <div className="tproj-slot-actions">
                  {f === "sovits" && !started ? (
                    // The two SoVITS cards of the old target step: 4.1 and 4.0 write into the
                    // same slot, and the choice is only available while it is empty.
                    (["4.1", "4.0"] as const).map((v) => (
                      <button
                        key={v}
                        className="training-btn small"
                        disabled={blocked}
                        onClick={() => void startFamily("sovits", v)}
                      >
                        {t("training.slotStartVersion", { version: v })}
                      </button>
                    ))
                  ) : (
                    <button
                      className="training-btn small primary"
                      disabled={blocked}
                      onClick={() => void startFamily(f)}
                    >
                      {started ? t("training.slotContinue") : t("training.slotStart")}
                    </button>
                  )}
                  {started && (
                    <button
                      className="training-btn small"
                      disabled={blocked}
                      onClick={() => void retrainFamily(f)}
                    >
                      {t("training.slotRetrain")}
                    </button>
                  )}
                </div>
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
                  onClick={() => void startDiff(sovitsVersionPinned)}
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
                    onClick={() => void startDiff(v)}
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
                className={`tproj-export-row ${e.installed ? "" : "gone"}`}
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
              </div>
            ))}
          </div>
        )}
      </div>
    </>,
  );
}
