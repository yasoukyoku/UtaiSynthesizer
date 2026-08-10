/**
 * Training store — mirrors the Rust TrainingManager (protocol v2, S37 rewrite).
 *
 * Event-driven (training-stage / training-step / training-ckpt / training-done),
 * NOT polled: install the module-level listeners once via setupTrainingListeners()
 * (msst-models.ts pattern — global, so progress survives the page being closed).
 * The Rust side keeps the authoritative loss history (get_training_history) so a
 * re-mounted page reconstructs the curve; live points append via training-step.
 */
import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import i18n from "../i18n";
import { backendErrorMessage, isBusyError } from "../lib/backendError";
import { hfBaseForMirror } from "../lib/models/msst-catalog";
import { useAppStore, type ConfirmButton } from "./app";
import { useMsstModelStore } from "./msst-models";

/** Mirror of Rust `commands::training::RequiredAssetStatus` (S66 pre-start asset check). */
interface RequiredAssetStatus {
  label: string;
  path: string;
  exists: boolean;
  pack: string | null;
  /** S75: license id when this file's pack carries its own terms (CC BY-NC-SA vocoder base). */
  license: string | null;
  /** Upstream release page (attribution link + offline escape hatch). */
  selfUrl: string | null;
}

/** Mirror of Rust `commands::settings::TrainingGpu` (S75 added id/variant/selectable/reason). */
export interface TrainingGpu {
  /** UI identity and the ONLY thing `config.gpu` / the start payload carries (`vendor:n`).
   *  `value` is unique only WITHIN a vendor, so it cannot serve as a key here. */
  id: string;
  label: string;
  /** Accelerator-native device mask. Resolved server-side from `id`; never sent by the UI. */
  value: string;
  /** Runtime variant that drives it ("nv-cu130"/"amd"/"xpu"); null = no training runtime. */
  variant: string | null;
  selectable: boolean;
  /** Stable CODE (backendError.ts) explaining an unselectable entry. */
  reason: string | null;
}

export interface DatasetFile {
  path: string;
  name: string;
  durationMs: number | null;
}

/** ①c multi-speaker co-training: one co-trained singer = a display name + its
 *  own files. Drives the SoVITS card's singer list (1 singer = single-speaker,
 *  the degenerate case). `id` is a stable React key (names may be blank /
 *  duplicate mid-edit). */
export interface SpeakerGroupDraft {
  id: string;
  name: string;
  files: DatasetFile[];
}

/** Generation counter for the checkpoint scan — see refreshProjectCkpts. */
let ckptScanSeq = 0;
/** Bumped by `enterProject`. Anything that stages data ASYNCHRONOUSLY has to check it after
 *  its await: probing a few hundred files takes long enough for the user to have moved to
 *  another project, and the result would land in THAT project's form. */
let projectEpoch = 0;
/** Nonce for the「文件落到这位歌手」pulse — the same singer must be able to re-flash. */
let flashSeq = 0;

export interface StageInfo {
  stage: string;
  done?: number | null;
  total?: number | null;
  progress?: number | null;
  message?: string | null;
}

export interface StepInfo {
  step: number;
  total_steps: number;
  epoch: number;
  total_epochs: number;
  lr: number;
  losses: Record<string, number>;
  eta_secs?: number | null;
}

export interface CkptInfo {
  kind: "periodic" | "best" | "final" | "stop";
  path: string;
  step: number;
  epoch: number;
  metric?: number | null;
}

/** Mirror of Rust `tproject::CkptRecord` (list_project_ckpts) — what is actually ON DISK.
 *  Distinct from CkptInfo, which is what the RUNNING sidecar emitted into memory: the event
 *  source vanishes when the app closes or「清空结果」is pressed, the scan does not. */
export interface CkptRecord {
  /** Path relative to the project dir — the ledger key (survives a data-dir move). */
  rel: string;
  path: string;
  family: string;
  /** ★§F2⒝ 批 2 ④ — 这份存档出自哪个 run(`trun::run_id_in_rel`)。
   *
   *  `""` = **槽根就是那个 run**(layout ≤2)—— 肯定事实,不是缺席。
   *  `null` = 只有前端造的 `pending` 行才会有:它的 `rel` 是事件里的绝对路径**砍成最后两段**
   *  拼的,`runs/<id>/` 那一层压根不在里面。填 `""` 会把它说成「槽根那个 run」,而那正是
   *  layout 3 下**不存在**的东西 ⇒ 分组时会凭空多出一个空 run。这一行只活到下一次磁盘扫描,
   *  而扫描出来的行带着真的 id,所以「不知道」才是它诚实的值。
   *
   *  存档中心一张表列整个 family:没有这个字段,两个 run 的存档只按 mtime 交织,行上没有
   *  任何东西说它属于哪个模型,而「导入」会给两边提议同一个名字。 */
  runId: string | null;
  /** base = the seeded pretrained (not the user's work); release/best = generator-only
   *  snapshots you import; resumable = training can continue from it. */
  /** Mirrors Rust `CkptKind` (six variants) plus the frontend-only `pending`. `final` was
   *  missing from this union AND from `training.ckptKind.*`, so the naturally-finished
   *  `weights/<slug>.pth` — the one artifact a user is most likely to want — rendered its raw
   *  i18n key in the archive list. */
  kind: "base" | "resumable" | "release" | "best" | "final" | "orphan" | "pending";
  /** Real training step. null = RVC's「只保留最新」sentinel name, which is not a step. */
  step: number | null;
  bytes: number;
  mtimeMs: number;
  imported: boolean;
  /** Files that belong to the same archive (a GAN pair's D); `bytes` already includes them. */
  companions: string[];
}

/** Same file, seen from the two sources. Paths come from Rust on one side and from the
 *  sidecar on the other, so compare them case- and separator-insensitively (Windows). */
function ckptKey(p: string): string {
  return p.replace(/\\/g, "/").toLowerCase();
}

/** THE single reconciliation of「运行中刚落的存档」with「磁盘上真实存在的存档」.
 *
 *  The disk scan wins wherever both know a file (it carries size / kind / imported), and a
 *  ckpt the sidecar just announced but the scan has not seen yet is still shown — otherwise a
 *  checkpoint would blink out of the list for the seconds between the event and the next
 *  scan. Pure + tested so the two sources can never drift into duplicate or flickering rows. */
export function mergeCkptSources(
  eventCkpts: CkptInfo[],
  scanned: CkptRecord[],
  family: string,
): CkptRecord[] {
  const seen = new Set(scanned.map((r) => ckptKey(r.path)));
  const extra: CkptRecord[] = eventCkpts
    .filter((c) => !seen.has(ckptKey(c.path)))
    .map((c) => ({
      rel: c.path.replace(/\\/g, "/").split("/").slice(-2).join("/"),
      path: c.path,
      family,
      // ★§F2⒝ 批 2 ④ —— **不猜**(见 `CkptRecord.runId` 的注释:`""` 是「槽根就是 run」这个
      // 肯定事实,不是「不知道」)。
      runId: null,
      // Do NOT infer a kind here. The event only says periodic/best/final/stop, and the same
      // file can be a release snapshot (rvc/sovits weights) or a resume point (diffusion
      // model_<step>.pt) depending on family — guessing made a diffusion ckpt read「快照」for
      // the few seconds before the scan caught up, then flip to「可续训」.
      kind: "pending" as const,
      step: c.step,
      bytes: 0,
      mtimeMs: Number.MAX_SAFE_INTEGER, // just written ⇒ newest
      imported: false,
      companions: [],
    }));
  return [...extra, ...scanned].sort((a, b) => b.mtimeMs - a.mtimeMs || a.rel.localeCompare(b.rel));
}

export interface StepPoint {
  step: number;
  lr: number;
  losses: Record<string, number>;
}

/** Mirror of Rust `tproject::ProjectSummary` (list_training_projects). `ProjectSizes` is
 *  FLATTENED into it server-side, so the size fields sit at the top level (pinned by a Rust
 *  test — `invoke` is stringly typed and nothing here could catch a rename). */
export interface ProjectSummary {
  id: string;
  name: string;
  note: string;
  createdMs: number;
  updatedMs: number;
  /** Migration could not classify the directory: contents preserved, training refused. */
  needsAttention: string | null;
  /** Architecture slots present on disk. */
  families: string[];
  hasDataset: boolean;
  totalBytes: number;
  datasetBytes: number;
  familyBytes: Record<string, number>;
  /** 0 = never measured — show「—」, not a confident「0 B」. */
  computedMs: number;
  /** Remembered by the cache, absent from disk. Listed, greyed, and barred from training. */
  missing: boolean;
}

/** Mirror of Rust `commands::training::RunDetail` —— **一个 run**,不是一个槽。
 *
 *  ★§F2⒝ 批 2 ④:这里每个字段都是 run 事实。以前它们挂在槽上,而回答它们的解析器在
 *  「一个槽有两个 run」时**拒绝作答** —— 于是四个槽经同一个 `Result` 收集,一个歧义槽会让
 *  整个项目详情页打不开。所以形状先变复数,再谈铸第二个 run。 */
export interface RunDetail {
  /** `trun` run id。`""` = 未迁移槽的槽根就是那个 run(layout ≤2);命令收 `runId?: string`,
   *  不传 = 同一个意思。 */
  id: string;
  /** 这个 **run** 的「本次训练名」(读它自己的 `run.json`)。"" = 它还没跑完过。 */
  modelName: string;
  info: WorkspaceInfo;
  /** null WITH hasResumePoint = RVC's「只保留最新」sentinel, whose name carries no step. */
  resumeStep: number | null;
  hasResumePoint: boolean;
  ckptCount: number;
  ckptBytes: number;
}

/** Mirror of Rust `commands::training::SlotDetail`. */
export interface SlotDetail {
  family: string;
  /** 这个槽的每个 run,按 id 排序。今天恒为 1 条。 */
  runs: RunDetail[];
  bytes: number;
  /** 槽**总计**(逐 run 求和)—— 「这个架构占多大」是槽问题,不是 run 问题。 */
  ckptCount: number;
  ckptBytes: number;
  /** ★§F2⒝ — PREPROCESSING pools in this slot (a different thing from this store's `poolCount`,
   *  which counts the project's imported dataset files). A preprocessing identity change keeps
   *  the old products as a sibling instead of deleting them, so this is where the disk goes. */
  prepPoolCount: number;
  prepPoolBytes: number;
}

/** Mirror of Rust `commands::training::ExportedModelStatus`. */
export interface ExportedModelStatus {
  name: string;
  modelType: string;
  fromCkptRel: string;
  atMs: number;
  /** ★§F2⒝ ④e —— 产它的那个 run(或整个架构)已被用户主动删除。行照列(「导出过」是历史),
   *  变的只是这一行还能**被用来做什么**:它不再保护任何快照不被清理,也不再算作
   *  「账本对得上盘」的证据。 */
  sourceDeleted: boolean;
  /** Live registry check — false = deleted in the resource manager since. */
  installed: boolean;
}

/** Mirror of Rust `tproject::DeleteReport` — what a destructive action actually did.
 *
 *  ⚠ Shared by Settings(删架构 / 删项目 / 清理快照)and the project detail page(§F2⒝ ④e 的
 *  per-run 删除)。两份手写镜像会各自漂,而它描述的是**破坏性操作真的做了什么**。 */
export interface DeleteReport {
  freedBytes: number;
  deleted: string[];
  kept: { rel: string; reason: string }[];
  /** rename done, background removal blocked — the archives are already unreachable. */
  deferred: boolean;
}

/** Mirror of Rust `commands::training::DatasetFileRow`. */
export interface DatasetFileRow {
  /** Path under `dataset/` — `000.wav` or `<slug>/000.wav`. */
  rel: string;
  /** Name at import time. "" = unrecorded (imported before batch 5) ⇒ show `rel`, never guess. */
  name: string;
  bytes: number;
  durationMs: number | null;
}

/** Mirror of Rust `commands::training::DatasetGroupRow`. */
export interface DatasetGroupRow {
  slug: string;
  /** "" = unrecoverable (`slugify` is one-way) ⇒ show the slug. */
  name: string;
  files: number;
  bytes: number;
}

/** Mirror of Rust `commands::training::DatasetSummary`. */
export interface DatasetSummary {
  files: number;
  bytes: number;
  /** Absolute path of `<project>/dataset` — join with a row's `rel` to read the audio. */
  datasetDir: string;
  /** Sorted slugs. `poolFlat` keys on its emptiness — the ordered, named view is `groups`. */
  speakers: string[];
  entries: DatasetFileRow[];
  groups: DatasetGroupRow[];
  /** False ⇒ the emb_g row numbers are NOT knowable; the UI must not print any. */
  orderKnown: boolean;
}

/** Mirror of Rust `commands::training::ProjectDetail`. */
export interface ProjectDetail {
  id: string;
  name: string;
  note: string;
  createdMs: number;
  updatedMs: number;
  needsAttention: string | null;
  dataset: DatasetSummary;
  slots: SlotDetail[];
  exported: ExportedModelStatus[];
}

/** Mirror of Rust `training::WorkspaceInfo` (get_training_slot_info). */
export interface WorkspaceInfo {
  exists: boolean;
  /** manifest family ("rvc"/"sovits"); "" when absent */
  family: string;
  /** manifest version ("v1"/"v2"/"4.1"/"4.0"); "" when absent */
  version: string;
  /** manifest sample rate ("32k"/"40k"/"48k"/"44k"); "" when absent */
  sample_rate: string;
  has_main_progress: boolean;
  /** max diffusion checkpoint step; 0 = none/base only */
  diff_steps: number;
  /**
   * ★S117 §F2⒜: step of the resumable BEST snapshot (`resume_best/`), or null when this slot
   * has none / it is half-written. Gates the「从最佳存档继续」button — offering it without a
   * complete snapshot would be a button that silently continues from the latest instead.
   */
  best_resume_step: number | null;
  /**
   * ★S118 §F8⒜: step of the SHALLOW-DIFFUSION best snapshot (`diffusion/resume_best/`), or null.
   * ⛔ A separate field, and the diffusion dialog must use THIS one: a `sovits_diff` probe
   * resolves to the sovits slot, so `best_resume_step` above is the MAIN GAN model's snapshot.
   * Labelling a diffusion button with it would print the wrong model's step.
   */
  diff_best_resume_step: number | null;
  /** manifest 数据增强份数 (S41) — what a diff run will inherit */
  aug_copies: number;
  /** ★§F2⒝ 批 2 ④d — 这个槽的**预处理**是不是带响度归一化建的。`null` = 盘上没有东西回答
   *  (从没跑过 / manifest 老到没这个键且池不唯一)。三态是**有意**的:`false` 要把复选框放回
   *  「关」,`null` 必须**不动**用户当前的值 —— 这个字段就是为了让这两件事分得开才存在的。 */
  loudnorm: boolean | null;
  /** a reusable shared slice pool exists — diff may start without importing */
  has_dataset: boolean;
  /** ①c resume config-diff: manifest vol_embedding (SoVITS); null when absent/not-sovits */
  vol_embedding: boolean | null;
  /** ①c: manifest n_speakers (multi-speaker); 1 when single-speaker */
  n_speakers: number;
  /** ①c: ordered speaker display names (index = emb_g id); empty for single-speaker */
  speakers: string[];
  /** ①c: manifest diff_k_step_max (sovits_diff); 0 when absent */
  diff_k_step_max: number;
}

/** S41 共享池模式 — THE single predicate for "a diff run may start without
 *  importing data" (root tab gating, DataStep next button, RunStep start
 *  guard all share it; Rust start_training re-verifies authoritatively). */
export function diffPoolReady(backend: string, info: WorkspaceInfo | null): boolean {
  return (
    backend === "sovits_diff" &&
    !!info?.exists &&
    info.family === "sovits" &&
    info.has_dataset &&
    // S76: shallow diffusion refuses multi-speaker workspaces (Rust side), and since the
    // dataset became a PROJECT-level shared layer a multi-singer project's dataset is stored
    // per singer — which this run, carrying no singer groups, cannot consume. Advertising
    // 免导入直训 there would let the user skip the data page straight into a refusal.
    info.n_speakers <= 1
  );
}

/** THE predicate for「这个项目盘上已经有可复用的数据集,本次运行不必再导入」.
 *
 *  For shallow diffusion it stays the sovits-host-specific `diffPoolReady`. For every other
 *  backend it is `poolFlat` — the project has an on-disk dataset AND that dataset is flat
 *  (single-speaker). The asymmetry mirrors the backend exactly: `try_start` reuses an empty
 *  `dataset_files` for a flat pool, but refuses a multi-speaker one for a run carrying no
 *  speaker groups (`PROJECT_DATASET_SHAPE`) — reconstructing the singer groups from disk so a
 *  multi-speaker project can resume without re-importing is batch 5's job. */
export function poolReusable(
  backend: string,
  poolFlat: boolean,
  diffInfo: WorkspaceInfo | null,
): boolean {
  return backend === "sovits_diff" ? diffPoolReady(backend, diffInfo) : poolFlat;
}

/** Where the training page is, EXPLICITLY (S76 batch 4).
 *
 *  It used to be a bare `wizard: 1|2|3|4`, which could not express「哪个项目」at all — the
 *  project was inferred from the editable model name at four independent call sites — and could
 *  not express「训练中,直接进运行段」either: that worked only because the wizard happened to be
 *  parked on 4, so a page refresh dropped the user back to step 1 mid-run.
 *
 *  `projectId` is "" only on the landing. Every other segment is ABOUT a project. */
export type TrainingSeg = "projects" | "detail" | "data" | "params" | "run" | "archive";

export interface TrainingRoute {
  seg: TrainingSeg;
  projectId: string;
}

/** Which of the four step tabs a segment belongs to. The landing and a project's detail are
 *  both「1 · 项目」— picking a project is a move WITHIN that step, not a step of its own. */
export function segTab(seg: TrainingSeg): 1 | 2 | 3 | 4 {
  switch (seg) {
    case "projects":
    case "detail":
      return 1;
    case "data":
      return 2;
    case "params":
      return 3;
    case "run":
    case "archive":
      // the archive is reached from a project's model card, not from the step rail; it rides
      // under 4 (运行/产物) so the tab highlight has somewhere sensible to sit.
      return 4;
  }
}

/** The five things this app can train. THE single source for the union — the form config, the
 *  slot cards and the「把运行中的 snapshot 回填进 config」路径 all speak it. */
export type TrainingBackend = "rvc" | "sovits" | "sovits_v2" | "sovits_diff" | "vocoder";

const TRAINING_BACKENDS: readonly TrainingBackend[] = [
  "rvc",
  "sovits",
  "sovits_v2",
  "sovits_diff",
  "vocoder",
];

/** `TrainingSnapshot.backend` is a plain string from Rust. Widening it into the form config
 *  without checking would let an unknown value drive every backend-keyed branch on the page. */
export function asTrainingBackend(s: string): TrainingBackend | null {
  return (TRAINING_BACKENDS as readonly string[]).includes(s) ? (s as TrainingBackend) : null;
}

/** Which architecture SLOT a backend trains into — mirrors Rust `training::backend_family`.
 *  Shallow diffusion is not a family of its own: it lives in the sovits slot because it shares
 *  the main model's preprocessing caches, which is the entire reason it exists. THE single
 *  frontend source for this mapping; do not re-derive it per call site. */
export function backendFamily(backend: string): string {
  return backend === "sovits_diff" ? "sovits" : backend;
}

/** ①c: which backends take a SINGER LIST (multi-speaker co-train) — SoVITS (α) + RVC (α′)
 *  + SoVITS 4.0-v2 (S68, natively multi-speaker upstream).
 *  THE single source for the DataStep singer-list gating so the store + page never drift.
 *  Shallow-diffusion / vocoder stay flat-dataset (their loaders assume one speaker). */
export function backendSupportsMultiSpeaker(backend: string): boolean {
  return backend === "sovits" || backend === "rvc" || backend === "sovits_v2";
}

/** THE single predicate for "step 2 (data) is satisfied" — shared by the root
 *  wizard gating (step3Ok) AND the DataStep next button so they never drift.
 *  ①c: SoVITS/RVC data is a SINGER LIST (default 1 singer = single-speaker, the
 *  degenerate case of N). Every singer needs files; with ≥2 singers each also
 *  needs a (unique) name. Other backends keep the flat-dataset / shared-pool rule. */
/** THE single predicate for「数据这一步满足了吗」 — the wizard gating (`step3Ok`), the data
 *  page's Next button, and「点某个架构该跳到哪一段」all ask this one, so they cannot disagree.
 *
 *  S78: it asks the DISK, not a staging form. Importing became its own act (files land in
 *  `<project>/dataset/` immediately), so「有没有数据」is a property of the project — which is
 *  also what makes an existing project trainable without re-importing anything.
 */
export function trainingDataOk(
  backend: string,
  ds: DatasetSummary | null,
  /** Shallow diffusion asks a different question: it needs the SoVITS slot's cached slices, not
   *  merely raw audio (`diffPoolReady`). */
  diffInfo: WorkspaceInfo | null,
): boolean {
  if (backend === "sovits_diff") return diffPoolReady(backend, diffInfo);
  if (!ds || ds.files === 0) return false;
  // A flat dataset feeds any backend. A per-singer one only feeds the co-training backends —
  // python's fingerprint hard-fails on a subdirectory for the flat ones, so offering to start
  // would just move the refusal later and further from the cause.
  return ds.speakers.length === 0 || backendSupportsMultiSpeaker(backend);
}


export interface TrainingSnapshot {
  state: "idle" | "starting" | "running" | "completed" | "stopped" | "error";
  error?: string | null;
  backend: string;
  model_name: string;
  model_slug: string;
  /** S76: the training PROJECT this run belongs to ("" while idle). */
  project_id: string;
  /** The run's family SLOT dir (`<data>/training/<project>/<family>`) — the pre-S76
   *  workspace root, so audition/weights paths keep their shape. */
  workspace: string;
  total_epochs: number;
  stage?: StageInfo | null;
  step?: StepInfo | null;
  ckpts: CkptInfo[];
  summary?: Record<string, unknown> | null;
  stop_requested: boolean;
  elapsed_secs: number;
  stderr_tail: string[];
  /** ①c: ordered speaker display names for a multi-speaker run (index = emb_g id); empty for
   *  single-speaker. Reflects the RUN (frozen at start), used by the audition speaker picker. */
  speakers?: string[];
  /** S114 §F5-1: stable CODEs for problems raised while the run is STILL RUNNING —
   *  localized through the same `backendErrorMessage` map as failures. Optional because
   *  the backend omits the field entirely for a healthy run (wire stays pre-S114 identical).
   *  A warning never changes `state`: the run may still recover, so this informs only. */
  warnings?: string[];
}

export interface TrainingFormConfig {
  modelName: string;
  /** ★§F2⒝ 批 2 ④ —— 这次开始训练**针对哪个 run**。`""` = 「这个槽最多只有一个 run」,
   *  正是 `trun::resolve_run_dir` 断言并**拒绝猜**的那个肯定事实。
   *
   *  ⚠ 它是**产物选择器**,不是锁表的一行(和 `resumeFrom` 同一个定位):它决定读写哪一份
   *  已有产物,不改变这个槽被允许持有什么。 */
  runId: string;
  /** sovits_v2 = SoVITS 4.0-v2 (VISinger2, S68) — its own backend/workspace
   *  family; it SHARES the sovits* form fields below (same 44.1k step-cadenced
   *  shape; v2-less switches — volEmbedding/fp16/allInMem — are hidden). */
  backend: TrainingBackend;
  version: "v1" | "v2";
  sampleRate: "32k" | "40k" | "48k";
  totalEpoch: number;
  batchSize: number;
  saveEveryEpoch: number;
  saveEveryWeights: boolean;
  keepOnlyLatest: boolean;
  cacheGpu: boolean;
  fp16: boolean;
  /** `TrainingGpu.id` of the picked device ("" = auto). S75: an IDENTITY, not the device mask —
   *  Rust resolves it to the mask server-side. It used to be the mask itself, which is unique
   *  only within a vendor, so on a multi-vendor box it named two different cards. S67 (the same
   *  bug one generation earlier): a raw WMI list index that silently CPU'd multi-adapter boxes. */
  gpu: string;
  forceCpu: boolean;
  /** S41 PSOLA 数据增强份数 (0-3, 0=off) — rvc card (per-card fields so
   *  switching cards never clobbers; diff has NO field: it inherits the
   *  workspace manifest's like loudnorm/vol_embedding) */
  augCopies: number;
  // ---- SoVITS (44.1kHz fixed; separate fields so switching cards never
  // clobbers the RVC values with SoVITS-scaled ones) ----
  sovitsVersion: "4.1" | "4.0";
  sovitsTotalEpoch: number;
  sovitsBatchSize: number;
  /** ckpt/eval cadence in global steps (upstream eval_interval) */
  sovitsSaveEverySteps: number;
  /** G_/D_ checkpoints kept on disk (upstream keep_ckpts) */
  sovitsKeepCkpts: number;
  sovitsFp16: boolean;
  /** 响度嵌入 — 4.1 only (couples vol_embedding + vol_aug like upstream --vol_aug) */
  sovitsVolEmbedding: boolean;
  /** resample 响度归一 — upstream default ON, ours OFF (lossy per upstream README) */
  sovitsLoudnorm: boolean;
  /** kmeans cluster centers instead of the retrieval matrix */
  sovitsKmeans: boolean;
  sovitsAllInMem: boolean;
  /** S41 PSOLA 数据增强份数 (0-3, 0=off) — sovits card; the value a later
   *  diff run inherits via the workspace manifest */
  sovitsAugCopies: number;
  // ---- 浅扩散 sovits_diff (separate fields — card switches must not clobber;
  // no loudnorm/vol_embedding here: a diff run INHERITS them from the
  // workspace manifest, flipping them would wipe the shared caches) ----
  diffVersion: "4.1" | "4.0";
  /** completion target in global steps (diffusion progress is step-based) */
  diffTotalSteps: number;
  diffBatchSize: number;
  /** save + validation cadence in steps (upstream interval_val) */
  diffSaveEverySteps: number;
  /** milestone keep cadence; Rust normalizes to a multiple of diffSaveEverySteps */
  diffForceSaveSteps: number;
  /** 0 = full diffusion (train all 1000 t) — most capable; 100/200/300 = shallow-only */
  diffKStepMax: number;
  /** amp fp16 (upstream amp_dtype; default fp32) */
  diffFp16: boolean;
  /** S78: augmentation copies for a DIFF-FIRST slot (no main model sharing the slice pool).
   *  With a main model present the run inherits the manifest's value and this is ignored —
   *  Rust decides, the form just carries a number. */
  diffAugCopies: number;
  diffCacheAllData: boolean;
  // ---- 声码器微调 vocoder (S40; separate fields — card switches must not
  // clobber). Steps are REAL optimizer rounds (the lightning GAN counts D+G
  // separately internally — sidecar handles the 2× mapping). ----
  /** completion target in REAL steps (official guidance: ~2000 finishes a fine-tune) */
  vocTotalSteps: number;
  /** save + validation cadence in REAL steps */
  vocSaveEverySteps: number;
  vocBatchSize: number;
  /** workspace lightning checkpoints kept (weights/ snapshots are never pruned) */
  vocKeepCkpts: number;
  /** dataset crop window in mel frames (32 = upstream 16G preset, 48 = 24G) */
  vocCropMelFrames: number;
  /** freeze the MPD discriminator (upstream README: may help small-step fine-tunes) */
  vocFreezeMpd: boolean;
  /** S41 PSOLA 数据增强份数 (0-3, 0=off) — vocoder card */
  vocAugCopies: number;
}

/** Exported so the run segment can substitute it when the one live snapshot belongs to a
 *  DIFFERENT project than the page is currently pointed at (S76 batch 4). */
export const IDLE_SNAPSHOT: TrainingSnapshot = {
  state: "idle",
  backend: "",
  model_name: "",
  model_slug: "",
  project_id: "",
  workspace: "",
  total_epochs: 0,
  ckpts: [],
  stop_requested: false,
  elapsed_secs: 0,
  stderr_tail: [],
};

const DEFAULT_CONFIG: TrainingFormConfig = {
  modelName: "",
  runId: "",
  backend: "rvc",
  version: "v2",
  sampleRate: "48k",
  totalEpoch: 200,
  batchSize: 6,
  saveEveryEpoch: 25,
  saveEveryWeights: true,
  keepOnlyLatest: true,
  cacheGpu: false,
  fp16: true,
  gpu: "",
  forceCpu: false,
  augCopies: 0,
  sovitsVersion: "4.1",
  sovitsTotalEpoch: 1000,
  sovitsBatchSize: 6,
  sovitsSaveEverySteps: 800,
  sovitsKeepCkpts: 3,
  sovitsFp16: false,
  sovitsVolEmbedding: true,
  sovitsLoudnorm: false,
  sovitsKmeans: false,
  sovitsAllInMem: false,
  sovitsAugCopies: 0,
  diffVersion: "4.1",
  diffTotalSteps: 100000,
  diffBatchSize: 48,
  diffSaveEverySteps: 2000,
  diffForceSaveSteps: 10000,
  diffKStepMax: 0,
  diffFp16: false,
  diffAugCopies: 0,
  diffCacheAllData: true,
  vocTotalSteps: 2000,
  vocSaveEverySteps: 500,
  vocBatchSize: 8,
  vocKeepCkpts: 5,
  vocCropMelFrames: 32,
  vocFreezeMpd: false,
  vocAugCopies: 0,
};

/** Client-side mirror of the Rust history cap: thin to half when exceeded. */
const HISTORY_CAP = 40000;

interface TrainingStoreState {
  snapshot: TrainingSnapshot;
  /** Wall-clock ms when `snapshot` was received — RunStep's elapsed ticker extrapolates from it. */
  snapshotAt: number;
  history: StepPoint[];
  config: TrainingFormConfig;
  route: TrainingRoute;
  starting: boolean;
  /** workspace info for the CURRENT diff host pick (null when backend≠diff or
   *  no pick) — fetched by the TrainingPage root effect, consumed everywhere
   *  via diffPoolReady() */
  diffWsInfo: WorkspaceInfo | null;
  /** The CURRENT project holds a flat (single-speaker) on-disk dataset a run may reuse without
   *  importing (S76 batch 4). Derived from `get_training_project` — by the TrainingPage root
   *  effect on route change AND by ProjectDetail after it edits the dataset — cleared on
   *  `enterProject`; consumed everywhere via `poolReusable`. Kept in the store (not local to
   *  ProjectDetail) because `step3Ok` in TrainingPage needs it after ProjectDetail unmounts. */
  poolFlat: boolean;
  /** How many files that on-disk dataset holds (0 = none). Same source and lifetime as
   *  `poolFlat`; the data page needs the COUNT to say what picking files would replace. */
  poolCount: number;
  /** The project's on-disk dataset as last read — the DATA the two derived flags above summarise.
   *  Held so the data page can list it without a fetch of its own (two fetches of the same thing
   *  is how two screens end up disagreeing about whether a project has data). */
  projectDataset: DatasetSummary | null;

  setRoute: (r: TrainingRoute) => void;
  /** Does any architecture slot of the CURRENT project hold work? Deleting or adding data then
   *  costs a full re-extraction on the next run — the one thing worth a confirmation. */
  projectHasProgress: boolean;

  /** ONE writer for every field derived from the project read — a setter each is exactly how
   *  they drift apart. `null` = unknown (no project / the read failed). */
  setProjectInfo: (d: ProjectDetail | null) => void;
  /** Re-read the CURRENT project's dataset from disk. Every mutation ends with this; the disk is
   *  the authority, so the UI never patches its own copy. */
  refreshProjectDataset: () => Promise<void>;
  /** Copy audio INTO the project's dataset (append). `speaker` = a co-trained singer's display
   *  name, null = the flat dataset. Throws the backend CODE for the caller to display. */
  importIntoProject: (files: string[], speaker?: string | null) => Promise<void>;
  /** Remove files from the project's dataset by their `rel`. Throws the backend CODE. */
  deleteFromProject: (rels: string[]) => Promise<void>;
  /** Move within the CURRENT project. Refuses when there is none — every segment past the
   *  landing is about a project, and a `projectId: ""` route would silently address
   *  `<training>/` itself. */
  goSeg: (seg: TrainingSeg) => void;
  /** THE way to enter (or leave) a project.
   *
   *  Switching projects must also drop every piece of per-RUN state, because all of it
   *  describes the project you are leaving: the staged dataset, the「本次训练名」, the
   *  shared-pool probe, the archive list. A plain `setRoute` leaves them behind, and they are
   *  not cosmetic — `start()` sends `route.projectId` together with `config`/`dataset`, so a
   *  stale form would train project B out of project A's audio and REPLACE B's dataset with it. */
  enterProject: (projectId: string, seg?: TrainingSeg) => void;
  setDiffWsInfo: (info: WorkspaceInfo | null) => void;
  /** The CURRENT backend's slot facts (`get_training_slot_info`). Drives the parameters page's
   *  read-only rendering of the resume-locked fields; null = no project / never ran / read failed. */
  slotInfo: WorkspaceInfo | null;
  setSlotInfo: (info: WorkspaceInfo | null) => void;
  /** The user arrived here via「再训一个」, i.e. they already chose to WIPE this architecture.
   *  A 重训 unlocks every resume-locked field (that is the only way to change one), but the
   *  choice is made on the project page and the parameters page is downstream of it — without
   *  carrying the intent, 再训一个 would land on a form whose version / sample rate / diffusion
   *  depth are all greyed out, which is precisely what that button exists to let you change.
   *  Cleared by `enterProject` and by every non-retrain entry into a slot. */
  retrainIntent: boolean;
  setRetrainIntent: (v: boolean) => void;
  updateConfig: (u: Partial<TrainingFormConfig>) => void;
  /** ①c: which singer card a file-drag is currently over (highlight target);
   *  null = none. Set by the drag handler, read by the DataStep cards. */
  dragOverSpeakerId: string | null;
  setDragOverSpeakerId: (id: string | null) => void;
  /** ①c: transient "files were just added to this singer" pulse — id + nonce so
   *  the SAME singer re-flashes on repeat adds (the nonce forces the animated
   *  node to remount). Only set with ≥2 singers; cleared on animation end. */
  flashSpeaker: { id: string; nonce: number } | null;
  /** clear the pulse, but only if `nonce` still matches — so a stale animationend
   *  from one singer cannot wipe a pulse just started on another. */
  clearFlashSpeaker: (nonce: number) => void;
  refresh: () => Promise<void>;
  /** S76: the project's on-disk checkpoint inventory (survives app restarts and「清空结果」). */
  projectCkpts: CkptRecord[];
  /** Rescan one project's on-disk archives. THE only way to refresh the inventory, so no call
   *  site can forget the family filter or race a stale response in.
   *
   *  S76 batch 4: takes the project ID. It used to resolve the project from the model NAME,
   *  because before the project pages existed the name was the only identity there was — and
   *  that name is now user-editable, so it would go stale the moment somebody renamed a run. */
  refreshProjectCkpts: (projectId: string, backend: string) => Promise<void>;
  /** `wipeConfirmed` = the user answered a destructive「重训」dialog for THIS run. The backend
   *  refuses a `fresh` start that would destroy checkpoints / an imported dataset without it. */
  /**
   * ★S117 §F2⒜ `resumeFrom`: "latest" (default = every previous release) or "best" — WHICH
   * archive a 续训 continues from. Ignored on a fresh run. The trainer falls back to the latest
   * LOUDLY if "best" is asked for and no complete snapshot exists.
   */
  start: (fresh: boolean, wipeConfirmed: boolean, resumeFrom?: "latest" | "best") => Promise<void>;
  stop: () => Promise<void>;
  forceStop: () => Promise<void>;
  /** Clear the finished run's display state (snapshot + curve) back to idle.
   *  Files are untouched — the workspace stays resumable. Resolves true only
   *  when the backend accepted (it refuses while running / audition in
   *  flight) — the caller's wizard jump must not fire on a refused clear. */
  resetRun: () => Promise<boolean>;
}

export const useTrainingStore = create<TrainingStoreState>((set, get) => ({
  snapshot: IDLE_SNAPSHOT,
  snapshotAt: Date.now(),
  history: [],
  dragOverSpeakerId: null,
  flashSpeaker: null,
  config: { ...DEFAULT_CONFIG },
  route: { seg: "projects", projectId: "" },
  starting: false,
  diffWsInfo: null,
  slotInfo: null,
  retrainIntent: false,
  poolFlat: false,
  poolCount: 0,
  projectDataset: null,
  projectHasProgress: false,

  setRoute: (r) => set({ route: r }),
  setProjectInfo: (d) =>
    set({
      projectDataset: d?.dataset ?? null,
      poolCount: d?.dataset.files ?? 0,
      poolFlat: !!d && d.dataset.files > 0 && d.dataset.speakers.length === 0,
      // ★§F2⒝ 批 2 ④ —— 对每个 run 求或(与 ProjectDetail 的 `dataHasDependents` 同一条规则)。
      projectHasProgress:
        d?.slots.some(
          (s) =>
            s.ckptCount > 0 || s.runs.some((r) => r.hasResumePoint || r.info.has_main_progress),
        ) ?? false,
    }) ,

  refreshProjectDataset: async () => {
    const pid = get().route.projectId;
    if (!pid) {
      get().setProjectInfo(null);
      return;
    }
    const epoch = projectEpoch;
    try {
      const d = await invoke<ProjectDetail>("get_training_project", { projectId: pid });
      // the user may have left for another project while this was in flight — landing now would
      // describe project A while the page is on B (same rule as `addFiles`)
      if (epoch !== projectEpoch) return;
      get().setProjectInfo(d);
    } catch {
      if (epoch === projectEpoch) get().setProjectInfo(null);
    }
  },

  importIntoProject: async (files, speaker = null) => {
    const pid = get().route.projectId;
    if (!pid || files.length === 0) return;
    await invoke("import_project_dataset", { projectId: pid, files, speaker });
    await get().refreshProjectDataset();
    // one-shot「落到这位歌手了」pulse, resolved AFTER the refresh so a brand-new singer (whose
    // card did not exist when the import started) flashes too
    if (speaker) {
      const g = get().projectDataset?.groups.find((x) => (x.name || x.slug) === speaker);
      if (g) set({ flashSpeaker: { id: g.slug, nonce: ++flashSeq } });
    }
  },

  deleteFromProject: async (rels) => {
    const pid = get().route.projectId;
    if (!pid || rels.length === 0) return;
    await invoke("delete_project_dataset_files", { projectId: pid, rels });
    await get().refreshProjectDataset();
  },
  enterProject: (projectId, seg = "detail") => {
    // Invalidate everything that is mid-flight FOR THE OLD PROJECT: an in-progress file probe
    // (`addFiles`) and an in-progress archive scan (`refreshProjectCkpts`) would otherwise land
    // after this clear and repopulate the new project's form with the old project's contents.
    projectEpoch++;
    ckptScanSeq++;
    set((s) => {
      // Re-entering the project the LIVE run belongs to is not a fresh start: that run already
      // fixed the identity, and blanking it would leave its own run segment describing「—」.
      const live = s.snapshot;
      const isLiveRun =
        !!projectId && live.project_id === projectId && live.state !== "idle";
      const backend = (isLiveRun && asTrainingBackend(live.backend)) || DEFAULT_CONFIG.backend;
      return {
        route: { seg: projectId ? seg : "projects", projectId },
        // Everything below belongs to the project we are LEAVING. Keeping any of it is how a run
        // ends up describing one project while writing into another.
        // `backend` too: the data segment renders a singer list or a flat list depending on it,
        // so leaving the previous project's pick behind means「导入数据」opens the wrong shape.
        config: { ...s.config, modelName: isLiveRun ? live.model_name : "", backend },
        diffWsInfo: null,
        slotInfo: null,
        retrainIntent: false,
        // Unknown until the new project's detail loads and re-derives it — leaving the old
        // project's value would let a run skip the data page on a project that has none.
        poolFlat: false,
        poolCount: 0,
        projectDataset: null,
        projectHasProgress: false,
        projectCkpts: [],
      };
    });
  },
  goSeg: (seg) =>
    set((s) =>
      seg === "projects" || s.route.projectId
        ? { route: { seg, projectId: seg === "projects" ? "" : s.route.projectId } }
        : {},
    ),
  setDiffWsInfo: (info) => set({ diffWsInfo: info }),
  setSlotInfo: (info) => set({ slotInfo: info }),
  setRetrainIntent: (v) => set({ retrainIntent: v }),
  updateConfig: (u) => set((s) => ({ config: { ...s.config, ...u } })),

  setDragOverSpeakerId: (id) => set({ dragOverSpeakerId: id }),
  clearFlashSpeaker: (nonce) =>
    set((s) => (s.flashSpeaker?.nonce === nonce ? { flashSpeaker: null } : {})),
  projectCkpts: [],
  refreshProjectCkpts: async (projectId, backend) => {
    const seq = ++ckptScanSeq;
    if (!projectId) {
      if (seq === ckptScanSeq) set({ projectCkpts: [] });
      return;
    }
    try {
      const rows = await invoke<CkptRecord[]>("list_project_ckpts", {
        projectId,
        family: backendFamily(backend),
      });
      // Drop a response a newer scan has already superseded — switching architecture cards
      // re-fires this, so out-of-order replies are the normal case, not the exotic one.
      if (seq === ckptScanSeq) set({ projectCkpts: rows });
    } catch (e) {
      console.error("project ckpt scan failed", e);
      if (seq === ckptScanSeq) set({ projectCkpts: [] });
    }
  },

  refresh: async () => {
    try {
      const snapshot = await invoke<TrainingSnapshot>("get_training_status");
      const history = await invoke<StepPoint[]>("get_training_history");
      set((s) => {
        // during a live run, step events may have appended points while we
        // awaited — keep the local tail newer than the fetched copy instead of
        // clobbering it (it re-syncs fully at done anyway)
        const lastFetched = history.length ? history[history.length - 1]!.step : -1;
        const localTail = s.history.filter((p) => p.step > lastFetched);
        return {
          snapshot,
          history: localTail.length ? [...history, ...localTail] : history,
          snapshotAt: Date.now(),
        };
      });
    } catch (e) {
      console.error("training refresh failed", e);
    }
  },

  start: async (fresh, wipeConfirmed, resumeFrom) => {
    // S64 release gating (the S43 decision): no REAL training interpreter (dev venv / runtime pack /
    // manual slot) → the spawn is doomed on an end-user machine; offer the runtime download instead.
    // Dev machines always resolve training/.venv, so this only ever fires on packaged installs.
    try {
      if (!(await invoke<boolean>("training_env_ready"))) {
        const c = await useAppStore.getState().showConfirm({
          title: i18n.t("training.envMissingTitle"),
          body: i18n.t("training.envMissingBody"),
          buttons: [
            { id: "cancel", label: i18n.t("common.cancel") },
            { id: "goto", label: i18n.t("training.envMissingGoto"), kind: "primary" },
          ],
        });
        if (c === "goto" && !useAppStore.getState().settingsOpen) useAppStore.getState().toggleSettings();
        return;
      }
    } catch {
      /* ready-check unavailable → fall through; start_training's own error still surfaces loudly */
    }
    // S66 base-model gating: resolve the run's required assets through the SAME Rust source
    // try_start verifies against (training_required_assets ↔ resolve_training_assets) and turn
    // "missing base model" from a toast into a dialog with a one-click pack download. The param
    // derivation mirrors the request builder below (backend → version/sample_rate/aug fields).
    try {
      const cfg = get().config;
      const assetParams =
        cfg.backend === "rvc"
          ? { backend: "rvc", version: cfg.version, sampleRate: cfg.sampleRate, augCopies: cfg.augCopies }
          : cfg.backend === "vocoder"
            ? { backend: "vocoder", version: "nsf_hifigan", sampleRate: "44k", augCopies: cfg.vocAugCopies }
            : cfg.backend === "sovits_diff"
              ? { backend: "sovits_diff", version: cfg.diffVersion, sampleRate: "44k", augCopies: 0 }
              : cfg.backend === "sovits_v2"
                ? { backend: "sovits_v2", version: "4.0-v2", sampleRate: "44k", augCopies: cfg.sovitsAugCopies }
                : { backend: "sovits", version: cfg.sovitsVersion, sampleRate: "44k", augCopies: cfg.sovitsAugCopies };
      const assets = await invoke<RequiredAssetStatus[]>("training_required_assets", assetParams);
      const missing = assets.filter((a) => !a.exists);
      if (missing.length > 0) {
        // S75: ONE dialog. The S66 split existed because the license-bound vocoder ckpt was the one
        // asset the download button could NOT fetch — now it is mirrored like everything else, so
        // the split has nothing left to separate. What survives from that review is the real
        // obligation: state the terms BEFORE fetching (license line + upstream link below).
        const packItems = missing.filter((a) => a.pack);
        if (packItems.length > 0) {
          const packs = [...new Set(packItems.map((a) => a.pack as string))];
          const files = packItems.map((a) => `· ${a.path}`).join("\n");
          // A licensed pack also carries its NOTICE files (attribution travels WITH the weights),
          // so the download may write MORE files than this list — never fewer.
          const licenses = [
            ...new Set(packItems.map((a) => a.license).filter((l): l is string => !!l)),
          ];
          const upstream = packItems.find((a) => a.selfUrl)?.selfUrl ?? null;
          const buttons: ConfirmButton[] = [{ id: "cancel", label: i18n.t("common.cancel") }];
          if (upstream) buttons.push({ id: "open", label: i18n.t("training.assetsUpstream") });
          buttons.push({ id: "dl", label: i18n.t("training.assetsMissingDl"), kind: "primary" });
          const c = await useAppStore.getState().showConfirm({
            title: i18n.t("training.assetsMissingTitle"),
            body:
              `${i18n.t("training.assetsMissingBody")}\n${files}` +
              (licenses.length > 0
                ? `\n\n${i18n.t("training.assetsLicenseNote", { license: licenses.join(" / ") })}`
                : ""),
            buttons,
          });
          if (c === "open" && upstream) void openUrl(upstream).catch(() => {});
          if (c === "dl") {
            if (!useAppStore.getState().settingsOpen) useAppStore.getState().toggleSettings();
            const hfBase = hfBaseForMirror(useMsstModelStore.getState().mirror);
            // pack downloads are single-flight Rust-side — chain them; progress/errors surface
            // in the Settings "Model Assets" section the dialog just opened.
            void (async () => {
              for (const p of packs) {
                try {
                  await invoke("download_asset_pack", { id: p, hfBase });
                } catch {
                  /* shown by the Settings asset section (busy/cancel/fail all covered there) */
                }
              }
            })();
          }
          return;
        }
        // missing but not pack-distributed (unexpected) → fall through; try_start errors loudly.
      }
    } catch {
      /* pre-flight unavailable → fall through; start_training's own gate still rejects loudly */
    }
    const { config, route } = get();
    set({ starting: true });
    try {
      // ①c: SoVITS (α) + RVC (α′) data is the singer list. 1 singer = single-speaker: send its
      // files as the flat dataset_files, NO `speakers` key -> byte-identical to pre-①c. ≥2
      // singers: send `speakers` (matching Rust StartTrainingRequest.speakers:
      // Vec<SpeakerGroup{name,files}>) + empty dataset_files. diff/vocoder keep the flat `dataset`.
      const isMulti = backendSupportsMultiSpeaker(config.backend);
      // S78: the data lives on disk BEFORE a run starts — importing is its own act now, so a
      // start never carries files. A per-singer project instead DECLARES its structure (every
      // group's `files` empty), which the backend validates against the directories that are
      // actually there and then consumes without copying anything. Names come from the disk
      // listing so the slug they derive to is the slug the data is already under.
      const diskGroups = get().projectDataset?.groups ?? [];
      const multi = isMulti && diskGroups.length > 1;
      const datasetFiles: string[] = [];
      const base = {
        model_name: config.modelName.trim(),
        // S76 batch 4: WHICH project, explicitly. `model_name` is now「本次训练名」— editable
        // and only the ARTIFACT identity — so letting the backend re-derive the directory from
        // it would fork a second project the first time somebody renames a run.
        project_id: route.projectId,
        backend: config.backend,
        dataset_files: datasetFiles,
        ...(multi
          ? {
              speakers: diskGroups.map((g) => ({
                name: g.name || g.slug,
                files: [] as string[],
              })),
            }
          : {}),
        gpu: config.gpu,
        force_cpu: config.forceCpu,
        spk_id: 0,
        fresh,
        wipe_confirmed: wipeConfirmed,
        // ⚠ empty string, not omitted: the Rust field is `#[serde(default)] String` and the
        // request builder normalizes "" -> "latest", so an old payload and a fresh run behave
        // exactly as they did before this existed.
        resume_from: fresh ? "" : (resumeFrom ?? "latest"),
        // ★§F2⒝ 批 2 ④ —— 同样是「空串而不是省略」:Rust 那边是 `#[serde(default)] String`,
        // 而 `run_id_of` 把空串归成 `None`,于是旧载荷与今天的每槽一 run 行为逐字节相同。
        run_id: config.runId,
      };
      const request =
        config.backend === "rvc"
          ? {
              ...base,
              version: config.version,
              sample_rate: config.sampleRate,
              total_epoch: config.totalEpoch,
              batch_size: config.batchSize,
              save_every_epoch: config.saveEveryEpoch,
              save_every_weights: config.saveEveryWeights,
              keep_only_latest: config.keepOnlyLatest,
              cache_gpu: config.cacheGpu,
              fp16: config.fp16,
              aug_copies: config.augCopies,
            }
          : config.backend === "vocoder"
            ? {
                ...base,
                // fixed markers, not user choices (一期单格式类); total_epoch 0
                // = the step-based sentinel (the UI hides epoch displays)
                version: "nsf_hifigan",
                sample_rate: "44k",
                total_epoch: 0,
                batch_size: config.vocBatchSize,
                total_steps: config.vocTotalSteps,
                save_every_steps: config.vocSaveEverySteps,
                keep_ckpts: config.vocKeepCkpts,
                crop_mel_frames: config.vocCropMelFrames,
                freeze_mpd: config.vocFreezeMpd,
                aug_copies: config.vocAugCopies,
              }
          : config.backend === "sovits_diff"
            ? {
                ...base,
                version: config.diffVersion,
                sample_rate: "44k",
                // sentinel: diffusion progress is step-based; the UI hides
                // epoch displays when total_epochs is 0
                total_epoch: 0,
                batch_size: config.diffBatchSize,
                save_every_steps: config.diffSaveEverySteps,
                total_steps: config.diffTotalSteps,
                k_step_max: config.diffKStepMax,
                interval_force_save: config.diffForceSaveSteps,
                cache_all_data: config.diffCacheAllData,
                fp16: config.diffFp16,
                // honoured only when the sovits slot holds no main model (see `eff_aug_copies`)
                aug_copies: config.diffAugCopies,
              }
          : config.backend === "sovits_v2"
            ? {
                ...base,
                // S68 VISinger2: fixed version marker; shares the sovits* form
                // fields. No vol_embedding/all_in_mem (v2 has neither); fp16
                // structurally off (pure fp32 upstream — Rust normalizes too)
                version: "4.0-v2",
                sample_rate: "44k",
                total_epoch: config.sovitsTotalEpoch,
                batch_size: config.sovitsBatchSize,
                save_every_steps: config.sovitsSaveEverySteps,
                keep_ckpts: config.sovitsKeepCkpts,
                fp16: false,
                loudnorm: config.sovitsLoudnorm,
                kmeans: config.sovitsKmeans,
                aug_copies: config.sovitsAugCopies,
              }
            : {
              ...base,
              version: config.sovitsVersion,
              sample_rate: "44k",
              total_epoch: config.sovitsTotalEpoch,
              batch_size: config.sovitsBatchSize,
              save_every_steps: config.sovitsSaveEverySteps,
              keep_ckpts: config.sovitsKeepCkpts,
              fp16: config.sovitsFp16,
              // 响度嵌入 is a 4.1 feature — the 4.0 card trains ecosystem-
              // compatible checkpoints, so it stays structurally off there
              vol_embedding:
                config.sovitsVersion === "4.1" ? config.sovitsVolEmbedding : false,
              loudnorm: config.sovitsLoudnorm,
              kmeans: config.sovitsKmeans,
              all_in_mem: config.sovitsAllInMem,
              aug_copies: config.sovitsAugCopies,
            };
      await invoke("start_training", { request });
      // Only on success, and before the refresh — a rejected start must leave the user on the
      // page they pressed the button from, with their form intact.
      set({ route: { seg: "run", projectId: route.projectId }, history: [] });
      useAppStore.getState().showToast(i18n.t("training.started"), "info");
      await get().refresh();
    } catch (e) {
      // Localize known backend CODEs (APP_BUSY: an audition/render holds the FlightGuard); the raw
      // error still rethrows unchanged for any caller that inspects it.
      useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
      throw e;
    } finally {
      set({ starting: false });
    }
  },

  stop: async () => {
    try {
      await invoke("stop_training");
      set((s) => ({ snapshot: { ...s.snapshot, stop_requested: true } }));
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
    }
  },

  forceStop: async () => {
    try {
      await invoke("force_stop_training");
    } catch (e) {
      useAppStore.getState().showToast(String(e), "error");
    }
  },

  resetRun: async () => {
    try {
      await invoke("reset_training_display");
      // only clear locally once the backend agreed (it refuses while running)
      set({ snapshot: IDLE_SNAPSHOT, history: [], snapshotAt: Date.now() });
      return true;
    } catch (e) {
      // APP_BUSY: an audition/render holds the FlightGuard → info, real failures stay errors.
      useAppStore.getState().showToast(backendErrorMessage(e) ?? String(e), isBusyError(e) ? "info" : "error");
      return false;
    }
  },
}));

let unlistens: UnlistenFn[] | null = null;
let installing = false;

/** Idempotent global listener install (App mount) — keeps the titlebar indicator
 *  and the loss history live even while the training page is closed. The sync
 *  sentinel closes the await window (StrictMode double-mount would double-install
 *  and duplicate every history point + toast). */
export async function setupTrainingListeners() {
  if (unlistens || installing) return;
  installing = true;
  unlistens = await Promise.all([
    listen<StageInfo>("training-stage", (e) => {
      useTrainingStore.setState((s) => ({
        snapshot: { ...s.snapshot, stage: e.payload },
      }));
    }),
    listen<StepInfo>("training-step", (e) => {
      useTrainingStore.setState((s) => {
        let history = s.history;
        if (history.length >= HISTORY_CAP) {
          history = history.filter((_, i) => i % 2 === 0);
        }
        // NB: snapshotAt is NOT touched here — it anchors the elapsed
        // extrapolation to the last full refresh (elapsed_secs base); resetting
        // it per step would freeze the displayed elapsed at the base value
        return {
          snapshot: { ...s.snapshot, state: "running", step: e.payload },
          history: [
            ...history,
            { step: e.payload.step, lr: e.payload.lr, losses: e.payload.losses },
          ],
        };
      });
    }),
    listen<CkptInfo>("training-ckpt", (e) => {
      useTrainingStore.setState((s) => {
        const kept =
          e.payload.kind === "best" || e.payload.kind === "final"
            ? s.snapshot.ckpts.filter((c) => c.kind !== e.payload.kind)
            : s.snapshot.ckpts;
        return { snapshot: { ...s.snapshot, ckpts: [...kept, e.payload] } };
      });
    }),
    listen<TrainingSnapshot>("training-done", (e) => {
      useTrainingStore.setState({ snapshot: e.payload, snapshotAt: Date.now() });
      const t = i18n.t.bind(i18n);
      const app = useAppStore.getState();
      if (e.payload.state === "completed") {
        app.showToast(t("training.doneCompleted"), "success");
      } else if (e.payload.state === "stopped") {
        app.showToast(t("training.doneStopped"), "info");
      } else if (e.payload.state === "error") {
        // snapshot.error carries the run_worker's stable CODE strings — localize known ones.
        const err = e.payload.error ?? "";
        app.showToast(`${t("training.doneError")}: ${backendErrorMessage(err) ?? err}`, "error");
      }
      // the final force-emitted step may have landed Rust-side only — resync once
      void useTrainingStore.getState().refresh();
    }),
    listen<string>("training-state", (e) => {
      if (e.payload === "running") {
        useTrainingStore.setState((s) => ({
          snapshot: { ...s.snapshot, state: "running" },
        }));
      }
    }),
  ]);
}
