/**
 * 续训锁 —— the frontend half of `src-tauri/src/training/resume_lock.rs`.
 *
 * The Rust table is the AUTHORITY (it is what `try_start` enforces). This mirror exists because
 * the parameters page has to render locked fields read-only BEFORE any start, and an `invoke`
 * per keystroke is not that. `src/lib/resumeLockParity.test.ts` reads the Rust source and fails
 * if the two ever describe different fields — the same trick `ipcParity` uses for commands.
 *
 * Editing rule: change `resume_lock.rs` first, then this file. Nothing here may add a field the
 * Rust table does not have — the guard would allow it and the UI would refuse it.
 */

/** `locked` = a resume is REFUSED outright (only 重训 unlocks it).
 *  `costly` = allowed, but it re-fingerprints the dataset ⇒ the next run re-preprocesses. */
export type LockTier = "locked" | "costly";

/** 这一项命名的是哪一层的产物 —— 镜像 `resume_lock::LockScope`。
 *
 *  与 `LockTier` **正交**:tier 问「续训改它会不会被拒」(关于守卫),scope 问「改了它之后
 *  哪一层的产物作废」(关于盘)。`version` 在 SoVITS 上两者都是,`volEmbedding` 是 locked 但
 *  与池无关,`augCopies` 是 costly 但整份预处理作废。 */
export type LockScope = "run" | "pool" | "both";

/** 改了这一项,这个槽已有的**预处理产物**还能不能用。参数页的代价提示就挂在它上面。 */
export function scopeInvalidatesPool(scope: LockScope): boolean {
  return scope !== "run";
}

export interface LockedField {
  id: string;
  tier: LockTier;
  scope: LockScope;
}

/** Mirrors `resume_lock::resume_locked_fields`. Ids are the shared vocabulary. */
export function resumeLockedFields(backend: string): LockedField[] {
  const sovitsFamily =
    backend === "sovits" || backend === "sovits_diff" || backend === "sovits_v2";
  // sovits 家的版本选的是 ContentVec 空间(`|enc=` 就在 fp_text 里);rvc 的 v1/v2 只是在同一个
  // 池里切两个**共存**的特征子目录;vocoder 的版本是常量标记。
  const verScope: LockScope = sovitsFamily ? "both" : "run";
  const out: LockedField[] = [
    { id: "version", tier: "locked", scope: verScope },
    { id: "sampleRate", tier: "locked", scope: "both" },
  ];
  if (backend === "sovits") out.push({ id: "volEmbedding", tier: "locked", scope: "run" });
  if (backend === "sovits" || backend === "rvc" || backend === "sovits_v2") {
    out.push({ id: "speakerCount", tier: "locked", scope: "both" });
    out.push({ id: "speakerSet", tier: "locked", scope: "both" });
  }
  if (backend === "sovits_diff") out.push({ id: "kStepMax", tier: "locked", scope: "run" });
  // ★§F2⒝ ④d 笔 1:sovits_v2 也送 loudnorm、也折进它自己的 fp_text —— 两侧此前一致地漏了这一行。
  if (backend === "sovits" || backend === "sovits_v2") {
    out.push({ id: "loudnorm", tier: "costly", scope: "pool" });
  }
  out.push({ id: "augCopies", tier: "costly", scope: "pool" });
  out.push({ id: "dataset", tier: "costly", scope: "pool" });
  return out;
}

/**
 * Is a start on this slot going to be a RESUME, i.e. are the locked fields actually pinned?
 *
 * A slot with nothing in it pins nothing, and 重训 unpins everything — but the params page is
 * upstream of that choice, so it shows the fields as locked whenever a resume WOULD be refused
 * and says out loud that 重训 is the way to change them.
 *
 * This must mirror the backend guard `check_resume_locks`, which fires the moment a VERSIONED
 * `run_manifest.json` exists — that manifest is written BEFORE the worker begins preprocessing
 * (mod.rs), so a run stopped mid-preprocess (no `G_*.pth` checkpoint yet) still pins
 * version/sampleRate/… on resume. Keying on a checkpoint (`has_main_progress`) would leave those
 * fields editable in that window while the backend refuses the resume — the exact "UI lets you
 * edit it, resume refuses it" trap this module exists to prevent (审查 S78).
 *
 * `sovits_diff` is the exception: its version is pinned by the MAIN model (not this manifest), so
 * the diffusion-specific progress signal (`diff_steps`) is the right gate there.
 */
export function resumeWouldBeGuarded(
  backend: string,
  info:
    | { exists: boolean; has_main_progress: boolean; diff_steps: number; version: string; sample_rate: string }
    | null,
): boolean {
  if (!info || !info.exists) return false;
  if (backend === "sovits_diff") return info.diff_steps > 0;
  // Mirrors check_resume_locks: a non-empty version OR sample_rate in the manifest = a resume is
  // guarded. They are always written together, so either being set means "a run has started here".
  return info.version !== "" || info.sample_rate !== "";
}

/** Which params-page controls a locked field owns. Empty = the field is not editable there
 *  (speakers live on the data page, `dataset` is the data page itself). */
export function lockedFieldIds(backend: string, tier: LockTier): Set<string> {
  return new Set(
    resumeLockedFields(backend)
      .filter((f) => f.tier === tier)
      .map((f) => f.id),
  );
}

/** 这个 backend 上「改了就得重跑预处理」的那些项 —— 参数页的代价提示读它。
 *
 *  ⛔ 判据是 **scope 而不是 tier**:tier=costly 说的是「不会被拒」,而屏幕上要说的是
 *  「会重跑」。今天两者的交集恰好相等,但那是巧合不是定义 —— 一个 locked 的池级项
 *  (sovits 的 `version`)同样会重跑,只是它走的是「续训锁定」那条渲染路径。 */
export function poolInvalidatingIds(backend: string): Set<string> {
  return new Set(
    resumeLockedFields(backend)
      .filter((f) => scopeInvalidatesPool(f.scope))
      .map((f) => f.id),
  );
}
