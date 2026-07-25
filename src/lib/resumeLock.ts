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

/** Mirrors `resume_lock::resume_locked_fields`. Ids are the shared vocabulary. */
export function resumeLockedFields(backend: string): { id: string; tier: LockTier }[] {
  const out: { id: string; tier: LockTier }[] = [
    { id: "version", tier: "locked" },
    { id: "sampleRate", tier: "locked" },
  ];
  if (backend === "sovits") out.push({ id: "volEmbedding", tier: "locked" });
  if (backend === "sovits" || backend === "rvc" || backend === "sovits_v2") {
    out.push({ id: "speakerCount", tier: "locked" });
    out.push({ id: "speakerSet", tier: "locked" });
  }
  if (backend === "sovits_diff") out.push({ id: "kStepMax", tier: "locked" });
  if (backend === "sovits") out.push({ id: "loudnorm", tier: "costly" });
  out.push({ id: "augCopies", tier: "costly" });
  out.push({ id: "dataset", tier: "costly" });
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
