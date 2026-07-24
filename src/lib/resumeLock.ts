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
  if (backend !== "sovits_diff") out.push({ id: "augCopies", tier: "costly" });
  out.push({ id: "dataset", tier: "costly" });
  return out;
}

/**
 * Is a start on this slot going to be a RESUME, i.e. are the locked fields actually pinned?
 *
 * A slot with nothing in it pins nothing, and 重训 unpins everything — but the params page is
 * upstream of that choice, so it shows the fields as locked whenever the slot HAS progress and
 * says out loud that 重训 is the way to change them. That matches the backend exactly: the
 * guard runs on `!fresh`.
 */
export function resumeWouldBeGuarded(
  backend: string,
  info: { exists: boolean; has_main_progress: boolean; diff_steps: number } | null,
): boolean {
  if (!info || !info.exists) return false;
  return backend === "sovits_diff" ? info.diff_steps > 0 : info.has_main_progress;
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
