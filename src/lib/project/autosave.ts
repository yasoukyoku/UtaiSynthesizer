import { invoke } from "@tauri-apps/api/core";
import { useProjectStore } from "../../store/project";
import { buildAutosaveJson } from "./bundle";

/**
 * Autosave — a fast, frequent crash-recovery snapshot. It reuses the SAME serialization as a real save
 * (`buildAutosaveJson` → `serializeProject`) so it can never drift from save as features are added, but
 * keeps media/render paths ABSOLUTE and copies nothing (it references your original files in place).
 *
 * The recovery unit is the DOCUMENT, not the store's `dirty` flag — `dirty` deliberately excludes baked
 * renders (processedOutputs), so relying on it would wrongly drop an unsaved render from recovery. Instead
 * we compare the actual serialized document to the last REAL save (`savedJson`): different → write a
 * snapshot; empty or identical-to-saved → clear (nothing to recover). The autosave FILE's existence is the
 * unclean-exit marker.
 */

const DEBOUNCE_MS = 1500; // write this long after the user stops editing
const MAX_INTERVAL_MS = 25000; // …but at least this often during continuous editing (debounce never settles)

let debounceTimer = 0;
let lastWrite = 0;
let clearEpoch = 0; // bumped by clearAutosave; flush re-clears if it changed across an in-flight write
let savedJson: string | null = null; // serialized doc as of the last REAL save / open / new (clean baseline)
let recoveryPending = false; // true while the startup recover prompt is open — don't clobber the slot
let current: Promise<void> | null = null; // the in-flight flush chain (single-flight guard + awaitable handle)
let pendingFlush = false; // a change arrived mid-write → run one trailing flush with the latest content

interface AutosaveEnvelope {
  filePath: string | null;
  name: string;
  savedAt: number;
  projectJson: string;
}

function currentProjectJson(): string {
  const s = useProjectStore.getState();
  return buildAutosaveJson(s.name || "Untitled", s.tracks, s.tempo, s.timeSignature);
}

function buildEnvelope(projectJson: string): string {
  const s = useProjectStore.getState();
  const env: AutosaveEnvelope = { filePath: s.filePath, name: s.name, savedAt: Date.now(), projectJson };
  return JSON.stringify(env);
}

/** One flush pass + any coalesced trailing rerun. The returned promise settles only when THIS
 *  call's work (write or clear, plus the trailing rerun it absorbed) is durably done — exit-like
 *  awaiters (S64 update install: the installer kills the process) depend on that. */
async function doFlush(): Promise<void> {
  const s = useProjectStore.getState();
  // Empty doc, or identical to the last real save → nothing worth recovering: drop any stale file.
  const json = s.tracks.length === 0 ? null : currentProjectJson();
  if (json === null || json === savedJson) {
    await clearAutosave();
    return;
  }
  lastWrite = Date.now(); // optimistic — self-throttle the force-flush branch while this write is in flight
  const epoch = clearEpoch;
  try {
    await invoke("write_autosave", { json: buildEnvelope(json) });
    // If a clear raced in while this write was in flight, the rename may have re-created the file → re-clear.
    if (clearEpoch !== epoch) void invoke("clear_autosave").catch(() => {});
  } catch {
    /* autosave is best-effort — a failed write must never disrupt editing */
  }
  if (pendingFlush) {
    pendingFlush = false;
    await doFlush(); // trailing run with the latest content — awaited so `current` covers it
  }
}

function flush(): Promise<void> {
  if (recoveryPending) return Promise.resolve(); // recovery decision pending — don't clobber the slot
  if (current) {
    pendingFlush = true; // coalesce: the in-flight chain reruns with the latest content before settling
    return current;
  }
  current = doFlush().finally(() => {
    current = null;
  });
  return current;
}

function schedule() {
  clearTimeout(debounceTimer);
  // Force a write if continuous editing has kept the debounce from ever settling past MAX_INTERVAL.
  if (Date.now() - lastWrite > MAX_INTERVAL_MS) {
    void flush();
  } else {
    debounceTimer = window.setTimeout(() => void flush(), DEBOUNCE_MS);
  }
}

/** Start autosave: snapshot the document (debounced) whenever it changes vs. the last real save. Returns
 *  an unsubscribe (HMR-safe). Subscribes to the PERSISTED document fields only — never `playheadTick`. */
export function installAutosave(): () => void {
  lastWrite = Date.now();
  savedJson = currentProjectJson(); // baseline = the (empty) initial document
  const unsub = useProjectStore.subscribe((st, prev) => {
    if (st.tracks !== prev.tracks || st.tempo !== prev.tempo || st.timeSignature !== prev.timeSignature) {
      schedule();
    }
  });
  return () => {
    clearTimeout(debounceTimer);
    unsub();
  };
}

/** True when the live document differs from the last real save (the SAME content-compare autosave uses).
 *  Unlike the store's `dirty`, this counts an unsaved render, ignores view-only toggles (expanded), and
 *  treats an empty project as nothing-to-save — so the window-close prompt matches what autosave would
 *  actually recover. */
export function hasUnsavedWork(): boolean {
  const s = useProjectStore.getState();
  if (s.tracks.length === 0) return false;
  return currentProjectJson() !== savedJson;
}

/** Mark the current document as the clean saved baseline (after a real save / open / new) and drop the
 *  recovery file — there's nothing unsaved to recover. */
export async function markAutosaveBaseline(): Promise<void> {
  savedJson = currentProjectJson();
  await clearAutosave();
}

/** Remove the autosave file (best-effort). Does NOT change the saved baseline — used when abandoning
 *  unsaved work (Don't Save / Discard / Dismiss). */
export async function clearAutosave(): Promise<void> {
  clearTimeout(debounceTimer);
  clearEpoch++;
  try {
    await invoke("clear_autosave");
  } catch {
    /* ignore */
  }
}

/** While true, flush() refuses to write — used to protect the recovery slot while the startup
 *  "Recover?" prompt is open (e.g. an OS file-drop must not overwrite the file before the user decides). */
export function setRecoveryPending(v: boolean): void {
  recoveryPending = v;
}

/** Whether a recovery decision is still pending. openProjectFile checks this before pruning usp_work —
 *  a crash-recovered .usp project's media lives there, and Ctrl+O while the recover prompt is open must
 *  not delete what "Recover" is about to hydrate. */
export function isRecoveryPending(): boolean {
  return recoveryPending;
}

/** Re-arm autosave after a freeze (e.g. an aborted exit): a flush() that fired while recoveryPending was
 *  true returned without writing or rescheduling, so edits made during the dialogs would wait for the next
 *  change. Call this when clearing the freeze to capture them promptly. */
export function rearmAutosave(): void {
  schedule();
}

/** Force an IMMEDIATE autosave (bypass the debounce) — for commit-like milestones worth snapshotting at
 *  once, e.g. a finished render/separation, so a fast reload right after doesn't lose it to the 1.5s wait.
 *  Returns the write's promise so exit-like callers (S64 update install: the installer kills the process)
 *  can AWAIT the snapshot; fire-and-forget callers just ignore it. */
export function flushAutosaveNow(): Promise<void> {
  clearTimeout(debounceTimer);
  return flush();
}

/** Read the autosave envelope left by an unclean exit (null if last session shut down cleanly). */
export async function readAutosave(): Promise<AutosaveEnvelope | null> {
  try {
    const raw = await invoke<string | null>("read_autosave");
    if (!raw) return null;
    return JSON.parse(raw) as AutosaveEnvelope;
  } catch {
    return null;
  }
}
