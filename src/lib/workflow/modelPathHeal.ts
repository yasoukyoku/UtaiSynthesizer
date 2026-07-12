// S64 portability — heal persisted ABSOLUTE model paths against the freshly-scanned model stores.
//
// Workflow node params (and therefore .usp files) persist `modelPath` absolute by design (bundle.ts:
// models stay external references). That breaks in exactly the fully-portable scenarios this release
// must support: the install dir copied elsewhere, the data dir migrated, or a project opened on
// another machine. The node-panel pickers already self-heal ON MOUNT (useVoiceModelSelection /
// SeparationNode's modelsDir effect) — but a render triggered WITHOUT ever opening the panel consumed
// the stale path and failed. These helpers are the shared USE-time counterpart: resolve the model by
// its STABLE identity (voiceName / modelFile — what the pickers key on) and prefer the store's path.
// They deliberately do NOT write back into node params: persistence stays owned by the pickers, so
// this can never fight their mount-time updates or create undo noise.
//
// (The vocal-track render path needs none of this — renderVocalPart already resolves by
// track.voiceModel name on every call. Track AVATARS are healed at project load, below.)

import { useVoiceModelStore } from "../../store/voice-models";
import { useMsstModelStore } from "../../store/msst-models";
import { DEFAULT_VOCAL_PARAMS } from "../../store/project";
import type { Track } from "../../types/project";

/** Voice-node heal: current path of the named model per the store scan, else the persisted path
 *  (model deleted/renamed — let the run fail with the model-missing error, same as before). */
export async function healVoiceModelPath(
  voiceType: "rvc" | "sovits",
  voiceName: string | undefined,
  modelPath: string | undefined,
): Promise<string | undefined> {
  if (!voiceName) return modelPath;
  const store = useVoiceModelStore.getState();
  if (store.models[voiceType].length === 0) {
    await store.fetchModels().catch(() => {});
  }
  const entry = useVoiceModelStore.getState().models[voiceType].find((m) => m.name === voiceName);
  return entry?.path ?? modelPath;
}

/** THE ckpt/pth→onnx path mapper for MSST models (moved from SeparationNode so the node UI and the
 *  engine-side heal can never drift on the extension rule). */
export function msstOnnxPath(modelsDir: string, filename: string): string {
  const onnxName = filename.replace(/\.(ckpt|th|pth)$/, ".onnx");
  return `${modelsDir.replace(/\\/g, "/")}/${onnxName}`;
}

/** Separation-node heal: recompute the onnx path from the CURRENT models dir + the node's stable
 *  modelFile. Falls back to the persisted path when the file identity or dir is unavailable. */
export async function healMsstModelPath(modelFile: string | undefined, modelPath: string): Promise<string> {
  if (!modelFile) return modelPath;
  const st = useMsstModelStore.getState();
  if (!st.modelsDir) {
    await st.fetchModelsDir().catch(() => {});
  }
  const dir = useMsstModelStore.getState().modelsDir;
  return dir ? msstOnnxPath(dir, modelFile) : modelPath;
}

/** Load-time avatar heal (openProjectFile / restoreAutosave, AFTER parse and BEFORE the store set —
 *  history is reset right after a load, so this never creates an undo step or dirties the project).
 *  `Track.voiceModelAvatar` persists absolute (bundle.ts: external reference); re-resolve it from the
 *  singer's registry entry — the exact lookup renderVocalPart uses — so avatars survive a machine/
 *  data-dir move. Unresolvable singers keep their stored value (the image may still exist). */
export async function healLoadedTrackAvatars(tracks: Track[]): Promise<void> {
  if (!tracks.some((tr) => tr.voiceModel)) return;
  const store = useVoiceModelStore.getState();
  if (store.models.rvc.length === 0 && store.models.sovits.length === 0) {
    await store.fetchModels().catch(() => {});
  }
  const models = useVoiceModelStore.getState().models;
  for (const tr of tracks) {
    if (!tr.voiceModel) continue;
    const backend = (tr.vocalParams ?? DEFAULT_VOCAL_PARAMS).backend;
    const entry = models[backend]?.find((m) => m.name === tr.voiceModel);
    if (entry) tr.voiceModelAvatar = entry.avatar_path ?? undefined;
  }
}
