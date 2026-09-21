// ② Singer selection (S58): THE single pick path shared by the vocal sidebar's singer <select> and the
// track-header singer popup — picking a voice sets Track.voiceModel (+avatar) AND auto-detects the
// backend from the model's type, in ONE undo step. Never duplicate this pair of writes (a fork would
// eventually disagree on the backend or the transaction bracket).
import { useProjectStore } from "../../store/project";
import { useHistoryStore } from "../../store/history";
/** The backend a singer model runs on, from its serde `model_type` ("Rvc" | "SoVits") — the model's TYPE
 *  drives the backend, so there's no manual toggle (§user: unified SoVITS/RVC singer list). */
export const backendOf = (m) => (m.model_type === "Rvc" ? "rvc" : "sovits");
export const backendLabel = (m) => (backendOf(m) === "sovits" ? "SoVITS" : "RVC");
/** Pick a singer for a track: voiceModel + avatar + backend, as ONE undo step. */
export function pickVoiceForTrack(trackId, m) {
    const hist = useHistoryStore.getState();
    const store = useProjectStore.getState();
    hist.beginTransaction();
    store.updateTrack(trackId, { voiceModel: m.name, voiceModelAvatar: m.avatar_path ?? undefined });
    store.setVocalParams(trackId, { backend: backendOf(m) });
    hist.commitTransaction();
}
