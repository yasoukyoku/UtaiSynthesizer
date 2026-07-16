import i18n from "../i18n";
import { useAppStore } from "../store/app";
import { isModalError } from "./backendError";

/**
 * S67c: route a localized backend-error display string to its vessel. Modal-class codes
 * (backendError CodeEntry.modal — fatal "the run stopped, act outside the app" errors like
 * INFERENCE_LOW_MEMORY) open the self-drawn ConfirmDialog in single-OK alert mode: their
 * guidance text is too long for a 5 s toast and invisible in a node tooltip. Everything
 * else returns false and the caller keeps its own toast/tooltip path. THE single modal
 * funnel — workflow engine + vocal render both call this; never fork a per-site copy.
 */
export function maybeShowErrorModal(raw: unknown, display: string): boolean {
  if (!isModalError(raw)) return false;
  // Collision guard (same rule as App.tsx's update-check funnel): showConfirm force-settles
  // any dialog the user is currently answering — never destroy it for a background failure.
  // Returning false hands the message back to the caller's toast path instead.
  if (useAppStore.getState().confirm) return false;
  void useAppStore.getState().showConfirm({
    title: i18n.t("common.errorModalTitle"),
    body: display,
    buttons: [{ id: "ok", label: i18n.t("common.ok"), kind: "primary" }],
  });
  return true;
}
