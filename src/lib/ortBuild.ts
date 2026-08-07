/**
 * "Would restarting actually change which ORT build loads?" — the predicate behind the Settings
 * device-section `buildMismatch` hint, extracted so it can be pinned by a test.
 *
 * WHY IT IS ITS OWN FILE. The hint is an ASSERTION rendered to the user ("restart the app for this
 * choice to take effect"), so it is only allowed to fire when a restart really would change the
 * build. The authority for that is `init_ort_runtime` in `src-tauri/src/lib.rs`; the hint is a
 * mirror, and an unchecked mirror is just a second opinion (same reasoning as `resumeLockParity`).
 *
 * ⛔ THE BUG THIS REPLACES (S116). The Auto/NVIDIA leg used to test `hw.cuda_available`, which is
 * `has_nvidia && is_cuda_available()` (settings.rs:181) = "some cudart64_12.dll is reachable via
 * PATH or CUDA_PATH". That is NOT what decides the build. The build gate is `cuda_pkg_supported()`
 * (lib.rs:280) — an nvidia-smi COMPUTE-CAP read that fails CLOSED (S74b). On a Blackwell or
 * pre-Turing NVIDIA box that still has our CUDA package on disk (or any box with the CUDA Toolkit
 * installed), `cuda_available` is true while `cuda_pkg_supported()` is false forever ⇒ the panel
 * rendered a red "restart to take effect" that no restart could ever satisfy, next to a note
 * saying the very same package is unsupported here.
 *
 * ⚠ `cuda_supported` ALONE is not the fix either: a fully supported NVIDIA box that never
 * downloaded the CUDA package has `cuda_supported === true`, no `runtime/ort/cuda/onnxruntime.dll`,
 * and `init_ort_runtime` logs "CUDA preferred but runtime/ort/cuda/ missing — using DirectML build"
 * (lib.rs:344-346). That is the same false promise with a different cause. `cudaReady`
 * (`is_cuda_runtime_ready`) is the term that closes it: it verifies the CUDA ORT build, its
 * providers DLL, the CUDA major it was built for, AND `cuda_provider_deps_resolvable` — the last of
 * which lib.rs's Auto arm calls too, by design ("Shared by is_cuda_runtime_ready AND lib.rs' Auto
 * build pick", settings.rs:1989-1990). It is strictly stronger than lib.rs's conjunction, and the
 * one case where it is stricter (CUDA ORT build present but no providers DLL) is a build that would
 * load and then fail EP registration — so staying silent there is the honest answer, not a miss.
 */
export interface OrtBuildHintInput {
  /** The saved inference device preference: "auto" | "cuda" | "directml" | "cpu". */
  device: string;
  /**
   * `HardwareInfo.ort_build` — which build THIS process actually loaded.
   * ⚠ Four possible values, not two: "CUDA" | "DirectML" | `dev/system (<path>)` | "system PATH"
   * (lib.rs `init_ort_runtime`). The last two mean "loaded from a fallback source, provider set
   * unknown"; they are neither of the legs below, so no hint fires — which is right, because we
   * cannot say what a restart would do. Rust asks this question through `ort_build_is_cuda`;
   * this file is the frontend mirror of the same two literals.
   */
  ortBuild: string;
  /** Vendor of the Auto-mode preferred GPU; undefined when the device is not Auto or none is picked. */
  autoVendor: string | undefined;
  /** `HardwareInfo.cuda_supported` = `cuda_pkg_supported()`, the S74b package gate (fail-CLOSED). */
  cudaSupported: boolean;
  /** `is_cuda_runtime_ready` — the CUDA package is present AND actually usable on this machine. */
  cudaReady: boolean;
}

/**
 * True iff the loaded build contradicts the current preference AND a restart would resolve it.
 *
 * The four legs, and what each one needs a restart to be able to deliver:
 *  - explicit "cuda" while on DirectML → needs the CUDA build to be loadable at all
 *    (`cudaSupported && cudaReady`);
 *  - explicit "directml" while on CUDA → the DirectML build is the unconditional default in
 *    `init_ort_runtime`'s search path, so a restart always delivers it;
 *  - Auto with a NON-NVIDIA preferred GPU while on CUDA → `picked_non_nvidia` forces the DirectML
 *    build on the next launch (lib.rs:293-297), again unconditionally;
 *  - Auto with an NVIDIA preferred GPU while on DirectML → the only leg that has to ASK, and the
 *    one the old `cuda_available` test got wrong.
 */
export function restartWouldChangeOrtBuild(i: OrtBuildHintInput): boolean {
  const cudaBuildIsReachable = i.cudaSupported && i.cudaReady;
  if (i.device === "cuda" && i.ortBuild === "DirectML") return cudaBuildIsReachable;
  if (i.device === "directml" && i.ortBuild === "CUDA") return true;
  if (i.autoVendor === undefined) return false;
  if (i.autoVendor !== "nvidia" && i.ortBuild === "CUDA") return true;
  if (i.autoVendor === "nvidia" && i.ortBuild === "DirectML") return cudaBuildIsReachable;
  return false;
}
