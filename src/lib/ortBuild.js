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
export function restartWouldChangeOrtBuild(i) {
    const cudaBuildIsReachable = i.cudaSupported && i.cudaReady;
    if (i.device === "cuda" && i.ortBuild === "DirectML")
        return cudaBuildIsReachable;
    if (i.device === "directml" && i.ortBuild === "CUDA")
        return true;
    if (i.autoVendor === undefined)
        return false;
    if (i.autoVendor !== "nvidia" && i.ortBuild === "CUDA")
        return true;
    if (i.autoVendor === "nvidia" && i.ortBuild === "DirectML")
        return cudaBuildIsReachable;
    return false;
}
