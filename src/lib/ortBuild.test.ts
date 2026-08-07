/**
 * The Settings `buildMismatch` hint is an ASSERTION shown to the user ("restart the app for this
 * choice to take effect"), so it needs two different gates and they catch different mistakes:
 *
 *  (1) a truth table over `restartWouldChangeOrtBuild` — catches someone changing the predicate;
 *  (2) a parity read of `src-tauri/src/lib.rs` — catches the BACKEND moving out from under the
 *      mirror, which no amount of frontend testing can see (same trick as `resumeLockParity`).
 *
 * Both are needed. S116's actual defect was neither a wrong truth table nor a backend change: the
 * mirror was wired to `cuda_available` (a cudart-on-PATH probe) while the build gate had always
 * been `cuda_pkg_supported` (an nvidia-smi compute-cap read). Only (2) states that relationship.
 */
import { describe, expect, it } from "vitest";
import { restartWouldChangeOrtBuild, type OrtBuildHintInput } from "./ortBuild";

// 前端 tsconfig 无 @types/node —— 同 ipcParity/resumeLockParity:用「变量说明符的动态 import」拿 fs。
type NodeFs = { readFileSync(p: string, enc: string): string };
const importFs = (): Promise<NodeFs> => {
  const spec = "node:fs";
  return import(/* @vite-ignore */ spec) as Promise<NodeFs>;
};

const LIB_RS = "src-tauri/src/lib.rs";
const SETTINGS_RS = "src-tauri/src/commands/settings.rs";

const base: OrtBuildHintInput = {
  device: "auto",
  ortBuild: "DirectML",
  autoVendor: undefined,
  cudaSupported: false,
  cudaReady: false,
};
const hint = (o: Partial<OrtBuildHintInput>) => restartWouldChangeOrtBuild({ ...base, ...o });

describe("restartWouldChangeOrtBuild", () => {
  it("★the S116 regression: a card our CUDA package cannot run is never promised a restart", () => {
    // Blackwell (or pre-Turing) NVIDIA box that still has the CUDA package on disk — or any box
    // with the CUDA Toolkit installed. `cuda_available` is true there, which is exactly why the
    // old predicate fired; `cuda_supported` is false forever, so no restart can ever deliver CUDA.
    expect(
      hint({ device: "auto", autoVendor: "nvidia", ortBuild: "DirectML", cudaSupported: false, cudaReady: true }),
      "Auto + NVIDIA pick on an unsupported card must NOT promise a restart",
    ).toBe(false);
    expect(
      hint({ device: "cuda", ortBuild: "DirectML", cudaSupported: false, cudaReady: true }),
      "the explicit-CUDA leg has the same false promise and must be closed too",
    ).toBe(false);
  });

  it("a supported card whose CUDA package is not installed is also not promised a restart", () => {
    // init_ort_runtime logs "CUDA preferred but runtime/ort/cuda/ missing — using DirectML build".
    // The panel already offers the Download button for this machine; a restart hint would be noise.
    expect(hint({ device: "auto", autoVendor: "nvidia", ortBuild: "DirectML", cudaSupported: true, cudaReady: false })).toBe(false);
    expect(hint({ device: "cuda", ortBuild: "DirectML", cudaSupported: true, cudaReady: false })).toBe(false);
  });

  it("the genuine case still fires — otherwise the fix would just be a mute button", () => {
    expect(hint({ device: "auto", autoVendor: "nvidia", ortBuild: "DirectML", cudaSupported: true, cudaReady: true })).toBe(true);
    expect(hint({ device: "cuda", ortBuild: "DirectML", cudaSupported: true, cudaReady: true })).toBe(true);
  });

  it("the two legs a restart always satisfies stay unconditional", () => {
    // The DirectML build is the unconditional default in init_ort_runtime's search path, and a
    // non-NVIDIA Auto pick forces it (`picked_non_nvidia`). Neither needs the CUDA terms.
    expect(hint({ device: "directml", ortBuild: "CUDA" })).toBe(true);
    expect(hint({ device: "auto", autoVendor: "intel", ortBuild: "CUDA" })).toBe(true);
    expect(hint({ device: "auto", autoVendor: "amd", ortBuild: "CUDA" })).toBe(true);
  });

  it("no contradiction, no hint", () => {
    expect(hint({ device: "cuda", ortBuild: "CUDA", cudaSupported: true, cudaReady: true })).toBe(false);
    expect(hint({ device: "directml", ortBuild: "DirectML" })).toBe(false);
    expect(hint({ device: "cpu", ortBuild: "DirectML" })).toBe(false);
    // Auto with no preferred GPU picked: ORT's own high-performance pick decides, and the hint
    // has nothing to compare against.
    expect(hint({ device: "auto", autoVendor: undefined, ortBuild: "CUDA", cudaSupported: true, cudaReady: true })).toBe(false);
  });
});

describe("buildMismatch ↔ init_ort_runtime 对拍", () => {
  /** The body of `init_ort_runtime` up to the point where the build decision is finished. */
  async function gateBody(): Promise<string> {
    const src = (await importFs()).readFileSync(LIB_RS, "utf8");
    const start = src.indexOf("pub fn init_ort_runtime");
    expect(start, `${LIB_RS} no longer defines init_ort_runtime`).toBeGreaterThan(0);
    const end = src.indexOf("let mut search_paths", start);
    expect(end, "init_ort_runtime no longer builds search_paths — re-anchor this test").toBeGreaterThan(start);
    const body = src.slice(start, end);
    // Self-check FIRST (S105): a parser that silently matched nothing makes every assertion below
    // vacuously true.
    expect(body.length, "the sliced gate body is implausibly short").toBeGreaterThan(400);
    expect(body, "expected the `prefer_cuda` decision inside the slice").toContain("let prefer_cuda");
    return body;
  }

  it("the build gate keys on cuda_pkg_supported — the term `cudaSupported` mirrors", async () => {
    expect(
      await gateBody(),
      "the ORT build gate stopped using cuda_pkg_supported; lib/ortBuild.ts mirrors it and must be re-derived",
    ).toContain("cuda_pkg_supported");
  });

  it("★the build gate does NOT key on is_cuda_available — the probe the hint used to mirror", async () => {
    // This is the assertion that states the S116 defect. `cuda_available` answers "is some cudart
    // reachable"; it has never decided which ORT build loads. If it ever does, this test is the
    // place that says so, and lib/ortBuild.ts must be updated in the same change.
    expect(await gateBody()).not.toContain("is_cuda_available");
  });

  it("the Auto arm requires the provider dependency set — the term `cudaReady` covers", async () => {
    expect(
      await gateBody(),
      "cuda_provider_deps_resolvable left the Auto arm; is_cuda_runtime_ready is no longer a superset of it",
    ).toContain("cuda_provider_deps_resolvable");
  });

  it("自检:cuda_available 与 cuda_supported 在后端是两个不同的问题", async () => {
    // If these two HardwareInfo fields were fed by the same probe, swapping one for the other
    // would be cosmetic and the fix above would prove nothing.
    const src = (await importFs()).readFileSync(SETTINGS_RS, "utf8");
    expect(src).toMatch(/cuda_available:\s*has_nvidia\s*&&\s*is_cuda_available\(\)/);
    expect(src).toMatch(/cuda_supported:\s*cuda_pkg_supported\(\)/);
    // …and that `cudaReady`'s backing command really is the superset we rely on.
    const ready = src.slice(src.indexOf("pub fn is_cuda_runtime_ready"));
    expect(ready.length).toBeGreaterThan(200);
    expect(ready.slice(0, 900)).toContain("cuda_provider_deps_resolvable");
  });
});
