import { describe, it, expect } from "vitest";
import { mergeCkptSources, type CkptInfo, type CkptRecord } from "./training";

/**
 * `mergeCkptSources` reconciles the two places a checkpoint can be known from:
 *  - the RUNNING sidecar's `training-ckpt` events (in memory; gone on app close or「清空结果」)
 *  - the on-disk scan (`list_project_ckpts`; the truth, but only as fresh as its last run)
 *
 * Getting this wrong is not cosmetic: a duplicate row means the user imports the same
 * checkpoint twice under two names, and a row that blinks out between the event and the next
 * scan looks exactly like a checkpoint that failed to save.
 */
const rec = (over: Partial<CkptRecord> = {}): CkptRecord => ({
  rel: "rvc/weights/m_e1_s100.pth",
  path: "D:\\data\\training\\p\\rvc\\weights\\m_e1_s100.pth",
  family: "rvc",
  kind: "release",
  step: 100,
  bytes: 55,
  mtimeMs: 1000,
  imported: false,
  companions: [],
  ...over,
});

const ev = (over: Partial<CkptInfo> = {}): CkptInfo => ({
  kind: "periodic",
  path: "D:\\data\\training\\p\\rvc\\weights\\m_e1_s100.pth",
  step: 100,
  epoch: 1,
  ...over,
});

describe("mergeCkptSources", () => {
  it("keeps the scanned record when both sources know the file", () => {
    const out = mergeCkptSources([ev()], [rec({ imported: true, bytes: 999 })], "rvc");
    expect(out).toHaveLength(1);
    // the scan carries size / kind / imported; the event carries none of it
    expect(out[0]!.imported).toBe(true);
    expect(out[0]!.bytes).toBe(999);
  });

  it("matches paths case- and separator-insensitively (Windows)", () => {
    const out = mergeCkptSources(
      [ev({ path: "d:/DATA/training/p/rvc/weights/M_E1_S100.PTH" })],
      [rec()],
      "rvc",
    );
    expect(out).toHaveLength(1);
  });

  it("still shows a checkpoint the sidecar just announced but the scan has not seen", () => {
    const out = mergeCkptSources(
      [ev({ path: "D:\\data\\training\\p\\rvc\\weights\\m_e2_s200.pth", step: 200 })],
      [rec()],
      "rvc",
    );
    expect(out).toHaveLength(2);
    // just written ⇒ sorts first, and is never claimed to be imported
    expect(out[0]!.step).toBe(200);
    expect(out[0]!.imported).toBe(false);
    expect(out[0]!.family).toBe("rvc");
  });

  it("never guesses a kind for an event-only row", () => {
    const out = mergeCkptSources(
      [ev({ path: "D:\\x\\best.pth", kind: "best" }), ev({ path: "D:\\x\\p.pth" })],
      [],
      "sovits",
    );
    // The event only knows periodic/best/final/stop, and the SAME shape is a release snapshot
    // for rvc/sovits weights but a resume point for diffusion `model_<step>.pt`. Guessing made
    // a diffusion ckpt read「快照」for the seconds before the scan caught up, then flip to
    // 「可续训」— so an event-only row says "pending" and claims nothing.
    expect(out.every((r) => r.kind === "pending")).toBe(true);
  });

  it("orders newest first and is stable for equal timestamps", () => {
    const out = mergeCkptSources(
      [],
      [
        rec({ rel: "rvc/b.pth", path: "D:\\b.pth", mtimeMs: 500 }),
        rec({ rel: "rvc/a.pth", path: "D:\\a.pth", mtimeMs: 500 }),
        rec({ rel: "rvc/c.pth", path: "D:\\c.pth", mtimeMs: 900 }),
      ],
      "rvc",
    );
    expect(out.map((r) => r.rel)).toEqual(["rvc/c.pth", "rvc/a.pth", "rvc/b.pth"]);
  });

  it("is a no-op on empty inputs", () => {
    expect(mergeCkptSources([], [], "rvc")).toEqual([]);
  });
});
