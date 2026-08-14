/**
 * `rangeRecordSig` 必须覆盖**每一个进入音域决策的项**。
 *
 * S146b 加了 `semitones_onset`(第二遍「像真唱」的探针,用来否决不安全的落点)——它**改的是
 * 救援深度**。如果签名不含它,后果不是报错,而是**这条修复对已经烤好的片段永远不生效**:
 * 用户重扫了模型、判决变了、而缓存音频一动不动 ⇒ 读起来就是「你改了,我什么都没听见」——
 * 那正是 `RANGE_ALGO_VERSION` 这一整套存在的唯一理由(S81/S86 R3 都栽在这个形状上)。
 *
 * ⛔ 断言写成**两份指纹的真实差异**,不是 `toContain`:一个「在字符串里但没被读」的项
 * 照样能通过包含检查(S88:「sig 自比不是测试」的同族)。
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Track, VocalTrackParams } from "../../types/project";

/** 可变的 mock:每个用例自己摆模型 sidecar 的内容。 */
const state: { models: Record<string, { name: string; path: string; config?: unknown }[]> } = {
  models: { sovits: [{ name: "V", path: "p" }] },
};
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k: string) => k } }));
vi.mock("../../store/voice-models", () => ({
  useVoiceModelStore: { getState: () => state },
}));

const { vocalTrackSig } = await import("./vocalRender");
const { DEFAULT_TRANSITION } = await import("../vocalNotes");

const vp = (): VocalTrackParams => ({
  backend: "sovits",
  speakerId: 0,
  langId: 2,
  transpose: 0,
  formant: 0,
  transition: DEFAULT_TRANSITION,
  breathToken: "AP",
  rangeExtend: true,
} as VocalTrackParams);

const track = (): Track =>
  ({
    id: "t", name: "V", trackType: "vocal", segments: [], volumeDb: 0, pan: 0,
    muted: false, solo: false, expanded: true, laneControls: {},
    voiceModel: "V", vocalParams: vp(),
  }) as Track;

const withRecord = (extra: Record<string, unknown>) => {
  state.models.sovits = [{
    name: "V", path: "p",
    config: {
      vocal_range: {
        speakers: {
          "0": { usable: [60, 80], comfort: [60, 79], semitones: { "79": [2, 0.97], "80": [2, 0.69] }, ...extra },
        },
      },
    },
  }];
};

describe("rangeRecordSig 覆盖 semitones_onset", () => {
  beforeEach(() => {
    state.models.sovits = [{ name: "V", path: "p" }];
  });

  it("加上第二遍探针的读数,签名必须变", () => {
    withRecord({});
    const before = vocalTrackSig(track(), 120);
    withRecord({ semitones_onset: { "79": [2, 0.78], "80": [2, 0.33] } });
    const after = vocalTrackSig(track(), 120);
    expect(after).not.toBe(before);
  });

  it("第二遍探针的读数**变了**(不是有无),签名也必须变", () => {
    // 这一条防的是「只判断键在不在」的实现:重扫一次模型、否决的格子换了,决策就变了。
    withRecord({ semitones_onset: { "79": [2, 0.78], "80": [2, 0.33] } });
    const a = vocalTrackSig(track(), 120);
    withRecord({ semitones_onset: { "79": [2, 0.95], "80": [2, 0.33] } });
    const b = vocalTrackSig(track(), 120);
    expect(b).not.toBe(a);
  });

  it("阴性对照:记录没动,签名逐字节相同", () => {
    // 否则上面两条在一个「每次都返回随机值」的实现上也会绿。
    withRecord({ semitones_onset: { "80": [2, 0.33] } });
    const a = vocalTrackSig(track(), 120);
    const b = vocalTrackSig(track(), 120);
    expect(b).toBe(a);
  });

  it("阴性对照:关掉音域扩展时这一项整个不参与(rr: 为空)", () => {
    withRecord({ semitones_onset: { "80": [2, 0.33] } });
    const t = track();
    (t.vocalParams as { rangeExtend?: boolean }).rangeExtend = false;
    expect(vocalTrackSig(t, 120)).toContain("|rr:|");
  });
});
