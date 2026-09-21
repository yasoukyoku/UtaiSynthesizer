// 智能编曲动作层测试:歌曲身份证 analyzeSourceProfile + 整组清空 clearAutoArrange。
import { describe, it, expect, beforeEach, vi } from "vitest";
vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.resolve() }));
vi.mock("../../i18n", () => ({ default: { t: (k) => k } }));
import { useProjectStore } from "../../store/project";
import { useHistoryStore, installHistory } from "../../store/history";
import { analyzeSourceProfile, clearAutoArrange } from "./autoArrange";
const BEAT = 480;
function notesTrack(id, pitches, beats) {
    const notes = [];
    for (let b = 0; b < beats; b++) {
        for (const p of pitches) {
            notes.push({ id: `n${b}_${p}`, tick: b * BEAT, duration: BEAT, pitch: p, lyric: "", velocity: 100 });
        }
    }
    const seg = {
        id: "s1", startTick: 0, durationTicks: beats * BEAT,
        content: { type: "notes", notes },
    };
    return {
        id, name: id, trackType: "vocal", segments: [seg],
        volumeDb: 0, pan: 0, muted: false, solo: false, expanded: false, laneControls: {},
    };
}
function instTrack(id, name) {
    return {
        id, name, trackType: "instrument",
        segments: [{ id: `${id}-s`, startTick: 0, durationTicks: BEAT, content: { type: "notes", notes: [] } }],
        volumeDb: 0, pan: 0, muted: false, solo: false, expanded: false, laneControls: {},
    };
}
function seed(tracks, tempo = 120) {
    useProjectStore.setState({
        name: "P", tracks, tempo, timeSignature: [4, 4],
        dirty: false, filePath: null, selectedNotes: [], playheadTick: 0,
    });
}
let uninstall = null;
beforeEach(() => {
    uninstall?.();
    uninstall = installHistory();
    useHistoryStore.getState().reset();
    useHistoryStore.getState().markSaved();
});
describe("analyzeSourceProfile 歌曲身份证", () => {
    it("C 大三和弦旋律 → C 调 / 120 BPM / 4 拍 / 2 小节 / 有和弦序列", () => {
        seed([notesTrack("mel", [60, 64, 67], 8)]);
        const p = analyzeSourceProfile("mel");
        expect(p).not.toBeNull();
        expect(p.bpm).toBe(120);
        expect(p.beatsPerBar).toBe(4);
        expect(p.beatUnit).toBe(4);
        expect(p.bars).toBe(2);
        expect(p.keyLabel).toBe("C");
        expect(p.noteCount).toBe(24); // 8 拍 × 3 音
        expect(p.chordLabels.length).toBeGreaterThan(0);
        expect(p.chordLabels[0]).toBe("C");
    });
    it("不存在的轨 / 无音符轨 → null", () => {
        seed([notesTrack("empty", [60], 0)]);
        expect(analyzeSourceProfile("nope")).toBeNull();
    });
    it("相邻小节同和弦 → 标签去重压缩", () => {
        seed([notesTrack("mel", [60, 64, 67], 16)]); // 4 小节全 C
        const p = analyzeSourceProfile("mel");
        expect(p.bars).toBe(4);
        expect(p.chordLabels).toEqual(["C"]); // 全曲同和弦只留一个
    });
    it("6/8 拍号 → beatUnit 如实为 8(不写死 /4)", () => {
        seed([notesTrack("mel", [60, 64, 67], 12)]);
        useProjectStore.setState({ timeSignature: [6, 8] });
        const p = analyzeSourceProfile("mel");
        expect(p.beatsPerBar).toBe(6);
        expect(p.beatUnit).toBe(8);
    });
});
describe("clearAutoArrange 整组清空", () => {
    it("删除源轨正下方连续的伴奏轨,遇到非伴奏轨即停", () => {
        seed([
            notesTrack("mel", [60, 64, 67], 4),
            instTrack("d", "🥁 Drums · 流行"),
            instTrack("b", "🎸 Bass · 流行"),
            instTrack("p", "🎹 Piano · 流行"),
            instTrack("mine", "我自己的轨"),
        ]);
        const removed = clearAutoArrange("mel");
        expect(removed).toBe(3);
        const names = useProjectStore.getState().tracks.map((t) => t.name);
        expect(names).toEqual(["mel", "我自己的轨"]);
    });
    it("源轨下没有伴奏轨 → 0 且不动其它轨", () => {
        seed([notesTrack("mel", [60, 64, 67], 4), instTrack("mine", "我自己的轨")]);
        expect(clearAutoArrange("mel")).toBe(0);
        expect(useProjectStore.getState().tracks).toHaveLength(2);
    });
});
