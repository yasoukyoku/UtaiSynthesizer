/**
 * 音源自动分发测试:专用音源优先 → GM 综合音源按 program 兜底。
 * 钉死 GM_PROGRAM 映射与 pickInstrumentSoundfont / presetForGmProgram 的选择契约。
 */
import { describe, it, expect } from "vitest";
import { GM_PROGRAM, defaultSoundfontFor, firstPresetOf, presetForGmProgram, pickInstrumentSoundfont, } from "./soundfont";
/** 造一个 SF2,含 0:0..0:`maxProgram` 的 GM preset。 */
function gmFont(id, maxProgram = 100) {
    return {
        id,
        name: id,
        format: "sf2",
        sizeBytes: 1000,
        presets: Array.from({ length: maxProgram + 1 }, (_, p) => ({ id: `0:${p}`, name: `P${p}` })),
    };
}
describe("GM_PROGRAM 映射", () => {
    it("11 类乐器全部有 program,且鼓为 0(鼓实际走 channel 9)", () => {
        const kinds = ["drums", "bass", "piano", "guitarArp", "guitarStrum", "epiano", "strings", "chords", "synthPad", "pluck", "lead"];
        for (const k of kinds)
            expect(GM_PROGRAM[k]).toBeGreaterThanOrEqual(0);
        expect(GM_PROGRAM.guitarArp).toBe(24);
        expect(GM_PROGRAM.strings).toBe(48);
        expect(GM_PROGRAM.epiano).toBe(4);
    });
});
describe("presetForGmProgram", () => {
    it("按 0:program 精确取音色", () => {
        const f = gmFont("generaluser_gs");
        expect(presetForGmProgram(f, 24)?.id).toBe("0:24");
        expect(presetForGmProgram(f, 48)?.id).toBe("0:48");
    });
    it("缺失 program 时退回默认 preset,不抛错", () => {
        const f = gmFont("tiny", 5); // 只到 0:5
        expect(presetForGmProgram(f, 48)?.id).toBe("0:0");
    });
});
describe("defaultSoundfontFor 关键字匹配", () => {
    it("吉他类优先匹配 guitar 音源", () => {
        const fonts = [gmFont("piano_grand"), gmFont("nylon_guitar")];
        expect(defaultSoundfontFor("guitarArp", fonts)?.id).toBe("nylon_guitar");
    });
    it("无匹配返回 null", () => {
        expect(defaultSoundfontFor("bass", [gmFont("weird_name")])).toBeNull();
    });
});
describe("pickInstrumentSoundfont 兜底链", () => {
    it("专用音源 + SF2 → 按对应 GM program 取 preset", () => {
        const got = pickInstrumentSoundfont("guitarStrum", [gmFont("steel_guitar")]);
        expect(got?.font.id).toBe("steel_guitar");
        expect(got?.preset.id).toBe("0:25");
    });
    it("无专用、但有任意多 preset SF2 → 兜底按 program 分发", () => {
        const got = pickInstrumentSoundfont("epiano", [gmFont("my_custom_font")]);
        expect(got?.font.id).toBe("my_custom_font");
        expect(got?.preset.id).toBe("0:4"); // 电钢 Rhodes
    });
    it("空音源列表 → null(不抛错,由 UI 提示导入)", () => {
        expect(pickInstrumentSoundfont("piano", [])).toBeNull();
    });
    it("SFZ 专用单音色音源 → 取其首 preset", () => {
        const sfz = {
            id: "jrhodes_ep", name: "jRhodes", format: "sfz", sizeBytes: 1,
            presets: [{ id: "p0", name: "Rhodes" }],
        };
        const got = pickInstrumentSoundfont("epiano", [sfz]);
        expect(got?.preset.id).toBe("p0");
    });
});
describe("firstPresetOf", () => {
    it("SF2 优先 0:0", () => {
        const f = { id: "x", name: "x", format: "sf2", sizeBytes: 1, presets: [{ id: "0:5", name: "a" }, { id: "0:0", name: "b" }] };
        expect(firstPresetOf(f)?.id).toBe("0:0");
    });
});
