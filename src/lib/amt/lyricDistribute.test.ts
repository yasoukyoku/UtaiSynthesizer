import { describe, expect, it } from "vitest";
import { distributeLyricsToNotes, mergeLyricLines } from "./lyricDistribute";

describe("distributeLyricsToNotes — Whisper 句 → 逐音符", () => {
  it("中文一句三字铺到三个音符（一字一音符）", () => {
    const segs = [{ start: 1.0, end: 2.5, text: "我爱你" }];
    const out = distributeLyricsToNotes(segs, [1.0, 1.5, 2.0]);
    expect(out.map((l) => l.text)).toEqual(["我", "爱", "你"]);
    expect(out.map((l) => l.start)).toEqual([1.0, 1.5, 2.0]);
  });

  it("音节多于音符：剩余字合并到最后一个音符", () => {
    const segs = [{ start: 0, end: 2, text: "我爱你" }];
    const out = distributeLyricsToNotes(segs, [0, 1]);
    expect(out.map((l) => l.text)).toEqual(["我", "爱你"]);
  });

  it("英文按空格分词", () => {
    const segs = [{ start: 0, end: 3, text: "I love you" }];
    const out = distributeLyricsToNotes(segs, [0, 1, 2]);
    expect(out.map((l) => l.text)).toEqual(["I", "love", "you"]);
  });

  it("英文音节多于音符时用空格连接", () => {
    const segs = [{ start: 0, end: 3, text: "I love you" }];
    const out = distributeLyricsToNotes(segs, [0, 1]);
    expect(out.map((l) => l.text)).toEqual(["I", "love you"]);
  });

  it("音符多于音节：多余音符不生成行", () => {
    const segs = [{ start: 0, end: 3, text: "你好" }];
    const out = distributeLyricsToNotes(segs, [0, 1, 2]);
    expect(out).toHaveLength(2);
  });

  it("句区间没有音符：整句原样保留", () => {
    const segs = [
      { start: 0, end: 1, text: "前奏" },
      { start: 5, end: 6, text: "开始" },
    ];
    const out = distributeLyricsToNotes(segs, [5.0, 5.5]);
    expect(out).toHaveLength(3);
    expect(out[0]).toEqual({ start: 0, end: 1, text: "前奏" });
    expect(out[1]!.text).toBe("开");
    expect(out[2]!.text).toBe("始");
  });

  it("空文本段被跳过", () => {
    const segs = [
      { start: 0, end: 1, text: "  " },
      { start: 2, end: 3, text: "词" },
    ];
    const out = distributeLyricsToNotes(segs, [2.2]);
    expect(out).toHaveLength(1);
  });

  it("乱序 onsets 内部排序后分配", () => {
    const segs = [{ start: 0, end: 3, text: "甲乙丙" }];
    const out = distributeLyricsToNotes(segs, [2, 0, 1]);
    expect(out.map((l) => l.text)).toEqual(["甲", "乙", "丙"]);
    expect(out.map((l) => l.start)).toEqual([0, 1, 2]);
  });
});

describe("mergeLyricLines — LRC 按句导出", () => {
  it("间隔小的行合并为一句（中文连写）", () => {
    const lines = [
      { start: 0, end: 0.4, text: "我" },
      { start: 0.5, end: 0.9, text: "爱" },
      { start: 1.0, end: 1.4, text: "你" },
    ];
    const out = mergeLyricLines(lines, 0.5);
    expect(out).toHaveLength(1);
    expect(out[0]!.text).toBe("我爱你");
    expect(out[0]!.start).toBe(0);
  });

  it("英文合并加空格", () => {
    const lines = [
      { start: 0, end: 0.4, text: "love" },
      { start: 0.5, end: 0.9, text: "you" },
    ];
    const out = mergeLyricLines(lines, 0.5);
    expect(out[0]!.text).toBe("love you");
  });

  it("间隔大的行保持独立（不同句）", () => {
    const lines = [
      { start: 0, end: 0.4, text: "你好" },
      { start: 3.0, end: 3.4, text: "世界" },
    ];
    const out = mergeLyricLines(lines, 0.5);
    expect(out).toHaveLength(2);
  });

  it("空文本行被过滤", () => {
    const lines = [
      { start: 0, end: 0.4, text: "" },
      { start: 1, end: 1.4, text: "词" },
    ];
    const out = mergeLyricLines(lines, 0.5);
    expect(out).toHaveLength(1);
  });
});

