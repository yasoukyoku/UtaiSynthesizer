import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { isTauri } from "../lib/tauri";
export const RECOMMENDED_SOUNDFONTS = [
    {
        purposeKey: "soundfont.recPurposes.gm",
        name: "GeneralUser GS v1.471",
        format: "sfz/sf2",
        size: "~31MB",
        url: "https://github.com/mrbumpy409/GeneralUser-GS",
        license: "免费可商用",
        matchKeywords: ["generaluser", "gm"],
    },
    {
        purposeKey: "soundfont.recPurposes.piano",
        name: "Salamander Grand Piano v3",
        format: "sfz",
        size: "394MB",
        url: "https://github.com/sfzinstruments/SalamanderGrandPiano",
        license: "CC BY 3.0",
        matchKeywords: ["salamander"],
    },
    {
        purposeKey: "soundfont.recPurposes.drums",
        name: "AVL Drumkits 1.1",
        format: "sfz",
        size: "~150MB",
        url: "http://www.bandshed.net/avldrumkits/",
        license: "CC BY-SA 3.0",
        matchKeywords: ["avl"],
    },
    {
        purposeKey: "soundfont.recPurposes.bass",
        name: "Karoryfer Swagbass",
        format: "sfz",
        size: "~50MB",
        url: "https://github.com/Karoryfer",
        license: "CC BY",
        matchKeywords: ["swagbass", "karoryfer"],
    },
    {
        purposeKey: "soundfont.recPurposes.strings",
        name: "VCSL Strings",
        format: "sfz",
        size: "~100MB+",
        url: "https://github.com/sgossner/VCSL",
        license: "CC0",
        matchKeywords: ["vcsl", "strings"],
    },
    {
        purposeKey: "soundfont.recPurposes.orchestra",
        name: "VSCO-2 CE",
        format: "sfz",
        size: "~3GB",
        url: "https://github.com/sgossner/VSCO-2-CE",
        license: "CC0",
        matchKeywords: ["vsco"],
    },
    {
        purposeKey: "soundfont.recPurposes.epiano",
        name: "jRhodes3c",
        format: "sfz",
        size: "~18MB",
        url: "https://github.com/sfzinstruments/johneriknorlander/jRhodes3c",
        license: "CC BY-NC-SA",
        matchKeywords: ["jrhodes"],
    },
    {
        purposeKey: "soundfont.recPurposes.orchestra",
        name: "Sonatina Symphonic Orchestra 1.0",
        format: "sfz",
        size: "~440MB",
        url: "https://musical-artifacts.com/artifacts/81",
        license: "CC Sampling+ 1.0",
        matchKeywords: ["sonatina", "sso"],
    },
    {
        purposeKey: "soundfont.recPurposes.orchestra",
        name: "Virtual Playing Orchestra",
        format: "sfz",
        size: "~600MB+",
        url: "https://virtualplaying.com/virtual-playing-orchestra/",
        license: "免费(CC 系列许可)",
        matchKeywords: ["vpo", "virtualplaying", "virtual_playing"],
    },
    {
        purposeKey: "soundfont.recPurposes.drums",
        name: "SM Drums",
        format: "sfz",
        size: "~2.2GB+",
        url: "https://smmdrums.wordpress.com/for-sfz-sforzando/",
        license: "Public Domain",
        matchKeywords: ["sm_drums", "smmdrums"],
    },
    {
        purposeKey: "soundfont.recPurposes.guitar",
        name: "Black and Green Guitars",
        format: "sfz",
        size: "~500MB",
        url: "https://github.com/sfzinstruments/karoryfer.black-and-green-guitars",
        license: "CC0",
        matchKeywords: ["black_and_green", "black-and-green", "guitar"],
    },
];
/**
 * 各乐器的 General MIDI 音色号(0-based,melodic channel)。
 * 只要用户本地有任意一个 GM 综合音源(如 GeneralUser GS),就能按此号自动分发不同音色,
 * 实现「一个综合音源覆盖全部乐器」的兜底发声。鼓走 channel 9,不需要 program。
 */
export const GM_PROGRAM = {
    drums: 0,
    piano: 0, // Acoustic Grand
    epiano: 4, // Electric Piano 1 (Rhodes)
    guitarArp: 24, // Nylon String Guitar
    guitarStrum: 25, // Steel String Guitar
    bass: 33, // Finger Bass
    strings: 48, // String Ensemble 1
    chords: 48, // 旧铺底轨 → 弦乐
    synthPad: 89, // Warm Pad
    pluck: 81, // Lead 2 (sawtooth),颗粒琶音
    lead: 80, // Lead 1 (square),主旋律
};
/** 把音轨类型(自动编曲的乐器轨)映射到默认音色:按 fontId 关键字匹配已导入音源。 */
export function defaultSoundfontFor(trackKind, fonts) {
    const keywordByKind = {
        drums: ["avl", "drum", "black_pearl"],
        bass: ["swagbass", "bass"],
        piano: ["salamander", "piano", "grand"],
        chords: ["vcsl", "strings", "generaluser", "gm"],
        lead: ["generaluser", "gm", "vcsl"],
        guitarArp: ["guitar", "nylon", "generaluser", "gm"],
        guitarStrum: ["guitar", "steel", "generaluser", "gm"],
        strings: ["vcsl", "strings", "vsco", "generaluser", "gm"],
        epiano: ["jrhodes", "rhodes", "epiano", "electric", "generaluser", "gm"],
        synthPad: ["pad", "synth", "generaluser", "gm", "vcsl"],
        pluck: ["pluck", "synth", "generaluser", "gm"],
    };
    const kws = keywordByKind[trackKind];
    for (const kw of kws) {
        const hit = fonts.find((f) => f.id.toLowerCase().includes(kw));
        if (hit)
            return hit;
    }
    return null;
}
/** 选取音源的默认 preset(第一个;SF2 优先 0:0 的 GM 钢琴)。 */
export function firstPresetOf(font) {
    if (font.presets.length === 0)
        return null;
    if (font.format === "sf2") {
        const gm = font.presets.find((p) => p.id === "0:0");
        if (gm)
            return gm;
    }
    return font.presets[0] ?? null;
}
/** 在 SF2 综合音源里按 GM program 号选对应 preset(id 形如 "0:24");找不到退回默认。 */
export function presetForGmProgram(font, program) {
    if (font.format === "sf2") {
        const exact = font.presets.find((p) => p.id === `0:${program}`);
        if (exact)
            return exact;
        // 某些音源 bank 非 0:退而求其次匹配 ":program" 后缀。
        const suffix = font.presets.find((p) => p.id.endsWith(`:${program}`));
        if (suffix)
            return suffix;
    }
    return firstPresetOf(font);
}
/**
 * 为一个编曲乐器轨自动挑选可用音色:
 *   1) 优先匹配该乐器的专用音源(吉他→吉他音源…);
 *   2) 否则用任意 GM 综合音源(generaluser/gm)的对应 program preset;
 *   3) 都没有 → null(轨道仍生成,由 UI 提示导入一个综合音源即可全乐器发声)。
 */
export function pickInstrumentSoundfont(kind, fonts) {
    if (fonts.length === 0)
        return null;
    const dedicated = defaultSoundfontFor(kind, fonts);
    if (dedicated) {
        // 专用音源:SF2 综合型按 GM program 取,专用单音色音源取首 preset。
        const preset = dedicated.format === "sf2"
            ? presetForGmProgram(dedicated, GM_PROGRAM[kind]) ?? firstPresetOf(dedicated)
            : firstPresetOf(dedicated);
        if (preset)
            return { font: dedicated, preset };
    }
    // 兜底:任意一个含多 preset 的 SF2(综合音源)按 program 分发。
    const gmFont = fonts.find((f) => f.format === "sf2" && /generaluser|gm|fluid|synth/i.test(f.id)) ??
        fonts.find((f) => f.format === "sf2" && f.presets.length > 1) ??
        null;
    if (gmFont) {
        const preset = presetForGmProgram(gmFont, GM_PROGRAM[kind]) ?? firstPresetOf(gmFont);
        if (preset)
            return { font: gmFont, preset };
    }
    return null;
}
function toFont(raw) {
    const presets = Array.isArray(raw.presets)
        ? raw.presets
            .filter((p) => p && typeof p === "object")
            .map((p) => ({
            id: String(p.id ?? ""),
            name: String(p.name ?? ""),
        }))
            .filter((p) => p.id)
        : [];
    return {
        id: String(raw.id ?? ""),
        name: String(raw.name ?? ""),
        format: raw.format === "sfz" ? "sfz" : "sf2",
        sizeBytes: Number(raw.size_bytes ?? 0) || 0,
        presets,
    };
}
export const useSoundfontStore = create((set, get) => ({
    fonts: [],
    scanning: false,
    importingPaths: [],
    refresh: async () => {
        if (!isTauri())
            return; // No Rust backend in browser preview — silently skip
        if (get().scanning)
            return;
        set({ scanning: true });
        try {
            const raw = await invoke("list_soundfonts");
            set({ fonts: (raw ?? []).map(toFont).filter((f) => f.id) });
        }
        catch (e) {
            console.error("[soundfont] list failed:", e);
        }
        finally {
            set({ scanning: false });
        }
    },
    importFont: async (path) => {
        if (get().importingPaths.includes(path))
            return null;
        set((s) => ({ importingPaths: [...s.importingPaths, path] }));
        try {
            const raw = await invoke("import_soundfont", { path });
            const font = toFont(raw ?? {});
            await get().refresh();
            return font.id ? font : null;
        }
        finally {
            set((s) => ({ importingPaths: s.importingPaths.filter((p) => p !== path) }));
        }
    },
    removeFont: async (id) => {
        try {
            await invoke("delete_soundfont", { id });
            await get().refresh();
            return true;
        }
        catch (e) {
            console.error("[soundfont] delete failed:", e);
            return false;
        }
    },
}));
